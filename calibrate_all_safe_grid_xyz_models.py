from __future__ import annotations

"""
calibrate_all_safe_grid.py

One-shot safety-gated calibration scan for the MAE 162 grocery bagger.

This is the "robust/easy-to-use" calibration script.

What it builds:
  1) Safety-gated XY/Z grid scan
  2) Height-indexed overhead homography lookup H(z)
  3) Visibility / workspace validity table
  4) Stereo EE logging, if stereo camera + stereo_calibration.npz are available
  5) Global stereo affine fits from logged stereo samples
  6) Support-point arrays for runtime confidence / nearest-neighbor checks
  7) Optional visible reference poses for later hidden-target bias correction

Outputs:
  robot_calibration_bundle.npz
  robot_calibration_scan_raw.csv
  robot_calibration_report.txt

Safety:
  - Requires robot.py to support fail-closed soft limits.
  - Requires soft_limits_config.json.
  - Refuses to run if soft limits are unavailable.
  - Every move goes through robot.move_cartesian().
  - Skips points that are unreachable, unsafe, not visible, or not detected.
  - Never lowers/substitutes Z for a calibration layer.
    If a point is unsafe at that Z layer, it is skipped for that layer.

Expected files in same folder:
  robot.py
  robot_config.py
  camera_config.py
  soft_limits.py
  soft_limits_config.json
  stereo_apriltag_viewer.py
  overhead_intrinsics.npz      optional, recommended
  stereo_calibration.npz       optional, for stereo XYZ logging
"""

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from robot import Robot
from robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    SOFT_LIMITS_CONFIG_PATH,
    print_startup_config,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_INDEX,
    OVERHEAD_INTRINSICS_PATH,
    OVERHEAD_WIDTH,
    STEREO_CALIBRATION_PATH,
    STEREO_FOURCC,
    STEREO_FPS,
    STEREO_HEIGHT,
    STEREO_INDEX,
    STEREO_WIDTH,
)
from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
    make_preview,
    average_angles_deg,
)


# ============================================================
# USER SETTINGS
# ============================================================

# User-facing floor-plane height convention.
# Internal calibration still stores robot_z_mm because robot.move_cartesian() uses robot Z.
# But these constants let reports/bundles show physical height above the floor.
DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM = 575.0
L1_TO_L2_HEIGHT_DIFFERENCE_MM = 55.0
J3_ZERO_TO_EE_DROP_MM = 400.0


def robot_z_to_floor_height_mm(robot_z_mm: float) -> float:
    return (
        DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM
        - L1_TO_L2_HEIGHT_DIFFERENCE_MM
        - (J3_ZERO_TO_EE_DROP_MM - float(robot_z_mm))
    )


def floor_height_to_robot_z_mm(floor_height_mm: float) -> float:
    return (
        float(floor_height_mm)
        - DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM
        + L1_TO_L2_HEIGHT_DIFFERENCE_MM
        + J3_ZERO_TO_EE_DROP_MM
    )


ENABLE_STEREO_LOGGING = True

# Pick one. Start with "smoke" first.
SCAN_MODE = "global_coarse"
# SCAN_MODE = "global_coarse"
# SCAN_MODE = "staging_refined"
# SCAN_MODE = "local_dense"

SCAN_PRESETS = {
    # First safe sanity check. Use this first after config/soft-limit changes.
    "smoke": {
        "x": [250, 350],
        "y": [300, 400],
        "z": [DEFAULT_TRAVEL_Z_MM],
    },

    # Good first real scan. Keep Z values conservative.
    "global_coarse": {
        "x": list(range(90, 400, 100)),
        "y": list(range(250, 550, 100)),
        "z": list(range(0, 200, 50)),
    },

    # Better accuracy in the main grocery staging area.
    "staging_refined": {
        "x": list(range(75, 500, 25)),
        "y": list(range(225, 600, 25)),
        "z": [0, 25.0],
    },

    # Only use this in a smaller region once everything else is stable.
    "local_dense": {
        "x": list(range(200, 401, 10)),
        "y": list(range(275, 476, 10)),
        "z": [50.0, 75.0],
    },
}

# Optional visible reference poses. These are useful later for correcting hidden targets
# like the bag drop location. Runtime can move to one of these visible poses, measure
# camera-vs-FK bias, and apply that bias to hidden taught targets.
#
# Format:
#   (name, x_mm, y_mm, z_mm, phi_deg)
#
# Keep them in areas where the overhead camera reliably sees the EE tag.
REFERENCE_POSES = [
    ("visible_ref_center", 300.0, 350.0, DEFAULT_TRAVEL_Z_MM, 0.0),
]

CAL_PHI_DEG = 0.0

# Motion.
MOVE_TIME_S = 0.90
SETTLE_S = 0.25
SYNC_AFTER_MOVE = True

# Detection / averaging.
SAMPLES_PER_POINT = 10
POINT_TIMEOUT_S = 3.0
MIN_OVERHEAD_SAMPLES = 4
MIN_STEREO_SAMPLES = 2

# Build filters.
MIN_POINTS_PER_HOMOGRAPHY = 4
RECOMMENDED_POINTS_PER_HOMOGRAPHY = 8

# Runtime confidence metadata thresholds. These are saved into the bundle.
MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM = 90.0
MAX_RUNTIME_HOMOGRAPHY_RMS_MM = 25.0

# ============================================================
# STEREO XYZ MODEL KNOBS
# ============================================================
# These are small calibration/axis knobs for the full 3D stereo model.
# They do NOT change the raw stereo triangulation; they only affect the
# fitted camera<->robot matrices saved into robot_calibration_bundle.npz.
#
# Convention:
#   robot_xyz = [robot_x_mm, robot_y_mm, robot_z_mm]
#   cam_xyz   = [stereo_x_mm, stereo_y_mm, stereo_z_mm]
#
# We fit:
#   B_cam_from_robot_xyz_3x4 @ [robot_x, robot_y, robot_z, 1] -> cam_xyz
#   A_robot_from_cam_xyz_3x4 @ [cam_x, cam_y, cam_z, 1]       -> robot_xyz
#
# Runtime delta convention:
#   delta_cam   = object_cam_xyz - ee_cam_xyz
#   delta_robot = A_robot_from_cam_xyz_3x4[:, :3] @ delta_cam
#
# Optional offsets below let you compensate for known fixed offsets between the
# triangulated EE tag center and the actual gripper/tool point. Leave zero first.
STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM = np.array([0.0, 0.0, 0.0], dtype=np.float64)
STEREO_CAM_X_OFFSET_MM = 0.0
STEREO_CAM_Y_OFFSET_MM = 0.0
STEREO_CAM_Z_OFFSET_MM = 0.0

# Require enough 3D spread to fit a meaningful XYZ model.
MIN_STEREO_XYZ_FIT_SAMPLES = 8
WARN_STEREO_XYZ_FIT_RMSE_MM = 25.0


# Files.
SOFT_LIMITS_PATH = SOFT_LIMITS_CONFIG_PATH

OUT_BUNDLE = Path("robot_calibration_bundle.npz")
OUT_RAW_CSV = Path("robot_calibration_scan_raw.csv")
OUT_REPORT = Path("robot_calibration_report.txt")

WINDOW_OVERHEAD = "Calibrate All - Overhead"
WINDOW_STEREO = "Calibrate All - Stereo"


# ============================================================
# SCAN PRESET RESOLUTION
# ============================================================

def get_scan_grids():
    if SCAN_MODE not in SCAN_PRESETS:
        raise KeyError(f"Unknown SCAN_MODE={SCAN_MODE!r}. Options={list(SCAN_PRESETS)}")

    preset = SCAN_PRESETS[SCAN_MODE]
    x_grid = [float(v) for v in preset["x"]]
    y_grid = [float(v) for v in preset["y"]]
    z_levels = [float(v) for v in preset["z"]]

    return x_grid, y_grid, z_levels


# ============================================================
# DATA
# ============================================================

@dataclass
class ScanRow:
    z_level_mm: float
    grid_name: str
    x_cmd_mm: float
    y_cmd_mm: float
    z_cmd_mm: float
    floor_height_cmd_mm: float

    is_reference_pose: bool

    reachable_ik: bool
    target_safe: bool
    path_safe: bool
    move_attempted: bool
    move_ok: bool

    reason: str

    robot_x_mm: float | None = None
    robot_y_mm: float | None = None
    robot_z_mm: float | None = None
    robot_phi_deg: float | None = None

    overhead_visible: bool = False
    overhead_used_for_homography: bool = False
    overhead_u_raw_px: float | None = None
    overhead_v_raw_px: float | None = None
    overhead_u_used_px: float | None = None
    overhead_v_used_px: float | None = None
    overhead_theta_deg: float | None = None
    overhead_samples: int = 0

    stereo_visible: bool = False
    stereo_x_mm: float | None = None
    stereo_y_mm: float | None = None
    stereo_z_mm: float | None = None
    stereo_left_u_px: float | None = None
    stereo_left_v_px: float | None = None
    stereo_right_u_px: float | None = None
    stereo_right_v_px: float | None = None
    stereo_disparity_px: float | None = None
    stereo_theta_deg: float | None = None
    stereo_samples: int = 0


# ============================================================
# CAMERA HELPERS
# ============================================================

def open_overhead_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def load_overhead_intrinsics(path: Path):
    if not path.exists():
        print(f"[Overhead intrinsics] Missing {path}. Using raw distorted tag centers.")
        return None, None, False

    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    K = data["camera_matrix"] if "camera_matrix" in keys else data["K"]

    if "dist_coeffs" in keys:
        dist = data["dist_coeffs"]
    elif "distortion_coefficients" in keys:
        dist = data["distortion_coefficients"]
    elif "dist" in keys:
        dist = data["dist"]
    else:
        raise KeyError(f"No distortion coeffs in {path}. Keys={data.files}")

    print(f"[Overhead intrinsics] Loaded {path.resolve()}")
    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64), True


def undistort_uv(uv_raw: np.ndarray, K, dist) -> np.ndarray:
    uv_raw = np.asarray(uv_raw, dtype=np.float64).reshape(2)
    if K is None or dist is None:
        return uv_raw
    pt = uv_raw.astype(np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def read_overhead_once(cap: cv2.VideoCapture, detector):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    dets = detect_tags(detector, frame)
    return frame, dets.get(EE_TAG_ID)


def draw_overhead(frame, det, lines):
    out = draw_detection(frame, det, f"OVERHEAD EE {EE_TAG_ID}")
    for i, line in enumerate(lines):
        y = 30 + i * 27
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ============================================================
# STEREO HELPERS
# ============================================================

def load_stereo_calibration(path: Path):
    if not path.exists():
        print(f"[Stereo] Missing {path}; stereo XYZ logging disabled, pixel logging still possible.")
        return None
    data = np.load(path, allow_pickle=False)
    calib = {k: np.asarray(data[k]) for k in data.files}
    print(f"[Stereo] Loaded calibration {path.resolve()}")
    return calib


def undistort_pixel_to_pixel_coords(point_xy: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pt = np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def triangulate_center_raw(left_xy: np.ndarray, right_xy: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    left_pt = undistort_pixel_to_pixel_coords(
        left_xy,
        np.asarray(calib["left_camera_matrix"], dtype=np.float64),
        np.asarray(calib["left_distortion_coefficients"], dtype=np.float64),
    ).reshape(2, 1)
    right_pt = undistort_pixel_to_pixel_coords(
        right_xy,
        np.asarray(calib["right_camera_matrix"], dtype=np.float64),
        np.asarray(calib["right_distortion_coefficients"], dtype=np.float64),
    ).reshape(2, 1)

    P_left = np.asarray(calib["projection_left_raw"], dtype=np.float64)
    P_right = np.asarray(calib["projection_right_raw"], dtype=np.float64)

    X_h = cv2.triangulatePoints(P_left, P_right, left_pt, right_pt)
    xyz = (X_h[:3] / X_h[3]).reshape(3).astype(np.float64)

    # Match the existing convention from your stereo PD scripts.
    xyz[2] *= -1.0
    return xyz


def read_stereo_once(stereo: SimpleStereoCamera, detector, stereo_calib):
    ok, _, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        return None, None, None, None, None

    det_l_all = detect_tags(detector, left)
    det_r_all = detect_tags(detector, right)
    det_l = det_l_all.get(EE_TAG_ID)
    det_r = det_r_all.get(EE_TAG_ID)

    xyz = None
    if stereo_calib is not None and det_l is not None and det_r is not None:
        xyz = triangulate_center_raw(
            np.asarray(det_l.center, dtype=np.float64),
            np.asarray(det_r.center, dtype=np.float64),
            stereo_calib,
        )

    return left, right, det_l, det_r, xyz


def draw_stereo(left, right, det_l, det_r, xyz, lines):
    if left is None or right is None:
        return None
    left_draw = draw_detection(left, det_l, "LEFT")
    right_draw = draw_detection(right, det_r, "RIGHT")
    if xyz is not None:
        lines = list(lines) + [f"stereo XYZ=({xyz[0]:+.1f},{xyz[1]:+.1f},{xyz[2]:+.1f}) mm"]
    return make_preview(left_draw, right_draw, lines)


# ============================================================
# SAFETY CHECKS
# ============================================================

def require_soft_limits(robot: Robot):
    missing = []
    for name in ("check_cartesian_pose_safe", "plan_cartesian_path", "set_soft_limits_enabled"):
        if not hasattr(robot, name):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "This script requires the fail-closed soft-limit robot.py. "
            f"Missing method(s): {missing}. Do not run autonomous calibration."
        )

    ok = robot.set_soft_limits_enabled(True)
    if not ok:
        raise RuntimeError("Could not enable soft limits. Do not run autonomous calibration.")

    if not getattr(robot.cfg, "soft_limits_enabled", False):
        raise RuntimeError("robot.cfg.soft_limits_enabled is False after enable attempt. Aborting.")

    print("[SAFETY] Soft limits are enabled.")


def precheck_point(robot: Robot, x: float, y: float, z: float, phi: float):
    q = robot.ik(x, y, z, phi)
    reachable = q is not None
    if not reachable:
        return False, False, False, "IK unreachable"

    target_safe, target_reason = robot.check_cartesian_pose_safe(x, y, z)
    if not target_safe:
        return reachable, False, False, f"target unsafe: {target_reason}"

    path_safe, path, path_reason = robot.plan_cartesian_path(x, y, z)
    if not path_safe:
        return reachable, target_safe, False, f"path unsafe: {path_reason}"

    return reachable, target_safe, path_safe, "precheck ok"


# ============================================================
# MEASUREMENT
# ============================================================

def measure_current_pose(row, robot, overhead_cap, stereo, detector, K_overhead, dist_overhead, stereo_calib):
    overhead_raw_samples = []
    overhead_used_samples = []
    overhead_theta_samples = []
    stereo_samples = []
    deadline = time.time() + POINT_TIMEOUT_S

    while time.time() < deadline:
        frame, det_o = read_overhead_once(overhead_cap, detector)

        if frame is not None and det_o is not None:
            raw_uv = np.asarray(det_o.center, dtype=np.float64).reshape(2)
            used_uv = undistort_uv(raw_uv, K_overhead, dist_overhead)
            overhead_raw_samples.append(raw_uv)
            overhead_used_samples.append(used_uv)
            overhead_theta_samples.append(float(det_o.theta_deg))

        left = right = det_l = det_r = xyz = None
        if stereo is not None:
            left, right, det_l, det_r, xyz = read_stereo_once(stereo, detector, stereo_calib)
            if det_l is not None and det_r is not None:
                ul, vl = det_l.center
                ur, vr = det_r.center
                theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
                srow = [float(ul), float(vl), float(ur), float(vr), float(ul - ur), float(theta)]
                if xyz is not None:
                    srow += [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                else:
                    srow += [np.nan, np.nan, np.nan]
                stereo_samples.append(srow)

        n_o = len(overhead_raw_samples)
        n_s = len(stereo_samples)
        lines = [
            f"Calibrate all | {row.grid_name} | z={row.z_level_mm:.1f}",
            f"overhead samples {n_o}/{SAMPLES_PER_POINT} | stereo samples {n_s}",
            f"cmd=({row.x_cmd_mm:.1f},{row.y_cmd_mm:.1f},{row.z_cmd_mm:.1f})",
            "q/ESC abort | SPACE accept early if enough samples",
        ]

        if frame is not None:
            cv2.imshow(WINDOW_OVERHEAD, draw_overhead(frame, det_o, lines))

        if stereo is not None and left is not None and right is not None:
            preview = draw_stereo(
                left, right, det_l, det_r, xyz,
                [
                    f"Stereo logger | {row.grid_name} z={row.z_level_mm:.1f}",
                    f"EE visible L/R: {det_l is not None}/{det_r is not None}",
                    f"samples: {n_s}",
                ],
            )
            if preview is not None:
                cv2.imshow(WINDOW_STEREO, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == 32 and n_o >= MIN_OVERHEAD_SAMPLES:
            break
        if n_o >= SAMPLES_PER_POINT:
            break

    row.overhead_samples = len(overhead_raw_samples)
    row.overhead_visible = row.overhead_samples >= MIN_OVERHEAD_SAMPLES
    if row.overhead_visible:
        raw_mean = np.mean(np.stack(overhead_raw_samples), axis=0)
        used_mean = np.mean(np.stack(overhead_used_samples), axis=0)
        row.overhead_u_raw_px = float(raw_mean[0])
        row.overhead_v_raw_px = float(raw_mean[1])
        row.overhead_u_used_px = float(used_mean[0])
        row.overhead_v_used_px = float(used_mean[1])
        row.overhead_theta_deg = float(np.mean(overhead_theta_samples))

        # Reference poses are saved, but not used for homography fitting.
        row.overhead_used_for_homography = not row.is_reference_pose

    row.stereo_samples = len(stereo_samples)
    row.stereo_visible = row.stereo_samples >= MIN_STEREO_SAMPLES
    if row.stereo_visible:
        s = np.asarray(stereo_samples, dtype=np.float64)
        sm = np.nanmean(s, axis=0)
        row.stereo_left_u_px = float(sm[0])
        row.stereo_left_v_px = float(sm[1])
        row.stereo_right_u_px = float(sm[2])
        row.stereo_right_v_px = float(sm[3])
        row.stereo_disparity_px = float(sm[4])
        row.stereo_theta_deg = float(sm[5])
        if not np.isnan(sm[6]):
            row.stereo_x_mm = float(sm[6])
            row.stereo_y_mm = float(sm[7])
            row.stereo_z_mm = float(sm[8])

    return row


# ============================================================
# SCAN POINT GENERATION
# ============================================================

def generate_scan_points():
    x_grid, y_grid, z_levels = get_scan_grids()

    points = []
    for z in z_levels:
        for y in y_grid:
            for x in x_grid:
                name = f"z{z:.0f}_x{x:.0f}_y{y:.0f}"
                points.append((name, float(x), float(y), float(z), CAL_PHI_DEG, False))

    for name, x, y, z, phi in REFERENCE_POSES:
        points.append((name, float(x), float(y), float(z), float(phi), True))

    return points, x_grid, y_grid, z_levels


# ============================================================
# SAVE RAW CSV
# ============================================================

def save_raw_csv(rows):
    fields = list(ScanRow.__dataclass_fields__.keys())
    with OUT_RAW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    print(f"[SAVE] Raw scan CSV -> {OUT_RAW_CSV.resolve()}")


# ============================================================
# MODEL BUILDING
# ============================================================

def build_homography_layers(rows):
    valid_z = []
    Hs = []
    Hs_inv = []
    rms_by_z = []
    max_by_z = []
    n_by_z = []
    report_lines = ["HEIGHT-INDEXED OVERHEAD HOMOGRAPHY", "=" * 72]

    z_values = sorted(set(r.z_level_mm for r in rows if not r.is_reference_pose))

    for z in z_values:
        layer = [
            r for r in rows
            if abs(r.z_level_mm - z) < 1e-9
            and not r.is_reference_pose
            and r.overhead_used_for_homography
            and r.overhead_u_used_px is not None
        ]

        if len(layer) < MIN_POINTS_PER_HOMOGRAPHY:
            msg = f"z={z:.1f} mm: SKIP, only {len(layer)} valid overhead points"
            print("[H] " + msg)
            report_lines.append(msg)
            continue

        img_pts = np.array([[r.overhead_u_used_px, r.overhead_v_used_px] for r in layer], dtype=np.float32)
        robot_pts = np.array([[r.robot_x_mm, r.robot_y_mm] for r in layer], dtype=np.float32)

        H, _ = cv2.findHomography(img_pts, robot_pts, method=0)
        H_inv, _ = cv2.findHomography(robot_pts, img_pts, method=0)

        if H is None or H_inv is None:
            msg = f"z={z:.1f} mm: SKIP, cv2.findHomography failed"
            print("[H] " + msg)
            report_lines.append(msg)
            continue

        pred = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = pred - robot_pts
        err_norm = np.linalg.norm(err, axis=1)
        rms = float(np.sqrt(np.mean(err_norm ** 2)))
        maxerr = float(np.max(err_norm))

        valid_z.append(float(z))
        Hs.append(H)
        Hs_inv.append(H_inv)
        rms_by_z.append(rms)
        max_by_z.append(maxerr)
        n_by_z.append(len(layer))

        msg = f"z={z:.1f} mm: RMS={rms:.2f} mm MAX={maxerr:.2f} mm n={len(layer)}"
        print("[H] " + msg)
        report_lines.append(msg)

        if len(layer) < RECOMMENDED_POINTS_PER_HOMOGRAPHY:
            warn = f"  WARNING: n={len(layer)} is valid but sparse; recommended >= {RECOMMENDED_POINTS_PER_HOMOGRAPHY}"
            print("[H] " + warn)
            report_lines.append(warn)

        for r, e, en in zip(layer, err, err_norm):
            report_lines.append(f"  {r.grid_name:18s} ex={e[0]:+7.2f} ey={e[1]:+7.2f} |e|={en:6.2f}")

    if not Hs:
        return None, report_lines

    return {
        "z_levels_mm": np.asarray(valid_z, dtype=np.float64),
        "H_img_to_robot_by_z": np.stack(Hs, axis=0),
        "H_robot_to_img_by_z": np.stack(Hs_inv, axis=0),
        "homography_rms_error_mm": np.asarray(rms_by_z, dtype=np.float64),
        "homography_max_error_mm": np.asarray(max_by_z, dtype=np.float64),
        "homography_n_points": np.asarray(n_by_z, dtype=np.int32),
    }, report_lines


def fit_stereo_models(rows):
    """Fit stereo-camera <-> robot affine models.

    Old/legacy model:
      [robot_x, robot_y, 1] -> [cam_x, cam_y, cam_z]
      [cam_x, cam_y, cam_z, 1] -> [robot_x, robot_y]

    New/full XYZ model:
      [robot_x, robot_y, robot_z, 1] -> [cam_x, cam_y, cam_z]
      [cam_x, cam_y, cam_z, 1] -> [robot_x, robot_y, robot_z]

    For final visual servoing, use the inverse linear block:
      delta_robot = A_robot_from_cam_xyz_3x4[:, :3] @ delta_cam

    where:
      delta_cam = object_cam_xyz - ee_cam_xyz
    """
    valid = [
        r for r in rows
        if r.move_ok
        and r.stereo_visible
        and r.stereo_x_mm is not None
        and r.stereo_y_mm is not None
        and r.stereo_z_mm is not None
        and r.robot_x_mm is not None
        and r.robot_y_mm is not None
        and r.robot_z_mm is not None
    ]

    report_lines = ["", "GLOBAL STEREO FITS", "=" * 72]

    if len(valid) < 4:
        msg = f"Stereo fit skipped: only {len(valid)} valid stereo samples."
        print("[StereoFit] " + msg)
        report_lines.append(msg)
        return None, report_lines

    robot_xyz = np.array(
        [[r.robot_x_mm, r.robot_y_mm, r.robot_z_mm] for r in valid],
        dtype=np.float64,
    )
    robot_xyz = robot_xyz + STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM.reshape(1, 3)

    cam_xyz = np.array(
        [[r.stereo_x_mm, r.stereo_y_mm, r.stereo_z_mm] for r in valid],
        dtype=np.float64,
    )
    cam_xyz = cam_xyz + np.array(
        [STEREO_CAM_X_OFFSET_MM, STEREO_CAM_Y_OFFSET_MM, STEREO_CAM_Z_OFFSET_MM],
        dtype=np.float64,
    ).reshape(1, 3)

    # ------------------------------------------------------------------
    # Legacy XY-only fits, kept for compatibility with existing scripts.
    # ------------------------------------------------------------------
    robot_xy1 = np.column_stack([robot_xyz[:, 0], robot_xyz[:, 1], np.ones(len(valid))])
    B_xy_T, *_ = np.linalg.lstsq(robot_xy1, cam_xyz, rcond=None)
    B_xy = B_xy_T.T

    cam_pred_xy = robot_xy1 @ B_xy_T
    cam_res_xy = cam_pred_xy - cam_xyz
    cam_rmse_xy = float(np.sqrt(np.mean(np.sum(cam_res_xy**2, axis=1))))

    cam_xyz1 = np.column_stack([cam_xyz, np.ones(len(valid))])
    robot_xy = robot_xyz[:, :2]
    A_xy_T, *_ = np.linalg.lstsq(cam_xyz1, robot_xy, rcond=None)
    A_xy = A_xy_T.T

    robot_xy_pred = cam_xyz1 @ A_xy_T
    robot_xy_res = robot_xy_pred - robot_xy
    robot_xy_rmse = float(np.sqrt(np.mean(np.sum(robot_xy_res**2, axis=1))))

    # ------------------------------------------------------------------
    # Full XYZ fits. This is the important new model.
    # ------------------------------------------------------------------
    robot_xyz1 = np.column_stack([robot_xyz, np.ones(len(valid))])

    # cam_xyz = robot_xyz1 @ B_xyz.T
    B_xyz_T, *_ = np.linalg.lstsq(robot_xyz1, cam_xyz, rcond=None)
    B_xyz = B_xyz_T.T  # 3 x 4
    cam_pred_xyz = robot_xyz1 @ B_xyz_T
    cam_res_xyz = cam_pred_xyz - cam_xyz
    cam_xyz_rmse = float(np.sqrt(np.mean(np.sum(cam_res_xyz**2, axis=1))))

    # robot_xyz = cam_xyz1 @ A_xyz.T
    A_xyz_T, *_ = np.linalg.lstsq(cam_xyz1, robot_xyz, rcond=None)
    A_xyz = A_xyz_T.T  # 3 x 4
    robot_xyz_pred = cam_xyz1 @ A_xyz_T
    robot_xyz_res = robot_xyz_pred - robot_xyz
    robot_xyz_rmse = float(np.sqrt(np.mean(np.sum(robot_xyz_res**2, axis=1))))

    robot_x_axis_cam = B_xyz[:, 0].copy()
    robot_y_axis_cam = B_xyz[:, 1].copy()
    robot_z_axis_cam = B_xyz[:, 2].copy()

    def unit(v):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.full_like(v, np.nan)
        return v / n

    robot_axes_cam_unit = np.stack([
        unit(robot_x_axis_cam),
        unit(robot_y_axis_cam),
        unit(robot_z_axis_cam),
    ], axis=0)

    msg1 = f"legacy cam_xyz_from_robot_xy RMSE = {cam_rmse_xy:.2f} mm, n={len(valid)}"
    msg2 = f"legacy robot_xy_from_cam_xyz RMSE = {robot_xy_rmse:.2f} mm, n={len(valid)}"
    msg3 = f"FULL cam_xyz_from_robot_xyz RMSE = {cam_xyz_rmse:.2f} mm, n={len(valid)}"
    msg4 = f"FULL robot_xyz_from_cam_xyz RMSE = {robot_xyz_rmse:.2f} mm, n={len(valid)}"

    for msg in (msg1, msg2, msg3, msg4):
        print("[StereoFit] " + msg)
        report_lines.append(msg)

    if len(valid) < MIN_STEREO_XYZ_FIT_SAMPLES:
        warn = f"WARNING: full XYZ stereo fit has only {len(valid)} samples; recommended >= {MIN_STEREO_XYZ_FIT_SAMPLES}"
        print("[StereoFit] " + warn)
        report_lines.append(warn)

    if robot_xyz_rmse > WARN_STEREO_XYZ_FIT_RMSE_MM:
        warn = f"WARNING: robot_xyz_from_cam_xyz RMSE {robot_xyz_rmse:.2f} mm exceeds warning threshold {WARN_STEREO_XYZ_FIT_RMSE_MM:.2f} mm"
        print("[StereoFit] " + warn)
        report_lines.append(warn)

    report_lines.append("")
    report_lines.append("Conventions:")
    report_lines.append("  cam_xyz = stereo triangulated EE tag center, in stereo camera coordinates")
    report_lines.append("  robot_xyz = FK estimate of EE/tool pose, in robot coordinates")
    report_lines.append("  Runtime delta: delta_cam = object_cam - ee_cam")
    report_lines.append("  Runtime correction: delta_robot = A_robot_from_cam_xyz_3x4[:, :3] @ delta_cam")
    report_lines.append("")
    report_lines.append("B_cam_from_robot_xyz_3x4 maps [robot_x, robot_y, robot_z, 1] -> cam_xyz:")
    report_lines.append(str(B_xyz))
    report_lines.append("A_robot_from_cam_xyz_3x4 maps [cam_x, cam_y, cam_z, 1] -> robot_xyz:")
    report_lines.append(str(A_xyz))
    report_lines.append("Robot axes expressed in stereo-camera coordinates, columns from B:")
    report_lines.append(f"  robot +X in cam = {robot_x_axis_cam}")
    report_lines.append(f"  robot +Y in cam = {robot_y_axis_cam}")
    report_lines.append(f"  robot +Z in cam = {robot_z_axis_cam}")
    report_lines.append(f"  unit axes rows [X,Y,Z] =\n{robot_axes_cam_unit}")

    return {
        # Legacy compatibility:
        "stereo_valid_count": np.asarray([len(valid)], dtype=np.int32),
        "B_cam_from_robot_xy_3x3": B_xy,
        "A_robot_from_cam_xyz_2x4": A_xy,
        "stereo_cam_fit_rmse_mm": np.asarray([cam_rmse_xy], dtype=np.float64),
        "stereo_robot_fit_rmse_mm": np.asarray([robot_xy_rmse], dtype=np.float64),

        # New full-XYZ model:
        "B_cam_from_robot_xyz_3x4": B_xyz,
        "A_robot_from_cam_xyz_3x4": A_xyz,
        "B_cam_from_robot_xyz_linear_3x3": B_xyz[:, :3],
        "A_robot_from_cam_xyz_linear_3x3": A_xyz[:, :3],
        "stereo_cam_xyz_fit_rmse_mm": np.asarray([cam_xyz_rmse], dtype=np.float64),
        "stereo_robot_xyz_fit_rmse_mm": np.asarray([robot_xyz_rmse], dtype=np.float64),
        "robot_x_axis_cam": robot_x_axis_cam,
        "robot_y_axis_cam": robot_y_axis_cam,
        "robot_z_axis_cam": robot_z_axis_cam,
        "robot_axes_cam_unit_rows_xyz": robot_axes_cam_unit,
        "stereo_model_convention": np.asarray([
            "delta_robot = A_robot_from_cam_xyz_3x4[:, :3] @ (object_cam_xyz - ee_cam_xyz)"
        ]),
        "stereo_ee_tag_to_tool_offset_robot_mm": STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM,
        "stereo_cam_offsets_mm": np.asarray(
            [STEREO_CAM_X_OFFSET_MM, STEREO_CAM_Y_OFFSET_MM, STEREO_CAM_Z_OFFSET_MM],
            dtype=np.float64,
        ),
    }, report_lines


def build_visibility_arrays(rows):
    cols = [
        "z_level_mm", "x_cmd_mm", "y_cmd_mm", "z_cmd_mm",
        "is_reference_pose",
        "reachable_ik", "target_safe", "path_safe", "move_ok",
        "overhead_visible", "stereo_visible",
        "robot_x_mm", "robot_y_mm", "robot_z_mm",
        "overhead_u_used_px", "overhead_v_used_px",
        "stereo_x_mm", "stereo_y_mm", "stereo_z_mm",
    ]

    arr = []
    for r in rows:
        arr.append([
            r.z_level_mm, r.x_cmd_mm, r.y_cmd_mm, r.z_cmd_mm,
            float(r.is_reference_pose),
            float(r.reachable_ik), float(r.target_safe), float(r.path_safe), float(r.move_ok),
            float(r.overhead_visible), float(r.stereo_visible),
            np.nan if r.robot_x_mm is None else r.robot_x_mm,
            np.nan if r.robot_y_mm is None else r.robot_y_mm,
            np.nan if r.robot_z_mm is None else r.robot_z_mm,
            np.nan if r.overhead_u_used_px is None else r.overhead_u_used_px,
            np.nan if r.overhead_v_used_px is None else r.overhead_v_used_px,
            np.nan if r.stereo_x_mm is None else r.stereo_x_mm,
            np.nan if r.stereo_y_mm is None else r.stereo_y_mm,
            np.nan if r.stereo_z_mm is None else r.stereo_z_mm,
        ])

    return np.asarray(arr, dtype=np.float64), np.asarray(cols)


def build_support_arrays(rows):
    valid = [
        r for r in rows
        if r.move_ok
        and not r.is_reference_pose
        and r.overhead_visible
        and r.robot_x_mm is not None
        and r.overhead_u_used_px is not None
    ]

    if not valid:
        return {
            "support_robot_xyz_mm": np.zeros((0, 3), dtype=np.float64),
            "support_overhead_uv_px": np.zeros((0, 2), dtype=np.float64),
            "support_z_levels_mm": np.zeros((0,), dtype=np.float64),
        }

    support_robot_xyz = np.array([[r.robot_x_mm, r.robot_y_mm, r.robot_z_mm] for r in valid], dtype=np.float64)
    support_overhead_uv = np.array([[r.overhead_u_used_px, r.overhead_v_used_px] for r in valid], dtype=np.float64)
    support_z_levels = np.array([r.z_level_mm for r in valid], dtype=np.float64)

    return {
        "support_robot_xyz_mm": support_robot_xyz,
        "support_overhead_uv_px": support_overhead_uv,
        "support_z_levels_mm": support_z_levels,
    }


def build_reference_arrays(rows):
    refs = [
        r for r in rows
        if r.is_reference_pose
        and r.move_ok
        and r.overhead_visible
        and r.robot_x_mm is not None
        and r.overhead_u_used_px is not None
    ]

    names = np.array([r.grid_name for r in refs])
    robot_xyz = np.array([[r.robot_x_mm, r.robot_y_mm, r.robot_z_mm] for r in refs], dtype=np.float64) if refs else np.zeros((0, 3))
    overhead_uv = np.array([[r.overhead_u_used_px, r.overhead_v_used_px] for r in refs], dtype=np.float64) if refs else np.zeros((0, 2))

    return {
        "reference_names": names,
        "reference_robot_xyz_mm": robot_xyz,
        "reference_overhead_uv_px": overhead_uv,
    }


def save_bundle(
    rows,
    H_bundle,
    stereo_bundle,
    K_overhead,
    dist_overhead,
    used_overhead_intrinsics,
    soft_limits_json_text,
    x_grid,
    y_grid,
    z_levels,
    report_lines,
):
    visibility_table, visibility_cols = build_visibility_arrays(rows)

    save_dict = {
        "created_unix_time": np.asarray([time.time()], dtype=np.float64),
        "scan_mode": np.asarray([SCAN_MODE]),
        "x_grid_mm": np.asarray(x_grid, dtype=np.float64),
        "y_grid_mm": np.asarray(y_grid, dtype=np.float64),
        "requested_z_levels_mm": np.asarray(z_levels, dtype=np.float64),
        "requested_floor_heights_mm": np.asarray([robot_z_to_floor_height_mm(z) for z in z_levels], dtype=np.float64),
        "distance_from_origin_to_floor_plane_mm": np.asarray([DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM], dtype=np.float64),
        "l1_to_l2_height_difference_mm": np.asarray([L1_TO_L2_HEIGHT_DIFFERENCE_MM], dtype=np.float64),
        "j3_zero_to_ee_drop_mm": np.asarray([J3_ZERO_TO_EE_DROP_MM], dtype=np.float64),
        "robot_z_zero_floor_height_mm": np.asarray([robot_z_to_floor_height_mm(0.0)], dtype=np.float64),
        "stereo_ee_tag_to_tool_offset_robot_mm": STEREO_EE_TAG_TO_TOOL_OFFSET_ROBOT_MM,
        "stereo_cam_offsets_mm": np.asarray([STEREO_CAM_X_OFFSET_MM, STEREO_CAM_Y_OFFSET_MM, STEREO_CAM_Z_OFFSET_MM], dtype=np.float64),
        "min_stereo_xyz_fit_samples": np.asarray([MIN_STEREO_XYZ_FIT_SAMPLES], dtype=np.int32),
        "warn_stereo_xyz_fit_rmse_mm": np.asarray([WARN_STEREO_XYZ_FIT_RMSE_MM], dtype=np.float64),
        "ee_tag_id": np.asarray([EE_TAG_ID], dtype=np.int32),
        "overhead_index": np.asarray([OVERHEAD_INDEX], dtype=np.int32),
        "stereo_index": np.asarray([STEREO_INDEX], dtype=np.int32),
        "used_overhead_intrinsics": np.asarray([bool(used_overhead_intrinsics)]),
        "overhead_camera_matrix": np.asarray(K_overhead if K_overhead is not None else np.eye(3), dtype=np.float64),
        "overhead_dist_coeffs": np.asarray(dist_overhead if dist_overhead is not None else np.zeros((1, 5)), dtype=np.float64),
        "visibility_table": visibility_table,
        "visibility_columns": visibility_cols,
        "soft_limits_json": np.asarray([soft_limits_json_text]),
        "max_runtime_nearest_sample_dist_mm": np.asarray([MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM], dtype=np.float64),
        "max_runtime_homography_rms_mm": np.asarray([MAX_RUNTIME_HOMOGRAPHY_RMS_MM], dtype=np.float64),
    }

    save_dict.update(build_support_arrays(rows))
    save_dict.update(build_reference_arrays(rows))

    if H_bundle is not None:
        save_dict.update(H_bundle)
    if stereo_bundle is not None:
        save_dict.update(stereo_bundle)

    np.savez(OUT_BUNDLE, **save_dict)
    print(f"[SAVE] Bundle -> {OUT_BUNDLE.resolve()}")

    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[SAVE] Report -> {OUT_REPORT.resolve()}")


# ============================================================
# MAIN SCAN
# ============================================================

def main():
    print("\nOne-shot Safety-Gated Robot Calibration")
    print("---------------------------------------")
    print("This script scans a safe XY/Z grid, builds H(z), logs stereo, and saves a calibration bundle.")
    print("It must use the fail-closed robot.py with soft limits enabled.\n")
    print_startup_config("calibrate_all_safe_grid.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("calibrate_all_safe_grid.py")

    if not SOFT_LIMITS_PATH.exists():
        raise RuntimeError(f"Missing {SOFT_LIMITS_PATH.resolve()}. Refusing to run calibration without keep-out zones.")

    soft_limits_json_text = SOFT_LIMITS_PATH.read_text(encoding="utf-8")
    scan_points, x_grid, y_grid, z_levels = generate_scan_points()

    detector = build_detector()
    overhead_cap = None
    stereo = None
    robot = Robot(ROBOT_CONFIG, connect=True)
    rows: list[ScanRow] = []

    try:
        require_soft_limits(robot)

        K_overhead, dist_overhead, used_overhead_intrinsics = load_overhead_intrinsics(OVERHEAD_INTRINSICS_PATH)
        overhead_cap = open_overhead_camera()

        stereo_calib = None
        if ENABLE_STEREO_LOGGING:
            try:
                stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
                stereo = SimpleStereoCamera(
                    index=STEREO_INDEX,
                    width=STEREO_WIDTH,
                    height=STEREO_HEIGHT,
                    fps=STEREO_FPS,
                    fourcc=STEREO_FOURCC,
                )
            except Exception as exc:
                print(f"[Stereo] Could not open stereo camera/logging: {exc!r}")
                stereo = None

        cv2.namedWindow(WINDOW_OVERHEAD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_OVERHEAD, 960, 540)
        if stereo is not None:
            cv2.namedWindow(WINDOW_STEREO, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_STEREO, 1280, 520)

        robot.enable(True)
        robot.init_drivers()

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()
        if choice == "h":
            if not robot.home():
                print("HOME failed. Aborting.")
                return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps():
                print("POS sync failed. Aborting.")
                return
        elif choice == "a":
            robot.assume_homed()
            robot.print_estimate()
        else:
            print("Unknown choice. Aborting.")
            return

        total = len(scan_points)
        print(f"\n[SCAN] Mode: {SCAN_MODE}")
        print(f"[SCAN] Candidate points: {total}")
        print(f"       X={x_grid}")
        print(f"       Y={y_grid}")
        print(f"       Robot Z={z_levels}")
        print(f"       Floor heights={[round(robot_z_to_floor_height_mm(z), 1) for z in z_levels]}")
        print(f"       Reference poses={REFERENCE_POSES}")
        print("\nType YES to start autonomous safety-gated grid scan.")
        ans = input("> ").strip()
        if ans != "YES":
            print("[SCAN] Cancelled.")
            return

        for idx, (name, x, y, z, phi, is_reference) in enumerate(scan_points, start=1):
            print("\n" + "-" * 76)
            print(f"[{idx}/{total}] Candidate {name}: x={x:.1f}, y={y:.1f}, robot_z={z:.1f}, floor_h={robot_z_to_floor_height_mm(z):.1f}, ref={is_reference}")

            row = ScanRow(
                z_level_mm=float(z),
                grid_name=name,
                x_cmd_mm=float(x),
                y_cmd_mm=float(y),
                z_cmd_mm=float(z),
                floor_height_cmd_mm=float(robot_z_to_floor_height_mm(z)),
                is_reference_pose=bool(is_reference),
                reachable_ik=False,
                target_safe=False,
                path_safe=False,
                move_attempted=False,
                move_ok=False,
                reason="not processed",
            )

            reachable, target_safe, path_safe, reason = precheck_point(robot, float(x), float(y), float(z), float(phi))
            row.reachable_ik = bool(reachable)
            row.target_safe = bool(target_safe)
            row.path_safe = bool(path_safe)
            row.reason = reason

            if not (reachable and target_safe and path_safe):
                print(f"[SKIP] {name}: {reason}")
                rows.append(row)
                save_raw_csv(rows)
                continue

            row.move_attempted = True
            ok = robot.move_cartesian(
                x_mm=float(x),
                y_mm=float(y),
                z_mm=float(z),
                phi_deg=float(phi),
                move_time_s=MOVE_TIME_S,
            )
            row.move_ok = bool(ok)

            if not ok:
                row.reason = "move_cartesian returned False"
                print(f"[SKIP] {name}: move failed")
                rows.append(row)
                save_raw_csv(rows)
                continue

            time.sleep(SETTLE_S)

            if SYNC_AFTER_MOVE:
                robot.sync_estimate_from_teensy_steps()

            rx, ry, rz, rphi = robot.fk()
            row.robot_x_mm = float(rx)
            row.robot_y_mm = float(ry)
            row.robot_z_mm = float(rz)
            row.robot_phi_deg = float(rphi)

            row.reason = "move ok; measuring"
            row = measure_current_pose(
                row,
                robot,
                overhead_cap,
                stereo,
                detector,
                K_overhead,
                dist_overhead,
                stereo_calib,
            )

            if row.overhead_visible:
                row.reason = "valid overhead"
                print(
                    f"[OK] {name}: overhead uv=({row.overhead_u_used_px:.1f},{row.overhead_v_used_px:.1f}) "
                    f"robot=({row.robot_x_mm:.1f},{row.robot_y_mm:.1f},{row.robot_z_mm:.1f})"
                )
            else:
                row.overhead_used_for_homography = False
                row.reason = "move ok but overhead not visible"
                print(f"[NO OVERHEAD] {name}: not enough overhead samples")

            if row.stereo_visible:
                print(f"[STEREO] {name}: xyz=({row.stereo_x_mm},{row.stereo_y_mm},{row.stereo_z_mm})")

            rows.append(row)
            save_raw_csv(rows)

        H_bundle, H_report = build_homography_layers(rows)
        stereo_bundle, stereo_report = fit_stereo_models(rows)

        report_lines = []
        report_lines.append("ROBOT CALIBRATION BUNDLE REPORT")
        report_lines.append("=" * 72)
        report_lines.append(f"Generated: {time.ctime()}")
        report_lines.append(f"Scan mode: {SCAN_MODE}")
        report_lines.append("Floor-plane height convention:")
        report_lines.append(f"  DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM={DISTANCE_FROM_ORIGIN_TO_FLOOR_PLANE_MM}")
        report_lines.append(f"  L1_TO_L2_HEIGHT_DIFFERENCE_MM={L1_TO_L2_HEIGHT_DIFFERENCE_MM}")
        report_lines.append(f"  J3_ZERO_TO_EE_DROP_MM={J3_ZERO_TO_EE_DROP_MM}")
        report_lines.append(f"  EE_floor_height = {robot_z_to_floor_height_mm(0.0):.1f} + robot_z")
        report_lines.append(f"Rows scanned: {len(rows)}")
        report_lines.append(f"Moves OK: {sum(r.move_ok for r in rows)}")
        report_lines.append(f"Overhead visible: {sum(r.overhead_visible for r in rows)}")
        report_lines.append(f"Stereo visible: {sum(r.stereo_visible for r in rows)}")
        report_lines.append(f"Reference poses visible: {sum(r.is_reference_pose and r.overhead_visible for r in rows)}")
        report_lines.append("")
        report_lines.append("Runtime confidence metadata:")
        report_lines.append(f"  MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM={MAX_RUNTIME_NEAREST_SAMPLE_DIST_MM}")
        report_lines.append(f"  MAX_RUNTIME_HOMOGRAPHY_RMS_MM={MAX_RUNTIME_HOMOGRAPHY_RMS_MM}")
        report_lines.append("")
        report_lines.extend(H_report)
        report_lines.extend(stereo_report)
        report_lines.append("")
        report_lines.append("Soft limits JSON used:")
        report_lines.append(soft_limits_json_text)

        save_bundle(
            rows,
            H_bundle,
            stereo_bundle,
            K_overhead,
            dist_overhead,
            used_overhead_intrinsics,
            soft_limits_json_text,
            x_grid,
            y_grid,
            z_levels,
            report_lines,
        )

        print("\n[DONE] Calibration scan complete.")

    except KeyboardInterrupt:
        print("\n[ABORT] User aborted.")
        if rows:
            save_raw_csv(rows)
            print("[ABORT] Partial CSV saved.")
    finally:
        if overhead_cap is not None:
            overhead_cap.release()
        if stereo is not None:
            stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
