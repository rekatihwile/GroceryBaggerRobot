from __future__ import annotations

"""
test_calibration_bundle_live_xyz_pickplace_stereo_z.py

Live test for robot_calibration_bundle.npz using:
  - overhead camera for target XY through height-indexed H(z)
  - stereo triangulation for target tag Z / EE tag Z
  - full XYZ calibration matrix for stereo_cam_xyz -> robot_xyz
  - pick/place with only three Z geometry knobs:
        TAG_TO_EE_Z_MM
        HOVER_HEIGHT_MM
        GRASP_OFFSET_MM

Core convention:
  1. Stereo triangulates target AprilTag ID2 in camera coordinates.
  2. A_robot_from_cam_xyz_3x4 maps that camera XYZ into robot XYZ.
  3. Stereo triangulates EE AprilTag ID0.
  4. EE tag is above the true EE/tool point:
          ee_tool_z_from_stereo = ee_tag_robot_z + TAG_TO_EE_Z_MM
  5. Compare that to FK z to compute a local stereo-Z bias:
          stereo_z_bias = fk_z - ee_tool_z_from_stereo
  6. Correct target tag robot Z:
          target_tag_z_robot = raw_target_tag_z_robot + stereo_z_bias
  7. Use target_tag_z_robot for:
          overhead homography lookup Z
          hover Z = target_tag_z_robot + HOVER_HEIGHT_MM
          grasp Z = target_tag_z_robot + GRASP_OFFSET_MM

Keys:
  m     coarse hover move to ID2 using overhead XY + stereo Z
  k     PICK ID2
  n     save current FK as drop zone
  f     PLACE at saved drop zone
  o/l   open/close claw
  [/]   jog robot Z
  ,/.   jog phi/J4
  v     validate overhead EE mapping
  b     print geometry sanity check once
  r     print stereo XYZ matrix
  h/c/a home/sync/assume home
  e/d   enable/disable motors
  q     quit
"""

from pathlib import Path
import time
try:
    import msvcrt
except ImportError:
    msvcrt = None

import cv2
import numpy as np

from hardware.robot import Robot
from config.robot_config import (
    ROBOT_CONFIG,
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    print_startup_config,
    require_soft_limits_configured,
)
from config.camera_config import (
    EE_TAG_ID,
    TARGET_TAG_ID,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    OVERHEAD_HEIGHT,
    OVERHEAD_FPS,
    OVERHEAD_FOURCC,
    STEREO_INDEX,
)
from hardware.cameras.stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
    average_angles_deg,
)


# ============================================================
# FILES / WINDOWS
# ============================================================

BUNDLE_PATH = Path("robot_calibration_bundle.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")
WINDOW = "Calibration Bundle Live Test - Stereo Z PickPlace"


# ============================================================
# ONLY Z GEOMETRY KNOBS YOU SHOULD NEED
# ============================================================

# EE AprilTag ID0 is mounted above the actual end-effector/tool point.
# If the tag center is about 180 mm ABOVE the EE/tool point in robot +Z,
# then:
#     ee_tool_z = ee_tag_z + TAG_TO_EE_Z_MM
# so this should usually be negative.
TAG_TO_EE_Z_MM = 0

# Hover height of the true EE/tool point above target tag/top surface.
# Desired hover:


# Grasp height of the true EE/tool point relative to target tag/top surface.
# Desired grasp:
#     z_ee_grasp = target_tag_z_robot + GRASP_OFFSET_MM
#
# Start conservative. If the claw needs to go below the tag/top, make this negative.
GRASP_OFFSET_MM = 115
#     z_ee_hover = target_tag_z_robot + HOVER_HEIGHT_MM
HOVER_HEIGHT_MM = GRASP_OFFSET_MM + 50.0 + 50
# If the target/object tag is physically above/below the object grasp surface, encode
# that directly in GRASP_OFFSET_MM, not with extra hidden offsets.


# ============================================================
# MOTION / CLAW KNOBS
# ============================================================

MIN_COARSE_TRAVEL_Z_MM = 50.0
COARSE_MOVE_TIME_S = 1.10
PICK_MOVE_TIME_S = 0.75

Z_JOG_MM = 5.0
PHI_JOG_DEG = 5.0

CLAW_OPEN_DEG = 75
CLAW_CLOSED_DEG = 5
CLAW_SETTLE_S = 0.30

PHI_OFFSET_DEG = 0.0

# Safety gates.
REFUSE_PICK_IF_TARGET_FAR = True
REFUSE_PICK_WITHOUT_STEREO_Z = True
REFUSE_PLACE_IF_NO_DROP_ZONE = True

# Runtime acceptance thresholds.
WARN_EE_ERROR_MM = 15.0
MAX_EE_ERROR_MM = 30.0
MAX_NEAREST_SAMPLE_DIST_MM_FALLBACK = 200.0


# ============================================================
# STEREO / MATRIX KNOBS
# ============================================================

# Use the current EE stereo triangulation + FK to remove local stereo-Z bias.
# This is the important "relative Z from current triangulated EE pos + FK" correction.
USE_EE_FK_Z_BIAS_CORRECTION = True

# Clamp target lookup Z to bundle z range. Keep this False so stereo-derived
# negative object heights can still affect the overhead XY correction.
CLAMP_STEREO_LOOKUP_Z_TO_BUNDLE = False

# Print sanity checks.
PRINT_GEOMETRY_SANITY = False
GEOMETRY_SANITY_PRINT_EVERY_SEC = 0.75


# ============================================================
# CAMERA HELPERS
# ============================================================

def open_overhead_camera():
    cap = cv2.VideoCapture(OVERHEAD_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open overhead camera index {OVERHEAD_INDEX}")

    if OVERHEAD_FOURCC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*OVERHEAD_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(OVERHEAD_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(OVERHEAD_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, int(OVERHEAD_FPS))

    print(
        f"[Overhead] index={OVERHEAD_INDEX} actual "
        f"{cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
        f"@{cap.get(cv2.CAP_PROP_FPS):.1f}"
    )
    return cap


def open_stereo_camera():
    stereo = SimpleStereoCamera(index=STEREO_INDEX)
    return stereo


# ============================================================
# BUNDLE / CALIBRATION LOADING
# ============================================================

def load_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing bundle: {path.resolve()}")

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    required = ["z_levels_mm", "H_img_to_robot_by_z"]
    for k in required:
        if k not in keys:
            raise KeyError(f"Bundle missing {k}. Did homography build fail? Keys={data.files}")

    z_levels = np.asarray(data["z_levels_mm"], dtype=np.float64)
    Hs = np.asarray(data["H_img_to_robot_by_z"], dtype=np.float64)

    used_intrinsics = bool(np.asarray(data.get("used_overhead_intrinsics", [False])).reshape(-1)[0])
    K = np.asarray(data["overhead_camera_matrix"], dtype=np.float64) if used_intrinsics else None
    dist = np.asarray(data["overhead_dist_coeffs"], dtype=np.float64) if used_intrinsics else None

    support_xyz = np.asarray(data.get("support_robot_xyz_mm", np.zeros((0, 3))), dtype=np.float64)
    support_uv = np.asarray(data.get("support_overhead_uv_px", np.zeros((0, 2))), dtype=np.float64)

    # if "max_runtime_nearest_sample_dist_mm" in keys:
    #     max_nearest = float(np.asarray(data["max_runtime_nearest_sample_dist_mm"]).reshape(-1)[0])
    # else:
    max_nearest = MAX_NEAREST_SAMPLE_DIST_MM_FALLBACK

    rms = np.asarray(data.get("homography_rms_error_mm", np.full(len(z_levels), np.nan)), dtype=np.float64)
    n_pts = np.asarray(data.get("homography_n_points", np.zeros(len(z_levels))), dtype=np.int32)

    A_xyz = np.asarray(data["A_robot_from_cam_xyz_3x4"], dtype=np.float64) if "A_robot_from_cam_xyz_3x4" in keys else None
    B_xyz = np.asarray(data["B_cam_from_robot_xyz_3x4"], dtype=np.float64) if "B_cam_from_robot_xyz_3x4" in keys else None
    A_lin = np.asarray(data["A_robot_from_cam_xyz_linear_3x3"], dtype=np.float64) if "A_robot_from_cam_xyz_linear_3x3" in keys else (A_xyz[:, :3] if A_xyz is not None else None)
    B_lin = np.asarray(data["B_cam_from_robot_xyz_linear_3x3"], dtype=np.float64) if "B_cam_from_robot_xyz_linear_3x3" in keys else (B_xyz[:, :3] if B_xyz is not None else None)
    xyz_rmse = float(np.asarray(data["stereo_robot_xyz_fit_rmse_mm"]).reshape(-1)[0]) if "stereo_robot_xyz_fit_rmse_mm" in keys else np.nan

    print(f"[Bundle] Loaded {path.resolve()}")
    print(f"[Bundle] z_levels={z_levels}")
    print(f"[Bundle] used_intrinsics={used_intrinsics}")
    print(f"[Bundle] homography RMS={rms}")
    print(f"[Bundle] homography n={n_pts}")
    print(f"[Bundle] support samples={len(support_xyz)}")
    print(f"[Bundle] max nearest sample dist={max_nearest:.1f} mm")
    print(f"[Bundle] full stereo XYZ model={'YES' if A_xyz is not None else 'NO'}")
    if A_xyz is not None:
        print(f"[Bundle] stereo robot_xyz_from_cam_xyz RMSE={xyz_rmse:.2f} mm")
        print("[Bundle] convention: robot_xyz = A_robot_from_cam_xyz_3x4 @ [cam_x,cam_y,cam_z,1]")

    return {
        "data": data,
        "z_levels": z_levels,
        "Hs": Hs,
        "K": K,
        "dist": dist,
        "used_intrinsics": used_intrinsics,
        "support_xyz": support_xyz,
        "support_uv": support_uv,
        "max_nearest": max_nearest,
        "rms": rms,
        "n_pts": n_pts,
        "A_robot_from_cam_xyz_3x4": A_xyz,
        "B_cam_from_robot_xyz_3x4": B_xyz,
        "A_robot_from_cam_xyz_linear_3x3": A_lin,
        "B_cam_from_robot_xyz_linear_3x3": B_lin,
        "stereo_robot_xyz_fit_rmse_mm": xyz_rmse,
    }


def load_stereo_calibration(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing stereo calibration: {path.resolve()}")
    data = np.load(path, allow_pickle=False)
    return {k: np.asarray(data[k]) for k in data.files}


# ============================================================
# STEREO TRIANGULATION
# ============================================================

class TriangulatedTag:
    def __init__(self, tag_id: int, xyz_cam_mm, left_px, right_px, disparity_px: float, theta_deg: float):
        self.tag_id = int(tag_id)
        self.xyz_cam_mm = np.asarray(xyz_cam_mm, dtype=np.float64).reshape(3)
        self.left_px = np.asarray(left_px, dtype=np.float64).reshape(2)
        self.right_px = np.asarray(right_px, dtype=np.float64).reshape(2)
        self.disparity_px = float(disparity_px)
        self.theta_deg = float(theta_deg)


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

    # Match existing project convention from calibration script.
    xyz[2] *= -1.0
    return xyz


def read_stereo_tags_once(stereo: SimpleStereoCamera, detector, stereo_calib):
    """Read a stereo pair, detect tags on each, and triangulate matching IDs.

    Returns:
      out       - dict[tag_id -> TriangulatedTag] with xyz in stereo cam coords (mm)
      left      - left camera frame (BGR) or None
      right     - right camera frame (BGR) or None
      det_l_all - dict[tag_id -> Detection] from left frame (for drawing)
      det_r_all - dict[tag_id -> Detection] from right frame (for drawing)
    """
    ok, _, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        return {}, None, None, {}, {}

    det_l_all = detect_tags(detector, left)
    det_r_all = detect_tags(detector, right)

    out: dict[int, TriangulatedTag] = {}
    for tag_id in sorted(set(det_l_all.keys()) & set(det_r_all.keys())):
        det_l = det_l_all[tag_id]
        det_r = det_r_all[tag_id]
        xyz = triangulate_center_raw(det_l.center, det_r.center, stereo_calib)
        theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
        out[tag_id] = TriangulatedTag(
            tag_id=tag_id,
            xyz_cam_mm=xyz,
            left_px=det_l.center,
            right_px=det_r.center,
            disparity_px=float(det_l.center[0] - det_r.center[0]),
            theta_deg=theta,
        )

    return out, left, right, det_l_all, det_r_all


def cam_xyz_to_robot_xyz(cam_xyz, bundle):
    A = bundle.get("A_robot_from_cam_xyz_3x4")
    if A is None:
        raise RuntimeError("Bundle has no A_robot_from_cam_xyz_3x4. Re-run XYZ calibration.")
    cam_h = np.array([float(cam_xyz[0]), float(cam_xyz[1]), float(cam_xyz[2]), 1.0], dtype=np.float64)
    return (A @ cam_h).astype(np.float64)


# ============================================================
# HOMOGRAPHY HELPERS
# ============================================================

def undistort_uv(uv_raw, K, dist):
    uv_raw = np.asarray(uv_raw, dtype=np.float64).reshape(2)
    if K is None or dist is None:
        return uv_raw
    pt = uv_raw.astype(np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def apply_H(uv, H):
    pt = np.asarray(uv, dtype=np.float32).reshape(1, 1, 2)
    return cv2.perspectiveTransform(pt, H)[0, 0].astype(np.float64)


def _normalize_key_code(raw: int):
    """Return a normalized key char/name from an OpenCV key code, or None."""
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        return None
    if raw < 0:
        return None

    code = raw & 0xFF
    if code in (27,):
        return "escape"
    if code in (8, 255):
        return None
    if 0 <= code <= 127:
        return chr(code).lower()

    special = {
        0x250000: "left",
        0x260000: "up",
        0x270000: "right",
        0x280000: "down",
    }
    return special.get(raw)


def read_cv2_key(delay_ms: int = 1):
    """Return a normalized key char/name from the OpenCV window, or None."""
    try:
        raw = cv2.waitKeyEx(delay_ms)
    except AttributeError:
        raw = cv2.waitKey(delay_ms)
    return _normalize_key_code(raw)


def read_console_key_nonblocking():
    """Return a normalized key from the Windows console, or None."""
    if msvcrt is None or not msvcrt.kbhit():
        return None

    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        if not msvcrt.kbhit():
            return None
        ext = msvcrt.getwch()
        return {
            "K": "left",
            "H": "up",
            "M": "right",
            "P": "down",
        }.get(ext)
    if ch == "\x1b":
        return "escape"
    if len(ch) == 1:
        return ch.lower()
    return None


def read_command_key(delay_ms: int = 1):
    return read_cv2_key(delay_ms) or read_console_key_nonblocking()


def require_nonnegative_robot_z(z_mm: float, context: str):
    z = float(z_mm)
    if z < 0.0:
        raise RuntimeError(
            f"{context} would command robot Z={z:.1f} mm. "
            "Negative Z is allowed only for homography lookup / stereo delta math, not final joint motion."
        )


def move_cartesian_nonnegative_z(robot, context: str, x_mm=None, y_mm=None, z_mm=None, phi_deg=None, move_time_s=None):
    if z_mm is not None:
        require_nonnegative_robot_z(float(z_mm), context)
    return robot.move_cartesian(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, phi_deg=phi_deg, move_time_s=move_time_s)


def jog_nonnegative_z(robot, context: str, dx=0.0, dy=0.0, dz=0.0, dphi=0.0, move_time_s=None):
    x, y, z, phi = robot.fk()
    target_z = float(z + dz)
    require_nonnegative_robot_z(target_z, context)
    return robot.move_cartesian(
        x_mm=x + dx,
        y_mm=y + dy,
        z_mm=target_z,
        phi_deg=phi + dphi,
        move_time_s=move_time_s,
    )


def clamp_lookup_z_to_bundle(z_query: float, bundle):
    z_levels = bundle["z_levels"]
    z_raw = float(z_query)
    if not CLAMP_STEREO_LOOKUP_Z_TO_BUNDLE:
        return z_raw, False
    z_clamped = float(np.clip(z_raw, float(z_levels[0]), float(z_levels[-1])))
    was_clamped = abs(z_clamped - z_raw) > 1e-9
    return z_clamped, was_clamped


def map_uv_z_to_robot_xy(uv_raw, z_query, bundle):
    uv = undistort_uv(uv_raw, bundle["K"], bundle["dist"])
    z_levels = bundle["z_levels"]
    Hs = bundle["Hs"]
    zq, _clamped = clamp_lookup_z_to_bundle(float(z_query), bundle)

    if len(z_levels) == 0:
        raise RuntimeError("No z levels in bundle.")

    if zq <= z_levels[0]:
        if len(z_levels) >= 2 and abs(float(z_levels[1] - z_levels[0])) > 1e-9:
            lo, hi = 0, 1
            alpha = float((zq - z_levels[lo]) / (z_levels[hi] - z_levels[lo]))
            xy_lo = apply_H(uv, Hs[lo])
            xy_hi = apply_H(uv, Hs[hi])
            xy = (1.0 - alpha) * xy_lo + alpha * xy_hi
            return xy, uv, lo, hi, alpha
        xy = apply_H(uv, Hs[0])
        return xy, uv, 0, 0, 0.0

    if zq >= z_levels[-1]:
        if len(z_levels) >= 2 and abs(float(z_levels[-1] - z_levels[-2])) > 1e-9:
            lo, hi = len(z_levels) - 2, len(z_levels) - 1
            alpha = float((zq - z_levels[lo]) / (z_levels[hi] - z_levels[lo]))
            xy_lo = apply_H(uv, Hs[lo])
            xy_hi = apply_H(uv, Hs[hi])
            xy = (1.0 - alpha) * xy_lo + alpha * xy_hi
            return xy, uv, lo, hi, alpha
        i = len(z_levels) - 1
        xy = apply_H(uv, Hs[i])
        return xy, uv, i, i, 0.0

    hi = int(np.searchsorted(z_levels, zq))
    lo = hi - 1
    alpha = float((zq - z_levels[lo]) / (z_levels[hi] - z_levels[lo]))

    xy_lo = apply_H(uv, Hs[lo])
    xy_hi = apply_H(uv, Hs[hi])
    xy = (1.0 - alpha) * xy_lo + alpha * xy_hi
    return xy, uv, lo, hi, alpha


def nearest_support_distance(xy, z, bundle):
    support = bundle["support_xyz"]
    if support.size == 0:
        return np.inf, -1

    query = np.array([float(xy[0]), float(xy[1]), float(z)], dtype=np.float64)
    d = np.linalg.norm(support - query.reshape(1, 3), axis=1)
    idx = int(np.argmin(d))
    return float(d[idx]), idx


# ============================================================
# GEOMETRY COMPUTATION
# ============================================================

def compute_stereo_target_geometry(robot, bundle, stereo_tags):
    """Compute target tag robot Z from stereo, using EE stereo + FK as local bias.

    Returns dict with:
      ee_tag_robot_xyz_raw
      ee_tool_z_from_stereo_raw
      stereo_z_bias_mm
      target_tag_robot_xyz_raw
      target_tag_robot_xyz_corrected
      lookup_z_used
      lookup_z_clamped
      hover_robot_z
      grasp_robot_z
    """
    x_fk, y_fk, z_fk, phi_fk = robot.fk()

    ee_tri = stereo_tags.get(EE_TAG_ID)
    tg_tri = stereo_tags.get(TARGET_TAG_ID)

    if tg_tri is None:
        return None

    target_raw = cam_xyz_to_robot_xyz(tg_tri.xyz_cam_mm, bundle)
    target_corr = target_raw.copy()

    ee_raw = None
    ee_tool_z_raw = None
    stereo_z_bias = 0.0

    if ee_tri is not None:
        ee_raw = cam_xyz_to_robot_xyz(ee_tri.xyz_cam_mm, bundle)
        ee_tool_z_raw = float(ee_raw[2] + TAG_TO_EE_Z_MM)
        if USE_EE_FK_Z_BIAS_CORRECTION:
            stereo_z_bias = float(z_fk - ee_tool_z_raw)
            target_corr[2] += stereo_z_bias

    lookup_z_used, lookup_z_clamped = clamp_lookup_z_to_bundle(float(target_corr[2]), bundle)
    hover_robot_z = float(target_corr[2] + HOVER_HEIGHT_MM)
    grasp_robot_z = float(target_corr[2] + GRASP_OFFSET_MM)

    return {
        "fk_xyz": np.array([x_fk, y_fk, z_fk], dtype=np.float64),
        "fk_phi": float(phi_fk),
        "ee_cam_xyz": None if ee_tri is None else ee_tri.xyz_cam_mm.copy(),
        "target_cam_xyz": tg_tri.xyz_cam_mm.copy(),
        "ee_tag_robot_xyz_raw": ee_raw,
        "ee_tool_z_from_stereo_raw": ee_tool_z_raw,
        "stereo_z_bias_mm": stereo_z_bias,
        "target_tag_robot_xyz_raw": target_raw,
        "target_tag_robot_xyz_corrected": target_corr,
        "lookup_z_used": lookup_z_used,
        "lookup_z_clamped": lookup_z_clamped,
        "hover_robot_z": hover_robot_z,
        "grasp_robot_z": grasp_robot_z,
    }


def _fmt_xyz(v, decimals: int = 1) -> str:
    """Format a 3-vector as [x=..., y=..., z=...] mm with signed values."""
    if v is None:
        return "  (not available)"
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"[x={a[0]:+8.{decimals}f}, y={a[1]:+8.{decimals}f}, z={a[2]:+8.{decimals}f}] mm"


def _fmt_dxyz(v, decimals: int = 1) -> str:
    """Format a 3-vector delta as [dx=..., dy=..., dz=...] mm."""
    if v is None:
        return "  (not available)"
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"[dx={a[0]:+8.{decimals}f}, dy={a[1]:+8.{decimals}f}, dz={a[2]:+8.{decimals}f}] mm"


def print_geometry_sanity(geom, prefix="[GEOM]"):
    """Sectioned sanity dump: camera frame, robot frame, deltas, offsets, commands.

    Key thing to read: section [3] -> '** dz EE->Target **' is the relative
    height difference in the robot frame between EE tag (ID0) and Target tag
    (ID2). That number should match a tape measure with sign convention:
      negative dz  => Target tag is BELOW EE tag in robot +Z.
      positive dz  => Target tag is ABOVE EE tag in robot +Z.
    """
    if geom is None:
        print(f"{prefix} No target stereo geometry available.")
        return

    fk_xyz      = geom["fk_xyz"]                              # robot frame, from joint angles
    ee_cam      = geom["ee_cam_xyz"]                          # stereo cam frame, raw triangulation of ID0
    tg_cam      = geom["target_cam_xyz"]                      # stereo cam frame, raw triangulation of ID2
    ee_robot    = geom["ee_tag_robot_xyz_raw"]                # robot frame, ID0 via A_robot_from_cam_xyz
    ee_tool_z   = geom["ee_tool_z_from_stereo_raw"]           # robot frame Z, ID0 + TAG_TO_EE_Z_MM
    bias        = geom["stereo_z_bias_mm"]                    # robot Z bias = FK_z - EE_tool_z_stereo
    tg_robot    = geom["target_tag_robot_xyz_raw"]            # robot frame, ID2 via A (no bias yet)
    tg_robot_c  = geom["target_tag_robot_xyz_corrected"]      # robot frame, ID2 with +bias applied to Z
    hover_z     = geom["hover_robot_z"]                       # robot Z command, target_z + HOVER
    grasp_z     = geom["grasp_robot_z"]                       # robot Z command, target_z + GRASP
    lookup_z    = geom["lookup_z_used"]                       # Z passed into overhead homography stack
    lookup_clamp= geom["lookup_z_clamped"]                    # True if clamped to bundle z range

    print("\n" + "=" * 78)
    print(f"{prefix} Stereo/FK geometry sanity check")
    print("-" * 78)

    # ---------------- [1] CAMERA FRAME ----------------
    print("[1] CAMERA FRAME  (stereo cam coords, mm) -- raw stereo triangulation")
    print(f"    EE tag (ID0)         = {_fmt_xyz(ee_cam)}")
    print(f"    Target tag (ID2)     = {_fmt_xyz(tg_cam)}")
    if ee_cam is not None and tg_cam is not None:
        d_cam = np.asarray(tg_cam) - np.asarray(ee_cam)
        print(f"    Target - EE (cam)    = {_fmt_dxyz(d_cam)}")
    print()

    # ---------------- [2] ROBOT FRAME ----------------
    print("[2] ROBOT FRAME   (after A_robot_from_cam_xyz transform, mm)")
    print(f"    FK estimate          = {_fmt_xyz(fk_xyz)}                <- forward kinematics from joints")
    print(f"    EE tag (ID0)         = {_fmt_xyz(ee_robot)}                <- stereo -> robot")
    print(f"    Target tag (ID2) raw = {_fmt_xyz(tg_robot)}                <- stereo -> robot (no Z bias)")
    print(f"    Target tag corrected = {_fmt_xyz(tg_robot_c)}                <- raw + stereo_Z_bias")
    print()

    # ---------------- [3] RELATIVE DELTAS (robot frame) ----------------
    print("[3] RELATIVE DELTAS  (robot frame, mm)  *** the key numbers ***")
    if ee_robot is not None:
        d_fk_ee = np.asarray(ee_robot) - np.asarray(fk_xyz)
        print(f"    EE_tag - FK          = {_fmt_dxyz(d_fk_ee)}  <- stereo agreement with FK at EE")

    if ee_robot is not None and tg_robot is not None:
        d_raw = np.asarray(tg_robot) - np.asarray(ee_robot)
        d_corr = np.asarray(tg_robot_c) - np.asarray(ee_robot)
        print(f"    Target - EE_tag raw  = {_fmt_dxyz(d_raw)}")
        print(f"    Target - EE_tag corr = {_fmt_dxyz(d_corr)}")
        sign = "BELOW" if d_corr[2] < 0 else "ABOVE"
        print(f"    ** dz EE -> Target   = {d_corr[2]:+8.1f} mm  -> Target is {sign} EE tag by {abs(d_corr[2]):.1f} mm **")
    elif tg_robot is not None:
        print(f"    Target - FK (robot)  = {_fmt_dxyz(np.asarray(tg_robot_c) - np.asarray(fk_xyz))}")
        print("    (EE tag not visible in stereo -- no EE_tag based delta)")
    print()

    # ---------------- [4] Z OFFSETS / BIAS ----------------
    print("[4] Z OFFSETS / BIAS  (robot Z, mm)")
    print(f"    TAG_TO_EE_Z_MM       = {TAG_TO_EE_Z_MM:+8.2f}            <- EE_tool_z = EE_tag_z + TAG_TO_EE_Z_MM")
    if ee_tool_z is not None:
        print(f"    EE tool z (stereo)   = {ee_tool_z:+8.2f}            <- EE_tag.z + TAG_TO_EE_Z_MM")
        print(f"    FK z                 = {fk_xyz[2]:+8.2f}            <- forward kinematics")
        print(f"    Stereo Z bias        = {bias:+8.2f}            <- FK_z - EE_tool_z_stereo  (added to target Z)")
    else:
        print("    EE tag not visible -- no stereo Z bias correction available.")
    print()

    # ---------------- [5] COMMANDS ----------------
    print("[5] COMMANDS  (robot Z, mm)")
    print(f"    HOVER_HEIGHT_MM      = {HOVER_HEIGHT_MM:+8.2f}            <- hover = target_z + HOVER")
    print(f"    GRASP_OFFSET_MM      = {GRASP_OFFSET_MM:+8.2f}            <- grasp = target_z + GRASP")
    print(f"    Target z (corrected) = {tg_robot_c[2]:+8.2f}")
    print(f"    Hover  z             = {hover_z:+8.2f}")
    print(f"    Grasp  z             = {grasp_z:+8.2f}")
    print(f"    Lookup z (overhead H)= {lookup_z:+8.2f}  {'<- CLAMPED to bundle range' if lookup_clamp else ''}")
    dz = grasp_z - fk_xyz[2]
    arrow = "DOWN" if dz < 0 else "UP"
    print(f"    dz FK -> grasp       = {dz:+8.2f}            <- robot will move {arrow} {abs(dz):.1f} mm")
    print("=" * 78 + "\n")


# ============================================================
# ROBOT SAFETY / DRAWING
# ============================================================

def require_soft_limits(robot: Robot):
    missing = []
    for name in ("check_cartesian_pose_safe", "plan_cartesian_path", "set_soft_limits_enabled"):
        if not hasattr(robot, name):
            missing.append(name)
    if missing:
        raise RuntimeError(f"robot.py missing soft-limit method(s): {missing}")
    if not robot.set_soft_limits_enabled(True):
        raise RuntimeError("Could not enable soft limits.")
    print("[SAFETY] Soft limits enabled.")


def draw_status(frame, det_ee, det_tg, lines):
    out = frame.copy()
    out = draw_detection(out, det_ee, f"EE {EE_TAG_ID}")

    if det_tg is not None:
        corners_i = np.round(det_tg.corners).astype(int)
        center_i = tuple(np.round(det_tg.center).astype(int))
        cv2.polylines(out, [corners_i], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, center_i, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(out, f"TARGET {TARGET_TAG_ID}", (center_i[0] + 8, center_i[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"TARGET {TARGET_TAG_ID}", (center_i[0] + 8, center_i[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    for i, line in enumerate(lines):
        y = 30 + i * 27
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ============================================================
# COMBINED CAMERA VIEW
# ============================================================
# Layout (single window):
#   +------------------------------------------+
#   |              OVERHEAD                     |
#   |             (annotated)                   |
#   +---------------------+--------------------+
#   |   STEREO LEFT       |   STEREO RIGHT     |
#   |   (annotated)       |   (annotated)      |
#   +---------------------+--------------------+

COMBINED_WIDTH_PX   = 1280   # final composed image width
OVERHEAD_DRAW_H_PX  = 600    # height to render overhead at (width = COMBINED_WIDTH_PX, aspect preserved by scale)
STEREO_DRAW_H_PX    = 380    # height to render each stereo half at

def _draw_stereo_pane(img, det_dict, side_label: str):
    """Draw detected tags on a stereo half-frame and label the pane."""
    if img is None:
        return None
    out = img.copy()
    for tag_id, det in det_dict.items():
        label = f"EE {tag_id}" if tag_id == EE_TAG_ID else (f"TGT {tag_id}" if tag_id == TARGET_TAG_ID else f"ID{tag_id}")
        out = draw_detection(out, det, label)
    cv2.putText(out, side_label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, side_label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def compose_camera_views(overhead_annotated, stereo_left, stereo_right,
                         det_l_all=None, det_r_all=None):
    """Return a single combined image: overhead on top, stereo pair on bottom.

    Inputs:
      overhead_annotated - overhead frame already annotated with overlay text/tags
      stereo_left/right  - raw stereo frames (drawing of tags happens here)
      det_l_all/det_r_all- per-image detection dicts for drawing
    """
    # Top pane: overhead, scaled to COMBINED_WIDTH_PX wide x OVERHEAD_DRAW_H_PX tall.
    if overhead_annotated is not None:
        top = cv2.resize(overhead_annotated, (COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX))
    else:
        top = np.zeros((OVERHEAD_DRAW_H_PX, COMBINED_WIDTH_PX, 3), dtype=np.uint8)
        cv2.putText(top, "Overhead camera not available", (20, OVERHEAD_DRAW_H_PX // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    # Bottom pane: stereo left + right side-by-side, each COMBINED_WIDTH_PX/2 wide.
    half_w = COMBINED_WIDTH_PX // 2

    if stereo_left is not None:
        left_drawn = _draw_stereo_pane(stereo_left, det_l_all or {}, "STEREO LEFT")
        bot_l = cv2.resize(left_drawn, (half_w, STEREO_DRAW_H_PX))
    else:
        bot_l = np.zeros((STEREO_DRAW_H_PX, half_w, 3), dtype=np.uint8)
        cv2.putText(bot_l, "stereo left N/A", (20, STEREO_DRAW_H_PX // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    if stereo_right is not None:
        right_drawn = _draw_stereo_pane(stereo_right, det_r_all or {}, "STEREO RIGHT")
        bot_r = cv2.resize(right_drawn, (half_w, STEREO_DRAW_H_PX))
    else:
        bot_r = np.zeros((STEREO_DRAW_H_PX, half_w, 3), dtype=np.uint8)
        cv2.putText(bot_r, "stereo right N/A", (20, STEREO_DRAW_H_PX // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    bottom = np.hstack([bot_l, bot_r])
    # Thin separator strip between top and bottom.
    sep = np.full((2, COMBINED_WIDTH_PX, 3), 80, dtype=np.uint8)
    return np.vstack([top, sep, bottom])


# ============================================================
# LABELED MATRIX DUMP (for 'r' command)
# ============================================================

def print_matrix_labeled(name: str, M: np.ndarray, row_labels, col_labels, units: str = "mm"):
    """Pretty-print a small matrix with row and column labels for debugging."""
    if M is None:
        print(f"{name}: (not available)")
        return
    M = np.asarray(M, dtype=np.float64)
    rows, cols = M.shape
    print(f"{name}  (units: {units}, shape {rows}x{cols})")
    head = "          " + "".join(f"{c:>12}" for c in col_labels[:cols])
    print(head)
    for i in range(rows):
        line = f"  {row_labels[i]:>6} |" + "".join(f"{M[i, j]:+12.5f}" for j in range(cols))
        print(line)
    print()


def choose_safe_travel_z(robot, target_x, target_y, hover_robot_z, target_tag_z):
    """Pick safe travel Z from stereo-derived hover Z."""
    desired = max(MIN_COARSE_TRAVEL_Z_MM, float(hover_robot_z))
    min_allowed = float(target_tag_z) + min(15.0, HOVER_HEIGHT_MM)

    candidates = [
        desired,
        HOME_Z_MM,
        DEFAULT_TRAVEL_Z_MM,
        100.0,
        75.0,
        65.0,
        50.0,
        40.0,
        25.0,
        0.0,
    ]

    clean = []
    for z in candidates:
        z = float(z)
        if z < min_allowed:
            continue
        if all(abs(z - old) > 1e-6 for old in clean):
            clean.append(z)

    for z in clean:
        ok_pose, reason_pose = robot.check_cartesian_pose_safe(target_x, target_y, z)
        if not ok_pose:
            print(f"[TRAVEL_Z] Reject robot_z={z:.1f}: target unsafe: {reason_pose}")
            continue

        ok_path, path, reason_path = robot.plan_cartesian_path(target_x, target_y, z)
        if not ok_path:
            print(f"[TRAVEL_Z] Reject robot_z={z:.1f}: path unsafe: {reason_path}")
            continue

        print(f"[TRAVEL_Z] Selected robot_z={z:.1f} mm")
        return z

    return None


def tag_phi_from_det(det, lookup_robot_z, bundle):
    """Estimate robot-world phi from target tag top edge using overhead homography."""
    try:
        c0_xy, _, _, _, _ = map_uv_z_to_robot_xy(det.corners[0], lookup_robot_z, bundle)
        c1_xy, _, _, _, _ = map_uv_z_to_robot_xy(det.corners[1], lookup_robot_z, bundle)
        dx = float(c1_xy[0] - c0_xy[0])
        dy = float(c1_xy[1] - c0_xy[1])
        phi = float(np.degrees(np.arctan2(dy, dx))) + PHI_OFFSET_DEG
        return phi
    except Exception as exc:
        print(f"[PHI] tag_phi_from_det failed: {exc}")
        return None


# ============================================================
# PICK / PLACE
# ============================================================

def execute_pick(robot, tg_xy, pick_phi, target_tag_robot_z, hover_robot_z, grasp_robot_z):
    x_t, y_t = float(tg_xy[0]), float(tg_xy[1])

    travel_z = choose_safe_travel_z(robot, x_t, y_t, hover_robot_z, target_tag_robot_z)
    if travel_z is None:
        print("[PICK] REFUSED: no soft-limit-safe travel Z found.")
        return False, None

    print("\n[PICK] Starting pick sequence")
    print(f"[PICK] target XY=({x_t:.1f},{y_t:.1f}) target_tag_z={target_tag_robot_z:.1f}")
    print(f"[PICK] TAG_TO_EE={TAG_TO_EE_Z_MM:+.1f}, HOVER={HOVER_HEIGHT_MM:+.1f}, GRASP={GRASP_OFFSET_MM:+.1f}")
    print(f"[PICK] travel_z={travel_z:.1f}, hover_z={hover_robot_z:.1f}, grasp_z={grasp_robot_z:.1f}")

    print(f"[PICK] Opening claw (servo={CLAW_OPEN_DEG} deg)")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)

    _, _, z_cur, phi_cur = robot.fk()
    phi = phi_cur if pick_phi is None else float(pick_phi)

    if abs(z_cur - travel_z) > 1.0:
        print(f"[PICK] Raising/moving Z to travel_z={travel_z:.1f}")
        if not move_cartesian_nonnegative_z(robot, "[PICK] travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
            print("[PICK] Failed moving to travel Z.")
            return False, None
        robot.sync_estimate_from_teensy_steps()

    print(f"[PICK] Approach XY=({x_t:.1f},{y_t:.1f}) phi={phi:.1f} z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PICK] approach", x_mm=x_t, y_mm=y_t, z_mm=travel_z, phi_deg=phi, move_time_s=COARSE_MOVE_TIME_S):
        print("[PICK] Approach move failed.")
        return False, None
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PICK] Lowering to grasp_robot_z={grasp_robot_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PICK] grasp Z", z_mm=float(grasp_robot_z), move_time_s=PICK_MOVE_TIME_S):
        print("[PICK] Failed lowering to grasp Z.")
        return False, None
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PICK] Closing claw (servo={CLAW_CLOSED_DEG} deg)")
    robot.servo(CLAW_CLOSED_DEG)
    time.sleep(CLAW_SETTLE_S)

    print(f"[PICK] Raising back to travel_z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PICK] post-pick travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
        print("[PICK] Failed raising after pick — opening claw for safety.")
        robot.servo(CLAW_OPEN_DEG)
        return False, None
    robot.sync_estimate_from_teensy_steps()

    print(f"[PICK] Done. grasp_robot_z={grasp_robot_z:.1f}, phi={phi:.1f}")
    return True, float(grasp_robot_z)


def execute_place(robot, drop_zone_xy, drop_zone_phi, place_robot_z):
    x_d, y_d = float(drop_zone_xy[0]), float(drop_zone_xy[1])
    drop_phi = float(drop_zone_phi)

    # Use current held item's place Z plus hover height as target travel candidate.
    target_tag_z_for_place = float(place_robot_z - GRASP_OFFSET_MM)
    hover_robot_z = float(target_tag_z_for_place + HOVER_HEIGHT_MM)

    travel_z = choose_safe_travel_z(robot, x_d, y_d, hover_robot_z, target_tag_z_for_place)
    if travel_z is None:
        print("[PLACE] REFUSED: no soft-limit-safe travel Z to drop zone.")
        return False

    print("\n[PLACE] Starting place sequence")
    print(f"[PLACE] drop XY=({x_d:.1f},{y_d:.1f}) phi={drop_phi:.1f} travel_z={travel_z:.1f} place_z={place_robot_z:.1f}")

    _, _, z_cur, _ = robot.fk()
    if abs(z_cur - travel_z) > 1.0:
        print(f"[PLACE] Raising to travel_z={travel_z:.1f}")
        if not move_cartesian_nonnegative_z(robot, "[PLACE] travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
            print("[PLACE] Failed moving to travel Z.")
            return False
        robot.sync_estimate_from_teensy_steps()

    print(f"[PLACE] Moving to drop zone XY=({x_d:.1f},{y_d:.1f}) phi={drop_phi:.1f} z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PLACE] approach", x_mm=x_d, y_mm=y_d, z_mm=travel_z, phi_deg=drop_phi, move_time_s=COARSE_MOVE_TIME_S):
        print("[PLACE] Move to drop zone failed.")
        return False
    robot.sync_estimate_from_teensy_steps()

    print(f"[PLACE] Lowering to place_z={place_robot_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PLACE] place Z", z_mm=float(place_robot_z), move_time_s=PICK_MOVE_TIME_S):
        print("[PLACE] Failed lowering to place Z.")
        return False
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PLACE] Opening claw (servo={CLAW_OPEN_DEG} deg)")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)

    print(f"[PLACE] Raising back to travel_z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PLACE] post-place travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
        print("[PLACE] Failed raising after place.")
        return False
    robot.sync_estimate_from_teensy_steps()

    print("[PLACE] Done.")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nCalibration Bundle Live Test - Stereo Z PickPlace")
    print("-------------------------------------------------")
    print_startup_config("test_calibration_bundle_live.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("test_calibration_bundle_live.py")
    print("[Z KNOBS]")
    print(f"  TAG_TO_EE_Z_MM={TAG_TO_EE_Z_MM:+.1f}")
    print(f"  HOVER_HEIGHT_MM={HOVER_HEIGHT_MM:+.1f}")
    print(f"  GRASP_OFFSET_MM={GRASP_OFFSET_MM:+.1f}")
    print(f"  USE_EE_FK_Z_BIAS_CORRECTION={USE_EE_FK_Z_BIAS_CORRECTION}")
    print(f"  CLAMP_STEREO_LOOKUP_Z_TO_BUNDLE={CLAMP_STEREO_LOOKUP_Z_TO_BUNDLE}")
    print(f"  PRINT_GEOMETRY_SANITY={PRINT_GEOMETRY_SANITY}")

    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)

    detector = build_detector()
    overhead_cap = open_overhead_camera()
    stereo = open_stereo_camera()
    robot = Robot(ROBOT_CONFIG, connect=True)

    last_target_xy = None
    last_target_support_dist = np.inf
    last_tag_phi = None
    last_geom = None

    has_item = False
    drop_zone_xy = None
    drop_zone_phi = None
    last_grasp_robot_z = None

    last_sanity_print = 0.0

    try:
        require_soft_limits(robot)

        robot.enable(True)
        robot.init_drivers()

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()
        if choice == "h":
            if not robot.home():
                return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps():
                return
        elif choice == "a":
            robot.assume_homed()
            robot.print_estimate()
        else:
            print("Unknown choice. Aborting.")
            return

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        # Combined window aspect (overhead + stereo stacked) is COMBINED_WIDTH_PX x (overhead_h + stereo_h + sep).
        cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + 2)

        while True:
            ok, frame = overhead_cap.read()
            if not ok or frame is None:
                continue

            dets_overhead = detect_tags(detector, frame)
            det_ee_overhead = dets_overhead.get(EE_TAG_ID)
            det_tg_overhead = dets_overhead.get(TARGET_TAG_ID)

            stereo_tags, left_img, right_img, det_l_all, det_r_all = read_stereo_tags_once(
                stereo, detector, stereo_calib
            )

            x_fk, y_fk, z_fk, phi_fk = robot.fk()
            geom = compute_stereo_target_geometry(robot, bundle, stereo_tags)
            last_geom = geom if geom is not None else last_geom

            lines = [
                "m hover | k PICK | f PLACE | n drop | b sanity | r matrix | o/l claw | [/] Z | ,/. phi | q quit",
                f"FK=({x_fk:.1f},{y_fk:.1f},z={z_fk:.1f},phi={phi_fk:.1f})",
                f"stereo EE={EE_TAG_ID in stereo_tags} target={TARGET_TAG_ID in stereo_tags} | overhead EE={det_ee_overhead is not None} target={det_tg_overhead is not None}",
            ]

            if geom is not None:
                used_lookup_z = geom["lookup_z_used"]
                tg_xy = None
                if det_tg_overhead is not None:
                    tg_xy, tg_uv, lo, hi, alpha = map_uv_z_to_robot_xy(det_tg_overhead.center, used_lookup_z, bundle)
                    support_dist, _ = nearest_support_distance(tg_xy, used_lookup_z, bundle)
                    last_target_xy = tg_xy
                    last_target_support_dist = support_dist
                    last_tag_phi = tag_phi_from_det(det_tg_overhead, used_lookup_z, bundle)

                    support_status = "OK" if support_dist <= bundle["max_nearest"] else "FAR"
                    lines.append(
                        f"ID{TARGET_TAG_ID}: xy=({tg_xy[0]:.1f},{tg_xy[1]:.1f}) "
                        f"lookup_z={used_lookup_z:.1f}{' CLAMP' if geom['lookup_z_clamped'] else ''} "
                        f"hover_z={geom['hover_robot_z']:.1f} grasp_z={geom['grasp_robot_z']:.1f} support={support_dist:.1f} {support_status}"
                    )
                else:
                    lines.append(
                        f"Stereo target z={geom['target_tag_robot_xyz_corrected'][2]:.1f}, "
                        f"lookup_z={used_lookup_z:.1f}, but overhead target not visible for XY."
                    )

                if geom["ee_tool_z_from_stereo_raw"] is not None:
                    lines.append(
                        f"Z sanity: EE_stereo_tool_z={geom['ee_tool_z_from_stereo_raw']:.1f}, "
                        f"FK_z={z_fk:.1f}, bias={geom['stereo_z_bias_mm']:+.1f}"
                    )
            else:
                lines.append("No stereo target geometry yet. Need target ID2 visible in both stereo cameras.")

            if det_ee_overhead is not None:
                ee_xy, _, lo, hi, alpha = map_uv_z_to_robot_xy(det_ee_overhead.center, z_fk, bundle)
                ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                ee_err_norm = float(np.linalg.norm(ee_err))
                status = "OK"
                if ee_err_norm > MAX_EE_ERROR_MM:
                    status = "BAD"
                elif ee_err_norm > WARN_EE_ERROR_MM:
                    status = "WARN"
                lines.append(f"Overhead EE {status}: err=({ee_err[0]:+.1f},{ee_err[1]:+.1f}) |e|={ee_err_norm:.1f}mm")

            lines.append(
                f"PICK: {'HOLDING' if has_item else 'empty'} | drop="
                f"{('(' + f'{drop_zone_xy[0]:.0f},{drop_zone_xy[1]:.0f},phi={drop_zone_phi:.0f}' + ')') if drop_zone_xy is not None else 'NOT SET'} "
                f"| grasp_z={f'{last_grasp_robot_z:.1f}' if last_grasp_robot_z is not None else 'N/A'}"
            )

            if PRINT_GEOMETRY_SANITY and geom is not None and (time.time() - last_sanity_print) > GEOMETRY_SANITY_PRINT_EVERY_SEC:
                print_geometry_sanity(geom)
                last_sanity_print = time.time()

            overhead_annotated = draw_status(frame, det_ee_overhead, det_tg_overhead, lines)
            combined = compose_camera_views(overhead_annotated, left_img, right_img, det_l_all, det_r_all)
            cv2.imshow(WINDOW, combined)

            key = read_command_key(1)

            if key in ("q", "escape"):
                break

            elif key == "b":
                print_geometry_sanity(last_geom)

            elif key == "m":
                if last_target_xy is None or det_tg_overhead is None:
                    print("[MOVE] Need overhead target ID2 for XY.")
                    continue
                if geom is None:
                    print("[MOVE] Need stereo target ID2 for Z.")
                    continue
                if last_target_support_dist > bundle["max_nearest"]:
                    print(f"[MOVE] REFUSED: target too far from calibration support ({last_target_support_dist:.1f} > {bundle['max_nearest']:.1f})")
                    continue

                x_t, y_t = float(last_target_xy[0]), float(last_target_xy[1])
                travel_z = choose_safe_travel_z(
                    robot,
                    x_t,
                    y_t,
                    geom["hover_robot_z"],
                    geom["target_tag_robot_xyz_corrected"][2],
                )
                if travel_z is None:
                    print("[MOVE] REFUSED: no soft-limit-safe travel Z found.")
                    continue

                print(f"[MOVE] Coarse hover to ID{TARGET_TAG_ID}: XY=({x_t:.1f},{y_t:.1f}) travel_z={travel_z:.1f}")
                _, _, z_cur, phi_cur = robot.fk()
                if abs(z_cur - travel_z) > 1.0:
                    if not move_cartesian_nonnegative_z(robot, "[MOVE] travel Z", z_mm=travel_z, move_time_s=0.75):
                        print("[MOVE] Failed moving to travel Z.")
                        continue
                    robot.sync_estimate_from_teensy_steps()

                if not move_cartesian_nonnegative_z(robot, "[MOVE] hover", x_mm=x_t, y_mm=y_t, z_mm=travel_z, phi_deg=phi_cur, move_time_s=COARSE_MOVE_TIME_S):
                    print("[MOVE] Coarse XY move failed.")
                    continue
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == "k":
                if last_target_xy is None or det_tg_overhead is None:
                    print("[PICK] Need overhead target ID2 for XY.")
                elif geom is None:
                    print("[PICK] Need stereo target ID2 for Z.")
                elif REFUSE_PICK_IF_TARGET_FAR and last_target_support_dist > bundle["max_nearest"]:
                    print(f"[PICK] REFUSED: target too far from calibration support ({last_target_support_dist:.1f} > {bundle['max_nearest']:.1f})")
                else:
                    print_geometry_sanity(geom, prefix="[PICK GEOM]")
                    ok, last_grasp_robot_z = execute_pick(
                        robot,
                        last_target_xy,
                        last_tag_phi,
                        geom["target_tag_robot_xyz_corrected"][2],
                        geom["hover_robot_z"],
                        geom["grasp_robot_z"],
                    )
                    has_item = ok
                    robot.print_estimate()

            elif key == "f":
                if not has_item:
                    print("[PLACE] No item held — run pick (k) first.")
                elif REFUSE_PLACE_IF_NO_DROP_ZONE and drop_zone_xy is None:
                    print("[PLACE] Drop zone not set — jog to drop location and press n.")
                elif drop_zone_phi is None:
                    print("[PLACE] Drop zone phi not set — press n again.")
                elif last_grasp_robot_z is None:
                    print("[PLACE] No grasp height recorded.")
                else:
                    ok = execute_place(robot, drop_zone_xy, drop_zone_phi, last_grasp_robot_z)
                    if ok:
                        has_item = False
                    robot.print_estimate()

            elif key == "n":
                x_cur, y_cur, _, phi_cur = robot.fk()
                drop_zone_xy = np.array([x_cur, y_cur], dtype=np.float64)
                drop_zone_phi = float(phi_cur)
                print(f"[DROP ZONE] Set to current FK: ({x_cur:.1f}, {y_cur:.1f}, phi={phi_cur:.1f} deg)")

            elif key == "[":
                print(f"[JOG] Z down by {Z_JOG_MM:.1f} mm")
                jog_nonnegative_z(robot, "[JOG] Z down", dz=-Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == "]":
                print(f"[JOG] Z up by {Z_JOG_MM:.1f} mm")
                jog_nonnegative_z(robot, "[JOG] Z up", dz=+Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ",":
                print(f"[JOG] Phi/J4 negative by {PHI_JOG_DEG:.1f} deg")
                jog_nonnegative_z(robot, "[JOG] phi negative", dphi=-PHI_JOG_DEG, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ".":
                print(f"[JOG] Phi/J4 positive by {PHI_JOG_DEG:.1f} deg")
                jog_nonnegative_z(robot, "[JOG] phi positive", dphi=+PHI_JOG_DEG, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == "v":
                if det_ee_overhead is None:
                    print("[VALIDATE] EE tag not visible overhead.")
                else:
                    ee_xy, _, lo, hi, alpha = map_uv_z_to_robot_xy(det_ee_overhead.center, z_fk, bundle)
                    ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                    support_dist, support_idx = nearest_support_distance(ee_xy, z_fk, bundle)
                    print("\n[VALIDATE OVERHEAD EE]")
                    print(f"  FK xy       = ({x_fk:.2f}, {y_fk:.2f}) at z={z_fk:.2f}")
                    print(f"  mapped xy   = ({ee_xy[0]:.2f}, {ee_xy[1]:.2f})")
                    print(f"  error       = ({ee_err[0]:+.2f}, {ee_err[1]:+.2f}) |e|={np.linalg.norm(ee_err):.2f} mm")
                    print(f"  layers      = {lo}/{hi}, alpha={alpha:.3f}")
                    print(f"  support     = {support_dist:.2f} mm nearest idx={support_idx}\n")

            elif key == "r":
                print("\n[STEREO XYZ MODEL]")
                A = bundle.get("A_robot_from_cam_xyz_3x4")
                B = bundle.get("B_cam_from_robot_xyz_3x4")
                A_lin = bundle.get("A_robot_from_cam_xyz_linear_3x3")
                if A is None:
                    print("  No full XYZ stereo model in bundle. Re-run calibration.")
                else:
                    print("  Convention:")
                    print("    robot_xyz = A_robot_from_cam_xyz_3x4 @ [cam_x, cam_y, cam_z, 1]")
                    print("    target_tag_robot_z_corrected = raw_target_tag_z + (FK_z - (raw_ee_tag_z + TAG_TO_EE_Z_MM))")
                    print("    lookup_z = target_tag_robot_z_corrected")
                    print("    hover_z = target_tag_robot_z_corrected + HOVER_HEIGHT_MM")
                    print("    grasp_z = target_tag_robot_z_corrected + GRASP_OFFSET_MM\n")
                    print(f"  RMSE = {bundle['stereo_robot_xyz_fit_rmse_mm']:.2f} mm")
                    print("  Reads: rows = output (robot_xyz components), cols = input (cam_xyz + bias)\n")
                    print_matrix_labeled(
                        "A_robot_from_cam_xyz_3x4  (stereo cam -> robot)",
                        A,
                        row_labels=["rob_x", "rob_y", "rob_z"],
                        col_labels=["cam_x", "cam_y", "cam_z", "bias"],
                        units="mm per mm (last col mm)",
                    )
                    if B is not None:
                        print_matrix_labeled(
                            "B_cam_from_robot_xyz_3x4  (robot -> stereo cam)",
                            B,
                            row_labels=["cam_x", "cam_y", "cam_z"],
                            col_labels=["rob_x", "rob_y", "rob_z", "bias"],
                            units="mm per mm (last col mm)",
                        )
                    if A_lin is not None:
                        print_matrix_labeled(
                            "A linear part (3x3)  -- how cam delta -> robot delta",
                            A_lin,
                            row_labels=["rob_x", "rob_y", "rob_z"],
                            col_labels=["cam_x", "cam_y", "cam_z"],
                            units="mm per mm",
                        )
                print()

            elif key == "p":
                robot.print_estimate()

            elif key == "h":
                robot.home()
                robot.print_estimate()

            elif key == "c":
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == "a":
                robot.assume_homed()
                robot.print_estimate()

            elif key == "e":
                robot.enable(True)
                robot.init_drivers()

            elif key == "d":
                robot.enable(False)

            elif key == "o":
                print(f"[CLAW] Opening (servo={CLAW_OPEN_DEG} deg)")
                robot.servo(CLAW_OPEN_DEG)

            elif key == "l":
                print(f"[CLAW] Closing (servo={CLAW_CLOSED_DEG} deg)")
                robot.servo(CLAW_CLOSED_DEG)

    finally:
        overhead_cap.release()
        stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
