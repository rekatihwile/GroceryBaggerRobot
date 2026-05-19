from __future__ import annotations

"""
homography_coarse_stereo_fine.py

Hybrid test for the grocery bagger:

    OVERHEAD WEBCAM:
        Detect target AprilTag ID2 on the grocery/box.
        Use saved overhead_homography_calibration.npz to map:
            target pixel (u,v) -> robot XY (x_mm, y_mm)
        Do one coarse absolute move to that XY.

    STEREO CAMERA:
        Detect EE AprilTag ID0 and target AprilTag ID2.
        Use saved/calibrated stereo Jacobian to do final PD alignment.

This is meant to test the exact pipeline:
    overhead homography coarse move -> stereo/PD fine alignment

Required files in same folder:
    robot.py
    stereo_apriltag_viewer.py
    stereo_calibration.npz
    overhead_homography_calibration.npz
    optionally stereo_pd_jacobian.npz

Recommended setup:
    EE tag      = ID 0
    target tag  = ID 2
    overhead webcam index = 2
    stereo camera index   = 1

Controls:
    Startup:
        h = HOME
        c = sync from Teensy POS, no homing
        a = assume configured home pose

    Main window:
        v = validate homography using visible EE tag and/or target tag
        t = save overhead target XY from target tag ID2
        m = coarse move only to saved overhead target XY
        k = calibrate stereo Jacobian from EE tag
        s = stereo fine PD only, using live target tag ID2
        g = GO: overhead target -> coarse move -> stereo fine PD
        o/l = open/close claw
        p = print robot estimate
        h/c/e/d = home/sync/enable/disable
        q/ESC = quit
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot import Robot
from robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
    require_robot_soft_limits_loaded,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_HOMOGRAPHY_PATH as HOMOGRAPHY_PATH,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    STEREO_CALIBRATION_PATH,
    STEREO_FOURCC,
    STEREO_FPS,
    STEREO_HEIGHT,
    STEREO_INDEX,
    STEREO_PD_JACOBIAN_PATH as JACOBIAN_PATH,
    STEREO_WIDTH,
    TARGET_TAG_ID,
)
from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    average_angles_deg,
    make_preview,
)


# ============================================================
# USER SETTINGS
# ============================================================

# Coarse move behavior.
COARSE_Z_MODE = "current"       # "current" keeps current Z; "fixed" uses COARSE_FIXED_Z_MM
COARSE_FIXED_Z_MM = LOW_Z_MM
COARSE_PHI_MODE = "current"     # "current" keeps current phi; "fixed" uses COARSE_FIXED_PHI_DEG
COARSE_FIXED_PHI_DEG = 0.0
COARSE_MOVE_TIME_S = 1.10
COARSE_SETTLE_S = 0.25

# Safety/workspace clamp. Edit to your reachable/visible region.
ENFORCE_WORKSPACE_LIMITS = True
X_MIN_MM, X_MAX_MM = 40.0, 520.0
Y_MIN_MM, Y_MAX_MM = 180.0, 620.0

# Measurement averaging.
OVERHEAD_TARGET_SAMPLES = 12
OVERHEAD_VALIDATE_SAMPLES = 8
OVERHEAD_TIMEOUT_S = 3.0

# Stereo Jacobian calibration.
JAC_CAL_STEP_MM = 40.0
JAC_CAL_MOVE_TIME_S = 0.65
JAC_CAL_SETTLE_S = 0.35
JAC_CAL_SAMPLES = 6
JAC_CAL_SAMPLE_TIMEOUT_S = 3.0

# Stereo fine PD tuning.
KP = 0.20
KD = 0.03
MAX_AXIS_STEP_MM = 10.0
DONE_THRESH_MM = 6.0
MAX_ITERS = 80
PD_MOVE_TIME_S = 0.16
PD_SETTLE_SEC = 0.00
MAX_WAIT_SEC = 0.35
FILTER_ALPHA = 0.45
STOP_IF_ERROR_GROWS = True
GROWTH_MARGIN_MM = 20.0
GROWTH_COUNT_LIMIT = 6

# Claw.
CLAW_OPEN_DEG = 70
CLAW_CLOSED_DEG = 20
CLAW_SETTLE_S = 0.45

WINDOW_OVERHEAD = "Overhead Homography Coarse"
WINDOW_STEREO = "Stereo Fine PD"


# ============================================================
# DATA TYPES / STATE
# ============================================================

@dataclass
class TriangulatedTag:
    tag_id: int
    xyz_cam_mm: np.ndarray
    left_px: np.ndarray
    right_px: np.ndarray
    disparity_px: float
    theta_deg: float


@dataclass
class OverheadMeasurement:
    tag_id: int
    uv_px: np.ndarray
    theta_deg: float
    xy_robot_mm: np.ndarray


JACOBIAN_MATRIX: np.ndarray | None = None


# ============================================================
# BASIC HELPERS
# ============================================================

def open_overhead_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def load_homography(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing homography file: {path.resolve()}")
    data = np.load(path, allow_pickle=False)
    H = np.asarray(data["H_img_to_robot"], dtype=np.float64)
    print(f"[H] Loaded {path.resolve()}")
    if "residual_norms_mm" in data.files:
        residuals = np.asarray(data["residual_norms_mm"], dtype=np.float64)
        print(f"[H] calibration residual RMS={np.sqrt(np.mean(residuals**2)):.2f} mm, max={np.max(residuals):.2f} mm")
    if "cal_z_mm" in data.files:
        print(f"[H] calibrated at z={float(np.asarray(data['cal_z_mm']).reshape(-1)[0]):.1f} mm")
    return H


def load_stereo_calibration(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stereo calibration: {path.resolve()}")
    data = np.load(path, allow_pickle=False)
    calib = {k: np.asarray(data[k]) for k in data.files}
    print(f"[Stereo] Loaded calibration: {path.resolve()}")
    if "baseline_mm" in calib:
        print(f"[Stereo] baseline_mm={float(np.asarray(calib['baseline_mm']).reshape(-1)[0]):.3f}")
    return calib


def load_jacobian_if_present(path: Path = JACOBIAN_PATH) -> np.ndarray | None:
    global JACOBIAN_MATRIX
    if not path.exists():
        print(f"[CAL] No saved Jacobian at {path.resolve()}. Press k to calibrate.")
        return None
    data = np.load(path, allow_pickle=False)
    J = np.asarray(data["J"], dtype=np.float64)
    if J.shape != (3, 2):
        print(f"[CAL] Bad J shape in {path}: {J.shape}. Ignoring.")
        return None
    JACOBIAN_MATRIX = J
    print(f"[CAL] Loaded Jacobian from {path.resolve()}:")
    print_jacobian(J)
    return J


def save_jacobian(J: np.ndarray, path: Path = JACOBIAN_PATH) -> None:
    np.savez(path, J=np.asarray(J, dtype=np.float64))
    print(f"[CAL] Saved Jacobian → {path.resolve()}")


def print_jacobian(J: np.ndarray) -> None:
    j_x, j_y = J[:, 0], J[:, 1]
    print(f"       +robot_X → Δcam = ({j_x[0]:+.3f}, {j_x[1]:+.3f}, {j_x[2]:+.3f}) mm/mm")
    print(f"       +robot_Y → Δcam = ({j_y[0]:+.3f}, {j_y[1]:+.3f}, {j_y[2]:+.3f}) mm/mm")


def image_to_robot_xy(H_img_to_robot: np.ndarray, u: float, v: float) -> np.ndarray:
    pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
    xy = cv2.perspectiveTransform(pt, H_img_to_robot.astype(np.float64))[0, 0]
    return xy.astype(np.float64)


def in_workspace(x: float, y: float) -> bool:
    return X_MIN_MM <= x <= X_MAX_MM and Y_MIN_MM <= y <= Y_MAX_MM


def clamp_workspace(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, X_MIN_MM, X_MAX_MM)),
        float(np.clip(y, Y_MIN_MM, Y_MAX_MM)),
    )


def draw_tag_custom(image: np.ndarray, det, label: str, color) -> np.ndarray:
    out = image.copy()
    if det is None:
        return out
    corners = np.round(det.corners).astype(int)
    center = tuple(np.round(det.center).astype(int))
    cv2.polylines(out, [corners], True, color, 2, cv2.LINE_AA)
    cv2.circle(out, center, 5, color, -1, cv2.LINE_AA)
    cv2.putText(out, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def render_overhead(frame, dets, H_img_to_robot, robot: Robot, saved_target_xy: np.ndarray | None, extra_lines=None):
    out = frame.copy()
    det_ee = dets.get(EE_TAG_ID)
    det_tg = dets.get(TARGET_TAG_ID)

    out = draw_tag_custom(out, det_ee, f"EE {EE_TAG_ID}", (0, 255, 255))
    out = draw_tag_custom(out, det_tg, f"TARGET {TARGET_TAG_ID}", (0, 255, 0))

    x_fk, y_fk, z_fk, phi_fk = robot.fk()
    lines = [
        "v validate | t save overhead target | m coarse | k calib stereo J | s stereo fine | g GO | q quit",
        f"EE visible={det_ee is not None} target visible={det_tg is not None} saved_target={saved_target_xy is not None}",
        f"FK XY=({x_fk:+.1f},{y_fk:+.1f}) Z={z_fk:+.1f} phi={phi_fk:+.1f}",
        f"J={'LOADED' if JACOBIAN_MATRIX is not None else 'NOT CALIBRATED'}",
    ]

    if det_ee is not None:
        ee_xy = image_to_robot_xy(H_img_to_robot, det_ee.center[0], det_ee.center[1])
        lines.append(f"overhead EE→XY=({ee_xy[0]:+.1f},{ee_xy[1]:+.1f}) err_vs_FK=({ee_xy[0]-x_fk:+.1f},{ee_xy[1]-y_fk:+.1f})")
    if det_tg is not None:
        tg_xy = image_to_robot_xy(H_img_to_robot, det_tg.center[0], det_tg.center[1])
        lines.append(f"overhead TARGET→XY=({tg_xy[0]:+.1f},{tg_xy[1]:+.1f})")
    if saved_target_xy is not None:
        lines.append(f"saved target XY=({saved_target_xy[0]:+.1f},{saved_target_xy[1]:+.1f})")
    if extra_lines:
        lines.extend(extra_lines)

    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# ============================================================
# OVERHEAD MEASUREMENT
# ============================================================

def read_overhead_once(cap: cv2.VideoCapture, detector):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, {}
    dets = detect_tags(detector, frame)
    return frame, dets


def measure_overhead_tag_average(
    cap: cv2.VideoCapture,
    detector,
    H_img_to_robot: np.ndarray,
    robot: Robot,
    tag_id: int,
    samples_needed: int,
    timeout_s: float,
    label: str,
    saved_target_xy: np.ndarray | None = None,
) -> OverheadMeasurement | None:
    samples = []
    deadline = time.time() + timeout_s

    while time.time() < deadline and len(samples) < samples_needed:
        frame, dets = read_overhead_once(cap, detector)
        if frame is None:
            continue

        det = dets.get(tag_id)
        if det is not None:
            samples.append([float(det.center[0]), float(det.center[1]), float(det.theta_deg)])

        lines = [
            f"{label}: collecting tag {tag_id} samples {len(samples)}/{samples_needed}",
            "SPACE accept early if >=4 | q abort",
        ]
        cv2.imshow(WINDOW_OVERHEAD, render_overhead(frame, dets, H_img_to_robot, robot, saved_target_xy, lines))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None
        if key == 32 and len(samples) >= 4:
            break

    if len(samples) < 4:
        print(f"[OVERHEAD] Not enough samples for tag {tag_id}. Got {len(samples)}.")
        return None

    arr = np.asarray(samples, dtype=np.float64)
    uvtheta = arr.mean(axis=0)
    xy = image_to_robot_xy(H_img_to_robot, uvtheta[0], uvtheta[1])

    return OverheadMeasurement(
        tag_id=tag_id,
        uv_px=uvtheta[:2],
        theta_deg=float(uvtheta[2]),
        xy_robot_mm=xy,
    )


def save_overhead_target(cap, detector, H_img_to_robot, robot) -> np.ndarray | None:
    meas = measure_overhead_tag_average(
        cap, detector, H_img_to_robot, robot,
        tag_id=TARGET_TAG_ID,
        samples_needed=OVERHEAD_TARGET_SAMPLES,
        timeout_s=OVERHEAD_TIMEOUT_S,
        label="Save overhead target",
    )
    if meas is None:
        print("[TARGET] Failed to save overhead target.")
        return None

    x, y = float(meas.xy_robot_mm[0]), float(meas.xy_robot_mm[1])
    print(f"\n[TARGET] overhead tag {TARGET_TAG_ID}: uv=({meas.uv_px[0]:.1f},{meas.uv_px[1]:.1f}) -> robot XY=({x:.1f},{y:.1f})")

    if ENFORCE_WORKSPACE_LIMITS and not in_workspace(x, y):
        xc, yc = clamp_workspace(x, y)
        print(f"[TARGET] WARNING: target outside workspace clamp.")
        print(f"         raw=({x:.1f},{y:.1f}) clamped=({xc:.1f},{yc:.1f})")
        x, y = xc, yc

    return np.array([x, y], dtype=np.float64)


def validate_homography_once(cap, detector, H_img_to_robot, robot, saved_target_xy):
    print("\n[VALIDATE] Reading overhead EE/target and mapping through H.")
    frame, dets = read_overhead_once(cap, detector)
    if frame is None:
        print("[VALIDATE] No overhead frame.")
        return

    x_fk, y_fk, z_fk, phi_fk = robot.fk()

    ee = measure_overhead_tag_average(
        cap, detector, H_img_to_robot, robot,
        tag_id=EE_TAG_ID,
        samples_needed=OVERHEAD_VALIDATE_SAMPLES,
        timeout_s=OVERHEAD_TIMEOUT_S,
        label="Validate EE",
        saved_target_xy=saved_target_xy,
    )
    if ee is not None:
        err = ee.xy_robot_mm - np.array([x_fk, y_fk], dtype=np.float64)
        print(f"[VALIDATE] EE overhead→XY=({ee.xy_robot_mm[0]:.1f},{ee.xy_robot_mm[1]:.1f})")
        print(f"           FK XY=({x_fk:.1f},{y_fk:.1f})  error=({err[0]:+.1f},{err[1]:+.1f}) |e|={np.linalg.norm(err):.1f} mm")

    tg = measure_overhead_tag_average(
        cap, detector, H_img_to_robot, robot,
        tag_id=TARGET_TAG_ID,
        samples_needed=OVERHEAD_VALIDATE_SAMPLES,
        timeout_s=OVERHEAD_TIMEOUT_S,
        label="Validate target",
        saved_target_xy=saved_target_xy,
    )
    if tg is not None:
        print(f"[VALIDATE] TARGET overhead→XY=({tg.xy_robot_mm[0]:.1f},{tg.xy_robot_mm[1]:.1f})")


# ============================================================
# STEREO TRIANGULATION / DISPLAY
# ============================================================

def build_rectification_maps(calib: dict[str, np.ndarray], image_size: tuple[int, int]):
    w, h = image_size
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        np.asarray(calib["left_camera_matrix"], dtype=np.float64),
        np.asarray(calib["left_distortion_coefficients"], dtype=np.float64),
        np.asarray(calib["rectification_left"], dtype=np.float64),
        np.asarray(calib["projection_left_rectified"], dtype=np.float64),
        (w, h),
        cv2.CV_32FC1,
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        np.asarray(calib["right_camera_matrix"], dtype=np.float64),
        np.asarray(calib["right_distortion_coefficients"], dtype=np.float64),
        np.asarray(calib["rectification_right"], dtype=np.float64),
        np.asarray(calib["projection_right_rectified"], dtype=np.float64),
        (w, h),
        cv2.CV_32FC1,
    )
    return map_lx, map_ly, map_rx, map_ry


def undistort_pixel_to_pixel_coords(point_xy: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pt = np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def triangulate_center_mm(left_xy: np.ndarray, right_xy: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
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
    xyz[2] *= -1.0
    return xyz


def make_stereo_tag(tag_id: int, det_l, det_r, calib: dict[str, np.ndarray]) -> TriangulatedTag | None:
    if det_l is None or det_r is None:
        return None
    left_xy = np.asarray(det_l.center, dtype=np.float64)
    right_xy = np.asarray(det_r.center, dtype=np.float64)
    xyz = triangulate_center_mm(left_xy, right_xy, calib)
    theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
    return TriangulatedTag(
        tag_id=tag_id,
        xyz_cam_mm=xyz,
        left_px=left_xy,
        right_px=right_xy,
        disparity_px=float(left_xy[0] - right_xy[0]),
        theta_deg=float(theta),
    )


def read_stereo_tags_once(stereo: SimpleStereoCamera, detector, calib):
    ok, _, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        return None, None, {}, {}, {}

    det_l = detect_tags(detector, left)
    det_r = detect_tags(detector, right)

    tags: dict[int, TriangulatedTag] = {}
    for tag_id in (EE_TAG_ID, TARGET_TAG_ID):
        tag = make_stereo_tag(tag_id, det_l.get(tag_id), det_r.get(tag_id), calib)
        if tag is not None:
            tags[tag_id] = tag

    return left, right, det_l, det_r, tags


def render_stereo_preview(left, right, det_l, det_r, tags, filtered_error=None):
    left_draw = draw_tag_custom(left, det_l.get(EE_TAG_ID), f"EE {EE_TAG_ID}", (0, 255, 255))
    left_draw = draw_tag_custom(left_draw, det_l.get(TARGET_TAG_ID), f"TARGET {TARGET_TAG_ID}", (0, 255, 0))
    right_draw = draw_tag_custom(right, det_r.get(EE_TAG_ID), f"EE {EE_TAG_ID}", (0, 255, 255))
    right_draw = draw_tag_custom(right_draw, det_r.get(TARGET_TAG_ID), f"TARGET {TARGET_TAG_ID}", (0, 255, 0))

    lines = [
        f"stereo fine PD | EE={EE_TAG_ID in tags} target={TARGET_TAG_ID in tags} J={'LOADED' if JACOBIAN_MATRIX is not None else 'NO'}",
        f"KP={KP:.2f} KD={KD:.2f} max_step={MAX_AXIS_STEP_MM:.1f} done={DONE_THRESH_MM:.1f}mm",
    ]

    if EE_TAG_ID in tags:
        e = tags[EE_TAG_ID].xyz_cam_mm
        lines.append(f"EE XYZ=({e[0]:+.1f},{e[1]:+.1f},{e[2]:+.1f})mm")
    if TARGET_TAG_ID in tags:
        t = tags[TARGET_TAG_ID].xyz_cam_mm
        lines.append(f"TG XYZ=({t[0]:+.1f},{t[1]:+.1f},{t[2]:+.1f})mm")
    if EE_TAG_ID in tags and TARGET_TAG_ID in tags and JACOBIAN_MATRIX is not None:
        delta_cam = tags[TARGET_TAG_ID].xyz_cam_mm - tags[EE_TAG_ID].xyz_cam_mm
        err = camera_delta_to_robot_delta(delta_cam)
        lines.append(f"live errXY=({err[0]:+.1f},{err[1]:+.1f}) |err|={np.linalg.norm(err):.1f}mm")
    if filtered_error is not None:
        lines.append(f"filtered errXY=({filtered_error[0]:+.1f},{filtered_error[1]:+.1f}) |err|={np.linalg.norm(filtered_error):.1f}mm")

    return make_preview(left_draw, right_draw, lines)


def measure_stereo_one_valid(stereo, detector, calib, required_ids: tuple[int, ...], max_wait_s=MAX_WAIT_SEC, filtered_error=None):
    start = time.time()
    while time.time() - start < max_wait_s:
        left, right, det_l, det_r, tags = read_stereo_tags_once(stereo, detector, calib)
        if left is None or right is None:
            continue

        cv2.imshow(WINDOW_STEREO, render_stereo_preview(left, right, det_l, det_r, tags, filtered_error))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None, True

        if all(tag_id in tags for tag_id in required_ids):
            return tags, False

    return None, False


def measure_ee_cam_average(stereo, detector, calib, n_samples: int, timeout_s: float) -> np.ndarray | None:
    samples: list[np.ndarray] = []
    deadline = time.time() + timeout_s
    while len(samples) < n_samples and time.time() < deadline:
        tags, abort = measure_stereo_one_valid(stereo, detector, calib, required_ids=(EE_TAG_ID,), max_wait_s=1.0)
        if abort:
            return None
        if tags is not None and EE_TAG_ID in tags:
            samples.append(tags[EE_TAG_ID].xyz_cam_mm.copy())
    if len(samples) < 2:
        return None
    return np.mean(np.stack(samples, axis=0), axis=0)


def camera_delta_to_robot_delta(delta_cam_xyz: np.ndarray) -> np.ndarray:
    if JACOBIAN_MATRIX is None:
        raise RuntimeError("No Jacobian loaded. Press k to calibrate.")
    delta_cam = np.asarray(delta_cam_xyz, dtype=np.float64).reshape(3)
    return np.linalg.pinv(JACOBIAN_MATRIX) @ delta_cam


def clip_axis(v: np.ndarray, max_abs: float) -> np.ndarray:
    return np.clip(v, -float(max_abs), float(max_abs)).astype(np.float64)


# ============================================================
# STEREO JACOBIAN / PD
# ============================================================

def run_jacobian_calibration(robot: Robot, stereo, detector, calib) -> bool:
    global JACOBIAN_MATRIX
    print("\n[CAL] Probing robot→stereo-camera Jacobian.")
    print(f"      step={JAC_CAL_STEP_MM:.1f}mm  samples_per_pose={JAC_CAL_SAMPLES}")
    print("      EE tag must stay visible at start, +X pose, and +Y pose.")

    step = float(JAC_CAL_STEP_MM)

    print("[CAL] Sampling start pose...")
    p0 = measure_ee_cam_average(stereo, detector, calib, JAC_CAL_SAMPLES, JAC_CAL_SAMPLE_TIMEOUT_S)
    if p0 is None:
        print("[CAL] Failed to see EE at start. Aborting.")
        return False
    print(f"      p0=({p0[0]:+.2f},{p0[1]:+.2f},{p0[2]:+.2f}) mm")

    print(f"[CAL] Jog +X by {step:.1f}mm...")
    if not robot.jog(dx=+step, dy=0.0, move_time_s=JAC_CAL_MOVE_TIME_S):
        return False
    time.sleep(JAC_CAL_SETTLE_S)
    px = measure_ee_cam_average(stereo, detector, calib, JAC_CAL_SAMPLES, JAC_CAL_SAMPLE_TIMEOUT_S)

    print("[CAL] Return -X...")
    if not robot.jog(dx=-step, dy=0.0, move_time_s=JAC_CAL_MOVE_TIME_S):
        return False
    time.sleep(JAC_CAL_SETTLE_S)
    if px is None:
        print("[CAL] Failed to see EE after +X jog.")
        return False
    print(f"      pX=({px[0]:+.2f},{px[1]:+.2f},{px[2]:+.2f}) mm")

    print(f"[CAL] Jog +Y by {step:.1f}mm...")
    if not robot.jog(dx=0.0, dy=+step, move_time_s=JAC_CAL_MOVE_TIME_S):
        return False
    time.sleep(JAC_CAL_SETTLE_S)
    py = measure_ee_cam_average(stereo, detector, calib, JAC_CAL_SAMPLES, JAC_CAL_SAMPLE_TIMEOUT_S)

    print("[CAL] Return -Y...")
    if not robot.jog(dx=0.0, dy=-step, move_time_s=JAC_CAL_MOVE_TIME_S):
        return False
    time.sleep(JAC_CAL_SETTLE_S)
    if py is None:
        print("[CAL] Failed to see EE after +Y jog.")
        return False
    print(f"      pY=({py[0]:+.2f},{py[1]:+.2f},{py[2]:+.2f}) mm")

    j_x = (px - p0) / step
    j_y = (py - p0) / step
    J = np.column_stack([j_x, j_y])

    cos_ang = float(np.dot(j_x, j_y) / (np.linalg.norm(j_x) * np.linalg.norm(j_y) + 1e-9))
    if abs(cos_ang) > 0.95:
        print(f"[CAL] WARNING: +X and +Y produced nearly parallel cam motion (|cos|={abs(cos_ang):.3f}).")

    print("\n[CAL] Built stereo Jacobian J:")
    print_jacobian(J)
    print(f"      |j_x|={np.linalg.norm(j_x):.3f}, |j_y|={np.linalg.norm(j_y):.3f}, axis_angle={np.degrees(np.arccos(np.clip(cos_ang,-1,1))):.1f}°")

    JACOBIAN_MATRIX = J
    save_jacobian(J, JACOBIAN_PATH)
    return True


def run_stereo_fine_pd_live_target(robot: Robot, stereo, detector, calib) -> bool:
    if JACOBIAN_MATRIX is None:
        print("[PD] No stereo Jacobian loaded. Press k first.")
        return False

    print("\n[PD] Fine alignment to LIVE stereo target tag.")
    print("     Requires EE ID0 and target ID2 visible in both stereo views.")

    prev_error: np.ndarray | None = None
    filtered_error: np.ndarray | None = None
    prev_t = time.perf_counter()
    best_err = float("inf")
    growth_count = 0

    for i in range(1, MAX_ITERS + 1):
        tags, abort = measure_stereo_one_valid(
            stereo,
            detector,
            calib,
            required_ids=(EE_TAG_ID, TARGET_TAG_ID),
            max_wait_s=MAX_WAIT_SEC,
            filtered_error=filtered_error,
        )
        if abort:
            return False
        if tags is None or EE_TAG_ID not in tags or TARGET_TAG_ID not in tags:
            print("[PD] Missing EE/target tag this cycle; retrying.")
            continue

        now = time.perf_counter()
        dt = max(now - prev_t, 1e-3)
        prev_t = now

        delta_cam = tags[TARGET_TAG_ID].xyz_cam_mm - tags[EE_TAG_ID].xyz_cam_mm
        measured_error = camera_delta_to_robot_delta(delta_cam)

        if filtered_error is None:
            filtered_error = measured_error.copy()
        else:
            filtered_error = FILTER_ALPHA * measured_error + (1.0 - FILTER_ALPHA) * filtered_error

        err_norm = float(np.linalg.norm(filtered_error))
        x_fk, y_fk, z_fk, phi_fk = robot.fk()

        if prev_error is None:
            d_error = np.zeros(2, dtype=np.float64)
        else:
            d_error = (filtered_error - prev_error) / dt
        prev_error = filtered_error.copy()

        raw_cmd = KP * filtered_error + KD * d_error
        cmd = clip_axis(raw_cmd, MAX_AXIS_STEP_MM)

        print(f"[{i:02d}] errXY=({filtered_error[0]:+.1f},{filtered_error[1]:+.1f}) |err|={err_norm:.1f}  FK=({x_fk:+.1f},{y_fk:+.1f},{z_fk:+.1f}) cmd=({cmd[0]:+.2f},{cmd[1]:+.2f})")

        if err_norm <= DONE_THRESH_MM:
            print(f"[PD] Done: error {err_norm:.1f} mm <= {DONE_THRESH_MM:.1f} mm.")
            return True

        if STOP_IF_ERROR_GROWS:
            if err_norm < best_err:
                best_err = err_norm
                growth_count = 0
            elif err_norm > best_err + GROWTH_MARGIN_MM:
                growth_count += 1
                print(f"[WARN] Error growing: best={best_err:.1f}, now={err_norm:.1f}, count={growth_count}/{GROWTH_COUNT_LIMIT}")
                if growth_count >= GROWTH_COUNT_LIMIT:
                    print("[PD] Stopping because error is growing. Re-run k or check signs.")
                    return False

        if abs(cmd[0]) < 0.05 and abs(cmd[1]) < 0.05:
            print("[PD] Command too small; stopping.")
            return False

        if not robot.jog(dx=float(cmd[0]), dy=float(cmd[1]), move_time_s=PD_MOVE_TIME_S):
            print("[PD] Robot jog failed.")
            return False

        if PD_SETTLE_SEC > 0:
            time.sleep(PD_SETTLE_SEC)

    print("[PD] Hit max iterations.")
    return False


# ============================================================
# COARSE MOVE / CLAW
# ============================================================

def run_homography_coarse_move(robot: Robot, target_xy: np.ndarray) -> bool:
    x_target, y_target = float(target_xy[0]), float(target_xy[1])

    if ENFORCE_WORKSPACE_LIMITS and not in_workspace(x_target, y_target):
        print(f"[COARSE] BLOCKED: homography target outside workspace bounds: ({x_target:.1f},{y_target:.1f}). Not clamping and not moving.")
        return False

    x_cur, y_cur, z_cur, phi_cur = robot.fk()
    z_cmd = z_cur if COARSE_Z_MODE == "current" else COARSE_FIXED_Z_MM
    phi_cmd = phi_cur if COARSE_PHI_MODE == "current" else COARSE_FIXED_PHI_DEG

    print(f"\n[COARSE] Homography move:")
    print(f"         current FK=({x_cur:.1f},{y_cur:.1f},{z_cur:.1f}, phi={phi_cur:.1f})")
    print(f"         target XY=({x_target:.1f},{y_target:.1f})  z={z_cmd:.1f} phi={phi_cmd:.1f}")

    # Fail closed before commanding. If the homography target lies inside a
    # keep-out zone at the commanded Z, do not move. The fix is to choose a
    # lower travel Z or skip this target; do not silently drive into the frame.
    if hasattr(robot, "check_cartesian_pose_safe"):
        safe, why = robot.check_cartesian_pose_safe(x_target, y_target, z_cmd)
        if not safe:
            zmax = robot.max_safe_z_at_xy(x_target, y_target) if hasattr(robot, "max_safe_z_at_xy") else None
            extra = f" Try z <= {zmax:.1f} mm, or skip this target." if zmax is not None else ""
            print(f"[COARSE] BLOCKED by soft limits: {why}.{extra}")
            return False

    ok = robot.move_cartesian(
        x_mm=x_target,
        y_mm=y_target,
        z_mm=z_cmd,
        phi_deg=phi_cmd,
        move_time_s=COARSE_MOVE_TIME_S,
    )
    if not ok:
        print("[COARSE] move_cartesian failed.")
        return False

    if COARSE_SETTLE_S > 0:
        time.sleep(COARSE_SETTLE_S)

    robot.sync_estimate_from_teensy_steps()
    robot.print_estimate()
    print("[COARSE] Done.")
    return True


def claw_open(robot: Robot) -> None:
    print(f"[CLAW] Opening (servo={CLAW_OPEN_DEG}°)")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)


def claw_close(robot: Robot) -> None:
    print(f"[CLAW] Closing (servo={CLAW_CLOSED_DEG}°)")
    robot.servo(CLAW_CLOSED_DEG)
    time.sleep(CLAW_SETTLE_S)


def run_go(robot, overhead, stereo, detector, H_img_to_robot, stereo_calib) -> bool:
    print("\n[GO] === overhead target -> coarse move -> stereo fine PD ===")
    target_xy = save_overhead_target(overhead, detector, H_img_to_robot, robot)
    if target_xy is None:
        print("[GO] Could not get overhead target. Aborting.")
        return False

    if not run_homography_coarse_move(robot, target_xy):
        print("[GO] Coarse move failed.")
        return False

    if JACOBIAN_MATRIX is None:
        print("[GO] No stereo Jacobian loaded. Coarse move complete; skipping fine PD.")
        return False

    return run_stereo_fine_pd_live_target(robot, stereo, detector, stereo_calib)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nHybrid Homography Coarse + Stereo Fine Test")
    print("-------------------------------------------")
    print("Overhead ID2 -> robot XY coarse move; stereo ID0/ID2 -> final PD.\n")
    print_startup_config("homography_coarse_stereo_fine_SAFE.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("homography_coarse_stereo_fine_SAFE.py")

    H_img_to_robot = load_homography(HOMOGRAPHY_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    load_jacobian_if_present(JACOBIAN_PATH)

    robot = Robot(ROBOT_CONFIG, connect=True)
    overhead = None
    stereo = None
    saved_target_xy: np.ndarray | None = None

    try:
        require_robot_soft_limits_loaded(robot, "homography_coarse_stereo_fine_SAFE.py")
        robot.enable(True)
        robot.init_drivers()
        claw_open(robot)

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
        else:
            print("Unknown choice. Aborting.")
            return

        robot.print_estimate()

        detector = build_detector()
        overhead = open_overhead_camera()
        stereo = SimpleStereoCamera(
            index=STEREO_INDEX,
            width=STEREO_WIDTH,
            height=STEREO_HEIGHT,
            fps=STEREO_FPS,
            fourcc=STEREO_FOURCC,
        )

        cv2.namedWindow(WINDOW_OVERHEAD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_OVERHEAD, 960, 540)
        cv2.namedWindow(WINDOW_STEREO, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_STEREO, 1280, 520)

        print("\nControls:")
        print("  v validate homography")
        print("  t save overhead target")
        print("  m coarse move only")
        print("  k calibrate stereo Jacobian")
        print("  s stereo fine PD only")
        print("  g GO: overhead target -> coarse -> stereo fine")
        print("  o/l open/close claw")
        print("  p print pose")
        print("  h/c/e/d home/sync/enable/disable")
        print("  q quit\n")

        while True:
            frame, dets = read_overhead_once(overhead, detector)
            if frame is not None:
                cv2.imshow(WINDOW_OVERHEAD, render_overhead(frame, dets, H_img_to_robot, robot, saved_target_xy))

            left, right, det_l, det_r, tags = read_stereo_tags_once(stereo, detector, stereo_calib)
            if left is not None and right is not None:
                cv2.imshow(WINDOW_STEREO, render_stereo_preview(left, right, det_l, det_r, tags))

            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), 27):
                break

            elif key == ord("v"):
                validate_homography_once(overhead, detector, H_img_to_robot, robot, saved_target_xy)

            elif key == ord("t"):
                target = save_overhead_target(overhead, detector, H_img_to_robot, robot)
                if target is not None:
                    saved_target_xy = target
                    print(f"[TARGET] Saved overhead target XY=({saved_target_xy[0]:.1f},{saved_target_xy[1]:.1f})")

            elif key == ord("m"):
                if saved_target_xy is None:
                    print("[COARSE] Press t first to save overhead target.")
                else:
                    run_homography_coarse_move(robot, saved_target_xy)

            elif key == ord("k"):
                run_jacobian_calibration(robot, stereo, detector, stereo_calib)

            elif key == ord("s"):
                if JACOBIAN_MATRIX is None:
                    print("[PD] Press k first or load stereo_pd_jacobian.npz.")
                else:
                    run_stereo_fine_pd_live_target(robot, stereo, detector, stereo_calib)

            elif key == ord("g"):
                run_go(robot, overhead, stereo, detector, H_img_to_robot, stereo_calib)

            elif key == ord("o"):
                claw_open(robot)

            elif key == ord("l"):
                claw_close(robot)

            elif key == ord("p"):
                robot.print_estimate()

            elif key == ord("h"):
                robot.home()
                robot.print_estimate()

            elif key == ord("c"):
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == ord("e"):
                robot.enable(True)
                robot.init_drivers()

            elif key == ord("d"):
                robot.enable(False)

    finally:
        if overhead is not None:
            overhead.release()
        if stereo is not None:
            stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
