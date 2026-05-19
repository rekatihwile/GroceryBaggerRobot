from __future__ import annotations

"""
overhead_homography_z_lookup_test_SAFE_Z.py

Height-indexed overhead homography test with soft-limit-aware Z jogging and
safe coarse moves.

What this does:
  - Opens overhead webcam.
  - Detects EE tag ID0 and target tag ID2.
  - Uses overhead_homography_z_lookup.npz to map tag pixels to robot XY.
  - Lets you adjust the assumed target/object height used for lookup.
  - Lets you jog the robot Z up/down through robot.move_cartesian(), so the
    robot.py fail-closed soft-limit wrapper checks the move before sending it.
  - Coarse move first retracts above the estimated target height by a clearance
    margin, then moves XY at that safe travel Z.

Important convention used here:
  z_mm larger = gripper more retracted / higher above the table.
  Example from your current setup: z=100 is lifted/home-ish, z=25 is lower.

Controls:
  z / x       decrease/increase TARGET/object height used for homography lookup
  [ / ]       jog robot Z down/up safely
  m           safe coarse move to target: retract above target, then XY move
  v           validate EE homography against robot FK at current robot Z
  p           print robot estimate
  h / c / a   home / sync / assume home pose
  e / d       enable / disable motors
  q / ESC     quit

Required files:
  robot.py                         # should be the fail-closed soft-limit robot
  soft_limits.py
  soft_limits_config.json
  stereo_apriltag_viewer.py
  overhead_homography_z_lookup.npz
"""

import time
from pathlib import Path

import cv2
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "run_pickplace_fast.py").exists()),
    Path(__file__).resolve().parents[1],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from hardware.robot import Robot
from config.robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
    require_robot_soft_limits_loaded,
    require_soft_limits_configured,
)
from config.camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    OVERHEAD_Z_LOOKUP_PATH as LOOKUP_PATH,
    STEREO_INDEX,
    TARGET_TAG_ID,
)
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import build_detector, detect_tags, draw_detection

# ============================================================
# USER SETTINGS
# ============================================================

WINDOW = "Overhead Z Lookup Test - SAFE Z"

# Target/object height used for the lookup map.
TARGET_Z_MM = 0.0
TARGET_Z_STEP_MM = 5.0

# Robot Z jog controls.
ROBOT_Z_JOG_MM = 5.0
Z_MIN_MM = -10.0
Z_MAX_MM = HOME_Z_MM

# Coarse move settings.
COARSE_MOVE_TIME_S = 1.10
Z_MOVE_TIME_S = 0.75
SETTLE_S = 0.20

# Coarse move retract height logic:
#   travel_z >= target_z + COARSE_CLEARANCE_ABOVE_TARGET_MM
# and never below MIN_COARSE_TRAVEL_Z_MM.
COARSE_CLEARANCE_ABOVE_TARGET_MM = 40.0
MIN_COARSE_TRAVEL_Z_MM = 45.0

# If target travel_z is unsafe due to frame/camera boxes, try lower travel Z
# values while preserving at least MIN_CLEARANCE_ABOVE_TARGET_MM clearance.
MIN_CLEARANCE_ABOVE_TARGET_MM = 50.0
TRAVEL_Z_SEARCH_STEP_MM = 5.0

# Workspace hard bounds for mapped homography target. This prevents weird
# homography extrapolation from commanding nonsense.
WORKSPACE_X_MIN = -50.0
WORKSPACE_X_MAX = 650.0
WORKSPACE_Y_MIN = 0.0
WORKSPACE_Y_MAX = 700.0

SURVEY_X = ROBOT_CONFIG.x_survey_mm
SURVEY_Y = ROBOT_CONFIG.y_survey_mm
SURVEY_Z = ROBOT_CONFIG.z_survey_mm
SURVEY_PHI = ROBOT_CONFIG.phi_survey_deg

# ============================================================
# CAMERA / LOOKUP HELPERS
# ============================================================

def open_overhead_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def load_lookup(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing lookup file: {path.resolve()}")
    data = np.load(path, allow_pickle=False)
    z = np.asarray(data["z_levels_mm"], dtype=np.float64)
    Hs = np.asarray(data["H_img_to_robot_by_z"], dtype=np.float64)
    used_intrinsics = bool(np.asarray(data.get("used_intrinsics", [False])).reshape(-1)[0])
    K = np.asarray(data["camera_matrix"], dtype=np.float64) if used_intrinsics else None
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64) if used_intrinsics else None
    print(f"[Lookup] Loaded {path.resolve()}")
    print(f"[Lookup] z levels: {z}")
    print(f"[Lookup] used_intrinsics={used_intrinsics}")
    return z, Hs, K, dist


def undistort_uv(uv: np.ndarray, K, dist) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    if K is None or dist is None:
        return uv
    pt = uv.astype(np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def apply_H(uv: np.ndarray, H: np.ndarray) -> np.ndarray:
    pt = np.asarray(uv, dtype=np.float32).reshape(1, 1, 2)
    return cv2.perspectiveTransform(pt, H)[0, 0].astype(np.float64)


def map_uv_z_to_robot_xy(uv_raw: np.ndarray, z_query: float, z_levels: np.ndarray, Hs: np.ndarray, K, dist):
    uv = undistort_uv(uv_raw, K, dist)
    zq = float(z_query)

    if zq <= z_levels[0]:
        return apply_H(uv, Hs[0]), uv, 0, 0, 0.0
    if zq >= z_levels[-1]:
        i = len(z_levels) - 1
        return apply_H(uv, Hs[i]), uv, i, i, 0.0

    hi = int(np.searchsorted(z_levels, zq))
    lo = hi - 1
    alpha = float((zq - z_levels[lo]) / (z_levels[hi] - z_levels[lo]))

    xy_lo = apply_H(uv, Hs[lo])
    xy_hi = apply_H(uv, Hs[hi])
    xy = (1.0 - alpha) * xy_lo + alpha * xy_hi
    return xy, uv, lo, hi, alpha

# ============================================================
# SAFETY HELPERS
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def within_workspace_xy(x: float, y: float) -> bool:
    return (WORKSPACE_X_MIN <= x <= WORKSPACE_X_MAX) and (WORKSPACE_Y_MIN <= y <= WORKSPACE_Y_MAX)


def robot_pose_safe(robot: Robot, x: float, y: float, z: float) -> tuple[bool, str]:
    """Ask robot.py soft-limit wrapper if a single pose is safe."""
    fn = getattr(robot, "check_cartesian_pose_safe", None)
    if not callable(fn):
        return False, "robot.py does not expose check_cartesian_pose_safe(); unsafe to continue"
    try:
        return fn(x_mm=x, y_mm=y, z_mm=z)
    except TypeError:
        try:
            return fn(x, y, z)
        except Exception as exc:
            return False, f"soft-limit pose check failed: {exc!r}"


def robot_plan_safe(robot: Robot, x: float, y: float, z: float) -> tuple[bool, str]:
    """Ask robot.py if the planned path from current pose to target pose is safe."""
    fn = getattr(robot, "plan_cartesian_path", None)
    if not callable(fn):
        return False, "robot.py does not expose plan_cartesian_path(); unsafe to continue"
    try:
        out = fn(x_mm=x, y_mm=y, z_mm=z)
    except TypeError:
        try:
            out = fn(x, y, z)
        except Exception as exc:
            return False, f"soft-limit path check failed: {exc!r}"
    except Exception as exc:
        return False, f"soft-limit path check failed: {exc!r}"

    if not isinstance(out, tuple) or len(out) < 3:
        return False, "bad plan_cartesian_path() return value"
    ok, _path, reason = out[:3]
    return bool(ok), str(reason)


def safe_move_cartesian(robot: Robot, *, x: float | None = None, y: float | None = None, z: float | None = None, phi: float | None = None, move_time_s: float = COARSE_MOVE_TIME_S) -> bool:
    x0, y0, z0, phi0 = robot.fk()
    xt = float(x0 if x is None else x)
    yt = float(y0 if y is None else y)
    zt = float(z0 if z is None else z)
    phit = float(phi0 if phi is None else phi)

    if not within_workspace_xy(xt, yt):
        print(f"[SAFE MOVE] BLOCKED: target XY ({xt:.1f}, {yt:.1f}) outside hard workspace bounds.")
        return False

    ok_pose, reason_pose = robot_pose_safe(robot, xt, yt, zt)
    if not ok_pose:
        print(f"[SAFE MOVE] BLOCKED target pose ({xt:.1f}, {yt:.1f}, {zt:.1f}): {reason_pose}")
        return False

    ok_path, reason_path = robot_plan_safe(robot, xt, yt, zt)
    if not ok_path:
        print(f"[SAFE MOVE] BLOCKED path to ({xt:.1f}, {yt:.1f}, {zt:.1f}): {reason_path}")
        return False

    print(f"[SAFE MOVE] OK: ({x0:.1f},{y0:.1f},{z0:.1f}) -> ({xt:.1f},{yt:.1f},{zt:.1f}) | {reason_path}")
    ok = robot.move_cartesian(xt, yt, zt, phit, move_time_s=move_time_s)
    if ok:
        time.sleep(SETTLE_S)
        robot.sync_estimate_from_teensy_steps()
    return bool(ok)


def choose_safe_travel_z(robot: Robot, target_x: float, target_y: float, target_object_z: float) -> tuple[float | None, str]:
    """Choose a safe travel Z above the object for coarse XY motion.

    We start from a desired retract height above the estimated object height.
    If that target XY is unsafe at that high Z due to frame/camera keepouts, try
    progressively lower Z values while preserving minimum clearance above object.
    If none work, return None and do not move.
    """
    desired = max(MIN_COARSE_TRAVEL_Z_MM, float(target_object_z) + COARSE_CLEARANCE_ABOVE_TARGET_MM)
    desired = clamp(desired, Z_MIN_MM, Z_MAX_MM)

    min_allowed = max(float(target_object_z) + MIN_CLEARANCE_ABOVE_TARGET_MM, Z_MIN_MM)
    min_allowed = min(min_allowed, Z_MAX_MM)

    # Try high-to-low. Higher is usually safer for object collision, lower may be needed
    # to avoid the overhead frame/camera keepout.
    candidates = []
    z = desired
    while z >= min_allowed - 1e-9:
        candidates.append(round(z, 6))
        z -= TRAVEL_Z_SEARCH_STEP_MM
    if min_allowed not in candidates:
        candidates.append(round(min_allowed, 6))

    reasons = []
    for zc in candidates:
        ok_pose, reason_pose = robot_pose_safe(robot, target_x, target_y, zc)
        ok_path, reason_path = robot_plan_safe(robot, target_x, target_y, zc) if ok_pose else (False, "pose unsafe")
        if ok_pose and ok_path:
            return float(zc), f"selected travel_z={zc:.1f}; {reason_path}"
        reasons.append(f"z={zc:.1f}: pose_ok={ok_pose} pose={reason_pose}; path_ok={ok_path} path={reason_path}")

    return None, "No safe travel Z above target. " + " | ".join(reasons[:4])


def safe_coarse_move_to_target(robot: Robot, target_x: float, target_y: float, target_object_z: float) -> bool:
    if not within_workspace_xy(target_x, target_y):
        print(f"[COARSE] BLOCKED: target XY ({target_x:.1f}, {target_y:.1f}) outside hard workspace bounds.")
        return False

    x_cur, y_cur, z_cur, phi_cur = robot.fk()
    travel_z, reason = choose_safe_travel_z(robot, target_x, target_y, target_object_z)
    if travel_z is None:
        print(f"[COARSE] BLOCKED: {reason}")
        print("[COARSE] Do not move. Lower the target height estimate, adjust soft limits, or pick a different approach point.")
        return False

    print(f"[COARSE] Target XY=({target_x:.1f},{target_y:.1f}), object_z={target_object_z:.1f}, current_z={z_cur:.1f}")
    print(f"[COARSE] {reason}")

    # Step 1: retract / set Z at current XY if needed. We do this even if moving down,
    # because the soft-limit wrapper will block unsafe vertical moves.
    if abs(z_cur - travel_z) > 0.5:
        print(f"[COARSE] Z pre-move at current XY: {z_cur:.1f} -> {travel_z:.1f}")
        if not safe_move_cartesian(robot, z=travel_z, phi=phi_cur, move_time_s=Z_MOVE_TIME_S):
            print("[COARSE] Failed Z pre-move. Aborting before XY.")
            return False

    # Step 2: XY move at safe travel Z.
    print(f"[COARSE] XY move at travel_z={travel_z:.1f}")
    if not safe_move_cartesian(robot, x=target_x, y=target_y, z=travel_z, phi=phi_cur, move_time_s=COARSE_MOVE_TIME_S):
        print("[COARSE] XY move blocked/failed.")
        return False

    print("[COARSE] Done. Robot is above target; no descent commanded by this test script.")
    return True

# ============================================================
# DRAWING
# ============================================================

def draw_status(frame, det_ee, det_tg, lines):
    out = frame.copy()
    out = draw_detection(out, det_ee, f"EE {EE_TAG_ID}")
    if det_tg is not None:
        corners_i = np.round(det_tg.corners).astype(int)
        center_i = tuple(np.round(det_tg.center).astype(int))
        cv2.polylines(out, [corners_i], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, center_i, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(out, f"TARGET {TARGET_TAG_ID}", (center_i[0] + 8, center_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(out, f"TARGET {TARGET_TAG_ID}", (center_i[0] + 8, center_i[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        y = 30 + i * 27
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# ============================================================
# MAIN
# ============================================================

def main():
    print_startup_config("overhead_homography_z_lookup_test.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("overhead_homography_z_lookup_test.py")

    z_levels, Hs, K, dist = load_lookup(LOOKUP_PATH)
    detector = build_detector()
    cap = open_overhead_camera()
    robot = Robot(ROBOT_CONFIG, connect=True)
    target_z = TARGET_Z_MM
    last_target_xy: np.ndarray | None = None

    try:
        robot.enable(True)
        robot.init_drivers()

        # Extra guard: if the fail-closed robot supports this, force-enable limits.
        if hasattr(robot, "set_soft_limits_enabled"):
            robot.set_soft_limits_enabled(True)
        require_robot_soft_limits_loaded(robot, "overhead_homography_z_lookup_test.py")

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()
        if choice == "h":
            if not robot.home(): return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps(): return
        elif choice == "a":
            robot.assume_homed(); robot.print_estimate()
        else:
            print("Unknown choice. Aborting."); return

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            dets = detect_tags(detector, frame)
            det_ee = dets.get(EE_TAG_ID)
            det_tg = dets.get(TARGET_TAG_ID)

            x_fk, y_fk, z_fk, phi_fk = robot.fk()
            lines = [
                "z/x target Z -/+ | [/]: robot Z -/+ | m safe coarse | v validate | p print | h/c | q",
                f"target/object z={target_z:.1f} mm  coarse clearance={COARSE_CLEARANCE_ABOVE_TARGET_MM:.0f} mm",
                f"FK XY=({x_fk:.1f}, {y_fk:.1f}) Z={z_fk:.1f}",
                f"EE visible={det_ee is not None} target visible={det_tg is not None}",
            ]

            if det_ee is not None:
                ee_xy, _ee_uv_used, lo, hi, a = map_uv_z_to_robot_xy(det_ee.center, z_fk, z_levels, Hs, K, dist)
                ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                lines.append(f"EE mapped @ FK z: ({ee_xy[0]:.1f},{ee_xy[1]:.1f}) err=({ee_err[0]:+.1f},{ee_err[1]:+.1f}) |e|={np.linalg.norm(ee_err):.1f}mm")

            if det_tg is not None:
                tg_xy, _tg_uv_used, lo, hi, a = map_uv_z_to_robot_xy(det_tg.center, target_z, z_levels, Hs, K, dist)
                last_target_xy = tg_xy
                safe_z, safe_reason = choose_safe_travel_z(robot, float(tg_xy[0]), float(tg_xy[1]), target_z)
                if safe_z is None:
                    lines.append(f"TARGET mapped: ({tg_xy[0]:.1f}, {tg_xy[1]:.1f}) -> BLOCKED")
                else:
                    lines.append(f"TARGET mapped: ({tg_xy[0]:.1f}, {tg_xy[1]:.1f}) travel_z={safe_z:.1f}")
                lines.append(f"lookup layers {lo}/{hi} alpha={a:.2f}")

            cv2.imshow(WINDOW, draw_status(frame, det_ee, det_tg, lines))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('z'):
                target_z = clamp(target_z - TARGET_Z_STEP_MM, Z_MIN_MM, Z_MAX_MM)
                print(f"target_z = {target_z:.1f} mm")
            elif key == ord('x'):
                target_z = clamp(target_z + TARGET_Z_STEP_MM, Z_MIN_MM, Z_MAX_MM)
                print(f"target_z = {target_z:.1f} mm")
            elif key == ord('['):
                new_z = clamp(z_fk - ROBOT_Z_JOG_MM, Z_MIN_MM, Z_MAX_MM)
                print(f"[Z JOG] robot z {z_fk:.1f} -> {new_z:.1f}")
                safe_move_cartesian(robot, z=new_z, phi=phi_fk, move_time_s=Z_MOVE_TIME_S)
            elif key == ord(']'):
                new_z = clamp(z_fk + ROBOT_Z_JOG_MM, Z_MIN_MM, Z_MAX_MM)
                print(f"[Z JOG] robot z {z_fk:.1f} -> {new_z:.1f}")
                safe_move_cartesian(robot, z=new_z, phi=phi_fk, move_time_s=Z_MOVE_TIME_S)
            elif key == ord('v'):
                if det_ee is None:
                    print("[VALIDATE] No EE tag visible.")
                else:
                    ee_xy, _uv, lo, hi, a = map_uv_z_to_robot_xy(det_ee.center, z_fk, z_levels, Hs, K, dist)
                    err = ee_xy - np.array([x_fk, y_fk])
                    print(f"[VALIDATE] FK=({x_fk:.1f},{y_fk:.1f}, z={z_fk:.1f})  mapped=({ee_xy[0]:.1f},{ee_xy[1]:.1f})  err=({err[0]:+.1f},{err[1]:+.1f}) |e|={np.linalg.norm(err):.1f} mm")
            elif key == ord('p'):
                robot.print_estimate()
            elif key == ord('h'):
                robot.home(); robot.print_estimate()
            elif key == ord('c'):
                robot.sync_estimate_from_teensy_steps(); robot.print_estimate()
            elif key == ord('e'):
                robot.enable(True); robot.init_drivers()
            elif key == ord('s'):
                robot.move_cartesian(SURVEY_X, SURVEY_Y, SURVEY_Z, SURVEY_PHI, move_time_s=1.0)
            elif key == ord('d'):
                robot.enable(False)
            elif key == ord('m'):
                if last_target_xy is None:
                    print("[COARSE] No target ID2 mapped yet.")
                    continue
                tx, ty = float(last_target_xy[0]), float(last_target_xy[1])
                safe_coarse_move_to_target(robot, tx, ty, target_z)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
