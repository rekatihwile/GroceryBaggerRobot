from __future__ import annotations

"""
test_calibration_bundle_live.py

Live test for robot_calibration_bundle.npz.

Use this AFTER calibrate_all_safe_grid.py global_coarse succeeds.

What it tests:
  1) Loads robot_calibration_bundle.npz.
  2) Opens overhead camera.
  3) Detects EE tag ID0 and target tag ID2.
  4) Maps tag pixels through height-indexed H(z).
  5) For EE tag, compares mapped XY to robot FK XY.
  6) Reports nearest calibration support distance.
  7) Optionally performs a safe coarse move to ID2 at travel height.

Controls:
  z / x     decrease/increase target/object z used for ID2 lookup
  [ / ]     jog robot Z down/up safely
  v         print current EE validation error
  m         safe coarse move to ID2 XY at travel height
  p         print robot estimate
  h / c / a home / sync / assume home
  e / d     enable / disable
  q / ESC   quit

Expected files:
  robot.py
  robot_config.py
  camera_config.py
  stereo_apriltag_viewer.py
  robot_calibration_bundle.npz
  soft_limits_config.json
"""

from pathlib import Path
import time
import math

import cv2
import numpy as np

from robot import Robot
from robot_config import (
    ROBOT_CONFIG,
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    print_startup_config,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    TARGET_TAG_ID,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    OVERHEAD_HEIGHT,
    OVERHEAD_FPS,
    OVERHEAD_FOURCC,
    STEREO_INDEX,
)
from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import build_detector, detect_tags, draw_detection


BUNDLE_PATH = Path("robot_calibration_bundle.npz")
WINDOW = "Calibration Bundle Live Test"

TARGET_Z_MM = 50.0
Z_STEP_MM = 5.0
Z_JOG_MM = 5.0

# Coarse move behavior.
COARSE_CLEARANCE_ABOVE_TARGET_MM = 150
MIN_COARSE_TRAVEL_Z_MM = 50.0
COARSE_MOVE_TIME_S = 1.10

# Runtime acceptance thresholds.
WARN_EE_ERROR_MM = 15.0
MAX_EE_ERROR_MM = 30.0
MAX_NEAREST_SAMPLE_DIST_MM_FALLBACK = 200.0

# ============================================================
# PHI / WRIST ORIENTATION SETTINGS
# ============================================================
# The AprilTag detector returns det.theta_deg from the target tag corners.
# This is probably NOT the same zero/sign convention as your robot phi.
# Tune:
#   - If wrist rotates the wrong direction: set PHI_SIGN = -1.0
#   - If it is consistently offset: change PHI_OFFSET_DEG, often +/-90 or 180.
USE_TARGET_TAG_THETA_FOR_PHI = True
PHI_SIGN = 1.0
PHI_OFFSET_DEG = 0.0

# Optional: snap orientations to nearest increment, e.g. 90.0. Leave None first.
PHI_SNAP_DEG = None

# Optional wrist command clamp. Leave None unless needed.
PHI_MIN_DEG = None
PHI_MAX_DEG = None


def wrap_deg_180(angle_deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def snap_deg(angle_deg: float, snap_deg_value) -> float:
    """Optionally snap angle to nearest snap increment."""
    if snap_deg_value is None or float(snap_deg_value) <= 0:
        return float(angle_deg)
    return round(float(angle_deg) / float(snap_deg_value)) * float(snap_deg_value)


def clamp_phi_deg(angle_deg: float) -> float:
    """Optionally clamp phi to a configured range."""
    a = float(angle_deg)
    if PHI_MIN_DEG is not None:
        a = max(float(PHI_MIN_DEG), a)
    if PHI_MAX_DEG is not None:
        a = min(float(PHI_MAX_DEG), a)
    return a


def marker_theta_to_robot_phi(theta_deg: float) -> float:
    """
    Convert target marker angle into robot wrist/world phi.

    First test this with only coarse move / no pick. The sign/offset will likely need tuning.
    """
    phi = PHI_SIGN * float(theta_deg) + PHI_OFFSET_DEG
    phi = wrap_deg_180(phi)
    phi = snap_deg(phi, PHI_SNAP_DEG)
    phi = wrap_deg_180(phi)
    phi = clamp_phi_deg(phi)
    return float(phi)


def open_overhead_camera():
    return SimpleOverheadCamera().cap

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

    if "max_runtime_nearest_sample_dist_mm" in keys:
        max_nearest = float(np.asarray(data["max_runtime_nearest_sample_dist_mm"]).reshape(-1)[0])
    else:
        max_nearest = MAX_NEAREST_SAMPLE_DIST_MM_FALLBACK

    rms = np.asarray(data.get("homography_rms_error_mm", np.full(len(z_levels), np.nan)), dtype=np.float64)
    n_pts = np.asarray(data.get("homography_n_points", np.zeros(len(z_levels))), dtype=np.int32)

    print(f"[Bundle] Loaded {path.resolve()}")
    print(f"[Bundle] z_levels={z_levels}")
    print(f"[Bundle] used_intrinsics={used_intrinsics}")
    print(f"[Bundle] homography RMS={rms}")
    print(f"[Bundle] homography n={n_pts}")
    print(f"[Bundle] support samples={len(support_xyz)}")
    print(f"[Bundle] max nearest sample dist={max_nearest:.1f} mm")

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
    }


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


def map_uv_z_to_robot_xy(uv_raw, z_query, bundle):
    uv = undistort_uv(uv_raw, bundle["K"], bundle["dist"])
    z_levels = bundle["z_levels"]
    Hs = bundle["Hs"]
    zq = float(z_query)

    if len(z_levels) == 0:
        raise RuntimeError("No z levels in bundle.")

    if zq <= z_levels[0]:
        xy = apply_H(uv, Hs[0])
        return xy, uv, 0, 0, 0.0

    if zq >= z_levels[-1]:
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

        # Draw target tag orientation arrow from detector theta.
        theta_rad = math.radians(float(det_tg.theta_deg))
        p0 = np.asarray(det_tg.center, dtype=float)
        p1 = p0 + 45.0 * np.array([math.cos(theta_rad), math.sin(theta_rad)])
        cv2.arrowedLine(
            out,
            tuple(np.round(p0).astype(int)),
            tuple(np.round(p1).astype(int)),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )

    for i, line in enumerate(lines):
        y = 30 + i * 27
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def choose_safe_travel_z(robot, target_x, target_y, target_z):
    """Pick travel Z above target, but only if soft-limit-safe."""
    desired = max(
        MIN_COARSE_TRAVEL_Z_MM,
        float(target_z) + COARSE_CLEARANCE_ABOVE_TARGET_MM,
    )

    # Try desired first, then lower values if frame soft limits make high Z invalid.
    # Never go below target_z + 15 mm.
    min_allowed = float(target_z) + 15.0
    candidates = [desired, DEFAULT_TRAVEL_Z_MM, HOME_Z_MM, 75.0, 65.0, 50.0, 40.0]

    # Unique, descending-ish but preserve order.
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
            print(f"[TRAVEL_Z] Reject z={z:.1f}: target unsafe: {reason_pose}")
            continue

        ok_path, path, reason_path = robot.plan_cartesian_path(target_x, target_y, z)
        if not ok_path:
            print(f"[TRAVEL_Z] Reject z={z:.1f}: path unsafe: {reason_path}")
            continue

        print(f"[TRAVEL_Z] Selected z={z:.1f} mm")
        return z

    return None


def main():
    print("\nCalibration Bundle Live Test")
    print("----------------------------")
    print_startup_config("test_calibration_bundle_live.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("test_calibration_bundle_live.py")

    print("\n[PHI SETTINGS]")
    print(f"  USE_TARGET_TAG_THETA_FOR_PHI = {USE_TARGET_TAG_THETA_FOR_PHI}")
    print(f"  PHI_SIGN = {PHI_SIGN}")
    print(f"  PHI_OFFSET_DEG = {PHI_OFFSET_DEG}")
    print(f"  PHI_SNAP_DEG = {PHI_SNAP_DEG}")
    print("  If orientation is wrong: flip PHI_SIGN or change PHI_OFFSET_DEG.\n")

    bundle = load_bundle(BUNDLE_PATH)
    detector = build_detector()
    cap = open_overhead_camera()
    robot = Robot(ROBOT_CONFIG, connect=True)

    target_z = TARGET_Z_MM
    last_target_xy = None
    last_target_support_dist = np.inf
    last_target_theta_deg = None
    last_target_phi_deg = None

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
                "z/x target height | [/] jog Z | v validate | m coarse ID2+phi | p print | h/c sync | q quit",
                f"FK=({x_fk:.1f},{y_fk:.1f},{z_fk:.1f}) phi={phi_fk:.1f}  target_z={target_z:.1f}",
                f"bundle z={bundle['z_levels'][0]:.0f}..{bundle['z_levels'][-1]:.0f}  support={len(bundle['support_xyz'])}",
                f"EE visible={det_ee is not None}  target ID{TARGET_TAG_ID} visible={det_tg is not None}",
            ]

            ee_err_norm = None
            if det_ee is not None:
                ee_xy, ee_uv, lo, hi, alpha = map_uv_z_to_robot_xy(det_ee.center, z_fk, bundle)
                ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                ee_err_norm = float(np.linalg.norm(ee_err))
                support_dist, _ = nearest_support_distance(ee_xy, z_fk, bundle)

                status = "OK"
                if ee_err_norm > MAX_EE_ERROR_MM:
                    status = "BAD"
                elif ee_err_norm > WARN_EE_ERROR_MM:
                    status = "WARN"

                lines.append(
                    f"EE map {status}: xy=({ee_xy[0]:.1f},{ee_xy[1]:.1f}) "
                    f"err=({ee_err[0]:+.1f},{ee_err[1]:+.1f}) |e|={ee_err_norm:.1f}mm "
                    f"support={support_dist:.1f}mm"
                )

            if det_tg is not None:
                tg_xy, tg_uv, lo, hi, alpha = map_uv_z_to_robot_xy(det_tg.center, target_z, bundle)
                support_dist, _ = nearest_support_distance(tg_xy, target_z, bundle)
                last_target_xy = tg_xy
                last_target_support_dist = support_dist
                last_target_theta_deg = float(det_tg.theta_deg)

                if USE_TARGET_TAG_THETA_FOR_PHI:
                    last_target_phi_deg = marker_theta_to_robot_phi(last_target_theta_deg)
                else:
                    last_target_phi_deg = phi_fk

                support_status = "OK" if support_dist <= bundle["max_nearest"] else "FAR"
                lines.append(
                    f"ID{TARGET_TAG_ID} map: xy=({tg_xy[0]:.1f},{tg_xy[1]:.1f}) "
                    f"z={target_z:.1f} layers={lo}/{hi} a={alpha:.2f} support={support_dist:.1f}mm {support_status}"
                )
                lines.append(
                    f"ID{TARGET_TAG_ID} theta={last_target_theta_deg:+.1f} deg "
                    f"-> phi_cmd={last_target_phi_deg:+.1f} deg "
                    f"(sign={PHI_SIGN:+.0f}, offset={PHI_OFFSET_DEG:+.1f})"
                )

            cv2.imshow(WINDOW, draw_status(frame, det_ee, det_tg, lines))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("z"):
                target_z -= Z_STEP_MM
                print(f"[Z] target_z = {target_z:.1f} mm")

            elif key == ord("x"):
                target_z += Z_STEP_MM
                print(f"[Z] target_z = {target_z:.1f} mm")

            elif key == ord("["):
                print(f"[JOG] Z down by {Z_JOG_MM:.1f} mm")
                robot.jog(dz=-Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ord("]"):
                print(f"[JOG] Z up by {Z_JOG_MM:.1f} mm")
                robot.jog(dz=+Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ord("v"):
                if det_ee is None:
                    print("[VALIDATE] EE tag not visible.")
                else:
                    ee_xy, _, lo, hi, alpha = map_uv_z_to_robot_xy(det_ee.center, z_fk, bundle)
                    ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                    support_dist, support_idx = nearest_support_distance(ee_xy, z_fk, bundle)
                    print("\n[VALIDATE EE]")
                    print(f"  FK xy       = ({x_fk:.2f}, {y_fk:.2f}) at z={z_fk:.2f}")
                    print(f"  mapped xy   = ({ee_xy[0]:.2f}, {ee_xy[1]:.2f})")
                    print(f"  error       = ({ee_err[0]:+.2f}, {ee_err[1]:+.2f}) |e|={np.linalg.norm(ee_err):.2f} mm")
                    print(f"  layers      = {lo}/{hi}, alpha={alpha:.3f}")
                    print(f"  support     = {support_dist:.2f} mm nearest idx={support_idx}\n")

            elif key == ord("m"):
                if last_target_xy is None or det_tg is None:
                    print("[MOVE] No visible/mapped target ID2.")
                    continue

                if last_target_support_dist > bundle["max_nearest"]:
                    print(
                        f"[MOVE] REFUSED: target is too far from calibration support "
                        f"({last_target_support_dist:.1f} > {bundle['max_nearest']:.1f} mm)."
                    )
                    continue

                x_t, y_t = float(last_target_xy[0]), float(last_target_xy[1])
                travel_z = choose_safe_travel_z(robot, x_t, y_t, target_z)
                if travel_z is None:
                    print("[MOVE] REFUSED: no soft-limit-safe travel Z found.")
                    continue

                _, _, z_cur, phi_cur = robot.fk()
                phi_cmd = float(last_target_phi_deg if last_target_phi_deg is not None else phi_cur)

                print(
                    f"[MOVE] Coarse move to ID{TARGET_TAG_ID}: "
                    f"XY=({x_t:.1f},{y_t:.1f}), travel_z={travel_z:.1f}, "
                    f"target_z={target_z:.1f}, phi_cmd={phi_cmd:+.1f}"
                )

                # Move Z first if needed. Keep current phi during pure Z retract.
                if abs(z_cur - travel_z) > 1e-6:
                    if not robot.move_cartesian(z_mm=travel_z, move_time_s=0.75):
                        print("[MOVE] Failed moving to travel Z.")
                        continue
                    robot.sync_estimate_from_teensy_steps()

                # Then move XY and wrist orientation together at travel Z.
                if not robot.move_cartesian(
                    x_mm=x_t,
                    y_mm=y_t,
                    z_mm=travel_z,
                    phi_deg=phi_cmd,
                    move_time_s=COARSE_MOVE_TIME_S,
                ):
                    print("[MOVE] Coarse XY+phi move failed.")
                    continue

                time.sleep(0.2)
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == ord("p"):
                robot.print_estimate()

            elif key == ord("h"):
                robot.home()
                robot.print_estimate()

            elif key == ord("c"):
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == ord("a"):
                robot.assume_homed()
                robot.print_estimate()

            elif key == ord("e"):
                robot.enable(True)
                robot.init_drivers()

            elif key == ord("d"):
                robot.enable(False)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
