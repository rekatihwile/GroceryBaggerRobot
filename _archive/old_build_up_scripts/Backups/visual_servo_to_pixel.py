from __future__ import annotations

"""
visual_servo_to_pixel.py

First closed-loop visual servo test for the grocery bagger.

Goal:
    Use the stereo camera to keep the end-effector AprilTag centered at a saved
    pixel target. This is NOT target-object tracking yet. It only proves:

        AprilTag pixel error -> inverse image Jacobian -> small robot XY jog

Controls in camera window:
    SPACE  save current EE tag pixel as target
    s      start visual servo loop to saved target
    n      nudge robot away from target (+X, +Y) for testing
    p      print robot estimate
    h      HOME robot
    c      sync Python estimate from Teensy POS, no homing
    e/d    enable / disable motors
    q/ESC  quit

Required files in same folder:
    robot.py
    stereo_apriltag_viewer.py
"""

import csv
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
from camera_config import EE_TAG_ID as TAG_ID, OVERHEAD_INDEX, STEREO_INDEX
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

# From your latest robot_camera_motion_test.py run.
# camera_delta_px = J_IMAGE @ robot_delta_mm
#
#   +X 10 mm -> avg du=-3.18 px,  avg dv=-9.66 px
#   +Y 10 mm -> avg du=-10.13 px, avg dv=-1.06 px
#
# So:
#   du/dx=-0.318, du/dy=-1.013
#   dv/dx=-0.966, dv/dy=-0.106
J_IMAGE = np.array([
    [-0.318, -1.013],
    [-0.966, -0.106],
], dtype=float)

# Servo behavior.
GAIN = 0.45                 # Start conservative. Try 0.25-0.60.
MAX_STEP_MM = 5.0           # Maximum XY jog per correction iteration.
DONE_THRESH_PX = 1.5        # Stop when pixel error norm is below this.
MAX_ITERS = 20
MOVE_TIME_S = 0.85
SETTLE_SEC = 0.35

# Measurement averaging.
SAMPLES_PER_MEASUREMENT = 12
MAX_WAIT_SEC = 4.0

# Test nudge.
NUDGE_DX_MM = 12.0
NUDGE_DY_MM = 0.0

LOG_PATH = Path("visual_servo_to_pixel_log.csv")


# ============================================================
# DATA
# ============================================================

@dataclass
class PixelMeasurement:
    t: float

    # Averaged stereo image coordinates.
    u: float
    v: float

    # Raw left/right values for logging.
    ul: float
    vl: float
    ur: float
    vr: float
    disparity: float
    theta_deg: float

    # Robot estimate at measurement time.
    robot_x: float
    robot_y: float
    robot_z: float
    robot_phi: float


# ============================================================
# CAMERA HELPERS
# ============================================================

def draw_cross(img: np.ndarray, center: tuple[int, int], color=(0, 255, 0), size: int = 12, thickness: int = 2):
    x, y = center
    cv2.line(img, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def read_one_detection(stereo: SimpleStereoCamera, detector, robot: Robot):
    ok, frame, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        return None, None, None, None

    det_l_all = detect_tags(detector, left)
    det_r_all = detect_tags(detector, right)

    det_l = det_l_all.get(TAG_ID)
    det_r = det_r_all.get(TAG_ID)

    if det_l is None or det_r is None:
        return left, right, det_l, det_r

    ul, vl = det_l.center
    ur, vr = det_r.center
    disparity = ul - ur
    theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)

    # Use average of left/right pixel centers for first 2D visual servo test.
    u = 0.5 * (float(ul) + float(ur))
    v = 0.5 * (float(vl) + float(vr))

    x, y, z, phi = robot.fk()

    meas = PixelMeasurement(
        t=time.time(),
        u=u,
        v=v,
        ul=float(ul),
        vl=float(vl),
        ur=float(ur),
        vr=float(vr),
        disparity=float(disparity),
        theta_deg=float(theta),
        robot_x=float(x),
        robot_y=float(y),
        robot_z=float(z),
        robot_phi=float(phi),
    )

    return left, right, det_l, det_r, meas


def measure_tag_average(
    stereo: SimpleStereoCamera,
    detector,
    robot: Robot,
    sample_count: int = SAMPLES_PER_MEASUREMENT,
    max_wait_s: float = MAX_WAIT_SEC,
    window_name: str = "Visual Servo To Pixel",
    target: PixelMeasurement | None = None,
    mode_line: str = "",
) -> PixelMeasurement | None:
    samples: list[PixelMeasurement] = []
    start = time.time()

    while time.time() - start < max_wait_s:
        result = read_one_detection(stereo, detector, robot)

        if result[0] is None:
            continue

        left, right, det_l, det_r, *maybe_meas = result
        meas = maybe_meas[0] if maybe_meas else None

        if meas is not None:
            samples.append(meas)

        left_draw = draw_detection(left, det_l, "LEFT")
        right_draw = draw_detection(right, det_r, "RIGHT")

        lines = [
            mode_line or "Measuring EE tag",
            f"valid samples: {len(samples)}/{sample_count}",
            f"target saved: {target is not None}",
            "q/ESC abort",
        ]

        if meas is not None:
            err_text = ""
            if target is not None:
                eu = target.u - meas.u
                ev = target.v - meas.v
                err_text = f" | err=({eu:+.1f},{ev:+.1f}) norm={np.hypot(eu, ev):.1f}px"
            lines.insert(2, f"u={meas.u:.1f}, v={meas.v:.1f}, disp={meas.disparity:+.2f}{err_text}")

        preview = make_preview(left_draw, right_draw, lines)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None

        if len(samples) >= sample_count:
            arr = np.array([
                [s.u, s.v, s.ul, s.vl, s.ur, s.vr, s.disparity, s.theta_deg,
                 s.robot_x, s.robot_y, s.robot_z, s.robot_phi]
                for s in samples
            ], dtype=float)
            mean = arr.mean(axis=0)

            return PixelMeasurement(
                t=time.time(),
                u=float(mean[0]),
                v=float(mean[1]),
                ul=float(mean[2]),
                vl=float(mean[3]),
                ur=float(mean[4]),
                vr=float(mean[5]),
                disparity=float(mean[6]),
                theta_deg=float(mean[7]),
                robot_x=float(mean[8]),
                robot_y=float(mean[9]),
                robot_z=float(mean[10]),
                robot_phi=float(mean[11]),
            )

    print("[WARN] Timed out waiting for EE tag.")
    return None


def live_preview_until_key(
    stereo: SimpleStereoCamera,
    detector,
    robot: Robot,
    target: PixelMeasurement | None,
    window_name: str = "Visual Servo To Pixel",
):
    while True:
        result = read_one_detection(stereo, detector, robot)
        if result[0] is None:
            continue

        left, right, det_l, det_r, *maybe_meas = result
        meas = maybe_meas[0] if maybe_meas else None

        left_draw = draw_detection(left, det_l, "LEFT")
        right_draw = draw_detection(right, det_r, "RIGHT")

        lines = [
            "SPACE target | s servo | n nudge | h home | c sync | p print | e/d enable | q quit",
            f"target saved: {target is not None}",
        ]

        if meas is not None:
            lines.append(f"EE pixel avg: u={meas.u:.1f}, v={meas.v:.1f}, disp={meas.disparity:+.2f}")
        else:
            lines.append("EE tag not detected in both views")

        if target is not None and meas is not None:
            eu = target.u - meas.u
            ev = target.v - meas.v
            lines.append(f"pixel error: eu={eu:+.1f}, ev={ev:+.1f}, norm={np.hypot(eu, ev):.1f}px")

        preview = make_preview(left_draw, right_draw, lines)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            return key, meas


# ============================================================
# SERVO
# ============================================================

def clamp_vector(dxdy: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(dxdy))
    if norm <= max_norm or norm < 1e-9:
        return dxdy
    return dxdy * (max_norm / norm)


def save_log(rows: list[dict]):
    if not rows:
        return

    with LOG_PATH.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved servo log to: {LOG_PATH.resolve()}")


def visual_servo_loop(robot: Robot, stereo: SimpleStereoCamera, detector, target: PixelMeasurement):
    print()
    print("Starting visual servo loop...")
    print(f"Target pixel: u={target.u:.2f}, v={target.v:.2f}")
    print("Keep hand near E-stop.")
    print()

    J_pinv = np.linalg.pinv(J_IMAGE)
    rows: list[dict] = []

    for k in range(MAX_ITERS):
        meas = measure_tag_average(
            stereo,
            detector,
            robot,
            target=target,
            mode_line=f"Servo iter {k+1}/{MAX_ITERS}",
        )

        if meas is None:
            print("Measurement failed/aborted. Stopping servo.")
            break

        error_px = np.array([target.u - meas.u, target.v - meas.v], dtype=float)
        err_norm = float(np.linalg.norm(error_px))

        print(f"[{k+1:02d}] error_px = [{error_px[0]:+.2f}, {error_px[1]:+.2f}], norm={err_norm:.2f}px")

        rows.append({
            "iter": k + 1,
            "t": f"{time.time():.6f}",
            "target_u": f"{target.u:.6f}",
            "target_v": f"{target.v:.6f}",
            "meas_u": f"{meas.u:.6f}",
            "meas_v": f"{meas.v:.6f}",
            "err_u": f"{error_px[0]:.6f}",
            "err_v": f"{error_px[1]:.6f}",
            "err_norm_px": f"{err_norm:.6f}",
            "robot_x": f"{meas.robot_x:.6f}",
            "robot_y": f"{meas.robot_y:.6f}",
            "disparity": f"{meas.disparity:.6f}",
            "cmd_dx": "",
            "cmd_dy": "",
        })

        if err_norm <= DONE_THRESH_PX:
            print("Done: pixel error is inside threshold.")
            break

        raw_delta_mm = GAIN * (J_pinv @ error_px)
        delta_mm = clamp_vector(raw_delta_mm, MAX_STEP_MM)

        dx = float(delta_mm[0])
        dy = float(delta_mm[1])

        rows[-1]["cmd_dx"] = f"{dx:.6f}"
        rows[-1]["cmd_dy"] = f"{dy:.6f}"

        print(f"      robot jog command: dx={dx:+.2f} mm, dy={dy:+.2f} mm")

        if not robot.jog(dx=dx, dy=dy, move_time_s=MOVE_TIME_S):
            print("Robot jog failed. Stopping servo.")
            break

        time.sleep(SETTLE_SEC)

    save_log(rows)
    print("Visual servo loop finished.")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("Visual Servo To Pixel")
    print("---------------------")
    print("This uses the EE AprilTag only. Save current pixel as target, nudge away, then servo back.")
    print()
    print_startup_config("visual_servo_to_pixel.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("visual_servo_to_pixel.py")

    robot = Robot(ROBOT_CONFIG, connect=True)
    stereo = None

    try:
        require_robot_soft_limits_loaded(robot, "visual_servo_to_pixel.py")
        robot.enable(True)
        robot.init_drivers()

        print()
        print("Startup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()

        if choice == "h":
            if not robot.home():
                print("HOME failed. Aborting.")
                return
        elif choice == "c":
            print("Reading POS and reconstructing Python estimate from Teensy steps...")
            if not robot.sync_estimate_from_teensy_steps():
                print("POS sync failed. Aborting.")
                return
        elif choice == "a":
            print("Assuming physical robot is at configured home_pose without moving.")
            robot.assume_homed()
        else:
            print("Unknown choice. Aborting.")
            return

        robot.print_estimate()

        detector = build_detector()
        stereo = SimpleStereoCamera()

        window_name = "Visual Servo To Pixel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 520)

        target: PixelMeasurement | None = None

        while True:
            key, live_meas = live_preview_until_key(stereo, detector, robot, target, window_name=window_name)

            if key in (ord("q"), 27):
                break

            elif key == ord(" "):
                print("Saving current EE tag pixel as target...")
                target = measure_tag_average(
                    stereo,
                    detector,
                    robot,
                    target=None,
                    mode_line="Saving target pixel",
                )
                if target is not None:
                    print(f"Target saved: u={target.u:.2f}, v={target.v:.2f}, disp={target.disparity:+.2f}")

            elif key == ord("s"):
                if target is None:
                    print("No target saved yet. Press SPACE first.")
                else:
                    visual_servo_loop(robot, stereo, detector, target)

            elif key == ord("n"):
                print(f"Nudging robot by dx={NUDGE_DX_MM:+.1f}, dy={NUDGE_DY_MM:+.1f} mm...")
                robot.jog(dx=NUDGE_DX_MM, dy=NUDGE_DY_MM, move_time_s=MOVE_TIME_S)

            elif key == ord("p"):
                robot.print_estimate()

            elif key == ord("h"):
                robot.home()
                robot.print_estimate()

            elif key == ord("c"):
                robot.sync_estimate_from_teensy_steps()

            elif key == ord("e"):
                robot.enable(True)
                robot.init_drivers()

            elif key == ord("d"):
                robot.enable(False)

    finally:
        if stereo is not None:
            stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
