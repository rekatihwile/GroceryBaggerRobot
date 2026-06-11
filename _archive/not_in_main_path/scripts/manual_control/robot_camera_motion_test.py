from __future__ import annotations

"""
robot_camera_motion_test.py

Build-up script #2:
- Connects to the robot
- Opens the stereo AprilTag viewer camera
- Moves to a known starting Cartesian pose
- Measures the end-effector AprilTag in both stereo images
- Commands small Cartesian jogs
- Measures how the tag actually moved in image/disparity space
- Saves a CSV log

This is NOT full PD control yet.
This is the sanity test that tells us:
    commanded +X robot motion -> observed camera motion
    commanded +Y robot motion -> observed camera motion
"""

import csv
import time
from dataclasses import dataclass
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
from config.camera_config import EE_TAG_ID as TAG_ID, OVERHEAD_INDEX, STEREO_INDEX

# Reuse the clean detector/camera code from stereo_apriltag_viewer.py
from hardware.cameras.stereo_apriltag_viewer import (
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

# Starting Cartesian pose after homing/syncing.
# Set START_Z_MM=None to keep whatever Z Python currently estimates.
START_X_MM = 200.0
START_Y_MM = 200.0
START_Z_MM = None
START_MOVE_TIME_S = 1.00

# Motion test settings.
JOG_MM = 10.0
MOVE_TIME_S = 0.90
SETTLE_SEC = 0.50

# Measurement averaging.
SAMPLES_PER_MEASUREMENT = 40
MAX_WAIT_SEC = 6.0

# Output.
LOG_PATH = Path("robot_camera_motion_test_log.csv")

WINDOW_NAME = "Robot Camera Motion Test"


# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass
class StereoTagMeasurement:
    name: str
    t: float

    # Robot estimate at time of measurement.
    robot_x: float
    robot_y: float
    robot_z: float
    robot_phi: float

    # Left/right image measurements.
    ul: float
    vl: float
    ur: float
    vr: float
    disparity: float
    theta_deg: float


# ============================================================
# CAMERA HELPERS
# ============================================================

def get_current_tag_frame(stereo: SimpleStereoCamera, detector, status_lines: list[str]):
    ok, frame, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        blank = np.zeros((480, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, "Could not read camera frame", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        return None, None, None, blank

    det_l_all = detect_tags(detector, left)
    det_r_all = detect_tags(detector, right)

    det_l = det_l_all.get(TAG_ID)
    det_r = det_r_all.get(TAG_ID)

    left_draw = draw_detection(left, det_l, "LEFT")
    right_draw = draw_detection(right, det_r, "RIGHT")

    lines = list(status_lines)
    lines.append(f"tag ID {TAG_ID}: left={det_l is not None}, right={det_r is not None}")

    if det_l is not None and det_r is not None:
        ul, vl = det_l.center
        ur, vr = det_r.center
        disp = ul - ur
        theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
        lines.append(f"L=({ul:.1f},{vl:.1f}) R=({ur:.1f},{vr:.1f}) disp={disp:+.2f}px theta={theta:+.1f}deg")

    preview = make_preview(left_draw, right_draw, lines)
    return det_l, det_r, left, preview


def wait_for_space(stereo: SimpleStereoCamera, detector, message: str) -> bool:
    """Show live camera until SPACE. Returns False if user quits."""
    print(message)
    print("Click/focus the camera window, then press SPACE. q/ESC aborts.")

    while True:
        _, _, _, preview = get_current_tag_frame(
            stereo,
            detector,
            [message, "SPACE = continue", "q/ESC = abort"],
        )
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            return True
        if key in (ord("q"), 27):
            return False


def measure_tag(
    stereo: SimpleStereoCamera,
    detector,
    name: str,
    robot: Robot,
    sample_count: int = SAMPLES_PER_MEASUREMENT,
    max_wait_s: float = MAX_WAIT_SEC,
) -> StereoTagMeasurement | None:
    """
    Read frames until the tag is detected in both left and right.
    Average sample_count valid detections.
    Also shows the live preview while waiting.
    """
    samples = []
    start = time.time()

    while time.time() - start < max_wait_s:
        det_l, det_r, _, preview = get_current_tag_frame(
            stereo,
            detector,
            [
                f"Measuring: {name}",
                f"Need {sample_count} valid stereo samples",
                f"Current valid samples: {len(samples)}",
                "q/ESC = abort",
            ],
        )

        if det_l is not None and det_r is not None:
            ul, vl = det_l.center
            ur, vr = det_r.center
            disp = ul - ur
            theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
            samples.append([ul, vl, ur, vr, disp, theta])

        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return None

        if len(samples) >= sample_count:
            arr = np.asarray(samples, dtype=float)
            mean = arr.mean(axis=0)
            x, y, z, phi = robot.fk()

            return StereoTagMeasurement(
                name=name,
                t=time.time(),
                robot_x=x,
                robot_y=y,
                robot_z=z,
                robot_phi=phi,
                ul=float(mean[0]),
                vl=float(mean[1]),
                ur=float(mean[2]),
                vr=float(mean[3]),
                disparity=float(mean[4]),
                theta_deg=float(mean[5]),
            )

    print(f"[WARN] Timed out waiting for stereo tag measurement: {name}")
    return None


def print_delta(label: str, a: StereoTagMeasurement, b: StereoTagMeasurement):
    du_l = b.ul - a.ul
    dv_l = b.vl - a.vl
    du_r = b.ur - a.ur
    dv_r = b.vr - a.vr
    ddisp = b.disparity - a.disparity

    dx_robot = b.robot_x - a.robot_x
    dy_robot = b.robot_y - a.robot_y
    dz_robot = b.robot_z - a.robot_z

    print()
    print("=" * 72)
    print(label)
    print(f"Robot estimated delta: dx={dx_robot:+.2f} mm, dy={dy_robot:+.2f} mm, dz={dz_robot:+.2f} mm")
    print(f"Left image delta:      du={du_l:+.2f} px, dv={dv_l:+.2f} px")
    print(f"Right image delta:     du={du_r:+.2f} px, dv={dv_r:+.2f} px")
    print(f"Disparity delta:       ddisp={ddisp:+.2f} px")
    print("=" * 72)


def save_log(measurements: list[StereoTagMeasurement]):
    with LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "t",
            "robot_x_mm", "robot_y_mm", "robot_z_mm", "robot_phi_deg",
            "ul_px", "vl_px", "ur_px", "vr_px", "disparity_px", "theta_deg",
        ])

        for m in measurements:
            writer.writerow([
                m.name, f"{m.t:.6f}",
                f"{m.robot_x:.6f}", f"{m.robot_y:.6f}", f"{m.robot_z:.6f}", f"{m.robot_phi:.6f}",
                f"{m.ul:.6f}", f"{m.vl:.6f}", f"{m.ur:.6f}", f"{m.vr:.6f}",
                f"{m.disparity:.6f}", f"{m.theta_deg:.6f}",
            ])

    print(f"\nSaved log to: {LOG_PATH.resolve()}")


def checked_move(label: str, ok: bool) -> bool:
    if ok:
        return True
    print(f"[ABORT] {label} failed. Not continuing because Python state would be wrong.")
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("Robot Camera Motion Test")
    print("------------------------")
    print("This will move to a start pose, measure the EE AprilTag, jog +X, return, jog +Y, return.")
    print("Keep your hand near power / E-stop. Use small jogs first.")
    print()
    print_startup_config("robot_camera_motion_test.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("robot_camera_motion_test.py")

    robot = Robot(ROBOT_CONFIG, connect=True)
    stereo = None

    try:
        require_robot_soft_limits_loaded(robot, "robot_camera_motion_test.py")
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
                print("HOME failed or timed out. Aborting before any jogs.")
                return
        elif choice == "c":
            print("Reading POS and reconstructing Python estimate from Teensy steps...")
            if not robot.sync_estimate_from_teensy_steps():
                print("Could not sync from POS. Aborting.")
                return
        elif choice == "a":
            print("Assuming physical robot is at configured home_pose without moving.")
            robot.assume_homed()
            robot.print_estimate()
        else:
            print("Unknown choice. Aborting.")
            return

        print(f"\nMoving to starting pose: x={START_X_MM:.1f} mm, y={START_Y_MM:.1f} mm...")
        if not checked_move(
            "Move to starting pose",
            robot.move_cartesian(
                x_mm=START_X_MM,
                y_mm=START_Y_MM,
                z_mm=START_Z_MM,
                move_time_s=START_MOVE_TIME_S,
            ),
        ):
            return

        # Rebuild Python estimate from Teensy step counters after the absolute move.
        # This does not physically home; it just keeps the shared state pot consistent.
        robot.sync_estimate_from_teensy_steps()

        detector = build_detector()
        stereo = SimpleStereoCamera()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 520)

        measurements: list[StereoTagMeasurement] = []

        if not wait_for_space(stereo, detector, "Put the EE tag in view for BASE measurement"):
            return
        base = measure_tag(stereo, detector, "base", robot)
        if base is None:
            return
        measurements.append(base)

        if not wait_for_space(stereo, detector, f"Ready to jog +X by {JOG_MM:.1f} mm"):
            return
        if not checked_move("Jog +X", robot.jog(dx=+JOG_MM, move_time_s=MOVE_TIME_S)):
            return
        time.sleep(SETTLE_SEC)
        plus_x = measure_tag(stereo, detector, "plus_x", robot)
        if plus_x is None:
            return
        measurements.append(plus_x)
        print_delta(f"Observed motion for commanded +X {JOG_MM:.1f} mm", base, plus_x)

        if not wait_for_space(stereo, detector, f"Ready to jog back -X by {JOG_MM:.1f} mm"):
            return
        if not checked_move("Jog -X return", robot.jog(dx=-JOG_MM, move_time_s=MOVE_TIME_S)):
            return
        time.sleep(SETTLE_SEC)
        back_from_x = measure_tag(stereo, detector, "back_from_x", robot)
        if back_from_x is None:
            return
        measurements.append(back_from_x)
        print_delta("Return-from-X residual relative to base", base, back_from_x)

        if not wait_for_space(stereo, detector, f"Ready to jog +Y by {JOG_MM:.1f} mm"):
            return
        if not checked_move("Jog +Y", robot.jog(dy=+JOG_MM, move_time_s=MOVE_TIME_S)):
            return
        time.sleep(SETTLE_SEC)
        plus_y = measure_tag(stereo, detector, "plus_y", robot)
        if plus_y is None:
            return
        measurements.append(plus_y)
        print_delta(f"Observed motion for commanded +Y {JOG_MM:.1f} mm", back_from_x, plus_y)

        if not wait_for_space(stereo, detector, f"Ready to jog back -Y by {JOG_MM:.1f} mm"):
            return
        if not checked_move("Jog -Y return", robot.jog(dy=-JOG_MM, move_time_s=MOVE_TIME_S)):
            return
        time.sleep(SETTLE_SEC)
        back_from_y = measure_tag(stereo, detector, "back_from_y", robot)
        if back_from_y is None:
            return
        measurements.append(back_from_y)
        print_delta("Return-from-Y residual relative to post-X-return", back_from_x, back_from_y)

        save_log(measurements)

        print()
        print("Done. The useful lines are the observed +X and +Y image/disparity deltas.")
        print("Those are the local camera Jacobian columns for this robot configuration.")

    finally:
        if stereo is not None:
            try:
                stereo.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
