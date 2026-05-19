# test_robot.py
# Small sanity test for robot.py.
# Prefer manual_cartesian_jog.py for normal use.

import time

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
from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX

print_startup_config("test_robot.py", OVERHEAD_INDEX, STEREO_INDEX)
require_soft_limits_configured("test_robot.py")

robot = Robot(ROBOT_CONFIG, connect=True)

try:
    require_robot_soft_limits_loaded(robot, "test_robot.py")
    robot.enable(True)
    robot.init_drivers()

    robot.print_estimate()

    choice = input("Run HOME first? [y/n]: ").strip().lower()
    if choice == "y":
        if not robot.home():
            print("HOME failed. Stopping.")
            raise SystemExit
    else:
        print("Continuing from Teensy step counters without physical homing...")
        if not robot.sync_estimate_from_teensy_steps():
            print("Could not sync from POS. Stopping.")
            raise SystemExit

    robot.print_estimate()

    input("Press Enter to move to x=200, y=200...")
    if not robot.move_cartesian(x_mm=200.0, y_mm=200.0, move_time_s=1.0):
        print("Move to 200,200 failed. Stopping.")
        raise SystemExit

    robot.sync_estimate_from_teensy_steps()
    robot.print_estimate()

    # Small jogs only. Use manual_cartesian_jog.py for more testing.
    for label, kwargs in [
        ("+X 10 mm", dict(dx=+10.0)),
        ("-X 10 mm", dict(dx=-10.0)),
        ("+Y 10 mm", dict(dy=+10.0)),
        ("-Y 10 mm", dict(dy=-10.0)),
    ]:
        input(f"Press Enter to jog {label}...")
        if not robot.jog(**kwargs, move_time_s=0.65):
            print(f"Jog {label} failed. Stopping.")
            break
        time.sleep(0.2)
        robot.print_estimate()

finally:
    robot.close()
