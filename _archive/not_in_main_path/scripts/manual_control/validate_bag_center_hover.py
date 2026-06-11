from __future__ import annotations

"""Motion-only bag center validator.

Loads the same configured place scene used by the planner, raises/retracts the
robot to a safe travel Z, then hovers over the bag center. No cameras, YOLO,
RAFT, point clouds, pick, or place release logic are touched.
"""

from pathlib import Path
import argparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.motion.z_safety_config import DEFAULT_Z_SAFETY, print_z_safety_settings, validate_z_command
from config.place import DEFAULT_PLACE, load_place_scene
from config.place.place_config import place_config_with_overrides
from config.robot_config import ROBOT_CONFIG
from hardware.robot import Robot


DEFAULT_HOVER_Z_MM = float(DEFAULT_Z_SAFETY.Z_MAX_MM)
DEFAULT_RAISE_TIME_S = 1.25
DEFAULT_XY_TIME_S = 1.50


def _confirm(prompt: str, *, enabled: bool) -> bool:
    if not enabled:
        return True
    return input(f"{prompt}\nType YES to continue: ").strip() == "YES"


def _startup_robot(*, enable_motors: bool, init_drivers: bool) -> Robot:
    robot = Robot(config=ROBOT_CONFIG, connect=True)
    if enable_motors:
        robot.enable(True)
    if init_drivers:
        robot.init_drivers()

    print("\nStartup pose options:")
    print("  h = run HOME now")
    print("  s = sync Python estimate from current Teensy step counters")
    print("  a = assume robot is physically at configured home_pose")
    choice = input("Choose h/s/a: ").strip().lower()
    if choice == "h":
        if not robot.home():
            raise RuntimeError("HOME failed")
    elif choice == "s":
        if not robot.sync_estimate_from_teensy_steps():
            raise RuntimeError("Teensy sync failed")
    elif choice == "a":
        robot.assume_homed()
    else:
        raise RuntimeError(f"unknown startup option {choice!r}")

    robot.print_estimate()
    return robot


def _validate_pose(robot: Robot, *, x_mm: float, y_mm: float, z_mm: float) -> bool:
    reason = validate_z_command(z_mm, "[BAG CENTER] hover", config=DEFAULT_Z_SAFETY)
    if reason:
        print(f"[BAG CENTER] REFUSED: {reason}")
        return False
    if hasattr(robot, "check_cartesian_pose_safe"):
        ok, pose_reason = robot.check_cartesian_pose_safe(float(x_mm), float(y_mm), float(z_mm))
        if not ok:
            print(f"[BAG CENTER] REFUSED: pose unsafe: {pose_reason}")
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retract and hover over configured bag center.")
    parser.add_argument("--scene", default=None, help="Override PLACE_SCENE_NAME from config/surface_zones.json.")
    parser.add_argument("--hover-z", type=float, default=DEFAULT_HOVER_Z_MM, help="Robot Z used for hover/retract.")
    parser.add_argument("--raise-time", type=float, default=DEFAULT_RAISE_TIME_S)
    parser.add_argument("--xy-time", type=float, default=DEFAULT_XY_TIME_S)
    parser.add_argument("--no-enable", action="store_true", help="Do not send EN 1 on startup.")
    parser.add_argument("--no-init-drivers", action="store_true", help="Do not send INITDRIVERS on startup.")
    parser.add_argument("--yes", action="store_true", help="Skip final YES confirmation.")
    args = parser.parse_args(argv)

    place_cfg = place_config_with_overrides(DEFAULT_PLACE, PLACE_SCENE_NAME=args.scene)
    zone = dict(load_place_scene(place_cfg, verbose=True))
    cx, cy = [float(v) for v in zone["center_xy_mm"]]
    phi = float(zone.get("default_phi_deg", 0.0))
    hover_z = float(args.hover_z)

    print("=" * 72)
    print("BAG CENTER HOVER VALIDATOR")
    print("=" * 72)
    print_z_safety_settings("[BAG CENTER] Z safety", config=DEFAULT_Z_SAFETY)
    print("[BAG CENTER] scene:")
    print(f"  name       = {zone.get('name')}")
    print(f"  center_xy  = ({cx:.1f}, {cy:.1f}) mm")
    print(f"  size       = {float(zone.get('width_mm', 0.0)):.1f} x {float(zone.get('depth_mm', 0.0)):.1f} mm")
    print(f"  surface_z  = {float(zone.get('surface_z_mm', 0.0)):.1f} mm")
    print(f"  hover_z    = {hover_z:.1f} mm")
    print(f"  phi        = {phi:.1f} deg")
    print("[BAG CENTER] sequence: raise/retract to hover_z, then move XY/phi to bag center.")

    robot = _startup_robot(
        enable_motors=not bool(args.no_enable),
        init_drivers=not bool(args.no_init_drivers),
    )
    try:
        if not _validate_pose(robot, x_mm=cx, y_mm=cy, z_mm=hover_z):
            return 1
        if not _confirm("[BAG CENTER] Real robot motion will execute.", enabled=not bool(args.yes)):
            print("[BAG CENTER] canceled.")
            return 1

        print(f"[BAG CENTER] 1/2 retract/raise to z={hover_z:.1f}")
        if not robot.move_cartesian(z_mm=hover_z, move_time_s=float(args.raise_time)):
            print("[BAG CENTER] raise failed.")
            return 1

        print(f"[BAG CENTER] 2/2 hover over center x={cx:.1f} y={cy:.1f} z={hover_z:.1f} phi={phi:.1f}")
        if not robot.move_cartesian(x_mm=cx, y_mm=cy, z_mm=hover_z, phi_deg=phi, move_time_s=float(args.xy_time)):
            print("[BAG CENTER] center hover move failed.")
            return 1

        robot.print_estimate()
        print("[BAG CENTER] Hovering over configured bag center. Visually confirm the bag is centered under the gripper.")
        return 0
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
