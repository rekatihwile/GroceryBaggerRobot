from __future__ import annotations

"""Interactive bag placement target sanity checker.

Use this when the bag corners trace correctly but autonomous placement targets
still feel wrong.  It does not run vision or pick anything.  It computes simple
bag slots plus an estimated gripper footprint, reports whether each target fits
inside the bag rectangle, and can jog the robot at max Z to selected targets.
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion.z_safety_config import DEFAULT_Z_SAFETY, print_z_safety_settings, validate_z_command  # noqa: E402

# Bag zone comes from scripts.pick_one_place_one._load_place_surface_zone().
TRACE_Z_MM = None  # None means use scripts.pick_one_place_one.Z_MAX_MM.

# Approximate object/placement knobs to test without running autonomy.
TEST_OBJECT_WIDTH_MM = 80.0
TEST_OBJECT_DEPTH_MM = 80.0
PAD_X_MM = 10.0
PAD_Y_MM = 10.0
ADJACENT_DIRECTION = "left"  # right/left/up/down

# Approximate claw footprint.  Length is L * sin(servo_angle), width is fixed.
SERVO_ANGLE_DEG = 55.0
GRIPPER_L_MM = 70.0
GRIPPER_WIDTH_MM = 60.0
GRIPPER_MARGIN_MM = 5.0
PLACE_PHI_DEG_OVERRIDE = None  # None means zone default phi.

# Motion.
CONNECT_AND_MOVE_ROBOT = True
RAISE_MOVE_TIME_S = 1.25
XY_MOVE_TIME_S = 1.25

# ============================================================

import math
import traceback

import numpy as np

from motion.pick_validation_motion import startup_robot
from scripts.pick_one_place_one import Z_MAX_MM, _configure_modules, _load_place_surface_zone


def _bag_bounds(zone: dict) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(zone["center_xy_mm"], dtype=np.float64).reshape(2)
    half = np.array([0.5 * float(zone.get("width_mm", 120.0)), 0.5 * float(zone.get("depth_mm", 120.0))], dtype=np.float64)
    return c - half, c + half


def _footprint_size() -> tuple[float, float]:
    length = abs(float(GRIPPER_L_MM) * math.sin(math.radians(float(SERVO_ANGLE_DEG))))
    length += 2.0 * float(GRIPPER_MARGIN_MM)
    width = float(GRIPPER_WIDTH_MM) + 2.0 * float(GRIPPER_MARGIN_MM)
    return max(1.0, length), max(1.0, width)


def _rect_corners(center_xy, phi_deg: float, length_mm: float, width_mm: float) -> np.ndarray:
    c = np.asarray(center_xy, dtype=np.float64).reshape(2)
    phi = math.radians(float(phi_deg))
    u = np.array([math.cos(phi), math.sin(phi)], dtype=np.float64)
    v = np.array([-math.sin(phi), math.cos(phi)], dtype=np.float64)
    return np.vstack([
        c - 0.5 * length_mm * u - 0.5 * width_mm * v,
        c + 0.5 * length_mm * u - 0.5 * width_mm * v,
        c + 0.5 * length_mm * u + 0.5 * width_mm * v,
        c - 0.5 * length_mm * u + 0.5 * width_mm * v,
    ])


def _inside_bounds(corners: np.ndarray, min_xy: np.ndarray, max_xy: np.ndarray) -> bool:
    return bool(
        np.all(corners[:, 0] >= min_xy[0])
        and np.all(corners[:, 0] <= max_xy[0])
        and np.all(corners[:, 1] >= min_xy[1])
        and np.all(corners[:, 1] <= max_xy[1])
    )


def _candidate_targets(zone: dict) -> list[tuple[str, np.ndarray]]:
    center = np.asarray(zone["center_xy_mm"], dtype=np.float64).reshape(2)
    object_size = np.array([float(TEST_OBJECT_WIDTH_MM), float(TEST_OBJECT_DEPTH_MM)], dtype=np.float64)
    padded_size = object_size + 2.0 * np.array([float(PAD_X_MM), float(PAD_Y_MM)], dtype=np.float64)

    d = str(ADJACENT_DIRECTION).strip().lower()
    offset = np.zeros(2, dtype=np.float64)
    if d == "right":
        offset[0] = padded_size[0]
    elif d == "left":
        offset[0] = -padded_size[0]
    elif d == "up":
        offset[1] = padded_size[1]
    elif d == "down":
        offset[1] = -padded_size[1]
    else:
        raise ValueError(f"bad ADJACENT_DIRECTION={ADJACENT_DIRECTION!r}")

    return [
        ("zone_center_object1", center),
        (f"adjacent_{d}_object2", center + offset),
        (f"stack_column1_object3", center),
        (f"stack_column2_object4", center + offset),
    ]


def _print_target_audit(zone: dict, targets: list[tuple[str, np.ndarray]]) -> None:
    min_xy, max_xy = _bag_bounds(zone)
    length, width = _footprint_size()
    phi = float(zone.get("default_phi_deg", 0.0) if PLACE_PHI_DEG_OVERRIDE is None else PLACE_PHI_DEG_OVERRIDE)
    print("[TUNE PLACE] bag bounds:")
    print(f"  x=[{min_xy[0]:.1f},{max_xy[0]:.1f}] y=[{min_xy[1]:.1f},{max_xy[1]:.1f}]")
    print(f"  center={zone['center_xy_mm']} size={zone.get('width_mm')} x {zone.get('depth_mm')} phi={phi:.1f}")
    print("[TUNE PLACE] gripper footprint:")
    print(f"  servo={SERVO_ANGLE_DEG:.1f} length=L*sin(servo)={length:.1f} width={width:.1f}")
    print("[TUNE PLACE] targets:")
    for idx, (name, xy) in enumerate(targets, start=1):
        corners = _rect_corners(xy, phi, length, width)
        ok = _inside_bounds(corners, min_xy, max_xy)
        print(f"  {idx}: {name:24s} xy=({xy[0]:.1f},{xy[1]:.1f}) footprint_inside={ok}")
        for cidx, corner in enumerate(corners, start=1):
            print(f"      fp{cidx}: x={corner[0]:.1f} y={corner[1]:.1f}")


def _move_target(robot, label: str, xy: np.ndarray, z: float, phi: float) -> bool:
    reason = validate_z_command(z, f"[TUNE PLACE] {label}", config=DEFAULT_Z_SAFETY)
    if reason:
        print(f"[TUNE PLACE] REFUSED: {reason}")
        return False
    if hasattr(robot, "check_cartesian_pose_safe"):
        ok, pose_reason = robot.check_cartesian_pose_safe(float(xy[0]), float(xy[1]), float(z))
        if not ok:
            print(f"[TUNE PLACE] REFUSED: pose unsafe: {pose_reason}")
            return False
    print(f"[TUNE PLACE] moving to {label}: x={xy[0]:.1f} y={xy[1]:.1f} z={z:.1f} phi={phi:.1f}")
    return bool(robot.move_cartesian(x_mm=float(xy[0]), y_mm=float(xy[1]), z_mm=float(z), phi_deg=float(phi), move_time_s=float(XY_MOVE_TIME_S)))


def main() -> int:
    _configure_modules()
    print("=" * 64)
    print("BAG PLACE SETTINGS TUNER")
    print("=" * 64)
    print_z_safety_settings("[TUNE PLACE] Z safety", config=DEFAULT_Z_SAFETY)
    zone = _load_place_surface_zone()
    targets = _candidate_targets(zone)
    _print_target_audit(zone, targets)

    trace_z = float(Z_MAX_MM if TRACE_Z_MM is None else TRACE_Z_MM)
    phi = float(zone.get("default_phi_deg", 0.0) if PLACE_PHI_DEG_OVERRIDE is None else PLACE_PHI_DEG_OVERRIDE)
    if not CONNECT_AND_MOVE_ROBOT:
        print("[TUNE PLACE] CONNECT_AND_MOVE_ROBOT=False; audit only.")
        return 0

    robot = startup_robot()
    if robot is None:
        return 1

    try:
        print("[TUNE PLACE] Raising to trace Z before target jogging.")
        if not robot.move_cartesian(z_mm=trace_z, move_time_s=float(RAISE_MOVE_TIME_S)):
            print("[TUNE PLACE] raise failed.")
            return 1
        while True:
            choice = input("[TUNE PLACE] enter target number, a=audit, q=quit: ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                return 0
            if choice in {"a", "audit"}:
                _print_target_audit(zone, targets)
                continue
            try:
                idx = int(choice)
            except ValueError:
                print("[TUNE PLACE] enter 1-4, a, or q.")
                continue
            if idx < 1 or idx > len(targets):
                print("[TUNE PLACE] target out of range.")
                continue
            label, xy = targets[idx - 1]
            _move_target(robot, label, xy, trace_z, phi)
    finally:
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[TUNE PLACE] interrupted by user")
    except Exception:
        traceback.print_exc()
        raise
