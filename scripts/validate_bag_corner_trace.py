from __future__ import annotations

"""Raise to max Z and trace the configured bag/place-zone corners.

This is a motion-only validation helper.  It does not open cameras, run YOLO,
or lower into the bag.  It loads the same place/surface zone used by the
pick/place scripts and traces the rectangle at Z_MAX_MM so you can verify the
bag XY math safely.
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

# Use the same zone name/config paths as scripts/pick_one_place_one.py.
BAG_ZONE_NAME = "New Bag Test"

# Trace only at safe travel height.  None means use scripts.pick_one_place_one.Z_MAX_MM.
TRACE_Z_MM = None

# Move to center first, then wait before tracing corners.
TRACE_CENTER_FIRST = True
WAIT_AT_CENTER_BEFORE_CORNERS = True
WAIT_BEFORE_EACH_CORNER = True

# Corner order.  The path closes itself by returning to the first corner.
TRACE_ORDER = "clockwise"  # "clockwise" or "counterclockwise"

# Motion timing.
RAISE_MOVE_TIME_S = 1.25
XY_MOVE_TIME_S = 1.50
DWELL_AT_CORNER_S = 0.20
TRACE_REPEATS = 1

# Confirmation before real motion.
REQUIRE_CONFIRM_BEFORE_TRACE = True

# ============================================================

import time
import traceback

import numpy as np

from motion.pick_validation_motion import startup_robot
from scripts.pick_one_place_one import (
    Z_MAX_MM,
    _configure_modules,
    _load_place_surface_zone,
)


def _confirm(prompt: str, enabled: bool) -> bool:
    if not enabled:
        return True
    answer = input(f"{prompt}\nType YES to continue: ").strip()
    return answer == "YES"


def _wait_for_enter(prompt: str) -> None:
    input(f"{prompt}\nPress Enter to continue...")


def _corners_for_zone(zone: dict) -> list[tuple[str, float, float]]:
    cx, cy = np.asarray(zone["center_xy_mm"], dtype=np.float64).reshape(2)
    half_w = 0.5 * float(zone.get("width_mm", 120.0))
    half_d = 0.5 * float(zone.get("depth_mm", 120.0))

    corners = [
        ("front_left", cx - half_w, cy - half_d),
        ("front_right", cx + half_w, cy - half_d),
        ("back_right", cx + half_w, cy + half_d),
        ("back_left", cx - half_w, cy + half_d),
    ]
    if str(TRACE_ORDER).strip().lower() == "counterclockwise":
        corners = [corners[0], corners[3], corners[2], corners[1]]
    return [(name, float(x), float(y)) for name, x, y in corners]


def _validate_trace_targets(robot, targets: list[tuple[str, float, float, float, float]]) -> bool:
    ok_all = True
    for label, x, y, z, phi in targets:
        reason = validate_z_command(z, f"[BAG TRACE] {label}", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[BAG TRACE] REFUSED {label}: {reason}")
            ok_all = False
            continue
        if hasattr(robot, "check_cartesian_pose_safe"):
            ok, pose_reason = robot.check_cartesian_pose_safe(float(x), float(y), float(z))
            if not ok:
                print(f"[BAG TRACE] REFUSED {label}: pose unsafe: {pose_reason}")
                ok_all = False
    return ok_all


def _move_or_abort(robot, label: str, *, x: float | None = None, y: float | None = None, z: float | None = None, phi: float | None = None, move_time_s: float) -> bool:
    print(
        f"[BAG TRACE] MOVE {label}: "
        f"x={'keep' if x is None else f'{x:.1f}'} "
        f"y={'keep' if y is None else f'{y:.1f}'} "
        f"z={'keep' if z is None else f'{z:.1f}'} "
        f"phi={'keep' if phi is None else f'{phi:.1f}'}"
    )
    return bool(robot.move_cartesian(x_mm=x, y_mm=y, z_mm=z, phi_deg=phi, move_time_s=float(move_time_s)))


def main() -> int:
    _configure_modules()
    print("=" * 64)
    print("BAG CORNER TRACE VALIDATION")
    print("=" * 64)
    print("[BAG TRACE] Motion-only check: raises to max Z and traces bag corners.")
    print("[BAG TRACE] It will not lower into the bag.")
    print_z_safety_settings("[BAG TRACE] Z safety", config=DEFAULT_Z_SAFETY)

    zone = _load_place_surface_zone()
    if str(zone.get("name", "")) != BAG_ZONE_NAME:
        print(f"[BAG TRACE] WARN: loaded zone name is {zone.get('name')!r}, requested label is {BAG_ZONE_NAME!r}.")

    trace_z = float(Z_MAX_MM if TRACE_Z_MM is None else TRACE_Z_MM)
    phi = float(zone.get("default_phi_deg", 0.0))
    cx, cy = np.asarray(zone["center_xy_mm"], dtype=np.float64).reshape(2)
    width = float(zone.get("width_mm", 120.0))
    depth = float(zone.get("depth_mm", 120.0))
    corners = _corners_for_zone(zone)

    print("[BAG TRACE] zone:")
    print(f"  name       = {zone.get('name')}")
    print(f"  source     = {zone.get('source')}")
    print(f"  center_xy  = ({cx:.1f}, {cy:.1f}) mm")
    print(f"  width/depth= {width:.1f} x {depth:.1f} mm")
    print(f"  surface_z  = {float(zone.get('surface_z_mm', 0.0)):.1f} mm")
    print(f"  trace_z    = {trace_z:.1f} mm")
    print(f"  phi        = {phi:.1f} deg")
    print("[BAG TRACE] corners:")
    for name, x, y in corners:
        print(f"  {name:12s} x={x:8.1f} y={y:8.1f} z={trace_z:8.1f}")

    targets: list[tuple[str, float, float, float, float]] = []
    if TRACE_CENTER_FIRST:
        targets.append(("center", float(cx), float(cy), trace_z, phi))
    for repeat_i in range(max(1, int(TRACE_REPEATS))):
        for name, x, y in corners + [corners[0]]:
            suffix = f"_{repeat_i + 1}" if TRACE_REPEATS > 1 else ""
            targets.append((f"{name}{suffix}", x, y, trace_z, phi))

    robot = startup_robot()
    if robot is None:
        print("[BAG TRACE] robot not connected.")
        return 1

    try:
        if not _validate_trace_targets(robot, targets):
            print("[BAG TRACE] One or more targets failed validation. No trace motion issued.")
            return 1

        print("[BAG TRACE] WARNING: real robot motion will execute at max Z.")
        if not _confirm("[BAG TRACE] Confirm corner trace motion.", REQUIRE_CONFIRM_BEFORE_TRACE):
            print("[BAG TRACE] canceled by user.")
            return 1

        if not _move_or_abort(robot, "raise_to_trace_z", z=trace_z, move_time_s=RAISE_MOVE_TIME_S):
            print("[BAG TRACE] raise failed.")
            return 1

        for idx, (label, x, y, z, target_phi) in enumerate(targets):
            if idx == 1 and WAIT_AT_CENTER_BEFORE_CORNERS:
                _wait_for_enter("[BAG TRACE] At bag center. Ready to trace the first corner.")
            elif idx > 1 and WAIT_BEFORE_EACH_CORNER:
                _wait_for_enter(f"[BAG TRACE] Ready to move to {label}.")

            if not _move_or_abort(robot, label, x=x, y=y, z=z, phi=target_phi, move_time_s=XY_MOVE_TIME_S):
                print(f"[BAG TRACE] move failed at {label}.")
                return 1
            if DWELL_AT_CORNER_S > 0.0:
                time.sleep(float(DWELL_AT_CORNER_S))

        print("[BAG TRACE] complete. Rectangle traced at safe Z.")
        return 0
    finally:
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[BAG TRACE] interrupted by user")
    except Exception:
        traceback.print_exc()
        raise
