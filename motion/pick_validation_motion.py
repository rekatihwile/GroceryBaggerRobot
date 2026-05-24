from __future__ import annotations

"""Robot startup and validation-only pick motion helpers."""

import math
import time
from typing import Any

import numpy as np

from hardware.robot import Robot
from scripts.pick_validation_display import _fmt_xy, _hr, print_validation
from test_calibration_bundle_live_stereo_z_pickplace import (
    move_cartesian_nonnegative_z,
    read_command_key,
    require_soft_limits,
)
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_xy_resolver import apply_xy_blend
from vision.pick_z_resolver import validate_z_command


CONNECT_ROBOT: bool = True
ENABLE_MOTORS_ON_START: bool = True
INIT_DRIVERS_ON_START: bool = True

X_SURVEY: float = 100.0
Y_SURVEY: float = 100.0
Z_SURVEY: float = 250.0

Z_MAX_MM: float = 275.0
MIN_VALID_OBJECT_POINTS: int = 300
FINE_TUNE_BLEND_STEP: float = 0.05

RAISE_MOVE_TIME_S: float = 1.00
XY_MOVE_TIME_S: float = 1.50
DESCENT_STEP_MM: float = 5.0
DESCENT_STEP_TIME_S: float = 0.35
FINE_TUNE_XY_MOVE_TIME_S: float = 0.35

CLAW_OPEN_DEG: int = 65
REQUIRE_OVERHEAD_XY_FOR_TEST: bool = False
REFUSE_TEST_IF_TOO_FEW_POINTS: bool = True


def _candidate_phi_or_current(robot: Robot, candidate: Any) -> float:
    if candidate.pick_phi_deg is not None and np.isfinite(float(candidate.pick_phi_deg)):
        return float(candidate.pick_phi_deg)
    _, _, _, phi_fk = robot.fk()
    return float(phi_fk)


def _print_fk(robot: Robot, prefix: str = "[FK]") -> tuple[float, float, float, float]:
    x, y, z, phi = robot.fk()
    print(f"{prefix} x={x:7.1f} y={y:7.1f} z={z:7.1f} phi={phi:6.2f}")
    return x, y, z, phi


def startup_robot() -> Robot | None:
    if not CONNECT_ROBOT:
        print("[ROBOT] CONNECT_ROBOT=False; validation only.")
        return None

    robot = Robot(connect=True)
    require_soft_limits(robot)

    if ENABLE_MOTORS_ON_START:
        robot.enable(True)
    if INIT_DRIVERS_ON_START:
        robot.init_drivers()

    print("\nStartup options:")
    print("  h = run HOME now")
    print("  c = continue from current Teensy step counters, no homing")
    print("  a = assume robot is physically at configured home_pose, no homing")
    choice = input("Choose h/c/a: ").strip().lower()

    if choice == "h":
        if not robot.home():
            raise RuntimeError("HOME failed")
    elif choice == "c":
        if not robot.sync_estimate_from_teensy_steps():
            raise RuntimeError("Teensy sync failed")
        robot.print_estimate()
    elif choice == "a":
        robot.assume_homed()
        robot.print_estimate()
    else:
        raise RuntimeError("Unknown startup choice")

    print("[ROBOT] ready")
    try:
        time.sleep(1)
        robot.move_cartesian(X_SURVEY, Y_SURVEY, Z_SURVEY, 0, move_time_s=2.0)
    except Exception:
        pass

    return robot


def execute_test_descent_no_claw(
    robot: Robot | None,
    dbg: CandidateDebug,
    bundle: dict,
) -> tuple[bool, bool]:
    if robot is None:
        print("[TEST] robot is not connected.")
        return False, False

    c = dbg.candidate
    if REQUIRE_OVERHEAD_XY_FOR_TEST and dbg.overhead_xy_mm is None:
        print("[TEST] refused: overhead XY unavailable and REQUIRE_OVERHEAD_XY_FOR_TEST=True")
        return False, False

    if REFUSE_TEST_IF_TOO_FEW_POINTS and c.valid_point_count < MIN_VALID_OBJECT_POINTS:
        print(f"[TEST] refused: points {c.valid_point_count} < {MIN_VALID_OBJECT_POINTS}")
        return False, False

    x = float(c.target_xy[0])
    y = float(c.target_xy[1])
    z_travel = float(Z_MAX_MM)
    z_hover = float(Z_MAX_MM)
    z_grasp = float(c.grasp_robot_z)
    phi = _candidate_phi_or_current(robot, c)

    for label, z in (("travel", z_travel), ("hover", z_hover), ("grasp", z_grasp)):
        reason = validate_z_command(z, f"[TEST] {label}")
        if reason:
            print(f"[TEST] ABORT: {reason}")
            return False, False

    _hr("TEST DESCENT - NO CLAW CLOSE", "-")
    print(f"[TEST] candidate [{c.index}] {c.yolo.class_name}")
    print(f"[TEST] XY={_fmt_xy(c.target_xy)} source={c.target_xy_source_effective}")
    print(f"[TEST] stereo_xy={_fmt_xy(dbg.stereo_xy_mm)} overhead_xy={_fmt_xy(dbg.overhead_xy_mm)}")
    print(f"[TEST] blend overhead={dbg.blend_weight_overhead:.2f} stereo={1.0 - dbg.blend_weight_overhead:.2f}")
    print(f"[TEST] phi={phi:.2f} deg source={c.pick_phi_source}")
    print(f"[TEST] Z plan: travel={z_travel:.1f}, hover={z_hover:.1f}, grasp={z_grasp:.1f}")
    print("[TEST] Opening claw only for clearance. It will NOT close.")
    try:
        robot.servo(CLAW_OPEN_DEG)
    except Exception:
        pass

    print("[TEST] 1/3 raise to Z_MAX")
    if not move_cartesian_nonnegative_z(robot, "[TEST] raise", z_mm=z_travel, move_time_s=RAISE_MOVE_TIME_S):
        return False, False

    print("[TEST] 2/3 move XY + semiminor phi at Z_MAX")
    if not move_cartesian_nonnegative_z(
        robot,
        "[TEST] XY+phi",
        x_mm=x,
        y_mm=y,
        z_mm=z_travel,
        phi_deg=phi,
        move_time_s=XY_MOVE_TIME_S,
    ):
        return False, False

    print("[TEST] 3/3 slow descent to grasp height")
    total_drop = max(0.0, z_hover - z_grasp)
    n_steps = max(1, int(math.ceil(total_drop / max(DESCENT_STEP_MM, 0.1))))
    for i, z in enumerate(np.linspace(z_hover, z_grasp, n_steps + 1)[1:], start=1):
        print(f"  descent {i:02d}/{n_steps}: z={float(z):.1f}")
        if not move_cartesian_nonnegative_z(robot, "[TEST] descent", z_mm=float(z), move_time_s=DESCENT_STEP_TIME_S):
            return False, False

    print("[TEST] reached grasp height. No claw close commanded.")
    request_resurvey = fine_tune_xy_blend_loop(robot, dbg, bundle, z_grasp, phi)
    return True, request_resurvey


def fine_tune_xy_blend_loop(
    robot: Robot,
    dbg: CandidateDebug,
    bundle: dict,
    z_current: float,
    phi_current: float,
) -> bool:
    _hr("FINE TUNE XY BLEND AT GRASP HEIGHT", "-")
    print("Controls:")
    print("  a = more stereo XY  (decrease overhead weight)")
    print("  d = more overhead XY (increase overhead weight)")
    print("  s = re-survey now (exit fine tune and run survey)")
    print("  v = print selected geometry")
    print("  q/ESC = exit fine tune")
    print("Each a/d move commands the same Z and phi, only changing XY by the blend ratio.")
    print()

    while True:
        c = dbg.candidate
        print(
            f"[FINE] w_overhead={dbg.blend_weight_overhead:.2f} "
            f"cmd_xy={_fmt_xy(c.target_xy)} "
            f"stereo={_fmt_xy(dbg.stereo_xy_mm)} overhead={_fmt_xy(dbg.overhead_xy_mm)}"
        )
        key = read_command_key(delay_ms=0)

        if key is None:
            key = input("[FINE] key a/d/s/v/q: ").strip().lower()[:1] or None

        if key == "s":
            print("[FINE] re-survey requested")
            return True

        if key in ("q", "escape", "\x1b"):
            print("[FINE] exit")
            return False

        if key == "v":
            tmp_state = SurveyState([], [], [], [dbg], None, [], 0)
            print_validation(tmp_state, 0)
            continue

        if key not in ("a", "d"):
            continue

        old_w = dbg.blend_weight_overhead
        if key == "a":
            new_w = max(0.0, old_w - FINE_TUNE_BLEND_STEP)
        else:
            new_w = min(1.0, old_w + FINE_TUNE_BLEND_STEP)

        apply_xy_blend(dbg, bundle, w_overhead=new_w)
        x = float(c.target_xy[0])
        y = float(c.target_xy[1])

        print(
            f"[FINE] moving XY at z={z_current:.1f}, phi={phi_current:.1f}: "
            f"w_overhead {old_w:.2f} -> {new_w:.2f}, xy={_fmt_xy(c.target_xy)}"
        )
        ok = move_cartesian_nonnegative_z(
            robot,
            "[FINE] XY blend adjust",
            x_mm=x,
            y_mm=y,
            z_mm=float(z_current),
            phi_deg=float(phi_current),
            move_time_s=FINE_TUNE_XY_MOVE_TIME_S,
        )
        if not ok:
            print("[FINE] move failed; staying in fine tune loop.")


print_fk = _print_fk
