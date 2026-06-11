from __future__ import annotations

"""Small FK <-> Teensy dynamic-lower frame validation tool."""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

TEST_X_MM = 100.0
TEST_Y_MM = 100.0
TEST_Z_MM = 140.0
TEST_PHI_DEG = 0.0
TEST_MOVE_TIME_S = 1.0

TEST_DLR_SERVO_DEG = 55.0
TEST_DLR_DERIV_THRESH_MA = 30.0
TEST_DLR_N_STEPS = 1
TEST_DLR_SIGNED_ONLY = False

CONNECT_ROBOT = True
ENABLE_MOTORS_ON_START = True
INIT_DRIVERS_ON_START = True
PRINT_POS_TRACE = True

# ============================================================

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.robot import Robot
from motion.pick_validation_motion import startup_robot
from test_calibration_bundle_live_stereo_z_pickplace import move_cartesian_nonnegative_z


def _print_trace(robot: Robot, target_z_mm: float) -> None:
    print("[TRACE]")
    x, y, z, phi = robot.fk()
    print(f"  fk_xyzphi = ({x:.2f}, {y:.2f}, {z:.2f}, {phi:.2f})")
    print(f"  q_est.z_mm = {float(robot.q_est.z_mm):.2f}")
    trace = robot.z_command_trace(float(target_z_mm), include_actual_pos=PRINT_POS_TRACE)
    print(f"  target_robot_z_mm = {trace.target_robot_z_mm:.2f}")
    print(f"  current_est_j3_steps = {trace.current_est_j3_steps}")
    print(f"  current_est_j3_mm = {trace.current_est_j3_mm:.2f}")
    print(f"  current_est_direct_j3_mm = {trace.current_est_direct_j3_mm:.2f}")
    print(f"  target_est_j3_steps = {trace.target_est_j3_steps}")
    print(f"  target_est_j3_mm = {trace.target_est_j3_mm:.2f}")
    print(f"  target_direct_j3_mm = {trace.target_direct_j3_mm:.2f}")
    if trace.actual_steps is not None:
        print(f"  actual_steps = {trace.actual_steps}")
    if trace.actual_j3_mm is not None:
        print(f"  actual_j3_mm = {trace.actual_j3_mm:.2f}")
    if trace.actual_direct_j3_mm is not None:
        print(f"  actual_direct_j3_mm = {trace.actual_direct_j3_mm:.2f}")
    if trace.inferred_robot_to_direct_offset_mm is not None:
        print(f"  inferred_robot_to_direct_offset_mm = {trace.inferred_robot_to_direct_offset_mm:.2f}")
    if trace.warnings:
        print(f"  warnings = {trace.warnings}")


def _print_help() -> None:
    print()
    print("validate_fk_teensy_dynamiclower_frame.py")
    print("=========================================")
    print("Controls:")
    print("  p = print FK + Z trace to TEST_Z_MM")
    print("  m = move to TEST_X/Y/Z/PHI with normal Cartesian path")
    print("  j = send J3 = TEST_Z_MM directly")
    print("  d = run DLR at TEST_Z_MM")
    print("  l = run legacy DL at TEST_Z_MM")
    print("  s = SHOW dynamic state")
    print("  q = quit")
    print()
    print(f"TEST target: x={TEST_X_MM:.1f} y={TEST_Y_MM:.1f} z={TEST_Z_MM:.1f} phi={TEST_PHI_DEG:.1f}")
    print(
        f"DLR args: servo={TEST_DLR_SERVO_DEG:.1f} deriv_thresh={TEST_DLR_DERIV_THRESH_MA:.1f} "
        f"N={TEST_DLR_N_STEPS} signed={int(bool(TEST_DLR_SIGNED_ONLY))}"
    )
    print()


def main() -> int:
    if not CONNECT_ROBOT:
        print("CONNECT_ROBOT=False; nothing to do.")
        return 0

    import motion.pick_validation_motion as _motion_mod

    _motion_mod.CONNECT_ROBOT = CONNECT_ROBOT
    _motion_mod.ENABLE_MOTORS_ON_START = ENABLE_MOTORS_ON_START
    _motion_mod.INIT_DRIVERS_ON_START = INIT_DRIVERS_ON_START
    _motion_mod.X_SURVEY = TEST_X_MM
    _motion_mod.Y_SURVEY = TEST_Y_MM
    _motion_mod.Z_SURVEY = max(TEST_Z_MM, 100.0)

    robot = startup_robot()
    if robot is None:
        print("Robot unavailable.")
        return 1

    try:
        _print_help()
        while True:
            key = input("[TRACE] key p/m/j/d/l/s/q: ").strip().lower()[:1]
            if key == "q":
                break
            if key == "p":
                _print_trace(robot, TEST_Z_MM)
                continue
            if key == "m":
                _print_trace(robot, TEST_Z_MM)
                ok = move_cartesian_nonnegative_z(
                    robot,
                    "[TRACE] move target",
                    x_mm=TEST_X_MM,
                    y_mm=TEST_Y_MM,
                    z_mm=TEST_Z_MM,
                    phi_deg=TEST_PHI_DEG,
                    move_time_s=TEST_MOVE_TIME_S,
                )
                print(f"[TRACE] move result: {ok}")
                _print_trace(robot, TEST_Z_MM)
                continue
            if key == "j":
                _print_trace(robot, TEST_Z_MM)
                robot.send(f"J3 = {float(TEST_Z_MM):.3f}")
                ok = robot.read_until("DONE", 30.0, match="exact")
                print(f"[TRACE] direct J3 result: {ok}")
                if ok:
                    robot.q_est.z_mm = float(TEST_Z_MM)
                _print_trace(robot, TEST_Z_MM)
                continue
            if key == "d":
                _print_trace(robot, TEST_Z_MM)
                result = robot.dynamic_lower_robot_z(
                    robot_z_mm=TEST_Z_MM,
                    deriv_thresh_ma=TEST_DLR_DERIV_THRESH_MA,
                    n_steps=TEST_DLR_N_STEPS,
                    servo_deg=TEST_DLR_SERVO_DEG,
                    signed_only=TEST_DLR_SIGNED_ONLY,
                )
                print(
                    "[TRACE] DLR result: "
                    f"ok={result.ok} z_empirical_mm={result.z_empirical_mm} "
                    f"servo_empirical_deg={result.servo_empirical_deg} error={result.error}"
                )
                robot.sync_estimate_from_teensy_steps()
                _print_trace(robot, TEST_Z_MM)
                continue
            if key == "l":
                _print_trace(robot, TEST_Z_MM)
                result = robot.dynamic_lower(
                    z_start_mm=TEST_Z_MM,
                    deriv_thresh_ma=TEST_DLR_DERIV_THRESH_MA,
                    n_steps=TEST_DLR_N_STEPS,
                    servo_deg=TEST_DLR_SERVO_DEG,
                    signed_only=TEST_DLR_SIGNED_ONLY,
                )
                print(
                    "[TRACE] legacy DL result: "
                    f"ok={result.ok} z_empirical_mm={result.z_empirical_mm} "
                    f"servo_empirical_deg={result.servo_empirical_deg} error={result.error}"
                )
                robot.sync_estimate_from_teensy_steps()
                _print_trace(robot, TEST_Z_MM)
                continue
            if key == "s":
                result = robot.show_dynamic_state()
                print(
                    "[TRACE] SHOW: "
                    f"ok={result.ok} z_empirical_mm={result.z_empirical_mm} "
                    f"servo_empirical_deg={result.servo_empirical_deg} error={result.error}"
                )
                continue
    finally:
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
