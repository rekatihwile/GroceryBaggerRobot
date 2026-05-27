from __future__ import annotations

"""Shared real-robot pick/place motion sequences."""

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from motion.pick_z_policy import PickZPlan
from motion.place_z_policy import PlaceZPlan
from motion.z_safety_config import ZSafetyConfig, validate_z_command
from test_calibration_bundle_live_stereo_z_pickplace import move_cartesian_nonnegative_z


MoveFn = Callable[..., bool]
PoseCheckFn = Callable[[object, float, float, float, str], bool]


@dataclass(frozen=True)
class PickSequenceSettings:
    claw_open_deg: float
    claw_closed_deg: float
    coarse_move_time_s: float
    z_move_time_s: float
    xy_move_time_s: float | None = None
    claw_settle_s: float = 0.0


@dataclass(frozen=True)
class PlaceSequenceSettings:
    release_servo_deg: float
    coarse_move_time_s: float
    z_move_time_s: float
    xy_move_time_s: float | None = None
    claw_settle_s: float = 0.0
    use_dynamic_release: bool = False
    dynamic_release_timeout_s: float = 45.0
    dynamic_release_max_open_deg: float | None = None


def _command_servo_angle(robot, angle_deg: float) -> bool:
    if hasattr(robot, "set_servo_fractional"):
        return bool(robot.set_servo_fractional(float(angle_deg)))
    result = robot.servo(int(round(float(angle_deg))))
    return True if result is None else bool(result)


def _default_pose_check(robot, x: float, y: float, z: float, label: str) -> bool:
    if hasattr(robot, "check_cartesian_pose_safe"):
        ok, reason = robot.check_cartesian_pose_safe(float(x), float(y), float(z))
        if not ok:
            print(f"{label} REFUSED: target pose unsafe: {reason}")
            return False
    return True


def _move_checked(
    robot,
    label: str,
    *,
    x_mm=None,
    y_mm=None,
    z_mm=None,
    phi_deg=None,
    move_time_s=None,
    move_fn: MoveFn | None = None,
    check_pose_safe_fn: PoseCheckFn | None = None,
) -> bool:
    x_cur, y_cur, z_cur, phi_cur = robot.fk()
    x = x_cur if x_mm is None else float(x_mm)
    y = y_cur if y_mm is None else float(y_mm)
    z = z_cur if z_mm is None else float(z_mm)
    phi = phi_cur if phi_deg is None else float(phi_deg)
    print(f"{label} command: x={x:.1f} y={y:.1f} z={z:.1f} phi={phi:.1f}")

    checker = _default_pose_check if check_pose_safe_fn is None else check_pose_safe_fn
    if not checker(robot, x, y, z, label):
        return False

    mover = move_cartesian_nonnegative_z if move_fn is None else move_fn
    return bool(
        mover(
            robot,
            label,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            phi_deg=phi_deg,
            move_time_s=move_time_s,
        )
    )


def _validate_plan_zs(
    label_prefix: str,
    targets: tuple[tuple[str, float], ...],
    *,
    config: ZSafetyConfig | None,
) -> bool:
    for label, z in targets:
        reason = validate_z_command(z, f"{label_prefix} {label}", config=config)
        if reason is not None:
            print(f"{label_prefix} ABORT (no motion issued): {reason}")
            return False
    return True


def execute_pick_sequence(
    robot,
    candidate_or_debug,
    *,
    target_xy_mm,
    target_phi_deg: float,
    pick_z_plan: PickZPlan,
    settings: PickSequenceSettings,
    config: ZSafetyConfig | None = None,
    move_fn: MoveFn | None = None,
    check_pose_safe_fn: PoseCheckFn | None = None,
    label_prefix: str = "[PICK]",
) -> bool:
    """Open claw, approach, descend to grasp, close claw, and retract."""
    if not pick_z_plan.valid:
        print(f"{label_prefix} ABORT (no motion issued): invalid pick Z plan {pick_z_plan.warnings}")
        return False

    x, y = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
    phi = float(target_phi_deg)
    approach_z = float(pick_z_plan.approach_z_mm)
    grasp_z = float(pick_z_plan.final_grasp_z_mm)
    retract_z = float(pick_z_plan.retract_z_mm)

    if not _validate_plan_zs(
        label_prefix,
        (("approach", approach_z), ("grasp", grasp_z), ("retract", retract_z)),
        config=config,
    ):
        return False

    checker = _default_pose_check if check_pose_safe_fn is None else check_pose_safe_fn
    if not checker(robot, float(x), float(y), approach_z, f"{label_prefix} approach"):
        return False
    if not checker(robot, float(x), float(y), grasp_z, f"{label_prefix} grasp"):
        return False

    print(f"{label_prefix} opening claw servo={settings.claw_open_deg:.2f}")
    if not _command_servo_angle(robot, settings.claw_open_deg):
        print(f"{label_prefix} ABORT: failed to open claw")
        return False
    if settings.claw_settle_s > 0.0:
        time.sleep(float(settings.claw_settle_s))

    xy_time = settings.coarse_move_time_s if settings.xy_move_time_s is None else settings.xy_move_time_s
    if not _move_checked(
        robot,
        f"{label_prefix} raise",
        z_mm=approach_z,
        move_time_s=settings.coarse_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False
    if not _move_checked(
        robot,
        f"{label_prefix} XY+phi",
        x_mm=float(x),
        y_mm=float(y),
        z_mm=approach_z,
        phi_deg=phi,
        move_time_s=xy_time,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False
    if not _move_checked(
        robot,
        f"{label_prefix} grasp",
        z_mm=grasp_z,
        move_time_s=settings.z_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False

    print(f"{label_prefix} closing claw servo={settings.claw_closed_deg:.2f}")
    if not _command_servo_angle(robot, settings.claw_closed_deg):
        print(f"{label_prefix} ABORT: failed to close claw")
        return False
    if settings.claw_settle_s > 0.0:
        time.sleep(float(settings.claw_settle_s))

    if not _move_checked(
        robot,
        f"{label_prefix} retract",
        z_mm=retract_z,
        move_time_s=settings.coarse_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False

    print(f"{label_prefix} OK - item should be held.")
    return True


def execute_place_sequence(
    robot,
    held_object,
    *,
    target_xy_mm,
    target_phi_deg: float,
    place_z_plan: PlaceZPlan,
    settings: PlaceSequenceSettings,
    config: ZSafetyConfig | None = None,
    move_fn: MoveFn | None = None,
    check_pose_safe_fn: PoseCheckFn | None = None,
    on_start_place_motion: Callable[[], None] | None = None,
    on_after_release: Callable[[], None] | None = None,
    label_prefix: str = "[PLACE]",
) -> bool:
    """Approach, descend to release, open claw, and retract."""
    if not place_z_plan.valid:
        print(f"{label_prefix} ABORT (no motion issued): invalid place Z plan {place_z_plan.warnings}")
        return False

    x, y = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
    phi = float(target_phi_deg)
    approach_z = float(place_z_plan.approach_z_mm)
    place_z = float(place_z_plan.final_release_z_mm)
    retract_z = float(place_z_plan.retract_z_mm)

    if not _validate_plan_zs(
        label_prefix,
        (("approach", approach_z), ("place", place_z), ("retract", retract_z)),
        config=config,
    ):
        return False

    checker = _default_pose_check if check_pose_safe_fn is None else check_pose_safe_fn
    if not checker(robot, float(x), float(y), approach_z, f"{label_prefix} approach"):
        return False
    if not checker(robot, float(x), float(y), place_z, f"{label_prefix} lower"):
        return False

    if on_start_place_motion is not None:
        print(f"{label_prefix} starting on-motion callback")
        try:
            on_start_place_motion()
        except Exception as exc:
            print(f"{label_prefix} WARN: on-motion callback failed: {exc}")

    xy_time = settings.coarse_move_time_s if settings.xy_move_time_s is None else settings.xy_move_time_s
    if not _move_checked(
        robot,
        f"{label_prefix} raise",
        z_mm=approach_z,
        move_time_s=settings.coarse_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False
    if not _move_checked(
        robot,
        f"{label_prefix} XY+phi",
        x_mm=float(x),
        y_mm=float(y),
        z_mm=approach_z,
        phi_deg=phi,
        move_time_s=xy_time,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False
    if not _move_checked(
        robot,
        f"{label_prefix} descend",
        z_mm=place_z,
        move_time_s=settings.z_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False

    release_ok = True
    if settings.use_dynamic_release:
        if hasattr(robot, "dynamic_release"):
            print(
                f"{label_prefix} triggering dynamic release (DR) "
                f"max_open_deg={settings.dynamic_release_max_open_deg}"
            )
            release_result = robot.dynamic_release(
                max_open_angle_deg=settings.dynamic_release_max_open_deg,
                timeout_s=float(settings.dynamic_release_timeout_s),
            )
            print(
                f"{label_prefix} dynamic release result: "
                f"ok={release_result.ok} "
                f"servo_release_empirical_deg={release_result.servo_release_empirical_deg} "
                f"error={release_result.error}"
            )
            release_ok = bool(release_result.ok)
        else:
            print(f"{label_prefix} ABORT: robot.dynamic_release unavailable")
            release_ok = False
    else:
        print(f"{label_prefix} opening claw servo={settings.release_servo_deg:.2f}")
        if not _command_servo_angle(robot, settings.release_servo_deg):
            print(f"{label_prefix} WARN: failed to command release servo angle")

    if not release_ok:
        print(f"{label_prefix} ABORT: release command failed")
        return False

    if on_after_release is not None:
        print(f"{label_prefix} starting after-release callback before retract")
        try:
            on_after_release()
        except Exception as exc:
            print(f"{label_prefix} WARN: after-release callback failed: {exc}")

    if settings.claw_settle_s > 0.0:
        time.sleep(float(settings.claw_settle_s))

    if not _move_checked(
        robot,
        f"{label_prefix} retract",
        z_mm=retract_z,
        move_time_s=settings.coarse_move_time_s,
        move_fn=move_fn,
        check_pose_safe_fn=check_pose_safe_fn,
    ):
        return False

    print(f"{label_prefix} OK - item released and robot retracted.")
    return True
