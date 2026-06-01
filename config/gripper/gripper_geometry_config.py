from __future__ import annotations

"""config/gripper/gripper_geometry_config.py

Shared gripper geometry for demo visualization and clearance audits.

The important footprint model while holding an object is:

    held_length_along_phi = max(object_length_along_phi, L * sin(servo_angle))
    held_width_perp_phi   = max(object_width_perp_phi, fixed_gripper_width)

This lets us conservatively reason about whether the claw + held object will
fit through a pick gap or inside the bag at a given yaw/phi.
"""

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class GripperGeometryConfig:
    # Footprint model used for pick/place clearance.
    FIXED_GRIPPER_WIDTH_MM: float = 60.0
    PIVOT_TO_TIP_LENGTH_MM: float = 70.0
    SERVO_MIN_DEG: float = 0.0
    SERVO_MAX_DEG: float = 70.0
    DEFAULT_SERVO_DEG: float = 55.0

    # Visual box model used in the demo render.
    FINGER_LENGTH_MM: float = 70.0
    FINGER_WIDTH_MM: float = 12.0
    FINGER_DEPTH_MM: float = 35.0
    PALM_WIDTH_MM: float = 40.0
    PALM_HEIGHT_MM: float = 18.0


DEFAULT_GRIPPER_GEOMETRY = GripperGeometryConfig()


def gripper_geometry_with_overrides(
    config: GripperGeometryConfig | None = None,
    **overrides,
) -> GripperGeometryConfig:
    base = DEFAULT_GRIPPER_GEOMETRY if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base


def clamp_servo_angle_deg(
    servo_angle_deg: float | int | None,
    config: GripperGeometryConfig = DEFAULT_GRIPPER_GEOMETRY,
) -> float:
    value = config.DEFAULT_SERVO_DEG if servo_angle_deg is None else float(servo_angle_deg)
    return float(min(config.SERVO_MAX_DEG, max(config.SERVO_MIN_DEG, value)))


def opening_length_mm_for_servo_deg(
    servo_angle_deg: float | int | None,
    config: GripperGeometryConfig = DEFAULT_GRIPPER_GEOMETRY,
) -> float:
    angle_deg = clamp_servo_angle_deg(servo_angle_deg, config)
    return abs(float(config.PIVOT_TO_TIP_LENGTH_MM) * math.sin(math.radians(angle_deg)))


def held_footprint_dims_mm(
    *,
    object_length_along_phi_mm: float,
    object_width_perp_phi_mm: float,
    servo_angle_deg: float | int | None,
    config: GripperGeometryConfig = DEFAULT_GRIPPER_GEOMETRY,
) -> tuple[float, float]:
    grip_length_mm = opening_length_mm_for_servo_deg(servo_angle_deg, config)
    hold_length_mm = max(float(object_length_along_phi_mm), float(grip_length_mm))
    hold_width_mm = max(float(object_width_perp_phi_mm), float(config.FIXED_GRIPPER_WIDTH_MM))
    return hold_length_mm, hold_width_mm
