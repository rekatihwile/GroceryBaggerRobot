from __future__ import annotations

"""config/pick/servo_config.py — Gripper servo geometry and dynamic pick policy.

The key insight: the initial servo opening angle is computed from the object's
width so the gripper wraps around the object without slamming shut.

    servo_angle_deg ≈ arcsin( min(object_half_width / L_gripper, 1.0) ) * 2

where L_gripper is GRIPPER_GEOMETRY_L_MM (the arm length from pivot to tip).

Dynamic pick modes
------------------
* DLR (Dynamic Lower Robot-Z): lowers until a current spike signals contact,
  records the empirical Z, then the robot moves to that corrected Z.
* DG  (Dynamic Grip): closes the servo from GRIP_START angle until a current
  spike signals the object is gripped.

Contact is detected when the derivative of the motor current exceeds a
threshold for a rolling window of N steps.

Usage:
    from config.pick.servo_config import DEFAULT_SERVO, ServoConfig
    _SERVO = DEFAULT_SERVO
    _SERVO = ServoConfig(CLAW_OPEN_DEG=70, DEFAULT_OBJECT_RIGIDITY="rigid")
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ServoConfig:
    # ── Claw angles ───────────────────────────────────────────────────────
    CLAW_OPEN_DEG: int = 60
    CLAW_CLOSED_DEG: int = 0

    # ── Dynamic pick master switch ────────────────────────────────────────
    ENABLE_DYNAMIC_PICK: bool = True
    # Use DLR to find empirical contact Z (True), or use feedforward grasp Z (False).
    USE_DYNAMIC_PICK_HEIGHT: bool = False
    # Use arcsin geometry to compute initial servo opening angle (True),
    # or open to CLAW_OPEN_DEG (False).
    USE_DYNAMIC_PICK_GRIP_ANGLE: bool = True
    DYNAMIC_PICK_FALLBACK_TO_FIXED: bool = False

    # ── Gripper geometry ──────────────────────────────────────────────────
    # Arm length from pivot to fingertip (mm).  Used in:
    #   servo_angle_deg = arcsin(min(half_object_width / L, 1.0)) * 2
    # Increase if the robot grips too close to fingertip; decrease if it clips.
    GRIPPER_GEOMETRY_L_MM: float = 70.0
    GRIPPER_SERVO_MIN_DEG: float = 0.0
    GRIPPER_SERVO_MAX_DEG: float = 70.0
    # Fallback opening when object width is unknown (no YOLO axis measurement).
    DYNAMIC_PICK_DEFAULT_SERVO_DEG: float = 55.0
    # Extra margin added to the geometry-computed angle for safety.
    DYNAMIC_SERVO_MARGIN_DEG: float = 10.0
    # How far below the feedforward grasp Z the DLR probe starts.
    DYNAMIC_LOWER_CLEARANCE_MM: float = 20.0

    # ── Contact detection thresholds ─────────────────────────────────────
    # DLR stops when current derivative > DYNAMIC_LOWER_DERIV_THRESH_MA for
    # DYNAMIC_CONTACT_LOOKBACK_COUNT consecutive steps.
    DYNAMIC_LOWER_DERIV_THRESH_MA: float = 30.0
    # DG stops when current derivative > DYNAMIC_GRIP_DERIV_THRESH_MA.
    DYNAMIC_GRIP_DERIV_THRESH_MA: float = 500.0
    DYNAMIC_CONTACT_LOOKBACK_COUNT: int = 3
    DYNAMIC_CONTACT_NONZERO_EPS_MA: float = 1.0
    DYNAMIC_CONTACT_SUM_GRIP_MA: float = 1500.0
    DYNAMIC_CONTACT_SUM_LOWER_MA: float = 50.0

    # ── Object rigidity ───────────────────────────────────────────────────
    # "squishable": after grip contact, close an extra SQUISHABLE_POST_CONTACT_EXTRA_CLOSE_DEG
    # "rigid":      stop exactly at contact
    DEFAULT_OBJECT_RIGIDITY: str = "squishable"
    SQUISHABLE_POST_CONTACT_EXTRA_CLOSE_DEG: float = 5.0

    # ── Debug ─────────────────────────────────────────────────────────────
    DYNAMIC_PICK_TRACE_DEBUG: bool = False
    DYNAMIC_PICK_TRACE_QUERY_POS: bool = False


DEFAULT_SERVO = ServoConfig()


def servo_config_with_overrides(
    config: ServoConfig | None = None,
    **overrides,
) -> ServoConfig:
    base = DEFAULT_SERVO if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
