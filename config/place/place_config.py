from __future__ import annotations

"""config/place/place_config.py — Placement zone, release, and timing configuration.

Covers:
  • Zone selection (which bag zone to target)
  • Place release height policy
  • Gripper footprint constraints
  • Packing geometry (AABB padding, adjacent direction)
  • Motion timing

Usage:
    from config.place.place_config import DEFAULT_PLACE, PlaceConfig
    _PLACE = DEFAULT_PLACE
    _PLACE = PlaceConfig(PLACE_SURFACE_ZONE_NAME="Left Bag", PAD_Z_MM=30.0)
"""

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class PlaceConfig:
    # ── Zone selection ────────────────────────────────────────────────────
    PLACE_SURFACE_ZONE_NAME: str = "New Bag Test"
    SURFACE_ZONE_CONFIG_PATH: Path = Path("config/surface_zones.json")
    # Legacy place_zones.json fallback (used if surface_zones.json lookup fails).
    PLACE_ZONE_NAME: str = "New Bag Test"
    PLACE_ZONE_CONFIG_PATH: Path = Path("config/place_zones.json")

    # ── Release Z policy ──────────────────────────────────────────────────
    # "shared"              — use shared place_z_policy only
    # "negative_bin_hang"   — estimate hang below gripper, weighted blend
    # "negative_bin_simple" — object height above bin floor
    PLACE_Z_POLICY_MODE: str = "negative_bin_hang"
    # Bin floor when using negative_bin_* modes (bag bottom in robot Z coords).
    PLACE_NEGATIVE_BIN_PLATFORM_Z_MM: float = -200.0
    PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM: float = 0.0
    PLACE_NEGATIVE_BIN_HANG_WEIGHT: float = 0.70
    PLACE_NEGATIVE_BIN_SIMPLE_WEIGHT: float = 0.30
    PLACE_NEGATIVE_BIN_CLEARANCE_MM: float = 0.0
    PLACE_NEGATIVE_BIN_USE_EXISTING_STACK: bool = True
    PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING: bool = False

    # ── Release Z uncertainty ─────────────────────────────────────────────
    PLACE_RELEASE_GAP_MM: float = 1.0
    PLACE_Z_UNCERTAINTY_GAIN: float = 0.30
    PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM: float = 10.0
    USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT: bool = False

    # ── Release servo angle ───────────────────────────────────────────────
    # "empirical_plus_offset" — use servo_empirical_deg + EMPIRICAL_OFFSET_DEG
    # "initial_minus_padding" — use initial_servo_angle_deg − INITIAL_PADDING_DEG
    PLACE_RELEASE_ANGLE_MODE: str = "empirical_plus_offset"
    PLACE_RELEASE_EMPIRICAL_OFFSET_DEG: float = 5.0
    PLACE_RELEASE_INITIAL_PADDING_DEG: float = 2.5
    # Fixed fallback open angle when no empirical angle is recorded.
    PLACE_CLAW_OPEN_DEG: int = 45

    # ── Dynamic release ───────────────────────────────────────────────────
    USE_DYNAMIC_RELEASE_FOR_PLACE: bool = False
    DYNAMIC_RELEASE_TIMEOUT_S: float = 45.0

    # ── Orientation / phi ─────────────────────────────────────────────────
    USE_PICK_PHI_FOR_PLACE: bool = False
    # Rotation offsets (deg) tried when optimising gripper footprint clearance.
    PLACE_ROTATION_CANDIDATE_OFFSETS_DEG: tuple[float, ...] = (0.0, 90.0)
    PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE: bool = True

    # ── Gripper footprint inside bag ──────────────────────────────────────
    # Aborts placement if the estimated claw footprint would leave the bag.
    PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG: bool = True
    PLACE_GRIPPER_FOOTPRINT_WIDTH_MM: float = 60.0
    # The opening dimension is L * sin(servo_angle):
    #   footprint_length = GRIPPER_FOOTPRINT_LENGTH_L_MM * sin(servo_deg)
    PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM: float = 70.0
    PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM: float = 1.0
    PLACE_GRIPPER_FOOTPRINT_DEFAULT_SERVO_DEG: float = 55.0

    # ── AABB padding (placed occupancy) ───────────────────────────────────
    # Inflate each placed object's AABB by this much on each side.
    # Increase X/Y when objects collide; increase Z to reserve stack clearance.
    PAD_X_MM: float = 0.0
    PAD_Y_MM: float = 0.0
    PAD_Z_MM: float = 20.0

    # ── Packing strategy ──────────────────────────────────────────────────
    # Direction for the 2nd object relative to the 1st: left/right/up/down.
    ADJACENT_DIRECTION: str = "left"
    # Efficient packing scores slot fit first, then picks by volume.
    EFFICIENT_PACKING_ENABLED: bool = True
    EFFICIENT_PACKING_REQUIRE_SLOT_FIT: bool = True

    # ── XY nudge ─────────────────────────────────────────────────────────
    # Small XY offsets tried to improve slot fit when the nominal target fails.
    PLACE_XY_NUDGE_ENABLED: bool = True
    PLACE_XY_NUDGE_STEP_MM: float = 5.0
    PLACE_XY_NUDGE_MAX_MM: float = 20.0

    # ── Motion timing ─────────────────────────────────────────────────────
    COARSE_MOVE_TIME_S: float = 1.10
    XY_MOVE_TIME_S: float = 1.50
    PICK_Z_MOVE_TIME_S: float = 0.60
    PLACE_Z_MOVE_TIME_S: float = 0.60

    # ── Robot connection ──────────────────────────────────────────────────
    CONNECT_ROBOT: bool = True
    ENABLE_MOTORS_ON_START: bool = True
    INIT_DRIVERS_ON_START: bool = True


DEFAULT_PLACE = PlaceConfig()


def place_config_with_overrides(
    config: PlaceConfig | None = None,
    **overrides,
) -> PlaceConfig:
    base = DEFAULT_PLACE if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
