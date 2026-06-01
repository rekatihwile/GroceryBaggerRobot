from __future__ import annotations

"""config/motion/z_safety_config.py — Robot Z-axis safety envelope.

This is the config/ canonical home for the ZSafetyConfig dataclass.
motion/z_safety_config.py exposes the same interface for backward compat.

Edit this file when the physical robot, gripper, or platform calibration
changes. Every pick/place Z decision ultimately clamps against these values.
"""

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class ZSafetyConfig:
    # ── Travel ceiling ─────────────────────────────────────────────────────
    # The highest robot Z used for all travel/approach/retract moves.
    Z_MAX_MM: float = 270.0

    # ── Pick / grasp floor ─────────────────────────────────────────────────
    # PLATFORM_MIN_GRIPPER_Z_MM is the hard stop below which the gripper tip
    # must never descend during a pick.  MIN_PICK_GRASP_Z_MM defaults to the
    # same value; override to allow slightly lower in unusual setups.
    PLATFORM_MIN_GRIPPER_Z_MM: float = 125.0
    MIN_PICK_GRASP_Z_MM: float = None  # resolved in __post_init__

    # ── Place / release floor ──────────────────────────────────────────────
    # Place release gets its own floor so objects can be released near the
    # calibrated surface (which may be close to 0) without forcing the arm
    # up to the pick floor.
    MIN_PLACE_Z_MM: float | None = 1.5
    # ── Gripper geometry ───────────────────────────────────────────────────
    # Distance from the robot Z coordinate to the empty gripper tip.
    # Used in pick Z planning: grasp_z = object_surface_z + GRIPPER_OFFSET_MM
    GRIPPER_OFFSET_MM: float = 140.0

    # ── Z safety padding ───────────────────────────────────────────────────
    MIN_PLACE_ITEM_HEIGHT_MM: float = 0.0
    PICK_SHORT_ITEM_EXTRA_BUFFER_MM: float = 0.5
    # Crush-prevention margin added above the estimated object surface.
    PLACE_Z_SAFETY_PADDING_MM: float = 15.0

    # Small release gap; the safety padding carries the main crush margin.
    PLACE_RELEASE_GAP_MM: float = 1.0

    # ── BagState query inflate ─────────────────────────────────────────────
    PLACE_STACK_QUERY_INFLATE_CM: float = 3.0

    def __post_init__(self) -> None:
        if self.MIN_PICK_GRASP_Z_MM is None:
            object.__setattr__(self, "MIN_PICK_GRASP_Z_MM", float(self.PLATFORM_MIN_GRIPPER_Z_MM))
        if self.MIN_PLACE_Z_MM is None:
            object.__setattr__(self, "MIN_PLACE_Z_MM", 0.0)


# ── Default instance ───────────────────────────────────────────────────────
DEFAULT_Z_SAFETY = ZSafetyConfig()

# Module-level aliases — import these for one-liner access in scripts.
Z_MAX_MM                    = DEFAULT_Z_SAFETY.Z_MAX_MM
PLATFORM_MIN_GRIPPER_Z_MM   = DEFAULT_Z_SAFETY.PLATFORM_MIN_GRIPPER_Z_MM
MIN_PICK_GRASP_Z_MM         = DEFAULT_Z_SAFETY.MIN_PICK_GRASP_Z_MM
MIN_PLACE_Z_MM              = DEFAULT_Z_SAFETY.MIN_PLACE_Z_MM
GRIPPER_OFFSET_MM           = DEFAULT_Z_SAFETY.GRIPPER_OFFSET_MM
MIN_PLACE_ITEM_HEIGHT_MM    = DEFAULT_Z_SAFETY.MIN_PLACE_ITEM_HEIGHT_MM
PICK_SHORT_ITEM_EXTRA_BUFFER_MM = DEFAULT_Z_SAFETY.PICK_SHORT_ITEM_EXTRA_BUFFER_MM
PLACE_Z_SAFETY_PADDING_MM   = DEFAULT_Z_SAFETY.PLACE_Z_SAFETY_PADDING_MM
PLACE_RELEASE_GAP_MM        = DEFAULT_Z_SAFETY.PLACE_RELEASE_GAP_MM
PLACE_STACK_QUERY_INFLATE_CM = DEFAULT_Z_SAFETY.PLACE_STACK_QUERY_INFLATE_CM


def z_safety_config_with_overrides(
    config: ZSafetyConfig | None = None,
    **overrides,
) -> ZSafetyConfig:
    """Return a ZSafetyConfig with explicit non-None overrides applied."""
    base = DEFAULT_Z_SAFETY if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base


def _z_target_from_label(label: str) -> str:
    return str(label).lower().split()[-1].strip("[]")


def validate_z_command(
    z_mm: float,
    label: str,
    *,
    config: ZSafetyConfig | None = None,
) -> str | None:
    """Return None if z_mm is valid for this command label, else a reason string."""
    cfg = DEFAULT_Z_SAFETY if config is None else config
    try:
        z = float(z_mm)
    except (TypeError, ValueError):
        return f"{label} z={z_mm} is not a number"

    if not math.isfinite(z):
        return f"{label} z={z_mm} is not finite"
    if z < 0.0:
        return f"{label} z={z:.1f} mm < 0"
    if z > cfg.Z_MAX_MM + 1e-6:
        return f"{label} z={z:.1f} mm > Z_MAX_MM={cfg.Z_MAX_MM:.1f}"

    target = _z_target_from_label(label)
    if target in {"grasp", "descent_target"} and z < cfg.MIN_PICK_GRASP_Z_MM:
        return (
            f"{label} z={z:.1f} mm < "
            f"MIN_PICK_GRASP_Z_MM={cfg.MIN_PICK_GRASP_Z_MM:.1f}"
        )
    if target in {"place", "release", "lower"} and z < cfg.MIN_PLACE_Z_MM:
        return f"{label} z={z:.1f} mm < MIN_PLACE_Z_MM={cfg.MIN_PLACE_Z_MM:.1f}"
    return None


def print_z_safety_settings(
    prefix: str = "[Z SAFETY]",
    *,
    config: ZSafetyConfig | None = None,
) -> None:
    """Print the active Z policy settings for startup diagnostics."""
    cfg = DEFAULT_Z_SAFETY if config is None else config
    print(
        f"{prefix} Z_MAX_MM={cfg.Z_MAX_MM:.1f} "
        f"PLATFORM_MIN_GRIPPER_Z_MM={cfg.PLATFORM_MIN_GRIPPER_Z_MM:.1f} "
        f"MIN_PICK_GRASP_Z_MM={cfg.MIN_PICK_GRASP_Z_MM:.1f} "
        f"MIN_PLACE_Z_MM={cfg.MIN_PLACE_Z_MM:.1f} "
        f"MIN_PLACE_ITEM_HEIGHT_MM={cfg.MIN_PLACE_ITEM_HEIGHT_MM:.1f} "
        f"PLACE_Z_SAFETY_PADDING_MM={cfg.PLACE_Z_SAFETY_PADDING_MM:.1f} "
        f"PLACE_RELEASE_GAP_MM={cfg.PLACE_RELEASE_GAP_MM:.1f} "
        f"GRIPPER_OFFSET_MM={cfg.GRIPPER_OFFSET_MM:.1f}"
    )
