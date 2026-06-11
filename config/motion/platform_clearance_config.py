from __future__ import annotations

"""config/motion/platform_clearance_config.py — Platform obstacle-avoidance knobs.

Two features are controlled here:

1. Safe pick→hover travel path
   After grasping an item the robot travels at Z_MAX to the bag hover position.
   If other items on the platform are tall enough that the held item's bottom would
   clip them, the robot detours via a perpendicular bypass waypoint instead of
   taking the direct path.

2. Intercept-and-clear mode (INTERCEPT_MODE = "clear")
   When the gripper's descent toward the target item would clip an adjacent item
   (too close in XY given the gripper width + phi), the robot picks the blocking
   item first, moves it to a temporary free spot on the platform, then picks the
   original target.  Default is "off" (legacy skip behaviour).
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PlatformClearanceConfig:
    # ── Legacy proximity check on pick descent ────────────────────────────────
    # Before descending to pick a target, every other candidate centroid is
    # checked.  If any centroid is within OBSTACLE_SKIP_MM the pick is rejected
    # (gripper would definitely clip on descent); within OBSTACLE_WARN_MM a
    # console warning is printed but the pick proceeds.
    OBSTACLE_AVOIDANCE_ENABLED: bool = True
    OBSTACLE_SKIP_MM: float = 15.0
    OBSTACLE_WARN_MM: float = 50.0

    # ── Safe pick→hover travel path ───────────────────────────────────────────
    # Master switch.  Set False to restore the old straight-line travel.
    TRAVEL_CLEARANCE_ENABLED: bool = True

    # Z margin: an obstacle whose top is within this many mm of the held item's
    # bottom (at Z_MAX) is treated as a potential blocker.
    TRAVEL_CLEARANCE_MARGIN_Z_MM: float = 0.0

    # Half-width of the gripper + held item swept envelope used when testing
    # whether the travel path passes over an obstacle.
    TRAVEL_GRIPPER_HALF_WIDTH_MM: float = 50.0

    # Extra XY margin added to each obstacle's footprint half-extents.
    TRAVEL_CLEARANCE_MARGIN_XY_MM: float = 10.0

    # The bypass waypoint is placed this far outside the platform X boundary.
    TRAVEL_BYPASS_X_MARGIN_MM: float = 50.0

    # ── Intercept-and-clear mode ──────────────────────────────────────────────
    # "off"   — legacy: skip the pick candidate when another item is too close.
    # "clear" — pick the blocking item first, relocate it to a free platform
    #            spot, then pick the original target.
    INTERCEPT_MODE: str = "off"

    # Gripper open half-width (mm) used for the cone-footprint collision check.
    # Should match approximately PLACE_GRIPPER_FOOTPRINT_WIDTH_MM / 2.
    INTERCEPT_GRIPPER_HALF_WIDTH_MM: float = 100.0

    # Gripper open half-depth (mm) — pivot-to-tip length / 2 at max open angle.
    INTERCEPT_GRIPPER_HALF_DEPTH_MM: float = 100.0

    # Extra margin added to each blocker's AABB half-extents during the check.
    INTERCEPT_MARGIN_MM: float = 50.0

    # Minimum clear radius required around the temporary relocation spot.
    INTERCEPT_TEMP_SPOT_CLEAR_RADIUS_MM: float = 80.0

    # Grid step for the temp-spot search.
    INTERCEPT_TEMP_SPOT_GRID_STEP_MM: float = 40.0

    # The arm lowers to this Z offset above the computed platform-surface place Z.
    # Increase if items bounce/tip when placed.
    INTERCEPT_TEMP_PLACE_Z_MARGIN_MM: float = 15.0


DEFAULT_PLATFORM_CLEARANCE = PlatformClearanceConfig()


def platform_clearance_with_overrides(
    config: PlatformClearanceConfig | None = None,
    **overrides,
) -> PlatformClearanceConfig:
    base = DEFAULT_PLATFORM_CLEARANCE if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
