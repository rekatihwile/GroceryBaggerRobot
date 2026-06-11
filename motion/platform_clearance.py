from __future__ import annotations

"""motion/platform_clearance.py — Safe pick→hover travel-path computation.

After the robot picks an item and retracts to Z_MAX, it sweeps XY to the bag
hover position.  If remaining platform candidates are tall enough that the held
item's bottom would clip them during this sweep, a perpendicular bypass waypoint
is inserted before the bag hover move.

Geometry
--------
At arm Z = Z_MAX, gripper tip is at Z_MAX - GRIPPER_OFFSET_MM.  The held item
occupies roughly [Z_MAX - GRIPPER_OFFSET - held_height, Z_MAX - GRIPPER_OFFSET]
in robot Z.  Any remaining platform candidate whose top_z exceeds
``held_bottom_z - clearance_margin_z`` is a *Z-blocking obstacle*.

For each such obstacle, we test whether the straight-line travel path passes
within ``obstacle.half_extents_xy + GRIPPER_HALF_WIDTH + clearance_margin_xy``
of the obstacle's XY centroid.  Obstacles that satisfy both criteria are
*path blockers*.

Bypass routing
--------------
Two alternatives are tried:
  • Route via X_min side: waypoint at (platform_x_min - bypass_x_margin, pick_y)
  • Route via X_max side: waypoint at (platform_x_max + bypass_x_margin, pick_y)

Each leg of both routes is checked for blockers.  The route with fewer
remaining blockers is chosen.  If neither fully clears (e.g. the robot arm
cannot physically avoid the obstacle), a warning is logged and the robot
proceeds with the best available option.
"""

import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObstacleBox:
    """Minimal platform-candidate representation for path-clearance geometry."""
    label: str
    center_xy_mm: np.ndarray   # shape (2,), robot-frame XY centroid
    top_z_mm: float            # robot-frame Z of the top of the item
    half_extents_xy_mm: np.ndarray  # shape (2,), AABB half-sizes in X and Y


@dataclass(frozen=True)
class ClearancePlan:
    """Result of compute_pick_to_hover_path()."""
    waypoints_xy_mm: tuple[np.ndarray, ...]
    """Intermediate XY waypoints to execute at Z_MAX before the bag hover move.
    Empty tuple means the direct path is clear."""

    held_bottom_z_mm: float
    """Computed Z of the held item's bottom at Z_MAX (robot frame)."""

    blocking_labels: tuple[str, ...]
    """Labels of obstacles that intersected the direct path."""

    bypass_side: str
    """'none' | 'x_min' | 'x_max' — which bypass was selected."""

    warnings: tuple[str, ...]
    """Non-fatal warnings (e.g. bypass still clips something)."""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _point_to_segment_dist_mm(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Minimum 2D distance from point p to line segment a→b."""
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _path_intersects_obstacle(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    obs: ObstacleBox,
    gripper_half_width_mm: float,
    clearance_margin_xy_mm: float,
) -> bool:
    """True if path segment a→b passes within keep-out radius of obs."""
    keep_out = float(np.max(obs.half_extents_xy_mm)) + gripper_half_width_mm + clearance_margin_xy_mm
    return _point_to_segment_dist_mm(obs.center_xy_mm, a_xy, b_xy) < keep_out


def _blockers_on_path(
    a_xy: np.ndarray,
    b_xy: np.ndarray,
    obstacles: list[ObstacleBox],
    gripper_half_width_mm: float,
    clearance_margin_xy_mm: float,
) -> list[ObstacleBox]:
    return [
        obs for obs in obstacles
        if _path_intersects_obstacle(a_xy, b_xy, obs, gripper_half_width_mm, clearance_margin_xy_mm)
    ]


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

def compute_pick_to_hover_path(
    pick_xy_mm: np.ndarray,
    bag_hover_xy_mm: np.ndarray,
    held_height_mm: float,
    obstacles: list[ObstacleBox],
    *,
    z_max_mm: float,
    gripper_offset_mm: float,
    platform_x_range_mm: tuple[float, float],
    gripper_half_width_mm: float = 35.0,
    clearance_margin_xy_mm: float = 20.0,
    clearance_margin_z_mm: float = 20.0,
    bypass_x_margin_mm: float = 50.0,
) -> ClearancePlan:
    """Compute a safe pick→bag-hover travel path at Z_MAX.

    Parameters
    ----------
    pick_xy_mm:
        Robot XY of the just-picked item (starting point of the travel).
    bag_hover_xy_mm:
        Robot XY of the bag hover position (destination).
    held_height_mm:
        Height of the held item (robot-frame Z size).
    obstacles:
        ObstacleBox list for remaining platform candidates.
    z_max_mm:
        Travel ceiling (Z_MAX_MM).
    gripper_offset_mm:
        Distance from robot arm Z to gripper tip (GRIPPER_OFFSET_MM).
    platform_x_range_mm:
        (x_min, x_max) of the pick platform, used to place bypass waypoints.
    gripper_half_width_mm:
        Half-width of the gripper + held-item envelope for path-sweep tests.
    clearance_margin_xy_mm:
        Extra XY keep-out margin beyond the obstacle's own footprint.
    clearance_margin_z_mm:
        Z tolerance: obstacles whose top is this far below held_bottom are safe.
    bypass_x_margin_mm:
        Lateral offset beyond platform X bounds for bypass waypoints.
    """
    pick_xy = np.asarray(pick_xy_mm, dtype=np.float64).reshape(2)
    bag_xy = np.asarray(bag_hover_xy_mm, dtype=np.float64).reshape(2)

    held_bottom_z = float(z_max_mm) - float(gripper_offset_mm) - float(held_height_mm)

    # Keep only obstacles that can Z-clip the held item at Z_MAX.
    z_blocking = [
        obs for obs in obstacles
        if obs.top_z_mm > held_bottom_z - float(clearance_margin_z_mm)
    ]

    if not z_blocking:
        return ClearancePlan(
            waypoints_xy_mm=(),
            held_bottom_z_mm=held_bottom_z,
            blocking_labels=(),
            bypass_side="none",
            warnings=(),
        )

    # Direct path check.
    direct_blockers = _blockers_on_path(
        pick_xy, bag_xy, z_blocking, gripper_half_width_mm, clearance_margin_xy_mm
    )
    if not direct_blockers:
        return ClearancePlan(
            waypoints_xy_mm=(),
            held_bottom_z_mm=held_bottom_z,
            blocking_labels=(),
            bypass_side="none",
            warnings=(),
        )

    blocking_labels = tuple(obs.label for obs in direct_blockers)

    # Build bypass waypoints: same Y as pick, shifted to X-min or X-max.
    x_min_wp = np.array([platform_x_range_mm[0] - float(bypass_x_margin_mm), pick_xy[1]], dtype=np.float64)
    x_max_wp = np.array([platform_x_range_mm[1] + float(bypass_x_margin_mm), pick_xy[1]], dtype=np.float64)

    def _route_blockers(wp: np.ndarray) -> list[ObstacleBox]:
        return (
            _blockers_on_path(pick_xy, wp, z_blocking, gripper_half_width_mm, clearance_margin_xy_mm)
            + _blockers_on_path(wp, bag_xy, z_blocking, gripper_half_width_mm, clearance_margin_xy_mm)
        )

    x_min_blockers = _route_blockers(x_min_wp)
    x_max_blockers = _route_blockers(x_max_wp)

    if len(x_min_blockers) <= len(x_max_blockers):
        chosen_wp, chosen_side, remaining = x_min_wp, "x_min", x_min_blockers
    else:
        chosen_wp, chosen_side, remaining = x_max_wp, "x_max", x_max_blockers

    warnings: list[str] = []
    if remaining:
        warnings.append(
            f"bypass via {chosen_side} still passes near "
            + ", ".join(obs.label for obs in remaining)
            + " — no fully-clear path found at Z_MAX"
        )

    return ClearancePlan(
        waypoints_xy_mm=(chosen_wp,),
        held_bottom_z_mm=held_bottom_z,
        blocking_labels=blocking_labels,
        bypass_side=chosen_side,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Candidate → ObstacleBox adapter
# ---------------------------------------------------------------------------

def obstacles_from_candidates(candidates: list) -> list[ObstacleBox]:
    """Build ObstacleBox list from CandidateDebug objects.

    Uses aabb_from_object_candidate (lazy import keeps this module usable in
    offline tests without the full vision stack).  Candidates that fail AABB
    extraction are silently skipped.
    """
    from planning.aabb_utils import aabb_from_object_candidate  # lazy

    result: list[ObstacleBox] = []
    for dbg in candidates:
        try:
            with io.StringIO() as _sink, redirect_stdout(_sink):
                aabb = aabb_from_object_candidate(
                    dbg.candidate,
                    default_label=str(
                        getattr(getattr(dbg.candidate, "yolo", None), "class_name", "unknown")
                    ),
                )
            center = np.asarray(aabb.center_xyz_mm, dtype=np.float64)
            size = np.asarray(aabb.size_xyz_mm, dtype=np.float64)
            result.append(ObstacleBox(
                label=str(aabb.label),
                center_xy_mm=center[:2].copy(),
                top_z_mm=float(center[2] + size[2] / 2.0),
                half_extents_xy_mm=(size[:2] / 2.0).copy(),
            ))
        except Exception:
            pass
    return result
