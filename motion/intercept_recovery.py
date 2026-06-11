from __future__ import annotations

"""motion/intercept_recovery.py — Pick-zone conflict detection and temp-spot finding.

Used by the "intercept-and-clear" mode in the main run script.

Workflow
--------
1. Before picking target T, call ``check_for_pick_interceptors()`` to see
   whether any other candidate's AABB overlaps the gripper's open footprint
   at T's pick XY / phi.

2. If interceptors are found and INTERCEPT_MODE == "clear":
   a. Pick the closest interceptor B first.
   b. Call ``find_temporary_platform_spot()`` to locate a free XY on the
      platform, then place B there.
   c. Now pick T normally.

3. The arm lowers to place B at:
       arm_Z = blocker_bottom_z + blocker_height + GRIPPER_OFFSET + Z_MARGIN
   so the item rests on the platform without being crushed.
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterceptBlocker:
    """A candidate that would be clipped by the gripper during target pick descent."""
    blocker_label: str
    blocker_index: int           # candidate.index
    separation_mm: float         # centre-to-centre XY distance from target
    overlap_mm: float            # how far the AABBs overlap (positive = more collision)
    reason: str


@dataclass(frozen=True)
class TemporaryPlaceSpot:
    """A free XY location on the platform for temporarily relocating a blocker."""
    xy_mm: np.ndarray            # shape (2,)
    min_clearance_mm: float      # distance to nearest occupied position
    reason: str


# ---------------------------------------------------------------------------
# Intercept detection
# ---------------------------------------------------------------------------

def _gripper_aa_half_extents_mm(
    phi_deg: float,
    half_width_mm: float,
    half_depth_mm: float,
) -> tuple[float, float]:
    """Axis-aligned half-extents of the gripper open footprint after rotation phi_deg."""
    rad = np.deg2rad(float(phi_deg))
    c, s = abs(float(np.cos(rad))), abs(float(np.sin(rad)))
    hx = c * float(half_depth_mm) + s * float(half_width_mm)
    hy = s * float(half_depth_mm) + c * float(half_width_mm)
    return hx, hy


def check_for_pick_interceptors(
    target_xy_mm: np.ndarray,
    target_phi_deg: float,
    other_aabbs: list[tuple[str, int, np.ndarray, np.ndarray]],
    *,
    gripper_half_width_mm: float,
    gripper_half_depth_mm: float,
    intercept_margin_mm: float = 10.0,
) -> list[InterceptBlocker]:
    """Return candidates whose AABB + margin overlaps the gripper footprint at target.

    Parameters
    ----------
    target_xy_mm:
        Robot XY of the target pick.
    target_phi_deg:
        Gripper rotation at pick (degrees).
    other_aabbs:
        List of ``(label, index, center_xyz_mm, size_xyz_mm)`` for all
        candidates except the target.
    gripper_half_width_mm:
        Half-width of the open gripper (perpendicular to phi axis).
    gripper_half_depth_mm:
        Half-depth of the open gripper (along phi axis, pivot-to-tip).
    intercept_margin_mm:
        Extra safety margin added to each candidate's AABB half-extents.
    """
    txy = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
    grip_hx, grip_hy = _gripper_aa_half_extents_mm(
        target_phi_deg, gripper_half_width_mm, gripper_half_depth_mm
    )

    blockers: list[InterceptBlocker] = []
    for label, idx, center_xyz, size_xyz in other_aabbs:
        other_xy = np.asarray(center_xyz, dtype=np.float64)[:2]
        other_hx = float(size_xyz[0]) / 2.0 + float(intercept_margin_mm)
        other_hy = float(size_xyz[1]) / 2.0 + float(intercept_margin_mm)

        dx = abs(float(txy[0]) - float(other_xy[0]))
        dy = abs(float(txy[1]) - float(other_xy[1]))
        hx_sum = grip_hx + other_hx
        hy_sum = grip_hy + other_hy

        if dx < hx_sum and dy < hy_sum:
            # Overlap in both axes → intercept
            overlap = min(hx_sum - dx, hy_sum - dy)
            sep = float(np.linalg.norm(txy - other_xy))
            blockers.append(InterceptBlocker(
                blocker_label=str(label),
                blocker_index=int(idx),
                separation_mm=sep,
                overlap_mm=float(overlap),
                reason="gripper_footprint_overlap",
            ))

    # Sort by overlap descending (worst first)
    blockers.sort(key=lambda b: -b.overlap_mm)
    return blockers


# ---------------------------------------------------------------------------
# Temporary spot finder
# ---------------------------------------------------------------------------

def find_temporary_platform_spot(
    occupied_xys_mm: list[np.ndarray],
    *,
    platform_x_range_mm: tuple[float, float],
    platform_y_range_mm: tuple[float, float],
    clear_radius_mm: float = 80.0,
    grid_step_mm: float = 40.0,
) -> TemporaryPlaceSpot | None:
    """Find the most-isolated free XY on the platform.

    Scans a grid inside the platform bounds and returns the point with the
    largest minimum distance to all occupied positions that is still >= clear_radius_mm.

    Returns None if no such point exists.
    """
    x_min, x_max = platform_x_range_mm
    y_min, y_max = platform_y_range_mm

    best_xy: np.ndarray | None = None
    best_dist = -1.0

    x = x_min + grid_step_mm / 2.0
    while x <= x_max:
        y = y_min + grid_step_mm / 2.0
        while y <= y_max:
            candidate = np.array([x, y], dtype=np.float64)
            if occupied_xys_mm:
                min_dist = float(min(
                    np.linalg.norm(candidate - np.asarray(occ, dtype=np.float64).reshape(2))
                    for occ in occupied_xys_mm
                ))
            else:
                min_dist = float("inf")

            if min_dist >= float(clear_radius_mm) and min_dist > best_dist:
                best_dist = min_dist
                best_xy = candidate.copy()

            y += grid_step_mm
        x += grid_step_mm

    if best_xy is None:
        return None
    return TemporaryPlaceSpot(
        xy_mm=best_xy,
        min_clearance_mm=best_dist,
        reason="grid_search",
    )


# ---------------------------------------------------------------------------
# Platform place-Z helper
# ---------------------------------------------------------------------------

def compute_platform_place_z_mm(
    blocker_bottom_z_mm: float,
    blocker_height_mm: float,
    gripper_offset_mm: float,
    z_margin_mm: float = 15.0,
    z_max_mm: float = 270.0,
) -> float:
    """Arm Z needed to lower a held blocker onto the platform at the temp spot.

    The gripper tip descends to (blocker_bottom_z + blocker_height + z_margin),
    which places the item just above its original surface height.
    Clamped to [0, z_max_mm].
    """
    place_z = (
        float(blocker_bottom_z_mm)
        + float(blocker_height_mm)
        + float(gripper_offset_mm)
        + float(z_margin_mm)
    )
    return float(np.clip(place_z, 0.0, z_max_mm))
