from __future__ import annotations

"""Simple adjacent placement planner for padded axis-aligned 3D boxes."""

from dataclasses import dataclass

import numpy as np

from planning.aabb_utils import AxisAlignedBox3D, PaddedBox3D, make_aabb_from_center_size


@dataclass
class AdjacentPlacementPlan:
    reference_box: PaddedBox3D
    moving_box: PaddedBox3D
    placed_box: AxisAlignedBox3D
    target_center_xy_mm: np.ndarray
    target_center_xyz_mm: np.ndarray
    target_phi_deg: float
    direction: str
    notes: str


def compute_adjacent_placement(
    reference_padded_box: PaddedBox3D,
    moving_padded_box: PaddedBox3D,
    direction: str = "right",
    surface_z_mm: float = 0.0,
    place_phi_deg: float = 0.0,
) -> AdjacentPlacementPlan:
    d = str(direction).strip().lower()
    if d not in {"right", "left", "up", "down"}:
        raise ValueError(f"unsupported direction={direction!r}; expected right/left/up/down")

    ref = reference_padded_box.padded_box
    mov = moving_padded_box.padded_box

    x_ref = float(ref.center_xyz_mm[0])
    y_ref = float(ref.center_xyz_mm[1])
    sx_ref = float(ref.size_xyz_mm[0])
    sy_ref = float(ref.size_xyz_mm[1])
    sx_mov = float(mov.size_xyz_mm[0])
    sy_mov = float(mov.size_xyz_mm[1])

    x_new = x_ref
    y_new = y_ref
    if d == "right":
        x_new = x_ref + 0.5 * sx_ref + 0.5 * sx_mov
    elif d == "left":
        x_new = x_ref - 0.5 * sx_ref - 0.5 * sx_mov
    elif d == "up":
        y_new = y_ref + 0.5 * sy_ref + 0.5 * sy_mov
    elif d == "down":
        y_new = y_ref - 0.5 * sy_ref - 0.5 * sy_mov

    z_new = float(surface_z_mm) + 0.5 * float(mov.size_xyz_mm[2])
    center = np.array([x_new, y_new, z_new], dtype=np.float64)

    placed_box = make_aabb_from_center_size(
        center_xyz_mm=center,
        size_xyz_mm=mov.size_xyz_mm,
        label=f"{mov.label}_placed_{d}",
    )

    plan = AdjacentPlacementPlan(
        reference_box=reference_padded_box,
        moving_box=moving_padded_box,
        placed_box=placed_box,
        target_center_xy_mm=center[:2].copy(),
        target_center_xyz_mm=center.copy(),
        target_phi_deg=float(place_phi_deg),
        direction=d,
        notes="flush placement using padded AABBs in placement frame",
    )
    print(
        "[ADJACENT PLAN] "
        f"direction={plan.direction} "
        f"target_xy=({plan.target_center_xy_mm[0]:.1f},{plan.target_center_xy_mm[1]:.1f}) "
        f"target_z={plan.target_center_xyz_mm[2]:.1f} "
        f"phi={plan.target_phi_deg:.1f}"
    )
    return plan
