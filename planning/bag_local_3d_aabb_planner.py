from __future__ import annotations

"""Deterministic bag-local 3D AABB placement planner."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from planning.aabb_utils import AxisAlignedBox3D, make_aabb_from_min_max

_EPS = 1e-6


@dataclass
class PlacementAttempt3D:
    attempt_i: int
    raw_box_local: AxisAlignedBox3D
    padded_box_local: AxisAlignedBox3D
    accepted: bool
    reasons: tuple[str, ...]
    support_kind: str


@dataclass
class PlacementPlan3D:
    raw_box_local: AxisAlignedBox3D
    padded_box_local: AxisAlignedBox3D
    target_center_xy_mm: np.ndarray
    target_min_xyz_mm: np.ndarray
    attempts: list[PlacementAttempt3D]
    notes: str


def _rounded_sorted(values: set[float]) -> list[float]:
    rounded = {round(float(v), 6) for v in values if np.isfinite(v)}
    return sorted(rounded)


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def aabbs_intersect_3d(a: AxisAlignedBox3D, b: AxisAlignedBox3D, eps: float = _EPS) -> bool:
    return (
        _overlap_1d(a.min_xyz_mm[0], a.max_xyz_mm[0], b.min_xyz_mm[0], b.max_xyz_mm[0]) > eps
        and _overlap_1d(a.min_xyz_mm[1], a.max_xyz_mm[1], b.min_xyz_mm[1], b.max_xyz_mm[1]) > eps
        and _overlap_1d(a.min_xyz_mm[2], a.max_xyz_mm[2], b.min_xyz_mm[2], b.max_xyz_mm[2]) > eps
    )


def _support_ok(
    raw_box_local: AxisAlignedBox3D,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    support_min_overlap_ratio: float,
) -> tuple[bool, str, tuple[str, ...]]:
    z0 = float(raw_box_local.min_xyz_mm[2])
    if z0 <= _EPS:
        return True, "floor", ()

    cx = float(raw_box_local.center_xyz_mm[0])
    cy = float(raw_box_local.center_xyz_mm[1])
    footprint = float(raw_box_local.size_xyz_mm[0] * raw_box_local.size_xyz_mm[1])
    supporters: list[str] = []

    for other in placed_raw_boxes_local:
        if abs(float(other.max_xyz_mm[2]) - z0) > _EPS:
            continue
        ox = _overlap_1d(
            raw_box_local.min_xyz_mm[0],
            raw_box_local.max_xyz_mm[0],
            other.min_xyz_mm[0],
            other.max_xyz_mm[0],
        )
        oy = _overlap_1d(
            raw_box_local.min_xyz_mm[1],
            raw_box_local.max_xyz_mm[1],
            other.min_xyz_mm[1],
            other.max_xyz_mm[1],
        )
        if ox <= _EPS or oy <= _EPS:
            continue
        overlap_area = ox * oy
        center_supported = (
            float(other.min_xyz_mm[0]) - _EPS <= cx <= float(other.max_xyz_mm[0]) + _EPS
            and float(other.min_xyz_mm[1]) - _EPS <= cy <= float(other.max_xyz_mm[1]) + _EPS
        )
        if center_supported or overlap_area >= footprint * float(support_min_overlap_ratio):
            supporters.append(other.label or "support")

    if supporters:
        return True, "stack", tuple(supporters)
    return False, "unsupported", ()


def plan_bag_local_aabb_placement(
    *,
    item_label: str,
    raw_size_xyz_mm: np.ndarray,
    padding_min_xyz_mm: np.ndarray,
    padding_max_xyz_mm: np.ndarray,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    support_min_overlap_ratio: float = 0.35,
) -> PlacementPlan3D:
    raw_size = np.asarray(raw_size_xyz_mm, dtype=np.float64).reshape(3)
    pad_min = np.asarray(padding_min_xyz_mm, dtype=np.float64).reshape(3)
    pad_max = np.asarray(padding_max_xyz_mm, dtype=np.float64).reshape(3)
    bag_size = np.asarray(bag_size_xyz_mm, dtype=np.float64).reshape(3)
    if np.any(raw_size <= 0.0):
        raise ValueError("item sizes must be positive")
    if np.any(pad_min < 0.0) or np.any(pad_max < 0.0):
        raise ValueError("padding must be non-negative")

    x_candidates = {0.0}
    y_candidates = {0.0}
    z_candidates = {0.0}

    for other in placed_raw_boxes_local:
        x_candidates.update(
            [
                float(other.min_xyz_mm[0]),
                float(other.max_xyz_mm[0]),
                float(other.min_xyz_mm[0] + max(0.0, (other.size_xyz_mm[0] - raw_size[0]) * 0.5)),
            ]
        )
        y_candidates.update(
            [
                float(other.min_xyz_mm[1]),
                float(other.max_xyz_mm[1]),
                float(other.min_xyz_mm[1] + max(0.0, (other.size_xyz_mm[1] - raw_size[1]) * 0.5)),
            ]
        )
        z_candidates.add(float(other.max_xyz_mm[2]))

    attempts: list[PlacementAttempt3D] = []
    attempt_i = 0
    for z0, y0, x0 in product(
        _rounded_sorted(z_candidates),
        _rounded_sorted(y_candidates),
        _rounded_sorted(x_candidates),
    ):
        raw_min = np.array([x0, y0, z0], dtype=np.float64)
        raw_max = raw_min + raw_size
        raw_box = make_aabb_from_min_max(raw_min, raw_max, label=item_label)

        padded_min = raw_min - pad_min
        padded_max = raw_max + pad_max
        padded_box = make_aabb_from_min_max(padded_min, padded_max, label=f"{item_label}_padded")

        reasons: list[str] = []
        if np.any(raw_min < -_EPS):
            reasons.append("raw_box_below_bag_origin")
        if np.any(raw_max - bag_size > _EPS):
            reasons.append("raw_box_outside_bag_bounds")

        collision_labels = [
            other.label or f"box_{i}"
            for i, other in enumerate(placed_padded_boxes_local, start=1)
            if aabbs_intersect_3d(padded_box, other)
        ]
        if collision_labels:
            reasons.append("intersects:" + ",".join(collision_labels))

        support_ok, support_kind, support_labels = _support_ok(
            raw_box,
            placed_raw_boxes_local,
            support_min_overlap_ratio=support_min_overlap_ratio,
        )
        if not support_ok:
            reasons.append("unsupported")
        elif support_labels:
            reasons.append("supports:" + ",".join(support_labels))

        accepted = not any(
            reason in {"raw_box_below_bag_origin", "raw_box_outside_bag_bounds", "unsupported"}
            or reason.startswith("intersects:")
            for reason in reasons
        )
        attempt_i += 1
        attempts.append(
            PlacementAttempt3D(
                attempt_i=attempt_i,
                raw_box_local=raw_box,
                padded_box_local=padded_box,
                accepted=accepted,
                reasons=tuple(reasons) if reasons else ("accepted",),
                support_kind=support_kind,
            )
        )
        if accepted:
            return PlacementPlan3D(
                raw_box_local=raw_box,
                padded_box_local=padded_box,
                target_center_xy_mm=raw_box.center_xyz_mm[:2].copy(),
                target_min_xyz_mm=raw_box.min_xyz_mm.copy(),
                attempts=attempts,
                notes=f"deterministic bag-local 3D search accepted at attempt {attempt_i}",
            )

    raise RuntimeError(
        f"no valid 3D AABB placement for {item_label} in bag "
        f"after {len(attempts)} deterministic attempts"
    )
