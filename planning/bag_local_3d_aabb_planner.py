from __future__ import annotations

"""Robot-aware 2.5D bag-local AABB planner."""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from planning.aabb_utils import AxisAlignedBox3D, make_aabb_from_min_max

_EPS = 1e-6
_FUTURE_REFINEMENT_CANDIDATES_PER_LAYER = 3
_FUTURE_LEGALITY_XY_LIMIT = 8


@dataclass(frozen=True)
class PlannerWeights:
    weight_priority: float = 1.05
    footprint_priority: float = 1.15
    stability_priority: float = 1.05
    height_priority: float = 0.95
    volume_priority: float = 0.90
    top_clipping_risk_priority: float = 1.15
    rigidity_priority: float = 0.95
    low_fragility_priority: float = 0.65
    z_penalty: float = 1.25
    stack_penalty: float = 0.25
    support_bonus: float = 1.20
    compactness_bonus: float = 0.08   # was 0.32 — reduced to stop items clustering on one side
    wall_contact_bonus: float = 0.22
    neighbor_contact_bonus: float = 0.06  # was 0.20 — reduced to stop cascade clustering
    support_risk_penalty: float = 1.10
    overhang_penalty: float = 1.00
    fragile_high_bonus: float = 0.45
    strong_low_bonus: float = 0.55
    slender_low_penalty: float = 1.20
    pin_corner_bonus: float = 0.25
    future_stranded_object_penalty: float = 2.75
    future_tall_object_stranded_penalty: float = 3.60
    tall_object_low_placement_reward: float = 1.10
    tall_object_reservation_penalty: float = 2.80


@dataclass(frozen=True)
class FutureItemSpec:
    label: str
    raw_size_xyz_mm: np.ndarray
    padding_min_xyz_mm: np.ndarray
    padding_max_xyz_mm: np.ndarray
    object_props: dict[str, Any]


@dataclass
class ScoreBreakdown:
    object_priority: float
    placement_quality: float
    z_penalty: float
    support_bonus: float
    compactness_bonus: float
    support_risk_penalty: float
    overhang_penalty: float
    future_height_penalty: float
    tall_object_low_reward: float
    tall_object_reservation_penalty: float
    total: float


@dataclass
class CandidatePlacement2p5D:
    raw_box_local: AxisAlignedBox3D
    padded_box_local: AxisAlignedBox3D
    yaw_deg: float
    orientation_label: str
    layer_z_mm: float
    score: float
    support_ratio: float
    overhang_ratio: float
    wall_contacts: int
    neighbor_contacts: int
    support_labels: tuple[str, ...]
    support_risk: float
    future_placeable_count: int
    future_total_count: int
    stranded_labels: tuple[str, ...]
    tallest_stranded_height_mm: float
    future_height_penalty: float
    top_clip_margin_mm: float
    future_feasibility_used: bool
    accepted: bool
    reasons: tuple[str, ...]
    breakdown: ScoreBreakdown


@dataclass
class PlacementPlan3D:
    raw_box_local: AxisAlignedBox3D
    padded_box_local: AxisAlignedBox3D
    target_center_xy_mm: np.ndarray
    target_min_xyz_mm: np.ndarray
    yaw_deg: float
    orientation_label: str
    score: float
    support_ratio: float
    layer_z_mm: float
    future_placeable_count: int
    future_total_count: int
    stranded_labels: tuple[str, ...]
    tallest_stranded_height_mm: float
    future_height_penalty: float
    top_clip_margin_mm: float
    future_feasibility_used: bool
    candidates_evaluated: int
    valid_candidates: int
    attempts: list[CandidatePlacement2p5D] = field(default_factory=list)
    notes: str = ""


def _as_vec3(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{label} must have 3 values")
    out = arr[:3].astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} must be finite")
    return out


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _rounded_sorted(values: set[float]) -> list[float]:
    return sorted({_rounded(v) for v in values if np.isfinite(v)})


def _clamp(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return lo
    return float(min(max(value, lo), hi))


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _overlap_area_xy(a: AxisAlignedBox3D, b: AxisAlignedBox3D) -> float:
    return _overlap_1d(a.min_xyz_mm[0], a.max_xyz_mm[0], b.min_xyz_mm[0], b.max_xyz_mm[0]) * _overlap_1d(
        a.min_xyz_mm[1], a.max_xyz_mm[1], b.min_xyz_mm[1], b.max_xyz_mm[1]
    )


def aabbs_intersect_3d(a: AxisAlignedBox3D, b: AxisAlignedBox3D, eps: float = _EPS) -> bool:
    return (
        _overlap_1d(a.min_xyz_mm[0], a.max_xyz_mm[0], b.min_xyz_mm[0], b.max_xyz_mm[0]) > eps
        and _overlap_1d(a.min_xyz_mm[1], a.max_xyz_mm[1], b.min_xyz_mm[1], b.max_xyz_mm[1]) > eps
        and _overlap_1d(a.min_xyz_mm[2], a.max_xyz_mm[2], b.min_xyz_mm[2], b.max_xyz_mm[2]) > eps
    )


def _make_box_from_min_size(min_xyz_mm: np.ndarray, size_xyz_mm: np.ndarray, label: str) -> AxisAlignedBox3D:
    return make_aabb_from_min_max(min_xyz_mm, min_xyz_mm + size_xyz_mm, label=label)


def _default_props(props: dict[str, Any] | None) -> dict[str, float | bool]:
    props = props or {}
    return {
        "weight": float(props.get("weight", 0.1)),
        "fragility": float(props.get("fragility", 0.3)),
        "compliance": float(props.get("compliance", 0.3)),
        "pin_slot": bool(props.get("pin_slot", False)),
    }


def _future_spec_props(spec: FutureItemSpec) -> dict[str, float | bool]:
    return _default_props(spec.object_props)


def _orientation_options(raw_size_xyz_mm: np.ndarray) -> list[tuple[np.ndarray, float, str]]:
    sx, sy, sz = [float(v) for v in raw_size_xyz_mm]
    options = [(np.array([sx, sy, sz], dtype=np.float64), 0.0, "yaw0_wh")]
    if abs(sx - sy) > max(5.0, 0.05 * max(sx, sy)):
        options.append((np.array([sy, sx, sz], dtype=np.float64), 90.0, "yaw90_swapped"))
    return options


def _layer_surfaces(bag_height_mm: float, item_height_mm: float, placed_raw_boxes_local: list[AxisAlignedBox3D]) -> list[float]:
    z_values = {0.0}
    for box in placed_raw_boxes_local:
        top_z = float(box.max_xyz_mm[2])
        if top_z + item_height_mm <= bag_height_mm + _EPS:
            z_values.add(top_z)
    return _rounded_sorted(z_values)


def _candidate_xy_positions(
    *,
    oriented_size_xyz_mm: np.ndarray,
    bag_size_xyz_mm: np.ndarray,
    layer_z_mm: float,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    object_props: dict[str, float | bool],
) -> list[tuple[float, float]]:
    sx = float(oriented_size_xyz_mm[0])
    sy = float(oriented_size_xyz_mm[1])
    max_x = max(0.0, float(bag_size_xyz_mm[0] - sx))
    max_y = max(0.0, float(bag_size_xyz_mm[1] - sy))

    x_edges = {0.0, max_x}
    y_edges = {0.0, max_y}
    combos: set[tuple[float, float]] = set()

    for box in placed_raw_boxes_local:
        x_edges.update([float(box.min_xyz_mm[0]), float(box.max_xyz_mm[0]), float(box.max_xyz_mm[0] - sx)])
        y_edges.update([float(box.min_xyz_mm[1]), float(box.max_xyz_mm[1]), float(box.max_xyz_mm[1] - sy)])
        if abs(float(box.max_xyz_mm[2]) - layer_z_mm) <= _EPS:
            combos.add((_rounded(_clamp(float(box.center_xyz_mm[0] - 0.5 * sx), 0.0, max_x)), _rounded(_clamp(float(box.center_xyz_mm[1] - 0.5 * sy), 0.0, max_y))))

    if bool(object_props.get("pin_slot", False)):
        combos.update(
            {
                (0.0, 0.0),
                (max_x, 0.0),
                (0.0, max_y),
                (max_x, max_y),
            }
        )

    xs = [_rounded(_clamp(x, 0.0, max_x)) for x in x_edges]
    ys = [_rounded(_clamp(y, 0.0, max_y)) for y in y_edges]
    for x, y in product(xs, ys):
        combos.add((x, y))

    return sorted(combos, key=lambda xy: (xy[1], xy[0]))


def _prioritize_xy_positions(
    positions: list[tuple[float, float]],
    *,
    max_x: float,
    max_y: float,
) -> list[tuple[float, float]]:
    cx = 0.5 * max_x
    cy = 0.5 * max_y
    return sorted(
        positions,
        key=lambda xy: (
            min(xy[0], max_x - xy[0], xy[1], max_y - xy[1]),
            abs(xy[0] - cx) + abs(xy[1] - cy),
            xy[1],
            xy[0],
        ),
    )


def _support_metrics(
    raw_box_local: AxisAlignedBox3D,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    support_props_by_label: dict[str, dict[str, Any]],
) -> tuple[float, tuple[str, ...], float]:
    z0 = float(raw_box_local.min_xyz_mm[2])
    if z0 <= _EPS:
        return 1.0, (), 0.0

    footprint_area = max(1.0, float(raw_box_local.size_xyz_mm[0] * raw_box_local.size_xyz_mm[1]))
    support_area = 0.0
    support_labels: list[str] = []
    risk_terms: list[float] = []
    for box in placed_raw_boxes_local:
        if abs(float(box.max_xyz_mm[2]) - z0) > _EPS:
            continue
        overlap_area = _overlap_area_xy(raw_box_local, box)
        if overlap_area <= _EPS:
            continue
        support_area += overlap_area
        support_labels.append(box.label or "support")
        props = _default_props(support_props_by_label.get(box.label or "", {}))
        support_strength = max(0.05, float(props["weight"])) * (1.15 - 0.55 * float(props["fragility"])) * (1.15 - 0.55 * float(props["compliance"]))
        risk_terms.append(overlap_area / footprint_area / support_strength)
    support_ratio = _clip01(support_area / footprint_area)
    support_risk_scale = max(risk_terms) if risk_terms else 0.0
    return support_ratio, tuple(sorted(set(support_labels))), support_risk_scale


def _heavy_on_fragile_support(
    *,
    object_props: dict[str, float | bool],
    support_labels: tuple[str, ...],
    support_props_by_label: dict[str, dict[str, Any]],
) -> tuple[bool, float]:
    if not support_labels:
        return False, 0.0
    obj_weight = float(object_props["weight"])
    worst_risk = 0.0
    for label in support_labels:
        props = _default_props(support_props_by_label.get(label, {}))
        fragility = float(props["fragility"])
        compliance = float(props["compliance"])
        support_weight = max(0.05, float(props["weight"]))
        risk = obj_weight * (0.35 + fragility + compliance) / support_weight
        worst_risk = max(worst_risk, risk)
        if obj_weight > support_weight * 1.35 and (fragility > 0.55 or compliance > 0.60):
            return True, worst_risk
    return False, worst_risk


def _vertical_descent_blocked(
    *,
    raw_box_local: AxisAlignedBox3D,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    clearance_xy_mm: float,
) -> bool:
    z0 = float(raw_box_local.min_xyz_mm[2])
    if z0 >= float(bag_size_xyz_mm[2]) - _EPS:
        return True
    sweep_min = raw_box_local.min_xyz_mm.copy()
    sweep_max = raw_box_local.max_xyz_mm.copy()
    sweep_min[0] -= clearance_xy_mm
    sweep_min[1] -= clearance_xy_mm
    sweep_min[2] = z0 + _EPS
    sweep_max[0] += clearance_xy_mm
    sweep_max[1] += clearance_xy_mm
    sweep_max[2] = float(bag_size_xyz_mm[2])
    sweep_box = make_aabb_from_min_max(sweep_min, sweep_max, label="descent_sweep")

    for other in placed_raw_boxes_local:
        if float(other.max_xyz_mm[2]) <= z0 + _EPS:
            continue
        if aabbs_intersect_3d(sweep_box, other):
            return True

    # TODO: add gripper swept-volume and finger-clearance checks here.
    return False


def _contact_counts(
    raw_box_local: AxisAlignedBox3D,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    tol_mm: float = 3.0,
) -> tuple[int, int]:
    wall_contacts = 0
    if abs(float(raw_box_local.min_xyz_mm[0])) <= tol_mm:
        wall_contacts += 1
    if abs(float(raw_box_local.min_xyz_mm[1])) <= tol_mm:
        wall_contacts += 1
    if abs(float(bag_size_xyz_mm[0] - raw_box_local.max_xyz_mm[0])) <= tol_mm:
        wall_contacts += 1
    if abs(float(bag_size_xyz_mm[1] - raw_box_local.max_xyz_mm[1])) <= tol_mm:
        wall_contacts += 1

    neighbor_contacts = 0
    for other in placed_raw_boxes_local:
        x_overlap = _overlap_1d(raw_box_local.min_xyz_mm[0], raw_box_local.max_xyz_mm[0], other.min_xyz_mm[0], other.max_xyz_mm[0])
        y_overlap = _overlap_1d(raw_box_local.min_xyz_mm[1], raw_box_local.max_xyz_mm[1], other.min_xyz_mm[1], other.max_xyz_mm[1])
        z_overlap = _overlap_1d(raw_box_local.min_xyz_mm[2], raw_box_local.max_xyz_mm[2], other.min_xyz_mm[2], other.max_xyz_mm[2])
        if x_overlap > 0.0 and z_overlap > 0.0 and abs(float(raw_box_local.min_xyz_mm[1] - other.max_xyz_mm[1])) <= tol_mm:
            neighbor_contacts += 1
        if x_overlap > 0.0 and z_overlap > 0.0 and abs(float(raw_box_local.max_xyz_mm[1] - other.min_xyz_mm[1])) <= tol_mm:
            neighbor_contacts += 1
        if y_overlap > 0.0 and z_overlap > 0.0 and abs(float(raw_box_local.min_xyz_mm[0] - other.max_xyz_mm[0])) <= tol_mm:
            neighbor_contacts += 1
        if y_overlap > 0.0 and z_overlap > 0.0 and abs(float(raw_box_local.max_xyz_mm[0] - other.min_xyz_mm[0])) <= tol_mm:
            neighbor_contacts += 1
    return wall_contacts, neighbor_contacts


def _top_clipping_risk(size_xyz_mm: np.ndarray, bag_size_xyz_mm: np.ndarray) -> float:
    height_norm = _clip01(float(size_xyz_mm[2]) / max(1.0, float(bag_size_xyz_mm[2])))
    return _clip01((height_norm - 0.45) / 0.55)


def _score_candidate(
    *,
    raw_box_local: AxisAlignedBox3D,
    layer_z_mm: float,
    bag_size_xyz_mm: np.ndarray,
    object_props: dict[str, float | bool],
    support_ratio: float,
    overhang_ratio: float,
    wall_contacts: int,
    neighbor_contacts: int,
    support_risk: float,
    support_labels: tuple[str, ...],
    future_placeable_count: int,
    future_total_count: int,
    tallest_stranded_height_mm: float,
    future_height_penalty: float,
    tallest_remaining_height_mm: float,
    strands_tallest_low_slot: bool,
    top_clip_margin_mm: float,
    weights: PlannerWeights,
) -> ScoreBreakdown:
    bag_area = max(1.0, float(bag_size_xyz_mm[0] * bag_size_xyz_mm[1]))
    footprint_area = float(raw_box_local.size_xyz_mm[0] * raw_box_local.size_xyz_mm[1])
    footprint_fraction = footprint_area / bag_area
    footprint_ratio = _clip01(footprint_area / bag_area * 5.5)
    stable_base = _clip01(np.sqrt(max(1.0, footprint_area)) / max(1.0, float(raw_box_local.size_xyz_mm[2])))
    height_norm = _clip01(float(raw_box_local.size_xyz_mm[2]) / max(1.0, float(bag_size_xyz_mm[2])))
    volume_norm = _clip01(float(np.prod(raw_box_local.size_xyz_mm)) / max(1.0, float(np.prod(bag_size_xyz_mm))))
    weight_value = _clip01(float(object_props["weight"]) / 0.40)
    fragility = _clip01(float(object_props["fragility"]))
    compliance = _clip01(float(object_props["compliance"]))
    rigidity = 1.0 - compliance
    z_norm = _clip01(layer_z_mm / max(1.0, float(bag_size_xyz_mm[2])))
    foundation_coverage = footprint_fraction * max(0.0, 1.0 - 0.35 * fragility - 0.15 * compliance)
    top_clipping_risk_norm = _top_clipping_risk(raw_box_local.size_xyz_mm, bag_size_xyz_mm)
    stranded_count = max(0, future_total_count - future_placeable_count)
    tallest_stranded_norm = _clip01(float(tallest_stranded_height_mm) / max(1.0, float(bag_size_xyz_mm[2])))
    tallest_remaining_norm = _clip01(float(tallest_remaining_height_mm) / max(1.0, float(bag_size_xyz_mm[2])))

    object_priority = (
        weights.weight_priority * weight_value
        + weights.footprint_priority * footprint_ratio
        + weights.stability_priority * stable_base
        + weights.height_priority * height_norm
        + weights.volume_priority * volume_norm
        + weights.top_clipping_risk_priority * top_clipping_risk_norm
        + weights.rigidity_priority * rigidity
        + weights.low_fragility_priority * (1.0 - fragility)
        + 7.00 * foundation_coverage * (1.0 - z_norm)
        + weights.strong_low_bonus * (weight_value + rigidity + stable_base) * (1.0 - z_norm)
        + weights.fragile_high_bonus * (fragility + compliance) * z_norm
        - weights.slender_low_penalty * (1.0 - stable_base) * (1.0 - z_norm)
    )
    compactness_bonus = weights.compactness_bonus * _clip01((wall_contacts + neighbor_contacts) / 6.0)
    tall_object_low_reward = weights.tall_object_low_placement_reward * height_norm * (1.0 - z_norm)
    tall_object_reservation_penalty = 0.0
    if strands_tallest_low_slot and layer_z_mm <= _EPS and height_norm < tallest_remaining_norm:
        tall_object_reservation_penalty = weights.tall_object_reservation_penalty * max(0.5, tallest_remaining_norm - height_norm)
    placement_quality = (
        weights.support_bonus * support_ratio
        + weights.wall_contact_bonus * wall_contacts
        + weights.neighbor_contact_bonus * neighbor_contacts
        + compactness_bonus
        + tall_object_low_reward
        - weights.z_penalty * z_norm
        - weights.stack_penalty * float(layer_z_mm > _EPS)
        - weights.support_risk_penalty * max(0.0, support_risk - 1.0)
        - weights.overhang_penalty * overhang_ratio
        - weights.future_stranded_object_penalty * stranded_count
        - weights.future_tall_object_stranded_penalty * tallest_stranded_norm
        - future_height_penalty
        - tall_object_reservation_penalty
    )
    if bool(object_props.get("pin_slot", False)):
        is_corner = (
            (abs(float(raw_box_local.min_xyz_mm[0])) <= 2.0 or abs(float(bag_size_xyz_mm[0] - raw_box_local.max_xyz_mm[0])) <= 2.0)
            and (abs(float(raw_box_local.min_xyz_mm[1])) <= 2.0 or abs(float(bag_size_xyz_mm[1] - raw_box_local.max_xyz_mm[1])) <= 2.0)
        )
        if is_corner:
            placement_quality += weights.pin_corner_bonus

    total = object_priority + placement_quality
    return ScoreBreakdown(
        object_priority=float(object_priority),
        placement_quality=float(placement_quality),
        z_penalty=float(weights.z_penalty * z_norm),
        support_bonus=float(weights.support_bonus * support_ratio),
        compactness_bonus=float(compactness_bonus),
        support_risk_penalty=float(weights.support_risk_penalty * max(0.0, support_risk - 1.0)),
        overhang_penalty=float(weights.overhang_penalty * overhang_ratio),
        future_height_penalty=float(future_height_penalty + weights.future_stranded_object_penalty * stranded_count + weights.future_tall_object_stranded_penalty * tallest_stranded_norm),
        tall_object_low_reward=float(tall_object_low_reward),
        tall_object_reservation_penalty=float(tall_object_reservation_penalty),
        total=float(total),
    )


def _has_any_legal_placement(
    *,
    item_spec: FutureItemSpec,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    support_props_by_label: dict[str, dict[str, Any]],
    support_min_overlap_ratio: float,
    max_overhang_ratio: float,
    descent_clearance_xy_mm: float,
    floor_only: bool = False,
) -> bool:
    raw_size = _as_vec3(item_spec.raw_size_xyz_mm, "item_spec.raw_size_xyz_mm")
    pad_min = _as_vec3(item_spec.padding_min_xyz_mm, "item_spec.padding_min_xyz_mm")
    pad_max = _as_vec3(item_spec.padding_max_xyz_mm, "item_spec.padding_max_xyz_mm")
    props = _future_spec_props(item_spec)
    layers = [0.0] if floor_only else _layer_surfaces(float(bag_size_xyz_mm[2]), float(raw_size[2]), placed_raw_boxes_local)
    for layer_z_mm in layers:
        for oriented_size, yaw_deg, orientation_label in _orientation_options(raw_size):
            max_x = max(0.0, float(bag_size_xyz_mm[0] - oriented_size[0]))
            max_y = max(0.0, float(bag_size_xyz_mm[1] - oriented_size[1]))
            xy_positions = _candidate_xy_positions(
                oriented_size_xyz_mm=oriented_size,
                bag_size_xyz_mm=bag_size_xyz_mm,
                layer_z_mm=layer_z_mm,
                placed_raw_boxes_local=placed_raw_boxes_local,
                object_props=props,
            )
            xy_positions = _prioritize_xy_positions(xy_positions, max_x=max_x, max_y=max_y)[:_FUTURE_LEGALITY_XY_LIMIT]
            for x_mm, y_mm in xy_positions:
                cand = _evaluate_candidate(
                    item_label=item_spec.label,
                    oriented_size_xyz_mm=oriented_size,
                    pad_min=pad_min,
                    pad_max=pad_max,
                    bag_size_xyz_mm=bag_size_xyz_mm,
                    placed_raw_boxes_local=placed_raw_boxes_local,
                    placed_padded_boxes_local=placed_padded_boxes_local,
                    object_props=props,
                    support_props_by_label=support_props_by_label,
                    yaw_deg=yaw_deg,
                    orientation_label=orientation_label,
                    layer_z_mm=layer_z_mm,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    weights=PlannerWeights(),
                    support_min_overlap_ratio=support_min_overlap_ratio,
                    max_overhang_ratio=max_overhang_ratio,
                    descent_clearance_xy_mm=descent_clearance_xy_mm,
                    remaining_item_specs=[],
                    enable_future_feasibility=False,
                )
                if cand.accepted:
                    return True
    return False


def _future_feasibility_metrics(
    *,
    candidate_raw_box: AxisAlignedBox3D,
    candidate_padded_box: AxisAlignedBox3D,
    candidate_object_props: dict[str, float | bool],
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    support_props_by_label: dict[str, dict[str, Any]],
    remaining_item_specs: list[FutureItemSpec],
    support_min_overlap_ratio: float,
    max_overhang_ratio: float,
    descent_clearance_xy_mm: float,
) -> dict[str, Any]:
    simulated_raw = list(placed_raw_boxes_local) + [candidate_raw_box]
    simulated_padded = list(placed_padded_boxes_local) + [candidate_padded_box]
    support_props = dict(support_props_by_label)
    support_props[candidate_raw_box.label] = dict(candidate_object_props)

    stranded_labels: list[str] = []
    tallest_stranded_height_mm = 0.0
    future_height_penalty = 0.0
    tallest_remaining: FutureItemSpec | None = None
    if remaining_item_specs:
        tallest_remaining = max(remaining_item_specs, key=lambda s: float(_as_vec3(s.raw_size_xyz_mm, "remaining.raw")[2]))

    for spec in remaining_item_specs:
        is_placeable = _has_any_legal_placement(
            item_spec=spec,
            bag_size_xyz_mm=bag_size_xyz_mm,
            placed_raw_boxes_local=simulated_raw,
            placed_padded_boxes_local=simulated_padded,
            support_props_by_label=support_props,
            support_min_overlap_ratio=support_min_overlap_ratio,
            max_overhang_ratio=max_overhang_ratio,
            descent_clearance_xy_mm=descent_clearance_xy_mm,
            floor_only=False,
        )
        if not is_placeable:
            stranded_labels.append(spec.label)
            spec_size = _as_vec3(spec.raw_size_xyz_mm, "remaining.raw")
            height_norm = _clip01(float(spec_size[2]) / max(1.0, float(bag_size_xyz_mm[2])))
            volume_norm = _clip01(float(np.prod(spec_size)) / max(1.0, float(np.prod(bag_size_xyz_mm))))
            tallest_stranded_height_mm = max(tallest_stranded_height_mm, float(spec_size[2]))
            future_height_penalty += 0.8 * height_norm + 0.6 * volume_norm

    tallest_remaining_height_mm = 0.0
    tallest_remaining_floor_feasible = True
    if tallest_remaining is not None:
        tallest_remaining_height_mm = float(_as_vec3(tallest_remaining.raw_size_xyz_mm, "tallest_remaining.raw")[2])
        tallest_remaining_floor_feasible = _has_any_legal_placement(
            item_spec=tallest_remaining,
            bag_size_xyz_mm=bag_size_xyz_mm,
            placed_raw_boxes_local=simulated_raw,
            placed_padded_boxes_local=simulated_padded,
            support_props_by_label=support_props,
            support_min_overlap_ratio=support_min_overlap_ratio,
            max_overhang_ratio=max_overhang_ratio,
            descent_clearance_xy_mm=descent_clearance_xy_mm,
            floor_only=True,
        )

    return {
        "remaining_placeable_count": len(remaining_item_specs) - len(stranded_labels),
        "remaining_total_count": len(remaining_item_specs),
        "stranded_labels": tuple(stranded_labels),
        "tallest_stranded_height_mm": float(tallest_stranded_height_mm),
        "future_height_penalty": float(future_height_penalty),
        "tallest_remaining_height_mm": float(tallest_remaining_height_mm),
        "tallest_remaining_floor_feasible": bool(tallest_remaining_floor_feasible),
    }


def _evaluate_candidate(
    *,
    item_label: str,
    oriented_size_xyz_mm: np.ndarray,
    pad_min: np.ndarray,
    pad_max: np.ndarray,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    object_props: dict[str, float | bool],
    support_props_by_label: dict[str, dict[str, Any]],
    yaw_deg: float,
    orientation_label: str,
    layer_z_mm: float,
    x_mm: float,
    y_mm: float,
    weights: PlannerWeights,
    support_min_overlap_ratio: float,
    max_overhang_ratio: float,
    descent_clearance_xy_mm: float,
    remaining_item_specs: list[FutureItemSpec] | None = None,
    enable_future_feasibility: bool = True,
) -> CandidatePlacement2p5D:
    raw_min = np.array([x_mm, y_mm, layer_z_mm], dtype=np.float64)
    raw_box = _make_box_from_min_size(raw_min, oriented_size_xyz_mm, label=item_label)
    padded_box = make_aabb_from_min_max(raw_box.min_xyz_mm - pad_min, raw_box.max_xyz_mm + pad_max, label=f"{item_label}_padded")

    reasons: list[str] = []
    if np.any(raw_box.min_xyz_mm < -_EPS) or np.any(raw_box.max_xyz_mm - bag_size_xyz_mm > _EPS):
        reasons.append("outside_bag")
    if float(raw_box.max_xyz_mm[2]) > float(bag_size_xyz_mm[2]) + _EPS:
        reasons.append("top_clip")
    if float(padded_box.max_xyz_mm[2]) > float(bag_size_xyz_mm[2]) + _EPS:
        reasons.append("padded_top_clip")

    raw_hits = [box.label or "placed" for box in placed_raw_boxes_local if aabbs_intersect_3d(raw_box, box)]
    if raw_hits:
        reasons.append("raw_intersection:" + ",".join(raw_hits))

    padded_hits = [box.label or "placed" for box in placed_padded_boxes_local if aabbs_intersect_3d(padded_box, box)]
    if padded_hits:
        reasons.append("padded_intersection:" + ",".join(padded_hits))

    support_ratio, support_labels, support_risk_scale = _support_metrics(raw_box, placed_raw_boxes_local, support_props_by_label)
    overhang_ratio = 0.0 if layer_z_mm <= _EPS else max(0.0, 1.0 - support_ratio)
    if layer_z_mm > _EPS and support_ratio + _EPS < support_min_overlap_ratio:
        reasons.append(f"support_ratio<{support_min_overlap_ratio:.2f}")
    if overhang_ratio > max_overhang_ratio + _EPS:
        reasons.append(f"overhang>{max_overhang_ratio:.2f}")

    heavy_on_fragile, heavy_support_risk = _heavy_on_fragile_support(
        object_props=object_props,
        support_labels=support_labels,
        support_props_by_label=support_props_by_label,
    )
    support_risk = max(support_risk_scale * float(object_props["weight"]), heavy_support_risk)
    if heavy_on_fragile:
        reasons.append("heavy_on_fragile_support")

    if _vertical_descent_blocked(
        raw_box_local=raw_box,
        bag_size_xyz_mm=bag_size_xyz_mm,
        placed_raw_boxes_local=placed_raw_boxes_local,
        clearance_xy_mm=descent_clearance_xy_mm,
    ):
        reasons.append("vertical_descent_blocked")

    wall_contacts, neighbor_contacts = _contact_counts(raw_box, bag_size_xyz_mm, placed_raw_boxes_local)
    remaining_item_specs = list(remaining_item_specs or [])
    future_placeable_count = len(remaining_item_specs)
    future_total_count = len(remaining_item_specs)
    stranded_labels: tuple[str, ...] = ()
    tallest_stranded_height_mm = 0.0
    future_height_penalty = 0.0
    tallest_remaining_height_mm = 0.0
    tallest_remaining_floor_feasible = True
    future_feasibility_used = False
    if enable_future_feasibility and not reasons and remaining_item_specs:
        future = _future_feasibility_metrics(
            candidate_raw_box=raw_box,
            candidate_padded_box=padded_box,
            candidate_object_props=object_props,
            bag_size_xyz_mm=bag_size_xyz_mm,
            placed_raw_boxes_local=placed_raw_boxes_local,
            placed_padded_boxes_local=placed_padded_boxes_local,
            support_props_by_label=support_props_by_label,
            remaining_item_specs=remaining_item_specs,
            support_min_overlap_ratio=support_min_overlap_ratio,
            max_overhang_ratio=max_overhang_ratio,
            descent_clearance_xy_mm=descent_clearance_xy_mm,
        )
        future_placeable_count = int(future["remaining_placeable_count"])
        future_total_count = int(future["remaining_total_count"])
        stranded_labels = tuple(future["stranded_labels"])
        tallest_stranded_height_mm = float(future["tallest_stranded_height_mm"])
        future_height_penalty = float(future["future_height_penalty"])
        tallest_remaining_height_mm = float(future["tallest_remaining_height_mm"])
        tallest_remaining_floor_feasible = bool(future["tallest_remaining_floor_feasible"])
        future_feasibility_used = True
        if layer_z_mm <= _EPS and tallest_remaining_height_mm > float(raw_box.size_xyz_mm[2]) + 5.0 and not tallest_remaining_floor_feasible:
            reasons.append("strands_tallest_floor_object")
        if stranded_labels:
            reasons.append("future_stranding:" + ",".join(stranded_labels))
    top_clip_margin_mm = float(bag_size_xyz_mm[2] - raw_box.max_xyz_mm[2])
    breakdown = _score_candidate(
        raw_box_local=raw_box,
        layer_z_mm=layer_z_mm,
        bag_size_xyz_mm=bag_size_xyz_mm,
        object_props=object_props,
        support_ratio=support_ratio,
        overhang_ratio=overhang_ratio,
        wall_contacts=wall_contacts,
        neighbor_contacts=neighbor_contacts,
        support_risk=support_risk,
        support_labels=support_labels,
        future_placeable_count=future_placeable_count,
        future_total_count=future_total_count,
        tallest_stranded_height_mm=tallest_stranded_height_mm,
        future_height_penalty=future_height_penalty,
        tallest_remaining_height_mm=tallest_remaining_height_mm,
        strands_tallest_low_slot=(layer_z_mm <= _EPS and not tallest_remaining_floor_feasible),
        top_clip_margin_mm=top_clip_margin_mm,
        weights=weights,
    )
    accepted = not reasons or (len(reasons) == 1 and reasons[0].startswith("future_stranding:"))
    return CandidatePlacement2p5D(
        raw_box_local=raw_box,
        padded_box_local=padded_box,
        yaw_deg=float(yaw_deg),
        orientation_label=orientation_label,
        layer_z_mm=float(layer_z_mm),
        score=float(breakdown.total),
        support_ratio=float(support_ratio),
        overhang_ratio=float(overhang_ratio),
        wall_contacts=wall_contacts,
        neighbor_contacts=neighbor_contacts,
        support_labels=support_labels,
        support_risk=float(support_risk),
        future_placeable_count=int(future_placeable_count),
        future_total_count=int(future_total_count),
        stranded_labels=tuple(stranded_labels),
        tallest_stranded_height_mm=float(tallest_stranded_height_mm),
        future_height_penalty=float(future_height_penalty),
        top_clip_margin_mm=float(top_clip_margin_mm),
        future_feasibility_used=bool(future_feasibility_used),
        accepted=accepted,
        reasons=tuple(reasons) if reasons else ("accepted",),
        breakdown=breakdown,
    )


def plan_bag_local_aabb_placement_bruteforce(
    *,
    item_label: str,
    raw_size_xyz_mm: np.ndarray,
    padding_min_xyz_mm: np.ndarray,
    padding_max_xyz_mm: np.ndarray,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    support_min_overlap_ratio: float = 0.72,
) -> PlacementPlan3D:
    raw_size = _as_vec3(raw_size_xyz_mm, "raw_size_xyz_mm")
    pad_min = _as_vec3(padding_min_xyz_mm, "padding_min_xyz_mm")
    pad_max = _as_vec3(padding_max_xyz_mm, "padding_max_xyz_mm")
    bag_size = _as_vec3(bag_size_xyz_mm, "bag_size_xyz_mm")

    x_candidates = {0.0}
    y_candidates = {0.0}
    z_candidates = {0.0}
    for other in placed_raw_boxes_local:
        x_candidates.update([float(other.min_xyz_mm[0]), float(other.max_xyz_mm[0])])
        y_candidates.update([float(other.min_xyz_mm[1]), float(other.max_xyz_mm[1])])
        z_candidates.add(float(other.max_xyz_mm[2]))

    attempts: list[CandidatePlacement2p5D] = []
    for z0, y0, x0 in product(_rounded_sorted(z_candidates), _rounded_sorted(y_candidates), _rounded_sorted(x_candidates)):
        cand = _evaluate_candidate(
            item_label=item_label,
            oriented_size_xyz_mm=raw_size,
            pad_min=pad_min,
            pad_max=pad_max,
            bag_size_xyz_mm=bag_size,
            placed_raw_boxes_local=placed_raw_boxes_local,
            placed_padded_boxes_local=placed_padded_boxes_local,
            object_props=_default_props(None),
            support_props_by_label={},
            yaw_deg=0.0,
            orientation_label="bruteforce",
            layer_z_mm=z0,
            x_mm=x0,
            y_mm=y0,
            weights=PlannerWeights(),
            support_min_overlap_ratio=support_min_overlap_ratio,
            max_overhang_ratio=1.0,
            descent_clearance_xy_mm=0.0,
            remaining_item_specs=[],
            enable_future_feasibility=False,
        )
        attempts.append(cand)
        if cand.accepted:
            return PlacementPlan3D(
                raw_box_local=cand.raw_box_local,
                padded_box_local=cand.padded_box_local,
                target_center_xy_mm=cand.raw_box_local.center_xyz_mm[:2].copy(),
                target_min_xyz_mm=cand.raw_box_local.min_xyz_mm.copy(),
                yaw_deg=0.0,
                orientation_label="bruteforce",
                score=cand.score,
                support_ratio=cand.support_ratio,
                layer_z_mm=cand.layer_z_mm,
                future_placeable_count=cand.future_placeable_count,
                future_total_count=cand.future_total_count,
                stranded_labels=cand.stranded_labels,
                tallest_stranded_height_mm=cand.tallest_stranded_height_mm,
                future_height_penalty=cand.future_height_penalty,
                top_clip_margin_mm=cand.top_clip_margin_mm,
                future_feasibility_used=cand.future_feasibility_used,
                candidates_evaluated=len(attempts),
                valid_candidates=1,
                attempts=attempts,
                notes="fallback brute-force placement",
            )
    raise RuntimeError(f"no brute-force placement for {item_label}")


def plan_bag_local_aabb_placement(
    *,
    item_label: str,
    raw_size_xyz_mm: np.ndarray,
    padding_min_xyz_mm: np.ndarray,
    padding_max_xyz_mm: np.ndarray,
    bag_size_xyz_mm: np.ndarray,
    placed_raw_boxes_local: list[AxisAlignedBox3D],
    placed_padded_boxes_local: list[AxisAlignedBox3D],
    object_properties: dict[str, Any] | None = None,
    support_properties_by_label: dict[str, dict[str, Any]] | None = None,
    weights: PlannerWeights | None = None,
    lower_layer_score_threshold: float = 0.10,
    support_min_overlap_ratio: float = 0.72,
    max_overhang_ratio: float = 0.28,
    descent_clearance_xy_mm: float = 0.0,
    remaining_item_specs: list[FutureItemSpec] | None = None,
    collect_debug_attempts: bool = False,
    allow_bruteforce_fallback: bool = True,
) -> PlacementPlan3D:
    raw_size = _as_vec3(raw_size_xyz_mm, "raw_size_xyz_mm")
    pad_min = _as_vec3(padding_min_xyz_mm, "padding_min_xyz_mm")
    pad_max = _as_vec3(padding_max_xyz_mm, "padding_max_xyz_mm")
    bag_size = _as_vec3(bag_size_xyz_mm, "bag_size_xyz_mm")
    if np.any(raw_size <= 0.0):
        raise ValueError("raw_size_xyz_mm must be positive")
    if np.any(pad_min < 0.0) or np.any(pad_max < 0.0):
        raise ValueError("padding must be non-negative")

    weights = weights or PlannerWeights()
    object_props = _default_props(object_properties)
    support_props_by_label = support_properties_by_label or {}
    remaining_item_specs = list(remaining_item_specs or [])
    attempts: list[CandidatePlacement2p5D] = []
    best_any_layer: CandidatePlacement2p5D | None = None
    candidates_evaluated = 0
    valid_candidates = 0
    use_two_stage_future = bool(remaining_item_specs) and not collect_debug_attempts

    layers = _layer_surfaces(float(bag_size[2]), float(raw_size[2]), placed_raw_boxes_local)
    for layer_z_mm in layers:
        layer_valid_base: list[CandidatePlacement2p5D] = []
        for oriented_size, yaw_deg, orientation_label in _orientation_options(raw_size):
            for x_mm, y_mm in _candidate_xy_positions(
                oriented_size_xyz_mm=oriented_size,
                bag_size_xyz_mm=bag_size,
                layer_z_mm=layer_z_mm,
                placed_raw_boxes_local=placed_raw_boxes_local,
                object_props=object_props,
            ):
                cand = _evaluate_candidate(
                    item_label=item_label,
                    oriented_size_xyz_mm=oriented_size,
                    pad_min=pad_min,
                    pad_max=pad_max,
                    bag_size_xyz_mm=bag_size,
                    placed_raw_boxes_local=placed_raw_boxes_local,
                    placed_padded_boxes_local=placed_padded_boxes_local,
                    object_props=object_props,
                    support_props_by_label=support_props_by_label,
                    yaw_deg=yaw_deg,
                    orientation_label=orientation_label,
                    layer_z_mm=layer_z_mm,
                    x_mm=x_mm,
                    y_mm=y_mm,
                    weights=weights,
                    support_min_overlap_ratio=support_min_overlap_ratio,
                    max_overhang_ratio=max_overhang_ratio,
                    descent_clearance_xy_mm=descent_clearance_xy_mm,
                    remaining_item_specs=remaining_item_specs,
                    enable_future_feasibility=not use_two_stage_future,
                )
                candidates_evaluated += 1
                if collect_debug_attempts or cand.accepted:
                    attempts.append(cand)
                if not cand.accepted:
                    continue
                valid_candidates += 1
                layer_valid_base.append(cand)

        if not layer_valid_base:
            continue

        if use_two_stage_future:
            layer_valid_base.sort(key=lambda c: (-c.score, c.layer_z_mm, c.raw_box_local.min_xyz_mm[1], c.raw_box_local.min_xyz_mm[0]))
            layer_valid: list[CandidatePlacement2p5D] = []
            for base_cand in layer_valid_base[:_FUTURE_REFINEMENT_CANDIDATES_PER_LAYER]:
                refined = _evaluate_candidate(
                    item_label=item_label,
                    oriented_size_xyz_mm=base_cand.raw_box_local.size_xyz_mm.copy(),
                    pad_min=pad_min,
                    pad_max=pad_max,
                    bag_size_xyz_mm=bag_size,
                    placed_raw_boxes_local=placed_raw_boxes_local,
                    placed_padded_boxes_local=placed_padded_boxes_local,
                    object_props=object_props,
                    support_props_by_label=support_props_by_label,
                    yaw_deg=base_cand.yaw_deg,
                    orientation_label=base_cand.orientation_label,
                    layer_z_mm=base_cand.layer_z_mm,
                    x_mm=float(base_cand.raw_box_local.min_xyz_mm[0]),
                    y_mm=float(base_cand.raw_box_local.min_xyz_mm[1]),
                    weights=weights,
                    support_min_overlap_ratio=support_min_overlap_ratio,
                    max_overhang_ratio=max_overhang_ratio,
                    descent_clearance_xy_mm=descent_clearance_xy_mm,
                    remaining_item_specs=remaining_item_specs,
                    enable_future_feasibility=True,
                )
                candidates_evaluated += 1
                if collect_debug_attempts or refined.accepted:
                    attempts.append(refined)
                if not refined.accepted:
                    continue
                layer_valid.append(refined)
                if best_any_layer is None or refined.score > best_any_layer.score + _EPS:
                    best_any_layer = refined
            if not layer_valid:
                layer_valid = layer_valid_base[:1]
                if best_any_layer is None or layer_valid[0].score > best_any_layer.score + _EPS:
                    best_any_layer = layer_valid[0]
        else:
            layer_valid = layer_valid_base
            for cand in layer_valid:
                if best_any_layer is None or cand.score > best_any_layer.score + _EPS:
                    best_any_layer = cand


        layer_valid.sort(key=lambda c: (-c.score, c.layer_z_mm, c.raw_box_local.min_xyz_mm[1], c.raw_box_local.min_xyz_mm[0]))
        best_layer = layer_valid[0]
        if layer_z_mm <= _EPS or best_layer.score >= lower_layer_score_threshold:
            return PlacementPlan3D(
                raw_box_local=best_layer.raw_box_local,
                padded_box_local=best_layer.padded_box_local,
                target_center_xy_mm=best_layer.raw_box_local.center_xyz_mm[:2].copy(),
                target_min_xyz_mm=best_layer.raw_box_local.min_xyz_mm.copy(),
                yaw_deg=best_layer.yaw_deg,
                orientation_label=best_layer.orientation_label,
                score=best_layer.score,
                support_ratio=best_layer.support_ratio,
                layer_z_mm=best_layer.layer_z_mm,
                future_placeable_count=best_layer.future_placeable_count,
                future_total_count=best_layer.future_total_count,
                stranded_labels=best_layer.stranded_labels,
                tallest_stranded_height_mm=best_layer.tallest_stranded_height_mm,
                future_height_penalty=best_layer.future_height_penalty,
                top_clip_margin_mm=best_layer.top_clip_margin_mm,
                future_feasibility_used=best_layer.future_feasibility_used,
                candidates_evaluated=candidates_evaluated,
                valid_candidates=valid_candidates,
                attempts=attempts,
                notes=f"2.5D planner selected layer={layer_z_mm:.1f} yaw={best_layer.yaw_deg:.0f} score={best_layer.score:.3f}",
            )

    if best_any_layer is not None:
        return PlacementPlan3D(
            raw_box_local=best_any_layer.raw_box_local,
            padded_box_local=best_any_layer.padded_box_local,
            target_center_xy_mm=best_any_layer.raw_box_local.center_xyz_mm[:2].copy(),
            target_min_xyz_mm=best_any_layer.raw_box_local.min_xyz_mm.copy(),
            yaw_deg=best_any_layer.yaw_deg,
            orientation_label=best_any_layer.orientation_label,
            score=best_any_layer.score,
            support_ratio=best_any_layer.support_ratio,
            layer_z_mm=best_any_layer.layer_z_mm,
            future_placeable_count=best_any_layer.future_placeable_count,
            future_total_count=best_any_layer.future_total_count,
            stranded_labels=best_any_layer.stranded_labels,
            tallest_stranded_height_mm=best_any_layer.tallest_stranded_height_mm,
            future_height_penalty=best_any_layer.future_height_penalty,
            top_clip_margin_mm=best_any_layer.top_clip_margin_mm,
            future_feasibility_used=best_any_layer.future_feasibility_used,
            candidates_evaluated=candidates_evaluated,
            valid_candidates=valid_candidates,
            attempts=attempts,
            notes=f"2.5D planner fallback-selected higher layer yaw={best_any_layer.yaw_deg:.0f} score={best_any_layer.score:.3f}",
        )

    if allow_bruteforce_fallback:
        return plan_bag_local_aabb_placement_bruteforce(
            item_label=item_label,
            raw_size_xyz_mm=raw_size,
            padding_min_xyz_mm=pad_min,
            padding_max_xyz_mm=pad_max,
            bag_size_xyz_mm=bag_size,
            placed_raw_boxes_local=placed_raw_boxes_local,
            placed_padded_boxes_local=placed_padded_boxes_local,
            support_min_overlap_ratio=support_min_overlap_ratio,
        )

    raise RuntimeError(f"no valid 2.5D placement for {item_label} after {candidates_evaluated} candidates")
