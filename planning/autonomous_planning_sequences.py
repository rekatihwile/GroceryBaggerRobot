from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from planning.aabb_utils import AxisAlignedBox3D, make_aabb_from_min_max
from planning.bag_local_3d_aabb_planner import (
    FutureItemSpec,
    PlacementPlan3D,
    plan_bag_local_aabb_placement,
)


BAG_LOCAL_AABB_JOINT = "bag_local_aabb_joint"
VOLUME_TOPDOWN_SAFE = "volume_topdown_safe"
FOUNDATION_FLOOR_FUTURE_AWARE = "foundation_floor_future_aware"

# When every strict placement fails, retry with this XY/Z overflow tolerance (mm).
# Items placed in desperate mode may extend slightly outside the bag boundary.
_DESPERATE_OVERFLOW_MM = 15.0
AVAILABLE_PLANNING_SEQUENCES = (
    BAG_LOCAL_AABB_JOINT,
    VOLUME_TOPDOWN_SAFE,
    FOUNDATION_FLOOR_FUTURE_AWARE,
)

_FOUNDATION_MIN_SUPPORT_RATIO = 0.20  # fraction of item footprint that must rest on something
_FOUNDATION_SUPPORT_Z_TOL_MM = 2.0    # z-tolerance when finding supporting boxes


@dataclass(frozen=True)
class PlanningSequenceContext:
    object_i: int
    robot: Any
    placed_boxes: list[Any]
    config: Any
    surface_zone: dict[str, Any]
    base_xy: np.ndarray
    base_phi_deg: float
    target_limit: int
    pad_xyz_mm: np.ndarray
    bag_height_mm: float


@dataclass(frozen=True)
class PlanningRuntime:
    choose_best_candidate: Callable[..., Any]
    aabb_from_object_candidate: Callable[..., Any]


@dataclass
class PlanningOverlayEntry:
    can_place: bool
    target_xy_mm: np.ndarray | None
    target_phi_deg: float | None
    clearance_mm: float | None
    reason: str


@dataclass
class SequenceTargetPlan:
    target_xy_mm: np.ndarray
    target_phi_deg: float
    footprint_clearance_mm: float
    aabb_clearance_mm: float
    fit_clearance_mm: float
    nudge_xy_mm: np.ndarray
    can_place: bool
    reason: str
    planner_score: float | None = None
    planner_yaw_deg: float | None = None
    planner_layer_z_mm: float | None = None
    future_placeable_count: int | None = None
    future_total_count: int | None = None


@dataclass
class PlanningSelection:
    result: Any
    overlay: dict[int, PlanningOverlayEntry]
    target_cache: dict[int, SequenceTargetPlan]


def list_planning_sequences() -> tuple[str, ...]:
    return AVAILABLE_PLANNING_SEQUENCES


def validate_planning_sequence_name(sequence_name: str) -> str:
    value = str(sequence_name or "").strip().lower()
    if value not in AVAILABLE_PLANNING_SEQUENCES:
        raise ValueError(
            f"unknown planning sequence {sequence_name!r}; "
            f"expected one of {', '.join(AVAILABLE_PLANNING_SEQUENCES)}"
        )
    return value


def _decision_for_candidate(result: Any, cand_dbg: Any) -> Any | None:
    return next((d for d in getattr(result, "decisions", []) if d.dbg is cand_dbg), None)


def _normalize_phi_deg(phi_deg: float) -> float:
    value = float(phi_deg)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _bag_local_frame(ctx: PlanningSequenceContext) -> tuple[np.ndarray, np.ndarray]:
    center_xy = np.asarray(ctx.surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    half_wh = np.array(
        [
            0.5 * float(ctx.surface_zone.get("width_mm", 290.0)),
            0.5 * float(ctx.surface_zone.get("depth_mm", 175.0)),
        ],
        dtype=np.float64,
    )
    bag_min_xyz = np.array(
        [float(center_xy[0] - half_wh[0]), float(center_xy[1] - half_wh[1]), float(ctx.surface_zone["surface_z_mm"])],
        dtype=np.float64,
    )
    bag_size_xyz = np.array(
        [
            float(ctx.surface_zone.get("width_mm", 290.0)),
            float(ctx.surface_zone.get("depth_mm", 175.0)),
            float(ctx.bag_height_mm),
        ],
        dtype=np.float64,
    )
    return bag_min_xyz, bag_size_xyz


def _box_to_bag_local(box: Any, *, bag_min_xyz: np.ndarray, label: str) -> Any:
    return make_aabb_from_min_max(
        np.asarray(box.min_xyz_mm, dtype=np.float64) - bag_min_xyz,
        np.asarray(box.max_xyz_mm, dtype=np.float64) - bag_min_xyz,
        label=label,
    )


def _box_xy_clearance_mm(box_local: Any, bag_size_xyz: np.ndarray) -> float:
    return float(
        min(
            float(box_local.min_xyz_mm[0]),
            float(box_local.min_xyz_mm[1]),
            float(bag_size_xyz[0] - box_local.max_xyz_mm[0]),
            float(bag_size_xyz[1] - box_local.max_xyz_mm[1]),
            float(bag_size_xyz[2] - box_local.max_xyz_mm[2]),
        )
    )


def _candidate_priority(cand_dbg: Any) -> float:
    candidate = getattr(cand_dbg, "candidate", None)
    volume_mm3 = float(np.prod(np.asarray(getattr(candidate, "bbox_size_mm", [0.0, 0.0, 0.0]), dtype=np.float64)))
    return volume_mm3


def _padded_size(raw_size_xyz_mm: np.ndarray, pad_xyz_mm: np.ndarray) -> np.ndarray:
    return np.asarray(raw_size_xyz_mm, dtype=np.float64).reshape(3) + 2.0 * np.asarray(pad_xyz_mm, dtype=np.float64).reshape(3)


def _safe_layer_z_values(
    *,
    item_height_mm: float,
    bag_height_mm: float,
    placed_raw_boxes_local: list[Any],
) -> list[float]:
    values = {0.0}
    for box in placed_raw_boxes_local:
        z = float(box.max_xyz_mm[2])
        if z + float(item_height_mm) <= float(bag_height_mm) + 1e-6:
            values.add(round(z, 6))
    return sorted(values)


def _safe_xy_starts(
    *,
    padded_size_xy_mm: np.ndarray,
    bag_size_xy_mm: np.ndarray,
    placed_padded_boxes_local: list[Any],
) -> list[tuple[float, float]]:
    sx, sy = [float(v) for v in np.asarray(padded_size_xy_mm, dtype=np.float64).reshape(2)]
    bx, by = [float(v) for v in np.asarray(bag_size_xy_mm, dtype=np.float64).reshape(2)]
    max_x = bx - sx
    max_y = by - sy
    if max_x < -1e-6 or max_y < -1e-6:
        return []

    xs = {0.0, max(0.0, max_x)}
    ys = {0.0, max(0.0, max_y)}
    for box in placed_padded_boxes_local:
        xs.update([float(box.min_xyz_mm[0]), float(box.max_xyz_mm[0]), float(box.max_xyz_mm[0] - sx)])
        ys.update([float(box.min_xyz_mm[1]), float(box.max_xyz_mm[1]), float(box.max_xyz_mm[1] - sy)])

    starts: set[tuple[float, float]] = set()
    for x in xs:
        for y in ys:
            x = float(min(max(x, 0.0), max(0.0, max_x)))
            y = float(min(max(y, 0.0), max(0.0, max_y)))
            starts.add((round(x, 6), round(y, 6)))

    bag_center_start = np.array([0.5 * max(0.0, max_x), 0.5 * max(0.0, max_y)], dtype=np.float64)
    return sorted(
        starts,
        key=lambda xy: (
            float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - bag_center_start)),
            xy[1],
            xy[0],
        ),
    )


def _overflow_xy_starts(
    *,
    padded_size_xy_mm: np.ndarray,
    bag_size_xy_mm: np.ndarray,
    placed_padded_boxes_local: list[Any],
    overflow_mm: float,
) -> list[tuple[float, float]]:
    """Like _safe_xy_starts but allows the padded box to extend up to overflow_mm outside
    the bag boundary.  Used only when strict placement is impossible."""
    sx, sy = [float(v) for v in np.asarray(padded_size_xy_mm, dtype=np.float64).reshape(2)]
    bx, by = [float(v) for v in np.asarray(bag_size_xy_mm, dtype=np.float64).reshape(2)]
    if sx > bx + overflow_mm + 1e-6 or sy > by + overflow_mm + 1e-6:
        return []
    max_x = bx - sx  # may be negative (item too wide)
    max_y = by - sy
    if max_x >= -1e-6 and max_y >= -1e-6:
        return _safe_xy_starts(
            padded_size_xy_mm=padded_size_xy_mm,
            bag_size_xy_mm=bag_size_xy_mm,
            placed_padded_boxes_local=placed_padded_boxes_local,
        )
    # Item overflows in at least one dimension — generate a small set of
    # candidate positions centered in the bag to minimise total overflow.
    starts: set[tuple[float, float]] = set()
    cx = round(float(max_x / 2.0), 6)  # negative/2 when item overflows
    cy = round(float(max_y / 2.0), 6)
    starts.add((cx, cy))
    for tx in [round(max(-(overflow_mm), min(0.0, max_x)), 6), round(max(0.0, max_x), 6)]:
        starts.add((tx, cy))
    for ty in [round(max(-(overflow_mm), min(0.0, max_y)), 6), round(max(0.0, max_y), 6)]:
        starts.add((cx, ty))

    def _overflow_score(xy: tuple[float, float]) -> float:
        x, y = xy
        return (max(0.0, -x) + max(0.0, x + sx - bx)
                + max(0.0, -y) + max(0.0, y + sy - by))

    return sorted(starts, key=_overflow_score)


def _intersects_any(box: Any, boxes: list[Any]) -> bool:
    for other in boxes:
        if (
            min(float(box.max_xyz_mm[0]), float(other.max_xyz_mm[0])) - max(float(box.min_xyz_mm[0]), float(other.min_xyz_mm[0])) > 1e-6
            and min(float(box.max_xyz_mm[1]), float(other.max_xyz_mm[1])) - max(float(box.min_xyz_mm[1]), float(other.min_xyz_mm[1])) > 1e-6
            and min(float(box.max_xyz_mm[2]), float(other.max_xyz_mm[2])) - max(float(box.min_xyz_mm[2]), float(other.min_xyz_mm[2])) > 1e-6
        ):
            return True
    return False


@dataclass
class _FoundationPlacement:
    is_floor: bool
    layer_z_mm: float
    yaw_deg: float
    orientation_label: str
    fit_clearance_mm: float
    raw_box_local: AxisAlignedBox3D
    padded_box_local: AxisAlignedBox3D


def _foundation_support_ok(
    raw_local: AxisAlignedBox3D,
    placed_raw_boxes_local: list[Any],
) -> bool:
    """Return True if raw_local has adequate XY support from boxes directly below it."""
    layer_z = float(raw_local.min_xyz_mm[2])
    if abs(layer_z) <= 1e-6:
        return True
    item_area = max(1e-6, float(raw_local.size_xyz_mm[0]) * float(raw_local.size_xyz_mm[1]))
    support_area = 0.0
    for box in placed_raw_boxes_local:
        if abs(float(box.max_xyz_mm[2]) - layer_z) > _FOUNDATION_SUPPORT_Z_TOL_MM:
            continue
        ox_min = max(float(raw_local.min_xyz_mm[0]), float(box.min_xyz_mm[0]))
        ox_max = min(float(raw_local.max_xyz_mm[0]), float(box.max_xyz_mm[0]))
        oy_min = max(float(raw_local.min_xyz_mm[1]), float(box.min_xyz_mm[1]))
        oy_max = min(float(raw_local.max_xyz_mm[1]), float(box.max_xyz_mm[1]))
        support_area += max(0.0, ox_max - ox_min) * max(0.0, oy_max - oy_min)
    return (support_area / item_area) >= _FOUNDATION_MIN_SUPPORT_RATIO


def _foundation_find_best_placement(
    raw_box: AxisAlignedBox3D,
    *,
    ctx: PlanningSequenceContext,
    placed_raw_boxes_local: list[Any],
    placed_padded_boxes_local: list[Any],
    bag_min_xyz: np.ndarray,
    bag_size_xyz: np.ndarray,
) -> _FoundationPlacement | None:
    """Find the best bag-local placement for raw_box, preferring floor over stack."""
    raw_size = np.asarray(raw_box.size_xyz_mm, dtype=np.float64).reshape(3)
    pad = np.asarray(ctx.pad_xyz_mm, dtype=np.float64).reshape(3)

    orientations: list[tuple[np.ndarray, float, str]] = [
        (raw_size.copy(), 0.0, "yaw0")
    ]
    if abs(float(raw_size[0] - raw_size[1])) > max(5.0, 0.05 * max(float(raw_size[0]), float(raw_size[1]))):
        orientations.append(
            (np.array([raw_size[1], raw_size[0], raw_size[2]], dtype=np.float64), 90.0, "yaw90")
        )

    # (fit_clearance, yaw_deg, orientation_label, layer_z, raw_local, padded_box)
    best_floor: tuple | None = None
    best_stack: tuple | None = None

    for oriented_size, yaw_deg, orientation_label in orientations:
        padded_size = _padded_size(oriented_size, pad)
        if padded_size[0] > bag_size_xyz[0] + 1e-6 or padded_size[1] > bag_size_xyz[1] + 1e-6:
            continue

        for layer_z in _safe_layer_z_values(
            item_height_mm=float(padded_size[2]),
            bag_height_mm=float(bag_size_xyz[2]),
            placed_raw_boxes_local=placed_raw_boxes_local,
        ):
            is_floor = abs(float(layer_z)) <= 1e-6
            xy_starts = _safe_xy_starts(
                padded_size_xy_mm=padded_size[:2],
                bag_size_xy_mm=bag_size_xyz[:2],
                placed_padded_boxes_local=placed_padded_boxes_local,
            )
            for padded_min_x, padded_min_y in xy_starts:
                padded_min = np.array([padded_min_x, padded_min_y, float(layer_z)], dtype=np.float64)
                padded_box = make_aabb_from_min_max(
                    padded_min,
                    padded_min + padded_size,
                    label="fnd_padded_local",
                )
                raw_min = padded_min + pad
                raw_local = make_aabb_from_min_max(
                    raw_min,
                    raw_min + oriented_size,
                    label="fnd_raw_local",
                )
                if padded_box.max_xyz_mm[2] > bag_size_xyz[2] + 1e-6:
                    continue
                if _intersects_any(padded_box, placed_padded_boxes_local):
                    continue
                if _intersects_any(raw_local, placed_raw_boxes_local):
                    continue
                if not is_floor and not _foundation_support_ok(raw_local, placed_raw_boxes_local):
                    continue

                fit_clearance = _box_xy_clearance_mm(padded_box, bag_size_xyz)
                entry = (fit_clearance, yaw_deg, orientation_label, layer_z, raw_local, padded_box)

                if is_floor:
                    if best_floor is None or fit_clearance > best_floor[0]:
                        best_floor = entry
                else:
                    if best_stack is None or fit_clearance > best_stack[0]:
                        best_stack = entry

    best = best_floor if best_floor is not None else best_stack
    if best is None:
        return None

    fit_clearance, yaw_deg, orientation_label, layer_z, raw_local, padded_box = best
    return _FoundationPlacement(
        is_floor=(best is best_floor),
        layer_z_mm=float(layer_z),
        yaw_deg=float(yaw_deg),
        orientation_label=str(orientation_label),
        fit_clearance_mm=float(fit_clearance),
        raw_box_local=raw_local,
        padded_box_local=padded_box,
    )


def _compute_volume_topdown_safe_target(
    cand_dbg: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
    overflow_mm: float = 0.0,
) -> SequenceTargetPlan:
    raw_box = runtime.aabb_from_object_candidate(
        cand_dbg.candidate,
        default_label=f"object{ctx.object_i}_volume_safe",
    )
    bag_min_xyz, bag_size_xyz = _bag_local_frame(ctx)
    placed_raw_boxes_local = [
        _box_to_bag_local(box.raw_box, bag_min_xyz=bag_min_xyz, label=f"{box.raw_box.label}_local")
        for box in ctx.placed_boxes
    ]
    placed_padded_boxes_local = [
        _box_to_bag_local(box.padded_box, bag_min_xyz=bag_min_xyz, label=f"{box.padded_box.label}_local")
        for box in ctx.placed_boxes
    ]

    raw_size = np.asarray(raw_box.size_xyz_mm, dtype=np.float64).reshape(3)
    pad = np.asarray(ctx.pad_xyz_mm, dtype=np.float64).reshape(3)
    orientations = [(raw_size.copy(), 0.0, "safe_yaw0")]
    if abs(float(raw_size[0] - raw_size[1])) > max(5.0, 0.05 * max(float(raw_size[0]), float(raw_size[1]))):
        orientations.append((np.array([raw_size[1], raw_size[0], raw_size[2]], dtype=np.float64), 90.0, "safe_yaw90"))

    best: tuple[float, float, float, np.ndarray, np.ndarray, float, str, Any, Any] | None = None
    for oriented_size, yaw_deg, orientation_label in orientations:
        padded_size = _padded_size(oriented_size, pad)
        if padded_size[0] > bag_size_xyz[0] + overflow_mm + 1e-6 or padded_size[1] > bag_size_xyz[1] + overflow_mm + 1e-6:
            continue
        for layer_z in _safe_layer_z_values(
            item_height_mm=float(padded_size[2]),
            bag_height_mm=float(bag_size_xyz[2]),
            placed_raw_boxes_local=placed_raw_boxes_local,
        ):
            if overflow_mm > 0.0:
                xy_starts = _overflow_xy_starts(
                    padded_size_xy_mm=padded_size[:2],
                    bag_size_xy_mm=bag_size_xyz[:2],
                    placed_padded_boxes_local=placed_padded_boxes_local,
                    overflow_mm=overflow_mm,
                )
            else:
                xy_starts = _safe_xy_starts(
                    padded_size_xy_mm=padded_size[:2],
                    bag_size_xy_mm=bag_size_xyz[:2],
                    placed_padded_boxes_local=placed_padded_boxes_local,
                )
            for padded_min_x, padded_min_y in xy_starts:
                padded_min = np.array([padded_min_x, padded_min_y, float(layer_z)], dtype=np.float64)
                padded_box = make_aabb_from_min_max(
                    padded_min,
                    padded_min + padded_size,
                    label=f"{raw_box.label}_safe_padded_local",
                )
                raw_min = padded_min + pad
                raw_local = make_aabb_from_min_max(
                    raw_min,
                    raw_min + oriented_size,
                    label=f"{raw_box.label}_safe_raw_local",
                )
                if padded_box.max_xyz_mm[2] > bag_size_xyz[2] + overflow_mm + 1e-6:
                    continue
                if _intersects_any(padded_box, placed_padded_boxes_local) or _intersects_any(raw_local, placed_raw_boxes_local):
                    continue
                center_bias = float(np.linalg.norm(raw_local.center_xyz_mm[:2] - 0.5 * bag_size_xyz[:2]))
                score = (
                    -float(layer_z),
                    -center_bias,
                    _box_xy_clearance_mm(padded_box, bag_size_xyz),
                )
                candidate = (
                    score[0],
                    score[1],
                    score[2],
                    raw_local.center_xyz_mm[:2].copy(),
                    padded_min.copy(),
                    float(yaw_deg),
                    orientation_label,
                    raw_local,
                    padded_box,
                )
                if best is None or score > best[:3]:
                    best = candidate

    if best is None:
        raise RuntimeError(f"no top-down safe footprint for {getattr(cand_dbg.candidate.yolo, 'class_name', raw_box.label)}")

    _score_z, _score_center, fit_clearance, local_center_xy, _padded_min, yaw_deg, orientation_label, raw_local, padded_box = best
    target_center_xy = bag_min_xyz[:2] + np.asarray(local_center_xy, dtype=np.float64).reshape(2)
    target_phi = _normalize_phi_deg(float(ctx.base_phi_deg) + float(yaw_deg))
    layer_z = float(raw_local.min_xyz_mm[2])
    return SequenceTargetPlan(
        target_xy_mm=target_center_xy.copy(),
        target_phi_deg=target_phi,
        footprint_clearance_mm=float(fit_clearance),
        aabb_clearance_mm=float(fit_clearance),
        fit_clearance_mm=float(fit_clearance),
        nudge_xy_mm=np.zeros(2, dtype=np.float64),
        can_place=True,
        reason="volume_topdown_safe",
        planner_score=float(_candidate_priority(cand_dbg)),
        planner_yaw_deg=float(yaw_deg),
        planner_layer_z_mm=float(layer_z),
        future_placeable_count=None,
        future_total_count=None,
    )


def _build_future_item_specs(
    state: Any,
    *,
    skip_candidate: Any,
    result: Any,
    runtime: PlanningRuntime,
    ctx: PlanningSequenceContext,
) -> list[FutureItemSpec]:
    future: list[FutureItemSpec] = []
    for cand_dbg in getattr(state, "candidates", []):
        if cand_dbg is skip_candidate:
            continue
        decision = _decision_for_candidate(result, cand_dbg)
        if decision is not None and not bool(decision.passed):
            continue
        try:
            raw_box = runtime.aabb_from_object_candidate(
                cand_dbg.candidate,
                default_label=f"future_{getattr(cand_dbg.candidate, 'index', len(future))}",
            )
        except Exception:
            continue
        future.append(
            FutureItemSpec(
                label=str(getattr(cand_dbg.candidate, "yolo", None).class_name if getattr(cand_dbg.candidate, "yolo", None) is not None else getattr(cand_dbg.candidate, "index", len(future))),
                raw_size_xyz_mm=np.asarray(raw_box.size_xyz_mm, dtype=np.float64).copy(),
                padding_min_xyz_mm=np.asarray(ctx.pad_xyz_mm, dtype=np.float64).copy(),
                padding_max_xyz_mm=np.asarray(ctx.pad_xyz_mm, dtype=np.float64).copy(),
                object_props={"weight": max(0.1, _candidate_priority(cand_dbg) / 1_000_000.0)},
            )
        )
    return future


def _compute_bag_local_target(
    cand_dbg: Any,
    *,
    state: Any,
    result: Any,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> SequenceTargetPlan:
    raw_box = runtime.aabb_from_object_candidate(
        cand_dbg.candidate,
        default_label=f"object{ctx.object_i}_bag_local",
    )
    bag_min_xyz, bag_size_xyz = _bag_local_frame(ctx)
    placed_raw_boxes_local = [
        _box_to_bag_local(box.raw_box, bag_min_xyz=bag_min_xyz, label=f"{box.raw_box.label}_local")
        for box in ctx.placed_boxes
    ]
    placed_padded_boxes_local = [
        _box_to_bag_local(box.padded_box, bag_min_xyz=bag_min_xyz, label=f"{box.padded_box.label}_local")
        for box in ctx.placed_boxes
    ]
    future_specs = _build_future_item_specs(
        state,
        skip_candidate=cand_dbg,
        result=result,
        runtime=runtime,
        ctx=ctx,
    )
    plan: PlacementPlan3D = plan_bag_local_aabb_placement(
        item_label=str(getattr(cand_dbg.candidate.yolo, "class_name", getattr(cand_dbg.candidate, "index", ctx.object_i))),
        raw_size_xyz_mm=np.asarray(raw_box.size_xyz_mm, dtype=np.float64),
        padding_min_xyz_mm=np.asarray(ctx.pad_xyz_mm, dtype=np.float64),
        padding_max_xyz_mm=np.asarray(ctx.pad_xyz_mm, dtype=np.float64),
        bag_size_xyz_mm=bag_size_xyz,
        placed_raw_boxes_local=placed_raw_boxes_local,
        placed_padded_boxes_local=placed_padded_boxes_local,
        remaining_item_specs=future_specs,
        collect_debug_attempts=False,
    )
    target_center_xy = bag_min_xyz[:2] + np.asarray(plan.target_center_xy_mm, dtype=np.float64).reshape(2)
    target_phi = _normalize_phi_deg(float(ctx.base_phi_deg) + float(plan.yaw_deg))
    fit_clearance = _box_xy_clearance_mm(plan.padded_box_local, bag_size_xyz)
    return SequenceTargetPlan(
        target_xy_mm=target_center_xy.copy(),
        target_phi_deg=target_phi,
        footprint_clearance_mm=float(fit_clearance),
        aabb_clearance_mm=float(fit_clearance),
        fit_clearance_mm=float(fit_clearance),
        nudge_xy_mm=np.zeros(2, dtype=np.float64),
        can_place=True,
        reason=str(plan.notes or "bag_local_aabb"),
        planner_score=float(plan.score),
        planner_yaw_deg=float(plan.yaw_deg),
        planner_layer_z_mm=float(plan.layer_z_mm),
        future_placeable_count=int(plan.future_placeable_count),
        future_total_count=int(plan.future_total_count),
    )


def _compute_foundation_floor_target(
    cand_dbg: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> SequenceTargetPlan:
    raw_box = runtime.aabb_from_object_candidate(
        cand_dbg.candidate,
        default_label=f"object{ctx.object_i}_foundation",
    )
    bag_min_xyz, bag_size_xyz = _bag_local_frame(ctx)
    placed_raw_boxes_local = [
        _box_to_bag_local(box.raw_box, bag_min_xyz=bag_min_xyz, label=f"{box.raw_box.label}_local")
        for box in ctx.placed_boxes
    ]
    placed_padded_boxes_local = [
        _box_to_bag_local(box.padded_box, bag_min_xyz=bag_min_xyz, label=f"{box.padded_box.label}_local")
        for box in ctx.placed_boxes
    ]
    placement = _foundation_find_best_placement(
        raw_box,
        ctx=ctx,
        placed_raw_boxes_local=placed_raw_boxes_local,
        placed_padded_boxes_local=placed_padded_boxes_local,
        bag_min_xyz=bag_min_xyz,
        bag_size_xyz=bag_size_xyz,
    )
    if placement is None:
        raise RuntimeError(
            f"foundation_floor_future_aware: no valid placement for "
            f"{str(getattr(getattr(cand_dbg.candidate, 'yolo', None), 'class_name', None) or 'unknown')}"
        )
    target_center_xy = bag_min_xyz[:2] + np.asarray(
        placement.raw_box_local.center_xyz_mm[:2], dtype=np.float64
    )
    target_phi = _normalize_phi_deg(float(ctx.base_phi_deg) + float(placement.yaw_deg))
    return SequenceTargetPlan(
        target_xy_mm=target_center_xy.copy(),
        target_phi_deg=target_phi,
        footprint_clearance_mm=float(placement.fit_clearance_mm),
        aabb_clearance_mm=float(placement.fit_clearance_mm),
        fit_clearance_mm=float(placement.fit_clearance_mm),
        nudge_xy_mm=np.zeros(2, dtype=np.float64),
        can_place=True,
        reason=(
            f"foundation_floor_future_aware: floor={placement.is_floor} "
            f"yaw={placement.yaw_deg:.0f} z={placement.layer_z_mm:.1f}"
        ),
        planner_score=None,
        planner_yaw_deg=float(placement.yaw_deg),
        planner_layer_z_mm=float(placement.layer_z_mm),
        future_placeable_count=None,
        future_total_count=None,
    )


def _select_foundation_floor_future_aware(
    state: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> PlanningSelection:
    from planning.grocery_properties import (
        foundation_strength_score,
        load_grocery_specs,
        properties_for_class,
    )

    specs = load_grocery_specs()
    result = runtime.choose_best_candidate(
        state, config=ctx.config, robot=ctx.robot, placed_boxes=ctx.placed_boxes
    )
    result.print_debug("[BEST]")

    overlay: dict[int, PlanningOverlayEntry] = {}
    target_cache: dict[int, SequenceTargetPlan] = {}

    if not getattr(state, "candidates", None):
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    bag_min_xyz, bag_size_xyz = _bag_local_frame(ctx)
    placed_raw_boxes_local = [
        _box_to_bag_local(box.raw_box, bag_min_xyz=bag_min_xyz, label=f"{box.raw_box.label}_local")
        for box in ctx.placed_boxes
    ]
    placed_padded_boxes_local = [
        _box_to_bag_local(box.padded_box, bag_min_xyz=bag_min_xyz, label=f"{box.padded_box.label}_local")
        for box in ctx.placed_boxes
    ]

    # ── phase 1: compute placement for every base-passed candidate ──────────
    # (idx, cand_dbg, decision, raw_box, placement, props, strength)
    candidate_entries: list[tuple] = []
    for idx, cand_dbg in enumerate(state.candidates):
        decision = _decision_for_candidate(result, cand_dbg)
        if decision is None or not bool(decision.passed):
            reject_reason = "selector_rejected"
            if decision is not None and getattr(decision, "reject_reasons", None):
                reject_reason = ";".join(list(decision.reject_reasons[:2]))
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, reject_reason)
            continue
        try:
            raw_box = runtime.aabb_from_object_candidate(
                cand_dbg.candidate,
                default_label=f"object{ctx.object_i}_foundation_{idx}",
            )
        except Exception as exc:
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, f"aabb_failed:{exc}")
            continue

        placement = _foundation_find_best_placement(
            raw_box,
            ctx=ctx,
            placed_raw_boxes_local=placed_raw_boxes_local,
            placed_padded_boxes_local=placed_padded_boxes_local,
            bag_min_xyz=bag_min_xyz,
            bag_size_xyz=bag_size_xyz,
        )
        if placement is None:
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, "no_valid_placement")
            continue

        class_name = str(
            getattr(getattr(cand_dbg.candidate, "yolo", None), "class_name", None) or ""
        )
        props = properties_for_class(class_name, specs)
        strength = foundation_strength_score(props)
        candidate_entries.append((idx, cand_dbg, decision, raw_box, placement, props, strength))

    # ── desperate fallback: nothing placed by foundation logic ──────────────
    if not candidate_entries:
        for idx, cand_dbg in enumerate(state.candidates):
            decision = _decision_for_candidate(result, cand_dbg)
            if decision is None or not bool(decision.passed):
                continue
            try:
                target = _compute_volume_topdown_safe_target(
                    cand_dbg, ctx=ctx, runtime=runtime, overflow_mm=_DESPERATE_OVERFLOW_MM
                )
                target_cache[idx] = target
                overlay[idx] = PlanningOverlayEntry(
                    can_place=True,
                    target_xy_mm=target.target_xy_mm.copy(),
                    target_phi_deg=float(target.target_phi_deg),
                    clearance_mm=float(target.fit_clearance_mm),
                    reason=f"desperate:{target.reason}",
                )
                result.selected = cand_dbg
                result.selected_decision = decision
                state.selected_index = state.candidates.index(cand_dbg)
                print(f"[FOUNDATION PLAN] desperate fallback: cand[{idx}]")
                break
            except Exception:
                pass
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    # ── phase 2: future feasibility for each candidate ──────────────────────
    print("[FOUNDATION PLAN]")
    scored: list[tuple] = []
    for idx, cand_dbg, decision, raw_box, placement, props, strength in candidate_entries:
        sim_raw = placed_raw_boxes_local + [placement.raw_box_local]
        sim_padded = placed_padded_boxes_local + [placement.padded_box_local]

        future_total = 0
        future_floor_count = 0
        future_any_count = 0
        for other_idx, other_cand_dbg, _, other_raw_box, _, _, _ in candidate_entries:
            if other_idx == idx:
                continue
            future_total += 1
            other_p = _foundation_find_best_placement(
                other_raw_box,
                ctx=ctx,
                placed_raw_boxes_local=sim_raw,
                placed_padded_boxes_local=sim_padded,
                bag_min_xyz=bag_min_xyz,
                bag_size_xyz=bag_size_xyz,
            )
            if other_p is not None:
                future_any_count += 1
                if other_p.is_floor:
                    future_floor_count += 1

        future_stack_forced = future_any_count - future_floor_count
        future_stranded = future_total - future_any_count

        is_floor = int(placement.is_floor)
        score_tuple = (
            is_floor,
            future_floor_count,
            -future_stack_forced,
            -future_stranded,
            strength,
            float(props.weight),
            -float(props.fragility),
            -float(props.compliance),
            float(placement.fit_clearance_mm),
            -idx,
        )

        target_center_xy = bag_min_xyz[:2] + np.asarray(
            placement.raw_box_local.center_xyz_mm[:2], dtype=np.float64
        )
        target_phi = _normalize_phi_deg(float(ctx.base_phi_deg) + float(placement.yaw_deg))
        target = SequenceTargetPlan(
            target_xy_mm=target_center_xy.copy(),
            target_phi_deg=target_phi,
            footprint_clearance_mm=float(placement.fit_clearance_mm),
            aabb_clearance_mm=float(placement.fit_clearance_mm),
            fit_clearance_mm=float(placement.fit_clearance_mm),
            nudge_xy_mm=np.zeros(2, dtype=np.float64),
            can_place=True,
            reason=(
                f"foundation_floor_future_aware: floor={placement.is_floor} "
                f"yaw={placement.yaw_deg:.0f} z={placement.layer_z_mm:.1f}"
            ),
            planner_score=float(strength),
            planner_yaw_deg=float(placement.yaw_deg),
            planner_layer_z_mm=float(placement.layer_z_mm),
            future_placeable_count=int(future_any_count),
            future_total_count=int(future_total),
        )
        target_cache[idx] = target
        overlay[idx] = PlanningOverlayEntry(
            can_place=True,
            target_xy_mm=target_center_xy.copy(),
            target_phi_deg=float(target_phi),
            clearance_mm=float(placement.fit_clearance_mm),
            reason=target.reason,
        )

        cand_class = str(
            getattr(getattr(cand_dbg.candidate, "yolo", None), "class_name", None)
            or f"cand[{idx}]"
        )
        print(
            f"  cand[{idx}] {cand_class:16s} floor={placement.is_floor} "
            f"target=({target_center_xy[0]:.1f},{target_center_xy[1]:.1f}) "
            f"props=w{props.weight:.2f} frag{props.fragility:.2f} comp{props.compliance:.2f} "
            f"floor_future={future_floor_count}/{future_total} "
            f"stack_forced={future_stack_forced} stranded={future_stranded} "
            f"strength={strength:.3f}"
        )
        scored.append((score_tuple, idx, cand_dbg, decision, target))

    if not scored:
        result.selected = None
        result.selected_decision = None
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    any_floor = any(s[0][0] for s in scored)
    scored.sort(reverse=True)

    winner_score, winner_idx, winner_dbg, winner_decision, winner_target = scored[0]

    for score_tuple, idx, cand_dbg, decision, target in scored[1:]:
        cand_class = str(
            getattr(getattr(cand_dbg.candidate, "yolo", None), "class_name", None)
            or f"cand[{idx}]"
        )
        if any_floor and not score_tuple[0]:
            print(f"  cand[{idx}] {cand_class:16s} -> rejected_as_winner: floor_options_exist")

    winner_class = str(
        getattr(getattr(winner_dbg.candidate, "yolo", None), "class_name", None)
        or f"cand[{winner_idx}]"
    )
    print(f"SELECTED cand[{winner_idx}] {winner_class} reason=floor_first_future_aware")

    result.selected = winner_dbg
    result.selected_decision = winner_decision
    state.selected_index = state.candidates.index(winner_dbg)
    return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)


def compute_candidate_place_target(
    sequence_name: str,
    cand_dbg: Any,
    *,
    state: Any,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> SequenceTargetPlan:
    name = validate_planning_sequence_name(sequence_name)
    if name == VOLUME_TOPDOWN_SAFE:
        return _compute_volume_topdown_safe_target(cand_dbg, ctx=ctx, runtime=runtime)
    if name == FOUNDATION_FLOOR_FUTURE_AWARE:
        return _compute_foundation_floor_target(cand_dbg, ctx=ctx, runtime=runtime)
    return _compute_bag_local_target(cand_dbg, state=state, result=None, ctx=ctx, runtime=runtime)


def _select_bag_local_joint(
    state: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> PlanningSelection:
    result = runtime.choose_best_candidate(state, config=ctx.config, robot=ctx.robot, placed_boxes=ctx.placed_boxes)
    result.print_debug("[BEST]")
    overlay: dict[int, PlanningOverlayEntry] = {}
    target_cache: dict[int, SequenceTargetPlan] = {}
    if not getattr(state, "candidates", None):
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    scored: list[tuple[float, float, float, float, int, Any]] = []
    for idx, cand_dbg in enumerate(state.candidates):
        decision = _decision_for_candidate(result, cand_dbg)
        if decision is None or not bool(decision.passed):
            reject_reason = "selector_rejected"
            if decision is not None and getattr(decision, "reject_reasons", None):
                reject_reason = ";".join(list(decision.reject_reasons[:2]))
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, reject_reason)
            continue
        try:
            target = _compute_bag_local_target(
                cand_dbg,
                state=state,
                result=result,
                ctx=ctx,
                runtime=runtime,
            )
            target_cache[idx] = target
            overlay[idx] = PlanningOverlayEntry(
                can_place=True,
                target_xy_mm=target.target_xy_mm.copy(),
                target_phi_deg=float(target.target_phi_deg),
                clearance_mm=float(target.fit_clearance_mm),
                reason=target.reason,
            )
            # is_floor=1 when the planner placed this item on the bag floor (layer_z≈0).
            # Making it the first sort key guarantees any floor placement beats any
            # stack, regardless of per-item scores.
            _is_floor = int(abs(float(target.planner_layer_z_mm or 0.0)) <= 1e-6)
            # For floor items: prefer SMALLER footprint first.  This lets small
            # items fill the base side-by-side; large items then go on top or in
            # the remaining space.  planner_score (which rewards large footprints)
            # is only the tie-breaker within the same footprint bucket.
            # Footprint proxy: use volume_mm3 ∝ footprint × height; since height
            # is roughly constant, smaller volume ≈ smaller floor footprint.
            _floor_volume = float(_candidate_priority(cand_dbg))
            # Negate so sort(reverse=True) puts the SMALLEST floor items first.
            _floor_rank = -_floor_volume if _is_floor else 0.0
            scored.append(
                (
                    _is_floor,
                    _floor_rank,
                    float(target.planner_score or 0.0),
                    float(target.future_placeable_count or 0.0),
                    float(target.fit_clearance_mm),
                    -float(target.planner_layer_z_mm or 0.0),
                    int(getattr(cand_dbg.candidate, "index", -1)),
                    cand_dbg,
                )
            )
        except Exception as exc:
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, str(exc))

    if not scored:
        # Desperate fallback: bag_local planner found nothing — try the simpler
        # volume_topdown_safe planner with overflow tolerance so we still move forward.
        for idx, cand_dbg in enumerate(state.candidates):
            decision = _decision_for_candidate(result, cand_dbg)
            if decision is None or not bool(decision.passed):
                continue
            try:
                target = _compute_volume_topdown_safe_target(
                    cand_dbg, ctx=ctx, runtime=runtime, overflow_mm=_DESPERATE_OVERFLOW_MM
                )
                target_cache[idx] = target
                overlay[idx] = PlanningOverlayEntry(
                    can_place=True,
                    target_xy_mm=target.target_xy_mm.copy(),
                    target_phi_deg=float(target.target_phi_deg),
                    clearance_mm=float(target.fit_clearance_mm),
                    reason=f"desperate:{target.reason}",
                )
                _is_floor = int(abs(float(target.planner_layer_z_mm or 0.0)) <= 1e-6)
                _floor_volume = float(_candidate_priority(cand_dbg))
                _floor_rank = -_floor_volume if _is_floor else 0.0
                scored.append((
                    _is_floor,
                    _floor_rank,
                    float(target.planner_score or 0.0),
                    0.0,
                    float(target.fit_clearance_mm),
                    0.0,
                    int(getattr(cand_dbg.candidate, "index", -1)),
                    cand_dbg,
                ))
            except Exception:
                pass

    if not scored:
        if getattr(result, "selected_decision", None) is not None:
            result.selected = result.selected_decision.dbg
            state.selected_index = state.candidates.index(result.selected)
        else:
            result.selected = None
            result.selected_decision = None
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    scored.sort(reverse=True)
    _is_floor, _floor_rank, _score, _future, _clearance, _layer, _candidate_i, selected_dbg = scored[0]
    best_decision = _decision_for_candidate(result, selected_dbg)
    if best_decision is not None:
        result.selected = selected_dbg
        result.selected_decision = best_decision
        state.selected_index = state.candidates.index(selected_dbg)
    return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)


def _select_volume_topdown_safe(
    state: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> PlanningSelection:
    result = runtime.choose_best_candidate(state, config=ctx.config, robot=ctx.robot, placed_boxes=ctx.placed_boxes)
    result.print_debug("[BEST]")
    overlay: dict[int, PlanningOverlayEntry] = {}
    target_cache: dict[int, SequenceTargetPlan] = {}
    if not getattr(state, "candidates", None):
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    scored: list[tuple[float, float, int, Any]] = []
    for idx, cand_dbg in enumerate(state.candidates):
        decision = _decision_for_candidate(result, cand_dbg)
        if decision is None or not bool(decision.passed):
            reject_reason = "selector_rejected"
            if decision is not None and getattr(decision, "reject_reasons", None):
                reject_reason = ";".join(list(decision.reject_reasons[:2]))
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, reject_reason)
            continue
        try:
            target = _compute_volume_topdown_safe_target(cand_dbg, ctx=ctx, runtime=runtime)
            target_cache[idx] = target
            overlay[idx] = PlanningOverlayEntry(
                can_place=True,
                target_xy_mm=target.target_xy_mm.copy(),
                target_phi_deg=float(target.target_phi_deg),
                clearance_mm=float(target.fit_clearance_mm),
                reason=target.reason,
            )
            _is_floor = int(abs(float(target.planner_layer_z_mm or 0.0)) <= 1e-6)
            _vol = float(getattr(decision, "volume_mm3", _candidate_priority(cand_dbg)))
            _floor_rank = -_vol if _is_floor else 0.0  # smallest floor footprint first
            scored.append(
                (
                    _is_floor,
                    _floor_rank,
                    _vol,
                    float(target.fit_clearance_mm),
                    int(getattr(cand_dbg.candidate, "index", -1)),
                    cand_dbg,
                )
            )
        except Exception as exc:
            overlay[idx] = PlanningOverlayEntry(False, None, None, None, str(exc))

    if not scored:
        # Desperate fallback: no candidate fits strictly — retry with overflow tolerance
        # so we still pick the least-bad option rather than giving up.
        for idx, cand_dbg in enumerate(state.candidates):
            decision = _decision_for_candidate(result, cand_dbg)
            if decision is None or not bool(decision.passed):
                continue
            try:
                target = _compute_volume_topdown_safe_target(
                    cand_dbg, ctx=ctx, runtime=runtime, overflow_mm=_DESPERATE_OVERFLOW_MM
                )
                target_cache[idx] = target
                overlay[idx] = PlanningOverlayEntry(
                    can_place=True,
                    target_xy_mm=target.target_xy_mm.copy(),
                    target_phi_deg=float(target.target_phi_deg),
                    clearance_mm=float(target.fit_clearance_mm),
                    reason=f"desperate:{target.reason}",
                )
                _is_floor = int(abs(float(target.planner_layer_z_mm or 0.0)) <= 1e-6)
                _vol = float(getattr(decision, "volume_mm3", _candidate_priority(cand_dbg)))
                _floor_rank = -_vol if _is_floor else 0.0
                scored.append((
                    _is_floor,
                    _floor_rank,
                    _vol,
                    float(target.fit_clearance_mm),
                    int(getattr(cand_dbg.candidate, "index", -1)),
                    cand_dbg,
                ))
            except Exception:
                pass

    if not scored:
        result.selected = None
        result.selected_decision = None
        return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)

    scored.sort(reverse=True)
    _is_floor, _floor_rank, _volume, _clearance, _candidate_i, selected_dbg = scored[0]
    best_decision = _decision_for_candidate(result, selected_dbg)
    if best_decision is not None:
        result.selected = selected_dbg
        result.selected_decision = best_decision
        state.selected_index = state.candidates.index(selected_dbg)
    return PlanningSelection(result=result, overlay=overlay, target_cache=target_cache)


def select_candidate_for_state(
    sequence_name: str,
    state: Any,
    *,
    ctx: PlanningSequenceContext,
    runtime: PlanningRuntime,
) -> PlanningSelection:
    name = validate_planning_sequence_name(sequence_name)
    if name == VOLUME_TOPDOWN_SAFE:
        return _select_volume_topdown_safe(state, ctx=ctx, runtime=runtime)
    if name == FOUNDATION_FLOOR_FUTURE_AWARE:
        return _select_foundation_floor_future_aware(state, ctx=ctx, runtime=runtime)
    return _select_bag_local_joint(state, ctx=ctx, runtime=runtime)
