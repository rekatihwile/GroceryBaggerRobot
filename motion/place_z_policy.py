from __future__ import annotations

"""Shared real-robot place Z policy."""

from dataclasses import dataclass
import math
from typing import Any

from motion.z_safety_config import (
    ZSafetyConfig,
    z_safety_config_with_overrides,
)


@dataclass
class PlaceZPlan:
    destination_floor_or_surface_z_mm: float
    stack_top_mm: float
    raw_item_height_mm: float
    object_height_mm: float
    place_z_safety_padding_mm: float
    release_gap_mm: float
    place_z_raw_mm: float
    final_release_z_mm: float
    approach_z_mm: float
    retract_z_mm: float
    warnings: list[str]
    valid: bool
    z_max_mm: float
    debug: str

    # Compatibility aliases for older diagnostics.
    destination_surface_z_mm: float
    place_uncertainty_clearance_mm: float


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bag_stack_height_mm(
    bag_state,
    *,
    x_cm: float | None,
    y_cm: float | None,
    w_cm: float | None,
    d_cm: float | None,
    inflate_cm: float,
) -> float | None:
    if bag_state is None or x_cm is None or y_cm is None or w_cm is None or d_cm is None:
        return None
    if not hasattr(bag_state, "stack_height_under_rect_mm"):
        return None
    return float(
        bag_state.stack_height_under_rect_mm(
            float(x_cm),
            float(y_cm),
            float(w_cm),
            float(d_cm),
            inflate_cm=float(inflate_cm),
        )
    )


def compute_place_z_plan(
    destination_surface_z_mm: float | None = None,
    object_height_mm: float | None = None,
    z_max_mm: float | None = None,
    release_gap_mm: float | None = None,
    object_uncertainty_clearance_mm: float = 0.0,
    place_uncertainty_gain: float = 0.25,
    place_uncertainty_clearance_max_mm: float = 5.0,
    *,
    destination_floor_or_surface_z_mm: float | None = None,
    stack_top_mm: float | None = None,
    bag_state=None,
    x_cm: float | None = None,
    y_cm: float | None = None,
    w_cm: float | None = None,
    d_cm: float | None = None,
    inflate_cm: float | None = None,
    config: ZSafetyConfig | None = None,
    min_place_item_height_mm: float | None = None,
    place_z_safety_padding_mm: float | None = None,
) -> PlaceZPlan:
    """Compute a safe place/release Z shared by real robot scripts.

    Equation:
        item_height_mm = max(raw_item_height_mm, MIN_PLACE_ITEM_HEIGHT_MM)
        place_z_raw = floor_or_surface + stack_top + item_height
                      + PLACE_Z_SAFETY_PADDING_MM + PLACE_RELEASE_GAP_MM
        final_release_z = max(place_z_raw, MIN_PLACE_Z_MM)
    """
    cfg = z_safety_config_with_overrides(
        config,
        Z_MAX_MM=z_max_mm,
        PLACE_RELEASE_GAP_MM=release_gap_mm,
        MIN_PLACE_ITEM_HEIGHT_MM=min_place_item_height_mm,
        PLACE_Z_SAFETY_PADDING_MM=place_z_safety_padding_mm,
        PLACE_STACK_QUERY_INFLATE_CM=inflate_cm,
    )
    warnings: list[str] = []

    surface_z = _finite_float(destination_surface_z_mm)
    base_z = _finite_float(destination_floor_or_surface_z_mm)
    if base_z is None:
        base_z = surface_z if surface_z is not None else 0.0
        if surface_z is None:
            warnings.append("destination_floor_or_surface_z_missing")

    raw_height = _finite_float(object_height_mm)
    if raw_height is None:
        raw_height = 0.0
        warnings.append("object_height_missing")
    raw_height = max(0.0, raw_height)
    item_height = max(raw_height, cfg.MIN_PLACE_ITEM_HEIGHT_MM)
    if item_height != raw_height:
        warnings.append("item_height_clamped_to_min")

    stack_candidates: list[float] = []
    explicit_stack = _finite_float(stack_top_mm)
    if explicit_stack is not None:
        stack_candidates.append(max(0.0, explicit_stack))

    bag_stack = _bag_stack_height_mm(
        bag_state,
        x_cm=x_cm,
        y_cm=y_cm,
        w_cm=w_cm,
        d_cm=d_cm,
        inflate_cm=cfg.PLACE_STACK_QUERY_INFLATE_CM,
    )
    if bag_stack is not None:
        stack_candidates.append(max(0.0, bag_stack))

    # When both a floor/base and a measured surface are provided, keep the
    # conservative height above the base rather than double-counting the base.
    if surface_z is not None and destination_floor_or_surface_z_mm is not None:
        stack_candidates.append(max(0.0, surface_z - base_z))

    stack = max(stack_candidates) if stack_candidates else 0.0

    scaled_uncertainty = float(object_uncertainty_clearance_mm) * float(place_uncertainty_gain)
    uncertainty_cap = max(0.0, float(place_uncertainty_clearance_max_mm))
    place_uncertainty_clearance_mm = max(0.0, min(uncertainty_cap, scaled_uncertainty))
    if place_uncertainty_clearance_mm > 0.0:
        warnings.append("legacy_place_uncertainty_ignored_by_shared_padding")

    place_z_raw = (
        base_z
        + stack
        + item_height
        + cfg.PLACE_Z_SAFETY_PADDING_MM
        + cfg.PLACE_RELEASE_GAP_MM
    )
    final_release_z = max(place_z_raw, cfg.MIN_PLACE_Z_MM)
    if final_release_z != place_z_raw:
        warnings.append("final_release_z_clamped_to_min")

    valid = True
    if final_release_z > cfg.Z_MAX_MM + 1e-6:
        valid = False
        warnings.append("final_release_z_above_z_max")

    debug = (
        f"floor/surface({base_z:.1f}) + stack({stack:.1f}) "
        f"+ raw_height({raw_height:.1f}) -> clamped_height({item_height:.1f}) "
        f"+ safety_padding({cfg.PLACE_Z_SAFETY_PADDING_MM:.1f}) "
        f"+ release_gap({cfg.PLACE_RELEASE_GAP_MM:.1f}) "
        f"= raw_place_z({place_z_raw:.1f}) -> final_place_z({final_release_z:.1f}) "
        f"warnings={warnings}"
    )

    plan = PlaceZPlan(
        destination_floor_or_surface_z_mm=float(base_z),
        stack_top_mm=float(stack),
        raw_item_height_mm=float(raw_height),
        object_height_mm=float(item_height),
        place_z_safety_padding_mm=float(cfg.PLACE_Z_SAFETY_PADDING_MM),
        release_gap_mm=float(cfg.PLACE_RELEASE_GAP_MM),
        place_z_raw_mm=float(place_z_raw),
        final_release_z_mm=float(final_release_z),
        approach_z_mm=float(cfg.Z_MAX_MM),
        retract_z_mm=float(cfg.Z_MAX_MM),
        warnings=warnings,
        valid=valid,
        z_max_mm=float(cfg.Z_MAX_MM),
        debug=debug,
        destination_surface_z_mm=float(base_z),
        place_uncertainty_clearance_mm=float(place_uncertainty_clearance_mm),
    )

    print(
        "[PLACE Z POLICY] "
        f"floor/surface={plan.destination_floor_or_surface_z_mm:.1f} "
        f"stack={plan.stack_top_mm:.1f} "
        f"raw_height={plan.raw_item_height_mm:.1f} "
        f"clamped_height={plan.object_height_mm:.1f} "
        f"padding={plan.place_z_safety_padding_mm:.1f} "
        f"gap={plan.release_gap_mm:.1f} "
        f"raw_place_z={plan.place_z_raw_mm:.1f} "
        f"final_place_z={plan.final_release_z_mm:.1f} "
        f"warnings={plan.warnings}"
    )
    return plan
