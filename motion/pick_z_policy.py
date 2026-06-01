from __future__ import annotations

"""Shared real-robot pick Z policy."""

from dataclasses import dataclass
import math
from typing import Any

from config.motion.z_safety_config import (
    DEFAULT_Z_SAFETY,
    ZSafetyConfig,
    z_safety_config_with_overrides,
)


@dataclass
class PickZPlan:
    object_surface_z_mm: float
    object_height_mm: float | None
    raw_grasp_z_mm: float
    final_grasp_z_mm: float
    hover_z_mm: float
    approach_z_mm: float
    retract_z_mm: float
    warnings: list[str]
    valid: bool
    z_max_mm: float

    # Compatibility aliases for older call sites.
    object_top_z_mm: float
    object_bottom_z_mm: float
    local_top_z_mm: float
    top_spread_mm: float
    pick_uncertainty_clearance_mm: float
    gripper_offset_mm: float
    extra_clearance_mm: float
    travel_z_mm: float


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_attr(obj: Any, attr: str) -> float | None:
    if obj is None:
        return None
    return _finite_float(getattr(obj, attr, None))


def _candidate_surface_z_mm(candidate: Any) -> float | None:
    if candidate is None:
        return None
    for attr in ("object_robot_xyz_raw", "object_robot_xyz_corrected", "target_cam_xyz"):
        value = getattr(candidate, attr, None)
        if value is None:
            continue
        try:
            z = _finite_float(value[2])
        except (TypeError, IndexError):
            z = None
        if z is not None:
            return z
    return None


def _candidate_height_mm(candidate: Any) -> float | None:
    if candidate is None:
        return None
    for attr, scale in (
        ("pointcloud_height_mm", 1.0),
        ("height_mm_from_pointcloud", 1.0),
        ("pointcloud_height_cm", 10.0),
        ("height_cm_from_pointcloud", 10.0),
    ):
        value = _finite_float(getattr(candidate, attr, None))
        if value is not None:
            return max(0.0, value * scale)
    return None


def compute_pick_z_plan(
    z_result=None,
    gripper_offset_mm: float | None = None,
    z_max_mm: float | None = None,
    pick_extra_clearance_mm: float = 0.0,
    pick_uncertainty_gain: float = 1.0,
    pick_uncertainty_clearance_max_mm: float = 20.0,
    *,
    object_surface_z_mm: float | None = None,
    object_height_mm: float | None = None,
    candidate=None,
    config: ZSafetyConfig | None = None,
) -> PickZPlan:
    """Compute a safe pick grasp Z shared by real robot scripts.

    Equation:
        raw_grasp_z = object_surface_z_mm + GRIPPER_OFFSET_MM
        if height is unknown or below MIN_PLACE_ITEM_HEIGHT_MM:
            raw_grasp_z += PICK_SHORT_ITEM_EXTRA_BUFFER_MM
        final_grasp_z = max(raw_grasp_z, MIN_PICK_GRASP_Z_MM)
    """
    cfg = z_safety_config_with_overrides(
        config,
        GRIPPER_OFFSET_MM=gripper_offset_mm,
        Z_MAX_MM=z_max_mm,
    )
    warnings: list[str] = []

    surface = _finite_float(object_surface_z_mm)
    if surface is None:
        surface = _safe_attr(z_result, "robust_top_z_mm")
    if surface is None:
        surface = _safe_attr(z_result, "grasp_z_mm")
        if surface is not None:
            surface -= cfg.GRIPPER_OFFSET_MM
            warnings.append("using_grasp_z_minus_offset_surface_fallback")
    if surface is None:
        surface = _candidate_surface_z_mm(candidate)
    if surface is None:
        surface = 0.0
        warnings.append("object_surface_z_missing")

    height = _finite_float(object_height_mm)
    if height is None:
        height = _safe_attr(z_result, "object_height_mm")
    if height is None:
        height = _candidate_height_mm(candidate)

    raw_grasp_z = surface + cfg.GRIPPER_OFFSET_MM
    if height is None or height < cfg.MIN_PLACE_ITEM_HEIGHT_MM:
        raw_grasp_z += cfg.PICK_SHORT_ITEM_EXTRA_BUFFER_MM
        warnings.append("short_or_unknown_item_pick_buffer_applied")

    # The old uncertainty knobs are accepted for compatibility, but the shared
    # real-robot equation uses the explicit short-item buffer above.
    if float(pick_extra_clearance_mm) != 0.0:
        warnings.append("pick_extra_clearance_ignored_by_shared_policy")
    final_grasp_z = max(raw_grasp_z, cfg.MIN_PICK_GRASP_Z_MM)
    if final_grasp_z != raw_grasp_z:
        warnings.append("final_grasp_z_clamped_to_min")
    valid = True
    if final_grasp_z > cfg.Z_MAX_MM + 1e-6:
        valid = False
        warnings.append("final_grasp_z_above_z_max")

    for msg in getattr(z_result, "warnings", []) or []:
        warnings.append(str(msg))

    object_bottom = _safe_attr(z_result, "robust_bottom_z_mm")
    if object_bottom is None:
        object_bottom = surface - max(0.0, height or 0.0)
    top_spread = max(0.0, _safe_attr(z_result, "top_spread_mm") or 0.0)

    plan = PickZPlan(
        object_surface_z_mm=float(surface),
        object_height_mm=None if height is None else max(0.0, float(height)),
        raw_grasp_z_mm=float(raw_grasp_z),
        final_grasp_z_mm=float(final_grasp_z),
        hover_z_mm=float(cfg.Z_MAX_MM),
        approach_z_mm=float(cfg.Z_MAX_MM),
        retract_z_mm=float(cfg.Z_MAX_MM),
        warnings=warnings,
        valid=valid,
        z_max_mm=float(cfg.Z_MAX_MM),
        object_top_z_mm=float(surface),
        object_bottom_z_mm=float(object_bottom),
        local_top_z_mm=float(_safe_attr(z_result, "z_p95_mm") or surface),
        top_spread_mm=float(top_spread),
        pick_uncertainty_clearance_mm=0.0,
        gripper_offset_mm=float(cfg.GRIPPER_OFFSET_MM),
        extra_clearance_mm=0.0,
        travel_z_mm=float(cfg.Z_MAX_MM),
    )

    height_s = "unknown" if plan.object_height_mm is None else f"{plan.object_height_mm:.1f}"
    print(
        "[PICK Z POLICY] "
        f"surface={plan.object_surface_z_mm:.1f} "
        f"height={height_s} "
        f"raw_grasp={plan.raw_grasp_z_mm:.1f} "
        f"final_grasp={plan.final_grasp_z_mm:.1f} "
        f"warnings={plan.warnings}"
    )
    return plan
