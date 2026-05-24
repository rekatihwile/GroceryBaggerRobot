from __future__ import annotations

"""Explicit pick Z policy built from robust object Z diagnostics."""

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class PickZPlan:
    object_top_z_mm: float
    object_bottom_z_mm: float
    object_height_mm: float
    local_top_z_mm: float
    top_spread_mm: float
    pick_uncertainty_clearance_mm: float
    gripper_offset_mm: float
    extra_clearance_mm: float
    final_grasp_z_mm: float
    hover_z_mm: float
    travel_z_mm: float
    warnings: list[str]


def _safe_attr(obj: Any, attr: str, default: float = 0.0) -> float:
    value = getattr(obj, attr, default)
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def compute_pick_z_plan(
    z_result,
    gripper_offset_mm,
    z_max_mm,
    pick_extra_clearance_mm=0.0,
    pick_uncertainty_gain=1.0,
    pick_uncertainty_clearance_max_mm=20.0,
) -> PickZPlan:
    warnings: list[str] = []

    object_top_z_mm = _safe_attr(z_result, "robust_top_z_mm", 0.0)
    object_bottom_z_mm = _safe_attr(z_result, "robust_bottom_z_mm", object_top_z_mm)
    object_height_mm = _safe_attr(z_result, "object_height_mm", max(0.0, object_top_z_mm - object_bottom_z_mm))
    local_top_z_mm = _safe_attr(z_result, "z_p95_mm", object_top_z_mm)
    top_spread_mm = max(0.0, _safe_attr(z_result, "top_spread_mm", 0.0))

    raw_uncertainty = top_spread_mm * float(pick_uncertainty_gain)
    pick_uncertainty_clearance_mm = _clamp(raw_uncertainty, 0.0, float(pick_uncertainty_clearance_max_mm))
    if pick_uncertainty_clearance_mm != raw_uncertainty:
        warnings.append("pick_uncertainty_clearance_clamped")

    grasp_unclamped = (
        object_top_z_mm
        + float(gripper_offset_mm)
        + float(pick_extra_clearance_mm)
        + pick_uncertainty_clearance_mm
    )

    final_grasp_z_mm = _clamp(grasp_unclamped, 0.0, float(z_max_mm))
    if final_grasp_z_mm != grasp_unclamped:
        warnings.append("final_grasp_z_clamped")

    for msg in getattr(z_result, "warnings", []) or []:
        warnings.append(str(msg))

    return PickZPlan(
        object_top_z_mm=object_top_z_mm,
        object_bottom_z_mm=object_bottom_z_mm,
        object_height_mm=max(0.0, object_height_mm),
        local_top_z_mm=local_top_z_mm,
        top_spread_mm=top_spread_mm,
        pick_uncertainty_clearance_mm=pick_uncertainty_clearance_mm,
        gripper_offset_mm=float(gripper_offset_mm),
        extra_clearance_mm=float(pick_extra_clearance_mm),
        final_grasp_z_mm=final_grasp_z_mm,
        hover_z_mm=float(z_max_mm),
        travel_z_mm=float(z_max_mm),
        warnings=warnings,
    )
