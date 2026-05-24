from __future__ import annotations

"""Explicit place Z policy decoupled from pick uncertainty behavior."""

from dataclasses import dataclass


@dataclass
class PlaceZPlan:
    destination_surface_z_mm: float
    object_height_mm: float
    release_gap_mm: float
    place_uncertainty_clearance_mm: float
    final_release_z_mm: float
    approach_z_mm: float
    retract_z_mm: float
    warnings: list[str]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def compute_place_z_plan(
    destination_surface_z_mm,
    object_height_mm,
    z_max_mm,
    release_gap_mm=1.0,
    object_uncertainty_clearance_mm=0.0,
    place_uncertainty_gain=0.25,
    place_uncertainty_clearance_max_mm=5.0,
) -> PlaceZPlan:
    warnings: list[str] = []

    scaled_uncertainty = float(object_uncertainty_clearance_mm) * float(place_uncertainty_gain)
    place_uncertainty_clearance_mm = _clamp(
        scaled_uncertainty,
        0.0,
        float(place_uncertainty_clearance_max_mm),
    )
    if place_uncertainty_clearance_mm != scaled_uncertainty:
        warnings.append("place_uncertainty_clearance_clamped")

    final_unclamped = (
        float(destination_surface_z_mm)
        + float(object_height_mm)
        + float(release_gap_mm)
        + place_uncertainty_clearance_mm
    )
    final_release_z_mm = _clamp(final_unclamped, 0.0, float(z_max_mm))
    if final_release_z_mm != final_unclamped:
        warnings.append("final_release_z_clamped")

    plan = PlaceZPlan(
        destination_surface_z_mm=float(destination_surface_z_mm),
        object_height_mm=float(object_height_mm),
        release_gap_mm=float(release_gap_mm),
        place_uncertainty_clearance_mm=float(place_uncertainty_clearance_mm),
        final_release_z_mm=float(final_release_z_mm),
        approach_z_mm=float(z_max_mm),
        retract_z_mm=float(z_max_mm),
        warnings=warnings,
    )

    print(
        "[PLACE Z POLICY] "
        f"surface={plan.destination_surface_z_mm:.1f} "
        f"height={plan.object_height_mm:.1f} "
        f"gap={plan.release_gap_mm:.1f} "
        f"uncertainty_in={float(object_uncertainty_clearance_mm):.1f} "
        f"gain={float(place_uncertainty_gain):.2f} "
        f"uncertainty_used={plan.place_uncertainty_clearance_mm:.1f} "
        f"final_release={plan.final_release_z_mm:.1f} "
        f"warnings={plan.warnings}"
    )
    return plan
