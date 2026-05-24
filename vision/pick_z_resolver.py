from __future__ import annotations

"""Validation-only pick Z policy helpers."""

from dataclasses import dataclass

import numpy as np


Z_MAX_MM: float = 275.0
GRIPPER_OFFSET_MM: float = 125.0
USE_ROBUST_OBJECT_Z = True
ROBUST_TOP_PERCENTILE = 95.0
ROBUST_BOTTOM_PERCENTILE = 5.0
TOP_SPREAD_LOW_PERCENTILE = 90.0
TOP_SPREAD_HIGH_PERCENTILE = 99.0
Z_UNCERTAINTY_CLEARANCE_GAIN = 1.0
Z_UNCERTAINTY_CLEARANCE_MIN_MM = 0.0
Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
Z_UNCERTAINTY_WARN_MM = 10.0
PICK_EXTRA_CLEARANCE_MM = 0.0
PLACE_RELEASE_GAP_MM = 8.0
MIN_OBJECT_HEIGHT_MM = 5.0
MAX_OBJECT_HEIGHT_MM = 180.0


@dataclass(frozen=True)
class PickZPlan:
    object_z: float
    travel_z: float
    hover_z: float
    grasp_z: float


@dataclass
class RobustZResult:
    raw_min_z_mm: float
    raw_max_z_mm: float
    z_p50_mm: float
    z_p90_mm: float
    z_p95_mm: float
    z_p99_mm: float
    robust_top_z_mm: float
    robust_bottom_z_mm: float
    object_height_mm: float
    top_spread_mm: float
    uncertainty_clearance_mm: float
    grasp_z_mm: float
    hover_z_mm: float
    travel_z_mm: float
    source: str
    warnings: list[str]
    place_release_z_mm: float | None = None


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _finite_z_from_points(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros(0, dtype=np.float64)
    z = arr[:, 2].reshape(-1)
    return z[np.isfinite(z)]


def _candidate_fallback_z(candidate) -> float:
    if candidate is None:
        return 0.0
    for attr in ("object_robot_xyz_raw", "object_robot_xyz_corrected", "target_cam_xyz"):
        val = getattr(candidate, attr, None)
        if val is None:
            continue
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.isfinite(arr[2]):
            return float(arr[2])
    return 0.0


def _candidate_range_z(candidate) -> np.ndarray:
    if candidate is None:
        return np.zeros(0, dtype=np.float64)
    rng = getattr(candidate, "pointcloud_height_robot_z_range_mm", None)
    if rng is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(rng, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.zeros(0, dtype=np.float64)
    lo = float(np.min(arr[:2]))
    hi = float(np.max(arr[:2]))
    return np.array([lo, hi], dtype=np.float64)


def _result_from_z_values(
    z_values: np.ndarray,
    *,
    source: str,
    warnings: list[str],
    gripper_offset_mm: float,
    z_max_mm: float,
    zone_floor_z_mm: float | None,
    place_release_gap_mm: float,
) -> RobustZResult:
    z = np.asarray(z_values, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    z_max = float(z_max_mm)

    raw_min = float(np.min(z))
    raw_max = float(np.max(z))
    z_p50 = float(np.percentile(z, 50.0))
    z_p90 = float(np.percentile(z, 90.0))
    z_p95 = float(np.percentile(z, 95.0))
    z_p99 = float(np.percentile(z, 99.0))

    robust_top = float(np.percentile(z, float(ROBUST_TOP_PERCENTILE)))
    robust_bottom = float(np.percentile(z, float(ROBUST_BOTTOM_PERCENTILE)))
    raw_height = robust_top - robust_bottom
    object_height = _clamp(raw_height, float(MIN_OBJECT_HEIGHT_MM), float(MAX_OBJECT_HEIGHT_MM))
    if object_height != raw_height:
        warnings.append("object_height_clamped")

    spread_hi = float(np.percentile(z, float(TOP_SPREAD_HIGH_PERCENTILE)))
    spread_lo = float(np.percentile(z, float(TOP_SPREAD_LOW_PERCENTILE)))
    top_spread = max(0.0, spread_hi - spread_lo)
    uncertainty = _clamp(
        top_spread * float(Z_UNCERTAINTY_CLEARANCE_GAIN),
        float(Z_UNCERTAINTY_CLEARANCE_MIN_MM),
        float(Z_UNCERTAINTY_CLEARANCE_MAX_MM),
    )
    if top_spread > float(Z_UNCERTAINTY_WARN_MM):
        warnings.append("noisy_top_surface")
    if uncertainty >= float(Z_UNCERTAINTY_CLEARANCE_MAX_MM) and top_spread * float(Z_UNCERTAINTY_CLEARANCE_GAIN) > uncertainty:
        warnings.append("uncertainty_clearance_clamped")

    travel_z = _clamp(z_max, 0.0, z_max)
    hover_z = _clamp(z_max, 0.0, z_max)
    grasp_unclamped = robust_top + float(gripper_offset_mm) + uncertainty + float(PICK_EXTRA_CLEARANCE_MM)
    grasp_z = _clamp(grasp_unclamped, 0.0, z_max)
    if grasp_z != grasp_unclamped:
        warnings.append("grasp_z_clamped")

    place_release_z = None
    if zone_floor_z_mm is not None:
        place_unclamped = (
            float(zone_floor_z_mm)
            + object_height
            + float(place_release_gap_mm)
            + uncertainty
        )
        place_release_z = _clamp(place_unclamped, 0.0, z_max)
        if place_release_z != place_unclamped:
            warnings.append("place_release_z_clamped")

    return RobustZResult(
        raw_min_z_mm=raw_min,
        raw_max_z_mm=raw_max,
        z_p50_mm=z_p50,
        z_p90_mm=z_p90,
        z_p95_mm=z_p95,
        z_p99_mm=z_p99,
        robust_top_z_mm=robust_top,
        robust_bottom_z_mm=robust_bottom,
        object_height_mm=object_height,
        top_spread_mm=top_spread,
        uncertainty_clearance_mm=uncertainty,
        grasp_z_mm=grasp_z,
        hover_z_mm=hover_z,
        travel_z_mm=travel_z,
        source=source,
        warnings=warnings,
        place_release_z_mm=place_release_z,
    )


def resolve_robust_object_z(
    points_robot: np.ndarray | None = None,
    points_cam: np.ndarray | None = None,
    candidate=None,
    gripper_offset_mm: float = GRIPPER_OFFSET_MM,
    z_max_mm: float = Z_MAX_MM,
    zone_floor_z_mm: float | None = None,
    place_release_gap_mm: float = PLACE_RELEASE_GAP_MM,
) -> RobustZResult:
    warnings: list[str] = []

    z = _finite_z_from_points(points_robot)
    source = "points_robot_percentiles"

    if len(z) == 0:
        z = _candidate_range_z(candidate)
        source = "candidate_robot_z_range"
        if len(z) > 0:
            warnings.append("using_candidate_z_range")

    if len(z) == 0 and candidate is not None:
        fallback = _candidate_fallback_z(candidate)
        z = np.array([fallback], dtype=np.float64)
        source = "candidate_object_robot_xyz_raw"
        warnings.append("fallback_no_usable_pointcloud")

    if len(z) == 0:
        z = _finite_z_from_points(points_cam)
        source = "points_cam_z_untransformed"
        if len(z) > 0:
            warnings.append("using_camera_z_without_robot_transform")

    if len(z) == 0:
        z = np.array([0.0], dtype=np.float64)
        source = "zero_fallback"
        warnings.append("fallback_no_z_available")

    if not USE_ROBUST_OBJECT_Z:
        fallback = _candidate_fallback_z(candidate) if candidate is not None else float(np.percentile(z, 50.0))
        z = np.array([max(0.0, fallback)], dtype=np.float64)
        source = "legacy_simple_z_policy"
        warnings.append("robust_object_z_disabled")

    return _result_from_z_values(
        z,
        source=source,
        warnings=warnings,
        gripper_offset_mm=float(gripper_offset_mm),
        z_max_mm=float(z_max_mm),
        zone_floor_z_mm=zone_floor_z_mm,
        place_release_gap_mm=float(place_release_gap_mm),
    )


def resolve_pick_z_from_stereo(
    stereo_z_mm: float,
    *,
    z_max_mm: float | None = None,
    gripper_offset_mm: float | None = None,
) -> PickZPlan:
    z_max = float(Z_MAX_MM if z_max_mm is None else z_max_mm)
    offset = float(GRIPPER_OFFSET_MM if gripper_offset_mm is None else gripper_offset_mm)
    object_z = max(0.0, float(stereo_z_mm))
    grasp_z = _clamp(object_z + offset, 0.0, z_max)
    return PickZPlan(
        object_z=float(object_z),
        travel_z=float(z_max),
        hover_z=float(z_max),
        grasp_z=float(grasp_z),
    )


def validate_z_command(
    z_mm: float,
    label: str,
    *,
    z_max_mm: float | None = None,
) -> str | None:
    z_max = float(Z_MAX_MM if z_max_mm is None else z_max_mm)
    if not np.isfinite(z_mm):
        return f"{label} z={z_mm} is not finite"
    if z_mm < 0.0:
        return f"{label} z={z_mm:.1f} mm < 0"
    if z_mm > z_max + 1e-6:
        return f"{label} z={z_mm:.1f} mm > Z_MAX_MM={z_max:.1f}"
    return None


_validate_z_command = validate_z_command
