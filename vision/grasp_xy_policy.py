from __future__ import annotations

"""Optional local-height-aware grasp XY adjustment policy."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GraspXYPlan:
    centroid_xy_mm: np.ndarray
    top_region_xy_mm: np.ndarray
    final_grasp_xy_mm: np.ndarray
    local_top_z_mm: float
    global_top_z_mm: float
    height_delta_mm: float
    mode: str
    warnings: list[str]


def _to_xy(arr, label: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.size < 2 or not np.isfinite(out[0]) or not np.isfinite(out[1]):
        raise ValueError(f"{label} must contain finite XY")
    return np.array([float(out[0]), float(out[1])], dtype=np.float64)


def _default_plan(default_centroid_xy_mm, default_target_xy_mm, warning: str) -> GraspXYPlan:
    centroid = _to_xy(default_centroid_xy_mm, "default_centroid_xy_mm")
    target = _to_xy(default_target_xy_mm, "default_target_xy_mm")
    return GraspXYPlan(
        centroid_xy_mm=centroid,
        top_region_xy_mm=target.copy(),
        final_grasp_xy_mm=target.copy(),
        local_top_z_mm=0.0,
        global_top_z_mm=0.0,
        height_delta_mm=0.0,
        mode="default",
        warnings=[warning],
    )


def compute_grasp_xy_with_local_height(
    points_robot_xyz,
    default_centroid_xy_mm,
    default_target_xy_mm,
    local_radius_mm=25.0,
    top_region_percentile=90.0,
    height_delta_threshold_mm=8.0,
    top_region_blend_weight=0.35,
    max_xy_shift_mm=40.0,
) -> GraspXYPlan:
    centroid_xy = _to_xy(default_centroid_xy_mm, "default_centroid_xy_mm")
    default_xy = _to_xy(default_target_xy_mm, "default_target_xy_mm")

    if points_robot_xyz is None:
        return _default_plan(centroid_xy, default_xy, "insufficient_points_default_xy")

    pts = np.asarray(points_robot_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return _default_plan(centroid_xy, default_xy, "insufficient_points_default_xy")

    finite_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
    pts = pts[finite_mask]
    if len(pts) < 20:
        return _default_plan(centroid_xy, default_xy, "insufficient_points_default_xy")

    xy = pts[:, :2]
    z = pts[:, 2]

    dxy = xy - default_xy.reshape(1, 2)
    radial = np.linalg.norm(dxy, axis=1)
    local = z[radial <= float(local_radius_mm)]
    if len(local) < 10:
        return _default_plan(centroid_xy, default_xy, "insufficient_local_points_default_xy")

    local_top_z = float(np.percentile(local, float(top_region_percentile)))

    threshold_z = float(np.percentile(z, float(top_region_percentile)))
    top_mask = z >= threshold_z
    top_xy = xy[top_mask]
    top_z = z[top_mask]
    if len(top_xy) < 10:
        return _default_plan(centroid_xy, default_xy, "insufficient_top_region_points_default_xy")

    top_region_xy = np.median(top_xy, axis=0).astype(np.float64)
    global_top_z = float(np.median(top_z))
    height_delta_mm = float(global_top_z - local_top_z)

    warnings: list[str] = []
    mode = "default"
    final_xy = default_xy.copy()

    if height_delta_mm > float(height_delta_threshold_mm):
        raw_shift = (top_region_xy - default_xy) * float(np.clip(top_region_blend_weight, 0.0, 1.0))
        raw_norm = float(np.linalg.norm(raw_shift))
        if raw_norm > float(max_xy_shift_mm) and raw_norm > 1e-9:
            raw_shift = raw_shift * (float(max_xy_shift_mm) / raw_norm)
            warnings.append("xy_shift_clamped")
        final_xy = default_xy + raw_shift
        mode = "shifted_toward_top_region"
        warnings.append("height_peak_not_at_centroid")

    return GraspXYPlan(
        centroid_xy_mm=centroid_xy,
        top_region_xy_mm=top_region_xy,
        final_grasp_xy_mm=final_xy,
        local_top_z_mm=local_top_z,
        global_top_z_mm=global_top_z,
        height_delta_mm=height_delta_mm,
        mode=mode,
        warnings=warnings,
    )
