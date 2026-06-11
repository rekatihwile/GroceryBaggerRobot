"""Smoke test for vision.grasp_xy_policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision.grasp_xy_policy import compute_grasp_xy_with_local_height


def _make_peak_pointcloud(global_peak_z: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    base_xy = rng.normal(loc=[0.0, 0.0], scale=[6.0, 6.0], size=(220, 2))
    base_z = np.full((220, 1), 10.0)

    lip_xy = rng.normal(loc=[30.0, 0.0], scale=[4.0, 4.0], size=(80, 2))
    lip_z = np.full((80, 1), global_peak_z)

    pts = np.vstack([
        np.hstack([base_xy, base_z]),
        np.hstack([lip_xy, lip_z]),
    ])
    return pts.astype(np.float64)


def main() -> int:
    default_centroid = np.array([0.0, 0.0], dtype=np.float64)
    default_target = np.array([0.0, 0.0], dtype=np.float64)

    shifted = compute_grasp_xy_with_local_height(
        points_robot_xyz=_make_peak_pointcloud(25.0),
        default_centroid_xy_mm=default_centroid,
        default_target_xy_mm=default_target,
        local_radius_mm=20.0,
        top_region_percentile=90.0,
        height_delta_threshold_mm=8.0,
        top_region_blend_weight=0.45,
        max_xy_shift_mm=40.0,
    )
    assert shifted.final_grasp_xy_mm[0] > 0.0
    assert "height_peak_not_at_centroid" in shifted.warnings

    not_shifted = compute_grasp_xy_with_local_height(
        points_robot_xyz=_make_peak_pointcloud(15.0),
        default_centroid_xy_mm=default_centroid,
        default_target_xy_mm=default_target,
        local_radius_mm=20.0,
        top_region_percentile=90.0,
        height_delta_threshold_mm=8.0,
        top_region_blend_weight=0.45,
        max_xy_shift_mm=40.0,
    )
    assert np.allclose(not_shifted.final_grasp_xy_mm, default_target)

    print("[PASS] smoke_grasp_xy_policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
