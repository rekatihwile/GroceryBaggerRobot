"""Smoke test for real-candidate AABB fallback geometry extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.aabb_utils import aabb_from_object_candidate


class _FakeCandidate:
    topdown_width_cm = 8
    topdown_depth_cm = 5
    pointcloud_height_cm = 3
    object_robot_xyz_corrected = [100, 200, 30]
    object_robot_xyz_raw = [100, 200, 30]
    yolo = None


def main() -> int:
    fake = _FakeCandidate()
    box = aabb_from_object_candidate(fake, default_label="fake")

    expected_size = np.array([80.0, 50.0, 30.0], dtype=np.float64)
    assert np.allclose(box.size_xyz_mm, expected_size), f"size mismatch: got {box.size_xyz_mm} expected {expected_size}"
    assert np.isfinite(box.center_xyz_mm).all(), f"center has non-finite values: {box.center_xyz_mm}"
    assert abs(float(box.center_xyz_mm[0]) - 100.0) < 1e-9
    assert abs(float(box.center_xyz_mm[1]) - 200.0) < 1e-9

    print("[PASS] smoke_real_candidate_aabb_fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
