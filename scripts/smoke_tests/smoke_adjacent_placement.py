"""Smoke test for adjacent padded-AABB placement math."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.aabb_utils import make_aabb_from_center_size, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement


def main() -> int:
    box1 = make_aabb_from_center_size([0, 0, 50], [120, 80, 100], label="box1")
    box2 = make_aabb_from_center_size([0, 0, 30], [90, 70, 60], label="box2")

    b1 = pad_aabb(box1, 10, 10, 0)
    b2 = pad_aabb(box2, 10, 10, 0)

    plan = compute_adjacent_placement(
        reference_padded_box=b1,
        moving_padded_box=b2,
        direction="right",
        surface_z_mm=0.0,
        place_phi_deg=0.0,
    )

    expected_x = b1.padded_box.center_xyz_mm[0] + 0.5 * b1.padded_box.size_xyz_mm[0] + 0.5 * b2.padded_box.size_xyz_mm[0]
    expected_y = b1.padded_box.center_xyz_mm[1]
    expected_z = 0.0 + 0.5 * b2.padded_box.size_xyz_mm[2]

    assert abs(plan.target_center_xyz_mm[0] - expected_x) < 1e-9
    assert abs(plan.target_center_xyz_mm[1] - expected_y) < 1e-9
    assert abs(plan.target_center_xyz_mm[2] - expected_z) < 1e-9

    print("[PASS] smoke_adjacent_placement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
