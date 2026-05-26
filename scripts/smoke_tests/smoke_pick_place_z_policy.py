"""Smoke test for motion.pick_z_policy and motion.place_z_policy."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motion.pick_z_policy import compute_pick_z_plan
from motion.place_z_policy import compute_place_z_plan


def main() -> int:
    z_result = SimpleNamespace(
        robust_top_z_mm=15.0,
        robust_bottom_z_mm=0.0,
        object_height_mm=100.0,
        z_p95_mm=15.0,
        top_spread_mm=12.0,
        warnings=[],
    )

    pick_plan = compute_pick_z_plan(
        z_result=z_result,
        gripper_offset_mm=135.0,
        z_max_mm=290.0,
    )

    place_plan = compute_place_z_plan(
        destination_surface_z_mm=20.0,
        object_height_mm=100.0,
        z_max_mm=290.0,
        release_gap_mm=1.0,
        object_uncertainty_clearance_mm=12.0,
        place_uncertainty_gain=0.25,
        place_uncertainty_clearance_max_mm=5.0,
    )

    # 20 + 100 + 50 safety padding + 1 release gap = 171
    assert abs(place_plan.final_release_z_mm - 171.0) < 1e-9
    assert abs(place_plan.place_uncertainty_clearance_mm - 3.0) < 1e-9

    capped_place_plan = compute_place_z_plan(
        destination_surface_z_mm=20.0,
        object_height_mm=100.0,
        z_max_mm=290.0,
        release_gap_mm=1.0,
        object_uncertainty_clearance_mm=200.0,
        place_uncertainty_gain=0.25,
        place_uncertainty_clearance_max_mm=5.0,
    )
    assert abs(capped_place_plan.place_uncertainty_clearance_mm - 5.0) < 1e-9

    assert abs(pick_plan.final_grasp_z_mm - 150.0) < 1e-9
    assert "legacy_place_uncertainty_ignored_by_shared_padding" in place_plan.warnings

    print("[PASS] smoke_pick_place_z_policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
