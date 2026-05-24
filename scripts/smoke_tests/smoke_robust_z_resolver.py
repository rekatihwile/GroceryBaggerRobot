from __future__ import annotations

"""
No-hardware smoke test for robust object Z resolution.

Run:
    python scripts/smoke_tests/smoke_robust_z_resolver.py
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision.pick_z_resolver import resolve_robust_object_z


def _cloud_from_z(z_values: np.ndarray) -> np.ndarray:
    z = np.asarray(z_values, dtype=np.float64).reshape(-1)
    x = np.linspace(-20.0, 20.0, len(z))
    y = np.linspace(10.0, -10.0, len(z))
    return np.column_stack([x, y, z])


def main() -> int:
    print("=" * 60)
    print("smoke_robust_z_resolver.py")
    print("=" * 60)

    flat_z = np.linspace(39.8, 40.2, 1000)
    flat = resolve_robust_object_z(
        points_robot=_cloud_from_z(flat_z),
        gripper_offset_mm=125.0,
        z_max_mm=275.0,
    )
    if flat.uncertainty_clearance_mm > 1.0:
        raise AssertionError(f"flat uncertainty too high: {flat.uncertainty_clearance_mm}")
    print(f"[PASS] flat object clearance is small: {flat.uncertainty_clearance_mm:.2f} mm")

    noisy_z = np.concatenate([np.full(900, 40.0), np.linspace(40.0, 55.0, 100)])
    noisy = resolve_robust_object_z(
        points_robot=_cloud_from_z(noisy_z),
        gripper_offset_mm=125.0,
        z_max_mm=275.0,
    )
    if noisy.uncertainty_clearance_mm < 8.0:
        raise AssertionError(f"noisy uncertainty too low: {noisy.uncertainty_clearance_mm}")
    if "noisy_top_surface" not in noisy.warnings:
        raise AssertionError(f"expected noisy_top_surface warning, got {noisy.warnings}")
    print(f"[PASS] noisy top adds clearance: {noisy.uncertainty_clearance_mm:.2f} mm")

    if noisy.grasp_z_mm <= flat.grasp_z_mm:
        raise AssertionError(f"expected noisy grasp z > flat grasp z, got {noisy.grasp_z_mm} <= {flat.grasp_z_mm}")
    print("[PASS] grasp_z increases with top spread.")

    high_z = np.linspace(260.0, 280.0, 1000)
    clamped = resolve_robust_object_z(
        points_robot=_cloud_from_z(high_z),
        gripper_offset_mm=125.0,
        z_max_mm=275.0,
    )
    if abs(clamped.grasp_z_mm - 275.0) > 1e-6:
        raise AssertionError(f"expected grasp clamp to 275.0, got {clamped.grasp_z_mm}")
    if "grasp_z_clamped" not in clamped.warnings:
        raise AssertionError(f"expected grasp_z_clamped warning, got {clamped.warnings}")
    print("[PASS] computed grasp_z clamps to z_max.")

    print("\n[PASS] smoke_robust_z_resolver: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
