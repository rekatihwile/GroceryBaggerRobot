from __future__ import annotations

"""
No-hardware smoke checks for pick validation resolver modules.

Run:
    python scripts/smoke_tests/smoke_pick_resolvers.py
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision.pick_phi_resolver import wrapped_phi_blend_deg, wrapped_phi_diff_deg
from vision.pick_xy_resolver import weighted_xy
from vision.pick_z_resolver import resolve_pick_z_from_stereo


def _assert_close(label: str, actual, expected, tol: float = 1e-6) -> None:
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not np.allclose(actual, expected, atol=tol):
            raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")
    elif abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def main() -> int:
    print("=" * 60)
    print("smoke_pick_resolvers.py")
    print("=" * 60)

    _assert_close("wrapped_phi_diff 5-175", wrapped_phi_diff_deg(5.0, 175.0), 10.0)
    _assert_close("wrapped_phi_diff 175-5", wrapped_phi_diff_deg(175.0, 5.0), -10.0)
    print("[PASS] wrapped_phi_diff_deg handles mod-180 wrap.")

    blended = wrapped_phi_blend_deg(85.0, 0.5, -85.0, 0.5)
    if abs(wrapped_phi_diff_deg(blended, 90.0)) > 1e-6:
        raise AssertionError(f"wrapped_phi_blend_deg: got {blended!r}, expected equivalent to 90 deg")
    print("[PASS] wrapped_phi_blend_deg blends equivalent gripper angles.")

    xy, source = weighted_xy(np.array([0.0, 0.0]), np.array([10.0, 20.0]), 0.25)
    _assert_close("weighted_xy", xy, np.array([2.5, 5.0]))
    if source != "weighted_overhead_0.25_stereo_0.75":
        raise AssertionError(f"weighted_xy source: got {source!r}")
    print("[PASS] weighted_xy blends stereo/overhead arrays.")

    z = resolve_pick_z_from_stereo(-10.0)
    _assert_close("z object clamp", z.object_z, 0.0)
    _assert_close("z travel", z.travel_z, 275.0)
    _assert_close("z hover", z.hover_z, 275.0)
    _assert_close("z grasp", z.grasp_z, 125.0)
    print("[PASS] resolve_pick_z_from_stereo clamps negative stereo z.")

    z = resolve_pick_z_from_stereo(42.0)
    _assert_close("z object positive", z.object_z, 42.0)
    _assert_close("z grasp positive", z.grasp_z, 167.0)
    print("[PASS] resolve_pick_z_from_stereo applies gripper offset.")

    print("\n[PASS] smoke_pick_resolvers: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
