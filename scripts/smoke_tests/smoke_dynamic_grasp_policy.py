from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion.dynamic_grasp_policy import (
    DEFAULT_DYNAMIC_PICK_SERVO_ANGLE_DEG,
    build_dynamic_pick_plan,
    estimate_grip_width_mm,
    servo_angle_from_grip_width_mm,
)


class _FakeRobot:
    def fk(self):
        return (0.0, 0.0, 200.0, 15.0)


def _fake_candidate(**overrides):
    base = dict(
        yolo=SimpleNamespace(class_name="test_item"),
        pick_phi_deg=0.0,
        pick_phi_source="test_phi",
        target_xy=np.array([120.0, 240.0], dtype=np.float64),
        object_robot_xyz_raw=np.array([120.0, 240.0, 40.0], dtype=np.float64),
        grasp_robot_z=160.0,
        topdown_oriented_rect_cm=None,
        pointcloud_footprint_cm=None,
        topdown_aabb_cm=None,
        topdown_width_cm=None,
        topdown_depth_cm=None,
        z_debug=SimpleNamespace(robust_top_z_mm=40.0),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def main() -> int:
    y = np.linspace(-20.0, 20.0, 80)
    x = np.linspace(-5.0, 5.0, 80)
    pts = np.column_stack([x, y, np.full_like(x, 40.0)])

    cand = _fake_candidate()
    width_mm, source, warnings = estimate_grip_width_mm(cand, points_robot_xyz=pts, phi_deg=0.0)
    assert width_mm is not None and 35.0 <= width_mm <= 45.0, width_mm
    assert source == "pointcloud_projection_closing_direction"
    print(f"[PASS] projected width from fake point cloud: {width_mm:.2f} mm ({source})")

    narrow = servo_angle_from_grip_width_mm(10.0, L_mm=50.0, margin_deg=8.0, min_deg=10.0, max_deg=70.0)
    wide = servo_angle_from_grip_width_mm(40.0, L_mm=50.0, margin_deg=8.0, min_deg=10.0, max_deg=70.0)
    assert 10.0 <= narrow <= 70.0
    assert 10.0 <= wide <= 70.0
    assert wide > narrow, (narrow, wide)
    print(f"[PASS] larger object width produces a more-open angle: {narrow:.2f} -> {wide:.2f}")

    clamped = servo_angle_from_grip_width_mm(200.0, L_mm=50.0, margin_deg=20.0, min_deg=10.0, max_deg=70.0)
    assert clamped == 70.0, clamped
    print("[PASS] servo angle clamps to [10, 70] deg")

    fallback_cand = _fake_candidate(pick_phi_deg=None, object_robot_xyz_raw=np.array([0.0, 0.0, 35.0]))
    plan = build_dynamic_pick_plan(
        fallback_cand,
        dbg=None,
        robot=_FakeRobot(),
        bundle=None,
        z_grasp_mm=155.0,
        z_max_mm=275.0,
        fallback_initial_servo_angle_deg=DEFAULT_DYNAMIC_PICK_SERVO_ANGLE_DEG,
    )
    assert abs(plan.initial_servo_angle_deg - DEFAULT_DYNAMIC_PICK_SERVO_ANGLE_DEG) < 1e-6
    assert any("using_default_initial_servo_angle" == w for w in plan.warnings), plan.warnings
    print("[PASS] missing geometry falls back to default dynamic servo angle with warning")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
