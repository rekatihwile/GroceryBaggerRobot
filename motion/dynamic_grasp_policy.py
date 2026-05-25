from __future__ import annotations

"""Dynamic pick planning from vision geometry and simple gripper geometry."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


DEFAULT_DYNAMIC_PICK_SERVO_ANGLE_DEG = 55.0


@dataclass
class DynamicPickPlan:
    object_class: str
    target_xy_mm: tuple[float, float]
    phi_deg: float
    phi_source: str
    measured_grip_width_mm: float | None
    grip_width_source: str
    initial_servo_angle_deg: float
    servo_angle_margin_deg: float
    dynamic_lower_start_z_mm: float
    dynamic_pregrasp_robot_z_mm: float
    dynamic_grip_start_angle_deg: float
    z_clearance_mm: float
    warnings: list[str]


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _candidate_xy(candidate: Any) -> tuple[float, float]:
    target_xy = np.asarray(getattr(candidate, "target_xy", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    x = float(target_xy[0]) if target_xy.size >= 1 and np.isfinite(target_xy[0]) else 0.0
    y = float(target_xy[1]) if target_xy.size >= 2 and np.isfinite(target_xy[1]) else 0.0
    return (x, y)


def _candidate_object_class(candidate: Any) -> str:
    yolo = getattr(candidate, "yolo", None)
    name = getattr(yolo, "class_name", None)
    return str(name) if name is not None else "unknown"


def _candidate_phi(candidate: Any, robot: Any) -> tuple[float, str]:
    phi_candidate = _finite_float(getattr(candidate, "pick_phi_deg", None))
    if phi_candidate is not None:
        return phi_candidate, str(getattr(candidate, "pick_phi_source", None) or "candidate_pick_phi_deg")

    if robot is not None and hasattr(robot, "fk"):
        try:
            _, _, _, phi_fk = robot.fk()
            phi_fk = float(phi_fk)
            if math.isfinite(phi_fk):
                return phi_fk, "robot_fk_phi"
        except Exception:
            pass

    return 0.0, "default_zero_phi"


def _candidate_object_top_z_mm(candidate: Any) -> float:
    z_debug = getattr(candidate, "z_debug", None)
    if z_debug is not None:
        top = _finite_float(getattr(z_debug, "robust_top_z_mm", None))
        if top is not None:
            return top
    try:
        z_raw = _finite_float(candidate.object_robot_xyz_raw[2])
        if z_raw is not None:
            return z_raw
    except Exception:
        pass
    return 0.0


def _candidate_dim_mm(candidate: Any, attr_name: str) -> float | None:
    dims = getattr(candidate, attr_name, None)
    if dims is None:
        return None
    try:
        values = tuple(dims)
    except TypeError:
        return None
    if len(values) < 2:
        return None
    a = _finite_float(values[0])
    b = _finite_float(values[1])
    if a is None or b is None or a <= 0.0 or b <= 0.0:
        return None
    return max(a, b) * 10.0


def _project_width_from_points(points_robot_xyz: np.ndarray, phi_deg: float) -> float | None:
    pts = np.asarray(points_robot_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 20:
        return None
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    pts = pts[finite]
    if len(pts) < 20:
        return None

    phi_rad = math.radians(float(phi_deg))
    closing_normal = np.array([-math.sin(phi_rad), math.cos(phi_rad)], dtype=np.float64)
    projected = pts[:, :2] @ closing_normal
    lo = float(np.percentile(projected, 5.0))
    hi = float(np.percentile(projected, 95.0))
    width = hi - lo
    if not math.isfinite(width) or width <= 0.0:
        return None
    return width


def estimate_grip_width_mm(candidate, points_robot_xyz=None, phi_deg=None) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []

    if phi_deg is None:
        phi_deg = _finite_float(getattr(candidate, "pick_phi_deg", None))
    if phi_deg is None:
        warnings.append("phi_unavailable_for_projection")
    else:
        width_proj = _project_width_from_points(points_robot_xyz, float(phi_deg)) if points_robot_xyz is not None else None
        if width_proj is not None:
            return float(width_proj), "pointcloud_projection_closing_direction", warnings
        if points_robot_xyz is None:
            warnings.append("pointcloud_projection_unavailable")
        else:
            warnings.append("pointcloud_projection_invalid")

    for attr_name, source in (
        ("topdown_oriented_rect_cm", "candidate_topdown_oriented_rect_cm"),
        ("pointcloud_footprint_cm", "candidate_pointcloud_footprint_cm"),
        ("topdown_aabb_cm", "candidate_topdown_aabb_cm"),
    ):
        width_mm = _candidate_dim_mm(candidate, attr_name)
        if width_mm is not None:
            warnings.append(f"{source}_used_without_axis_alignment")
            return width_mm, source, warnings

    width_cm = _finite_float(getattr(candidate, "topdown_width_cm", None))
    depth_cm = _finite_float(getattr(candidate, "topdown_depth_cm", None))
    if width_cm is not None and depth_cm is not None and width_cm > 0.0 and depth_cm > 0.0:
        warnings.append("candidate_topdown_width_depth_used_without_axis_alignment")
        return max(width_cm, depth_cm) * 10.0, "candidate_topdown_width_depth_cm", warnings

    warnings.append("grip_width_unavailable")
    return None, "fallback_default", warnings


def servo_angle_from_grip_width_mm(width_mm, *, L_mm, margin_deg, min_deg, max_deg) -> float:
    """Estimate a conservative jaw-open servo angle from width.

    We model the jaw projection with a single-link `L * cos(theta)` relation in
    the linkage frame. In the servo angle convention used here, that becomes an
    equivalent `asin(width / L)` mapping, where larger object width produces a
    larger servo angle (more open).
    """
    width = max(0.0, float(width_mm))
    L = max(1e-6, float(L_mm))
    ratio = _clamp(width / L, 0.0, 1.0)
    base_angle = math.degrees(math.asin(ratio))
    angle = base_angle + float(margin_deg)
    return _clamp(angle, float(min_deg), float(max_deg))


def build_dynamic_pick_plan(
    candidate,
    dbg,
    robot,
    bundle,
    *,
    z_grasp_mm=None,
    gripper_offset_mm=0.0,
    z_max_mm,
    dynamic_lower_clearance_mm=20.0,
    servo_angle_margin_deg=8.0,
    gripper_geometry_l_mm=50.0,
    gripper_servo_min_deg=10.0,
    gripper_servo_max_deg=70.0,
    fallback_initial_servo_angle_deg=DEFAULT_DYNAMIC_PICK_SERVO_ANGLE_DEG,
) -> DynamicPickPlan:
    warnings: list[str] = []
    phi_deg, phi_source = _candidate_phi(candidate, robot)
    target_xy_mm = _candidate_xy(candidate)

    points_robot_xyz = None
    if dbg is not None and getattr(dbg, "points_cam", None) is not None and bundle is not None:
        try:
            from vision.pointcloud import cam_points_to_robot_xyz

            points_robot_xyz = cam_points_to_robot_xyz(dbg.points_cam, bundle)
        except Exception as exc:
            warnings.append(f"pointcloud_robot_transform_failed:{exc}")

    measured_grip_width_mm, grip_width_source, width_warnings = estimate_grip_width_mm(
        candidate,
        points_robot_xyz=points_robot_xyz,
        phi_deg=phi_deg,
    )
    warnings.extend(width_warnings)

    if measured_grip_width_mm is None:
        initial_servo_angle_deg = _clamp(
            float(fallback_initial_servo_angle_deg),
            float(gripper_servo_min_deg),
            float(gripper_servo_max_deg),
        )
        warnings.append("using_default_initial_servo_angle")
    else:
        initial_servo_angle_deg = servo_angle_from_grip_width_mm(
            measured_grip_width_mm,
            L_mm=gripper_geometry_l_mm,
            margin_deg=servo_angle_margin_deg,
            min_deg=gripper_servo_min_deg,
            max_deg=gripper_servo_max_deg,
        )

    object_top_z_mm = _candidate_object_top_z_mm(candidate)
    if z_grasp_mm is not None and float(z_grasp_mm) > float(object_top_z_mm) + 1e-6:
        warnings.append("dynamic_lower_start_uses_object_top_plus_gripper_offset")

    pregrasp_robot_z_mm = (
        float(object_top_z_mm)
        + float(gripper_offset_mm)
        + float(dynamic_lower_clearance_mm)
    )
    probe_start = pregrasp_robot_z_mm
    probe_start = _clamp(probe_start, 0.0, float(z_max_mm))

    return DynamicPickPlan(
        object_class=_candidate_object_class(candidate),
        target_xy_mm=target_xy_mm,
        phi_deg=float(phi_deg),
        phi_source=phi_source,
        measured_grip_width_mm=None if measured_grip_width_mm is None else float(measured_grip_width_mm),
        grip_width_source=grip_width_source,
        initial_servo_angle_deg=float(initial_servo_angle_deg),
        servo_angle_margin_deg=float(servo_angle_margin_deg),
        dynamic_lower_start_z_mm=float(probe_start),
        dynamic_pregrasp_robot_z_mm=float(pregrasp_robot_z_mm),
        dynamic_grip_start_angle_deg=float(initial_servo_angle_deg),
        z_clearance_mm=float(dynamic_lower_clearance_mm),
        warnings=warnings,
    )
