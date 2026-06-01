from __future__ import annotations

"""Axis-aligned 3D box helpers for simple placement occupancy planning."""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _vec3(values: Any, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{label} must have at least 3 values")
    out = arr[:3].astype(np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} must contain finite values")
    return out


@dataclass
class AxisAlignedBox3D:
    center_xyz_mm: np.ndarray
    size_xyz_mm: np.ndarray
    min_xyz_mm: np.ndarray
    max_xyz_mm: np.ndarray
    label: str = ""


@dataclass
class PaddedBox3D:
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D
    padding_xyz_mm: np.ndarray


def make_aabb_from_center_size(center_xyz_mm, size_xyz_mm, label: str = "") -> AxisAlignedBox3D:
    center = _vec3(center_xyz_mm, "center_xyz_mm")
    size = _vec3(size_xyz_mm, "size_xyz_mm")
    if np.any(size <= 0.0):
        raise ValueError("size_xyz_mm must be strictly positive")
    half = 0.5 * size
    min_xyz = center - half
    max_xyz = center + half
    return AxisAlignedBox3D(
        center_xyz_mm=center,
        size_xyz_mm=size,
        min_xyz_mm=min_xyz,
        max_xyz_mm=max_xyz,
        label=str(label),
    )


def make_aabb_from_min_max(min_xyz_mm, max_xyz_mm, label: str = "") -> AxisAlignedBox3D:
    min_xyz = _vec3(min_xyz_mm, "min_xyz_mm")
    max_xyz = _vec3(max_xyz_mm, "max_xyz_mm")
    size = max_xyz - min_xyz
    if np.any(size <= 0.0):
        raise ValueError("max_xyz_mm must be greater than min_xyz_mm on all axes")
    center = 0.5 * (min_xyz + max_xyz)
    return AxisAlignedBox3D(
        center_xyz_mm=center,
        size_xyz_mm=size,
        min_xyz_mm=min_xyz,
        max_xyz_mm=max_xyz,
        label=str(label),
    )


def pad_aabb(
    box: AxisAlignedBox3D,
    pad_x_mm: float,
    pad_y_mm: float,
    pad_z_mm: float,
    label_suffix: str = "_padded",
) -> PaddedBox3D:
    pad = _vec3([pad_x_mm, pad_y_mm, pad_z_mm], "padding_xyz_mm")
    if np.any(pad < 0.0):
        raise ValueError("padding must be non-negative")
    padded_size = box.size_xyz_mm + 2.0 * pad
    padded = make_aabb_from_center_size(
        center_xyz_mm=box.center_xyz_mm,
        size_xyz_mm=padded_size,
        label=f"{box.label}{label_suffix}",
    )
    return PaddedBox3D(raw_box=box, padded_box=padded, padding_xyz_mm=pad)


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _candidate_size_mm(candidate) -> tuple[np.ndarray, str]:
    # Priority 1: overhead-projected AABB footprint (most reliable when available).
    width_mm = None
    depth_mm = None
    top_w = _finite_or_none(getattr(candidate, "topdown_width_cm", None))
    top_d = _finite_or_none(getattr(candidate, "topdown_depth_cm", None))
    if top_w is not None and top_w > 0.0 and top_d is not None and top_d > 0.0:
        width_mm = top_w * 10.0
        depth_mm = top_d * 10.0
        xy_source = "topdown_width_depth_cm"
    else:
        # Priority 2: PCA-oriented minimum bounding rectangle — gives the actual
        # physical dimensions of the object regardless of its orientation relative
        # to the robot X/Y axes, so the AABB is as tight as possible.
        oriented = getattr(candidate, "topdown_oriented_rect_cm", None)
        if oriented is not None:
            try:
                ow = _finite_or_none(oriented[0])
                od = _finite_or_none(oriented[1])
            except Exception:
                ow, od = None, None
            if ow is not None and ow > 0.0 and od is not None and od > 0.0:
                # PCA tends to slightly overestimate because the point cloud
                # extends to the outer boundary of the measurement noise.
                # Trim ~4 % so items that are within sensor noise of the bag
                # boundary still get floor placements instead of being forced
                # to stack.  The trim is small enough that collision avoidance
                # is not meaningfully degraded.
                _PCA_TRIM = 0.96
                width_mm = ow * 10.0 * _PCA_TRIM
                depth_mm = od * 10.0 * _PCA_TRIM
                xy_source = "pca_oriented_rect_cm"

        # Priority 3: axis-aligned pointcloud span (fallback — inflated for
        # diagonal objects but always available).
        if width_mm is None:
            footprint = getattr(candidate, "pointcloud_footprint_cm", None)
            if footprint is not None:
                try:
                    fw = _finite_or_none(footprint[0])
                    fd = _finite_or_none(footprint[1])
                except Exception:
                    fw, fd = None, None
                if fw is not None and fw > 0.0 and fd is not None and fd > 0.0:
                    width_mm = fw * 10.0
                    depth_mm = fd * 10.0
                    xy_source = "pointcloud_footprint_cm"
                else:
                    xy_source = "none"
            else:
                xy_source = "none"

    if width_mm is None or depth_mm is None:
        width_mm = 80.0
        depth_mm = 80.0
        xy_source = "default_xy_80mm"

    height_mm = None
    z_debug = getattr(candidate, "z_debug", None)
    if z_debug is not None:
        z_h = _finite_or_none(getattr(z_debug, "object_height_mm", None))
        if z_h is not None and z_h > 0.0:
            height_mm = z_h
            z_source = "z_debug.object_height_mm"
        else:
            z_source = "none"
    else:
        z_source = "none"

    if height_mm is None:
        pc_h_cm = _finite_or_none(getattr(candidate, "pointcloud_height_cm", None))
        if pc_h_cm is not None and pc_h_cm > 0.0:
            height_mm = pc_h_cm * 10.0
            z_source = "pointcloud_height_cm"

    if height_mm is None:
        z_rng = getattr(candidate, "pointcloud_height_robot_z_range_mm", None)
        if z_rng is not None:
            try:
                z0 = _finite_or_none(z_rng[0])
                z1 = _finite_or_none(z_rng[1])
            except Exception:
                z0, z1 = None, None
            if z0 is not None and z1 is not None:
                dh = abs(z1 - z0)
                if dh > 0.0:
                    height_mm = dh
                    z_source = "pointcloud_height_robot_z_range_mm"

    if height_mm is None:
        height_mm = 60.0
        z_source = "default_z_60mm"

    size = np.array([
        max(1.0, float(width_mm)),
        max(1.0, float(depth_mm)),
        max(1.0, float(height_mm)),
    ], dtype=np.float64)
    return size, f"xy={xy_source},z={z_source}"


def aabb_from_object_candidate(candidate, default_label: str = "object") -> AxisAlignedBox3D:
    if candidate is None:
        raise ValueError("candidate is required")

    # Accept either ObjectCandidate or CandidateDebug-like wrappers.
    if hasattr(candidate, "candidate") and getattr(candidate, "object_robot_xyz_corrected", None) is None:
        inner = getattr(candidate, "candidate")
        if inner is not None:
            candidate = inner

    center_src = getattr(candidate, "object_robot_xyz_corrected", None)
    if center_src is None:
        center_src = getattr(candidate, "object_robot_xyz_raw", None)
    if center_src is None:
        raise ValueError("candidate missing object robot XYZ")

    center = _vec3(center_src, "candidate.object_robot_xyz")
    size, source = _candidate_size_mm(candidate)

    cand_label = str(getattr(getattr(candidate, "yolo", None), "class_name", "") or "").strip()
    label = cand_label if cand_label else str(default_label)
    box = make_aabb_from_center_size(center, size, label=label)
    print(
        "[AABB] "
        f"label={box.label} "
        f"center=({box.center_xyz_mm[0]:.1f},{box.center_xyz_mm[1]:.1f},{box.center_xyz_mm[2]:.1f}) "
        f"size=({box.size_xyz_mm[0]:.1f},{box.size_xyz_mm[1]:.1f},{box.size_xyz_mm[2]:.1f}) "
        f"source={source}"
    )
    return box
