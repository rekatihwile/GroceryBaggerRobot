from __future__ import annotations

"""vision/object_geometry.py

ObjectCandidate dataclass and related geometry helpers.

Extracted from run_pickplace_fast.py.  All module-level constants match the
original defaults.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from config.camera_config import EE_TAG_ID
from vision.yolo_segmenter import YOLODetection
from vision.pointcloud import (
    cam_points_to_robot_xyz,
    cam_xyz_to_robot_xyz,
    choose_object_target_point,
    estimate_pick_phi_from_centroid_rays,
    estimate_pick_phi_from_mask_minor_axis,
    estimate_pick_phi_from_pointcloud_shortest_path,
    estimate_pick_phi_from_pointcloud_short_side,
)

if TYPE_CHECKING:
    from hardware.robot import Robot

from test_calibration_bundle_live_stereo_z_pickplace import (
    map_uv_z_to_robot_xy,
    nearest_support_distance,
    clamp_lookup_z_to_bundle,
)


# ============================================================
# KNOBS (defaults match run_pickplace_fast.py)
# ============================================================

TAG_TO_EE_Z_MM: float = 0.0
USE_EE_FK_Z_BIAS_CORRECTION: bool = True
WARN_STEREO_Z_BIAS_MM: float = 50.0
MAX_ALLOWED_STEREO_Z_BIAS_MM: float = 80.0
TARGET_XY_SOURCE: str = "stereo_xyz"
HOVER_HEIGHT_MM: float = 215.0     # GRASP_OFFSET_MM + 50 + 50
GRASP_OFFSET_MM: float = 115.0
SURVEY_MATCH_MAX_CENTROID_PX: float = 60.0
PICK_PHI_MODE: str = "mask_minor_axis_pointcloud"


# ============================================================
# Data container
# ============================================================

@dataclass
class ObjectCandidate:
    index: int
    yolo: YOLODetection
    frame_i: int
    valid_point_count: int
    centroid_cam_xyz: np.ndarray
    top_cam_xyz: np.ndarray
    target_cam_xyz: np.ndarray
    object_robot_xyz_raw: np.ndarray
    object_robot_xyz_corrected: np.ndarray
    target_xy: np.ndarray
    target_xy_source_requested: str
    target_xy_source_effective: str
    pick_phi_deg: float | None
    pick_phi_source: str
    overhead_centroid_px: np.ndarray | None
    overhead_bbox_px: tuple[float, float, float, float] | None
    topdown_width_cm: float | None
    topdown_depth_cm: float | None
    topdown_area_cm2: float | None
    topdown_bbox_robot_xy_mm: np.ndarray | None
    topdown_aabb_cm: tuple[float, float] | None
    topdown_oriented_rect_cm: tuple[float, float] | None
    topdown_footprint_source: str | None
    pointcloud_height_cm: float | None
    pointcloud_height_robot_z_range_mm: tuple[float, float] | None
    pointcloud_footprint_cm: tuple[float, float] | None   # (width_cm, depth_cm) from stereo XY span
    lookup_z_used: float
    lookup_z_clamped: bool
    support_distance_mm: float
    support_index: int
    hover_robot_z: float
    grasp_robot_z: float
    ee_cam_xyz: np.ndarray | None
    ee_tag_robot_xyz_raw: np.ndarray | None
    ee_tool_z_from_stereo_raw: float | None
    stereo_z_bias_mm: float
    ee_stereo_visible: bool
    fk_xyz_at_update: np.ndarray
    fk_phi_at_update: float


# ============================================================
# Candidate construction
# ============================================================

def estimate_pointcloud_height_cm(
    points_cam: np.ndarray,
    bundle: Any,
) -> tuple[float | None, tuple[float, float] | None]:
    """Estimate physical object height from the stereo pointcloud Z span.

    object_robot_xyz_raw[2] is an absolute robot-frame Z plane used for pick
    and homography lookup. It is not the object's physical height. For packing
    height we transform all masked stereo points into robot XYZ and use a
    robust vertical span, so a calibration/table offset can shift the whole
    cloud below zero without forcing every item back to a default height.
    """
    if points_cam is None or len(points_cam) < 10:
        return None, None

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
    except Exception:
        return None, None

    robot_z = np.asarray(points_robot[:, 2], dtype=np.float64)
    robot_z = robot_z[np.isfinite(robot_z)]
    if len(robot_z) < 10:
        return None, None

    z_low = float(np.percentile(robot_z, 5.0))
    z_high = float(np.percentile(robot_z, 95.0))
    height_mm = z_high - z_low
    if not np.isfinite(height_mm) or height_mm <= 0.0:
        return None, None

    return float(height_mm / 10.0), (z_low, z_high)


def estimate_pointcloud_footprint_cm(
    points_cam: np.ndarray,
    bundle: Any,
) -> tuple[float, float] | None:
    """Estimate object top-down footprint (width_cm, depth_cm) from the stereo pointcloud XY span.

    Uses robot-frame X and Y percentile spans (5th–95th) so outlier disparity
    errors don't inflate the footprint estimate.  Returns None if the cloud is
    too sparse or the transform fails.
    """
    if points_cam is None or len(points_cam) < 10:
        return None

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
    except Exception:
        return None

    rx = np.asarray(points_robot[:, 0], dtype=np.float64)
    ry = np.asarray(points_robot[:, 1], dtype=np.float64)
    rx = rx[np.isfinite(rx)]
    ry = ry[np.isfinite(ry)]
    if len(rx) < 10 or len(ry) < 10:
        return None

    width_mm = float(np.percentile(rx, 95.0)) - float(np.percentile(rx, 5.0))
    depth_mm = float(np.percentile(ry, 95.0)) - float(np.percentile(ry, 5.0))
    if not (np.isfinite(width_mm) and np.isfinite(depth_mm) and width_mm > 0.0 and depth_mm > 0.0):
        return None

    return float(width_mm / 10.0), float(depth_mm / 10.0)


def build_object_candidate(
    index: int,
    yolo_det: YOLODetection,
    points_cam: np.ndarray,
    point_uv_px: np.ndarray,
    robot: "Robot",
    bundle: Any,
    stereo_tags: dict[int, Any],
    frame_i: int,
) -> ObjectCandidate:
    centroid_cam, top_cam, target_cam = choose_object_target_point(points_cam, bundle)
    raw_robot = cam_xyz_to_robot_xyz(target_cam, bundle)
    pointcloud_height_cm, pointcloud_height_range_mm = estimate_pointcloud_height_cm(
        points_cam, bundle
    )
    pointcloud_footprint_cm = estimate_pointcloud_footprint_cm(points_cam, bundle)
    if PICK_PHI_MODE == "triangulated_short_side":
        pick_phi, pick_phi_source = estimate_pick_phi_from_pointcloud_short_side(
            points_cam, bundle
        )
    elif PICK_PHI_MODE == "pointcloud_shortest_path":
        pick_phi, pick_phi_source = estimate_pick_phi_from_pointcloud_shortest_path(
            points_cam, bundle
        )
    elif PICK_PHI_MODE == "centroid_longest_ray_perp":
        pick_phi, pick_phi_source = estimate_pick_phi_from_centroid_rays(
            yolo_det,
            points_cam,
            point_uv_px,
            bundle,
            select="longest",
            perpendicular=True,
        )
    elif PICK_PHI_MODE == "centroid_shortest_ray_parallel":
        pick_phi, pick_phi_source = estimate_pick_phi_from_centroid_rays(
            yolo_det,
            points_cam,
            point_uv_px,
            bundle,
            select="shortest",
            perpendicular=False,
        )
    elif PICK_PHI_MODE == "overhead_minor_axis":
        pick_phi, pick_phi_source = None, "overhead_minor_axis_pending"
    elif PICK_PHI_MODE == "overhead_semi_minor_projected":
        pick_phi, pick_phi_source = None, "overhead_semi_minor_projected_pending"
    elif PICK_PHI_MODE == "mask_minor_axis_pointcloud":
        pick_phi, pick_phi_source = estimate_pick_phi_from_mask_minor_axis(
            yolo_det, points_cam, point_uv_px, bundle
        )
    elif PICK_PHI_MODE == "current_fk":
        pick_phi, pick_phi_source = None, "current_fk"
    else:
        raise ValueError(f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}")

    candidate = ObjectCandidate(
        index=index,
        yolo=yolo_det,
        frame_i=frame_i,
        valid_point_count=int(len(points_cam)),
        centroid_cam_xyz=centroid_cam,
        top_cam_xyz=top_cam,
        target_cam_xyz=target_cam,
        object_robot_xyz_raw=raw_robot,
        object_robot_xyz_corrected=raw_robot.copy(),
        target_xy=raw_robot[:2].copy(),
        target_xy_source_requested=TARGET_XY_SOURCE,
        target_xy_source_effective="stereo_xyz",
        pick_phi_deg=pick_phi,
        pick_phi_source=pick_phi_source,
        overhead_centroid_px=None,
        overhead_bbox_px=None,
        topdown_width_cm=None,
        topdown_depth_cm=None,
        topdown_area_cm2=None,
        topdown_bbox_robot_xy_mm=None,
        topdown_aabb_cm=None,
        topdown_oriented_rect_cm=None,
        topdown_footprint_source=None,
        pointcloud_height_cm=pointcloud_height_cm,
        pointcloud_height_robot_z_range_mm=pointcloud_height_range_mm,
        pointcloud_footprint_cm=pointcloud_footprint_cm,
        lookup_z_used=float(raw_robot[2]),
        lookup_z_clamped=False,
        support_distance_mm=float("inf"),
        support_index=-1,
        hover_robot_z=float(raw_robot[2] + HOVER_HEIGHT_MM),
        grasp_robot_z=float(raw_robot[2] + GRASP_OFFSET_MM),
        ee_cam_xyz=None,
        ee_tag_robot_xyz_raw=None,
        ee_tool_z_from_stereo_raw=None,
        stereo_z_bias_mm=0.0,
        ee_stereo_visible=False,
        fk_xyz_at_update=np.zeros(3, dtype=np.float64),
        fk_phi_at_update=0.0,
    )
    refresh_candidate_z_bias(candidate, robot, bundle, stereo_tags, warn=False)
    return candidate


def refresh_candidate_z_bias(
    candidate: ObjectCandidate,
    robot: "Robot",
    bundle: Any,
    stereo_tags: dict[int, Any] | None,
    *,
    warn: bool = True,
) -> None:
    x_fk, y_fk, z_fk, phi_fk = robot.fk()
    candidate.fk_xyz_at_update = np.array([x_fk, y_fk, z_fk], dtype=np.float64)
    candidate.fk_phi_at_update = float(phi_fk)

    ee_tri = (stereo_tags or {}).get(EE_TAG_ID)
    ee_raw = None
    ee_tool_z_raw = None
    stereo_z_bias = 0.0

    if ee_tri is not None:
        ee_raw = cam_xyz_to_robot_xyz(ee_tri.xyz_cam_mm, bundle)
        ee_tool_z_raw = float(ee_raw[2] + TAG_TO_EE_Z_MM)
        if USE_EE_FK_Z_BIAS_CORRECTION:
            stereo_z_bias = float(z_fk - ee_tool_z_raw)

    corrected = candidate.object_robot_xyz_raw.copy()
    corrected[2] += stereo_z_bias

    lookup_z_used, lookup_z_clamped = clamp_lookup_z_to_bundle(float(corrected[2]), bundle)

    xy_source_effective = TARGET_XY_SOURCE
    if TARGET_XY_SOURCE == "overhead_homography" and candidate.overhead_centroid_px is not None:
        target_xy, _, _, _, _ = map_uv_z_to_robot_xy(
            candidate.overhead_centroid_px, lookup_z_used, bundle
        )
    else:
        if TARGET_XY_SOURCE == "overhead_homography":
            xy_source_effective = "stereo_xyz"
        target_xy = corrected[:2].copy()

    support_dist, support_idx = nearest_support_distance(target_xy, lookup_z_used, bundle)

    candidate.object_robot_xyz_corrected = corrected
    candidate.target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    candidate.target_xy_source_requested = TARGET_XY_SOURCE
    candidate.target_xy_source_effective = xy_source_effective
    candidate.lookup_z_used = float(lookup_z_used)
    candidate.lookup_z_clamped = bool(lookup_z_clamped)
    candidate.support_distance_mm = float(support_dist)
    candidate.support_index = int(support_idx)
    candidate.hover_robot_z = float(corrected[2] + HOVER_HEIGHT_MM)
    candidate.grasp_robot_z = float(corrected[2] + GRASP_OFFSET_MM)
    candidate.ee_cam_xyz = None if ee_tri is None else ee_tri.xyz_cam_mm.copy()
    candidate.ee_tag_robot_xyz_raw = ee_raw
    candidate.ee_tool_z_from_stereo_raw = ee_tool_z_raw
    candidate.stereo_z_bias_mm = float(stereo_z_bias)
    candidate.ee_stereo_visible = ee_tri is not None

    if warn and ee_tri is not None and abs(stereo_z_bias) > WARN_STEREO_Z_BIAS_MM:
        print(
            f"[Z BIAS WARN] stereo_z_bias={stereo_z_bias:+.1f} mm "
            f"(warn>{WARN_STEREO_Z_BIAS_MM:.1f}, refuse>{MAX_ALLOWED_STEREO_Z_BIAS_MM:.1f})"
        )


# ============================================================
# Candidate ranking / fusion
# ============================================================

def candidate_score(candidate: ObjectCandidate) -> float:
    return float(candidate.yolo.confidence) * float(
        np.log1p(float(candidate.valid_point_count))
    )


def pre_pointcloud_detection_score(det: YOLODetection) -> float:
    x1, y1, x2, y2 = det.bbox
    bbox_area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    return float(det.confidence) * np.log1p(float(det.mask_area)) * np.log1p(bbox_area)


def fuse_survey_candidates(raw_candidates: list[ObjectCandidate]) -> list[ObjectCandidate]:
    """Merge per-frame candidates by class + centroid proximity; return top-9."""
    grouped: list[ObjectCandidate] = []
    for cand in sorted(raw_candidates, key=candidate_score, reverse=True):
        replaced = False
        for i, existing in enumerate(grouped):
            same_class = cand.yolo.class_name == existing.yolo.class_name
            px_dist = float(
                np.linalg.norm(cand.yolo.centroid_px - existing.yolo.centroid_px)
            )
            if same_class and px_dist <= SURVEY_MATCH_MAX_CENTROID_PX:
                if candidate_score(cand) > candidate_score(existing):
                    grouped[i] = cand
                replaced = True
                break
        if not replaced:
            grouped.append(cand)

    grouped = sorted(grouped, key=candidate_score, reverse=True)[:9]
    for i, cand in enumerate(grouped, start=1):
        cand.index = i
    return grouped
