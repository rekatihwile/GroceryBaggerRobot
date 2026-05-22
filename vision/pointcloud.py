from __future__ import annotations

"""vision/pointcloud.py

Stereo point cloud extraction and target point selection.

Extracted from run_pickplace_fast.py.  Module-level constants match the knobs
in run_pickplace_fast.py so importing callers need no changes.
"""

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from vision.stereo_rectifier import _rectified_xyz_to_bundle_cam_convention, _available_calib_keys_message

if TYPE_CHECKING:
    from vision.yolo_segmenter import YOLODetection


# ============================================================
# KNOBS (defaults match run_pickplace_fast.py)
# ============================================================

MIN_DISPARITY_PX: float = 1.0
POINTCLOUD_MAX_POINTS: int = 20000
POINTCLOUD_Z_MIN_MM: float = -2000.0
POINTCLOUD_Z_MAX_MM: float = 2000.0
POINTCLOUD_REMOVE_OUTLIERS: bool = True

OBJECT_TARGET_MODE: str = "centroid"        # "centroid" | "top_surface"
TOP_SURFACE_PERCENTILE: float = 90.0
TOP_SURFACE_MEDIAN_BAND_MM: float = 10.0

USE_OBJECT_MASK_MINOR_AXIS_PHI: bool = True
PHI_AXIS_ENDPOINT_PERCENTILE: float = 15.0
PHI_AXIS_MIN_POINTS_PER_SIDE: int = 8


# ============================================================
# Point cloud extraction
# ============================================================

def masked_disparity_to_pointcloud(
    mask: np.ndarray,
    disparity: np.ndarray,
    stereo_calib: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Project a mask-filtered rectified disparity map into stereo camera XYZ.

    Returns:
        points_cam_mm — Nx3 points in the left-camera convention used by the
                        calibration bundle (matches A_robot_from_cam_xyz_3x4).
        uv_px         — Nx2 rectified stereo-left pixel coordinates.
    """
    if mask.shape != disparity.shape[:2]:
        raise ValueError(f"mask/disparity shape mismatch: {mask.shape} vs {disparity.shape}")

    mask_bool = mask.astype(bool)
    disp = np.asarray(disparity, dtype=np.float32)
    valid = mask_bool & np.isfinite(disp) & (disp > MIN_DISPARITY_PX)

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    if len(xs) > POINTCLOUD_MAX_POINTS:
        step = int(np.ceil(len(xs) / POINTCLOUD_MAX_POINTS))
        xs = xs[::step][:POINTCLOUD_MAX_POINTS]
        ys = ys[::step][:POINTCLOUD_MAX_POINTS]

    d = disp[ys, xs].astype(np.float64)
    u = xs.astype(np.float64)
    v = ys.astype(np.float64)

    if "disparity_to_depth_Q" in stereo_calib:
        q = np.asarray(stereo_calib["disparity_to_depth_Q"], dtype=np.float64)
        pts_h = np.column_stack([u, v, d, np.ones_like(d)]) @ q.T
        w = pts_h[:, 3]
        good_w = np.isfinite(w) & (np.abs(w) > 1e-9)
        xyz_rect = pts_h[good_w, :3] / w[good_w, None]
        uv = np.column_stack([u[good_w], v[good_w]])
    else:
        required = ["projection_left_rectified"]
        missing = [k for k in required if k not in stereo_calib]
        if missing:
            print(_available_calib_keys_message(stereo_calib))
            raise KeyError(f"Cannot project disparity: missing {missing}")

        p_left = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
        fx = float(p_left[0, 0])
        fy = float(p_left[1, 1])
        cx = float(p_left[0, 2])
        cy = float(p_left[1, 2])

        if "baseline_mm" in stereo_calib:
            baseline = float(np.asarray(stereo_calib["baseline_mm"]).reshape(-1)[0])
        elif "projection_right_rectified" in stereo_calib:
            p_right = np.asarray(stereo_calib["projection_right_rectified"], dtype=np.float64)
            baseline = abs(float(p_right[0, 3]) / fx)
        else:
            print(_available_calib_keys_message(stereo_calib))
            raise KeyError("Cannot project disparity: missing baseline_mm and projection_right_rectified")

        z = fx * baseline / d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz_rect = np.column_stack([x, y, z])
        uv = np.column_stack([u, v])

    points_cam = _rectified_xyz_to_bundle_cam_convention(xyz_rect, stereo_calib)
    finite = np.all(np.isfinite(points_cam), axis=1)
    z_ok = (points_cam[:, 2] >= POINTCLOUD_Z_MIN_MM) & (points_cam[:, 2] <= POINTCLOUD_Z_MAX_MM)
    keep = finite & z_ok
    points_cam = points_cam[keep]
    uv = uv[keep]

    if POINTCLOUD_REMOVE_OUTLIERS and len(points_cam) >= 30:
        points_cam, uv = remove_pointcloud_outliers(points_cam, uv)

    return points_cam, uv


def remove_pointcloud_outliers(
    points_cam: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(points_cam, axis=0)
    dist = np.linalg.norm(points_cam - med.reshape(1, 3), axis=1)
    cutoff = float(np.percentile(dist, 95.0))
    keep = dist <= max(cutoff, 1e-6)
    return points_cam[keep], uv[keep]


# ============================================================
# Robot XYZ transform helpers
# ============================================================

def cam_points_to_robot_xyz(points_cam: np.ndarray, bundle: Any) -> np.ndarray:
    """Transform Nx3 camera points to Nx3 robot XYZ using the calibration bundle."""
    a = bundle.get("A_robot_from_cam_xyz_3x4")
    if a is None:
        raise RuntimeError("Bundle has no A_robot_from_cam_xyz_3x4. Re-run XYZ calibration.")
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return pts_h @ np.asarray(a, dtype=np.float64).T


def cam_xyz_to_robot_xyz(cam_xyz: np.ndarray, bundle: Any) -> np.ndarray:
    """Transform a single 3-element camera XYZ to robot XYZ."""
    pts = np.asarray(cam_xyz, dtype=np.float64).reshape(1, 3)
    return cam_points_to_robot_xyz(pts, bundle).reshape(3)


# ============================================================
# Target point selection
# ============================================================

def choose_object_target_point(
    points_cam: np.ndarray, bundle: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose centroid, top surface, and final target point from a point cloud.

    Returns:
        centroid_cam — median of all cloud points in camera frame
        top_cam      — median of top-surface band in camera frame
        target_cam   — selected by OBJECT_TARGET_MODE
    """
    if len(points_cam) == 0:
        raise ValueError("No valid object points.")

    centroid_cam = np.median(points_cam, axis=0).astype(np.float64)
    top_cam = centroid_cam.copy()

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
        robot_z = points_robot[:, 2]
        z_cut = float(np.percentile(robot_z, TOP_SURFACE_PERCENTILE))
        band = robot_z >= (z_cut - TOP_SURFACE_MEDIAN_BAND_MM)
        if int(band.sum()) >= 5:
            top_cam = np.median(points_cam[band], axis=0).astype(np.float64)
    except Exception as exc:
        print(f"[POINTCLOUD] Top-surface extraction failed; using centroid: {exc}")

    if OBJECT_TARGET_MODE == "top_surface":
        target_cam = top_cam
    elif OBJECT_TARGET_MODE == "centroid":
        target_cam = centroid_cam
    else:
        raise ValueError(f"Unknown OBJECT_TARGET_MODE={OBJECT_TARGET_MODE!r}")

    return centroid_cam, top_cam, target_cam.astype(np.float64)


# ============================================================
# Pick angle helpers
# ============================================================

def normalize_phi_0_180(phi_deg: float) -> float:
    return float(phi_deg % 180.0)


def estimate_pick_phi_from_mask_minor_axis(
    yolo_det: "YOLODetection",
    points_cam: np.ndarray,
    point_uv_px: np.ndarray,
    bundle: Any,
) -> tuple[float | None, str]:
    """Estimate pick wrist angle from the mask semi-minor axis projected into robot XY.

    Returns:
        (phi_deg, source_label)  where phi_deg is None if the estimate fails.
    """
    if not USE_OBJECT_MASK_MINOR_AXIS_PHI:
        return None, "current_fk"

    if (
        point_uv_px is None
        or len(point_uv_px) != len(points_cam)
        or len(points_cam) < (2 * PHI_AXIS_MIN_POINTS_PER_SIDE)
    ):
        return None, "mask_minor_axis_unavailable"

    theta = np.radians(float(yolo_det.minor_axis_angle_deg))
    axis_uv = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    uv_offsets = (
        np.asarray(point_uv_px, dtype=np.float64).reshape(-1, 2)
        - yolo_det.centroid_px.reshape(1, 2)
    )
    proj = uv_offsets @ axis_uv

    lo_q = float(np.percentile(proj, PHI_AXIS_ENDPOINT_PERCENTILE))
    hi_q = float(np.percentile(proj, 100.0 - PHI_AXIS_ENDPOINT_PERCENTILE))
    low_mask = proj <= lo_q
    high_mask = proj >= hi_q

    if (
        int(low_mask.sum()) < PHI_AXIS_MIN_POINTS_PER_SIDE
        or int(high_mask.sum()) < PHI_AXIS_MIN_POINTS_PER_SIDE
    ):
        return None, "mask_minor_axis_too_few_endpoint_points"

    low_cam = np.median(points_cam[low_mask], axis=0)
    high_cam = np.median(points_cam[high_mask], axis=0)
    low_robot = cam_xyz_to_robot_xyz(low_cam, bundle)
    high_robot = cam_xyz_to_robot_xyz(high_cam, bundle)
    delta_xy = high_robot[:2] - low_robot[:2]

    if not np.all(np.isfinite(delta_xy)) or float(np.linalg.norm(delta_xy)) < 1e-6:
        return None, "mask_minor_axis_degenerate_robot_delta"

    phi = normalize_phi_0_180(float(np.degrees(np.arctan2(delta_xy[1], delta_xy[0]))))
    return phi, "mask_minor_axis_pointcloud"


def estimate_pick_phi_from_pointcloud_short_side(
    points_cam: np.ndarray,
    bundle: Any,
) -> tuple[float | None, str]:
    """Estimate wrist angle along the shortest triangulated object side.

    This uses PCA on the object point cloud after transforming it into robot
    XY. The eigenvector with the smaller variance is treated as the short side.
    """
    if points_cam is None or len(points_cam) < (2 * PHI_AXIS_MIN_POINTS_PER_SIDE):
        return None, "pointcloud_short_side_unavailable"

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
    except Exception as exc:
        print(f"[POINTCLOUD] Short-side phi failed; transform error: {exc}")
        return None, "pointcloud_short_side_transform_failed"

    xy = np.asarray(points_robot[:, :2], dtype=np.float64)
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if len(xy) < (2 * PHI_AXIS_MIN_POINTS_PER_SIDE):
        return None, "pointcloud_short_side_too_few_points"

    centered = xy - np.median(xy, axis=0).reshape(1, 2)
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None, "pointcloud_short_side_degenerate"

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    short_vec = eigenvectors[:, int(np.argmin(eigenvalues))]
    if not np.all(np.isfinite(short_vec)) or float(np.linalg.norm(short_vec)) < 1e-9:
        return None, "pointcloud_short_side_degenerate"

    phi = normalize_phi_0_180(float(np.degrees(np.arctan2(short_vec[1], short_vec[0]))))
    return phi, "pointcloud_short_side"
