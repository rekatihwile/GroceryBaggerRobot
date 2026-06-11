from __future__ import annotations

"""Recover per-point RGB from the rectified-left source photograph."""

from pathlib import Path

import numpy as np


def load_stereo_calibration(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return None


def project_bundle_camera_points_to_uv(
    points_cam: np.ndarray,
    stereo_calib: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    """Project saved bundle-convention camera XYZ back to rectified-left pixels."""
    if stereo_calib is None or "projection_left_rectified" not in stereo_calib:
        return None
    points = np.asarray(points_cam, dtype=np.float64).copy().reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    points[:, 2] *= -1.0
    if "rectification_left" in stereo_calib:
        points = points @ np.asarray(stereo_calib["rectification_left"], dtype=np.float64).T
    projection = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1.0)
    safe_z = np.where(valid, z, 1.0)
    u = np.where(valid, projection[0, 0] * points[:, 0] / safe_z + projection[0, 2], np.nan)
    v = np.where(valid, projection[1, 1] * points[:, 1] / safe_z + projection[1, 2], np.nan)
    return np.column_stack([u, v])


def photo_colors_for_points(
    image_rgb: np.ndarray | None,
    uv_px: np.ndarray | None,
    point_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return RGB for in-frame points and the corresponding point mask."""
    if image_rgb is None or uv_px is None or len(uv_px) != point_count:
        return None, None
    height, width = image_rgb.shape[:2]
    valid = (
        np.all(np.isfinite(uv_px[:, :2]), axis=1)
        & (uv_px[:, 0] >= 0.0)
        & (uv_px[:, 0] <= width - 1)
        & (uv_px[:, 1] >= 0.0)
        & (uv_px[:, 1] <= height - 1)
    )
    if not np.any(valid):
        return None, valid
    x = np.rint(uv_px[valid, 0]).astype(np.int32)
    y = np.rint(uv_px[valid, 1]).astype(np.int32)
    colors = np.clip(image_rgb[y, x, :3], 0.0, 1.0).astype(np.float32)
    return colors, valid
