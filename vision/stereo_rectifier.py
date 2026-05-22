from __future__ import annotations

"""vision/stereo_rectifier.py

Stereo rectification helper.

Extracted from run_pickplace_fast.py.
"""

import cv2
import numpy as np


class StereoRectifier:
    """Lazily builds and caches OpenCV rectification maps per image size."""

    _REQUIRED_KEYS = [
        "left_camera_matrix",
        "left_distortion_coefficients",
        "right_camera_matrix",
        "right_distortion_coefficients",
        "rectification_left",
        "rectification_right",
        "projection_left_rectified",
        "projection_right_rectified",
    ]

    def __init__(self, stereo_calib: dict[str, np.ndarray]) -> None:
        self.calib = stereo_calib
        self._maps_by_size: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

        missing = [k for k in self._REQUIRED_KEYS if k not in stereo_calib]
        if missing:
            raise KeyError(
                f"Stereo calibration missing keys: {missing}. "
                f"{_available_calib_keys_message(stereo_calib)}"
            )

    def rectify(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError(
                f"Stereo frame shape mismatch: {left_bgr.shape} vs {right_bgr.shape}"
            )
        h, w = left_bgr.shape[:2]
        key = (w, h)
        if key not in self._maps_by_size:
            self._maps_by_size[key] = self._build_maps(w, h)
        map_lx, map_ly, map_rx, map_ry = self._maps_by_size[key]
        left_rect = cv2.remap(left_bgr, map_lx, map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_bgr, map_rx, map_ry, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def _build_maps(
        self, w: int, h: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self.calib
        map_lx, map_ly = cv2.initUndistortRectifyMap(
            np.asarray(c["left_camera_matrix"], dtype=np.float64),
            np.asarray(c["left_distortion_coefficients"], dtype=np.float64),
            np.asarray(c["rectification_left"], dtype=np.float64),
            np.asarray(c["projection_left_rectified"], dtype=np.float64),
            (int(w), int(h)),
            cv2.CV_32FC1,
        )
        map_rx, map_ry = cv2.initUndistortRectifyMap(
            np.asarray(c["right_camera_matrix"], dtype=np.float64),
            np.asarray(c["right_distortion_coefficients"], dtype=np.float64),
            np.asarray(c["rectification_right"], dtype=np.float64),
            np.asarray(c["projection_right_rectified"], dtype=np.float64),
            (int(w), int(h)),
            cv2.CV_32FC1,
        )
        print(f"[Stereo] Built rectification maps for {w}x{h}")
        return map_lx, map_ly, map_rx, map_ry


# ------------------------------------------------------------------ #
# Module-level helpers (also used by vision/pointcloud.py)
# ------------------------------------------------------------------ #

def _available_calib_keys_message(stereo_calib: dict[str, np.ndarray]) -> str:
    return "Available keys: " + ", ".join(sorted(stereo_calib.keys()))


def _rectified_xyz_to_bundle_cam_convention(
    xyz_rect_mm: np.ndarray,
    stereo_calib: dict[str, np.ndarray],
) -> np.ndarray:
    """Rotate rectified-frame XYZ back to the original left-camera convention used by
    the calibration bundle, then flip Z to match the existing bundle sign convention.
    """
    xyz = np.asarray(xyz_rect_mm, dtype=np.float64).copy()
    if "rectification_left" in stereo_calib:
        r_left = np.asarray(stereo_calib["rectification_left"], dtype=np.float64)
        xyz = xyz @ r_left
    xyz[:, 2] *= -1.0
    return xyz
