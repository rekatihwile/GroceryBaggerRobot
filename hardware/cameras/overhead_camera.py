from __future__ import annotations

"""
overhead_camera.py

Reusable OOP wrapper for the overhead webcam.

This module only handles the camera stream and optional AprilTag detection.
It does not import or command the robot.
"""

import cv2
import numpy as np

from config.camera_config import (
    OVERHEAD_INDEX,
    apply_overhead_camera_settings,
)
from hardware.cameras.stereo_apriltag_viewer import build_detector, detect_tags, draw_detection


class SimpleOverheadCamera:
    def __init__(
        self,
        index: int = OVERHEAD_INDEX,
    ) -> None:
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open overhead camera index {index}")

        apply_overhead_camera_settings(self.cap, label="[Overhead]", index=index)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def read_detections(self, detector=None):
        ok, frame = self.read()
        if not ok:
            return False, None, {}

        if detector is None:
            detector = build_detector()

        detections = detect_tags(detector, frame)
        return True, frame, detections

    def draw_detections(self, frame: np.ndarray, detections: dict[int, object]) -> np.ndarray:
        drawn = frame
        if detections:
            for det in detections.values():
                drawn = draw_detection(drawn, det, f"OVERHEAD ID {det.marker_id}")
        else:
            drawn = draw_detection(drawn, None, "OVERHEAD no tags")
        return drawn

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
