from __future__ import annotations

"""
stereo_apriltag_viewer.py

Clean first build-up script:
- Opens the single USB stereo stream at 1280x480
- Splits it into left/right 640x480 images
- Detects an AprilTag / ArUco marker on the end effector
- Draws the tag center and tag orientation in both images
- Prints left/right pixel centers and image-plane theta

This intentionally does NOT depend on your older config.py or StereoCameraSystem.
True metric X/Y/Z requires stereo calibration, which we will add after this sanity check.
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config.camera_config import (
    EE_TAG_ID as TAG_ID,
    STEREO_FOURCC as FOURCC,
    STEREO_FPS as FPS,
    STEREO_HEIGHT as STREAM_HEIGHT,
    STEREO_INDEX as CAMERA_INDEX,
    STEREO_WIDTH as STREAM_WIDTH,
    apply_stereo_camera_settings,
)


# -----------------------------
# User settings
# -----------------------------

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_25h9

DISPLAY_SCALE = 1.0
PRINT_EVERY_SEC = 0.15


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class TagDetection:
    marker_id: int
    center: np.ndarray      # shape (2,), [u, v]
    corners: np.ndarray     # shape (4, 2)
    theta_deg: float        # image-plane orientation of top edge


# -----------------------------
# Camera
# -----------------------------

class SimpleStereoCamera:
    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = STREAM_WIDTH,
        height: int = STREAM_HEIGHT,
        fps: int = FPS,
        fourcc: str = FOURCC,
    ) -> None:
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")

        apply_stereo_camera_settings(
            self.cap,
            label="[Stereo]",
            index=index,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_w % 2 != 0:
            print("[WARN] Frame width is not even. Stereo split may be wrong.")

        if actual_w < actual_h * 2:
            print("[WARN] Frame does not look like side-by-side stereo. Check camera mode.")

    def read_pair(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None, None, None

        mid = frame.shape[1] // 2
        left = frame[:, :mid].copy()
        right = frame[:, mid:].copy()

        return True, frame, left, right

    def capture_burst(self, n: int = 5, delay_s: float = 0.05) -> list[dict]:
        """Capture n stereo frame pairs.

        Returns a list of dicts, each with keys:
            "frame"  — full side-by-side frame (HxW)
            "left"   — left half (Hx(W//2))
            "right"  — right half (Hx(W//2))

        Frames where read fails are skipped silently.
        """
        import time
        frames: list[dict] = []
        for i in range(int(n)):
            ok, frame, left, right = self.read_pair()
            if ok and frame is not None:
                frames.append({"frame": frame, "left": left, "right": right})
            if i < int(n) - 1 and delay_s > 0:
                time.sleep(delay_s)
        return frames

    def release(self):
        self.cap.release()

    # Context manager support
    def __enter__(self) -> "SimpleStereoCamera":
        return self

    def __exit__(self, *args) -> None:
        self.release()


# -----------------------------
# AprilTag detection
# -----------------------------

def build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICT)
    params = cv2.aruco.DetectorParameters()

    # These are mild robustness tweaks. Keep simple.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30

    return cv2.aruco.ArucoDetector(dictionary, params)


def compute_tag_orientation_deg(corners: np.ndarray) -> float:
    pts = corners.reshape(4, 2).astype(np.float64)

    # OpenCV order is usually:
    # 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left
    top_left = pts[0]
    top_right = pts[1]

    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]

    return float(np.degrees(np.arctan2(dy, dx)))


def detect_tags(detector, image_bgr: np.ndarray) -> dict[int, TagDetection]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners_list, ids, _ = detector.detectMarkers(gray)

    detections: dict[int, TagDetection] = {}

    if ids is None:
        return detections

    for corners, marker_id in zip(corners_list, ids.flatten()):
        pts = corners.reshape(4, 2).astype(np.float32)
        center = pts.mean(axis=0)
        theta_deg = compute_tag_orientation_deg(pts)

        detections[int(marker_id)] = TagDetection(
            marker_id=int(marker_id),
            center=center,
            corners=pts,
            theta_deg=theta_deg,
        )

    return detections


def average_angles_deg(a: float, b: float) -> float:
    ar = np.radians(a)
    br = np.radians(b)

    x = np.cos(ar) + np.cos(br)
    y = np.sin(ar) + np.sin(br)

    return float(np.degrees(np.arctan2(y, x)))


def draw_detection(image_bgr: np.ndarray, det: TagDetection | None, label: str) -> np.ndarray:
    out = image_bgr.copy()

    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    if det is None:
        return out

    corners_i = np.round(det.corners).astype(int)
    center_i = tuple(np.round(det.center).astype(int))

    cv2.polylines(out, [corners_i], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(out, center_i, 5, (0, 0, 255), -1, cv2.LINE_AA)

    # Draw tag-local axes
    pts = det.corners.astype(np.float64)
    center = pts.mean(axis=0)

    x_vec = pts[1] - pts[0]
    y_vec = pts[3] - pts[0]

    x_norm = np.linalg.norm(x_vec)
    y_norm = np.linalg.norm(y_vec)

    if x_norm > 1e-9:
        x_vec /= x_norm
    if y_norm > 1e-9:
        y_vec /= y_norm

    axis_len = 45.0
    x_end = tuple(np.round(center + axis_len * x_vec).astype(int))
    y_end = tuple(np.round(center + axis_len * y_vec).astype(int))
    c0 = tuple(np.round(center).astype(int))

    cv2.arrowedLine(out, c0, x_end, (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(out, c0, y_end, (255, 0, 0), 3, cv2.LINE_AA, tipLength=0.25)

    u, v = det.center
    cv2.putText(
        out,
        f"ID {det.marker_id}  u={u:.1f}, v={v:.1f}, th={det.theta_deg:+.1f}",
        (12, out.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_preview(left_draw: np.ndarray, right_draw: np.ndarray, lines: list[str]) -> np.ndarray:
    if DISPLAY_SCALE != 1.0:
        left_draw = cv2.resize(left_draw, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        right_draw = cv2.resize(right_draw, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

    divider = np.full((left_draw.shape[0], 6, 3), 32, dtype=np.uint8)
    preview = cv2.hconcat([left_draw, divider, right_draw])

    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(preview, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    return preview


# -----------------------------
# Main
# -----------------------------

def main():
    detector = build_detector()
    stereo = SimpleStereoCamera()

    cv2.namedWindow("Stereo AprilTag Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stereo AprilTag Viewer", 1280, 520)

    last_print = 0.0

    try:
        while True:
            ok, frame, left, right = stereo.read_pair()
            if not ok:
                print("[WARN] Could not read frame.")
                continue

            det_l_all = detect_tags(detector, left)
            det_r_all = detect_tags(detector, right)

            det_l = det_l_all.get(TAG_ID)
            det_r = det_r_all.get(TAG_ID)

            left_draw = draw_detection(left, det_l, "LEFT")
            right_draw = draw_detection(right, det_r, "RIGHT")

            lines = [
                f"tag25h9 ID {TAG_ID}",
                f"left detected: {det_l is not None} | right detected: {det_r is not None}",
                "q/ESC = quit",
            ]

            now = time.time()

            if det_l is not None and det_r is not None:
                ul, vl = det_l.center
                ur, vr = det_r.center
                disparity = ul - ur
                theta_avg = average_angles_deg(det_l.theta_deg, det_r.theta_deg)

                lines.insert(
                    2,
                    f"L=({ul:.1f},{vl:.1f}) R=({ur:.1f},{vr:.1f}) disp={disparity:+.1f}px theta={theta_avg:+.1f}deg",
                )

                if now - last_print > PRINT_EVERY_SEC:
                    print(
                        f"TAG_PIXELS "
                        f"ul={ul:.2f} vl={vl:.2f} "
                        f"ur={ur:.2f} vr={vr:.2f} "
                        f"disp={disparity:+.2f} "
                        f"theta={theta_avg:+.2f}"
                    )
                    last_print = now

            preview = make_preview(left_draw, right_draw, lines)
            cv2.imshow("Stereo AprilTag Viewer", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        stereo.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
