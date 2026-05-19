from __future__ import annotations

"""
camera_oop_smoke_test.py

No-robot smoke test for the OOP camera wrappers.

Opens:
  - SimpleOverheadCamera from overhead_camera.py
  - SimpleStereoCamera from stereo_apriltag_viewer.py

Then reads live frames and runs AprilTag detections on overhead, left stereo,
and right stereo views.

Controls when display is enabled:
  q / ESC  quit
"""

import argparse
import time

import cv2

from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
    make_preview,
)


def draw_all_detections(frame, detections, label_prefix: str):
    drawn = frame
    if detections:
        for det in detections.values():
            drawn = draw_detection(drawn, det, f"{label_prefix} ID {det.marker_id}")
    else:
        drawn = draw_detection(drawn, None, f"{label_prefix} no tags")
    return drawn


def format_ids(detections: dict[int, object]) -> str:
    if not detections:
        return "-"
    return ",".join(str(tag_id) for tag_id in sorted(detections))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0, help="Max runtime for the smoke test.")
    parser.add_argument("--overhead-index", type=int, default=None, help="Override camera_config.OVERHEAD_INDEX.")
    parser.add_argument("--stereo-index", type=int, default=None, help="Override camera_config.STEREO_INDEX.")
    args = parser.parse_args()

    detector = build_detector()
    overhead = None
    stereo = None
    overhead_seen = False
    stereo_left_seen = False
    stereo_right_seen = False
    frames = 0
    start = time.time()
    last_print = 0.0

    try:
        overhead = SimpleOverheadCamera() if args.overhead_index is None else SimpleOverheadCamera(args.overhead_index)
        stereo = SimpleStereoCamera() if args.stereo_index is None else SimpleStereoCamera(index=args.stereo_index)

        while time.time() - start < args.seconds:
            ok_o, overhead_frame, overhead_dets = overhead.read_detections(detector)
            ok_s, stereo_frame, left, right = stereo.read_pair()

            if not ok_o:
                print("[Smoke] Overhead read failed.")
                break
            if not ok_s:
                print("[Smoke] Stereo read failed.")
                break

            left_dets = detect_tags(detector, left)
            right_dets = detect_tags(detector, right)

            overhead_seen = overhead_seen or bool(overhead_dets)
            stereo_left_seen = stereo_left_seen or bool(left_dets)
            stereo_right_seen = stereo_right_seen or bool(right_dets)
            frames += 1

            now = time.time()
            if now - last_print >= 0.5:
                print(
                    "[Smoke] "
                    f"overhead IDs={format_ids(overhead_dets)} | "
                    f"left IDs={format_ids(left_dets)} | "
                    f"right IDs={format_ids(right_dets)}"
                )
                last_print = now

            overhead_drawn = overhead.draw_detections(overhead_frame, overhead_dets)
            left_drawn = draw_all_detections(left, left_dets, "LEFT")
            right_drawn = draw_all_detections(right, right_dets, "RIGHT")
            stereo_preview = make_preview(
                left_drawn,
                right_drawn,
                [
                    f"Overhead IDs: {format_ids(overhead_dets)}",
                    f"Stereo L/R IDs: {format_ids(left_dets)} / {format_ids(right_dets)}",
                ],
            )

            cv2.imshow("OOP Smoke - Overhead", overhead_drawn)
            cv2.imshow("OOP Smoke - Stereo", stereo_preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        print(f"[Smoke] Frames read: {frames}")
        print(f"[Smoke] Overhead tag seen: {overhead_seen}")
        print(f"[Smoke] Stereo left tag seen: {stereo_left_seen}")
        print(f"[Smoke] Stereo right tag seen: {stereo_right_seen}")

        return 0

    finally:
        if overhead is not None:
            overhead.release()
        if stereo is not None:
            stereo.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
