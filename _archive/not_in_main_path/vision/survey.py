from __future__ import annotations

"""vision/survey.py

Workspace survey: burst capture, YOLO segmentation, RAFT disparity, point-cloud
candidate extraction.

Extracted from run_pickplace_fast.py.  Module-level constants match the knobs
in run_pickplace_fast.py so calling code needs no changes.
"""

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from vision.yolo_segmenter import YOLODetection, YOLOSegmenter
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import masked_disparity_to_pointcloud
from vision.object_geometry import (
    ObjectCandidate,
    build_object_candidate,
    candidate_score,
    fuse_survey_candidates,
    pre_pointcloud_detection_score,
)

if TYPE_CHECKING:
    from hardware.robot import Robot

from test_calibration_bundle_live_stereo_z_pickplace import read_stereo_tags_once


# ============================================================
# KNOBS (defaults match run_pickplace_fast.py)
# ============================================================

SURVEY_BURST_COUNT: int = 5
SURVEY_FRAME_DELAY_S: float = 0.05
SURVEY_RAFT_MODE: str = "best_frame_only"   # "best_frame_only" | "all_frames" | "manual_current_frame"
SURVEY_USE_YOLO_BATCH: bool = True
MIN_VALID_OBJECT_POINTS: int = 300


# ============================================================
# Frame capture
# ============================================================

def capture_survey_frames(
    stereo: Any,
    detector: Any,
    stereo_calib: dict[str, np.ndarray],
    rectifier: StereoRectifier,
) -> list[dict[str, Any]]:
    count = 1 if SURVEY_RAFT_MODE == "manual_current_frame" else int(SURVEY_BURST_COUNT)
    frames: list[dict[str, Any]] = []
    for frame_i in range(count):
        stereo_tags, left_raw, right_raw, _, _ = read_stereo_tags_once(
            stereo, detector, stereo_calib
        )
        if left_raw is None or right_raw is None:
            print(f"[SURVEY] frame {frame_i + 1}: stereo read failed")
            continue
        left_rect, right_rect = rectifier.rectify(left_raw, right_raw)
        frames.append(
            {
                "frame_i": frame_i,
                "left_rect": left_rect,
                "right_rect": right_rect,
                "stereo_tags": stereo_tags,
            }
        )
        if count > 1:
            time.sleep(SURVEY_FRAME_DELAY_S)
    return frames


# ============================================================
# YOLO
# ============================================================

def run_yolo_for_survey(
    yolo: YOLOSegmenter, left_frames: list[np.ndarray]
) -> list[list[YOLODetection]]:
    if SURVEY_USE_YOLO_BATCH:
        return yolo.segment_batch(left_frames)
    return [yolo.segment(frame) for frame in left_frames]


def select_best_yolo_detection(
    frames: list[dict[str, Any]],
    detections_by_frame: list[list[YOLODetection]],
) -> tuple[int, YOLODetection] | None:
    best: tuple[int, YOLODetection] | None = None
    best_score = -float("inf")
    for frame_idx, detections in enumerate(detections_by_frame):
        for det in detections:
            score = pre_pointcloud_detection_score(det)
            if score > best_score:
                best_score = score
                best = (frame_idx, det)
    return best


# ============================================================
# Timing print
# ============================================================

def print_survey_timing(
    *,
    capture_time_s: float,
    yolo_time_s: float,
    frame_selection_time_s: float,
    raft_time_s: float,
    pointcloud_time_s: float,
    total_start_s: float,
    selected_device: Any,
    raft_cached: bool,
) -> None:
    total_s = time.perf_counter() - total_start_s
    print("[SURVEY TIMING]")
    print(f"  capture_time_s         = {capture_time_s:.3f}")
    print(f"  yolo_time_s            = {yolo_time_s:.3f}")
    print(f"  frame_selection_time_s = {frame_selection_time_s:.3f}")
    print(f"  raft_time_s            = {raft_time_s:.3f}")
    print(f"  pointcloud_time_s      = {pointcloud_time_s:.3f}")
    print(f"  total_s                = {total_s:.3f}")
    print(f"  selected device        = {selected_device}")
    print(f"  raft cached/reused     = {raft_cached}")


# ============================================================
# Main survey
# ============================================================

def run_survey_workspace(
    stereo: Any,
    detector: Any,
    stereo_calib: dict[str, np.ndarray],
    rectifier: StereoRectifier,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    robot: "Robot",
    bundle: Any,
) -> list[ObjectCandidate]:
    print("\n[SURVEY] Starting YOLO + RAFT burst survey")
    print(
        f"[SURVEY] frames={SURVEY_BURST_COUNT}, raft_mode={SURVEY_RAFT_MODE}, "
        f"yolo_batch={SURVEY_USE_YOLO_BATCH}"
    )

    t_total = time.perf_counter()
    capture_time_s = yolo_time_s = selection_time_s = raft_time_s = pointcloud_time_s = 0.0
    raft_cached = False
    raw_candidates: list[ObjectCandidate] = []

    def _timing() -> None:
        print_survey_timing(
            capture_time_s=capture_time_s,
            yolo_time_s=yolo_time_s,
            frame_selection_time_s=selection_time_s,
            raft_time_s=raft_time_s,
            pointcloud_time_s=pointcloud_time_s,
            total_start_s=t_total,
            selected_device=raft.device,
            raft_cached=raft_cached,
        )

    try:
        t0 = time.perf_counter()
        frames = capture_survey_frames(stereo, detector, stereo_calib, rectifier)
        capture_time_s = time.perf_counter() - t0

        if not frames:
            print("[SURVEY] No stereo frames captured.")
            _timing()
            return []

        t0 = time.perf_counter()
        left_frames = [f["left_rect"] for f in frames]
        detections_by_frame = run_yolo_for_survey(yolo, left_frames)
        yolo_time_s = time.perf_counter() - t0

        counts = [len(dets) for dets in detections_by_frame]
        print(f"[SURVEY] YOLO candidates per frame: {counts}")

        if SURVEY_RAFT_MODE not in ("best_frame_only", "all_frames", "manual_current_frame"):
            print(f"[SURVEY] Unknown SURVEY_RAFT_MODE={SURVEY_RAFT_MODE!r}; refusing.")
            _timing()
            return []

        if SURVEY_RAFT_MODE in ("best_frame_only", "manual_current_frame"):
            t0 = time.perf_counter()
            selected = select_best_yolo_detection(frames, detections_by_frame)
            selection_time_s = time.perf_counter() - t0
            if selected is None:
                print("[SURVEY] No YOLO detections; no RAFT call needed.")
                _timing()
                return []

            frame_idx, det = selected
            frame = frames[frame_idx]
            left_rect = frame["left_rect"]
            right_rect = frame["right_rect"]
            frame_shape = left_rect.shape[:2]

            t0 = time.perf_counter()
            if raft.can_reuse_cache(frame_shape, det.centroid_px):
                print("[RAFT] Reusing cached disparity")
                disparity = raft.last_disparity
                raft_cached = True
            else:
                disparity = raft.predict_disparity(left_rect, right_rect, color="BGR")
                raft.remember_cache(disparity, frame_shape, det.centroid_px)
            raft_time_s = time.perf_counter() - t0

            if disparity is None:
                print("[SURVEY] REFUSED: disparity cache unexpectedly empty.")
                _timing()
                return []

            t0 = time.perf_counter()
            points_cam, uv = masked_disparity_to_pointcloud(det.mask, disparity, stereo_calib)
            if len(points_cam) < MIN_VALID_OBJECT_POINTS:
                print(
                    f"[SURVEY] {det.class_name}: only {len(points_cam)} valid points "
                    f"(<{MIN_VALID_OBJECT_POINTS})."
                )
                pointcloud_time_s = time.perf_counter() - t0
                _timing()
                return []

            cand = build_object_candidate(
                index=1,
                yolo_det=det,
                points_cam=points_cam,
                point_uv_px=uv,
                robot=robot,
                bundle=bundle,
                stereo_tags=frame["stereo_tags"],
                frame_i=int(frame["frame_i"]),
            )
            raw_candidates = [cand]
            pointcloud_time_s = time.perf_counter() - t0

        else:
            # all_frames mode
            next_index = 1
            t0_pc = time.perf_counter()
            for frame_idx, frame in enumerate(frames):
                detections = detections_by_frame[frame_idx]
                if not detections:
                    continue
                left_rect = frame["left_rect"]
                right_rect = frame["right_rect"]
                frame_shape = left_rect.shape[:2]

                t0 = time.perf_counter()
                disparity = raft.predict_disparity(left_rect, right_rect, color="BGR")
                raft_time_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                for det in detections:
                    points_cam, uv = masked_disparity_to_pointcloud(det.mask, disparity, stereo_calib)
                    if len(points_cam) < MIN_VALID_OBJECT_POINTS:
                        continue
                    cand = build_object_candidate(
                        index=next_index,
                        yolo_det=det,
                        points_cam=points_cam,
                        point_uv_px=uv,
                        robot=robot,
                        bundle=bundle,
                        stereo_tags=frame["stereo_tags"],
                        frame_i=int(frame["frame_i"]),
                    )
                    raw_candidates.append(cand)
                    next_index += 1
                pointcloud_time_s += time.perf_counter() - t0

    except Exception as exc:
        print(f"[SURVEY] Exception: {exc}")
        _timing()
        raise

    fused = fuse_survey_candidates(raw_candidates)
    _timing()
    print(f"[SURVEY] {len(fused)} fused candidate(s) after burst survey.")
    for c in fused:
        xy = c.target_xy
        pc_h = c.pointcloud_height_cm if c.pointcloud_height_cm is not None else 0.0
        print(
            f"  [{c.index}] {c.yolo.class_name:<14} "
            f"conf={c.yolo.confidence:.2f} pts={c.valid_point_count} "
            f"robot_xy=({xy[0]:.1f},{xy[1]:.1f}) "
            f"pc_h={pc_h:.1f}cm "
            f"bias={c.stereo_z_bias_mm:+.1f}mm"
        )
    return fused
