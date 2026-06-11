from __future__ import annotations

"""Burst capture, YOLO burst inference, and centroid-based detection tracking."""

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from test_calibration_bundle_live_stereo_z_pickplace import read_stereo_tags_once
from vision.yolo_segmenter import YOLODetection, YOLOSegmenter


BURST_COUNT: int = 8
MIN_BURST_HITS: int = 3
BURST_FRAME_DELAY_S: float = 0.05
BURST_CLUSTER_MAX_CENTROID_PX: float = 75.0
BURST_REQUIRE_SAME_CLASS: bool = True
TARGET_CLASS_NAMES: list[str] = []

# Stereo-camera pre-flush.  DirectShow (CAP_DSHOW) buffers frames continuously
# while no reader drains them.  During the miss-check burst and place descent
# the stereo camera is idle but still filling its internal queue, so the first
# frames read in the next survey are stale.  Discard this many frames (and wait
# STEREO_BURST_PREFRESH_DELAY_S between each) before starting the YOLO burst.
# Set to 0 to restore the old (no-flush) behaviour.
STEREO_BURST_PREFRESH_COUNT: int = 4
STEREO_BURST_PREFRESH_DELAY_S: float = 0.01

# When True, burst_tracking prints elapsed time for the prefresh drain and
# each frame capture so per-phase timing is visible.
PROFILE_SURVEY_TIMING: bool = True


@dataclass
class BurstFrame:
    frame_i: int
    left_rect: np.ndarray
    right_rect: np.ndarray
    stereo_tags: dict[int, Any]
    detections: list[YOLODetection] = field(default_factory=list)


@dataclass
class DetectionObservation:
    frame_i: int
    detection: YOLODetection


@dataclass
class DetectionTrack:
    track_id: int
    class_name: str
    observations: list[DetectionObservation] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        return len({obs.frame_i for obs in self.observations})

    @property
    def mean_centroid_px(self) -> np.ndarray:
        pts = [np.asarray(obs.detection.centroid_px, dtype=np.float64).reshape(2) for obs in self.observations]
        if not pts:
            return np.array([np.nan, np.nan], dtype=np.float64)
        return np.mean(np.vstack(pts), axis=0)

    def best_observation(self) -> DetectionObservation:
        if not self.observations:
            raise RuntimeError("DetectionTrack has no observations.")
        return max(
            self.observations,
            key=lambda obs: float(obs.detection.confidence) * math.log1p(float(obs.detection.mask_area)),
        )


def _hr(title: str = "", char: str = "=", width: int = 96) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"{char * 3} {title} {char * pad}"[:width])
    else:
        print(char * width)


def capture_burst_frames(
    stereo: Any,
    detector: Any,
    stereo_calib: dict,
    rectifier: Any,
) -> list[BurstFrame]:
    # Drain the stereo camera's DirectShow buffer before starting the burst so
    # that frames from during the previous miss-check / place descent aren't
    # mixed into the new survey.
    _prefresh = max(0, int(STEREO_BURST_PREFRESH_COUNT))
    if _prefresh > 0:
        _t_pre = time.perf_counter()
        print(f"[BURST] flushing {_prefresh} stale stereo frame(s) before burst")
        for _fi in range(_prefresh):
            stereo.read_pair()
            if STEREO_BURST_PREFRESH_DELAY_S > 0.0:
                time.sleep(float(STEREO_BURST_PREFRESH_DELAY_S))
        if PROFILE_SURVEY_TIMING:
            print(f"[BURST TIMING] stereo_prefresh={time.perf_counter() - _t_pre:.3f}s  ({_prefresh} frames)")

    frames: list[BurstFrame] = []
    for frame_i in range(int(BURST_COUNT)):
        stereo_tags, left_raw, right_raw, _, _ = read_stereo_tags_once(
            stereo, detector, stereo_calib
        )
        if left_raw is None or right_raw is None:
            print(f"[BURST] frame {frame_i + 1}/{BURST_COUNT}: stereo read failed")
            continue

        try:
            left_rect, right_rect = rectifier.rectify(left_raw, right_raw)
        except Exception as exc:
            print(f"[BURST] frame {frame_i + 1}/{BURST_COUNT}: rectification failed: {exc}")
            continue

        frames.append(
            BurstFrame(
                frame_i=frame_i,
                left_rect=left_rect,
                right_rect=right_rect,
                stereo_tags=stereo_tags,
            )
        )
        print(f"[BURST] captured frame {frame_i + 1}/{BURST_COUNT}")
        if frame_i < BURST_COUNT - 1 and BURST_FRAME_DELAY_S > 0:
            time.sleep(float(BURST_FRAME_DELAY_S))
    return frames


def run_yolo_on_burst(yolo: YOLOSegmenter, frames: list[BurstFrame]) -> None:
    if not frames:
        return

    left_images = [f.left_rect for f in frames]
    detections_by_frame = yolo.segment_batch(left_images)

    for frame, detections in zip(frames, detections_by_frame):
        if TARGET_CLASS_NAMES:
            detections = [d for d in detections if d.class_name in TARGET_CLASS_NAMES]
        frame.detections = detections
        print(
            f"[YOLO BURST] frame {frame.frame_i + 1:02d}: "
            f"{len(detections)} detection(s): "
            + ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in detections)
        )


def _find_matching_track(
    tracks: list[DetectionTrack],
    det: YOLODetection,
) -> DetectionTrack | None:
    det_c = np.asarray(det.centroid_px, dtype=np.float64).reshape(2)
    best_track = None
    best_dist = float("inf")

    for track in tracks:
        if BURST_REQUIRE_SAME_CLASS and track.class_name != det.class_name:
            continue
        dist = float(np.linalg.norm(det_c - track.mean_centroid_px))
        if dist < best_dist:
            best_dist = dist
            best_track = track

    if best_track is not None and best_dist <= BURST_CLUSTER_MAX_CENTROID_PX:
        return best_track
    return None


def cluster_burst_detections(frames: list[BurstFrame]) -> tuple[list[DetectionTrack], list[DetectionTrack]]:
    tracks: list[DetectionTrack] = []

    for frame in frames:
        for det in frame.detections:
            track = _find_matching_track(tracks, det)
            if track is None:
                track = DetectionTrack(
                    track_id=len(tracks) + 1,
                    class_name=str(det.class_name),
                )
                tracks.append(track)
            track.observations.append(DetectionObservation(frame_i=frame.frame_i, detection=det))

    kept = [t for t in tracks if t.hit_count >= MIN_BURST_HITS]
    kept.sort(
        key=lambda t: (
            t.hit_count,
            max(float(obs.detection.confidence) for obs in t.observations),
        ),
        reverse=True,
    )

    _hr("BURST TRACKS", "-")
    print(f"[BURST] total tracks={len(tracks)} kept={len(kept)} min_hits={MIN_BURST_HITS}/{BURST_COUNT}")
    for t in tracks:
        status = "KEEP" if t in kept else "drop"
        c = t.mean_centroid_px
        best = t.best_observation().detection if t.observations else None
        conf = float(best.confidence) if best is not None else float("nan")
        print(
            f"  [{t.track_id:02d}] {status:4s} {t.class_name:14s} "
            f"hits={t.hit_count:2d}/{BURST_COUNT} "
            f"mean_px=({c[0]:6.1f},{c[1]:6.1f}) "
            f"best_conf={conf:.2f}"
        )
    return tracks, kept
