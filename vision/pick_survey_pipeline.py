from __future__ import annotations

"""Survey orchestration for real-object pick validation."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.pick_validation_display import _hr, print_survey_candidate_summary
from vision.burst_tracking import capture_burst_frames, cluster_burst_detections, run_yolo_on_burst
from vision.pick_candidate_builder import CandidateDebug, SurveyState, build_candidate_from_detection, colorize_disparity
from vision.pick_phi_resolver import resolve_pick_phi
from vision.pick_xy_resolver import apply_xy_blend, match_overhead_xy_to_candidate
from vision.raft_runner import RAFTStereoRunner
from vision.yolo_segmenter import YOLODetection, YOLOSegmenter


YOLO_WEIGHTS_PATH = Path("yolo_weights/full_data.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("yolo_weights/validate_V2.pt")
RAFT_ROOT = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")

YOLO_IMGSZ: int = 640
YOLO_CONF: float = 0.35
YOLO_IOU: float = 0.50
YOLO_RETINA_MASKS: bool = True
TARGET_CLASS_NAMES: list[str] = []

RAFT_VALID_ITERS: int = 16
RAFT_DOWNSCALE: float = 1.0
RAFT_MIXED_PRECISION: bool = True
MIN_MASK_AREA_PX: int = 500
RAFT_ONCE_PER_SURVEY: bool = True

# OpenCV/DirectShow webcams can return queued frames if the stream has not been
# drained while stereo burst + RAFT work is running. Discard a few frames before
# using the overhead image for matching/display.
OVERHEAD_FRESH_READ_DISCARD_FRAMES: int = 6
OVERHEAD_FRESH_READ_DELAY_S: float = 0.02


def load_vision(device_info: Any) -> tuple[YOLOSegmenter, RAFTStereoRunner]:
    weights = YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.is_file() else YOLO_FALLBACK_WEIGHTS_PATH
    yolo = YOLOSegmenter(
        weights_path=str(weights),
        device_info=device_info,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        retina_masks=YOLO_RETINA_MASKS,
        min_mask_area_px=MIN_MASK_AREA_PX,
        target_class_names=TARGET_CLASS_NAMES if TARGET_CLASS_NAMES else None,
    )
    yolo.warmup()
    print(f"[VISION] YOLO loaded: {weights} conf={YOLO_CONF}")

    raft = RAFTStereoRunner(
        raft_root=str(RAFT_ROOT),
        checkpoint_path=str(RAFT_CHECKPOINT_PATH),
        device_info=device_info,
        valid_iters=RAFT_VALID_ITERS,
        downscale=RAFT_DOWNSCALE,
        mixed_precision=RAFT_MIXED_PRECISION,
    )
    raft.warmup()
    print(f"[VISION] RAFT loaded: {RAFT_CHECKPOINT_PATH} iters={RAFT_VALID_ITERS}")
    return yolo, raft


def _match_overhead_to_candidates(
    overhead_dets: list[YOLODetection],
    candidate_debugs: list[CandidateDebug],
    bundle: dict,
) -> None:
    if not overhead_dets:
        print("[MATCH] no overhead detections; candidates will use stereo-only XY")
        for dbg in candidate_debugs:
            resolve_pick_phi(dbg, None, bundle)
            apply_xy_blend(dbg, bundle)
        return

    _hr("OVERHEAD MATCH", "-")
    for dbg in candidate_debugs:
        matched_overhead_det = match_overhead_xy_to_candidate(overhead_dets, dbg, bundle)
        resolve_pick_phi(dbg, matched_overhead_det, bundle)
        apply_xy_blend(dbg, bundle)


def _read_fresh_overhead(overhead: Any) -> tuple[bool, Any]:
    last_ok = False
    last_frame = None
    discard_count = max(0, int(OVERHEAD_FRESH_READ_DISCARD_FRAMES))
    total_reads = discard_count + 1
    if discard_count > 0:
        print(f"[OVERHEAD] flushing {discard_count} queued frame(s) before survey match")

    for i in range(total_reads):
        ok, frame = overhead.read()
        if ok and frame is not None:
            last_ok = True
            last_frame = frame
        if i < total_reads - 1 and OVERHEAD_FRESH_READ_DELAY_S > 0.0:
            time.sleep(float(OVERHEAD_FRESH_READ_DELAY_S))

    if last_ok:
        print(f"[OVERHEAD] fresh frame read after {total_reads} read(s)")
    return last_ok, last_frame


def _select_shared_raft_frame(frames: list[Any], kept_tracks: list[Any]) -> Any | None:
    if not frames or not kept_tracks:
        return None

    frame_by_i = {int(f.frame_i): f for f in frames}
    best_key: tuple[int, float, int] | None = None
    best_frame_i: int | None = None
    for frame in frames:
        frame_i = int(frame.frame_i)
        hits = 0
        conf_sum = 0.0
        for track in kept_tracks:
            obs = next((o for o in track.observations if int(o.frame_i) == frame_i), None)
            if obs is None:
                continue
            hits += 1
            conf_sum += float(getattr(obs.detection, "confidence", 0.0))
        key = (hits, conf_sum, -frame_i)
        if best_key is None or key > best_key:
            best_key = key
            best_frame_i = frame_i
    return None if best_frame_i is None else frame_by_i.get(best_frame_i)


def _build_candidates_with_one_disparity(
    *,
    frames: list[Any],
    kept_tracks: list[Any],
    raft: RAFTStereoRunner,
    stereo_calib: dict,
    robot: Any,
    bundle: dict,
) -> list[CandidateDebug]:
    shared_frame = _select_shared_raft_frame(frames, kept_tracks)
    if shared_frame is None:
        return []

    print(
        f"[RAFT] running once for survey on frame {int(shared_frame.frame_i) + 1}; "
        f"tracks={len(kept_tracks)}"
    )
    try:
        disparity = raft.predict_disparity(shared_frame.left_rect, shared_frame.right_rect, color="BGR")
    except Exception as exc:
        print(f"[RAFT] shared survey disparity failed: {exc}")
        return []
    _ = colorize_disparity(disparity)

    candidates: list[CandidateDebug] = []
    for idx, track in enumerate(kept_tracks, start=1):
        obs = next((o for o in track.observations if int(o.frame_i) == int(shared_frame.frame_i)), None)
        if obs is None:
            print(
                f"[RAFT] track {track.track_id} {track.class_name}: "
                f"not visible in shared frame {int(shared_frame.frame_i) + 1}; drop"
            )
            continue
        print(
            f"[RAFT] track {track.track_id} {track.class_name}: "
            f"shared_frame={int(shared_frame.frame_i) + 1}, hits={track.hit_count}/{len(frames)}"
        )
        dbg = build_candidate_from_detection(
            track=track,
            frame=shared_frame,
            det=obs.detection,
            disparity=disparity,
            stereo_calib=stereo_calib,
            robot=robot,
            bundle=bundle,
            index=idx,
        )
        if dbg is not None:
            candidates.append(dbg)
    return candidates


def run_survey(
    overhead: Any,
    stereo: Any,
    detector: Any,
    stereo_calib: dict,
    rectifier: Any,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    robot: Any,
    bundle: dict,
    seed_overhead_frame: Any | None = None,
) -> SurveyState:
    _hr("SURVEY REAL OBJECTS", "=")
    t0 = time.perf_counter()

    frames = capture_burst_frames(stereo, detector, stereo_calib, rectifier)
    if not frames:
        return SurveyState([], [], [], [], None, [], 0)

    run_yolo_on_burst(yolo, frames)
    tracks, kept = cluster_burst_detections(frames)

    candidates: list[CandidateDebug] = _build_candidates_with_one_disparity(
        frames=frames,
        kept_tracks=kept,
        raft=raft,
        stereo_calib=stereo_calib,
        robot=robot,
        bundle=bundle,
    )

    for idx, dbg in enumerate(candidates, start=1):
        dbg.candidate.index = idx

    if seed_overhead_frame is not None:
        # Reuse a frame captured by the miss-check burst immediately before this
        # survey — the platform scene hasn't changed, so capturing a fresh frame
        # is redundant.  Flush the stale backlog anyway to keep the queue clean.
        try:
            overhead.read()  # discard one stale frame so backlog doesn't grow
        except Exception:
            pass
        overhead_frame = seed_overhead_frame
        overhead_dets = yolo.segment(overhead_frame)
        if TARGET_CLASS_NAMES:
            overhead_dets = [d for d in overhead_dets if d.class_name in TARGET_CLASS_NAMES]
        print(f"[OVERHEAD] reusing miss-check frame; YOLO detections={len(overhead_dets)}")
        for i, d in enumerate(overhead_dets):
            print(
                f"  [OH {i}] {d.class_name:14s} conf={d.confidence:.2f} "
                f"centroid=({d.centroid_px[0]:.1f},{d.centroid_px[1]:.1f})"
            )
    else:
        ok_oh, overhead_frame = _read_fresh_overhead(overhead)
        if ok_oh and overhead_frame is not None:
            overhead_dets = yolo.segment(overhead_frame)
            if TARGET_CLASS_NAMES:
                overhead_dets = [d for d in overhead_dets if d.class_name in TARGET_CLASS_NAMES]
            print(f"[OVERHEAD] YOLO detections={len(overhead_dets)}")
            for i, d in enumerate(overhead_dets):
                print(
                    f"  [OH {i}] {d.class_name:14s} conf={d.confidence:.2f} "
                    f"centroid=({d.centroid_px[0]:.1f},{d.centroid_px[1]:.1f})"
                )
        else:
            overhead_frame = None
            overhead_dets = []
            print("[OVERHEAD] frame unavailable")

    _match_overhead_to_candidates(overhead_dets, candidates, bundle)

    print_survey_candidate_summary(candidates)
    print(f"[SURVEY] done in {time.perf_counter() - t0:.2f}s")

    return SurveyState(
        burst_frames=frames,
        tracks=tracks,
        kept_tracks=kept,
        candidates=candidates,
        overhead_frame=overhead_frame.copy() if overhead_frame is not None else None,
        overhead_detections=overhead_dets,
        selected_index=0 if candidates else 0,
    )
