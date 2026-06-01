from __future__ import annotations

"""Candidate debug containers and point-cloud candidate construction."""

import traceback
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from vision.burst_tracking import BurstFrame, DetectionTrack
from vision.object_geometry import ObjectCandidate, build_object_candidate
from vision.pick_z_resolver import resolve_robust_object_z
from vision.pointcloud import cam_points_to_robot_xyz
from vision.pointcloud import masked_disparity_to_pointcloud
from vision.yolo_segmenter import YOLODetection


BURST_COUNT: int = 10
MIN_VALID_OBJECT_POINTS: int = 300
OVERHEAD_XY_BLEND_WEIGHT: float = 0.45


@dataclass
class CandidateDebug:
    candidate: ObjectCandidate
    track: DetectionTrack
    best_frame_i: int
    best_detection: YOLODetection
    disparity: np.ndarray
    disparity_overlay: np.ndarray
    left_overlay: np.ndarray
    point_count: int
    point_uv_px: np.ndarray
    points_cam: np.ndarray

    # Geometry copies used for runtime blend tuning.
    stereo_xy_mm: np.ndarray
    overhead_xy_mm: np.ndarray | None
    blend_weight_overhead: float
    xy_disagreement_mm: float | None

    # Phi resolution metadata, populated by resolve_pick_phi(). All optional.
    phi_stereo_deg: float | None = None
    phi_overhead_deg: float | None = None
    phi_stereo_confidence: float | None = None
    phi_overhead_confidence: float | None = None
    phi_disagreement_deg: float | None = None
    phi_blend_source: str | None = None


@dataclass
class SurveyState:
    burst_frames: list[BurstFrame]
    tracks: list[DetectionTrack]
    kept_tracks: list[DetectionTrack]
    candidates: list[CandidateDebug]
    overhead_frame: np.ndarray | None
    overhead_detections: list[YOLODetection]
    selected_index: int = 0


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    disp = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(disp)
    if not np.any(finite):
        return np.zeros((*disp.shape[:2], 3), dtype=np.uint8)

    vals = disp[finite]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0

    norm = np.clip((disp - lo) / (hi - lo), 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO)


def overlay_detection_on_image(
    image_bgr: np.ndarray,
    det: YOLODetection,
    *,
    label: str,
    alpha: float = 0.40,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    mask = np.asarray(det.mask).astype(bool)
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    color = np.array([0, 255, 255], dtype=np.uint8)
    overlay = out.copy()
    overlay[mask] = color
    out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)

    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

    c = tuple(np.round(np.asarray(det.centroid_px, dtype=np.float64)).astype(int))
    cv2.circle(out, c, 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw semiminor image axis for visual sanity.
    phi = float(det.minor_axis_angle_deg)
    rad = np.deg2rad(phi)
    length = max(20.0, 0.5 * float(det.minor_axis_length_px))
    dx = int(round(np.cos(rad) * length))
    dy = int(round(np.sin(rad) * length))
    cv2.line(out, (c[0] - dx, c[1] - dy), (c[0] + dx, c[1] + dy), (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(
        out,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def build_candidate_from_track(
    *,
    track: DetectionTrack,
    frames: list[BurstFrame],
    raft: Any,
    stereo_calib: dict,
    robot: Any,
    bundle: dict,
    index: int,
) -> CandidateDebug | None:
    obs = track.best_observation()
    frame = next((f for f in frames if f.frame_i == obs.frame_i), None)
    if frame is None:
        print(f"[TRACK {track.track_id}] best frame missing")
        return None

    print(
        f"[RAFT] track {track.track_id} {track.class_name}: "
        f"best_frame={obs.frame_i + 1}, hits={track.hit_count}/{BURST_COUNT}"
    )

    try:
        disparity = raft.predict_disparity(frame.left_rect, frame.right_rect, color="BGR")
    except Exception as exc:
        print(f"[RAFT] track {track.track_id}: failed: {exc}")
        return None

    return build_candidate_from_detection(
        track=track,
        frame=frame,
        det=obs.detection,
        disparity=disparity,
        stereo_calib=stereo_calib,
        robot=robot,
        bundle=bundle,
        index=index,
    )


def build_candidate_from_detection(
    *,
    track: DetectionTrack,
    frame: BurstFrame,
    det: YOLODetection,
    disparity: np.ndarray,
    stereo_calib: dict,
    robot: Any,
    bundle: dict,
    index: int,
) -> CandidateDebug | None:
    disp_color = colorize_disparity(disparity)
    disp_overlay = overlay_detection_on_image(
        disp_color,
        det,
        label=f"#{index} {det.class_name} hits={track.hit_count}/{BURST_COUNT}",
        alpha=0.45,
    )
    left_overlay = overlay_detection_on_image(
        frame.left_rect,
        det,
        label=f"#{index} {det.class_name} conf={det.confidence:.2f}",
        alpha=0.35,
    )

    try:
        points_cam, point_uv_px = masked_disparity_to_pointcloud(
            det.mask,
            disparity,
            stereo_calib,
        )
    except Exception as exc:
        print(f"[POINTCLOUD] track {track.track_id}: failed: {exc}")
        return None

    if len(points_cam) < MIN_VALID_OBJECT_POINTS:
        print(
            f"[POINTCLOUD] track {track.track_id}: "
            f"{len(points_cam)} valid points < {MIN_VALID_OBJECT_POINTS}; drop"
        )
        return None

    try:
        cand = build_object_candidate(
            index=index,
            yolo_det=det,
            points_cam=points_cam,
            point_uv_px=point_uv_px,
            robot=robot,
            bundle=bundle,
            stereo_tags=frame.stereo_tags,
            frame_i=int(frame.frame_i),
        )
    except Exception as exc:
        print(f"[CANDIDATE] track {track.track_id}: build failed: {exc}")
        traceback.print_exc()
        return None

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
    except Exception as exc:
        print(f"[Z] robot-frame point cloud transform failed; using fallback Z: {exc}")
        points_robot = None
    z_result = resolve_robust_object_z(
        points_robot=points_robot,
        points_cam=points_cam,
        candidate=cand,
    )
    cand.object_robot_xyz_raw[2] = z_result.robust_top_z_mm
    cand.object_robot_xyz_corrected[2] = z_result.robust_top_z_mm
    cand.hover_robot_z = z_result.hover_z_mm
    cand.grasp_robot_z = z_result.grasp_z_mm
    cand.stereo_z_bias_mm = 0.0
    cand.z_debug = z_result

    return CandidateDebug(
        candidate=cand,
        track=track,
        best_frame_i=int(frame.frame_i),
        best_detection=det,
        disparity=disparity,
        disparity_overlay=disp_overlay,
        left_overlay=left_overlay,
        point_count=int(len(points_cam)),
        point_uv_px=point_uv_px,
        points_cam=points_cam,
        stereo_xy_mm=np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2),
        overhead_xy_mm=None,
        blend_weight_overhead=float(OVERHEAD_XY_BLEND_WEIGHT),
        xy_disagreement_mm=None,
    )
