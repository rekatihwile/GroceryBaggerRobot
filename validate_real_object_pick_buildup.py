from __future__ import annotations

"""
scripts/validate_real_object_pick_buildup.py

Real-object pick validation buildup.

This script deliberately stops BEFORE a real pick:
  - no BLB
  - no place
  - no claw close during test
  - no autonomous cycle

Controls
--------
  s   survey real objects with a 10-frame burst
  r   rotate through kept burst detections / candidates
  v   validate and print all geometry for the selected candidate
  t   test approach + slow descent to grasp height, but DO NOT close claw
      After reaching grasp height, enter XY-blend fine tune:
          a = more stereo XY
          d = more overhead H(z) XY
          s = re-survey now
          v = print current blend geometry
          q/esc = exit fine tune
  q   quit

Survey logic
------------
1. Capture BURST_COUNT stereo pairs.
2. Rectify each pair.
3. Run YOLO segmentation on all burst-left frames.
4. Cluster detections across burst frames by class + centroid.
5. Keep only tracks detected in at least MIN_BURST_HITS frames.
6. For each kept track, pick its best detection frame.
7. Run RAFT on that selected frame.
8. Overlay the kept mask on the RAFT disparity visualization.
9. Make point cloud from the detection mask + disparity.
10. Build ObjectCandidate with the same geometry path as run_buildup_pickplace.py.
11. Match overhead YOLO detection and fuse overhead H(z) XY with stereo XY.

Motion logic
------------
  travel_z = Z_MAX_MM
  hover_z  = Z_MAX_MM
  grasp_z  = z_stereo + GRIPPER_OFFSET_MM

X/Y command:
  xy_cmd = w * xy_overhead_Hz + (1 - w) * xy_stereo

where w starts at OVERHEAD_XY_BLEND_WEIGHT and can be adjusted after the slow
descent for debugging.
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

BUNDLE_PATH = Path("robot_calibration_bundle.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")

YOLO_WEIGHTS_PATH = Path("yolo_weights/Validate_Only_100_Training_Best.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("yolo_weights/validate_V2.pt")

RAFT_ROOT = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")

YOLO_IMGSZ: int = 640
YOLO_CONF: float = 0.35
YOLO_IOU: float = 0.50
YOLO_RETINA_MASKS: bool = True
TARGET_CLASS_NAMES: list[str] = []  # [] = all classes

USE_CUDA: bool = True
USE_HALF: bool = True
RAFT_VALID_ITERS: int = 16
RAFT_DOWNSCALE: float = 1.0
RAFT_MIXED_PRECISION: bool = True

MIN_MASK_AREA_PX: int = 500
MIN_DISPARITY_PX: float = 1.0
MIN_VALID_OBJECT_POINTS: int = 300

# Burst detection validation.
BURST_COUNT: int = 10
MIN_BURST_HITS: int = 3
BURST_FRAME_DELAY_S: float = 0.05
BURST_CLUSTER_MAX_CENTROID_PX: float = 75.0
BURST_REQUIRE_SAME_CLASS: bool = True

# Pick phi mode. The first two ray-cast from the mask centroid; the third uses
# the stereo point cloud projected into robot XY.
PICK_PHI_MODE: str = "centroid_shortest_ray_parallel"
VALID_PICK_PHI_MODES = {
    "centroid_longest_ray_perp",
    "centroid_shortest_ray_parallel",
    "pointcloud_shortest_path",
    "overhead_minor_axis",
    "mask_minor_axis_pointcloud",
    "triangulated_short_side",
    "current_fk",
}

# Z policy for validation-only picking.
Z_MAX_MM: float = 275.0
GRIPPER_OFFSET_MM: float = 125.0
HOVER_HEIGHT_MM: float = Z_MAX_MM
GRASP_OFFSET_MM: float = GRIPPER_OFFSET_MM

# XY fusion.
OVERHEAD_MATCH_MAX_DIST_MM: float = 140.0
OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True
OVERHEAD_XY_BLEND_WEIGHT: float = 0.45
OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI: bool = True
XY_DISAGREEMENT_WARN_MM: float = 50.0
FINE_TUNE_BLEND_STEP: float = 0.05

# Phi (gripper yaw) confidence-weighted blend.
# Stereo and overhead each produce a phi estimate from their YOLO mask's
# minor-axis angle. Each estimate gets a confidence in [0, 1] derived from
# (a) mask aspect ratio (a near-circle gives an undefined minor axis), and
# (b) per-camera geometric reliability (overhead obliquity / stereo height).
# We blend the two using their normalized confidences on the doubled-angle
# unit circle (mod-180 safe). If only one source is usable, we use it; if
# neither is usable, we keep stereo phi or fall back to current EE phi.
USE_CONFIDENCE_PHI_BLEND: bool = True
PHI_DISAGREEMENT_WARN_DEG: float = 25.0
PHI_MIN_CONFIDENCE: float = 0.20
PHI_ASPECT_DECAY: float = 0.8          # higher = needs more elongation before phi is trusted
PHI_STEREO_HEIGHT_DECAY_CM: float = 8.0  # taller objects -> stereo phi penalized more
PHI_FALLBACK_TO_CURRENT_EE_PHI: bool = False  # if both sources unusable, hold current EE phi

# Motion timing.
COARSE_MOVE_TIME_S: float = 1.10
XY_MOVE_TIME_S: float = 1.50
RAISE_MOVE_TIME_S: float = 1.00
DESCENT_STEP_MM: float = 5.0
DESCENT_STEP_TIME_S: float = 0.35
FINE_TUNE_XY_MOVE_TIME_S: float = 0.35

# Robot connection.
CONNECT_ROBOT: bool = True
ENABLE_MOTORS_ON_START: bool = True
INIT_DRIVERS_ON_START: bool = True

# Display.
COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 560
STEREO_DRAW_H_PX: int = 390
STATUS_H_PX: int = 140
WINDOW: str = "Real Object Pick Validation Buildup"

# Claw commands are not used during test, but manual open is convenient.
CLAW_OPEN_DEG: int = 65

# Safety-ish gates.
REQUIRE_OVERHEAD_XY_FOR_TEST: bool = False
REFUSE_TEST_IF_TOO_FEW_POINTS: bool = True


X_SURVEY = 100.0
Y_SURVEY = 100.0
Z_SURVEY = 250.0

# ============================================================
# IMPORTS / MODULE PATCHES
# ============================================================

import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Patch object geometry BEFORE importing ObjectCandidate/build_object_candidate.
import vision.object_geometry as _geom_mod

_geom_mod.USE_EE_FK_Z_BIAS_CORRECTION = False
_geom_mod.TARGET_XY_SOURCE = "overhead_homography"
_geom_mod.HOVER_HEIGHT_MM = HOVER_HEIGHT_MM
_geom_mod.GRASP_OFFSET_MM = GRASP_OFFSET_MM
_geom_mod.PICK_PHI_MODE = PICK_PHI_MODE

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import estimate_mask_centroid_ray_angle_deg, masked_disparity_to_pointcloud
from vision.object_geometry import ObjectCandidate, build_object_candidate, candidate_score

from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from hardware.robot import Robot

from test_calibration_bundle_live_stereo_z_pickplace import (
    load_bundle,
    load_stereo_calibration,
    read_stereo_tags_once,
    map_uv_z_to_robot_xy,
    nearest_support_distance,
    clamp_lookup_z_to_bundle,
    read_command_key,
    require_soft_limits,
    move_cartesian_nonnegative_z,
    jog_nonnegative_z,
    print_matrix_labeled,
)


# ============================================================
# DATA CLASSES
# ============================================================

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

    # Phi resolution metadata, populated by _resolve_pick_phi(). All optional.
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


# ============================================================
# GENERAL HELPERS
# ============================================================

def _hr(title: str = "", char: str = "=", width: int = 96) -> None:
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"{char * 3} {title} {char * pad}"[:width])
    else:
        print(char * width)


def _fmt_xy(xy: Any) -> str:
    if xy is None:
        return "(--, --)"
    a = np.asarray(xy, dtype=np.float64).reshape(-1)
    if a.size < 2:
        return "(--, --)"
    return f"({a[0]:7.1f}, {a[1]:7.1f})"


def _fmt_xyz(xyz: Any) -> str:
    if xyz is None:
        return "(--, --, --)"
    a = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if a.size < 3:
        return "(--, --, --)"
    return f"({a[0]:7.1f}, {a[1]:7.1f}, {a[2]:7.1f})"


def _candidate_phi_or_current(robot: Robot, candidate: ObjectCandidate) -> float:
    if candidate.pick_phi_deg is not None and np.isfinite(float(candidate.pick_phi_deg)):
        return float(candidate.pick_phi_deg)
    _, _, _, phi_fk = robot.fk()
    return float(phi_fk)


def _fmt_candidate_phi(candidate: ObjectCandidate) -> str:
    if candidate.pick_phi_deg is None:
        return f"current ({candidate.pick_phi_source})"
    return f"{float(candidate.pick_phi_deg):+6.1f} ({candidate.pick_phi_source})"


def _validate_z_command(z_mm: float, label: str) -> str | None:
    if not np.isfinite(z_mm):
        return f"{label} z={z_mm} is not finite"
    if z_mm < 0.0:
        return f"{label} z={z_mm:.1f} mm < 0"
    if z_mm > Z_MAX_MM + 1e-6:
        return f"{label} z={z_mm:.1f} mm > Z_MAX_MM={Z_MAX_MM:.1f}"
    return None


def _print_fk(robot: Robot, prefix: str = "[FK]") -> tuple[float, float, float, float]:
    x, y, z, phi = robot.fk()
    print(f"{prefix} x={x:7.1f} y={y:7.1f} z={z:7.1f} phi={phi:6.2f}")
    return x, y, z, phi


# ============================================================
# VISION LOADING
# ============================================================

def load_vision(device_info) -> tuple[YOLOSegmenter, RAFTStereoRunner]:
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


# ============================================================
# BURST SURVEY
# ============================================================

def capture_burst_frames(
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict,
    rectifier: StereoRectifier,
) -> list[BurstFrame]:
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
    raft: RAFTStereoRunner,
    stereo_calib: dict,
    robot: Robot,
    bundle: dict,
    index: int,
) -> CandidateDebug | None:
    obs = track.best_observation()
    frame = next((f for f in frames if f.frame_i == obs.frame_i), None)
    if frame is None:
        print(f"[TRACK {track.track_id}] best frame missing")
        return None

    det = obs.detection
    print(
        f"[RAFT] track {track.track_id} {track.class_name}: "
        f"best_frame={obs.frame_i + 1}, hits={track.hit_count}/{BURST_COUNT}"
    )

    try:
        disparity = raft.predict_disparity(frame.left_rect, frame.right_rect, color="BGR")
    except Exception as exc:
        print(f"[RAFT] track {track.track_id}: failed: {exc}")
        return None

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

    # Simpler validation Z policy.
    z_stereo = max(0.0, float(cand.object_robot_xyz_raw[2]))
    cand.object_robot_xyz_raw[2] = z_stereo
    cand.object_robot_xyz_corrected[2] = z_stereo
    cand.hover_robot_z = float(Z_MAX_MM)
    cand.grasp_robot_z = float(z_stereo + GRIPPER_OFFSET_MM)
    cand.stereo_z_bias_mm = 0.0

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


# ============================================================
# OVERHEAD MATCH + WEIGHTED XY
# ============================================================

def _overhead_bbox_corners_px(det: YOLODetection) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in det.bbox)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)


def _overhead_mask_rect_corners_px(det: YOLODetection) -> np.ndarray | None:
    mask = getattr(det, "mask", None)
    if mask is None:
        return None
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    pts = cv2.findNonZero(mask_u8)
    if pts is None or len(pts) < 4:
        return None
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float64)


def _project_overhead_centroid_to_robot_xy(
    centroid_px: np.ndarray,
    z_mm: float,
    bundle: dict,
) -> tuple[np.ndarray, float, bool, float, int]:
    lookup_z, clamped = clamp_lookup_z_to_bundle(max(0.0, float(z_mm)), bundle)
    xy_raw, _uv_undist, _lo, _hi, _alpha = map_uv_z_to_robot_xy(
        np.asarray(centroid_px, dtype=np.float64).reshape(2),
        lookup_z,
        bundle,
    )
    xy = np.asarray(xy_raw, dtype=np.float64).reshape(2)
    support_dist, support_idx = nearest_support_distance(xy, lookup_z, bundle)
    return xy, float(lookup_z), bool(clamped), float(support_dist), int(support_idx)


def _weighted_xy(stereo_xy: np.ndarray, overhead_xy: np.ndarray | None, w_overhead: float) -> tuple[np.ndarray, str]:
    st = np.asarray(stereo_xy, dtype=np.float64).reshape(2)
    if overhead_xy is None:
        return st.copy(), "stereo_only_no_overhead"

    w = float(np.clip(w_overhead, 0.0, 1.0))
    oh = np.asarray(overhead_xy, dtype=np.float64).reshape(2)
    xy = w * oh + (1.0 - w) * st
    return xy.astype(np.float64), f"weighted_overhead_{w:.2f}_stereo_{1.0 - w:.2f}"


def _apply_xy_blend(debug: CandidateDebug, bundle: dict, w_overhead: float | None = None) -> None:
    cand = debug.candidate
    if w_overhead is None:
        w_overhead = debug.blend_weight_overhead

    stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))
    debug.stereo_xy_mm = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2)

    overhead_xy = None
    if cand.overhead_centroid_px is not None:
        try:
            overhead_xy, lookup_z, lookup_clamped, support_dist, support_idx = _project_overhead_centroid_to_robot_xy(
                cand.overhead_centroid_px,
                stereo_z,
                bundle,
            )
            cand.lookup_z_used = float(lookup_z)
            cand.lookup_z_clamped = bool(lookup_clamped)
            cand.support_distance_mm = float(support_dist)
            cand.support_index = int(support_idx)
        except Exception as exc:
            print(f"[XY BLEND] overhead mapping failed for cand[{cand.index}]: {exc}")
            cand.overhead_centroid_px = None
            overhead_xy = None

    debug.overhead_xy_mm = None if overhead_xy is None else overhead_xy.copy()

    xy, source = _weighted_xy(debug.stereo_xy_mm, overhead_xy, float(w_overhead))
    debug.blend_weight_overhead = float(np.clip(w_overhead, 0.0, 1.0))

    if overhead_xy is not None:
        debug.xy_disagreement_mm = float(np.linalg.norm(overhead_xy - debug.stereo_xy_mm))
    else:
        debug.xy_disagreement_mm = None

    cand.target_xy = xy.copy()
    cand.target_xy_source_effective = source
    cand.object_robot_xyz_corrected = np.array([xy[0], xy[1], stereo_z], dtype=np.float64)
    cand.hover_robot_z = float(Z_MAX_MM)
    cand.grasp_robot_z = float(stereo_z + GRIPPER_OFFSET_MM)


# ============================================================
# PHI (gripper yaw) CONFIDENCE-WEIGHTED BLEND
# ============================================================

def wrapped_phi_blend_deg(
    phi_a_deg: float,
    w_a: float,
    phi_b_deg: float,
    w_b: float,
) -> float:
    """Weighted blend of two angles that are equivalent mod 180° (gripper yaw).
    Uses doubled-angle unit vectors so +85° and -85° correctly blend to ±90°
    (≈10° physical disagreement), not 0° (the worst possible answer)."""
    ang_a = np.deg2rad(2.0 * float(phi_a_deg))
    ang_b = np.deg2rad(2.0 * float(phi_b_deg))
    vx = float(w_a) * np.cos(ang_a) + float(w_b) * np.cos(ang_b)
    vy = float(w_a) * np.sin(ang_a) + float(w_b) * np.sin(ang_b)
    if vx == 0.0 and vy == 0.0:
        # Exactly antipodal with equal weights: pick the average naively.
        return float(0.5 * (float(phi_a_deg) + float(phi_b_deg)))
    return float(np.rad2deg(np.arctan2(vy, vx)) / 2.0)


def wrapped_phi_diff_deg(phi_a_deg: float, phi_b_deg: float) -> float:
    """Signed difference (phi_a - phi_b) wrapped to [-90, 90)."""
    return float(((float(phi_a_deg) - float(phi_b_deg) + 90.0) % 180.0) - 90.0)


def aspect_phi_confidence(det: YOLODetection) -> float:
    """Confidence that the YOLO mask's minor axis is well defined.
    Near-circular masks (aspect ≈ 1) give garbage minor axes; oblong masks
    give a sharply defined one. Returns 0 at aspect=1, ~0.92 at aspect=3."""
    minor = max(float(det.minor_axis_length_px), 1.0)
    major = max(float(det.major_axis_length_px), minor)
    ratio = major / minor
    decay = max(1e-6, float(PHI_ASPECT_DECAY))
    return float(np.clip(1.0 - np.exp(-(ratio - 1.0) / decay), 0.0, 1.0))


def overhead_obliquity_confidence(det: YOLODetection, bundle: dict) -> float:
    """Penalize overhead phi when the object is far from the overhead's
    principal point: off-axis objects show more side than top, distorting
    the projected minor axis. Returns 1.0 at principal point, ~0.0 at corner."""
    K = bundle.get("overhead_camera_matrix")
    if K is None:
        return 1.0  # no intrinsics -> can't measure obliquity, don't penalize
    K = np.asarray(K, dtype=np.float64)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if cx <= 0.0 or cy <= 0.0:
        return 1.0
    dx = float(det.centroid_px[0]) - cx
    dy = float(det.centroid_px[1]) - cy
    radial = float(np.hypot(dx, dy)) / float(np.hypot(cx, cy))
    return float(np.clip(1.0 - radial, 0.0, 1.0))


def stereo_phi_confidence(cand: ObjectCandidate) -> float:
    """Penalize stereo phi when the object is tall: the stereo cameras
    look at the table from a side angle, so tall objects expose a lot of
    side surface that distorts the stereo-mask minor axis the same way
    overhead obliquity does. Returns 1.0 for flat objects, decays with height."""
    h_cm = getattr(cand, "pointcloud_height_cm", None)
    if h_cm is None or not np.isfinite(float(h_cm)):
        return 1.0  # unknown height -> don't penalize
    decay = max(1e-6, float(PHI_STEREO_HEIGHT_DECAY_CM))
    return float(np.clip(np.exp(-max(0.0, float(h_cm)) / decay), 0.0, 1.0))


def _overhead_centroid_ray_phi(
    det: YOLODetection,
    z_mm: float,
    bundle: dict,
    *,
    select: str,
    perpendicular: bool,
) -> tuple[float | None, str]:
    image_angle, source = estimate_mask_centroid_ray_angle_deg(
        det,
        select=select,
        perpendicular=perpendicular,
    )
    if image_angle is None:
        return None, source

    centroid = np.asarray(det.centroid_px, dtype=np.float64).reshape(2)
    theta = np.deg2rad(float(image_angle))
    axis_px = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    half_len_px = max(
        12.0,
        0.5 * float(det.minor_axis_length_px if select == "shortest" else det.major_axis_length_px),
    )
    p0 = centroid - axis_px * half_len_px
    p1 = centroid + axis_px * half_len_px

    try:
        xy0, *_ = _project_overhead_centroid_to_robot_xy(p0, z_mm, bundle)
        xy1, *_ = _project_overhead_centroid_to_robot_xy(p1, z_mm, bundle)
    except Exception as exc:
        print(f"[PHI] overhead centroid ray projection failed: {exc}")
        return None, f"{source}_overhead_projection_failed"

    delta = np.asarray(xy1, dtype=np.float64).reshape(2) - np.asarray(xy0, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-6:
        return None, f"{source}_overhead_degenerate_robot_delta"

    phi = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0)
    return phi, f"{source}_overhead"


def _resolve_pick_phi(
    debug: CandidateDebug,
    overhead_det: YOLODetection | None,
    bundle: dict,
) -> None:
    """Confidence-weighted phi resolution. Updates cand.pick_phi_deg /
    cand.pick_phi_source and fills debug.phi_* fields. Falls through to the
    legacy "overhead-overrides-stereo" behavior when USE_CONFIDENCE_PHI_BLEND
    is False."""
    cand = debug.candidate
    stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))

    if PICK_PHI_MODE in {"centroid_longest_ray_perp", "centroid_shortest_ray_parallel"}:
        select = "longest" if PICK_PHI_MODE == "centroid_longest_ray_perp" else "shortest"
        perpendicular = PICK_PHI_MODE == "centroid_longest_ray_perp"
        phi: float | None = None
        source: str = "unknown"
        if overhead_det is not None:
            phi, source = _overhead_centroid_ray_phi(
                overhead_det,
                stereo_z,
                bundle,
                select=select,
                perpendicular=perpendicular,
            )
        if phi is None:
            phi = cand.pick_phi_deg
            source = cand.pick_phi_source
        cand.pick_phi_deg = None if phi is None else float(phi)
        cand.pick_phi_source = source
        debug.phi_stereo_deg = cand.pick_phi_deg
        debug.phi_overhead_deg = cand.pick_phi_deg if overhead_det is not None else None
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = None
        debug.phi_blend_source = source
        return

    if PICK_PHI_MODE == "pointcloud_shortest_path":
        cand.pick_phi_source = cand.pick_phi_source or "pointcloud_shortest_path"
        debug.phi_stereo_deg = cand.pick_phi_deg
        debug.phi_overhead_deg = None
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = None
        debug.phi_blend_source = cand.pick_phi_source
        return

    # Stereo phi always available (the stereo burst detection is cand.yolo).
    phi_stereo = float(cand.yolo.minor_axis_angle_deg)
    debug.phi_stereo_deg = phi_stereo if np.isfinite(phi_stereo) else None

    phi_overhead: float | None = None
    if overhead_det is not None and OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI:
        p = float(overhead_det.minor_axis_angle_deg)
        if np.isfinite(p):
            phi_overhead = p
    debug.phi_overhead_deg = phi_overhead

    # ---- legacy behavior: overhead overrides stereo if available ----
    if not USE_CONFIDENCE_PHI_BLEND:
        debug.phi_stereo_confidence = None
        debug.phi_overhead_confidence = None
        debug.phi_disagreement_deg = (
            wrapped_phi_diff_deg(phi_overhead, phi_stereo)
            if (phi_overhead is not None and debug.phi_stereo_deg is not None) else None
        )
        if phi_overhead is not None:
            cand.pick_phi_deg = phi_overhead
            cand.pick_phi_source = "overhead_minor_axis"
            debug.phi_blend_source = cand.pick_phi_source
        # else: leave whatever build_object_candidate set (stereo)
        return

    # ---- confidence-weighted blend ----
    c_stereo = aspect_phi_confidence(cand.yolo) * stereo_phi_confidence(cand)
    debug.phi_stereo_confidence = float(c_stereo)

    c_overhead = 0.0
    if phi_overhead is not None:
        c_overhead = aspect_phi_confidence(overhead_det) * overhead_obliquity_confidence(overhead_det, bundle)
    debug.phi_overhead_confidence = float(c_overhead)

    # Disagreement (wrapped) and warning.
    if phi_overhead is not None and debug.phi_stereo_deg is not None:
        diff = wrapped_phi_diff_deg(phi_overhead, phi_stereo)
        debug.phi_disagreement_deg = diff
        if abs(diff) > float(PHI_DISAGREEMENT_WARN_DEG):
            print(
                f"[PHI WARN] cand[{cand.index}] overhead={phi_overhead:+.1f} "
                f"stereo={phi_stereo:+.1f} diff={diff:+.1f}°  "
                f"c_oh={c_overhead:.2f} c_st={c_stereo:.2f}"
            )
    else:
        debug.phi_disagreement_deg = None

    # Build usable list.
    usable: list[tuple[str, float, float]] = []
    if phi_overhead is not None and c_overhead >= float(PHI_MIN_CONFIDENCE):
        usable.append(("overhead", phi_overhead, c_overhead))
    if debug.phi_stereo_deg is not None and c_stereo >= float(PHI_MIN_CONFIDENCE):
        usable.append(("stereo", debug.phi_stereo_deg, c_stereo))

    if not usable:
        if PHI_FALLBACK_TO_CURRENT_EE_PHI:
            cand.pick_phi_deg = None
            cand.pick_phi_source = "fallback_current_ee_phi"
        else:
            # Keep stereo phi if we have it; otherwise leave whatever's there.
            if debug.phi_stereo_deg is not None:
                cand.pick_phi_deg = debug.phi_stereo_deg
                cand.pick_phi_source = "fallback_stereo_low_conf"
            else:
                cand.pick_phi_source = "fallback_no_phi_available"
        debug.phi_blend_source = cand.pick_phi_source
        return

    if len(usable) == 1:
        label, phi, conf = usable[0]
        cand.pick_phi_deg = float(phi)
        cand.pick_phi_source = f"{label}_minor_axis_c{conf:.2f}"
        debug.phi_blend_source = cand.pick_phi_source
        return

    # Both usable -> wrapped blend with normalized confidences.
    (la, pa, ca), (lb, pb, cb) = usable
    total = ca + cb
    blended = wrapped_phi_blend_deg(pa, ca / total, pb, cb / total)
    cand.pick_phi_deg = float(blended)
    cand.pick_phi_source = f"blend_{la[:2]}{ca:.2f}_{lb[:2]}{cb:.2f}"
    debug.phi_blend_source = cand.pick_phi_source


def _match_overhead_to_candidates(
    overhead_dets: list[YOLODetection],
    candidate_debugs: list[CandidateDebug],
    bundle: dict,
) -> None:
    if not overhead_dets:
        print("[MATCH] no overhead detections; candidates will use stereo-only XY")
        for dbg in candidate_debugs:
            _resolve_pick_phi(dbg, None, bundle)
            _apply_xy_blend(dbg, bundle)
        return

    _hr("OVERHEAD MATCH", "-")
    for dbg in candidate_debugs:
        cand = dbg.candidate
        stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))
        stereo_xy = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2)

        pool = overhead_dets
        if OVERHEAD_MATCH_PREFER_SAME_CLASS:
            same = [d for d in overhead_dets if str(d.class_name) == str(cand.yolo.class_name)]
            if same:
                pool = same

        best_det = None
        best_dist = float("inf")
        best_xy = None

        for det in pool:
            try:
                xy, *_ = _project_overhead_centroid_to_robot_xy(det.centroid_px, stereo_z, bundle)
                dist = float(np.linalg.norm(xy - stereo_xy))
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best_det = det
                best_xy = xy

        matched_overhead_det: YOLODetection | None = None
        if best_det is not None and best_dist <= OVERHEAD_MATCH_MAX_DIST_MM:
            cand.overhead_centroid_px = np.asarray(best_det.centroid_px, dtype=np.float64).reshape(2)
            cand.overhead_bbox_px = tuple(float(v) for v in best_det.bbox)
            dbg.overhead_xy_mm = np.asarray(best_xy, dtype=np.float64).reshape(2) if best_xy is not None else None
            matched_overhead_det = best_det

            print(
                f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} "
                f"-> overhead {best_det.class_name:14s} dist={best_dist:6.1f} mm"
            )
        else:
            reason = "no projected overhead candidate" if best_det is None else f"best_dist={best_dist:.1f} > {OVERHEAD_MATCH_MAX_DIST_MM:.1f}"
            print(f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} -> NO MATCH ({reason})")

        # Phi resolution must happen before _apply_xy_blend so the
        # printed/displayed pick_phi reflects the confidence-weighted result.
        _resolve_pick_phi(dbg, matched_overhead_det, bundle)
        _apply_xy_blend(dbg, bundle)


def run_survey(
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict,
    rectifier: StereoRectifier,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    robot: Robot,
    bundle: dict,
) -> SurveyState:
    _hr("SURVEY REAL OBJECTS", "=")
    t0 = time.perf_counter()

    frames = capture_burst_frames(stereo, detector, stereo_calib, rectifier)
    if not frames:
        return SurveyState([], [], [], [], None, [], 0)

    run_yolo_on_burst(yolo, frames)
    tracks, kept = cluster_burst_detections(frames)

    candidates: list[CandidateDebug] = []
    for idx, track in enumerate(kept, start=1):
        dbg = build_candidate_from_track(
            track=track,
            frames=frames,
            raft=raft,
            stereo_calib=stereo_calib,
            robot=robot,
            bundle=bundle,
            index=idx,
        )
        if dbg is not None:
            candidates.append(dbg)

    # Re-index after pointcloud filtering.
    for idx, dbg in enumerate(candidates, start=1):
        dbg.candidate.index = idx

    ok_oh, overhead_frame = overhead.read()
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


# ============================================================
# PRINTING / VALIDATION
# ============================================================

def print_survey_candidate_summary(candidates: list[CandidateDebug]) -> None:
    _hr("CANDIDATES", "-")
    if not candidates:
        print("[CANDIDATES] none")
        return

    for dbg in candidates:
        c = dbg.candidate
        dxy = "--" if dbg.xy_disagreement_mm is None else f"{dbg.xy_disagreement_mm:.1f}mm"
        print(
            f"  [{c.index}] {c.yolo.class_name:14s} "
            f"hits={dbg.track.hit_count:2d}/{BURST_COUNT} "
            f"best_frame={dbg.best_frame_i + 1:02d} "
            f"pts={dbg.point_count:5d} "
            f"stereo_xy={_fmt_xy(dbg.stereo_xy_mm)} "
            f"overhead_xy={_fmt_xy(dbg.overhead_xy_mm)} "
            f"cmd_xy={_fmt_xy(c.target_xy)} "
            f"dXY={dxy:>8s} "
            f"Z={c.object_robot_xyz_raw[2]:7.1f} "
            f"grasp={c.grasp_robot_z:7.1f} "
            f"phi={_fmt_candidate_phi(c)}"
        )


def print_validation(state: SurveyState | None, selected_index: int) -> None:
    if state is None or not state.candidates:
        print("[VALIDATE] no survey candidates. Press s first.")
        return

    _hr("VALIDATION — ALL FRAME INFORMATION", "=")
    print(f"[BURST] frames captured={len(state.burst_frames)} requested={BURST_COUNT}")
    for frame in state.burst_frames:
        print(
            f"  frame {frame.frame_i + 1:02d}: "
            f"detections={len(frame.detections):2d} "
            + ", ".join(f"{d.class_name}:{d.confidence:.2f}" for d in frame.detections)
        )

    _hr("TRACKS", "-")
    for track in state.tracks:
        keep = "KEEP" if track in state.kept_tracks else "drop"
        c = track.mean_centroid_px
        print(
            f"  track[{track.track_id:02d}] {keep:4s} {track.class_name:14s} "
            f"hits={track.hit_count:2d}/{BURST_COUNT} mean_px=({c[0]:.1f},{c[1]:.1f})"
        )

    _hr("CANDIDATE GEOMETRY", "-")
    print_survey_candidate_summary(state.candidates)

    dbg = state.candidates[selected_index % len(state.candidates)]
    c = dbg.candidate

    _hr(f"SELECTED [{c.index}] {c.yolo.class_name}", "-")
    print(f"best_frame_i           = {dbg.best_frame_i + 1}/{BURST_COUNT}")
    print(f"burst_hits             = {dbg.track.hit_count}/{BURST_COUNT}")
    print(f"valid_point_count      = {c.valid_point_count}")
    print(f"mask centroid px       = {np.round(c.yolo.centroid_px, 2)}")
    print(f"mask area px           = {c.yolo.mask_area}")
    print(f"bbox px                = {tuple(round(float(v), 1) for v in c.yolo.bbox)}")
    print(f"major axis px/deg      = {c.yolo.major_axis_length_px:.1f} @ {c.yolo.major_axis_angle_deg:.1f}°")
    print(f"minor axis px/deg      = {c.yolo.minor_axis_length_px:.1f} @ {c.yolo.minor_axis_angle_deg:.1f}°")
    print(f"pick phi               = {_fmt_candidate_phi(c)}")
    print(f"phi_stereo_deg         = {dbg.phi_stereo_deg}")
    print(f"phi_overhead_deg       = {dbg.phi_overhead_deg}")
    print(f"phi_stereo_confidence  = {dbg.phi_stereo_confidence}")
    print(f"phi_overhead_confidence= {dbg.phi_overhead_confidence}")
    print(f"phi_disagreement_deg   = {dbg.phi_disagreement_deg}")
    print(f"phi_blend_source       = {dbg.phi_blend_source}")
    print()
    print(f"centroid_cam_xyz       = {_fmt_xyz(c.centroid_cam_xyz)}")
    print(f"top_cam_xyz            = {_fmt_xyz(c.top_cam_xyz)}")
    print(f"target_cam_xyz         = {_fmt_xyz(c.target_cam_xyz)}")
    print()
    print(f"object_robot_xyz_raw   = {_fmt_xyz(c.object_robot_xyz_raw)}")
    print(f"object_robot_xyz_corr  = {_fmt_xyz(c.object_robot_xyz_corrected)}")
    print(f"stereo_xy              = {_fmt_xy(dbg.stereo_xy_mm)}")
    print(f"overhead_centroid_px   = {None if c.overhead_centroid_px is None else np.round(c.overhead_centroid_px, 2)}")
    print(f"overhead_xy            = {_fmt_xy(dbg.overhead_xy_mm)}")
    print(f"target_xy              = {_fmt_xy(c.target_xy)}")
    print(f"xy_source              = {c.target_xy_source_effective}")
    print(f"blend_weight_overhead  = {dbg.blend_weight_overhead:.2f}")
    print(f"xy_disagreement_mm     = {dbg.xy_disagreement_mm}")
    print()
    print(f"lookup_z_used          = {c.lookup_z_used:.1f}  clamped={c.lookup_z_clamped}")
    print(f"support_distance_mm    = {c.support_distance_mm:.1f} nearest_idx={c.support_index}")
    print(f"pointcloud_height_cm   = {getattr(c, 'pointcloud_height_cm', None)}")
    print(f"pointcloud_z_range_mm  = {getattr(c, 'pointcloud_height_robot_z_range_mm', None)}")
    print()
    print(f"Z_MAX_MM               = {Z_MAX_MM:.1f}")
    print(f"GRIPPER_OFFSET_MM      = {GRIPPER_OFFSET_MM:.1f}")
    print(f"hover_robot_z          = {c.hover_robot_z:.1f}")
    print(f"grasp_robot_z          = {c.grasp_robot_z:.1f}")
    print("=" * 96)


# ============================================================
# ROBOT MOTION
# ============================================================

def startup_robot() -> Robot | None:
    if not CONNECT_ROBOT:
        print("[ROBOT] CONNECT_ROBOT=False; validation only.")
        return None

    robot = Robot(connect=True)
    require_soft_limits(robot)

    if ENABLE_MOTORS_ON_START:
        robot.enable(True)
    if INIT_DRIVERS_ON_START:
        robot.init_drivers()

    print("\nStartup options:")
    print("  h = run HOME now")
    print("  c = continue from current Teensy step counters, no homing")
    print("  a = assume robot is physically at configured home_pose, no homing")
    choice = input("Choose h/c/a: ").strip().lower()

    if choice == "h":
        if not robot.home():
            raise RuntimeError("HOME failed")
    elif choice == "c":
        if not robot.sync_estimate_from_teensy_steps():
            raise RuntimeError("Teensy sync failed")
        robot.print_estimate()
    elif choice == "a":
        robot.assume_homed()
        robot.print_estimate()
    else:
        raise RuntimeError("Unknown startup choice")

    print("[ROBOT] ready")
    try:
        # robot.send("HOMEJ3")
        time.sleep(1)
        robot.move_cartesian(X_SURVEY, Y_SURVEY, Z_SURVEY, 0, move_time_s=2.0)

    except Exception:
        pass

    return robot


def execute_test_descent_no_claw(
    robot: Robot | None,
    dbg: CandidateDebug,
    bundle: dict,
) -> tuple[bool, bool]:
    if robot is None:
        print("[TEST] robot is not connected.")
        return False, False

    c = dbg.candidate
    if REQUIRE_OVERHEAD_XY_FOR_TEST and dbg.overhead_xy_mm is None:
        print("[TEST] refused: overhead XY unavailable and REQUIRE_OVERHEAD_XY_FOR_TEST=True")
        return False, False

    if REFUSE_TEST_IF_TOO_FEW_POINTS and c.valid_point_count < MIN_VALID_OBJECT_POINTS:
        print(f"[TEST] refused: points {c.valid_point_count} < {MIN_VALID_OBJECT_POINTS}")
        return False, False

    x = float(c.target_xy[0])
    y = float(c.target_xy[1])
    z_travel = float(Z_MAX_MM)
    z_hover = float(Z_MAX_MM)
    z_grasp = float(c.grasp_robot_z)
    phi = _candidate_phi_or_current(robot, c)

    for label, z in (("travel", z_travel), ("hover", z_hover), ("grasp", z_grasp)):
        reason = _validate_z_command(z, f"[TEST] {label}")
        if reason:
            print(f"[TEST] ABORT: {reason}")
            return False, False

    _hr("TEST DESCENT — NO CLAW CLOSE", "-")
    print(f"[TEST] candidate [{c.index}] {c.yolo.class_name}")
    print(f"[TEST] XY={_fmt_xy(c.target_xy)} source={c.target_xy_source_effective}")
    print(f"[TEST] stereo_xy={_fmt_xy(dbg.stereo_xy_mm)} overhead_xy={_fmt_xy(dbg.overhead_xy_mm)}")
    print(f"[TEST] blend overhead={dbg.blend_weight_overhead:.2f} stereo={1.0 - dbg.blend_weight_overhead:.2f}")
    print(f"[TEST] phi={phi:.2f} deg source={c.pick_phi_source}")
    print(f"[TEST] Z plan: travel={z_travel:.1f}, hover={z_hover:.1f}, grasp={z_grasp:.1f}")
    print("[TEST] Opening claw only for clearance. It will NOT close.")
    try:
        robot.servo(CLAW_OPEN_DEG)
    except Exception:
        pass

    print("[TEST] 1/3 raise to Z_MAX")
    if not move_cartesian_nonnegative_z(robot, "[TEST] raise", z_mm=z_travel, move_time_s=RAISE_MOVE_TIME_S):
        return False, False

    print("[TEST] 2/3 move XY + semiminor phi at Z_MAX")
    if not move_cartesian_nonnegative_z(
        robot,
        "[TEST] XY+phi",
        x_mm=x,
        y_mm=y,
        z_mm=z_travel,
        phi_deg=phi,
        move_time_s=XY_MOVE_TIME_S,
    ):
        return False, False

    print("[TEST] 3/3 slow descent to grasp height")
    total_drop = max(0.0, z_hover - z_grasp)
    n_steps = max(1, int(math.ceil(total_drop / max(DESCENT_STEP_MM, 0.1))))
    for i, z in enumerate(np.linspace(z_hover, z_grasp, n_steps + 1)[1:], start=1):
        print(f"  descent {i:02d}/{n_steps}: z={float(z):.1f}")
        if not move_cartesian_nonnegative_z(robot, "[TEST] descent", z_mm=float(z), move_time_s=DESCENT_STEP_TIME_S):
            return False, False

    print("[TEST] reached grasp height. No claw close commanded.")
    request_resurvey = fine_tune_xy_blend_loop(robot, dbg, bundle, z_grasp, phi)
    return True, request_resurvey


def fine_tune_xy_blend_loop(
    robot: Robot,
    dbg: CandidateDebug,
    bundle: dict,
    z_current: float,
    phi_current: float,
) -> bool:
    _hr("FINE TUNE XY BLEND AT GRASP HEIGHT", "-")
    print("Controls:")
    print("  a = more stereo XY  (decrease overhead weight)")
    print("  d = more overhead XY (increase overhead weight)")
    print("  s = re-survey now (exit fine tune and run survey)")
    print("  v = print selected geometry")
    print("  q/ESC = exit fine tune")
    print("Each a/d move commands the same Z and phi, only changing XY by the blend ratio.")
    print()

    while True:
        c = dbg.candidate
        print(
            f"[FINE] w_overhead={dbg.blend_weight_overhead:.2f} "
            f"cmd_xy={_fmt_xy(c.target_xy)} "
            f"stereo={_fmt_xy(dbg.stereo_xy_mm)} overhead={_fmt_xy(dbg.overhead_xy_mm)}"
        )
        key = read_command_key(delay_ms=0)

        if key is None:
            key = input("[FINE] key a/d/s/v/q: ").strip().lower()[:1] or None

        if key == "s":
            print("[FINE] re-survey requested")
            return True

        if key in ("q", "escape", "\x1b"):
            print("[FINE] exit")
            return False

        if key == "v":
            tmp_state = SurveyState([], [], [], [dbg], None, [], 0)
            print_validation(tmp_state, 0)
            continue

        if key not in ("a", "d"):
            continue

        old_w = dbg.blend_weight_overhead
        if key == "a":
            new_w = max(0.0, old_w - FINE_TUNE_BLEND_STEP)
        else:
            new_w = min(1.0, old_w + FINE_TUNE_BLEND_STEP)

        _apply_xy_blend(dbg, bundle, w_overhead=new_w)
        x = float(c.target_xy[0])
        y = float(c.target_xy[1])

        print(
            f"[FINE] moving XY at z={z_current:.1f}, phi={phi_current:.1f}: "
            f"w_overhead {old_w:.2f} -> {new_w:.2f}, xy={_fmt_xy(c.target_xy)}"
        )
        ok = move_cartesian_nonnegative_z(
            robot,
            "[FINE] XY blend adjust",
            x_mm=x,
            y_mm=y,
            z_mm=float(z_current),
            phi_deg=float(phi_current),
            move_time_s=FINE_TUNE_XY_MOVE_TIME_S,
        )
        if not ok:
            print("[FINE] move failed; staying in fine tune loop.")


# ============================================================
# DISPLAY
# ============================================================

def put_text_outline(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def resize_to_panel(img: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
    if img is None:
        out = np.zeros((height, width, 3), dtype=np.uint8)
        put_text_outline(out, label + " unavailable", (20, height // 2), scale=0.7, color=(80, 80, 255), thickness=2)
        return out
    out = cv2.resize(img, (width, height))
    put_text_outline(out, label, (10, 26), scale=0.65, color=(255, 255, 255), thickness=2)
    return out


def draw_overhead_panel(state: SurveyState | None, live_frame: np.ndarray | None) -> np.ndarray:
    frame = None
    if state is not None and state.overhead_frame is not None:
        frame = state.overhead_frame.copy()
    elif live_frame is not None:
        frame = live_frame.copy()

    if frame is None:
        return resize_to_panel(None, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX, "OVERHEAD")

    # Draw overhead detections.
    if state is not None:
        for i, det in enumerate(state.overhead_detections):
            x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cx, cy = np.round(det.centroid_px).astype(int)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
            put_text_outline(frame, f"OH {i} {det.class_name}", (x1, max(20, y1 - 6)), scale=0.5, color=(255, 160, 0), thickness=1)

        for idx, dbg in enumerate(state.candidates):
            c = dbg.candidate
            if c.overhead_centroid_px is None:
                continue
            px, py = np.round(c.overhead_centroid_px).astype(int)
            sel = idx == state.selected_index
            color = (0, 255, 0) if sel else (0, 220, 255)
            cv2.circle(frame, (int(px), int(py)), 14 if sel else 9, color, 2)
            put_text_outline(frame, str(idx + 1), (int(px) + 14, int(py) - 8), scale=0.65, color=color, thickness=2)

    return resize_to_panel(frame, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX, "OVERHEAD")


def make_display(
    state: SurveyState | None,
    live_overhead: np.ndarray | None,
    live_left: np.ndarray | None,
    live_right: np.ndarray | None,
) -> np.ndarray:
    W = COMBINED_WIDTH_PX
    top = draw_overhead_panel(state, live_overhead)

    half = W // 2
    if state is not None and state.candidates:
        dbg = state.candidates[state.selected_index % len(state.candidates)]
        left_panel = resize_to_panel(dbg.left_overlay, half, STEREO_DRAW_H_PX, f"BEST BURST LEFT #{state.selected_index + 1}")
        right_panel = resize_to_panel(dbg.disparity_overlay, half, STEREO_DRAW_H_PX, "RAFT DISPARITY + MASK")
    else:
        left_panel = resize_to_panel(live_left, half, STEREO_DRAW_H_PX, "LIVE LEFT")
        right_panel = resize_to_panel(live_right, half, STEREO_DRAW_H_PX, "LIVE RIGHT")

    bottom = np.hstack([left_panel, right_panel])

    status = np.zeros((STATUS_H_PX, W, 3), dtype=np.uint8)
    if state is None or not state.candidates:
        lines = [
            "No survey yet. Press s to run 10-frame burst survey.",
            "Keys: s=survey | r=rotate kept candidates | v=validate | t=test slow descent/no claw close | q=quit",
        ]
    else:
        dbg = state.candidates[state.selected_index % len(state.candidates)]
        c = dbg.candidate
        c_oh = dbg.phi_overhead_confidence
        c_st = dbg.phi_stereo_confidence
        phi_conf_tag = ""
        if c_oh is not None or c_st is not None:
            phi_conf_tag = f" [cOH={(c_oh or 0.0):.2f} cST={(c_st or 0.0):.2f}]"
        lines = [
            f"Selected [{state.selected_index + 1}/{len(state.candidates)}] {c.yolo.class_name} | hits={dbg.track.hit_count}/{BURST_COUNT} | best_frame={dbg.best_frame_i + 1}",
            f"cmd_xy={_fmt_xy(c.target_xy)}  stereo={_fmt_xy(dbg.stereo_xy_mm)}  overhead={_fmt_xy(dbg.overhead_xy_mm)}  wOH={dbg.blend_weight_overhead:.2f}",
            f"Z={c.object_robot_xyz_raw[2]:.1f}  grasp={c.grasp_robot_z:.1f}  phi={_fmt_candidate_phi(c)}{phi_conf_tag}  pts={c.valid_point_count}",
            "Keys: s=survey | r=rotate candidates | v=validate print | t=test descent/no close | q=quit",
        ]
    for i, line in enumerate(lines):
        put_text_outline(status, line, (8, 24 + i * 25), scale=0.58, color=(230, 230, 230), thickness=1)

    return np.vstack([top, bottom, status])


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if PICK_PHI_MODE not in VALID_PICK_PHI_MODES:
        raise ValueError(
            f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}; "
            f"expected one of {sorted(VALID_PICK_PHI_MODES)}"
        )

    _hr("REAL OBJECT PICK VALIDATION BUILDUP", "=")
    print(f"[MAIN] burst: {BURST_COUNT} frames, keep if hits >= {MIN_BURST_HITS}")
    print(f"[MAIN] Z policy: travel=hover=Z_MAX={Z_MAX_MM:.1f}, grasp=z_stereo+offset={GRIPPER_OFFSET_MM:.1f}")
    print(f"[MAIN] XY blend: overhead weight={OVERHEAD_XY_BLEND_WEIGHT:.2f}, stereo={1.0 - OVERHEAD_XY_BLEND_WEIGHT:.2f}")
    print(f"[MAIN] phi from overhead semiminor after match: {OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI}")
    print(
        f"[MAIN] phi confidence blend: USE={USE_CONFIDENCE_PHI_BLEND}  "
        f"min_conf={PHI_MIN_CONFIDENCE:.2f}  disagree_warn={PHI_DISAGREEMENT_WARN_DEG:.1f}°  "
        f"aspect_decay={PHI_ASPECT_DECAY:.2f}  stereo_h_decay_cm={PHI_STEREO_HEIGHT_DECAY_CM:.1f}"
    )
    print(f"[MAIN] phi mode: {PICK_PHI_MODE}")
    print("[MAIN] This script never closes the claw during test.")
    _hr("", "=")

    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)

    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    detector = build_detector()

    print_matrix_labeled(
        "A_robot_from_cam_xyz_3x4",
        bundle.get("A_robot_from_cam_xyz_3x4"),
        ["robot_x", "robot_y", "robot_z"],
        ["cam_x", "cam_y", "cam_z", "1"],
    )

    yolo, raft = load_vision(device_info)
    rectifier = StereoRectifier(stereo_calib)

    print(f"[MAIN] opening overhead camera index {OVERHEAD_INDEX}")
    overhead = SimpleOverheadCamera(OVERHEAD_INDEX)
    print(f"[MAIN] opening stereo camera index {STEREO_INDEX}")
    stereo = SimpleStereoCamera(STEREO_INDEX)

    robot = startup_robot()

    state: SurveyState | None = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + STATUS_H_PX)

    running = True
    try:
        while running:
            ok_oh, live_overhead = overhead.read()
            if not ok_oh:
                live_overhead = None

            ok_st, _full, live_left, live_right = stereo.read_pair()
            if not ok_st:
                live_left = None
                live_right = None

            display = make_display(state, live_overhead, live_left, live_right)
            cv2.imshow(WINDOW, display)

            key = read_command_key(delay_ms=1)
            if key is None:
                continue

            if key in ("q", "escape", "\x1b"):
                print("[MAIN] quit requested")
                running = False

            elif key == "s":
                state = run_survey(
                    overhead=overhead,
                    stereo=stereo,
                    detector=detector,
                    stereo_calib=stereo_calib,
                    rectifier=rectifier,
                    yolo=yolo,
                    raft=raft,
                    robot=robot,
                    bundle=bundle,
                )
                if state.candidates:
                    state.selected_index = 0
                    print("[SURVEY] selected candidate 1. Press r to rotate.")
                else:
                    print("[SURVEY] no kept candidates after burst/pointcloud filtering.")

            elif key == "r":
                if state is None or not state.candidates:
                    print("[ROTATE] no candidates. Press s first.")
                else:
                    state.selected_index = (state.selected_index + 1) % len(state.candidates)
                    dbg = state.candidates[state.selected_index]
                    print(
                        f"[ROTATE] selected [{state.selected_index + 1}/{len(state.candidates)}] "
                        f"{dbg.candidate.yolo.class_name} hits={dbg.track.hit_count}/{BURST_COUNT}"
                    )

            elif key == "v":
                if state is None or not state.candidates:
                    print("[VALIDATE] no candidates. Press s first.")
                else:
                    print_validation(state, state.selected_index)

            elif key == "t":
                if state is None or not state.candidates:
                    print("[TEST] no candidates. Press s first.")
                else:
                    dbg = state.candidates[state.selected_index]
                    print_validation(state, state.selected_index)
                    ok, request_resurvey = execute_test_descent_no_claw(robot, dbg, bundle)
                    print(f"[TEST] {'success' if ok else 'failed'}")
                    if ok and request_resurvey:
                        print("[TEST] running requested re-survey...")
                        state = run_survey(
                            overhead=overhead,
                            stereo=stereo,
                            detector=detector,
                            stereo_calib=stereo_calib,
                            rectifier=rectifier,
                            yolo=yolo,
                            raft=raft,
                            robot=robot,
                            bundle=bundle,
                        )
                        if state.candidates:
                            state.selected_index = 0
                            print("[SURVEY] selected candidate 1. Press r to rotate.")
                        else:
                            print("[SURVEY] no kept candidates after burst/pointcloud filtering.")

            elif key == "o" and robot is not None:
                robot.servo(CLAW_OPEN_DEG)
                print(f"[CLAW] open ({CLAW_OPEN_DEG}°)")

            elif key == "p" and robot is not None:
                _print_fk(robot, "[FK]")

            elif key == "[" and robot is not None:
                jog_nonnegative_z(robot, "[JOG]", dz=-5.0, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            elif key == "]" and robot is not None:
                jog_nonnegative_z(robot, "[JOG]", dz=+5.0, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            elif key == "e" and robot is not None:
                robot.enable(True)
                robot.init_drivers()
                print("[ROBOT] enabled")

            elif key == "d" and robot is not None:
                robot.enable(False)
                print("[ROBOT] disabled")
            
            elif key == "g" and robot is not None:
                robot.move_cartesian(X_SURVEY, Y_SURVEY, Z_SURVEY, 0, move_time_s=3.0)
                print("[ROBOT] moved to survey pose")

    finally:
        try:
            overhead.release()
        except Exception:
            pass
        try:
            stereo.release()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[MAIN] interrupted by user")
        cv2.destroyAllWindows()
    except Exception:
        traceback.print_exc()
        cv2.destroyAllWindows()
        raise
