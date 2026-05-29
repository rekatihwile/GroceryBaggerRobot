from __future__ import annotations

"""Autonomous adjacent placement with YOLO-only missed-pick recovery.

Experimental script cloned from pick_place_two_objects_autonomous.py.  The
existing autonomous scripts are intentionally left untouched.
"""

# ============================================================
# USER SETTINGS - TUNE THESE FIRST
# ============================================================

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion.z_safety_config import (  # noqa: E402
    DEFAULT_Z_SAFETY,
    PLACE_RELEASE_GAP_MM as SHARED_PLACE_RELEASE_GAP_MM,
    print_z_safety_settings,
    validate_z_command,
)

# General run control.  Set RUN_UNTIL_NO_VALID_CANDIDATE=True for a feeder loop.
TARGET_OBJECT_COUNT = 10
RUN_UNTIL_NO_VALID_CANDIDATE = False
MAX_OBJECT_COUNT_SAFETY = 12
REQUIRE_CONFIRM_BEFORE_REAL_MOTION = False
NO_CANDIDATE_RETRY_COUNT = 1
NO_CANDIDATE_RETRY_DELAY_S = 1.0
NO_CANDIDATE_AFTER_RETRIES_MODE = "continuous"  # "continuous", "wait_for_resume", or "stop"
NO_CANDIDATE_RESUME_KEY = "r"
# When no candidate is found and continuous mode re-surveys, move to the survey/home
# pose first so the arm is out of the camera's view.
NO_CANDIDATE_RECOVERY_MOVE_ENABLED = True

# Placement geometry and packing fit.
# Lower PAD_X_MM / PAD_Y_MM / PAD_Z_MM to make packing tighter across all objects.
# Increase them if you want more conservative spacing between packed AABBs.
PAD_X_MM = 0
PAD_Y_MM = 0
PAD_Z_MM = 20.0
ADJACENT_DIRECTION = "left"
# Efficient packing chooses the best object for the next slot by fit first, then volume.
EFFICIENT_PACKING_ENABLED = True
EFFICIENT_PACKING_REQUIRE_SLOT_FIT = True
# Small XY nudges are tried only when they improve slot fit.
PLACE_XY_NUDGE_ENABLED = True
PLACE_XY_NUDGE_STEP_MM = 5.0
PLACE_XY_NUDGE_MAX_MM = 20.0
# The gripper footprint check still enforces bag containment.
PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG = True
PLACE_GRIPPER_FOOTPRINT_WIDTH_MM = 60.0
PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM = 70.0
PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM = 1.0
PLACE_GRIPPER_FOOTPRINT_DEFAULT_SERVO_DEG = 55.0
PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE = True
PLACE_ROTATION_CANDIDATE_OFFSETS_DEG = [0.0, 90.0]

# Autonomous survey timing. The next survey is submitted right before the place
# descent move, so capture begins while the robot is lowering to release.
PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT = True
PREFETCH_PLACE_DESCENT_DELAY_S = 0.0
SELECTION_DISPLAY_HOLD_S = 0.25
HOLD_WINDOW_AFTER_RUN = True
CLEAR_BOX_Z_MM = 275.0
CLEAR_BOX_MOVE_TIME_S = 1.25

# Overhead camera freshness. If the overhead view looks stale/phantom, increase
# discard frames. This is applied immediately before overhead YOLO matching.
OVERHEAD_FRESH_READ_DISCARD_FRAMES = 6
OVERHEAD_FRESH_READ_DELAY_S = 0.02

# Platform footprint.  Keep this aligned with
# calibration/calibrate_all_safe_grid_xyz_models.py SCAN_PRESETS["staging_refined"].
PLATFORM_GRID_X_MM = list(range(40, 380, 100))
PLATFORM_GRID_Y_MM = list(range(40, 515, 150))
PLATFORM_GRID_Z_MM = [0, 50, 150, 200]
PLATFORM_X_MIN_MM = float(min(PLATFORM_GRID_X_MM))
PLATFORM_X_MAX_MM = float(max(PLATFORM_GRID_X_MM))
PLATFORM_Y_MIN_MM = float(min(PLATFORM_GRID_Y_MM))
PLATFORM_Y_MAX_MM = float(max(PLATFORM_GRID_Y_MM))

# Missed-pick YOLO watchdog.  This is deliberately cheap: YOLO burst first,
# RAFT/pointcloud only after this watchdog says the object is probably still
# at the pick site.
MISS_CHECK_ENABLED = True
MISS_CHECK_CAMERA = "overhead"  # "overhead" is easiest to compare to the original pick-site detection.
MISS_BURST_COUNT = 8
MISS_MIN_HITS = 3
MISS_CHECK_TIMEOUT_S = 4.0
MISS_CHECK_PERIOD_S = 0.15
MISS_CONFIRM_AT_PLACE_HOVER_ONLY = True
MISS_MATCH_MAX_ROBOT_DIST_MM = 35.0
MISS_MATCH_MIN_IOU = 0.20
MISS_MATCH_AREA_RATIO_MIN = 0.50
MISS_MATCH_AREA_RATIO_MAX = 2.00
MISS_SCORE_THRESHOLD = 0.65
MISS_CLASS_WEIGHT = 0.25
MISS_ROBOT_DIST_WEIGHT = 0.35
MISS_IOU_WEIGHT = 0.20
MISS_AREA_WEIGHT = 0.20

# Recovery pose / J3 rehome.  The recovery move is validated with the shared Z
# safety policy before any motion is sent.  XYZ is loaded from
# scripts/pick_one_place_one.py's survey pose so survey and HOMEJ3 recovery use
# the same physical staging pose.
RECOVERY_POSE_X_MM = None
RECOVERY_POSE_Y_MM = None
RECOVERY_POSE_Z_MM = None
RECOVERY_POSE_PHI_DEG = 0.0
RECOVERY_MOVE_TIME_S = 1.50
RECOVERY_REHOME_J3_ENABLED = True
RECOVERY_DROP_Z_BEFORE_REHOME_MM = None
RECOVERY_REHOME_TIMEOUT_S = 120.0

# Retry grasp.  A retry uses the same pick machinery with a copied candidate,
# a higher/saner grasp target, and a wider starting claw.
MAX_PICK_RETRIES_PER_OBJECT = 1
RETRY_SAFE_PICK_ENABLED = True
RETRY_GRASP_Z_OFFSET_MM = 10.0
RETRY_START_CLAW_EXTRA_DEG = 8.0
RETRY_START_CLAW_MAX_DEG = 80.0
RETRY_XY_SAME_THRESHOLD_MM = 15.0
RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY = True
RETRY_RELOCALIZE_WITH_RAFT_ONLY_AFTER_MISS = True
ON_RETRY_FAIL = "stop"  # "stop" or "skip"

# Best-candidate filters. These are intentionally conservative and easy to tune.
BEST_REQUIRE_POSITIVE_PLATFORM_XY = True
BEST_PLATFORM_MIN_X_MM = PLATFORM_X_MIN_MM
BEST_PLATFORM_MIN_Y_MM = PLATFORM_Y_MIN_MM
BEST_WORKSPACE_X_MIN_MM = PLATFORM_X_MIN_MM
BEST_WORKSPACE_X_MAX_MM = PLATFORM_X_MAX_MM
BEST_WORKSPACE_Y_MIN_MM = PLATFORM_Y_MIN_MM
BEST_WORKSPACE_Y_MAX_MM = PLATFORM_Y_MAX_MM
BEST_REQUIRE_ROBOTFRAME_CENTROID_XY_IN_PLATFORM_BOUNDS = True
BEST_ROBOTFRAME_CENTROID_X_MIN_MM = PLATFORM_X_MIN_MM
BEST_ROBOTFRAME_CENTROID_X_MAX_MM = PLATFORM_X_MAX_MM
BEST_ROBOTFRAME_CENTROID_Y_MIN_MM = PLATFORM_Y_MIN_MM
BEST_ROBOTFRAME_CENTROID_Y_MAX_MM = PLATFORM_Y_MAX_MM
BEST_REQUIRE_MIN_ROBOTFRAME_CENTROID_Z = False
BEST_MIN_ROBOTFRAME_CENTROID_Z_MM = -200.0
BEST_USE_ROBOT_REACH_CHECK = True
BEST_ROBOT_REACH_MARGIN_MM = 2.0
BEST_REQUIRE_SOFT_POSE_SAFE = True
BEST_SOFT_POSE_CHECK_Z_MM = 275.0
BEST_CENTER_GATE_ENABLED = False
BEST_MAX_IMAGE_CENTER_NORM_RADIUS = 0.85
BEST_CLUSTER_GATE_ENABLED = False
BEST_MAX_CLUSTER_DISTANCE_MM = 600.0
BEST_REJECT_PLACED_OVERLAP = True
BEST_PLACED_OVERLAP_MARGIN_MM = 25.0
BEST_MIN_VOLUME_MM3 = 1.0
BEST_MAX_VOLUME_CM3 = 3000.0

# IMPORTANT:
# This flag may disable optional dynamic correction, but it must not bypass the
# shared minimum height, safety padding, or minimum Z clamp.
USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT = False
PLACE_RELEASE_GAP_MM = SHARED_PLACE_RELEASE_GAP_MM
PLACE_Z_UNCERTAINTY_GAIN = 0.1
PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 3.0
PLACE_Z_POLICY_MODE = "negative_bin_hang"  # "shared", "negative_bin_hang", or "negative_bin_simple"
PLACE_NEGATIVE_BIN_PLATFORM_Z_MM = -200.0
PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM = 0.0
PLACE_NEGATIVE_BIN_USE_EXISTING_STACK = True
PLACE_NEGATIVE_BIN_HANG_WEIGHT = 0.70
PLACE_NEGATIVE_BIN_SIMPLE_WEIGHT = 0.30
PLACE_NEGATIVE_BIN_CLEARANCE_MM = 0.0
PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING = False
USE_DYNAMIC_RELEASE_FOR_PLACE = False
DYNAMIC_RELEASE_TIMEOUT_S = 45.0

# Placement should only crack the claw open so it does not hit the object already placed.
PLACE_CLAW_OPEN_DEG = 45
COARSE_MOVE_TIME_S = 1.10
XY_MOVE_TIME_S = 1.50
PLACE_Z_MOVE_TIME_S = 0.60

WINDOW = "Autonomous Pick Place - Largest Volume"

# ============================================================

import copy
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
import math
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import cv2
import numpy as np

from motion.place_z_policy import compute_place_z_plan
from motion.pick_z_policy import compute_pick_z_plan
from motion.pick_place_sequence import (
    PlaceSequenceSettings,
    _command_servo_angle as _sequence_command_servo_angle,
    _move_checked as _sequence_move_checked,
)
from planning.aabb_utils import aabb_from_object_candidate, make_aabb_from_center_size, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement
import scripts.pick_one_place_one as _pick_one_mod
from scripts.autonomous_best_candidate import (
    BestCandidateConfig,
    BestCandidateResult,
    choose_best_candidate,
)
from scripts.pick_one_place_one import (
    BUNDLE_PATH,
    CLAW_OPEN_DEG,
    STEREO_CALIBRATION_PATH,
    USE_CUDA,
    USE_HALF,
    OVERHEAD_INDEX,
    STEREO_INDEX,
    COMBINED_WIDTH_PX,
    OVERHEAD_DRAW_H_PX,
    STEREO_DRAW_H_PX,
    STATUS_H_PX,
    Z_MAX_MM,
    _configure_modules,
    _confirm,
    get_use_z_ground_model_for_pick_surface,
    _load_place_surface_zone,
    execute_pick_selected,
)
from scripts.pick_validation_display import _hr, make_display, put_text_outline
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_xy_resolver import project_overhead_centroid_to_robot_xy
from vision.pick_survey_pipeline import load_vision, run_survey
import vision.pick_survey_pipeline as _survey_pipeline_mod
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLODetection
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from motion.pick_validation_motion import startup_robot
from test_calibration_bundle_live_stereo_z_pickplace import (
    load_bundle,
    load_stereo_calibration,
    print_matrix_labeled,
    read_command_key,
)


class UserAbort(RuntimeError):
    pass


class ClearBoxRequest(RuntimeError):
    pass


@dataclass
class PickAttemptRecord:
    object_i: int
    candidate_debug: CandidateDebug
    class_name: str
    original_robot_xy_mm: np.ndarray
    original_target_xy_mm: np.ndarray
    original_bbox_xyxy: tuple[float, float, float, float] | None
    original_mask_area_px: int | None
    original_bbox_area_px: float | None
    original_object_height_mm: float
    original_phi_deg: float
    attempt_number: int
    timestamp_s: float


@dataclass
class MissMatchResult:
    is_miss: bool
    score: float
    hits: int
    burst_count: int
    best_detection: YOLODetection | None
    best_robot_xy_mm: np.ndarray | None
    robot_dist_mm: float | None
    iou: float | None
    area_ratio: float | None
    same_class: bool
    reason: str


@dataclass
class RecoveredTarget:
    candidate_debug: CandidateDebug
    target_xy_mm: np.ndarray
    target_z_mm: float
    phi_deg: float
    object_height_mm: float
    safe_retry_required: bool
    reason: str


@dataclass
class PreparedPlaceMove:
    held_object: CandidateDebug
    target_xy_mm: np.ndarray
    target_phi_deg: float
    place_plan: Any
    settings: PlaceSequenceSettings
    object_height_mm: float
    destination_surface_z_mm: float
    release_max_open_deg: float


@dataclass
class PlaceabilityOverlayEntry:
    can_place: bool
    target_xy_mm: np.ndarray | None
    target_phi_deg: float | None
    clearance_mm: float | None
    reason: str


@dataclass
class OptimizedPlaceTarget:
    target_xy_mm: np.ndarray
    target_phi_deg: float
    footprint_clearance_mm: float
    aabb_clearance_mm: float
    fit_clearance_mm: float
    nudge_xy_mm: np.ndarray
    can_place: bool
    reason: str


def _best_candidate_config() -> BestCandidateConfig:
    return BestCandidateConfig(
        require_positive_platform_xy=BEST_REQUIRE_POSITIVE_PLATFORM_XY,
        platform_min_x_mm=BEST_PLATFORM_MIN_X_MM,
        platform_min_y_mm=BEST_PLATFORM_MIN_Y_MM,
        workspace_x_min_mm=BEST_WORKSPACE_X_MIN_MM,
        workspace_x_max_mm=BEST_WORKSPACE_X_MAX_MM,
        workspace_y_min_mm=BEST_WORKSPACE_Y_MIN_MM,
        workspace_y_max_mm=BEST_WORKSPACE_Y_MAX_MM,
        require_robotframe_centroid_xy_in_platform_bounds=BEST_REQUIRE_ROBOTFRAME_CENTROID_XY_IN_PLATFORM_BOUNDS,
        robotframe_centroid_x_min_mm=BEST_ROBOTFRAME_CENTROID_X_MIN_MM,
        robotframe_centroid_x_max_mm=BEST_ROBOTFRAME_CENTROID_X_MAX_MM,
        robotframe_centroid_y_min_mm=BEST_ROBOTFRAME_CENTROID_Y_MIN_MM,
        robotframe_centroid_y_max_mm=BEST_ROBOTFRAME_CENTROID_Y_MAX_MM,
        require_min_robotframe_centroid_z=BEST_REQUIRE_MIN_ROBOTFRAME_CENTROID_Z,
        min_robotframe_centroid_z_mm=BEST_MIN_ROBOTFRAME_CENTROID_Z_MM,
        use_robot_reach_check=BEST_USE_ROBOT_REACH_CHECK,
        robot_reach_margin_mm=BEST_ROBOT_REACH_MARGIN_MM,
        require_soft_pose_safe=BEST_REQUIRE_SOFT_POSE_SAFE,
        soft_pose_check_z_mm=BEST_SOFT_POSE_CHECK_Z_MM,
        center_gate_enabled=BEST_CENTER_GATE_ENABLED,
        max_image_center_norm_radius=BEST_MAX_IMAGE_CENTER_NORM_RADIUS,
        cluster_gate_enabled=BEST_CLUSTER_GATE_ENABLED,
        max_cluster_distance_mm=BEST_MAX_CLUSTER_DISTANCE_MM,
        reject_placed_overlap=BEST_REJECT_PLACED_OVERLAP,
        placed_overlap_margin_mm=BEST_PLACED_OVERLAP_MARGIN_MM,
        min_volume_mm3=BEST_MIN_VOLUME_MM3,
        max_volume_mm3=BEST_MAX_VOLUME_CM3 * 1000.0,
    )


def _candidate_class_name(dbg: CandidateDebug | None) -> str:
    if dbg is None:
        return "unknown"
    return str(getattr(getattr(dbg.candidate, "yolo", None), "class_name", "unknown"))


def _candidate_target_xy(dbg: CandidateDebug) -> np.ndarray:
    return np.asarray(getattr(dbg.candidate, "target_xy", [np.nan, np.nan]), dtype=np.float64).reshape(-1)[:2]


def _candidate_robot_xy(dbg: CandidateDebug) -> np.ndarray:
    c = dbg.candidate
    for attr in ("object_robot_xyz_corrected", "object_robot_xyz_raw"):
        value = getattr(c, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return arr[:2].astype(np.float64)
    return _candidate_target_xy(dbg)


def _candidate_surface_z_mm(dbg: CandidateDebug) -> float:
    c = dbg.candidate
    z_debug = getattr(c, "z_debug", None)
    for obj, attr in (
        (z_debug, "robust_top_z_mm"),
        (z_debug, "z_p95_mm"),
    ):
        try:
            value = float(getattr(obj, attr, np.nan))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass

    for attr in ("object_robot_xyz_corrected", "object_robot_xyz_raw", "target_cam_xyz"):
        value = getattr(c, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 3 and np.isfinite(arr[2]):
            return float(arr[2])
    return 0.0


def _candidate_height_mm(dbg: CandidateDebug) -> float:
    c = dbg.candidate
    z_debug = getattr(c, "z_debug", None)
    try:
        h = float(getattr(z_debug, "object_height_mm", np.nan))
        if np.isfinite(h):
            return max(0.0, h)
    except (TypeError, ValueError):
        pass
    h_cm = getattr(c, "pointcloud_height_cm", None)
    try:
        h = float(h_cm) * 10.0
        if np.isfinite(h):
            return max(0.0, h)
    except (TypeError, ValueError):
        pass
    return 0.0


def _candidate_phi_deg(robot, dbg: CandidateDebug) -> float:
    value = getattr(dbg.candidate, "pick_phi_deg", None)
    try:
        phi = float(value)
        if np.isfinite(phi):
            return phi
    except (TypeError, ValueError):
        pass
    try:
        return float(robot.fk()[3])
    except Exception:
        return 0.0


def _aabb_from_object_candidate_quiet(candidate, default_label: str):
    # aabb_from_object_candidate prints verbose diagnostics; suppress in tight loops.
    with io.StringIO() as _sink, redirect_stdout(_sink):
        return aabb_from_object_candidate(candidate, default_label=default_label)


def _bbox_area_px(bbox_xyxy: tuple[float, float, float, float] | None) -> float | None:
    if bbox_xyxy is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return area if np.isfinite(area) and area > 0.0 else None


def _candidate_comparable_bbox(dbg: CandidateDebug) -> tuple[float, float, float, float] | None:
    c = dbg.candidate
    if MISS_CHECK_CAMERA == "overhead":
        bbox = getattr(c, "overhead_bbox_px", None)
        if bbox is not None:
            return tuple(float(v) for v in bbox)
        print("[MISS CHECK WARN] original overhead bbox unavailable; falling back to stereo detection bbox for IoU audit.")
    det = getattr(dbg, "best_detection", None) or getattr(c, "yolo", None)
    bbox = getattr(det, "bbox", None)
    if bbox is None:
        return None
    return tuple(float(v) for v in bbox)


def _candidate_mask_area_px(dbg: CandidateDebug) -> int | None:
    det = getattr(dbg, "best_detection", None) or getattr(dbg.candidate, "yolo", None)
    value = getattr(det, "mask_area", None)
    try:
        area = int(value)
        return area if area > 0 else None
    except (TypeError, ValueError):
        return None


def _detection_area_px(det: YOLODetection, *, prefer_bbox: bool) -> float | None:
    if not prefer_bbox:
        try:
            area = float(getattr(det, "mask_area", np.nan))
            if np.isfinite(area) and area > 0.0:
                return area
        except (TypeError, ValueError):
            pass
    return _bbox_area_px(tuple(float(v) for v in det.bbox))


def _bbox_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = _bbox_area_px(a)
    area_b = _bbox_area_px(b)
    if area_a is None or area_b is None:
        return None
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return None
    return float(inter / denom)


def _area_score(area_ratio: float | None) -> float | None:
    if area_ratio is None or not np.isfinite(area_ratio) or area_ratio <= 0.0:
        return None
    lo = float(MISS_MATCH_AREA_RATIO_MIN)
    hi = float(MISS_MATCH_AREA_RATIO_MAX)
    if lo <= area_ratio <= hi:
        return 1.0
    if area_ratio < lo:
        return float(np.clip(area_ratio / max(lo, 1e-6), 0.0, 1.0))
    return float(np.clip(hi / area_ratio, 0.0, 1.0))


def _project_miss_detection_to_robot_xy(
    det: YOLODetection,
    attempt: PickAttemptRecord,
    bundle: dict,
) -> np.ndarray | None:
    if MISS_CHECK_CAMERA != "overhead":
        return None
    try:
        xy, lookup_z, clamped, support_dist, _support_idx = project_overhead_centroid_to_robot_xy(
            det.centroid_px,
            _candidate_surface_z_mm(attempt.candidate_debug),
            bundle,
        )
        if clamped:
            print(f"[MISS CHECK WARN] overhead homography lookup Z clamped to {lookup_z:.1f} mm.")
        if np.isfinite(support_dist) and support_dist > 80.0:
            print(f"[MISS CHECK WARN] overhead XY support distance is high ({support_dist:.1f} mm).")
        return np.asarray(xy, dtype=np.float64).reshape(2)
    except Exception as exc:
        print(f"[MISS CHECK WARN] robot-frame XY unavailable from YOLO-only overhead frame: {exc}")
        return None


def _score_miss_detection(
    det: YOLODetection,
    attempt: PickAttemptRecord,
    bundle: dict,
) -> tuple[float, np.ndarray | None, float | None, float | None, float | None, bool, str]:
    same_class = str(det.class_name) == str(attempt.class_name)
    robot_xy = _project_miss_detection_to_robot_xy(det, attempt, bundle)
    robot_dist = None
    robot_dist_score = None
    if robot_xy is not None and np.all(np.isfinite(robot_xy)):
        robot_dist = float(np.linalg.norm(robot_xy - attempt.original_target_xy_mm))
        robot_dist_score = float(np.clip(1.0 - robot_dist / max(1e-6, float(MISS_MATCH_MAX_ROBOT_DIST_MM)), 0.0, 1.0))

    det_bbox = tuple(float(v) for v in det.bbox)
    iou = _bbox_iou(attempt.original_bbox_xyxy, det_bbox)
    iou_score = None if iou is None else float(np.clip(iou / max(1e-6, float(MISS_MATCH_MIN_IOU)), 0.0, 1.0))

    prefer_bbox = MISS_CHECK_CAMERA == "overhead" and attempt.original_bbox_area_px is not None
    det_area = _detection_area_px(det, prefer_bbox=prefer_bbox)
    original_area = attempt.original_bbox_area_px if prefer_bbox else attempt.original_mask_area_px
    if original_area is None or original_area <= 0.0:
        original_area = attempt.original_bbox_area_px
    area_ratio = None
    if det_area is not None and original_area is not None and original_area > 0.0:
        area_ratio = float(det_area / float(original_area))
    area_score = _area_score(area_ratio)

    weighted_parts: list[tuple[float, float]] = [(float(MISS_CLASS_WEIGHT), 1.0 if same_class else 0.0)]
    missing: list[str] = []
    if robot_dist_score is None:
        missing.append("robot_xy")
    else:
        weighted_parts.append((float(MISS_ROBOT_DIST_WEIGHT), robot_dist_score))
    if iou_score is None:
        missing.append("iou")
    else:
        weighted_parts.append((float(MISS_IOU_WEIGHT), iou_score))
    if area_score is None:
        missing.append("area")
    else:
        weighted_parts.append((float(MISS_AREA_WEIGHT), area_score))

    total_weight = sum(w for w, _ in weighted_parts)
    score = 0.0 if total_weight <= 0.0 else sum(w * s for w, s in weighted_parts) / total_weight
    reason = (
        f"class={same_class} dist={_fmt_optional(robot_dist)} iou={_fmt_optional(iou)} "
        f"area_ratio={_fmt_optional(area_ratio)} missing={','.join(missing) if missing else 'none'}"
    )
    return float(score), robot_xy, robot_dist, iou, area_ratio, same_class, reason


def _fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def make_pick_attempt_record(
    *,
    object_i: int,
    cand_dbg: CandidateDebug,
    robot,
    attempt_number: int,
) -> PickAttemptRecord:
    c = cand_dbg.candidate
    bbox = _candidate_comparable_bbox(cand_dbg)
    record = PickAttemptRecord(
        object_i=int(object_i),
        candidate_debug=cand_dbg,
        class_name=_candidate_class_name(cand_dbg),
        original_robot_xy_mm=_candidate_robot_xy(cand_dbg),
        original_target_xy_mm=_candidate_target_xy(cand_dbg),
        original_bbox_xyxy=bbox,
        original_mask_area_px=_candidate_mask_area_px(cand_dbg),
        original_bbox_area_px=_bbox_area_px(bbox),
        original_object_height_mm=_candidate_height_mm(cand_dbg),
        original_phi_deg=_candidate_phi_deg(robot, cand_dbg),
        attempt_number=int(attempt_number),
        timestamp_s=time.perf_counter(),
    )
    print("[PICK ATTEMPT AUDIT]")
    print(
        f"object={record.object_i} attempt={record.attempt_number} "
        f"candidate_index={getattr(c, 'index', '?')} class={record.class_name}"
    )
    print(
        f"target_xy=({record.original_target_xy_mm[0]:.1f},{record.original_target_xy_mm[1]:.1f}) "
        f"robot_xy=({record.original_robot_xy_mm[0]:.1f},{record.original_robot_xy_mm[1]:.1f}) "
        f"height={record.original_object_height_mm:.1f} phi={record.original_phi_deg:.1f}"
    )
    print(
        f"bbox={record.original_bbox_xyxy} "
        f"bbox_area={_fmt_optional(record.original_bbox_area_px, 1)} "
        f"mask_area={record.original_mask_area_px}"
    )
    return record


def _resolve_held_object_height_and_uncertainty(held_object: CandidateDebug | None) -> tuple[float, float, list[str]]:
    if held_object is None:
        return 0.0, 0.0, ["no_held_object"]

    warnings: list[str] = []
    c = held_object.candidate
    z_debug = getattr(c, "z_debug", None)

    if z_debug is not None:
        return (
            max(0.0, float(z_debug.object_height_mm)),
            max(0.0, float(z_debug.uncertainty_clearance_mm)),
            list(getattr(z_debug, "warnings", []) or []),
        )

    h_cm = getattr(c, "pointcloud_height_cm", None)
    if h_cm is not None and np.isfinite(float(h_cm)):
        warnings.append("using_pointcloud_height_cm_fallback")
        return max(0.0, float(h_cm) * 10.0), 0.0, warnings

    warnings.append("no_object_height_available")
    return 0.0, 0.0, warnings


def _held_pick_grasp_z_mm(held_object: CandidateDebug | None) -> float | None:
    if held_object is None:
        return None
    c = held_object.candidate
    for attr in ("z_empirical_robot_frame_mm", "grasp_robot_z"):
        try:
            value = float(getattr(c, attr, np.nan))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    z_debug = getattr(c, "z_debug", None)
    try:
        value = float(getattr(z_debug, "grasp_z_mm", np.nan))
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return None


def _held_bottom_z_at_pick_mm(held_object: CandidateDebug | None) -> float | None:
    if held_object is None:
        return None
    c = held_object.candidate
    z_debug = getattr(c, "z_debug", None)
    try:
        value = float(getattr(z_debug, "robust_bottom_z_mm", np.nan))
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    height = _candidate_height_mm(held_object)
    top = _candidate_surface_z_mm(held_object)
    if np.isfinite(height) and np.isfinite(top):
        return float(top - height)
    return None


def _held_bottom_hang_below_gripper_mm(held_object: CandidateDebug | None, object_height_mm: float) -> tuple[float, list[str]]:
    warnings: list[str] = []
    grasp_z = _held_pick_grasp_z_mm(held_object)
    bottom_z = _held_bottom_z_at_pick_mm(held_object)
    height = max(0.0, float(object_height_mm))
    if grasp_z is not None and bottom_z is not None:
        gripper_offset = float(getattr(_pick_one_mod, "GRIPPER_OFFSET_MM", DEFAULT_Z_SAFETY.GRIPPER_OFFSET_MM))
        object_contact_z = float(grasp_z) - gripper_offset
        hang = max(0.0, object_contact_z - float(bottom_z))
        hang = min(max(0.0, height), hang)
        if hang > height * 2.0 + 50.0:
            warnings.append("bottom_hang_large_check_grasp_z_or_bottom_z")
        warnings.append("bottom_hang_uses_grasp_z_minus_gripper_offset")
        return hang, warnings
    warnings.append("bottom_hang_fallback_to_object_height")
    return height, warnings


def _estimate_existing_bag_stack_top_mm(destination_surface_z_mm: float) -> float:
    if not PLACE_NEGATIVE_BIN_USE_EXISTING_STACK:
        return 0.0
    # destination_surface_z_mm is height above bag floor (not robot_z).
    # robot_z of stack top = PLACE_NEGATIVE_BIN_PLATFORM_Z_MM + destination.
    return max(0.0, float(destination_surface_z_mm))


def _apply_negative_bin_place_policy(place_plan, held_object: CandidateDebug, object_height_mm: float, destination_surface_z_mm: float):
    mode = str(PLACE_Z_POLICY_MODE).strip().lower()
    if mode == "shared":
        return place_plan
    if mode not in {"negative_bin_hang", "negative_bin_simple"}:
        print(f"[PLACE AUTO Z POLICY WARN] unknown PLACE_Z_POLICY_MODE={PLACE_Z_POLICY_MODE!r}; using shared plan.")
        return place_plan

    stack_top = _estimate_existing_bag_stack_top_mm(destination_surface_z_mm)
    height = max(0.0, float(object_height_mm))
    hang, hang_warnings = _held_bottom_hang_below_gripper_mm(held_object, height)

    simple_release_z = float(PLACE_NEGATIVE_BIN_PLATFORM_Z_MM) + stack_top + height
    hang_release_z = float(PLACE_NEGATIVE_BIN_PLATFORM_Z_MM) + stack_top + hang
    if mode == "negative_bin_hang":
        weighted = (
            float(PLACE_NEGATIVE_BIN_HANG_WEIGHT) * hang_release_z
            + float(PLACE_NEGATIVE_BIN_SIMPLE_WEIGHT) * simple_release_z
        )
        raw_release_z = weighted
    else:
        raw_release_z = simple_release_z

    if PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING:
        raw_release_z += float(PLACE_RELEASE_GAP_MM) + float(place_plan.place_z_safety_padding_mm)
    raw_release_z += float(PLACE_NEGATIVE_BIN_CLEARANCE_MM)
    final_release_z = max(float(PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM), raw_release_z)
    final_release_z = max(float(DEFAULT_Z_SAFETY.MIN_PLACE_Z_MM), final_release_z)
    final_release_z = min(float(Z_MAX_MM), final_release_z)

    warnings = list(place_plan.warnings) + hang_warnings + [f"experimental_{mode}"]
    if final_release_z != raw_release_z:
        warnings.append("negative_bin_release_z_clamped")

    print("[PLACE AUTO NEGATIVE BIN Z PLAN]")
    print(f"mode = {mode}")
    print(f"bin_platform_z_mm = {PLACE_NEGATIVE_BIN_PLATFORM_Z_MM:.1f}")
    print(f"existing_stack_top_mm = {stack_top:.1f}")
    print(f"held_object_height_mm = {height:.1f}")
    print(f"bottom_hang_below_gripper_mm = {hang:.1f}")
    print(f"simple_release_z_mm = {simple_release_z:.1f}")
    print(f"hang_release_z_mm = {hang_release_z:.1f}")
    print(f"include_release_gap_padding = {PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING}")
    print(f"raw_release_z_with_gap_padding = {raw_release_z:.1f}")
    print(f"final_release_z_mm = {final_release_z:.1f}")
    print(f"warnings = {warnings}")

    place_plan.stack_top_mm = float(stack_top)
    place_plan.raw_item_height_mm = float(height)
    place_plan.object_height_mm = float(height)
    place_plan.place_z_raw_mm = float(raw_release_z)
    place_plan.final_release_z_mm = float(final_release_z)
    place_plan.destination_floor_or_surface_z_mm = float(PLACE_NEGATIVE_BIN_PLATFORM_Z_MM)
    place_plan.destination_surface_z_mm = float(PLACE_NEGATIVE_BIN_PLATFORM_Z_MM)
    place_plan.warnings = warnings
    place_plan.valid = bool(final_release_z <= float(Z_MAX_MM) + 1e-6)
    place_plan.debug = (
        f"{mode}: platform({PLACE_NEGATIVE_BIN_PLATFORM_Z_MM:.1f}) + stack({stack_top:.1f}) "
        f"+ hang({hang:.1f})/height({height:.1f}) + gap/padding -> {final_release_z:.1f}"
    )
    return place_plan


def _execute_place_at_target(
    robot,
    held_object: CandidateDebug,
    *,
    target_xy_mm: np.ndarray,
    target_phi_deg: float,
    destination_surface_z_mm: float,
    on_start_place_descent: Callable[[], None] | None = None,
) -> bool:
    prepared = _prepare_place_at_target(
        robot,
        held_object,
        target_xy_mm=target_xy_mm,
        target_phi_deg=target_phi_deg,
        destination_surface_z_mm=destination_surface_z_mm,
    )
    if prepared is None:
        return False
    return _finish_place_from_hover(robot, prepared, on_start_place_descent=on_start_place_descent)


def _prepare_place_at_target(
    robot,
    held_object: CandidateDebug,
    *,
    target_xy_mm: np.ndarray,
    target_phi_deg: float,
    destination_surface_z_mm: float,
) -> PreparedPlaceMove | None:
    object_height_mm, object_uncertainty_clearance_mm, object_warnings = _resolve_held_object_height_and_uncertainty(held_object)
    place_plan = compute_place_z_plan(
        destination_surface_z_mm=destination_surface_z_mm,
        object_height_mm=object_height_mm,
        z_max_mm=Z_MAX_MM,
        release_gap_mm=PLACE_RELEASE_GAP_MM,
        object_uncertainty_clearance_mm=object_uncertainty_clearance_mm,
        place_uncertainty_gain=PLACE_Z_UNCERTAINTY_GAIN,
        place_uncertainty_clearance_max_mm=PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
        config=DEFAULT_Z_SAFETY,
    )
    place_plan = _apply_negative_bin_place_policy(place_plan, held_object, object_height_mm, destination_surface_z_mm)

    place_z = float(place_plan.final_release_z_mm)
    place_source = f"place_z_policy:{PLACE_Z_POLICY_MODE}"
    if not USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT:
        print("[PLACE AUTO] dynamic place correction disabled; shared Z safety policy still controls release height.")

    x = float(target_xy_mm[0])
    y = float(target_xy_mm[1])
    phi = float(target_phi_deg)
    travel_z = float(place_plan.approach_z_mm)

    initial_pick_open_deg = getattr(held_object.candidate, "initial_servo_angle_deg", None)
    try:
        release_max_open_deg = float(initial_pick_open_deg)
    except (TypeError, ValueError):
        release_max_open_deg = float(CLAW_OPEN_DEG)
    if not np.isfinite(release_max_open_deg):
        release_max_open_deg = float(CLAW_OPEN_DEG)
    release_max_open_deg = float(np.clip(release_max_open_deg, 0.0, 180.0))

    for label, z in (("travel", travel_z), ("place", place_z), ("retract", float(place_plan.retract_z_mm))):
        reason = validate_z_command(z, f"[PLACE AUTO] {label}", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[PLACE AUTO] ABORT: {reason}")
            return False

    print("[PLACE AUTO Z PLAN]")
    print(f"destination_surface_z_mm = {destination_surface_z_mm:.3f}")
    print(f"object_height_mm = {object_height_mm:.3f}")
    print(f"release_gap_mm = {PLACE_RELEASE_GAP_MM:.3f}")
    print(f"object_uncertainty_clearance_mm = {object_uncertainty_clearance_mm:.3f}")
    print(f"place_uncertainty_gain = {PLACE_Z_UNCERTAINTY_GAIN:.3f}")
    print(f"place_uncertainty_clearance_mm = {place_plan.place_uncertainty_clearance_mm:.3f}")
    print(f"raw_item_height_mm = {place_plan.raw_item_height_mm:.3f}")
    print(f"clamped_item_height_mm = {place_plan.object_height_mm:.3f}")
    print(f"safety_padding_mm = {place_plan.place_z_safety_padding_mm:.3f}")
    print(f"raw_place_z_mm = {place_plan.place_z_raw_mm:.3f}")
    print(f"final_release_z_mm = {place_plan.final_release_z_mm:.3f}")
    print(f"approach/retract_z_mm = {place_plan.approach_z_mm:.3f}/{place_plan.retract_z_mm:.3f}")
    print(f"warnings = {place_plan.warnings + object_warnings}")
    print(f"target_xy_mm = ({x:.1f}, {y:.1f}) target_phi_deg = {phi:.1f} source = {place_source}")
    print(
        f"release_mode = {'dynamic_release_DR' if USE_DYNAMIC_RELEASE_FOR_PLACE else 'fixed_servo_open'} "
        f"fallback_open_deg = {PLACE_CLAW_OPEN_DEG:.2f}"
    )
    print(f"dynamic_release_max_open_deg = {release_max_open_deg:.2f} (from pick initial angle)")
    print("[PLACE AUTO] motion is split: hover first, then missed-pick decision, then release descent if clear.")

    if not _confirm("[PLACE AUTO] Real place-hover motion will execute. Keep clear of the robot.", REQUIRE_CONFIRM_BEFORE_REAL_MOTION):
        print("[PLACE AUTO] canceled by user")
        return None

    settings = PlaceSequenceSettings(
        release_servo_deg=PLACE_CLAW_OPEN_DEG,
        coarse_move_time_s=COARSE_MOVE_TIME_S,
        xy_move_time_s=XY_MOVE_TIME_S,
        z_move_time_s=PLACE_Z_MOVE_TIME_S,
        use_dynamic_release=USE_DYNAMIC_RELEASE_FOR_PLACE,
        dynamic_release_timeout_s=DYNAMIC_RELEASE_TIMEOUT_S,
        dynamic_release_max_open_deg=release_max_open_deg,
    )

    checker = None
    xy_time = settings.coarse_move_time_s if settings.xy_move_time_s is None else settings.xy_move_time_s

    print("[PLACE AUTO] WARNING: moving to place hover now.")
    if not _sequence_move_checked(
        robot,
        "[PLACE AUTO] raise",
        z_mm=travel_z,
        move_time_s=settings.coarse_move_time_s,
        check_pose_safe_fn=checker,
    ):
        return None
    if not _sequence_move_checked(
        robot,
        "[PLACE AUTO] XY+phi hover",
        x_mm=x,
        y_mm=y,
        z_mm=travel_z,
        phi_deg=phi,
        move_time_s=xy_time,
        check_pose_safe_fn=checker,
    ):
        return None

    print("[PLACE AUTO] at place XY hover; release descent is pending.")
    return PreparedPlaceMove(
        held_object=held_object,
        target_xy_mm=np.array([x, y], dtype=np.float64),
        target_phi_deg=phi,
        place_plan=place_plan,
        settings=settings,
        object_height_mm=object_height_mm,
        destination_surface_z_mm=destination_surface_z_mm,
        release_max_open_deg=release_max_open_deg,
    )


def _finish_place_from_hover(
    robot,
    prepared: PreparedPlaceMove,
    *,
    on_start_place_descent: Callable[[], None] | None = None,
) -> bool:
    place_z = float(prepared.place_plan.final_release_z_mm)
    retract_z = float(prepared.place_plan.retract_z_mm)
    settings = prepared.settings

    for label, z in (("place", place_z), ("retract", retract_z)):
        reason = validate_z_command(z, f"[PLACE AUTO] {label}", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[PLACE AUTO] ABORT: {reason}")
            return False

    if on_start_place_descent is not None:
        print("[PLACE AUTO] starting place-descent callback")
        try:
            on_start_place_descent()
        except Exception as exc:
            print(f"[PLACE AUTO] WARN: place-descent callback failed: {exc}")

    print("[PLACE AUTO] WARNING: descending to release height now.")
    if not _sequence_move_checked(
        robot,
        "[PLACE AUTO] descend",
        z_mm=place_z,
        move_time_s=settings.z_move_time_s,
    ):
        return False

    release_ok = True
    if settings.use_dynamic_release:
        if hasattr(robot, "dynamic_release"):
            print(
                "[PLACE AUTO] triggering dynamic release (DR) "
                f"max_open_deg={settings.dynamic_release_max_open_deg}"
            )
            release_result = robot.dynamic_release(
                max_open_angle_deg=settings.dynamic_release_max_open_deg,
                timeout_s=float(settings.dynamic_release_timeout_s),
            )
            print(
                "[PLACE AUTO] dynamic release result: "
                f"ok={release_result.ok} "
                f"servo_release_empirical_deg={release_result.servo_release_empirical_deg} "
                f"error={release_result.error}"
            )
            release_ok = bool(release_result.ok)
        else:
            print("[PLACE AUTO] ABORT: robot.dynamic_release unavailable")
            release_ok = False
    else:
        print(f"[PLACE AUTO] opening claw servo={settings.release_servo_deg:.2f}")
        if not _sequence_command_servo_angle(robot, settings.release_servo_deg):
            print("[PLACE AUTO] WARN: failed to command release servo angle")

    if not release_ok:
        print("[PLACE AUTO] ABORT: release command failed")
        return False

    if settings.claw_settle_s > 0.0:
        time.sleep(float(settings.claw_settle_s))

    if not _sequence_move_checked(
        robot,
        "[PLACE AUTO] retract",
        z_mm=retract_z,
        move_time_s=settings.coarse_move_time_s,
    ):
        return False

    print("[PLACE AUTO] OK - item released and robot retracted.")
    return True


def _placed_occupancy_from_plan(center_xyz_mm: np.ndarray, size_xyz_mm: np.ndarray, label: str):
    raw = make_aabb_from_center_size(center_xyz_mm=center_xyz_mm, size_xyz_mm=size_xyz_mm, label=label)
    return pad_aabb(raw, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)


def _held_initial_servo_deg(held_object: CandidateDebug | None) -> float:
    if held_object is not None:
        try:
            value = float(getattr(held_object.candidate, "initial_servo_angle_deg", np.nan))
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return float(PLACE_GRIPPER_FOOTPRINT_DEFAULT_SERVO_DEG)


def _gripper_footprint_size_mm(servo_deg: float) -> tuple[float, float]:
    angle_rad = np.deg2rad(float(np.clip(servo_deg, 0.0, 180.0)))
    length = abs(float(PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM) * float(np.sin(angle_rad)))
    length += 2.0 * float(PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM)
    width = float(PLACE_GRIPPER_FOOTPRINT_WIDTH_MM) + 2.0 * float(PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM)
    return max(1.0, length), max(1.0, width)


def _oriented_rect_corners_xy(center_xy_mm: np.ndarray, phi_deg: float, length_mm: float, width_mm: float) -> np.ndarray:
    c = np.asarray(center_xy_mm, dtype=np.float64).reshape(2)
    phi = np.deg2rad(float(phi_deg))
    u = np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)
    v = np.array([-np.sin(phi), np.cos(phi)], dtype=np.float64)
    hl = 0.5 * float(length_mm)
    hw = 0.5 * float(width_mm)
    return np.vstack([
        c - hl * u - hw * v,
        c + hl * u - hw * v,
        c + hl * u + hw * v,
        c - hl * u + hw * v,
    ])


def _normalize_phi_deg(phi_deg: float) -> float:
    phi = float(phi_deg)
    while phi <= -180.0:
        phi += 360.0
    while phi > 180.0:
        phi -= 360.0
    return phi


def _bag_bounds_xy(surface_zone: dict) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    half_w = 0.5 * float(surface_zone.get("width_mm", 120.0))
    half_d = 0.5 * float(surface_zone.get("depth_mm", 120.0))
    min_xy = center - np.array([half_w, half_d], dtype=np.float64)
    max_xy = center + np.array([half_w, half_d], dtype=np.float64)
    return min_xy, max_xy


def _footprint_signed_edge_clearance_mm(
    center_xy_mm: np.ndarray,
    phi_deg: float,
    length_mm: float,
    width_mm: float,
    *,
    surface_zone: dict,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    min_xy, max_xy = _bag_bounds_xy(surface_zone)
    corners = _oriented_rect_corners_xy(center_xy_mm, phi_deg, length_mm, width_mm)
    signed_clearances = np.column_stack((
        corners[:, 0] - min_xy[0],
        max_xy[0] - corners[:, 0],
        corners[:, 1] - min_xy[1],
        max_xy[1] - corners[:, 1],
    ))
    return float(np.min(signed_clearances)), corners, min_xy, max_xy


def _aabb_signed_edge_clearance_mm(
    center_xy_mm: np.ndarray,
    size_xy_mm: np.ndarray,
    *,
    surface_zone: dict,
) -> float:
    min_xy, max_xy = _bag_bounds_xy(surface_zone)
    center = np.asarray(center_xy_mm, dtype=np.float64).reshape(2)
    size_xy = np.asarray(size_xy_mm, dtype=np.float64).reshape(2)
    half = 0.5 * size_xy
    signed_clearances = np.array([
        center[0] - half[0] - min_xy[0],
        max_xy[0] - (center[0] + half[0]),
        center[1] - half[1] - min_xy[1],
        max_xy[1] - (center[1] + half[1]),
    ], dtype=np.float64)
    return float(np.min(signed_clearances))


def _evaluate_place_phi_for_edge_clearance(
    held_object: CandidateDebug | None,
    *,
    target_xy_mm: np.ndarray,
    base_phi_deg: float,
    surface_zone: dict,
    label: str,
    verbose: bool,
) -> tuple[float, float, bool]:
    base_phi = _normalize_phi_deg(base_phi_deg)
    if not PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE or not PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG:
        clearance_mm, _corners, _min_xy, _max_xy = _footprint_signed_edge_clearance_mm(
            np.asarray(target_xy_mm, dtype=np.float64).reshape(2),
            base_phi,
            *_gripper_footprint_size_mm(_held_initial_servo_deg(held_object)),
            surface_zone=surface_zone,
        )
        return base_phi, clearance_mm, bool(clearance_mm >= 0.0)

    servo_deg = _held_initial_servo_deg(held_object)
    length_mm, width_mm = _gripper_footprint_size_mm(servo_deg)
    offsets = [float(v) for v in PLACE_ROTATION_CANDIDATE_OFFSETS_DEG]
    candidate_phis: list[float] = []
    for offset_deg in offsets if offsets else [0.0]:
        phi = _normalize_phi_deg(base_phi + offset_deg)
        if any(abs(phi - existing) <= 1e-6 for existing in candidate_phis):
            continue
        candidate_phis.append(phi)

    best_phi = base_phi
    best_clearance = -float("inf")
    best_inside = False
    if verbose:
        print("[PLACE ROTATION SEARCH]")
        print(
            f"label={label} base_phi={base_phi:.1f} servo_deg={servo_deg:.1f} "
            f"footprint_length={length_mm:.1f} width={width_mm:.1f} candidates={candidate_phis}"
        )
    for phi in candidate_phis:
        clearance_mm, corners, min_xy, max_xy = _footprint_signed_edge_clearance_mm(
            np.asarray(target_xy_mm, dtype=np.float64).reshape(2),
            phi,
            length_mm,
            width_mm,
            surface_zone=surface_zone,
        )
        inside = bool(clearance_mm >= 0.0)
        if verbose:
            print(
                f"  phi={phi:.1f} clearance_mm={clearance_mm:.1f} inside={inside} "
                f"bag_x=[{min_xy[0]:.1f},{max_xy[0]:.1f}] bag_y=[{min_xy[1]:.1f},{max_xy[1]:.1f}]"
            )
            for idx, corner in enumerate(corners, start=1):
                print(f"    corner{idx}: x={corner[0]:.1f} y={corner[1]:.1f}")
        if inside and not best_inside:
            best_phi = phi
            best_clearance = clearance_mm
            best_inside = True
            continue
        if inside == best_inside and clearance_mm > best_clearance + 1e-6:
            best_phi = phi
            best_clearance = clearance_mm

    if verbose:
        print(
            f"[PLACE ROTATION SEARCH] selected_phi={best_phi:.1f} "
            f"clearance_mm={best_clearance:.1f} inside={best_inside}"
        )
    return best_phi, best_clearance, best_inside


def _xy_nudge_candidates() -> list[np.ndarray]:
    if not PLACE_XY_NUDGE_ENABLED:
        return [np.array([0.0, 0.0], dtype=np.float64)]

    step = max(0.0, float(PLACE_XY_NUDGE_STEP_MM))
    max_nudge = max(0.0, float(PLACE_XY_NUDGE_MAX_MM))
    if step <= 0.0 or max_nudge <= 0.0:
        return [np.array([0.0, 0.0], dtype=np.float64)]

    offsets = [0.0]
    current = step
    while current <= max_nudge + 1e-6:
        offsets.append(float(current))
        current += step

    candidates: list[np.ndarray] = [np.array([0.0, 0.0], dtype=np.float64)]
    for dx in offsets:
        for dy in offsets:
            for sign_x in (-1.0, 1.0):
                for sign_y in (-1.0, 1.0):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    candidates.append(np.array([sign_x * dx, sign_y * dy], dtype=np.float64))

    unique: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for offset in candidates:
        key = (round(float(offset[0]), 6), round(float(offset[1]), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(offset)
    unique.sort(key=lambda arr: (float(arr[0] ** 2 + arr[1] ** 2), float(arr[0]), float(arr[1])))
    return unique


def _evaluate_slot_fit(
    held_object: CandidateDebug | None,
    *,
    target_xy_mm: np.ndarray,
    base_phi_deg: float,
    surface_zone: dict,
    label: str,
    verbose: bool,
) -> OptimizedPlaceTarget:
    base_xy = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
    base_phi = _normalize_phi_deg(base_phi_deg)
    servo_deg = _held_initial_servo_deg(held_object)
    footprint_length_mm, footprint_width_mm = _gripper_footprint_size_mm(servo_deg)
    footprint_size_xy_mm = np.array([footprint_length_mm, footprint_width_mm], dtype=np.float64)
    try:
        candidate_box = _aabb_from_object_candidate_quiet(held_object.candidate, default_label="candidate_fit") if held_object is not None else None
        candidate_size_xy_mm = (
            np.asarray(candidate_box.size_xyz_mm, dtype=np.float64).reshape(3)[:2]
            if candidate_box is not None
            else footprint_size_xy_mm.copy()
        )
    except Exception:
        candidate_box = None
        candidate_size_xy_mm = footprint_size_xy_mm.copy()

    if verbose:
        print("[PLACE FIT SEARCH]")
        print(
            f"label={label} base_xy=({base_xy[0]:.1f},{base_xy[1]:.1f}) base_phi={base_phi:.1f} "
            f"servo_deg={servo_deg:.1f} footprint={footprint_length_mm:.1f}x{footprint_width_mm:.1f} "
            f"candidate_aabb_xy={candidate_size_xy_mm[0]:.1f}x{candidate_size_xy_mm[1]:.1f}"
        )

    best: OptimizedPlaceTarget | None = None
    for offset in _xy_nudge_candidates():
        nudged_xy = base_xy + offset
        for phi in (base_phi,) if not PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE else [base_phi + float(v) for v in PLACE_ROTATION_CANDIDATE_OFFSETS_DEG]:
            phi_norm = _normalize_phi_deg(phi)
            footprint_clearance_mm, _corners, _min_xy, _max_xy = _footprint_signed_edge_clearance_mm(
                nudged_xy,
                phi_norm,
                footprint_length_mm,
                footprint_width_mm,
                surface_zone=surface_zone,
            )
            aabb_clearance_mm = _aabb_signed_edge_clearance_mm(
                nudged_xy,
                candidate_size_xy_mm,
                surface_zone=surface_zone,
            )
            fit_clearance_mm = min(float(footprint_clearance_mm), float(aabb_clearance_mm))
            can_place = bool(fit_clearance_mm >= 0.0)
            candidate = OptimizedPlaceTarget(
                target_xy_mm=nudged_xy.copy(),
                target_phi_deg=float(phi_norm),
                footprint_clearance_mm=float(footprint_clearance_mm),
                aabb_clearance_mm=float(aabb_clearance_mm),
                fit_clearance_mm=float(fit_clearance_mm),
                nudge_xy_mm=offset.copy(),
                can_place=can_place,
                reason="ok" if can_place else "fit_or_footprint_outside_bag",
            )
            if verbose:
                print(
                    f"  phi={candidate.target_phi_deg:.1f} nudge=({candidate.nudge_xy_mm[0]:.1f},{candidate.nudge_xy_mm[1]:.1f}) "
                    f"footprint_clearance={candidate.footprint_clearance_mm:.1f} aabb_clearance={candidate.aabb_clearance_mm:.1f} "
                    f"fit_clearance={candidate.fit_clearance_mm:.1f} can_place={candidate.can_place}"
                )

            if best is None:
                best = candidate
                continue

            if candidate.can_place != best.can_place:
                if candidate.can_place:
                    best = candidate
                continue

            if candidate.fit_clearance_mm > best.fit_clearance_mm + 1e-6:
                best = candidate
                continue

            if abs(candidate.fit_clearance_mm - best.fit_clearance_mm) <= 1e-6:
                candidate_shift = float(np.linalg.norm(candidate.nudge_xy_mm))
                best_shift = float(np.linalg.norm(best.nudge_xy_mm))
                if candidate_shift < best_shift - 1e-6:
                    best = candidate
                    continue
                if abs(candidate_shift - best_shift) <= 1e-6 and float(candidate.footprint_clearance_mm) > float(best.footprint_clearance_mm) + 1e-6:
                    best = candidate

    if best is None:
        best = OptimizedPlaceTarget(
            target_xy_mm=base_xy.copy(),
            target_phi_deg=float(base_phi),
            footprint_clearance_mm=float("nan"),
            aabb_clearance_mm=float("nan"),
            fit_clearance_mm=float("nan"),
            nudge_xy_mm=np.array([0.0, 0.0], dtype=np.float64),
            can_place=False,
            reason="no_candidates",
        )

    if verbose:
        print(
            f"[PLACE FIT SEARCH] selected_xy=({best.target_xy_mm[0]:.1f},{best.target_xy_mm[1]:.1f}) "
            f"selected_phi={best.target_phi_deg:.1f} nudge=({best.nudge_xy_mm[0]:.1f},{best.nudge_xy_mm[1]:.1f}) "
            f"fit_clearance={best.fit_clearance_mm:.1f} can_place={best.can_place}"
        )
    return best


def _choose_place_phi_for_edge_clearance(
    held_object: CandidateDebug | None,
    *,
    target_xy_mm: np.ndarray,
    base_phi_deg: float,
    surface_zone: dict,
    label: str,
) -> float:
    best_phi, _best_clearance, _best_inside = _evaluate_place_phi_for_edge_clearance(
        held_object,
        target_xy_mm=target_xy_mm,
        base_phi_deg=base_phi_deg,
        surface_zone=surface_zone,
        label=label,
        verbose=True,
    )
    return best_phi


def _optimize_place_target_for_slot(
    held_object: CandidateDebug | None,
    *,
    target_xy_mm: np.ndarray,
    base_phi_deg: float,
    surface_zone: dict,
    label: str,
    verbose: bool,
) -> OptimizedPlaceTarget:
    if not EFFICIENT_PACKING_ENABLED:
        base_xy = np.asarray(target_xy_mm, dtype=np.float64).reshape(2)
        base_phi = _normalize_phi_deg(base_phi_deg)
        servo_deg = _held_initial_servo_deg(held_object)
        footprint_length_mm, footprint_width_mm = _gripper_footprint_size_mm(servo_deg)
        footprint_clearance_mm, _corners, _min_xy, _max_xy = _footprint_signed_edge_clearance_mm(
            base_xy,
            base_phi,
            footprint_length_mm,
            footprint_width_mm,
            surface_zone=surface_zone,
        )
        aabb_clearance_mm = _aabb_signed_edge_clearance_mm(base_xy, np.array([footprint_length_mm, footprint_width_mm], dtype=np.float64), surface_zone=surface_zone)
        fit_clearance_mm = min(float(footprint_clearance_mm), float(aabb_clearance_mm))
        return OptimizedPlaceTarget(
            target_xy_mm=base_xy,
            target_phi_deg=base_phi,
            footprint_clearance_mm=float(footprint_clearance_mm),
            aabb_clearance_mm=float(aabb_clearance_mm),
            fit_clearance_mm=float(fit_clearance_mm),
            nudge_xy_mm=np.array([0.0, 0.0], dtype=np.float64),
            can_place=bool(fit_clearance_mm >= 0.0),
            reason="efficient_packing_disabled",
        )
    return _evaluate_slot_fit(
        held_object,
        target_xy_mm=target_xy_mm,
        base_phi_deg=base_phi_deg,
        surface_zone=surface_zone,
        label=label,
        verbose=verbose,
    )


def _validate_gripper_footprint_inside_bag(
    held_object: CandidateDebug | None,
    *,
    target_xy_mm: np.ndarray,
    target_phi_deg: float,
    surface_zone: dict,
    label: str,
) -> bool:
    if not PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG:
        return True

    servo_deg = _held_initial_servo_deg(held_object)
    length_mm, width_mm = _gripper_footprint_size_mm(servo_deg)
    clearance_mm, corners, min_xy, max_xy = _footprint_signed_edge_clearance_mm(
        np.asarray(target_xy_mm, dtype=np.float64).reshape(2),
        target_phi_deg,
        length_mm,
        width_mm,
        surface_zone=surface_zone,
    )
    inside = bool(clearance_mm >= 0.0)

    print("[PLACE FOOTPRINT CHECK]")
    print(f"label={label} target_xy=({float(target_xy_mm[0]):.1f},{float(target_xy_mm[1]):.1f}) phi={float(target_phi_deg):.1f}")
    print(f"servo_deg={servo_deg:.1f} footprint_length={length_mm:.1f} width={width_mm:.1f}")
    print(f"bag_bounds x=[{min_xy[0]:.1f},{max_xy[0]:.1f}] y=[{min_xy[1]:.1f},{max_xy[1]:.1f}]")
    print(f"signed_edge_clearance_mm={clearance_mm:.1f}")
    for idx, corner in enumerate(corners, start=1):
        print(f"  corner{idx}: x={corner[0]:.1f} y={corner[1]:.1f}")
    if not inside:
        print("[PLACE FOOTPRINT CHECK] ABORT: gripper footprint would leave the bag rectangle.")
    return inside


def _raise_or_hold_safe_z(robot, label: str, z_mm: float | None = None, move_time_s: float | None = None) -> bool:
    target_z = float(Z_MAX_MM if z_mm is None else z_mm)
    reason = validate_z_command(target_z, label, config=DEFAULT_Z_SAFETY)
    if reason:
        print(f"{label} ABORT: {reason}")
        return False
    print(f"{label} WARNING: commanding safe Z hold/raise to {target_z:.1f} mm.")
    return bool(robot.move_cartesian(z_mm=target_z, move_time_s=float(move_time_s or COARSE_MOVE_TIME_S)))


def _survey_pose_from_pick_one() -> tuple[float, float, float, float]:
    return (
        float(_pick_one_mod.X_SURVEY),
        float(_pick_one_mod.Y_SURVEY),
        float(_pick_one_mod.Z_SURVEY),
        float(RECOVERY_POSE_PHI_DEG),
    )


def _recovery_pose_xyzphi() -> tuple[float, float, float, float]:
    survey_x, survey_y, survey_z, survey_phi = _survey_pose_from_pick_one()
    x = survey_x if RECOVERY_POSE_X_MM is None else float(RECOVERY_POSE_X_MM)
    y = survey_y if RECOVERY_POSE_Y_MM is None else float(RECOVERY_POSE_Y_MM)
    z = survey_z if RECOVERY_POSE_Z_MM is None else float(RECOVERY_POSE_Z_MM)
    return x, y, z, survey_phi


def _move_to_recovery_pose(robot) -> bool:
    print("RECOVERY: moving to recalibration pose")
    recovery_x, recovery_y, recovery_z, recovery_phi = _recovery_pose_xyzphi()
    print(
        "[RECOVERY] WARNING: robot will move to "
        f"({recovery_x:.1f}, {recovery_y:.1f}, {recovery_z:.1f}, "
        f"phi={recovery_phi:.1f}). Keep clear."
    )
    reason = validate_z_command(float(recovery_z), "[RECOVERY] pose z", config=DEFAULT_Z_SAFETY)
    if reason:
        print(f"[RECOVERY] ABORT: {reason}")
        return False

    if not _raise_or_hold_safe_z(robot, "[RECOVERY] pre-move safe z", float(recovery_z), RECOVERY_MOVE_TIME_S):
        return False
    return _sequence_move_checked(
        robot,
        "[RECOVERY] XY+phi",
        x_mm=float(recovery_x),
        y_mm=float(recovery_y),
        z_mm=float(recovery_z),
        phi_deg=float(recovery_phi),
        move_time_s=float(RECOVERY_MOVE_TIME_S),
    )


def _rehome_j3_if_requested(robot) -> bool:
    if not RECOVERY_REHOME_J3_ENABLED:
        print("[RECOVERY] J3 rehome disabled.")
        return True

    print("RECOVERY: rehoming J3")
    recovery_z = _recovery_pose_xyzphi()[2]
    if RECOVERY_DROP_Z_BEFORE_REHOME_MM is not None:
        z = float(RECOVERY_DROP_Z_BEFORE_REHOME_MM)
    else:
        z = float(recovery_z)
    if z is not None:
        reason = validate_z_command(z, "[RECOVERY] pre-HOMEJ3 z", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[RECOVERY] ABORT: {reason}")
            return False
        print(f"[RECOVERY] WARNING: moving to configured pre-HOMEJ3 z={z:.1f} mm.")
        if not robot.move_cartesian(z_mm=z, move_time_s=float(RECOVERY_MOVE_TIME_S)):
            print("[RECOVERY] ABORT: failed pre-HOMEJ3 Z move.")
            return False

    if not hasattr(robot, "send") or not hasattr(robot, "read_until"):
        print("[RECOVERY] WARN: robot has no raw HOMEJ3 API; falling back to sync_estimate_from_teensy_steps if available.")
        if hasattr(robot, "sync_estimate_from_teensy_steps"):
            ok = bool(robot.sync_estimate_from_teensy_steps())
            print(f"[RECOVERY] J3 sync fallback result={ok}")
            return ok
        return False

    robot.send("HOMEJ3")
    ok = bool(robot.read_until("HOMED", float(RECOVERY_REHOME_TIMEOUT_S), match="exact"))
    print(f"[RECOVERY] HOMEJ3 result={ok}")
    if ok and hasattr(robot, "sync_estimate_from_teensy_steps"):
        sync_ok = bool(robot.sync_estimate_from_teensy_steps())
        print(f"[RECOVERY] post-HOMEJ3 sync result={sync_ok}")
        ok = ok and sync_ok
    return ok


def _match_recovered_candidate(
    survey_state: SurveyState,
    attempt: PickAttemptRecord,
) -> tuple[CandidateDebug | None, str]:
    candidates = list(survey_state.candidates)
    if not candidates:
        return None, "survey returned no candidates"

    same_class = [dbg for dbg in candidates if _candidate_class_name(dbg) == attempt.class_name]
    pool = same_class if same_class else candidates
    if not same_class:
        print("[RECOVERY WARN] no same-class candidate in recovery survey; considering nearest candidate by XY.")

    max_dist = max(float(MISS_MATCH_MAX_ROBOT_DIST_MM) * 3.0, 90.0)
    ranked: list[tuple[float, CandidateDebug]] = []
    for dbg in pool:
        xy = _candidate_target_xy(dbg)
        if not np.all(np.isfinite(xy)):
            continue
        dist = float(np.linalg.norm(xy - attempt.original_target_xy_mm))
        ranked.append((dist, dbg))

    if not ranked:
        return None, "no recovered candidates had finite XY"

    ranked.sort(key=lambda item: item[0])
    dist, dbg = ranked[0]
    if dist > max_dist:
        return None, f"nearest candidate too far from missed pick ({dist:.1f} > {max_dist:.1f} mm)"
    return dbg, f"matched class={_candidate_class_name(dbg)} dist={dist:.1f} mm"


def recover_target_after_confirmed_miss(
    *,
    attempt: PickAttemptRecord,
    miss_result: MissMatchResult,
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict,
    rectifier: StereoRectifier,
    yolo,
    raft,
    robot,
    bundle: dict,
    camera_lock: threading.Lock,
) -> RecoveredTarget | None:
    if not miss_result.is_miss:
        print("[RECOVERY] skipped: miss was not confirmed.")
        return None

    if not _move_to_recovery_pose(robot):
        return None
    rehome_ok = _rehome_j3_if_requested(robot)
    print(f"[RECOVERY] J3 rehome result={rehome_ok}")
    if not rehome_ok:
        return None

    if not RETRY_RELOCALIZE_WITH_RAFT_ONLY_AFTER_MISS:
        print("[RECOVERY] RAFT relocalization disabled; retrying with original target.")
        dbg = copy.deepcopy(attempt.candidate_debug)
        return RecoveredTarget(
            candidate_debug=dbg,
            target_xy_mm=attempt.original_target_xy_mm.copy(),
            target_z_mm=_candidate_surface_z_mm(dbg),
            phi_deg=attempt.original_phi_deg,
            object_height_mm=attempt.original_object_height_mm,
            safe_retry_required=True,
            reason="RAFT recovery disabled after YOLO-confirmed miss",
        )

    print("RECOVERY: running RAFT/pointcloud once")
    print("[RECOVERY] expensive localization is running only after YOLO-confirmed miss.")
    with camera_lock:
        survey_state = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)

    recovered_dbg, reason = _match_recovered_candidate(survey_state, attempt)
    if recovered_dbg is None:
        print(f"[RECOVERY] failed to match missed object: {reason}")
        return None

    recovered_xy = _candidate_target_xy(recovered_dbg)
    same_xy_dist = float(np.linalg.norm(recovered_xy - attempt.original_target_xy_mm))
    safe_retry_required = bool(RETRY_SAFE_PICK_ENABLED) and (
        same_xy_dist <= float(RETRY_XY_SAME_THRESHOLD_MM) and bool(RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY)
    )
    target_z = _candidate_surface_z_mm(recovered_dbg)
    object_height = _candidate_height_mm(recovered_dbg)
    phi = _candidate_phi_deg(robot, recovered_dbg)
    print(
        "[RECOVERY] recovered target: "
        f"class={_candidate_class_name(recovered_dbg)} xy=({recovered_xy[0]:.1f},{recovered_xy[1]:.1f}) "
        f"z={target_z:.1f} height={object_height:.1f} phi={phi:.1f} same_xy_dist={same_xy_dist:.1f}"
    )
    return RecoveredTarget(
        candidate_debug=recovered_dbg,
        target_xy_mm=recovered_xy.copy(),
        target_z_mm=float(target_z),
        phi_deg=float(phi),
        object_height_mm=float(object_height),
        safe_retry_required=safe_retry_required,
        reason=reason,
    )


def _preview_pick_grasp_z_mm(dbg: CandidateDebug) -> float | None:
    c = dbg.candidate
    try:
        x = float(c.target_xy[0])
        y = float(c.target_xy[1])
    except Exception:
        x = y = float("nan")

    pick_surface_override_mm = None
    if bool(getattr(_pick_one_mod, "USE_Z_GROUND_MODEL_FOR_PICK_SURFACE", False)):
        try:
            model = _pick_one_mod._load_z_ground_model_cached()
            z_ground_mm = _pick_one_mod._predict_z_ground_mm(model, x, y)
            object_height_mm = _pick_one_mod._candidate_object_height_mm_for_zground(c)
            if z_ground_mm is not None and object_height_mm is not None:
                pick_surface_override_mm = float(z_ground_mm + object_height_mm)
        except Exception as exc:
            print(f"[RETRY PICK] preview z-ground fallback: {exc}")

    try:
        plan = compute_pick_z_plan(
            z_result=getattr(c, "z_debug", None),
            object_surface_z_mm=pick_surface_override_mm,
            candidate=c,
            gripper_offset_mm=float(_pick_one_mod.GRIPPER_OFFSET_MM),
            z_max_mm=float(Z_MAX_MM),
            pick_extra_clearance_mm=float(getattr(_pick_one_mod, "PICK_EXTRA_CLEARANCE_MM", 0.0)),
            pick_uncertainty_gain=float(getattr(_pick_one_mod, "PICK_Z_UNCERTAINTY_GAIN", 0.0)),
            pick_uncertainty_clearance_max_mm=float(getattr(_pick_one_mod, "PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM", 0.0)),
            config=DEFAULT_Z_SAFETY,
        )
        return float(plan.final_grasp_z_mm)
    except Exception as exc:
        print(f"[RETRY PICK] grasp Z preview unavailable: {exc}")
        return None


def _snapshot_pick_geometry(dbg: CandidateDebug) -> dict[str, Any]:
    c = dbg.candidate
    names = (
        "z_debug",
        "object_robot_xyz_raw",
        "object_robot_xyz_corrected",
        "target_cam_xyz",
        "hover_robot_z",
        "grasp_robot_z",
    )
    return {name: copy.deepcopy(getattr(c, name, None)) for name in names}


def _restore_pick_geometry(dbg: CandidateDebug, snapshot: dict[str, Any]) -> None:
    c = dbg.candidate
    for name, value in snapshot.items():
        if value is not None or hasattr(c, name):
            setattr(c, name, copy.deepcopy(value))


def _add_retry_grasp_z_offset(dbg: CandidateDebug, offset_mm: float) -> None:
    if offset_mm <= 0.0:
        return
    c = dbg.candidate
    z_debug = getattr(c, "z_debug", None)
    if z_debug is not None:
        for attr in (
            "raw_min_z_mm",
            "raw_max_z_mm",
            "z_p50_mm",
            "z_p90_mm",
            "z_p95_mm",
            "z_p99_mm",
            "robust_top_z_mm",
            "grasp_z_mm",
            "hover_z_mm",
            "travel_z_mm",
        ):
            if hasattr(z_debug, attr):
                try:
                    setattr(z_debug, attr, float(getattr(z_debug, attr)) + float(offset_mm))
                except (TypeError, ValueError):
                    pass
        if hasattr(z_debug, "object_height_mm"):
            try:
                setattr(z_debug, "object_height_mm", max(0.0, float(getattr(z_debug, "object_height_mm")) + float(offset_mm)))
            except (TypeError, ValueError):
                pass
        if hasattr(z_debug, "warnings"):
            z_debug.warnings = list(getattr(z_debug, "warnings", []) or []) + [f"retry_grasp_z_offset_{offset_mm:.1f}mm_for_pick_only"]
        return

    for attr in ("object_robot_xyz_raw", "object_robot_xyz_corrected", "target_cam_xyz"):
        value = getattr(c, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).copy()
        if arr.size >= 3 and np.isfinite(arr.reshape(-1)[2]):
            arr.reshape(-1)[2] += float(offset_mm)
            setattr(c, attr, arr)


def _patch_pick_globals_for_retry() -> tuple[dict[str, Any], float, float, float]:
    names = (
        "CLAW_OPEN_DEG",
        "DYNAMIC_SERVO_MARGIN_DEG",
        "GRIPPER_SERVO_MAX_DEG",
        "DYNAMIC_PICK_DEFAULT_SERVO_DEG",
    )
    old = {name: getattr(_pick_one_mod, name, None) for name in names}
    base_open = float(old["CLAW_OPEN_DEG"])
    retry_open = float(np.clip(base_open + float(RETRY_START_CLAW_EXTRA_DEG), 0.0, float(RETRY_START_CLAW_MAX_DEG)))
    base_default = float(old["DYNAMIC_PICK_DEFAULT_SERVO_DEG"])
    retry_default = float(np.clip(base_default + float(RETRY_START_CLAW_EXTRA_DEG), 0.0, float(RETRY_START_CLAW_MAX_DEG)))

    _pick_one_mod.CLAW_OPEN_DEG = retry_open
    _pick_one_mod.DYNAMIC_SERVO_MARGIN_DEG = float(old["DYNAMIC_SERVO_MARGIN_DEG"]) + float(RETRY_START_CLAW_EXTRA_DEG)
    _pick_one_mod.GRIPPER_SERVO_MAX_DEG = float(RETRY_START_CLAW_MAX_DEG)
    _pick_one_mod.DYNAMIC_PICK_DEFAULT_SERVO_DEG = retry_default
    if hasattr(_pick_one_mod, "_motion_mod"):
        _pick_one_mod._motion_mod.CLAW_OPEN_DEG = retry_open
    return old, base_open, retry_open, retry_default


def _restore_pick_globals(old: dict[str, Any]) -> None:
    for name, value in old.items():
        if value is not None:
            setattr(_pick_one_mod, name, value)
    if hasattr(_pick_one_mod, "_motion_mod") and old.get("CLAW_OPEN_DEG") is not None:
        _pick_one_mod._motion_mod.CLAW_OPEN_DEG = old["CLAW_OPEN_DEG"]


def execute_retry_pick_selected(
    robot,
    recovered: RecoveredTarget,
    *,
    original_attempt: PickAttemptRecord,
    bundle: dict | None,
) -> tuple[bool, CandidateDebug | None]:
    retry_dbg = copy.deepcopy(recovered.candidate_debug)
    retry_dbg.candidate.target_xy = np.asarray(recovered.target_xy_mm, dtype=np.float64).reshape(2).copy()

    same_xy_dist = float(np.linalg.norm(recovered.target_xy_mm - original_attempt.original_target_xy_mm))
    force_safe = bool(RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY) and same_xy_dist <= float(RETRY_XY_SAME_THRESHOLD_MM)
    use_safe = bool(RETRY_SAFE_PICK_ENABLED) and (bool(recovered.safe_retry_required) or force_safe)

    if not use_safe:
        print("[RETRY PICK] safe retry disabled/not required; using normal pick helper.")
        ok = execute_pick_selected(robot, retry_dbg, bundle=bundle)
        return bool(ok), retry_dbg if ok else None

    print("RETRY PICK: safe sequence")
    print(
        f"[RETRY PICK] reason={recovered.reason}; same_xy_dist={same_xy_dist:.1f} mm "
        f"force_safe_if_same_xy={force_safe}"
    )
    print(
        f"[RETRY PICK] target before: class={_candidate_class_name(retry_dbg)} "
        f"xy=({original_attempt.original_target_xy_mm[0]:.1f},{original_attempt.original_target_xy_mm[1]:.1f}) "
        f"phi={original_attempt.original_phi_deg:.1f} height={original_attempt.original_object_height_mm:.1f}"
    )
    print(
        f"[RETRY PICK] target after:  class={_candidate_class_name(retry_dbg)} "
        f"xy=({recovered.target_xy_mm[0]:.1f},{recovered.target_xy_mm[1]:.1f}) "
        f"phi={recovered.phi_deg:.1f} height={recovered.object_height_mm:.1f}"
    )

    before_grasp_z = _preview_pick_grasp_z_mm(retry_dbg)
    geometry_snapshot = _snapshot_pick_geometry(retry_dbg)
    _add_retry_grasp_z_offset(retry_dbg, float(RETRY_GRASP_Z_OFFSET_MM))
    after_grasp_z = _preview_pick_grasp_z_mm(retry_dbg)

    old_globals, base_open, retry_open, retry_default = _patch_pick_globals_for_retry()
    print(
        f"[RETRY PICK] claw open: {base_open:.1f} -> {retry_open:.1f} deg; "
        f"dynamic fallback start={retry_default:.1f} deg; max={RETRY_START_CLAW_MAX_DEG:.1f} deg"
    )
    print(
        f"[RETRY PICK] grasp Z preview: {_fmt_optional(before_grasp_z, 1)} -> "
        f"{_fmt_optional(after_grasp_z, 1)} mm using retry_offset={RETRY_GRASP_Z_OFFSET_MM:.1f} mm"
    )
    print("[RETRY PICK] WARNING: executing retry pick motion now.")

    ok = False
    try:
        ok = bool(execute_pick_selected(robot, retry_dbg, bundle=bundle))
    finally:
        _restore_pick_globals(old_globals)
        _restore_pick_geometry(retry_dbg, geometry_snapshot)

    if ok:
        print("[RETRY PICK] OK - safe retry pick reports object should be held.")
        return True, retry_dbg

    print("[RETRY PICK] failed.")
    return False, None


def _read_live_frames(overhead: SimpleOverheadCamera, stereo: SimpleStereoCamera):
    ok_oh, live_overhead = overhead.read()
    if not ok_oh:
        live_overhead = None
    ok_st, _full, live_left, live_right = stereo.read_pair()
    if not ok_st:
        live_left = None
        live_right = None
    return live_overhead, live_left, live_right


def _make_autonomous_display(
    state: SurveyState | None,
    live_overhead: np.ndarray | None,
    live_left: np.ndarray | None,
    live_right: np.ndarray | None,
    status_lines: list[str],
    placeability_overlay: dict[int, PlaceabilityOverlayEntry] | None = None,
) -> np.ndarray:
    frame = make_display(state, live_overhead, live_left, live_right)
    if state is not None and placeability_overlay:
        overhead_frame = getattr(state, "overhead_frame", None)
        if overhead_frame is not None:
            src_h, src_w = overhead_frame.shape[:2]
            scale_x = float(COMBINED_WIDTH_PX) / max(float(src_w), 1.0)
            scale_y = float(OVERHEAD_DRAW_H_PX) / max(float(src_h), 1.0)
            reject_count = 0
            for idx, dbg in enumerate(state.candidates):
                entry = placeability_overlay.get(idx)
                if entry is None or entry.can_place:
                    continue
                px = getattr(dbg.candidate, "overhead_centroid_px", None)
                if px is None:
                    continue
                cx = int(round(float(px[0]) * scale_x))
                cy = int(round(float(px[1]) * scale_y))
                reject_count += 1
                cv2.circle(frame, (cx, cy), 18, (0, 0, 255), 2)
                cv2.line(frame, (cx - 12, cy - 12), (cx + 12, cy + 12), (0, 0, 255), 2)
                cv2.line(frame, (cx - 12, cy + 12), (cx + 12, cy - 12), (0, 0, 255), 2)
                clearance_text = "n/a" if entry.clearance_mm is None else f"{entry.clearance_mm:.1f}mm"
                put_text_outline(frame, f"NP {clearance_text}", (cx + 20, max(20, cy - 14)), scale=0.5, color=(80, 80, 255), thickness=1)
            if reject_count > 0:
                put_text_outline(
                    frame,
                    f"Not placeable in current slot: {reject_count}",
                    (12, 48),
                    scale=0.6,
                    color=(80, 80, 255),
                    thickness=2,
                )
    y0 = int(OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, y0), (w, h), (8, 8, 8), -1)
    for i, line in enumerate(status_lines[:5]):
        color = (170, 255, 170) if i == 0 else (230, 230, 230)
        put_text_outline(frame, line, (8, y0 + 24 + i * 23), scale=0.55, color=color, thickness=1)
    return frame


def _show_display(
    *,
    state: SurveyState | None,
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    status_lines: list[str],
    read_live: bool,
    placeability_overlay: dict[int, PlaceabilityOverlayEntry] | None = None,
) -> None:
    if read_live:
        live_overhead, live_left, live_right = _read_live_frames(overhead, stereo)
    else:
        live_overhead = live_left = live_right = None
    cv2.imshow(
        WINDOW,
        _make_autonomous_display(
            state,
            live_overhead,
            live_left,
            live_right,
            status_lines,
            placeability_overlay=placeability_overlay,
        ),
    )
    _check_abort_key()


def _check_abort_key(delay_ms: int = 1) -> None:
    key = read_command_key(delay_ms=delay_ms)
    if key in ("q", "escape", "\x1b"):
        raise UserAbort("quit requested")
    if key == "c":
        raise ClearBoxRequest("clear box requested")


def _read_yolo_watchdog_frame(
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    *,
    frame_i: int,
) -> np.ndarray | None:
    if MISS_CHECK_CAMERA == "overhead":
        if frame_i == 0 and OVERHEAD_FRESH_READ_DISCARD_FRAMES > 0:
            print(f"[MISS CHECK] flushing {OVERHEAD_FRESH_READ_DISCARD_FRAMES} overhead frame(s) before YOLO burst")
            for _ in range(max(0, int(OVERHEAD_FRESH_READ_DISCARD_FRAMES))):
                overhead.read()
                if OVERHEAD_FRESH_READ_DELAY_S > 0.0:
                    time.sleep(float(OVERHEAD_FRESH_READ_DELAY_S))
        ok, frame = overhead.read()
        return frame if ok else None

    if MISS_CHECK_CAMERA == "stereo_left":
        ok, _full, left, _right = stereo.read_pair()
        return left if ok else None

    raise ValueError(f"Unsupported MISS_CHECK_CAMERA={MISS_CHECK_CAMERA!r}")


def run_yolo_only_miss_check(
    *,
    attempt: PickAttemptRecord,
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    yolo,
    bundle: dict,
    camera_lock: threading.Lock,
) -> MissMatchResult:
    if not MISS_CHECK_ENABLED:
        return MissMatchResult(False, 0.0, 0, 0, None, None, None, None, None, False, "miss check disabled")

    print("MISS CHECK: collecting YOLO burst")
    print("[MISS CHECK] RAFT/disparity/pointcloud are intentionally skipped in this watchdog.")
    frames: list[np.ndarray] = []
    deadline = time.perf_counter() + float(MISS_CHECK_TIMEOUT_S)

    with camera_lock:
        for frame_i in range(max(1, int(MISS_BURST_COUNT))):
            if time.perf_counter() > deadline:
                print(f"[MISS CHECK] burst timeout after {len(frames)} frame(s)")
                break
            frame = _read_yolo_watchdog_frame(overhead, stereo, frame_i=frame_i)
            if frame is not None:
                frames.append(frame.copy())
            _check_abort_key(delay_ms=1)
            if frame_i < int(MISS_BURST_COUNT) - 1 and MISS_CHECK_PERIOD_S > 0.0:
                time.sleep(float(MISS_CHECK_PERIOD_S))

        detections_by_frame = yolo.segment_batch(frames) if frames else []

    burst_count = len(frames)
    if burst_count == 0:
        return MissMatchResult(False, 0.0, 0, 0, None, None, None, None, None, False, "no frames captured")

    best_score = -1.0
    best_det: YOLODetection | None = None
    best_xy: np.ndarray | None = None
    best_robot_dist: float | None = None
    best_iou: float | None = None
    best_area_ratio: float | None = None
    best_same_class = False
    best_reason = "no detections"
    hit_scores: list[float] = []

    for frame_i, dets in enumerate(detections_by_frame):
        frame_best: tuple[float, YOLODetection, np.ndarray | None, float | None, float | None, float | None, bool, str] | None = None
        for det in dets:
            score, robot_xy, robot_dist, iou, area_ratio, same_class, reason = _score_miss_detection(det, attempt, bundle)
            if frame_best is None or score > frame_best[0]:
                frame_best = (score, det, robot_xy, robot_dist, iou, area_ratio, same_class, reason)
        if frame_best is None:
            print(f"[MISS CHECK] frame {frame_i + 1}/{burst_count}: no YOLO detections")
            continue

        score, det, robot_xy, robot_dist, iou, area_ratio, same_class, reason = frame_best
        print(
            f"[MISS CHECK] frame {frame_i + 1}/{burst_count}: "
            f"best={det.class_name} score={score:.3f} {reason}"
        )
        if score > best_score:
            best_score = float(score)
            best_det = det
            best_xy = None if robot_xy is None else np.asarray(robot_xy, dtype=np.float64).reshape(2)
            best_robot_dist = robot_dist
            best_iou = iou
            best_area_ratio = area_ratio
            best_same_class = bool(same_class)
            best_reason = reason
        if score >= float(MISS_SCORE_THRESHOLD):
            hit_scores.append(float(score))

    hits = len(hit_scores)
    aggregate_score = float(np.mean(hit_scores)) if hit_scores else max(0.0, float(best_score))
    is_miss = hits >= int(MISS_MIN_HITS) and aggregate_score >= float(MISS_SCORE_THRESHOLD)
    reason = (
        f"hits={hits}/{burst_count} aggregate_score={aggregate_score:.3f} "
        f"threshold={MISS_SCORE_THRESHOLD:.3f}; best: {best_reason}"
    )
    if best_xy is None and MISS_CHECK_CAMERA == "overhead":
        reason += "; robot-frame distance unavailable, using class/IoU/area fallback"
    if is_miss:
        print("MISS DETECTED: same object still at pick site")
    else:
        print(f"[MISS CHECK] no miss: {reason}")

    return MissMatchResult(
        is_miss=bool(is_miss),
        score=aggregate_score,
        hits=hits,
        burst_count=burst_count,
        best_detection=best_det,
        best_robot_xy_mm=best_xy,
        robot_dist_mm=best_robot_dist,
        iou=best_iou,
        area_ratio=best_area_ratio,
        same_class=best_same_class,
        reason=reason,
    )


def _select_best_for_state(
    state: SurveyState,
    *,
    robot,
    placed_boxes: list,
    config: BestCandidateConfig,
) -> BestCandidateResult:
    result = choose_best_candidate(state, config=config, robot=robot, placed_boxes=placed_boxes)
    result.print_debug("[BEST]")
    if result.selected is not None:
        state.selected_index = state.candidates.index(result.selected)
    return result


def _select_best_for_state_by_slot_fit(
    state: SurveyState,
    *,
    object_i: int,
    robot,
    placed_boxes: list,
    config: BestCandidateConfig,
    surface_zone: dict,
    base_xy: np.ndarray,
    base_phi_deg: float,
    column_xy_primary: np.ndarray | None,
    column_xy_secondary: np.ndarray | None,
    target_limit: int,
) -> tuple[BestCandidateResult, dict[int, PlaceabilityOverlayEntry]]:
    result = choose_best_candidate(state, config=config, robot=robot, placed_boxes=placed_boxes)
    result.print_debug("[BEST]")
    if not state.candidates:
        return result, {}

    overlay: dict[int, PlaceabilityOverlayEntry] = {}
    scored: list[tuple[float, float, float, int, CandidateDebug, OptimizedPlaceTarget]] = []

    def compute_target_for_candidate(cand_dbg: CandidateDebug) -> OptimizedPlaceTarget:
        raw_box = _aabb_from_object_candidate_quiet(cand_dbg.candidate, default_label=f"object{object_i}_overlay")
        if object_i == 1:
            target_xy = np.asarray(base_xy, dtype=np.float64).reshape(2).copy()
            target_phi = float(base_phi_deg)
        elif object_i == 2:
            if not placed_boxes:
                raise ValueError("waiting_for_reference_box")
            moving_padded = pad_aabb(raw_box, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
            adjacent_plan = compute_adjacent_placement(
                reference_padded_box=placed_boxes[-1],
                moving_padded_box=moving_padded,
                direction=ADJACENT_DIRECTION,
                surface_z_mm=float(surface_zone["surface_z_mm"]),
                place_phi_deg=float(base_phi_deg),
            )
            target_xy = adjacent_plan.target_center_xy_mm
            target_phi = float(adjacent_plan.target_phi_deg)
        else:
            if column_xy_primary is None or column_xy_secondary is None:
                raise ValueError("waiting_for_column_anchor")
            target_xy = np.asarray(column_xy_primary if object_i % 2 == 1 else column_xy_secondary, dtype=np.float64).reshape(2).copy()
            target_phi = float(base_phi_deg)

        return _optimize_place_target_for_slot(
            cand_dbg,
            target_xy_mm=target_xy,
            base_phi_deg=target_phi,
            surface_zone=surface_zone,
            label=f"object{object_i}",
            verbose=False,
        )

    for idx, cand_dbg in enumerate(state.candidates):
        try:
            optimized = compute_target_for_candidate(cand_dbg)
            decision = next((d for d in result.decisions if d.dbg is cand_dbg), None)
            if decision is None or not decision.passed:
                reject_reason = "selector_rejected"
                if decision is not None and decision.reject_reasons:
                    reject_reason = ";".join(decision.reject_reasons[:2])
                overlay[idx] = PlaceabilityOverlayEntry(
                    can_place=False,
                    target_xy_mm=optimized.target_xy_mm.copy(),
                    target_phi_deg=float(optimized.target_phi_deg),
                    clearance_mm=float(optimized.fit_clearance_mm),
                    reason=reject_reason,
                )
                continue
            overlay[idx] = PlaceabilityOverlayEntry(
                can_place=bool(optimized.can_place),
                target_xy_mm=optimized.target_xy_mm.copy(),
                target_phi_deg=float(optimized.target_phi_deg),
                clearance_mm=float(optimized.fit_clearance_mm),
                reason=optimized.reason,
            )
            if optimized.can_place:
                volume_mm3 = float(decision.volume_mm3)
                scored.append((float(optimized.fit_clearance_mm), volume_mm3, -float(np.dot(optimized.nudge_xy_mm, optimized.nudge_xy_mm)), int(getattr(cand_dbg.candidate, 'index', -1)), cand_dbg, optimized))
        except Exception as exc:
            overlay[idx] = PlaceabilityOverlayEntry(
                can_place=False,
                target_xy_mm=None,
                target_phi_deg=None,
                clearance_mm=None,
                reason=str(exc),
            )

    if not scored:
        # Do not hard-stop the run: fall back to largest valid candidate and let
        # placement logic continue trying to find a feasible target.
        if result.selected_decision is not None:
            result.selected = result.selected_decision.dbg
            state.selected_index = state.candidates.index(result.selected)
            print(
                f"[BEST PACK WARN] no slot-fit candidate for object {object_i}; "
                f"falling back to largest valid candidate [{result.selected_decision.candidate_index}] "
                f"{result.selected_decision.class_name}."
            )
        else:
            result.selected = None
            result.selected_decision = None
        return result, overlay

    scored.sort(key=lambda item: (item[0], item[1], item[2], -item[3]), reverse=True)
    best = scored[0]
    selected_dbg = best[4]
    selected_optimized = best[5]
    best_decision = next((d for d in result.decisions if d.dbg is selected_dbg), None)
    if best_decision is not None:
        result.selected = selected_dbg
        result.selected_decision = best_decision
        state.selected_index = state.candidates.index(selected_dbg)
        print(
            f"[BEST PACK] object={object_i} selected={best_decision.candidate_index} {best_decision.class_name} "
            f"fit_clearance={selected_optimized.fit_clearance_mm:.1f}mm volume={best_decision.volume_cm3:.1f}cm3"
        )
    return result, overlay


def _print_knob_group(title: str, lines: list[str]) -> None:
    print(f"\n[{title}]")
    for line in lines:
        print(f"  {line}")


def print_knob_overview() -> None:
    _hr("USER SETTINGS - MISSED PICK RECOVERY", "=")
    _print_knob_group(
        "1. Run count / loop control",
        [
            f"TARGET_OBJECT_COUNT={TARGET_OBJECT_COUNT}",
            f"RUN_UNTIL_NO_VALID_CANDIDATE={RUN_UNTIL_NO_VALID_CANDIDATE}",
            f"MAX_OBJECT_COUNT_SAFETY={MAX_OBJECT_COUNT_SAFETY}",
            f"NO_CANDIDATE_RETRY_COUNT={NO_CANDIDATE_RETRY_COUNT}",
            f"NO_CANDIDATE_RETRY_DELAY_S={NO_CANDIDATE_RETRY_DELAY_S}",
            f"NO_CANDIDATE_AFTER_RETRIES_MODE={NO_CANDIDATE_AFTER_RETRIES_MODE!r}",
            f"NO_CANDIDATE_RESUME_KEY={NO_CANDIDATE_RESUME_KEY!r}",
            f"ON_RETRY_FAIL={ON_RETRY_FAIL!r}",
        ],
    )
    _print_knob_group(
        "2. Pick location / pick target interpretation",
        [
            f"PLATFORM_GRID_X_MM={PLATFORM_GRID_X_MM}",
            f"PLATFORM_GRID_Y_MM={PLATFORM_GRID_Y_MM}",
            f"PLATFORM_GRID_Z_MM={PLATFORM_GRID_Z_MM}",
            f"platform_bounds_x=[{PLATFORM_X_MIN_MM},{PLATFORM_X_MAX_MM}]",
            f"platform_bounds_y=[{PLATFORM_Y_MIN_MM},{PLATFORM_Y_MAX_MM}]",
            f"startup_survey_pose_from_pick_one=({_pick_one_mod.X_SURVEY},{_pick_one_mod.Y_SURVEY},{_pick_one_mod.Z_SURVEY})",
            f"PICK_PHI_MODE={_pick_one_mod.PICK_PHI_MODE}",
            f"USE_LOCAL_HEIGHT_AWARE_GRASP_XY={_pick_one_mod.USE_LOCAL_HEIGHT_AWARE_GRASP_XY}",
            f"LOCAL_GRASP_RADIUS_MM={_pick_one_mod.LOCAL_GRASP_RADIUS_MM}",
            f"LOCAL_GRASP_BLEND_WEIGHT={_pick_one_mod.LOCAL_GRASP_BLEND_WEIGHT}",
            f"USE_Z_GROUND_MODEL_FOR_PICK_SURFACE={get_use_z_ground_model_for_pick_surface()}",
            f"REQUIRE_OVERHEAD_XY_FOR_PICK={_pick_one_mod.REQUIRE_OVERHEAD_XY_FOR_PICK}",
        ],
    )
    _print_knob_group(
        "3. Packing conservatism / slot fit",
        [
            f"PLACE_SURFACE_ZONE_NAME={_pick_one_mod.PLACE_SURFACE_ZONE_NAME!r}",
            f"ADJACENT_DIRECTION={ADJACENT_DIRECTION!r}",
            f"EFFICIENT_PACKING_ENABLED={EFFICIENT_PACKING_ENABLED}",
            f"EFFICIENT_PACKING_REQUIRE_SLOT_FIT={EFFICIENT_PACKING_REQUIRE_SLOT_FIT}",
            f"PLACE_XY_NUDGE_ENABLED={PLACE_XY_NUDGE_ENABLED} step={PLACE_XY_NUDGE_STEP_MM} max={PLACE_XY_NUDGE_MAX_MM}",
            f"PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG={PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG}",
            f"PLACE_GRIPPER_FOOTPRINT_WIDTH_MM={PLACE_GRIPPER_FOOTPRINT_WIDTH_MM}",
            f"PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM={PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM}",
            f"PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM={PLACE_GRIPPER_FOOTPRINT_EXTRA_MARGIN_MM}",
            f"PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE={PLACE_OPTIMIZE_ROTATION_FOR_EDGE_CLEARANCE}",
            f"PLACE_ROTATION_CANDIDATE_OFFSETS_DEG={PLACE_ROTATION_CANDIDATE_OFFSETS_DEG}",
            f"USE_PICK_PHI_FOR_PLACE={_pick_one_mod.USE_PICK_PHI_FOR_PLACE}",
            "Lower PAD_* for tighter packing; higher values are more conservative spacing.",
        ],
    )
    _print_knob_group(
        "4. AABB / padding / placed overlap",
        [
            f"PAD_X_MM={PAD_X_MM}",
            f"PAD_Y_MM={PAD_Y_MM}",
            f"PAD_Z_MM={PAD_Z_MM}",
            f"BEST_REJECT_PLACED_OVERLAP={BEST_REJECT_PLACED_OVERLAP}",
            f"BEST_PLACED_OVERLAP_MARGIN_MM={BEST_PLACED_OVERLAP_MARGIN_MM}",
        ],
    )
    _print_knob_group(
        "5. Best-candidate filters",
        [
            f"BEST_REQUIRE_POSITIVE_PLATFORM_XY={BEST_REQUIRE_POSITIVE_PLATFORM_XY}",
            f"platform_min=({BEST_PLATFORM_MIN_X_MM},{BEST_PLATFORM_MIN_Y_MM})",
            f"workspace_x=[{BEST_WORKSPACE_X_MIN_MM},{BEST_WORKSPACE_X_MAX_MM}]",
            f"workspace_y=[{BEST_WORKSPACE_Y_MIN_MM},{BEST_WORKSPACE_Y_MAX_MM}]",
            f"robotframe_centroid_xy_gate={BEST_REQUIRE_ROBOTFRAME_CENTROID_XY_IN_PLATFORM_BOUNDS}",
            f"robotframe_centroid_x=[{BEST_ROBOTFRAME_CENTROID_X_MIN_MM},{BEST_ROBOTFRAME_CENTROID_X_MAX_MM}]",
            f"robotframe_centroid_y=[{BEST_ROBOTFRAME_CENTROID_Y_MIN_MM},{BEST_ROBOTFRAME_CENTROID_Y_MAX_MM}]",
            f"robotframe_centroid_z_min_gate={BEST_REQUIRE_MIN_ROBOTFRAME_CENTROID_Z} z_min={BEST_MIN_ROBOTFRAME_CENTROID_Z_MM}",
            f"BEST_USE_ROBOT_REACH_CHECK={BEST_USE_ROBOT_REACH_CHECK} margin={BEST_ROBOT_REACH_MARGIN_MM}",
            f"BEST_REQUIRE_SOFT_POSE_SAFE={BEST_REQUIRE_SOFT_POSE_SAFE} z={BEST_SOFT_POSE_CHECK_Z_MM}",
            f"BEST_CENTER_GATE_ENABLED={BEST_CENTER_GATE_ENABLED} radius={BEST_MAX_IMAGE_CENTER_NORM_RADIUS}",
            f"BEST_CLUSTER_GATE_ENABLED={BEST_CLUSTER_GATE_ENABLED} dist={BEST_MAX_CLUSTER_DISTANCE_MM}",
            f"volume_mm3=[{BEST_MIN_VOLUME_MM3},{BEST_MAX_VOLUME_CM3 * 1000.0}]",
        ],
    )
    _print_knob_group(
        "6. Missed-pick YOLO watchdog",
        [
            f"MISS_CHECK_ENABLED={MISS_CHECK_ENABLED}",
            f"MISS_CHECK_CAMERA={MISS_CHECK_CAMERA!r}",
            f"MISS_BURST_COUNT={MISS_BURST_COUNT}",
            f"MISS_MIN_HITS={MISS_MIN_HITS}",
            f"MISS_CHECK_TIMEOUT_S={MISS_CHECK_TIMEOUT_S}",
            f"MISS_CHECK_PERIOD_S={MISS_CHECK_PERIOD_S}",
            f"MISS_CONFIRM_AT_PLACE_HOVER_ONLY={MISS_CONFIRM_AT_PLACE_HOVER_ONLY}",
            f"MATCH_MAX_ROBOT_DIST={MISS_MATCH_MAX_ROBOT_DIST_MM} MIN_IOU={MISS_MATCH_MIN_IOU}",
            f"AREA_RATIO=[{MISS_MATCH_AREA_RATIO_MIN},{MISS_MATCH_AREA_RATIO_MAX}]",
            f"SCORE_THRESHOLD={MISS_SCORE_THRESHOLD}",
            f"weights class/dist/iou/area={MISS_CLASS_WEIGHT}/{MISS_ROBOT_DIST_WEIGHT}/{MISS_IOU_WEIGHT}/{MISS_AREA_WEIGHT}",
        ],
    )
    _print_knob_group(
        "7. Recovery pose / J3 rehome",
        [
            f"RECOVERY_POSE=survey_pose unless overridden: {_recovery_pose_xyzphi()}",
            f"RECOVERY_MOVE_TIME_S={RECOVERY_MOVE_TIME_S}",
            f"RECOVERY_REHOME_J3_ENABLED={RECOVERY_REHOME_J3_ENABLED}",
            f"RECOVERY_DROP_Z_BEFORE_REHOME_MM={RECOVERY_DROP_Z_BEFORE_REHOME_MM} (None means survey/recovery Z)",
            f"RECOVERY_REHOME_TIMEOUT_S={RECOVERY_REHOME_TIMEOUT_S}",
        ],
    )
    _print_knob_group(
        "8. Retry grasp offsets",
        [
            f"MAX_PICK_RETRIES_PER_OBJECT={MAX_PICK_RETRIES_PER_OBJECT}",
            f"RETRY_SAFE_PICK_ENABLED={RETRY_SAFE_PICK_ENABLED}",
            f"RETRY_GRASP_Z_OFFSET_MM={RETRY_GRASP_Z_OFFSET_MM}",
            f"RETRY_START_CLAW_EXTRA_DEG={RETRY_START_CLAW_EXTRA_DEG}",
            f"RETRY_START_CLAW_MAX_DEG={RETRY_START_CLAW_MAX_DEG}",
            f"RETRY_XY_SAME_THRESHOLD_MM={RETRY_XY_SAME_THRESHOLD_MM}",
            f"RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY={RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY}",
            f"RETRY_RELOCALIZE_WITH_RAFT_ONLY_AFTER_MISS={RETRY_RELOCALIZE_WITH_RAFT_ONLY_AFTER_MISS}",
        ],
    )
    _print_knob_group(
        "9. Z safety / place release policy",
        [
            f"USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT={USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT}",
            f"PLACE_RELEASE_GAP_MM={PLACE_RELEASE_GAP_MM}",
            f"PLACE_Z_UNCERTAINTY_GAIN={PLACE_Z_UNCERTAINTY_GAIN}",
            f"PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM={PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM}",
            f"PLACE_Z_POLICY_MODE={PLACE_Z_POLICY_MODE!r}",
            f"PLACE_NEGATIVE_BIN_PLATFORM_Z_MM={PLACE_NEGATIVE_BIN_PLATFORM_Z_MM}",
            f"PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM={PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM}",
            f"PLACE_NEGATIVE_BIN_HANG_WEIGHT={PLACE_NEGATIVE_BIN_HANG_WEIGHT}",
            f"PLACE_NEGATIVE_BIN_SIMPLE_WEIGHT={PLACE_NEGATIVE_BIN_SIMPLE_WEIGHT}",
            f"PLACE_NEGATIVE_BIN_CLEARANCE_MM={PLACE_NEGATIVE_BIN_CLEARANCE_MM}",
            f"PLACE_NEGATIVE_BIN_USE_EXISTING_STACK={PLACE_NEGATIVE_BIN_USE_EXISTING_STACK}",
            f"PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING={PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING}",
            f"USE_DYNAMIC_RELEASE_FOR_PLACE={USE_DYNAMIC_RELEASE_FOR_PLACE}",
            f"DYNAMIC_RELEASE_TIMEOUT_S={DYNAMIC_RELEASE_TIMEOUT_S}",
            f"PLACE_CLAW_OPEN_DEG={PLACE_CLAW_OPEN_DEG}",
        ],
    )
    _print_knob_group(
        "10. Camera freshness / burst timing",
        [
            f"OVERHEAD_FRESH_READ_DISCARD_FRAMES={OVERHEAD_FRESH_READ_DISCARD_FRAMES}",
            f"OVERHEAD_FRESH_READ_DELAY_S={OVERHEAD_FRESH_READ_DELAY_S}",
            f"survey BURST_COUNT={_pick_one_mod.BURST_COUNT} MIN_BURST_HITS={_pick_one_mod.MIN_BURST_HITS}",
            "miss watchdog uses YOLO-only burst; RAFT runs once only after confirmed miss",
        ],
    )
    _print_knob_group(
        "11. Motion timing",
        [
            f"COARSE_MOVE_TIME_S={COARSE_MOVE_TIME_S}",
            f"XY_MOVE_TIME_S={XY_MOVE_TIME_S}",
            f"PLACE_Z_MOVE_TIME_S={PLACE_Z_MOVE_TIME_S}",
            f"PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT={PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT}",
            f"PREFETCH_PLACE_DESCENT_DELAY_S={PREFETCH_PLACE_DESCENT_DELAY_S}",
            f"CLEAR_BOX_Z_MM={CLEAR_BOX_Z_MM} CLEAR_BOX_MOVE_TIME_S={CLEAR_BOX_MOVE_TIME_S}",
        ],
    )
    _print_knob_group(
        "12. Display / manual abort keys",
        [
            f"WINDOW={WINDOW!r}",
            f"SELECTION_DISPLAY_HOLD_S={SELECTION_DISPLAY_HOLD_S}",
            f"HOLD_WINDOW_AFTER_RUN={HOLD_WINDOW_AFTER_RUN}",
            "q/ESC aborts; c raises arm and clears bag-state/placed_boxes when feasible",
            f"REQUIRE_CONFIRM_BEFORE_REAL_MOTION={REQUIRE_CONFIRM_BEFORE_REAL_MOTION}",
        ],
    )


def main() -> int:
    _configure_modules()
    _survey_pipeline_mod.OVERHEAD_FRESH_READ_DISCARD_FRAMES = OVERHEAD_FRESH_READ_DISCARD_FRAMES
    _survey_pipeline_mod.OVERHEAD_FRESH_READ_DELAY_S = OVERHEAD_FRESH_READ_DELAY_S
    _pick_one_mod.REQUIRE_CONFIRM_BEFORE_PICK = bool(REQUIRE_CONFIRM_BEFORE_REAL_MOTION)
    _pick_one_mod.REQUIRE_CONFIRM_BEFORE_PLACE = bool(REQUIRE_CONFIRM_BEFORE_REAL_MOTION)
    _hr("AUTONOMOUS PICK PLACE - MISSED PICK RECOVERY", "=")
    print("[MAIN] Experimental clone of scripts/pick_place_two_objects_autonomous.py.")
    print("[MAIN] YOLO-only miss watchdog runs before any RAFT recovery localization.")
    print_knob_overview()
    print_z_safety_settings("[MAIN] Z safety", config=DEFAULT_Z_SAFETY)
    print("[MAIN] camera window: q=quit, c=clear box/reset stack. No s key is needed.")

    selector_config = _best_candidate_config()
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
    overhead = SimpleOverheadCamera(OVERHEAD_INDEX)
    stereo = SimpleStereoCamera(STEREO_INDEX)
    robot = startup_robot()

    surface_zone = _load_place_surface_zone()
    surface_z = float(surface_zone["surface_z_mm"])
    base_xy = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    place_phi = float(surface_zone["default_phi_deg"])

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + STATUS_H_PX)

    placed_boxes: list = []
    state: SurveyState | None = None
    held: CandidateDebug | None = None
    column_xy_primary: np.ndarray | None = None
    column_xy_secondary: np.ndarray | None = None
    status_lines = ["AUTO: starting up", "The first survey will begin automatically.", "q=quit | c=clear box"]

    camera_lock = threading.Lock()
    survey_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera_work")
    prefetched_future: Future | None = None

    target_limit = int(MAX_OBJECT_COUNT_SAFETY if RUN_UNTIL_NO_VALID_CANDIDATE else max(1, TARGET_OBJECT_COUNT))
    continuous_after_clear = False
    run_ok = True

    def set_status(lines: list[str]) -> None:
        nonlocal status_lines
        status_lines = list(lines)

    def run_until_no_valid_active() -> bool:
        return bool(RUN_UNTIL_NO_VALID_CANDIDATE or continuous_after_clear)

    def flow_label(object_i: int) -> str:
        if run_until_no_valid_active():
            return f"object {object_i}/continuous"
        return f"object {object_i}/{target_limit}"

    def compute_candidate_place_target(
        cand_dbg: CandidateDebug,
        *,
        object_i: int,
    ) -> OptimizedPlaceTarget:
        raw_box = _aabb_from_object_candidate_quiet(cand_dbg.candidate, default_label=f"object{object_i}_overlay")
        if object_i == 1:
            target_xy = np.asarray(base_xy, dtype=np.float64).reshape(2).copy()
            target_phi = float(place_phi)
            return _optimize_place_target_for_slot(
                cand_dbg,
                target_xy_mm=target_xy,
                base_phi_deg=target_phi,
                surface_zone=surface_zone,
                label=f"object{object_i}_base",
                verbose=False,
            )
        if object_i == 2:
            if not placed_boxes:
                raise ValueError("waiting_for_reference_box")
            moving_padded = pad_aabb(raw_box, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
            adjacent_plan = compute_adjacent_placement(
                reference_padded_box=placed_boxes[-1],
                moving_padded_box=moving_padded,
                direction=ADJACENT_DIRECTION,
                surface_z_mm=surface_z,
                place_phi_deg=place_phi,
            )
            return _optimize_place_target_for_slot(
                cand_dbg,
                target_xy_mm=adjacent_plan.target_center_xy_mm.copy(),
                base_phi_deg=float(adjacent_plan.target_phi_deg),
                surface_zone=surface_zone,
                label=f"object{object_i}_adjacent",
                verbose=False,
            )

        if column_xy_primary is None or column_xy_secondary is None:
            raise ValueError("waiting_for_column_anchor")
        target_xy = column_xy_primary if object_i % 2 == 1 else column_xy_secondary
        return _optimize_place_target_for_slot(
            cand_dbg,
            target_xy_mm=np.asarray(target_xy, dtype=np.float64).reshape(2).copy(),
            base_phi_deg=float(place_phi),
            surface_zone=surface_zone,
            label=f"object{object_i}_stack",
            verbose=False,
        )

    def build_placeability_overlay(
        display_state: SurveyState | None,
        *,
        next_object_i: int | None = None,
    ) -> dict[int, PlaceabilityOverlayEntry] | None:
        if display_state is None or not display_state.candidates:
            return None

        object_i = int(next_object_i or max(1, len(placed_boxes) + 1))
        overlay: dict[int, PlaceabilityOverlayEntry] = {}
        for idx, cand_dbg in enumerate(display_state.candidates):
            try:
                optimized = compute_candidate_place_target(cand_dbg, object_i=object_i)
                overlay[idx] = PlaceabilityOverlayEntry(
                    can_place=bool(optimized.can_place),
                    target_xy_mm=optimized.target_xy_mm.copy(),
                    target_phi_deg=float(optimized.target_phi_deg),
                    clearance_mm=float(optimized.fit_clearance_mm),
                    reason=str(optimized.reason),
                )
            except Exception as exc:
                overlay[idx] = PlaceabilityOverlayEntry(
                    can_place=False,
                    target_xy_mm=None,
                    target_phi_deg=None,
                    clearance_mm=None,
                    reason=str(exc),
                )
        return overlay

    def handle_clear_box_request() -> None:
        nonlocal placed_boxes, state, held, column_xy_primary, column_xy_secondary, prefetched_future
        print("[CLEAR BOX] requested from camera window.")
        set_status([
            "CLEAR BOX: raising arm",
            "Empty the box when motion completes.",
            "Then press Enter in the terminal.",
            "After clear: continuous mode, no object count limit.",
        ])
        _show_display(
            state=state,
            overhead=overhead,
            stereo=stereo,
            status_lines=status_lines,
            read_live=True,
            placeability_overlay=build_placeability_overlay(state),
        )

        if prefetched_future is not None and not prefetched_future.done():
            print("[CLEAR BOX] canceling queued survey before reset.")
            if not prefetched_future.cancel():
                print("[CLEAR BOX] survey is already running; waiting for it to release the cameras.")
                try:
                    prefetched_future.result()
                except Exception as exc:
                    print(f"[CLEAR BOX] background survey finished with warning: {exc}")
        prefetched_future = None

        clear_z = float(CLEAR_BOX_Z_MM)
        reason = validate_z_command(clear_z, "[CLEAR BOX] raise", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[CLEAR BOX] WARN: configured clear Z rejected ({reason}); using Z_MAX_MM={Z_MAX_MM:.1f}")
            clear_z = float(Z_MAX_MM)

        print(f"[CLEAR BOX] raising to z={clear_z:.1f} mm. Keep hands clear until motion stops.")
        if not robot.move_cartesian(z_mm=clear_z, move_time_s=float(CLEAR_BOX_MOVE_TIME_S)):
            print("[CLEAR BOX] WARN: raise command failed; still waiting for operator confirmation.")

        set_status([
            "CLEAR BOX: arm raised",
            "Empty the box now.",
            "Press Enter in the terminal to restart.",
            "After clear: continuous mode, no object count limit.",
        ])
        _show_display(
            state=state,
            overhead=overhead,
            stereo=stereo,
            status_lines=status_lines,
            read_live=True,
            placeability_overlay=build_placeability_overlay(state),
        )
        input("[CLEAR BOX] Empty the box, then press Enter to restart autonomous picking...")

        placed_boxes = []
        state = None
        held = None
        column_xy_primary = None
        column_xy_secondary = None
        print("[CLEAR BOX] placement occupancy reset. Continuing in continuous mode until no valid candidate.")
        set_status([
            "CLEAR BOX: reset complete",
            "Assuming the box is empty.",
            "Continuous mode is active; no object count limit.",
            "Survey will restart now.",
        ])

    def run_survey_now(label: str) -> SurveyState:
        nonlocal state
        print(f"[SURVEY AUTO] starting blocking survey for {label}")
        set_status([f"SURVEY: {label}", "Capturing burst, segmenting, computing disparity...", "Selection will use largest valid volume.", "q=quit | c=clear box after current operation"])
        _show_display(
            state=state,
            overhead=overhead,
            stereo=stereo,
            status_lines=status_lines,
            read_live=True,
            placeability_overlay=build_placeability_overlay(state),
        )
        with camera_lock:
            surveyed = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
        state = surveyed
        print(f"[SURVEY AUTO] completed blocking survey for {label}: candidates={len(surveyed.candidates)}")
        return surveyed

    def prefetch_worker(next_object_i: int) -> SurveyState:
        if PREFETCH_PLACE_DESCENT_DELAY_S > 0.0:
            time.sleep(float(PREFETCH_PLACE_DESCENT_DELAY_S))
        print(f"[PREFETCH] object {next_object_i}: running survey in background from place descent")
        with camera_lock:
            surveyed = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
        print(f"[PREFETCH] object {next_object_i}: survey done candidates={len(surveyed.candidates)}")
        return surveyed

    def start_prefetch_on_place_descent(next_object_i: int) -> None:
        nonlocal prefetched_future
        if not PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT:
            print("[PREFETCH] disabled")
            return
        if prefetched_future is not None and not prefetched_future.done():
            print("[PREFETCH] already running; not submitting another survey")
            return
        set_status([
            f"PREFETCH: object {next_object_i} survey queued",
            "Started at the beginning of place descent.",
            "Place the next object on the platform now.",
            "Selection will be audited after this survey finishes.",
        ])
        print(f"[PREFETCH] submitting next survey for object {next_object_i} at place descent")
        prefetched_future = survey_executor.submit(prefetch_worker, next_object_i)

    def get_survey_for_object(object_i: int) -> SurveyState:
        nonlocal state, prefetched_future
        if prefetched_future is None:
            return run_survey_now(flow_label(object_i))

        set_status([
            f"WAITING: prefetched survey for object {object_i}",
            "Camera reads are owned by the survey thread.",
            "q=quit | c=clear box",
        ])
        while not prefetched_future.done():
            _show_display(
                state=state,
                overhead=overhead,
                stereo=stereo,
                status_lines=status_lines,
                read_live=False,
                placeability_overlay=build_placeability_overlay(state, next_object_i=object_i),
            )
            time.sleep(0.05)

        try:
            state = prefetched_future.result()
            print(f"[FLOW] using place-descent prefetched survey for object {object_i}")
            return state
        except Exception as exc:
            print(f"[FLOW WARN] prefetched survey failed for object {object_i}: {exc}")
            return run_survey_now(f"{flow_label(object_i)} fallback")
        finally:
            prefetched_future = None

    def wait_for_resume_or_quit_after_no_candidate(object_i: int) -> str:
        mode = str(NO_CANDIDATE_AFTER_RETRIES_MODE).strip().lower()
        if mode == "continuous":
            print("[FLOW] no valid candidate after retries; continuous survey mode will keep looking until q/ESC.")
            return "resume"
        if mode == "stop":
            print("[FLOW] no valid candidate after retries; mode=stop.")
            return "stop"

        resume_key = str(NO_CANDIDATE_RESUME_KEY or "r").lower()[:1]
        print(
            f"[FLOW] no valid candidate for object {object_i} after retries. "
            f"Press {resume_key!r} in the camera window to resume surveying, or q/ESC to quit."
        )
        set_status([
            "NO VALID CANDIDATE after retries",
            f"Press {resume_key.upper()} to resume automation.",
            "Press q/ESC to stop.",
            "Place/adjust the grocery on the platform.",
        ])
        while True:
            live_overhead, live_left, live_right = _read_live_frames(overhead, stereo)
            cv2.imshow(WINDOW, _make_autonomous_display(state, live_overhead, live_left, live_right, status_lines))
            key = read_command_key(delay_ms=25)
            if key in ("q", "escape", "\x1b"):
                raise UserAbort("quit requested")
            if key == "c":
                raise ClearBoxRequest("clear box requested")
            if key == resume_key:
                print("[FLOW] resume requested; surveying again.")
                return "resume"
            time.sleep(0.05)

    def _no_miss_result(reason: str) -> MissMatchResult:
        return MissMatchResult(False, 0.0, 0, 0, None, None, None, None, None, False, reason)

    def start_miss_watchdog(attempt: PickAttemptRecord) -> Future | None:
        if not MISS_CHECK_ENABLED:
            print("[MISS CHECK] disabled; RAFT recovery path will not run.")
            return None
        set_status([
            "MISS CHECK: collecting YOLO burst",
            f"object {attempt.object_i} attempt {attempt.attempt_number}",
            "RAFT is skipped unless this confirms a miss.",
            "q=quit | c=clear box",
        ])
        print("[MISS CHECK] submitting YOLO-only watchdog to the single camera executor.")
        return survey_executor.submit(
            run_yolo_only_miss_check,
            attempt=attempt,
            overhead=overhead,
            stereo=stereo,
            yolo=yolo,
            bundle=bundle,
            camera_lock=camera_lock,
        )

    def wait_for_miss_watchdog(future: Future | None, attempt: PickAttemptRecord) -> MissMatchResult:
        if future is None:
            return _no_miss_result("miss check disabled")
        set_status([
            "MISS CHECK: confirming at place hover",
            "Waiting for YOLO-only burst result.",
            "No release descent until the watchdog clears.",
            "q=quit | c=clear box",
        ])
        while not future.done():
            _show_display(
                state=state,
                overhead=overhead,
                stereo=stereo,
                status_lines=status_lines,
                read_live=False,
                placeability_overlay=build_placeability_overlay(state, next_object_i=attempt.object_i),
            )
            time.sleep(0.03)
        result = future.result()
        print("[MISS CHECK AUDIT]")
        print(
            f"object={attempt.object_i} attempt={attempt.attempt_number} "
            f"is_miss={result.is_miss} score={result.score:.3f} hits={result.hits}/{result.burst_count}"
        )
        print(
            f"components: same_class={result.same_class} "
            f"robot_dist={_fmt_optional(result.robot_dist_mm, 1)} "
            f"iou={_fmt_optional(result.iou)} area_ratio={_fmt_optional(result.area_ratio)}"
        )
        print(f"reason={result.reason}")
        print("[MISS CHECK AUDIT] RAFT skipped." if not result.is_miss else "[MISS CHECK AUDIT] RAFT will run once during recovery.")
        return result

    def place_or_detect_miss(
        *,
        held_object: CandidateDebug,
        attempt: PickAttemptRecord,
        target_xy: np.ndarray,
        target_phi: float,
        destination_surface_for_call: float,
        place_descent_cb: Callable[[], None] | None,
    ) -> tuple[str, MissMatchResult | None]:
        miss_future = start_miss_watchdog(attempt)
        prepared = _prepare_place_at_target(
            robot,
            held_object,
            target_xy_mm=target_xy,
            target_phi_deg=target_phi,
            destination_surface_z_mm=destination_surface_for_call,
        )
        if prepared is None:
            if miss_future is not None and not miss_future.done():
                miss_future.cancel()
            return "failed", None

        miss_result = wait_for_miss_watchdog(miss_future, attempt)
        if miss_result.is_miss and MISS_CONFIRM_AT_PLACE_HOVER_ONLY:
            print("[MISS CHECK] preliminary burst says miss; running YOLO-only hover confirmation before recovery.")
            set_status([
                "MISS CHECK: hover confirmation",
                "Preliminary burst saw the object at pick site.",
                "Confirming with YOLO only; RAFT still skipped.",
                "q=quit | c=clear box",
            ])
            hover_result = run_yolo_only_miss_check(
                attempt=attempt,
                overhead=overhead,
                stereo=stereo,
                yolo=yolo,
                bundle=bundle,
                camera_lock=camera_lock,
            )
            print(
                f"[MISS CHECK HOVER AUDIT] is_miss={hover_result.is_miss} "
                f"score={hover_result.score:.3f} hits={hover_result.hits}/{hover_result.burst_count} "
                f"reason={hover_result.reason}"
            )
            miss_result = hover_result
        if miss_result.is_miss:
            print("[MISS ABORT] confirmed before release descent. Canceling pending place sequence.")
            set_status([
                "MISS DETECTED: object still at pick site",
                "Release descent canceled.",
                "Moving to recovery/recalibration pose.",
                "RAFT will run once after recovery.",
            ])
            _raise_or_hold_safe_z(robot, "[MISS ABORT] hold at safe hover", float(prepared.place_plan.approach_z_mm), COARSE_MOVE_TIME_S)
            return "miss", miss_result

        set_status([
            "MISS CHECK: clear",
            "Proceeding to place descent and release.",
            "Next survey may prefetch during descent.",
            "q=quit | c=clear box",
        ])
        if not _finish_place_from_hover(robot, prepared, on_start_place_descent=place_descent_cb):
            return "failed", miss_result
        return "placed", miss_result

    try:
        _show_display(
            state=state,
            overhead=overhead,
            stereo=stereo,
            status_lines=status_lines,
            read_live=True,
            placeability_overlay=build_placeability_overlay(state),
        )

        i = 1
        while run_until_no_valid_active() or i <= target_limit:
            try:
                print(f"\n[FLOW] Object {i}/{'continuous' if run_until_no_valid_active() else target_limit}")
                survey_state = get_survey_for_object(i)

                if EFFICIENT_PACKING_ENABLED:
                    selection, _slot_overlay = _select_best_for_state_by_slot_fit(
                        survey_state,
                        object_i=i,
                        robot=robot,
                        placed_boxes=placed_boxes,
                        config=selector_config,
                        surface_zone=surface_zone,
                        base_xy=base_xy,
                        base_phi_deg=place_phi,
                        column_xy_primary=column_xy_primary,
                        column_xy_secondary=column_xy_secondary,
                        target_limit=target_limit,
                    )
                else:
                    selection = _select_best_for_state(
                        survey_state,
                        robot=robot,
                        placed_boxes=placed_boxes,
                        config=selector_config,
                    )

                retry_i = 0
                while selection.selected is None and retry_i < int(NO_CANDIDATE_RETRY_COUNT):
                    retry_i += 1
                    print(f"[FLOW] no valid candidate for object {i}; retry {retry_i}/{NO_CANDIDATE_RETRY_COUNT}")
                    set_status([
                        f"NO VALID CANDIDATE for object {i}",
                        f"Retrying survey in {NO_CANDIDATE_RETRY_DELAY_S:.1f}s.",
                        "Check terminal for exact rejection reasons.",
                        "q=quit | c=clear box",
                    ])
                    deadline = time.time() + float(NO_CANDIDATE_RETRY_DELAY_S)
                    while time.time() < deadline:
                        _show_display(
                            state=survey_state,
                            overhead=overhead,
                            stereo=stereo,
                            status_lines=status_lines,
                            read_live=True,
                            placeability_overlay=build_placeability_overlay(survey_state, next_object_i=i),
                        )
                        time.sleep(0.05)
                    survey_state = run_survey_now(f"{flow_label(i)} retry {retry_i}")
                    if EFFICIENT_PACKING_ENABLED:
                        selection, _slot_overlay = _select_best_for_state_by_slot_fit(
                            survey_state,
                            object_i=i,
                            robot=robot,
                            placed_boxes=placed_boxes,
                            config=selector_config,
                            surface_zone=surface_zone,
                            base_xy=base_xy,
                            base_phi_deg=place_phi,
                            column_xy_primary=column_xy_primary,
                            column_xy_secondary=column_xy_secondary,
                            target_limit=target_limit,
                        )
                    else:
                        selection = _select_best_for_state(
                            survey_state,
                            robot=robot,
                            placed_boxes=placed_boxes,
                            config=selector_config,
                        )

                set_status(selection.display_lines(max_lines=5))
                _show_display(
                    state=survey_state,
                    overhead=overhead,
                    stereo=stereo,
                    status_lines=status_lines,
                    read_live=False,
                    placeability_overlay=build_placeability_overlay(survey_state, next_object_i=i),
                )
                if SELECTION_DISPLAY_HOLD_S > 0.0:
                    time.sleep(float(SELECTION_DISPLAY_HOLD_S))

                if selection.selected is None:
                    print(f"[FLOW] no valid candidate for object {i} after {NO_CANDIDATE_RETRY_COUNT + 1} survey attempt(s)")
                    no_candidate_action = wait_for_resume_or_quit_after_no_candidate(i)
                    if no_candidate_action == "resume":
                        if NO_CANDIDATE_RECOVERY_MOVE_ENABLED:
                            print("[FLOW] no candidate: retracting to clear-box Z before survey/home move.")
                            _raise_or_hold_safe_z(robot, "[FLOW] pre-recovery retract", CLEAR_BOX_Z_MM, CLEAR_BOX_MOVE_TIME_S)
                            print("[FLOW] no candidate: moving to survey/home pose.")
                            _move_to_recovery_pose(robot)
                            _rehome_j3_if_requested(robot)
                        continue
                    print(f"[FLOW] stopping: no valid candidate for object {i}")
                    if run_until_no_valid_active() and str(NO_CANDIDATE_AFTER_RETRIES_MODE).strip().lower() != "stop":
                        break
                    run_ok = False
                    break

                attempt: PickAttemptRecord | None = None
                held = None
                max_local_pick_attempts = max(1, min(5, len(survey_state.candidates)))
                excluded_pick_indices: set[int] = set()

                for local_pick_try in range(1, max_local_pick_attempts + 1):
                    if selection.selected is None:
                        break

                    cand_dbg = selection.selected
                    c = cand_dbg.candidate
                    c_idx = int(getattr(c, "index", -1))
                    if c_idx in excluded_pick_indices:
                        continue

                    print(
                        f"[FLOW] picking object {i}: candidate [{c.index}] {c.yolo.class_name} "
                        f"xy=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f}) "
                        f"local_try={local_pick_try}/{max_local_pick_attempts}"
                    )

                    attempt = make_pick_attempt_record(object_i=i, cand_dbg=cand_dbg, robot=robot, attempt_number=1)
                    if execute_pick_selected(robot, cand_dbg, bundle=bundle):
                        held = cand_dbg
                        break

                    excluded_pick_indices.add(c_idx)
                    print(
                        f"[FLOW WARN] object {i} pick failed for candidate [{c.index}] {c.yolo.class_name}; "
                        "trying next best candidate."
                    )

                    survey_state.candidates = [
                        dbg
                        for dbg in survey_state.candidates
                        if int(getattr(dbg.candidate, "index", -1)) not in excluded_pick_indices
                    ]
                    if not survey_state.candidates:
                        selection.selected = None
                        selection.selected_decision = None
                        break

                    if EFFICIENT_PACKING_ENABLED:
                        selection, _slot_overlay = _select_best_for_state_by_slot_fit(
                            survey_state,
                            object_i=i,
                            robot=robot,
                            placed_boxes=placed_boxes,
                            config=selector_config,
                            surface_zone=surface_zone,
                            base_xy=base_xy,
                            base_phi_deg=place_phi,
                            column_xy_primary=column_xy_primary,
                            column_xy_secondary=column_xy_secondary,
                            target_limit=target_limit,
                        )
                    else:
                        selection = _select_best_for_state(
                            survey_state,
                            robot=robot,
                            placed_boxes=placed_boxes,
                            config=selector_config,
                        )

                if held is None or attempt is None:
                    print(f"[FLOW] object {i} pick failed for all local candidates; re-surveying current slot.")
                    continue

                raw_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=f"object{i}")
                object_height_mm = float(raw_box.size_xyz_mm[2])

                destination_surface_for_call = float(surface_z)
                below_top_z_mm: float | None = None

                optimized_target = compute_candidate_place_target(cand_dbg, object_i=i)
                target_xy = optimized_target.target_xy_mm.copy()
                target_phi = float(optimized_target.target_phi_deg)
                if i == 1:
                    column_xy_primary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                elif i == 2:
                    column_xy_secondary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                print(
                    f"[FLOW] object{i} optimized target xy=({target_xy[0]:.1f},{target_xy[1]:.1f}) "
                    f"phi={target_phi:.1f} nudge=({optimized_target.nudge_xy_mm[0]:.1f},{optimized_target.nudge_xy_mm[1]:.1f}) "
                    f"fit_clearance={optimized_target.fit_clearance_mm:.1f} can_place={optimized_target.can_place}"
                )
                if i >= 3 and len(placed_boxes) >= 1:
                    below_idx = i - 2
                    below_box = placed_boxes[below_idx - 1]
                    below_top_z_mm = float(below_box.raw_box.max_xyz_mm[2])
                    destination_surface_for_call = float(below_top_z_mm)
                    print(
                        f"[FLOW] object{i} stacking target (2-per-layer): "
                        f"xy=({target_xy[0]:.1f},{target_xy[1]:.1f}) "
                        f"below_object={below_idx} below_top_z_mm={below_top_z_mm:.1f} "
                        "release_z=shared_policy(surface=below_top_z)"
                    )

                next_i = i + 1
                should_prefetch_next = bool(PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT) and (
                    run_until_no_valid_active() or next_i <= target_limit
                )
                place_descent_cb = (
                    (lambda next_object_i=next_i: start_prefetch_on_place_descent(next_object_i))
                    if should_prefetch_next
                    else None
                )

                target_phi = _choose_place_phi_for_edge_clearance(
                    held,
                    target_xy_mm=target_xy,
                    base_phi_deg=target_phi,
                    surface_zone=surface_zone,
                    label=f"object{i}",
                )

                if not _validate_gripper_footprint_inside_bag(
                    held,
                    target_xy_mm=target_xy,
                    target_phi_deg=target_phi,
                    surface_zone=surface_zone,
                    label=f"object{i}",
                ):
                    held = None
                    print("[FLOW] placement footprint gate failed; re-surveying current slot.")
                    continue

                place_status, miss_result = place_or_detect_miss(
                    held_object=held,
                    attempt=attempt,
                    target_xy=target_xy,
                    target_phi=target_phi,
                    destination_surface_for_call=destination_surface_for_call,
                    place_descent_cb=place_descent_cb,
                )

                if place_status == "miss":
                    print("[FLOW] confirmed missed pick; no placed_boxes update will be made for this attempt.")
                    if int(MAX_PICK_RETRIES_PER_OBJECT) <= 0:
                        print("[FLOW] retries disabled; stopping after confirmed miss.")
                        held = None
                        run_ok = False
                        break

                    recovered = recover_target_after_confirmed_miss(
                        attempt=attempt,
                        miss_result=miss_result or _no_miss_result("missing miss result"),
                        overhead=overhead,
                        stereo=stereo,
                        detector=detector,
                        stereo_calib=stereo_calib,
                        rectifier=rectifier,
                        yolo=yolo,
                        raft=raft,
                        robot=robot,
                        bundle=bundle,
                        camera_lock=camera_lock,
                    )
                    if recovered is None:
                        print("[FLOW] recovery failed after confirmed miss.")
                        held = None
                        if str(ON_RETRY_FAIL).lower() == "skip":
                            print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot; remove/adjust the failed item if needed.")
                            continue
                        run_ok = False
                        break

                    retry_ok, retry_held = execute_retry_pick_selected(
                        robot,
                        recovered,
                        original_attempt=attempt,
                        bundle=bundle,
                    )
                    if not retry_ok or retry_held is None:
                        print("[FLOW] retry pick failed after recovery.")
                        held = None
                        if str(ON_RETRY_FAIL).lower() == "skip":
                            print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot; failed candidate was not placed.")
                            continue
                        run_ok = False
                        break

                    held = retry_held
                    retry_attempt = make_pick_attempt_record(object_i=i, cand_dbg=retry_held, robot=robot, attempt_number=2)
                    retry_raw_box = _aabb_from_object_candidate_quiet(retry_held.candidate, default_label=f"object{i}_retry")
                    retry_object_height_mm = float(retry_raw_box.size_xyz_mm[2])

                    if not _validate_gripper_footprint_inside_bag(
                        held,
                        target_xy_mm=target_xy,
                        target_phi_deg=target_phi,
                        surface_zone=surface_zone,
                        label=f"object{i}_retry",
                    ):
                        held = None
                        if str(ON_RETRY_FAIL).lower() == "skip":
                            print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot after retry footprint rejection.")
                            continue
                        run_ok = False
                        break

                    retry_place_status, _retry_miss_result = place_or_detect_miss(
                        held_object=held,
                        attempt=retry_attempt,
                        target_xy=target_xy,
                        target_phi=target_phi,
                        destination_surface_for_call=destination_surface_for_call,
                        place_descent_cb=place_descent_cb,
                    )
                    if retry_place_status == "miss":
                        print("[FLOW] retry also appears missed; no placed_boxes update will be made.")
                        held = None
                        if str(ON_RETRY_FAIL).lower() == "skip":
                            print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot after retry miss.")
                            continue
                        run_ok = False
                        break
                    if retry_place_status != "placed":
                        print("[FLOW] retry place failed.")
                        held = None
                        if str(ON_RETRY_FAIL).lower() == "skip":
                            print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot after retry place failure.")
                            continue
                        run_ok = False
                        break

                    print("PLACE: retry succeeded")
                    raw_box = retry_raw_box
                    object_height_mm = retry_object_height_mm
                elif place_status != "placed":
                    print(f"[FLOW] object {i} place failed")
                    held = None
                    if str(ON_RETRY_FAIL).lower() == "skip":
                        print("[FLOW] ON_RETRY_FAIL='skip': re-surveying this bag slot after place failure.")
                        continue
                    run_ok = False
                    break

                if i == 1:
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array([float(target_xy[0]), float(target_xy[1]), surface_z + 0.5 * object_height_mm], dtype=np.float64),
                        size_xyz_mm=raw_box.size_xyz_mm,
                        label="object1_placed",
                    )
                    placed_boxes.append(placed)
                elif i == 2:
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array(
                            [float(target_xy[0]), float(target_xy[1]), surface_z + 0.5 * object_height_mm],
                            dtype=np.float64,
                        ),
                        size_xyz_mm=raw_box.size_xyz_mm,
                        label="object2_placed",
                    )
                    placed_boxes.append(placed)
                else:
                    if below_top_z_mm is None:
                        print(f"[FLOW] missing below_top_z_mm for object {i}")
                        run_ok = False
                        break
                    stack_center_z = below_top_z_mm + 0.5 * object_height_mm
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array([float(target_xy[0]), float(target_xy[1]), stack_center_z], dtype=np.float64),
                        size_xyz_mm=raw_box.size_xyz_mm,
                        label=f"object{i}_stacked",
                    )
                    placed_boxes.append(placed)

                held = None
                print(f"[FLOW] placed_boxes count = {len(placed_boxes)}")
                i += 1
            except ClearBoxRequest:
                handle_clear_box_request()
                continuous_after_clear = True
                i = 1
                continue

        if run_ok:
            if run_until_no_valid_active():
                print(f"[FLOW] complete: placed {len(placed_boxes)} object(s); stopped because no valid candidate was available.")
            elif len(placed_boxes) >= target_limit:
                print(f"[FLOW] success: placed {len(placed_boxes)} objects")
            else:
                print(f"[FLOW] incomplete: placed {len(placed_boxes)}/{target_limit} objects")
                run_ok = False

        set_status([
            f"DONE: placed {len(placed_boxes)} object(s)" if run_ok else f"STOPPED: placed {len(placed_boxes)} object(s)",
            "Check terminal for full candidate audit trail.",
            "q=quit | c=clear box",
        ])

        if HOLD_WINDOW_AFTER_RUN:
            print("[MAIN] run ended. Press q in the camera window to close.")
            while True:
                _show_display(
                    state=state,
                    overhead=overhead,
                    stereo=stereo,
                    status_lines=status_lines,
                    read_live=True,
                    placeability_overlay=build_placeability_overlay(state),
                )
                time.sleep(0.05)

    except UserAbort as exc:
        print(f"[MAIN] {exc}")
        run_ok = False
    finally:
        try:
            if prefetched_future is not None and not prefetched_future.done():
                prefetched_future.cancel()
        except Exception:
            pass
        try:
            survey_executor.shutdown(wait=False)
        except Exception:
            pass
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

    return 0 if run_ok else 1


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
