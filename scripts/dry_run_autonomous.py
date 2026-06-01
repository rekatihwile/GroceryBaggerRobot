from __future__ import annotations

"""scripts/dry_run_autonomous.py

Hardware-free dry run of the full autonomous pick-place pipeline.

What it does
------------
1. Finds stereo image pairs in Training_Images/ (same path as
   random_yolo_raft_pointcloud_validation.py).
2. Loads YOLO + RAFT models from the same paths as the wet run.
3. For each pair, runs the IDENTICAL vision pipeline that run_survey() uses
   (YOLO seg → RAFT disparity → masked point cloud → build_object_candidate
   → Z resolver → phi resolver → XY blend).
4. Runs the IDENTICAL candidate selector (autonomous_best_candidate).
5. Runs the IDENTICAL placement planner (adjacent_placement, slot fit, AABB
   padding).
6. Prints a full simulation trace — "would pick X, place at Y" — without
   sending any commands to hardware.

Shared config pot
-----------------
Import from config/ dataclasses and override here to change behaviour in BOTH
this dry run and autonomous_missed_pick_recovery.py simultaneously:

    from config.survey.survey_config import SurveyConfig
    _SURVEY = SurveyConfig(YOLO_CONF=0.50)   ← applies everywhere

Run
---
    python scripts/dry_run_autonomous.py
    python scripts/dry_run_autonomous.py --images path/to/other/images
    python scripts/dry_run_autonomous.py --index 3   # specific pair index
"""

# ============================================================
# USER SETTINGS  ← same knobs as autonomous_missed_pick_recovery.py
# ============================================================

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.survey.survey_config import DEFAULT_SURVEY, SurveyConfig
from config.pick.pick_config import DEFAULT_PICK, PickConfig
from config.place.place_config import DEFAULT_PLACE, PlaceConfig
from config.place.place_scene_config import load_place_scene
from config.pick.servo_config import DEFAULT_SERVO, ServoConfig
from config.runtime_context import resolve_runtime_context, resolve_workspace_profile_name
from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from config.motion.z_safety_config import DEFAULT_Z_SAFETY

# Override individual fields here without touching the shared defaults:
_SURVEY = DEFAULT_SURVEY
_PICK   = DEFAULT_PICK
_PLACE  = DEFAULT_PLACE
_SERVO  = DEFAULT_SERVO
_RUNTIME_CONTEXT = resolve_runtime_context("saved_photo_test")
_WORKSPACE = get_workspace_filter_config(resolve_workspace_profile_name(_RUNTIME_CONTEXT.workspace_profile_name))

# Candidate-filter workspace bounds.
PLATFORM_GRID_X_MM = [float(v) for v in _WORKSPACE.platform_grid_x_mm]
PLATFORM_GRID_Y_MM = [float(v) for v in _WORKSPACE.platform_grid_y_mm]
PLATFORM_X_MIN_MM = float(min(PLATFORM_GRID_X_MM))
PLATFORM_X_MAX_MM = float(max(PLATFORM_GRID_X_MM))
PLATFORM_Y_MIN_MM = float(min(PLATFORM_GRID_Y_MM))
PLATFORM_Y_MAX_MM = float(max(PLATFORM_GRID_Y_MM))

TARGET_OBJECT_COUNT = 10   # how many objects to simulate packing

# Set True when Training_Images/ pairs are already rectified (default from capture script).
IMAGES_ALREADY_RECTIFIED = True

# ============================================================

import argparse
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import masked_disparity_to_pointcloud, cam_points_to_robot_xyz
from vision.pick_z_resolver import resolve_robust_object_z
from vision.pick_phi_resolver import resolve_pick_phi
from vision.pick_xy_resolver import apply_xy_blend
from vision.pick_candidate_builder import (
    CandidateDebug, SurveyState,
    colorize_disparity, overlay_detection_on_image,
)
from planning.aabb_utils import aabb_from_object_candidate, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement
from scripts.autonomous_best_candidate import BestCandidateConfig, choose_best_candidate


# ── Null robot stub ────────────────────────────────────────────────────────
# build_object_candidate → refresh_candidate_z_bias always calls robot.fk().
# With stereo_tags={} (no AprilTag EE), the bias correction is skipped and only
# fk_xyz_at_update / fk_phi_at_update metadata is set — harmless defaults.

class _NullRobot:
    def fk(self):
        return 0.0, 0.0, 0.0, 0.0
    def check_cartesian_pose_safe(self, x, y, z):
        return True, "no_robot"
    @property
    def cfg(self):
        return None


# ── Mock types that satisfy build_candidate_from_track() interface ─────────
# The real pipeline uses BurstFrame / DetectionTrack from vision/burst_tracking.
# The dry run builds equivalent objects from a single frozen image pair so the
# downstream build_object_candidate() call is identical to the wet run.

@dataclass
class _MockObservation:
    frame_i: int
    detection: Any


@dataclass
class _MockTrack:
    track_id: int
    class_name: str
    hit_count: int = 1
    _obs: Any = None

    def best_observation(self) -> _MockObservation:
        return self._obs


@dataclass
class _MockBurstFrame:
    frame_i: int
    left_rect: np.ndarray
    right_rect: np.ndarray
    stereo_tags: list = field(default_factory=list)


# ── Stereo pair discovery ──────────────────────────────────────────────────

@dataclass
class StereoPair:
    index: int
    left_path: Path
    right_path: Path


def find_stereo_pairs(training_dir: Path) -> list[StereoPair]:
    left_re  = re.compile(r"Stereo_Left_(\d+)\.(jpg|jpeg|png)$",  re.IGNORECASE)
    right_re = re.compile(r"Stereo_Right_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    lefts, rights = {}, {}
    for p in training_dir.iterdir():
        m = left_re.match(p.name)
        if m:
            lefts[int(m.group(1))] = p; continue
        m = right_re.match(p.name)
        if m:
            rights[int(m.group(1))] = p
    common = sorted(set(lefts) & set(rights))
    if not common:
        raise FileNotFoundError(
            f"No Stereo_Left_#### / Stereo_Right_#### pairs found in {training_dir}"
        )
    return [StereoPair(i, lefts[i], rights[i]) for i in common]


# ── Vision-module configuration ────────────────────────────────────────────

def _configure_vision_modules() -> None:
    """Push config values into vision module globals — same as pick_one_place_one._configure_modules()."""
    import vision.pointcloud as _pc
    import vision.object_geometry as _geom
    import vision.burst_tracking as _burst
    import vision.pick_candidate_builder as _cand
    import vision.pick_z_resolver as _z
    import vision.pick_xy_resolver as _xy
    import vision.pick_phi_resolver as _phi
    import vision.pick_survey_pipeline as _survey

    _pc.MIN_DISPARITY_PX = _SURVEY.MIN_DISPARITY_PX

    _geom.USE_EE_FK_Z_BIAS_CORRECTION = False
    _geom.TARGET_XY_SOURCE = "overhead_homography"
    _geom.HOVER_HEIGHT_MM  = float(_PICK.Z_MAX_MM)
    _geom.GRASP_OFFSET_MM  = float(_PICK.GRIPPER_OFFSET_MM)
    _geom.PICK_PHI_MODE    = _PICK.PICK_PHI_MODE

    _burst.BURST_COUNT                 = _SURVEY.BURST_COUNT
    _burst.MIN_BURST_HITS              = _SURVEY.MIN_BURST_HITS
    _burst.BURST_FRAME_DELAY_S         = _SURVEY.BURST_FRAME_DELAY_S
    _burst.BURST_CLUSTER_MAX_CENTROID_PX = _SURVEY.BURST_CLUSTER_MAX_CENTROID_PX
    _burst.BURST_REQUIRE_SAME_CLASS    = _SURVEY.BURST_REQUIRE_SAME_CLASS
    _burst.TARGET_CLASS_NAMES          = list(_SURVEY.TARGET_CLASS_NAMES)

    _cand.BURST_COUNT             = _SURVEY.BURST_COUNT
    _cand.MIN_VALID_OBJECT_POINTS = _SURVEY.MIN_VALID_OBJECT_POINTS
    _cand.OVERHEAD_XY_BLEND_WEIGHT = _SURVEY.OVERHEAD_XY_BLEND_WEIGHT

    _z.Z_MAX_MM                    = float(_PICK.Z_MAX_MM)
    _z.GRIPPER_OFFSET_MM           = float(_PICK.GRIPPER_OFFSET_MM)
    _z.USE_ROBUST_OBJECT_Z         = _PICK.USE_ROBUST_OBJECT_Z
    _z.ROBUST_TOP_PERCENTILE       = _PICK.ROBUST_TOP_PERCENTILE
    _z.ROBUST_BOTTOM_PERCENTILE    = _PICK.ROBUST_BOTTOM_PERCENTILE
    _z.TOP_SPREAD_LOW_PERCENTILE   = _PICK.TOP_SPREAD_LOW_PERCENTILE
    _z.TOP_SPREAD_HIGH_PERCENTILE  = _PICK.TOP_SPREAD_HIGH_PERCENTILE
    _z.Z_UNCERTAINTY_CLEARANCE_GAIN = _PICK.PICK_Z_UNCERTAINTY_GAIN
    _z.Z_UNCERTAINTY_CLEARANCE_MIN_MM = _PICK.Z_UNCERTAINTY_CLEARANCE_MIN_MM
    _z.Z_UNCERTAINTY_CLEARANCE_MAX_MM = _PICK.PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM
    _z.Z_UNCERTAINTY_WARN_MM       = _PICK.Z_UNCERTAINTY_WARN_MM
    _z.PICK_EXTRA_CLEARANCE_MM     = _PICK.PICK_EXTRA_CLEARANCE_MM
    _z.MIN_OBJECT_HEIGHT_MM        = _PICK.MIN_OBJECT_HEIGHT_MM
    _z.MAX_OBJECT_HEIGHT_MM        = _PICK.MAX_OBJECT_HEIGHT_MM

    _xy.OVERHEAD_MATCH_MAX_DIST_MM      = _SURVEY.OVERHEAD_MATCH_MAX_DIST_MM
    _xy.OVERHEAD_MATCH_PREFER_SAME_CLASS = _SURVEY.OVERHEAD_MATCH_PREFER_SAME_CLASS

    _phi.PICK_PHI_MODE                     = _PICK.PICK_PHI_MODE
    _phi.OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI = _PICK.OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI
    _phi.USE_CONFIDENCE_PHI_BLEND          = _PICK.USE_CONFIDENCE_PHI_BLEND
    _phi.PHI_DISAGREEMENT_WARN_DEG         = _PICK.PHI_DISAGREEMENT_WARN_DEG
    _phi.PHI_MIN_CONFIDENCE                = _PICK.PHI_MIN_CONFIDENCE
    _phi.PHI_ASPECT_DECAY                  = _PICK.PHI_ASPECT_DECAY
    _phi.PHI_STEREO_HEIGHT_DECAY_CM        = _PICK.PHI_STEREO_HEIGHT_DECAY_CM
    _phi.PHI_FALLBACK_TO_CURRENT_EE_PHI    = _PICK.PHI_FALLBACK_TO_CURRENT_EE_PHI

    _survey.YOLO_WEIGHTS_PATH          = _SURVEY.YOLO_WEIGHTS_PATH
    _survey.YOLO_FALLBACK_WEIGHTS_PATH = _SURVEY.YOLO_FALLBACK_WEIGHTS_PATH
    _survey.RAFT_ROOT                  = _SURVEY.RAFT_ROOT
    _survey.RAFT_CHECKPOINT_PATH       = _SURVEY.RAFT_CHECKPOINT_PATH
    _survey.YOLO_IMGSZ                 = _SURVEY.YOLO_IMGSZ
    _survey.YOLO_CONF                  = _SURVEY.YOLO_CONF
    _survey.YOLO_IOU                   = _SURVEY.YOLO_IOU
    _survey.YOLO_RETINA_MASKS          = _SURVEY.YOLO_RETINA_MASKS
    _survey.TARGET_CLASS_NAMES         = list(_SURVEY.TARGET_CLASS_NAMES)
    _survey.RAFT_VALID_ITERS           = _SURVEY.RAFT_VALID_ITERS
    _survey.RAFT_DOWNSCALE             = _SURVEY.RAFT_DOWNSCALE
    _survey.RAFT_MIXED_PRECISION       = _SURVEY.RAFT_MIXED_PRECISION
    _survey.MIN_MASK_AREA_PX           = _SURVEY.MIN_MASK_AREA_PX


# ── Single-frame survey from saved images ─────────────────────────────────

def survey_from_images(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    rectifier: StereoRectifier,
    stereo_calib: dict,
    bundle: dict,
) -> SurveyState:
    """Build a SurveyState from one static stereo pair.

    Mirrors run_survey() but replaces the live burst-capture loop with a
    single rectified image pair.  Every downstream call — build_object_candidate,
    Z resolver, phi resolver, XY blend — is IDENTICAL to the wet run.
    """
    from vision.object_geometry import build_object_candidate

    # 1. Rectify (skip if Training_Images are pre-rectified)
    if IMAGES_ALREADY_RECTIFIED:
        rect_left, rect_right = left_bgr, right_bgr
    else:
        rect_left, rect_right = rectifier.rectify(left_bgr, right_bgr)

    # 2. YOLO on rect-left (same call as the wet run)
    detections = yolo.segment(rect_left)
    print(f"[DRY SURVEY] {len(detections)} YOLO detection(s)")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d.class_name:16s} conf={d.confidence:.2f}  mask={d.mask_area}px²")

    if not detections:
        print("[DRY SURVEY] no detections — returning empty survey")
        return SurveyState([], [], [], [], None, [], 0)

    # 3. RAFT — run ONCE, shared across all detections (efficient)
    print("[DRY SURVEY] running RAFT disparity …")
    t0 = time.perf_counter()
    disparity = raft.predict_disparity(rect_left, rect_right, color="BGR")
    print(f"[DRY SURVEY] RAFT done in {time.perf_counter() - t0:.2f}s")

    # 4. Build one candidate per detection
    candidates: list[CandidateDebug] = []
    for idx, det in enumerate(detections, start=1):
        print(f"\n[DRY SURVEY] building candidate {idx}: {det.class_name}")

        # — Point cloud (same call as build_candidate_from_track) ——————
        try:
            points_cam, point_uv_px = masked_disparity_to_pointcloud(
                det.mask, disparity, stereo_calib
            )
        except Exception as exc:
            print(f"  [PC FAIL] {exc}")
            continue

        n_pts = len(points_cam)
        if n_pts < _SURVEY.MIN_VALID_OBJECT_POINTS:
            print(f"  [PC SKIP] {n_pts} pts < {_SURVEY.MIN_VALID_OBJECT_POINTS} required")
            continue
        print(f"  [PC OK]   {n_pts} valid points")

        # — ObjectCandidate (same call as real pipeline) ────────────────
        try:
            cand = build_object_candidate(
                index=idx,
                yolo_det=det,
                points_cam=points_cam,
                point_uv_px=point_uv_px,
                robot=_NullRobot(),  # stub: no real robot in dry run
                bundle=bundle,
                stereo_tags={},      # no AprilTag data for static images
                frame_i=0,
            )
        except Exception as exc:
            print(f"  [CAND FAIL] {exc}")
            traceback.print_exc()
            continue

        # — Robot-frame Z (same call) ────────────────────────────────────
        try:
            points_robot = cam_points_to_robot_xyz(points_cam, bundle)
        except Exception as exc:
            print(f"  [Z] robot transform failed: {exc}")
            points_robot = None

        z_result = resolve_robust_object_z(points_robot, points_cam, candidate=cand)
        cand.object_robot_xyz_raw[2]       = z_result.robust_top_z_mm
        cand.object_robot_xyz_corrected[2] = z_result.robust_top_z_mm
        cand.hover_robot_z                 = z_result.hover_z_mm
        cand.grasp_robot_z                 = z_result.grasp_z_mm
        cand.stereo_z_bias_mm              = 0.0
        cand.z_debug                       = z_result

        # Build minimal mock track for CandidateDebug (compatible with real type)
        mock_obs   = _MockObservation(frame_i=0, detection=det)
        mock_track = _MockTrack(track_id=idx, class_name=det.class_name, _obs=mock_obs)

        dbg = CandidateDebug(
            candidate         = cand,
            track             = mock_track,
            best_frame_i      = 0,
            best_detection    = det,
            disparity         = disparity,
            disparity_overlay = colorize_disparity(disparity),
            left_overlay      = rect_left.copy(),
            point_count       = n_pts,
            point_uv_px       = point_uv_px,
            points_cam        = points_cam,
            stereo_xy_mm      = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64).reshape(2),
            overhead_xy_mm    = None,           # no overhead in dry run
            blend_weight_overhead = 0.0,
            xy_disagreement_mm = None,
        )

        # — Phi + XY blend (same calls as _match_overhead_to_candidates) ─
        resolve_pick_phi(dbg, None, bundle)
        apply_xy_blend(dbg, bundle)

        candidates.append(dbg)
        print(
            f"  [OK] target_xy=({float(cand.target_xy[0]):.1f},"
            f"{float(cand.target_xy[1]):.1f})  "
            f"top_z={z_result.robust_top_z_mm:.1f}mm  "
            f"height={z_result.object_height_mm:.1f}mm"
        )

    print(f"\n[DRY SURVEY] built {len(candidates)} candidate(s) from {len(detections)} detection(s)")
    return SurveyState([], [], [], candidates, None, [], 0)


# ── Candidate filter config ────────────────────────────────────────────────

def _best_candidate_config() -> BestCandidateConfig:
    workspace = _WORKSPACE
    wx0, wx1, wy0, wy1 = workspace_bounds_mm(workspace)
    return BestCandidateConfig(
        require_positive_platform_xy=bool(workspace.require_positive_platform_xy),
        platform_min_x_mm=wx0 if workspace.require_positive_platform_xy else None,
        platform_min_y_mm=wy0 if workspace.require_positive_platform_xy else None,
        workspace_x_min_mm=wx0 if workspace.enable_workspace_bounds else None,
        workspace_x_max_mm=wx1 if workspace.enable_workspace_bounds else None,
        workspace_y_min_mm=wy0 if workspace.enable_workspace_bounds else None,
        workspace_y_max_mm=wy1 if workspace.enable_workspace_bounds else None,
        require_robotframe_centroid_xy_in_platform_bounds=bool(workspace.enable_robotframe_centroid_xy_platform_bounds),
        robotframe_centroid_x_min_mm=wx0 if workspace.enable_robotframe_centroid_xy_platform_bounds else None,
        robotframe_centroid_x_max_mm=wx1 if workspace.enable_robotframe_centroid_xy_platform_bounds else None,
        robotframe_centroid_y_min_mm=wy0 if workspace.enable_robotframe_centroid_xy_platform_bounds else None,
        robotframe_centroid_y_max_mm=wy1 if workspace.enable_robotframe_centroid_xy_platform_bounds else None,
        use_robot_reach_check=False,    # no robot in dry run
        require_soft_pose_safe=False,   # no robot in dry run
        center_gate_enabled=bool(workspace.center_gate_enabled),
        max_image_center_norm_radius=float(workspace.max_image_center_norm_radius),
        cluster_gate_enabled=bool(workspace.cluster_gate_enabled),
        max_cluster_distance_mm=float(workspace.max_cluster_distance_mm),
        reject_placed_overlap=bool(workspace.reject_placed_overlap),
        placed_overlap_margin_mm=float(workspace.placed_overlap_margin_mm),
        min_volume_mm3=float(workspace.min_volume_mm3),
        max_volume_mm3=float(workspace.max_volume_cm3) * 1000.0,
    )


# ── Placement planning helpers ─────────────────────────────────────────────

def _compute_place_target(
    cand_dbg: CandidateDebug,
    *,
    object_i: int,
    base_xy: np.ndarray,
    placed_boxes: list,
    surface_zone: dict,
) -> tuple[np.ndarray, float]:
    """Return (target_xy_mm, target_phi_deg) for the i-th object."""
    raw_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=f"obj{object_i}")
    padded  = pad_aabb(raw_box, _PLACE.PAD_X_MM, _PLACE.PAD_Y_MM, _PLACE.PAD_Z_MM)

    surface_z = float(surface_zone.get("surface_z_mm", 0.0))
    place_phi  = float(surface_zone.get("default_phi_deg", 0.0))

    if object_i == 1:
        return np.asarray(base_xy, dtype=np.float64).reshape(2).copy(), place_phi

    if object_i == 2 and placed_boxes:
        plan = compute_adjacent_placement(
            reference_padded_box=placed_boxes[-1],
            moving_padded_box=padded,
            direction="left",
            surface_z_mm=surface_z,
            place_phi_deg=place_phi,
        )
        return plan.target_center_xy_mm.copy(), float(plan.target_phi_deg)

    # Object 3+: alternate between column_1 (odd) and column_2 (even) XY
    if len(placed_boxes) >= 2:
        col_xy = np.asarray(placed_boxes[object_i % 2].padded_box.center_xyz_mm[:2], dtype=np.float64)
        return col_xy.copy(), place_phi

    return np.asarray(base_xy, dtype=np.float64).reshape(2).copy(), place_phi


def _load_surface_zone() -> dict:
    return dict(load_place_scene(_PLACE, verbose=False))


# ── Main dry-run loop ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hardware-free dry run of the autonomous pipeline")
    parser.add_argument("--images", default=str(REPO_ROOT / "Training_Images"),
                        help="Folder containing Stereo_Left_#### / Stereo_Right_#### pairs")
    parser.add_argument("--index", type=int, default=None,
                        help="Run only this stereo pair index (default: all pairs)")
    parser.add_argument("--max-pairs", type=int, default=1,
                        help="Max image pairs to process (default: 1)")
    args = parser.parse_args(argv)

    training_dir = Path(args.images)
    print("=" * 70)
    print("DRY RUN AUTONOMOUS  —  hardware-free pipeline validation")
    print("=" * 70)
    print(f"images         : {training_dir}")
    print(f"YOLO weights   : {_SURVEY.YOLO_WEIGHTS_PATH}")
    print(f"RAFT checkpoint: {_SURVEY.RAFT_CHECKPOINT_PATH}")
    print(f"bundle         : {_SURVEY.BUNDLE_PATH}")
    print(f"stereo calib   : {_SURVEY.STEREO_CALIBRATION_PATH}")
    print(f"place scene    : {_PLACE.PLACE_SCENE_NAME}")
    print(f"runtime        : context={_RUNTIME_CONTEXT.name} workspace_profile={_WORKSPACE.name}")
    print(f"packing        : planner={_PLACE.PLACE_PLANNING_SEQUENCE_NAME} pad_xyz=({_PLACE.PAD_X_MM},{_PLACE.PAD_Y_MM},{_PLACE.PAD_Z_MM})")
    print()

    # ── Configure modules ────────────────────────────────────────────────
    _configure_vision_modules()

    # ── Load calibration ─────────────────────────────────────────────────
    calib_path = REPO_ROOT / _SURVEY.STEREO_CALIBRATION_PATH
    if not calib_path.exists():
        calib_path = _SURVEY.STEREO_CALIBRATION_PATH
    if not calib_path.exists():
        print(f"[ERROR] stereo calibration not found: {calib_path}")
        return 1
    stereo_data = np.load(str(calib_path), allow_pickle=False)
    stereo_calib = {k: np.asarray(stereo_data[k]) for k in stereo_data.files}
    print(f"[CALIB] loaded {calib_path}")

    bundle_path = REPO_ROOT / _SURVEY.BUNDLE_PATH
    if not bundle_path.exists():
        bundle_path = _SURVEY.BUNDLE_PATH
    if not bundle_path.exists():
        print(f"[WARN] calibration bundle not found: {bundle_path}. Robot-frame XY will be approximate.")
        bundle: dict = {}
    else:
        from test_calibration_bundle_live_stereo_z_pickplace import load_bundle
        bundle = load_bundle(bundle_path)
        print(f"[CALIB] bundle loaded (processed): {bundle_path}")

    # ── Load models ──────────────────────────────────────────────────────
    print("[INIT] selecting torch device …")
    device_info = select_torch_device(use_cuda=_SURVEY.USE_CUDA, use_half=_SURVEY.USE_HALF)

    weights = REPO_ROOT / _SURVEY.YOLO_WEIGHTS_PATH
    if not weights.exists():
        weights = REPO_ROOT / _SURVEY.YOLO_FALLBACK_WEIGHTS_PATH
    print(f"[INIT] loading YOLO from {weights} …")
    yolo = YOLOSegmenter(
        weights_path=str(weights),
        device_info=device_info,
        imgsz=_SURVEY.YOLO_IMGSZ,
        conf=_SURVEY.YOLO_CONF,
        iou=_SURVEY.YOLO_IOU,
        retina_masks=_SURVEY.YOLO_RETINA_MASKS,
        min_mask_area_px=_SURVEY.MIN_MASK_AREA_PX,
        target_class_names=list(_SURVEY.TARGET_CLASS_NAMES) if _SURVEY.TARGET_CLASS_NAMES else None,
    )
    yolo.warmup()

    raft_root = REPO_ROOT / _SURVEY.RAFT_ROOT
    raft_ckpt = REPO_ROOT / _SURVEY.RAFT_CHECKPOINT_PATH
    print(f"[INIT] loading RAFT from {raft_ckpt} …")
    raft = RAFTStereoRunner(
        raft_root=str(raft_root),
        checkpoint_path=str(raft_ckpt),
        device_info=device_info,
        valid_iters=_SURVEY.RAFT_VALID_ITERS,
        downscale=_SURVEY.RAFT_DOWNSCALE,
        mixed_precision=_SURVEY.RAFT_MIXED_PRECISION,
    )
    raft.warmup()

    rectifier    = StereoRectifier(stereo_calib)
    surface_zone = _load_surface_zone()
    base_xy      = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64)
    selector_cfg = _best_candidate_config()

    print(f"\n[ZONE] {surface_zone['name']}  center=({base_xy[0]:.1f},{base_xy[1]:.1f})  "
          f"surface_z={surface_zone['surface_z_mm']:.1f}mm  "
          f"size=({surface_zone['width_mm']:.0f}x{surface_zone['depth_mm']:.0f})mm")

    # ── Find image pairs ─────────────────────────────────────────────────
    pairs = find_stereo_pairs(training_dir)
    print(f"\n[IMAGES] found {len(pairs)} stereo pair(s) in {training_dir}")

    if args.index is not None:
        pairs = [p for p in pairs if p.index == args.index]
        if not pairs:
            print(f"[ERROR] no pair with index {args.index}")
            return 1

    pairs = pairs[:args.max_pairs]

    # ── Per-pair simulation ───────────────────────────────────────────────
    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"PAIR {pair.index:04d}  left={pair.left_path.name}  right={pair.right_path.name}")
        print("=" * 70)

        left  = cv2.imread(str(pair.left_path),  cv2.IMREAD_COLOR)
        right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(f"[ERROR] could not load images")
            continue

        # Simulate the full pack run for this scene
        placed_boxes: list = []
        remaining_survey: SurveyState | None = None

        for object_i in range(1, TARGET_OBJECT_COUNT + 1):
            print(f"\n--- Object {object_i}/{TARGET_OBJECT_COUNT} ---")

            # Survey: re-use for first object, re-run for subsequent
            # (in real wet run a prefetched survey would exist; here we
            #  always run fresh since we have only static images)
            if remaining_survey is None or object_i > 1:
                survey = survey_from_images(
                    left, right,
                    yolo=yolo, raft=raft, rectifier=rectifier,
                    stereo_calib=stereo_calib, bundle=bundle,
                )
            else:
                survey = remaining_survey

            if not survey.candidates:
                print(f"[FLOW] no candidates for object {object_i} — stopping")
                break

            # Candidate selection (same logic as autonomous script)
            result = choose_best_candidate(
                survey, config=selector_cfg, robot=None, placed_boxes=placed_boxes
            )
            result.print_debug("[BEST]")

            if result.selected is None:
                print(f"[FLOW] no valid candidate passed filters — stopping")
                break

            cand_dbg = result.selected
            c = cand_dbg.candidate
            print(
                f"\n[FLOW] WOULD PICK: [{getattr(c, 'index', '?')}] {c.yolo.class_name}"
                f"  xy=({float(c.target_xy[0]):.1f},{float(c.target_xy[1]):.1f})"
                f"  phi={float(getattr(c, 'pick_phi_deg', 0.0)):.1f}°"
            )

            # Compute place target
            try:
                target_xy, target_phi = _compute_place_target(
                    cand_dbg,
                    object_i=object_i,
                    base_xy=base_xy,
                    placed_boxes=placed_boxes,
                    surface_zone=surface_zone,
                )
                print(
                    f"[FLOW] WOULD PLACE: target_xy=({target_xy[0]:.1f},{target_xy[1]:.1f})"
                    f"  phi={target_phi:.1f}°"
                )
            except Exception as exc:
                print(f"[FLOW] placement planning failed: {exc}")
                break

            # Build placed AABB and record for next iteration
            try:
                raw_box = aabb_from_object_candidate(c, default_label=f"obj{object_i}")
                placed  = pad_aabb(raw_box, _PLACE.PAD_X_MM, _PLACE.PAD_Y_MM, _PLACE.PAD_Z_MM)
                placed_boxes.append(placed)
                print(
                    f"[FLOW] placed_boxes={len(placed_boxes)}  "
                    f"raw_size=({raw_box.size_xyz_mm[0]:.0f}x"
                    f"{raw_box.size_xyz_mm[1]:.0f}x"
                    f"{raw_box.size_xyz_mm[2]:.0f})mm"
                )
            except Exception as exc:
                print(f"[FLOW] could not build placed AABB: {exc}")
                break

            # Remove just-placed candidate from survey for next iteration
            survey.candidates = [
                dbg for dbg in survey.candidates
                if dbg is not cand_dbg
            ]
            remaining_survey = survey

        print(f"\n[DONE] simulated placing {len(placed_boxes)} object(s) for pair {pair.index:04d}")

    print("\n[DRY RUN COMPLETE]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[DRY RUN] interrupted")
    except Exception:
        traceback.print_exc()
        raise
