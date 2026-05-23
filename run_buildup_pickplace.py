"""
run_buildup_pickplace.py

Main buildup pick/place pipeline.

Differences from run_pickplace_fast.py:
  1. No EE AprilTag required for Z correction.
     Object height comes directly from stereo triangulation via
     A_robot_from_cam_xyz_3x4 (stereo_z_bias always = 0).
  2. Overhead YOLO -> H(z) homography for robot X,Y, BLENDED with stereo XY
     by OVERHEAD_XY_BLEND_WEIGHT (default 0.7 overhead / 0.3 stereo).
  3. BLB 2D packing planner for automated bag placement.
     Bag geometry defined in USER SETTINGS (robot-frame mm).
     No manual drop-zone 'n' key; BLB auto-plans each spot.
  4. Simplified Z policy:
        travel_z = Z_MAX_MM
        hover_z  = Z_MAX_MM
        grasp_z  = z_stereo + GRIPPER_OFFSET_MM (+50 for short items)
        place_z  = BAG_FLOOR_GRIPPER_Z_MM + stack + item_height + gap
     Validated up front in execute_place_blb so a bad plan never starts motion.
  5. Auto-cycle test mode ('a') picks-and-places every surveyed candidate
     in scored order, tracks holding state so user can retry on partial fail.

Keys:
  s       survey workspace (stereo YOLO+RAFT + overhead YOLO match)
  1-9     select candidate from last survey
  g       move to survey pose
  m       hover to selected object (without picking)
  k       PICK selected object
  f       PLACE using BLB-planned spot
  a       AUTO cycle: pick+place every surveyed candidate until bag full
  b       print geometry sanity check
  r       print calibration matrices
  w       print current bag state / BLB layout
  x       reset bag state (clear all placed items)
  [/]     jog Z down / up
  ,/.     jog phi (J4) down / up
  o/l     open / close claw
  p       print current FK pose
  h/c/u   home / sync-home / assume-home
  e/d     enable / disable motors
  q/ESC   quit
"""

from __future__ import annotations

# ============================================================
# USER SETTINGS — edit these before running
# ============================================================

# --- Calibration ---
from pathlib import Path

BUNDLE_PATH = Path("robot_calibration_bundle.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")

# --- Vision ---
YOLO_WEIGHTS_PATH = Path("yolo_weights/Validate_Only_100_Training_Best.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("yolo_weights/validate_V2.pt")
RAFT_ROOT = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")
YOLO_IMGSZ: int = 640
YOLO_CONF: float = 0.35
YOLO_IOU: float = 0.50
YOLO_RETINA_MASKS: bool = True
TARGET_CLASS_NAMES: list[str] = []   # [] = detect all classes
USE_CUDA: bool = True
USE_HALF: bool = True
RAFT_VALID_ITERS: int = 16
RAFT_DOWNSCALE: float = 1.0
RAFT_MIXED_PRECISION: bool = True

# --- Point cloud ---
MIN_MASK_AREA_PX: int = 500
MIN_DISPARITY_PX: float = 1.0
MIN_VALID_OBJECT_POINTS: int = 300
POINTCLOUD_MAX_POINTS: int = 20_000

# --- Object size estimation ---
USE_POINTCLOUD_SIZE_BLEND: bool = True
POINTCLOUD_SIZE_BLEND_WEIGHT: float = 0.75

# --- Z geometry (PICK / HOVER / TRAVEL) ---
Z_MAX_MM: float = 250
GRIPPER_OFFSET_MM: float = 100

# Compatibility shim for vision.object_geometry.
HOVER_HEIGHT_MM: float = Z_MAX_MM
GRASP_OFFSET_MM: float = GRIPPER_OFFSET_MM

# --- Overhead/stereo XY blending ---
OVERHEAD_MATCH_MAX_DIST_MM: float = 120.0
OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True
OVERHEAD_XY_BLEND_WEIGHT: float = 0.45
XY_DISAGREEMENT_WARN_MM: float = 50.0

# --- Pick orientation ---
PICK_PHI_MODE: str = "triangulated_short_side"
VALID_PICK_PHI_MODES = {
    "centroid_longest_ray_perp",
    "centroid_shortest_ray_parallel",
    "overhead_minor_axis",
    "pointcloud_shortest_path",
    "triangulated_short_side",
    "mask_minor_axis_pointcloud",
    "current_fk",
}

# --- Survey ---
SURVEY_BURST_COUNT: int = 5
SURVEY_FRAME_DELAY_S: float = 0.05
SURVEY_RAFT_MODE: str = "best_frame_only"
SURVEY_USE_YOLO_BATCH: bool = True

# --- Bag / BLB packing ---
BAG_ORIGIN_X_MM: float = 700
BAG_ORIGIN_Y_MM: float = 200
BAG_WIDTH_MM: float = 1000 # bag extent along robot +X axis
BAG_DEPTH_MM: float =1000  # bag extent along robot +Y axis

# --- Z geometry (PLACE side) ---
# Robot Z at which the EMPTY gripper tip just touches the empty bag floor.
# Calibrate once: lower the empty gripper into the empty bag until the tip
# kisses the floor, then read FK Z and put that number here.
BAG_FLOOR_GRIPPER_Z_MM: float = 0       # <-- CALIBRATE
PLACE_RELEASE_GAP_MM: float = 1.0
BAG_PLACE_PHI_DEG: float | None = None

# --- Motion ---
COARSE_MOVE_TIME_S: float = 1.10
PICK_MOVE_TIME_S: float = 0.75
Z_JOG_MM: float = 5.0
PHI_JOG_DEG: float = 5.0
CLAW_OPEN_DEG: int = 65
CLAW_CLOSED_DEG: int = 0
CLAW_SETTLE_S: float = 0.30

# --- Safety ---
REFUSE_PICK_IF_TOO_FEW_POINTS: bool = True
MAX_NEAREST_SUPPORT_DIST_MM: float = 1000

# --- Survey pose ---
SURVEY_POSE_X_MM: float = 200
SURVEY_POSE_Y_MM: float = 200
SURVEY_POSE_Z_MM: float = 200
SURVEY_POSE_PHI_DEG: float = 0.0

# --- Display ---
COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 600
STEREO_DRAW_H_PX: int = 380
WINDOW: str = "Buildup Pipeline"

# --- Debug overlays ---
DEBUG_MASK_OVERLAY: bool = True
DEBUG_INFER_EVERY_N_FRAMES: int = 1
DEBUG_MASK_ALPHA: float = 0.35

# --- Debug printing ---
# Verbose prints the per-motion FK verification after each step.
DEBUG_VERBOSE_MOTION: bool = True

# ============================================================
# END USER SETTINGS
# ============================================================

import math
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np

# Patch vision.object_geometry module-level knobs BEFORE importing vision.survey
import vision.object_geometry as _geom_mod
_geom_mod.USE_EE_FK_Z_BIAS_CORRECTION = False
_geom_mod.TARGET_XY_SOURCE = "overhead_homography"
_geom_mod.HOVER_HEIGHT_MM = HOVER_HEIGHT_MM
_geom_mod.GRASP_OFFSET_MM = GRIPPER_OFFSET_MM
_geom_mod.PICK_PHI_MODE = PICK_PHI_MODE

import vision.survey as _survey_mod
_survey_mod.SURVEY_BURST_COUNT = SURVEY_BURST_COUNT
_survey_mod.SURVEY_FRAME_DELAY_S = SURVEY_FRAME_DELAY_S
_survey_mod.SURVEY_RAFT_MODE = SURVEY_RAFT_MODE
_survey_mod.SURVEY_USE_YOLO_BATCH = SURVEY_USE_YOLO_BATCH
_survey_mod.MIN_VALID_OBJECT_POINTS = MIN_VALID_OBJECT_POINTS

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import estimate_mask_centroid_ray_angle_deg, masked_disparity_to_pointcloud
from vision.object_geometry import (
    ObjectCandidate,
    build_object_candidate,
    fuse_survey_candidates,
    candidate_score,
)
from vision.survey import run_survey_workspace

from planning.grocery_item import GroceryItem
from planning.bag_state import BagState, PlacedItem
from planning.planner_2d_blb import choose_placement_spot_2d

from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from hardware.robot import Robot

from test_calibration_bundle_live_stereo_z_pickplace import (
    load_bundle,
    load_stereo_calibration,
    read_stereo_tags_once,
    cam_xyz_to_robot_xyz,
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
# Derived settings
# ============================================================

_BAG_WIDTH_CM = BAG_WIDTH_MM / 10.0
_BAG_DEPTH_CM = BAG_DEPTH_MM / 10.0


# ============================================================
# Debug printing helpers
# ============================================================

def _hr(title: str = "", char: str = "=", width: int = 78) -> None:
    """Print a horizontal rule with optional title for visual section breaks."""
    if not title:
        print(char * width)
        return
    pad = max(0, width - len(title) - 2)
    print(f"{char * 3} {title} {char * pad}"[:width])


def _fmt_xyz(xyz, fmt: str = "{:7.1f}") -> str:
    """Format a 3-vector like '( 100.0, 200.0,  50.0)'."""
    try:
        x, y, z = (float(v) for v in xyz[:3])
    except (TypeError, ValueError, IndexError):
        return "(   --,    --,    --)"
    return f"({fmt.format(x)},{fmt.format(y)},{fmt.format(z)})"


def _fmt_optional(value, fmt: str = "{:.1f}", missing: str = "--") -> str:
    if value is None:
        return missing
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _print_fk(robot: Robot, prefix: str = "[FK]") -> tuple[float, float, float, float]:
    x, y, z, phi = robot.fk()
    print(f"{prefix} x={x:7.1f} y={y:7.1f} z={z:7.1f} phi={phi:6.2f}")
    return x, y, z, phi


def _print_motion_verify(
    robot: Robot,
    label: str,
    target_x: float | None = None,
    target_y: float | None = None,
    target_z: float | None = None,
    target_phi: float | None = None,
) -> None:
    """After a motion, print FK and the per-axis error vs commanded target."""
    if not DEBUG_VERBOSE_MOTION:
        return
    x, y, z, phi = robot.fk()
    parts = [f"FK=({x:7.1f},{y:7.1f},{z:7.1f}) phi={phi:6.2f}"]
    errs = []
    if target_x is not None: errs.append(f"dx={x - target_x:+5.1f}")
    if target_y is not None: errs.append(f"dy={y - target_y:+5.1f}")
    if target_z is not None: errs.append(f"dz={z - target_z:+5.1f}")
    if target_phi is not None: errs.append(f"dphi={phi - target_phi:+5.2f}")
    if errs:
        parts.append("err: " + " ".join(errs))
    print(f"{label} {'  '.join(parts)}")


# ============================================================
# Vision loading
# ============================================================

def load_fast_vision(device_info) -> tuple[YOLOSegmenter, RAFTStereoRunner]:
    """Load YOLO and RAFT models; warmup."""
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
    print(f"[VISION] YOLO loaded: {weights.name}  conf={YOLO_CONF}  iou={YOLO_IOU}  imgsz={YOLO_IMGSZ}")

    raft = RAFTStereoRunner(
        raft_root=str(RAFT_ROOT),
        checkpoint_path=str(RAFT_CHECKPOINT_PATH),
        device_info=device_info,
        valid_iters=RAFT_VALID_ITERS,
        downscale=RAFT_DOWNSCALE,
        mixed_precision=RAFT_MIXED_PRECISION,
    )
    raft.warmup()
    print(f"[VISION] RAFT loaded  iters={RAFT_VALID_ITERS}  downscale={RAFT_DOWNSCALE}  fp16={RAFT_MIXED_PRECISION}")
    return yolo, raft


# ============================================================
# Bundle quality summary
# ============================================================

def _print_bundle_summary(bundle: dict) -> None:
    """Print calibration-quality summary at startup so the user knows what they're flying with."""
    _hr("CALIBRATION BUNDLE SUMMARY", "=")
    scan_mode = bundle.get("scan_mode")
    if scan_mode is not None:
        try:
            print(f"[BUNDLE] scan_mode = {str(scan_mode[0])}")
        except Exception:
            pass

    z_levels = bundle.get("z_levels_mm")
    rms = bundle.get("homography_rms_error_mm")
    maxerr = bundle.get("homography_max_error_mm")
    npts = bundle.get("homography_n_points")
    if z_levels is not None and rms is not None:
        print("[BUNDLE] Homography layers H(z):")
        for i, z in enumerate(np.asarray(z_levels).ravel()):
            r = float(np.asarray(rms).ravel()[i]) if rms is not None else float("nan")
            m = float(np.asarray(maxerr).ravel()[i]) if maxerr is not None else float("nan")
            n = int(np.asarray(npts).ravel()[i]) if npts is not None else -1
            print(f"           z={float(z):6.1f} mm  RMS={r:5.2f} mm  MAX={m:5.2f} mm  n={n}")

    s_rmse = bundle.get("stereo_robot_xyz_fit_rmse_mm")
    if s_rmse is not None:
        try:
            print(f"[BUNDLE] Stereo robot_from_cam_xyz fit RMSE = {float(np.asarray(s_rmse).ravel()[0]):.2f} mm")
        except Exception:
            pass

    support = bundle.get("support_robot_xyz_mm")
    if support is not None:
        print(f"[BUNDLE] Support points: {len(np.asarray(support))}")

    used_intr = bundle.get("used_overhead_intrinsics")
    if used_intr is not None:
        try:
            print(f"[BUNDLE] Overhead intrinsics used: {bool(np.asarray(used_intr).ravel()[0])}")
        except Exception:
            pass

    floor_gripper = bundle.get("robot_z_zero_floor_height_mm")
    if floor_gripper is not None:
        try:
            print(f"[BUNDLE] EE floor height at robot_z=0: {float(np.asarray(floor_gripper).ravel()[0]):.1f} mm")
        except Exception:
            pass
    _hr("", "=")


# ============================================================
# Overhead YOLO matching
# ============================================================

def _overhead_bbox_corners_px(det: YOLODetection) -> np.ndarray:
    """Return the overhead detection bbox corners in image pixel coordinates."""
    x1, y1, x2, y2 = (float(v) for v in det.bbox)
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float64,
    )


def _overhead_mask_rect_corners_px(det: YOLODetection) -> np.ndarray | None:
    mask = getattr(det, "mask", None)
    if mask is None:
        return None
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    points = cv2.findNonZero(mask_u8)
    if points is None or len(points) < 4:
        return None
    rect = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rect).astype(np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None
    return corners


def _project_overhead_corners_to_robot_xy_mm(
    corners_px: np.ndarray,
    z_mm: float,
    bundle: dict,
) -> np.ndarray | None:
    projected: list[np.ndarray] = []
    lookup_z, _ = clamp_lookup_z_to_bundle(max(0.0, float(z_mm)), bundle)
    for corner_px in np.asarray(corners_px, dtype=np.float64).reshape(-1, 2):
        try:
            xy_raw, *_ = map_uv_z_to_robot_xy(corner_px, lookup_z, bundle)
            xy = np.asarray(xy_raw, dtype=np.float64).reshape(-1)[:2]
        except Exception:
            return None
        if xy.shape[0] != 2 or not np.all(np.isfinite(xy)):
            return None
        projected.append(xy)
    if len(projected) < 4:
        return None
    return np.vstack(projected)


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

    lookup_z, _ = clamp_lookup_z_to_bundle(max(0.0, float(z_mm)), bundle)
    centroid = np.asarray(det.centroid_px, dtype=np.float64).reshape(2)
    theta = np.deg2rad(float(image_angle))
    axis_px = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    half_len_px = max(
        12.0,
        0.5 * float(det.minor_axis_length_px if select == "shortest" else det.major_axis_length_px),
    )

    try:
        xy0, *_ = map_uv_z_to_robot_xy(centroid - axis_px * half_len_px, lookup_z, bundle)
        xy1, *_ = map_uv_z_to_robot_xy(centroid + axis_px * half_len_px, lookup_z, bundle)
    except Exception:
        return None, f"{source}_overhead_projection_failed"

    delta = np.asarray(xy1, dtype=np.float64).reshape(2) - np.asarray(xy0, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(delta)) or float(np.linalg.norm(delta)) < 1e-6:
        return None, f"{source}_overhead_degenerate_robot_delta"

    phi = float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0)
    return phi, f"{source}_overhead"


def _footprint_dims_from_robot_corners_cm(
    robot_corners_xy_mm: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float] | None] | None:
    corners = np.asarray(robot_corners_xy_mm, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 4 or not np.all(np.isfinite(corners)):
        return None

    width_cm = (float(np.max(corners[:, 0])) - float(np.min(corners[:, 0]))) / 10.0
    depth_cm = (float(np.max(corners[:, 1])) - float(np.min(corners[:, 1]))) / 10.0
    if width_cm <= 0.0 or depth_cm <= 0.0:
        return None

    oriented_cm: tuple[float, float] | None = None
    if len(corners) == 4:
        side_mm = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if np.all(np.isfinite(side_mm)) and np.all(side_mm > 0.0):
            side_a_cm = float((side_mm[0] + side_mm[2]) * 0.5 / 10.0)
            side_b_cm = float((side_mm[1] + side_mm[3]) * 0.5 / 10.0)
            oriented_cm = (side_a_cm, side_b_cm)

    return (width_cm, depth_cm), oriented_cm


def _set_overhead_footprint_from_detection(
    cand: ObjectCandidate,
    det: YOLODetection,
    bundle: dict,
) -> None:
    cand.overhead_bbox_px = tuple(float(v) for v in det.bbox)
    try:
        stereo_z_mm = max(0.0, float(cand.object_robot_xyz_raw[2]))
    except (IndexError, TypeError, ValueError):
        return

    bbox_corners = _overhead_bbox_corners_px(det)
    corner_sets: list[tuple[str, np.ndarray]] = []
    mask_corners = _overhead_mask_rect_corners_px(det)
    if mask_corners is not None:
        corner_sets.append(("overhead_projected_mask_rect", mask_corners))
    corner_sets.append(("overhead_projected_bbox", bbox_corners))

    for source, corners_px in corner_sets:
        robot_corners = _project_overhead_corners_to_robot_xy_mm(corners_px, stereo_z_mm, bundle)
        if robot_corners is None:
            continue
        dims = _footprint_dims_from_robot_corners_cm(robot_corners)
        if dims is None:
            continue
        aabb_cm, oriented_cm = dims
        cand.topdown_width_cm = float(aabb_cm[0])
        cand.topdown_depth_cm = float(aabb_cm[1])
        cand.topdown_area_cm2 = float(aabb_cm[0] * aabb_cm[1])
        cand.topdown_bbox_robot_xy_mm = robot_corners
        cand.topdown_aabb_cm = aabb_cm
        cand.topdown_oriented_rect_cm = oriented_cm
        cand.topdown_footprint_source = source
        return


def _clear_overhead_footprint(cand: ObjectCandidate) -> None:
    cand.overhead_bbox_px = None
    cand.topdown_width_cm = None
    cand.topdown_depth_cm = None
    cand.topdown_area_cm2 = None
    cand.topdown_bbox_robot_xy_mm = None
    cand.topdown_aabb_cm = None
    cand.topdown_oriented_rect_cm = None
    cand.topdown_footprint_source = None


def _valid_footprint_pair_cm(value: object) -> tuple[float, float] | None:
    try:
        w, d = value  # type: ignore[misc]
        width = float(w)
        depth = float(d)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(width) or not np.isfinite(depth):
        return None
    if width <= 0.0 or depth <= 0.0:
        return None
    return width, depth


def _apply_pointcloud_size_blend(cand: ObjectCandidate) -> None:
    if not USE_POINTCLOUD_SIZE_BLEND:
        return
    pointcloud_dims = _valid_footprint_pair_cm(cand.pointcloud_footprint_cm)
    if pointcloud_dims is None:
        return
    homography_dims = _valid_footprint_pair_cm(cand.topdown_aabb_cm)
    w = float(np.clip(POINTCLOUD_SIZE_BLEND_WEIGHT, 0.0, 1.0))

    if homography_dims is None:
        blend_w, blend_d = pointcloud_dims
        source = "pointcloud_only"
    else:
        blend_w = w * pointcloud_dims[0] + (1.0 - w) * homography_dims[0]
        blend_d = w * pointcloud_dims[1] + (1.0 - w) * homography_dims[1]
        source = f"pointcloud_homography_blend_w{w:.2f}"

    cand.topdown_width_cm = float(blend_w)
    cand.topdown_depth_cm = float(blend_d)
    cand.topdown_area_cm2 = float(blend_w * blend_d)
    cand.topdown_aabb_cm = (float(blend_w), float(blend_d))
    cand.topdown_oriented_rect_cm = None
    cand.topdown_footprint_source = source


def _match_overhead_detections_to_candidates(
    overhead_dets: list[YOLODetection],
    candidates: list[ObjectCandidate],
    bundle: dict,
) -> None:
    """Match each stereo candidate to its closest overhead YOLO detection."""
    if not overhead_dets or not candidates:
        return

    n_matched = 0
    for cand in candidates:
        stereo_z = float(cand.object_robot_xyz_raw[2])
        stereo_xy = cand.object_robot_xyz_raw[:2].copy()

        pool = overhead_dets
        if OVERHEAD_MATCH_PREFER_SAME_CLASS:
            same_class = [d for d in overhead_dets if d.class_name == cand.yolo.class_name]
            if same_class:
                pool = same_class

        best_det: YOLODetection | None = None
        best_dist = float("inf")
        for det in pool:
            try:
                proj_xy, *_ = map_uv_z_to_robot_xy(det.centroid_px, stereo_z, bundle)
                dist = float(np.linalg.norm(np.asarray(proj_xy, dtype=np.float64) - stereo_xy))
                if dist < best_dist:
                    best_dist = dist
                    best_det = det
            except Exception:
                continue

        if best_det is not None and best_dist <= OVERHEAD_MATCH_MAX_DIST_MM:
            cand.overhead_centroid_px = np.asarray(
                best_det.centroid_px, dtype=np.float64
            ).flatten()[:2]
            _set_overhead_footprint_from_detection(cand, best_det, bundle)
            if PICK_PHI_MODE == "overhead_minor_axis":
                cand.pick_phi_deg = float(best_det.minor_axis_angle_deg)
                cand.pick_phi_source = "overhead_minor_axis"
            elif PICK_PHI_MODE in {"centroid_longest_ray_perp", "centroid_shortest_ray_parallel"}:
                select = "longest" if PICK_PHI_MODE == "centroid_longest_ray_perp" else "shortest"
                phi, source = _overhead_centroid_ray_phi(
                    best_det,
                    stereo_z,
                    bundle,
                    select=select,
                    perpendicular=PICK_PHI_MODE == "centroid_longest_ray_perp",
                )
                if phi is not None:
                    cand.pick_phi_deg = float(phi)
                    cand.pick_phi_source = source
            n_matched += 1
            print(
                f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} "
                f"-> overhead {best_det.class_name:14s}  "
                f"dist={best_dist:5.1f} mm"
            )
        else:
            reason = "no overhead dets in pool" if best_det is None else f"best dist {best_dist:.1f} > {OVERHEAD_MATCH_MAX_DIST_MM:.0f}"
            print(f"[MATCH] cand[{cand.index}] {cand.yolo.class_name:14s} -> NO MATCH ({reason})")

    print(f"[MATCH] total {n_matched}/{len(candidates)} candidates matched to overhead")


def _clamp_candidate_z_floor(cand: ObjectCandidate) -> None:
    """Clamp candidate object Z to >= -150 to keep grasp_z computation sane."""
    z_raw = float(cand.object_robot_xyz_raw[2])
    z_clamped = max(-150.0, z_raw)
    if z_clamped != z_raw:
        print(f"[Z CLAMP] cand[{cand.index}] z {z_raw:.1f} -> {z_clamped:.1f} mm")
    cand.object_robot_xyz_raw[2] = z_clamped
    if cand.object_robot_xyz_corrected.shape[0] >= 3:
        cand.object_robot_xyz_corrected[2] = z_clamped


# ============================================================
# Pick-side Z helpers
# ============================================================

def _compute_grasp_robot_z(cand: ObjectCandidate) -> float:
    """Compute grasp Z from stereo surface Z and measured object height.

    Convention:
        gripper_tip_Z = robot_Z - GRIPPER_OFFSET_MM

    Tall items: tip at stereo Z (z_stereo).
    Short or unknown-height items: add a +50 mm buffer so the fingers have
    room to close around the object without bottoming out on the table.
    """
    z_stereo = float(cand.object_robot_xyz_raw[2])
    item_height_cm = cand.pointcloud_height_cm
    item_height_mm = (
        None if item_height_cm is None else float(item_height_cm) * 10.0
    )

    if item_height_mm is None or item_height_mm < 50.0:
        grasp_z = z_stereo + GRIPPER_OFFSET_MM + 50.0
        regime = "short/unknown(+50mm buffer)"
    else:
        grasp_z = z_stereo + GRIPPER_OFFSET_MM
        regime = "tall"

    h_str = "None" if item_height_mm is None else f"{item_height_mm:.1f} mm"
    print(
        f"[GRASP Z] cand[{cand.index}] z_stereo={z_stereo:7.1f} mm  "
        f"item_h={h_str:>9s}  regime={regime:24s}  "
        f"-> grasp_z={grasp_z:7.1f} mm"
    )

    if grasp_z < 0.0:
        print(f"[GRASP Z] WARNING: computed {grasp_z:.1f} mm < 0; clamping to 0")
        return 0.0
    return float(grasp_z)


def _apply_simple_pick_z(cand: ObjectCandidate) -> None:
    cand.hover_robot_z = float(Z_MAX_MM)
    cand.grasp_robot_z = _compute_grasp_robot_z(cand)


def _reapply_overhead_xy(
    cand: ObjectCandidate,
    bundle: dict,
) -> None:
    """Re-compute candidate.target_xy as a weighted blend of overhead + stereo XY."""
    if cand.overhead_centroid_px is None:
        return

    stereo_z = max(0.0, float(cand.object_robot_xyz_raw[2]))
    lookup_z, lookup_z_clamped = clamp_lookup_z_to_bundle(stereo_z, bundle)

    try:
        overhead_xy_raw, *_ = map_uv_z_to_robot_xy(
            cand.overhead_centroid_px, lookup_z, bundle
        )
    except Exception as exc:
        print(f"[XY BLEND] cand[{cand.index}] overhead XY map failed: {exc}")
        cand.overhead_centroid_px = None
        _clear_overhead_footprint(cand)
        return

    overhead_xy = np.asarray(overhead_xy_raw, dtype=np.float64).flatten()[:2]
    stereo_xy = cand.object_robot_xyz_raw[:2].copy()
    disagreement_mm = float(np.linalg.norm(overhead_xy - stereo_xy))

    tag = ""
    if disagreement_mm > XY_DISAGREEMENT_WARN_MM:
        tag = "  <-- WARN"
    print(
        f"[XY BLEND] cand[{cand.index}] overhead=({overhead_xy[0]:7.1f},{overhead_xy[1]:7.1f}) "
        f"stereo=({stereo_xy[0]:7.1f},{stereo_xy[1]:7.1f}) "
        f"disagreement={disagreement_mm:5.1f} mm{tag}"
    )

    w = OVERHEAD_XY_BLEND_WEIGHT
    target_xy = w * overhead_xy + (1.0 - w) * stereo_xy
    support_dist, support_idx = nearest_support_distance(target_xy, lookup_z, bundle)

    cand.target_xy = target_xy.copy()
    cand.target_xy_source_effective = "merged_overhead_stereo"
    cand.lookup_z_used = float(lookup_z)
    cand.lookup_z_clamped = bool(lookup_z_clamped)
    cand.support_distance_mm = float(support_dist)
    cand.support_index = int(support_idx)
    cand.object_robot_xyz_corrected = np.array(
        [target_xy[0], target_xy[1], stereo_z], dtype=np.float64
    )
    cand.hover_robot_z = float(Z_MAX_MM)
    cand.grasp_robot_z = stereo_z + float(GRIPPER_OFFSET_MM)
    cand.stereo_z_bias_mm = 0.0


# ============================================================
# Place-side Z helpers
# ============================================================

def _compute_place_robot_z(
    item: GroceryItem,
    bag_state: BagState,
    spot,
) -> tuple[float, float, float, str]:
    """Compute robot Z to release `item` at (spot.x, spot.y) in bag coords.

    Convention (must match _compute_grasp_robot_z):
        gripper_tip_Z = robot_Z - GRIPPER_OFFSET_MM
        BAG_FLOOR_GRIPPER_Z_MM = robot_Z when EMPTY gripper tip is at bag floor.

    Releasing an item so its bottom lands at (floor + stack + gap), with the
    item held near its top so it dangles ~item_height below the tip:
        robot_Z = BAG_FLOOR_GRIPPER_Z_MM + stack_top + item_height + gap
    The tip-vs-robot offset is already baked into the calibrated
    BAG_FLOOR_GRIPPER_Z_MM, so it does NOT appear again here.
    """
    item_height_mm = _item_height_mm(item)
    stack_top_mm = bag_state.stack_height_at_mm(spot.x, spot.y)
    place_z = (
        BAG_FLOOR_GRIPPER_Z_MM
        + stack_top_mm
        + item_height_mm
        + PLACE_RELEASE_GAP_MM
    )
    debug = (
        f"floor({BAG_FLOOR_GRIPPER_Z_MM:.1f}) + stack({stack_top_mm:.1f}) "
        f"+ item({item_height_mm:.1f}) + gap({PLACE_RELEASE_GAP_MM:.1f}) "
        f"= {place_z:.1f} mm"
    )
    return place_z, item_height_mm, stack_top_mm, debug


def _validate_z_command(z_mm: float, label: str) -> str | None:
    """Return None if z_mm is a safe robot Z to command, else a reason string."""
    if not np.isfinite(z_mm):
        return f"{label} z={z_mm} is not finite"
    if z_mm < 0.0:
        return (
            f"{label} z={z_mm:.1f} mm < 0. "
            f"BAG_FLOOR_GRIPPER_Z_MM={BAG_FLOOR_GRIPPER_Z_MM:.1f} likely needs calibration."
        )
    if z_mm > Z_MAX_MM + 1e-6:
        return f"{label} z={z_mm:.1f} mm > Z_MAX_MM={Z_MAX_MM:.1f}"
    return None


# ============================================================
# Survey (wraps vision.survey + overhead matching)
# ============================================================

@dataclass
class SurveyDisplayFreeze:
    overhead_frame: np.ndarray | None
    left_frame: np.ndarray | None
    right_frame: np.ndarray | None


@dataclass
class SurveyResult:
    candidates: list[ObjectCandidate]
    freeze: SurveyDisplayFreeze | None


def _capture_stereo_freeze_frames(
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict,
    rectifier: StereoRectifier,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        _, left_raw, right_raw, _, _ = read_stereo_tags_once(
            stereo, detector, stereo_calib
        )
        if left_raw is None or right_raw is None:
            return None, None
        return rectifier.rectify(left_raw, right_raw)
    except Exception as exc:
        print(f"[SURVEY] stereo freeze frame unavailable: {exc}")
        return None, None


def _format_candidate_row(c: ObjectCandidate, choice_index: int) -> str:
    """One-line per-candidate diagnostic summary, aligned for scanning."""
    xy_src = c.target_xy_source_effective
    src_tag = {
        "merged_overhead_stereo": "MERGED",
        "overhead_homography": "OVHD",
    }.get(xy_src, "STEREO")
    pc_h = "--" if c.pointcloud_height_cm is None else f"{c.pointcloud_height_cm:.1f}cm"
    return (
        f"  [{choice_index}] {c.yolo.class_name:14s} "
        f"XY=({c.target_xy[0]:7.1f},{c.target_xy[1]:7.1f}) "
        f"Z={c.object_robot_xyz_raw[2]:6.1f}mm  "
        f"h={pc_h:>7s}  "
        f"src={src_tag:6s}  "
        f"pts={c.valid_point_count:5d}  "
        f"sup={c.support_distance_mm:5.1f}mm  "
        f"phi={_fmt_candidate_phi(c)}  "
        f"score={candidate_score(c):.2f}"
    )


def do_survey(
    overhead: SimpleOverheadCamera,
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict,
    rectifier: StereoRectifier,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    robot: Robot,
    bundle: dict,
) -> SurveyResult | None:
    """Run full survey: stereo pipeline + overhead YOLO match + XY re-apply."""
    _hr("SURVEY", "-")
    t0 = time.perf_counter()
    _print_fk(robot, "[SURVEY] FK before:")

    # 1. Stereo burst survey
    raw_candidates = run_survey_workspace(
        stereo=stereo,
        detector=detector,
        stereo_calib=stereo_calib,
        rectifier=rectifier,
        yolo=yolo,
        raft=raft,
        robot=robot,
        bundle=bundle,
    )
    t_stereo = time.perf_counter()
    print(f"[SURVEY] stereo pipeline: {len(raw_candidates)} candidate(s) in {t_stereo - t0:.2f}s")

    if not raw_candidates:
        print("[SURVEY] no stereo candidates found.")
        return SurveyResult([], None)

    # 2. Capture one overhead frame for YOLO
    ok, overhead_frame = overhead.read()
    if ok and overhead_frame is not None:
        overhead_dets = yolo.segment(overhead_frame)
        if TARGET_CLASS_NAMES:
            overhead_dets = [d for d in overhead_dets if d.class_name in TARGET_CLASS_NAMES]
        print(f"[SURVEY] overhead YOLO: {len(overhead_dets)} detection(s)")
        for i, d in enumerate(overhead_dets):
            print(f"           [{i}] {d.class_name:14s} conf={d.confidence:.2f}  centroid_px=({d.centroid_px[0]:.0f},{d.centroid_px[1]:.0f})")
    else:
        overhead_dets = []
        print("[SURVEY] overhead frame unavailable; XY will fall back to stereo only")

    # 3. Match overhead -> candidates
    _match_overhead_detections_to_candidates(overhead_dets, raw_candidates, bundle)

    # 4. Clamp floor Z, re-apply overhead XY blend, then apply Z policy
    for cand in raw_candidates:
        _clamp_candidate_z_floor(cand)
        _reapply_overhead_xy(cand, bundle)
        _apply_pointcloud_size_blend(cand)
        _apply_simple_pick_z(cand)

    overhead_matched = sum(1 for c in raw_candidates if c.overhead_centroid_px is not None)
    if PICK_PHI_MODE == "overhead_minor_axis":
        missing_phi = [c for c in raw_candidates if c.pick_phi_source == "overhead_minor_axis_pending"]
        if missing_phi:
            print(f"[SURVEY] {len(missing_phi)} candidate(s) had no overhead phi match; will keep current wrist phi if picked.")

    t_total = time.perf_counter() - t0
    print(
        f"[SURVEY] DONE: {len(raw_candidates)} candidate(s), "
        f"{overhead_matched} with overhead XY blend ({t_total:.2f}s total)"
    )

    left_freeze, right_freeze = _capture_stereo_freeze_frames(
        stereo, detector, stereo_calib, rectifier
    )
    return SurveyResult(
        candidates=raw_candidates,
        freeze=SurveyDisplayFreeze(
            overhead_frame=overhead_frame.copy() if overhead_frame is not None else None,
            left_frame=left_freeze.copy() if left_freeze is not None else None,
            right_frame=right_freeze.copy() if right_freeze is not None else None,
        ),
    )


def _overlay_yolo_debug(
    frame: np.ndarray | None,
    detections: list[YOLODetection],
    tag: str,
) -> np.ndarray | None:
    if frame is None:
        return None
    out = frame.copy()
    if not detections:
        cv2.putText(out, f"{tag}: no masks", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (110, 110, 110), 1, cv2.LINE_AA)
        return out

    h, w = out.shape[:2]
    for i, det in enumerate(detections):
        color = (
            int((37 * (i + 3)) % 255),
            int((97 * (i + 5)) % 255),
            int((167 * (i + 7)) % 255),
        )
        mask = det.mask.astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        overlay = out.copy()
        overlay[mask] = color
        out = cv2.addWeighted(overlay, DEBUG_MASK_ALPHA, out, 1.0 - DEBUG_MASK_ALPHA, 0.0)

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cx = int(round(float(det.centroid_px[0])))
        cy = int(round(float(det.centroid_px[1])))
        cv2.circle(out, (cx, cy), 4, (255, 255, 255), -1)

        phi = float(det.minor_axis_angle_deg)
        half_len = max(12, int(round(det.minor_axis_length_px * 0.5)))
        rad = np.deg2rad(phi)
        dx = int(round(np.cos(rad) * half_len))
        dy = int(round(np.sin(rad) * half_len))
        cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 255, 255), 2)

        lbl = f"{det.class_name} {det.confidence:.2f} phi={phi:.1f}"
        cv2.putText(out, lbl, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 2, cv2.LINE_AA)
    return out


def _handle_runtime_error(exc: Exception) -> bool:
    print(f"\n[ERROR] {exc}")
    traceback.print_exc()
    print("[ERROR] Press 't' to try again, or 'q' to quit.")
    while True:
        k = read_command_key(delay_ms=100)
        if k in ("t", " ", "r"):
            print("[ERROR] retrying...")
            return True
        if k in ("q", "\x1b"):
            print("[ERROR] quit requested after error")
            return False


# ============================================================
# Motion helpers
# ============================================================

def _choose_safe_travel_z(current_z: float, target_z: float) -> float:
    """Single safe travel altitude: Z_MAX_MM. Always go high between XY moves."""
    return float(Z_MAX_MM)


def _item_height_mm(item: GroceryItem) -> float:
    h_cm = getattr(item, "padded_height_cm", None)
    if h_cm is not None:
        return float(h_cm) * 10.0
    h_mm = getattr(item, "padded_height_mm", None)
    if h_mm is not None:
        return float(h_mm)
    box = getattr(item, "padded_box_xyz_cm", None)
    try:
        if box is not None and len(box) >= 3:
            return float(box[2]) * 10.0
    except (TypeError, ValueError):
        pass
    return float(item.height_cm) * 10.0


def _fmt_rect_cm(rect: object) -> str:
    if rect is None:
        return "--"
    try:
        w, d = rect
        return f"{float(w):.1f}x{float(d):.1f}cm"
    except (TypeError, ValueError):
        return "--"


def _grocery_size_debug(item: GroceryItem) -> str:
    metadata = item.metadata
    measured = _fmt_rect_cm(metadata.get("measured_rect_xy_cm"))
    padded = _fmt_rect_cm(item.padded_rect_xy_cm)
    return (
        f"measured={measured} padded={padded} "
        f"h={item.height_cm:.1f}cm "
        f"src_xy={metadata.get('size_source_xy_detail') or metadata.get('size_source_xy')} "
        f"src_z={metadata.get('size_source_z')}"
    )


def _grocery_size_debug_from_candidate(cand: ObjectCandidate) -> tuple[GroceryItem | None, str]:
    try:
        item = GroceryItem.from_object_candidate(cand)
    except Exception as exc:
        return None, f"size_error={exc}"
    return item, _grocery_size_debug(item)


def _candidate_phi_or_current(robot: Robot, candidate: ObjectCandidate) -> float:
    if candidate.pick_phi_deg is not None:
        return float(candidate.pick_phi_deg)
    _, _, _, phi_fk = robot.fk()
    return float(phi_fk)


def _fmt_candidate_phi(candidate: ObjectCandidate) -> str:
    if candidate.pick_phi_deg is None:
        return f"current ({candidate.pick_phi_source})"
    return f"{float(candidate.pick_phi_deg):+6.1f} ({candidate.pick_phi_source})"


def execute_hover_to_object(robot: Robot, candidate: ObjectCandidate) -> bool:
    """Raise to safe Z, move XY over object, then descend to hover Z."""
    _hr("HOVER", "-")
    x, y = float(candidate.target_xy[0]), float(candidate.target_xy[1])
    hover_z = float(candidate.hover_robot_z)
    phi = _candidate_phi_or_current(robot, candidate)
    cx, cy, cz, _ = robot.fk()
    travel_z = _choose_safe_travel_z(cz, hover_z)

    print(
        f"[HOVER] cand[{candidate.index}] {candidate.yolo.class_name}: "
        f"xy=({x:.1f},{y:.1f}) phi={phi:+6.1f} hover_z={hover_z:.1f} travel_z={travel_z:.1f}"
    )

    if not move_cartesian_nonnegative_z(robot, "[HOVER] raise", z_mm=travel_z, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[HOVER] raise verify", target_z=travel_z)

    if not move_cartesian_nonnegative_z(robot, "[HOVER] XY+phi", x_mm=x, y_mm=y, phi_deg=phi, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[HOVER] XY+phi verify", target_x=x, target_y=y, target_phi=phi)

    if not move_cartesian_nonnegative_z(robot, "[HOVER] descend", z_mm=hover_z, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[HOVER] descend verify", target_z=hover_z)
    print("[HOVER] done.")
    return True


def execute_pick(robot: Robot, candidate: ObjectCandidate) -> bool:
    """Full pick sequence."""
    _hr("PICK", "-")
    x, y = float(candidate.target_xy[0]), float(candidate.target_xy[1])
    hover_z = float(candidate.hover_robot_z)
    pointcloud_z = float(candidate.object_robot_xyz_raw[2])
    cx, cy, cz, _ = robot.fk()
    grasp_z = _compute_grasp_robot_z(candidate)
    candidate.grasp_robot_z = float(grasp_z)
    phi = _candidate_phi_or_current(robot, candidate)
    travel_z = _choose_safe_travel_z(cz, hover_z)

    # Validate everything before any motion
    for label, z in (("travel", travel_z), ("hover", hover_z), ("grasp", grasp_z)):
        reason = _validate_z_command(z, f"[PICK] {label}")
        if reason is not None:
            print(f"[PICK] ABORT (no motion issued): {reason}")
            return False

    if REFUSE_PICK_IF_TOO_FEW_POINTS and candidate.valid_point_count < MIN_VALID_OBJECT_POINTS:
        print(f"[PICK] REFUSE: only {candidate.valid_point_count} valid points < {MIN_VALID_OBJECT_POINTS}")
        return False

    if candidate.support_distance_mm > MAX_NEAREST_SUPPORT_DIST_MM:
        print(
            f"[PICK] REFUSE: support_dist={candidate.support_distance_mm:.1f} mm "
            f"> {MAX_NEAREST_SUPPORT_DIST_MM:.1f} mm — possibly bad calibration"
        )
        return False

    print(
        f"[PICK] cand[{candidate.index}] {candidate.yolo.class_name}  "
        f"xy=({x:.1f},{y:.1f})  phi={phi:+6.1f} ({candidate.pick_phi_source})"
    )
    print(
        f"[PICK] z plan:  travel={travel_z:.1f}  hover={hover_z:.1f}  "
        f"grasp={grasp_z:.1f}  (z_stereo={pointcloud_z:.1f} + offset={GRIPPER_OFFSET_MM:.1f})"
    )
    print(f"[PICK] quality: pts={candidate.valid_point_count}  support={candidate.support_distance_mm:.1f}mm")

    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)
    print(f"[PICK] claw open ({CLAW_OPEN_DEG}°)")

    if not move_cartesian_nonnegative_z(robot, "[PICK] raise", z_mm=travel_z, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[PICK] raise verify", target_z=travel_z)

    if not move_cartesian_nonnegative_z(robot, "[PICK] XY+phi", x_mm=x, y_mm=y, phi_deg=phi, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[PICK] XY+phi verify", target_x=x, target_y=y, target_phi=phi)

    if not move_cartesian_nonnegative_z(robot, "[PICK] hover", z_mm=hover_z, move_time_s=PICK_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[PICK] hover verify", target_z=hover_z)

    if not move_cartesian_nonnegative_z(robot, "[PICK] grasp", z_mm=grasp_z, move_time_s=PICK_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[PICK] grasp verify", target_z=grasp_z)

    robot.servo(CLAW_CLOSED_DEG)
    time.sleep(CLAW_SETTLE_S)
    print(f"[PICK] claw closed ({CLAW_CLOSED_DEG}°)")

    if not move_cartesian_nonnegative_z(robot, "[PICK] raise after grasp", z_mm=hover_z, move_time_s=PICK_MOVE_TIME_S):
        return False
    _print_motion_verify(robot, "[PICK] raise verify", target_z=hover_z)

    print("[PICK] OK — item should be in gripper.")
    return True


def execute_place_blb(
    robot: Robot,
    candidate: ObjectCandidate,
    bag_state: BagState,
    bundle: dict,
) -> tuple[bool, PlacedItem | None]:
    """Plan BLB placement and execute place.

    All motion targets are computed and validated up front. If any target is
    out of bounds, the function aborts BEFORE issuing any motion, so the robot
    stays put with the item still held and you can investigate.
    """
    _hr("PLACE (BLB)", "-")

    try:
        item = GroceryItem.from_object_candidate(candidate)
    except Exception as exc:
        print(f"[PLACE] GroceryItem construction failed: {exc}")
        return False, None
    print(f"[PLACE] item: {item.class_name}  {_grocery_size_debug(item)}")

    spot = choose_placement_spot_2d(item, bag_state)
    if spot is None:
        print("[PLACE] BLB: no valid placement spot found — bag may be full.")
        return False, None

    # --- Pre-compute all motion targets ---
    place_x = BAG_ORIGIN_X_MM + spot.x * 10.0
    place_y = BAG_ORIGIN_Y_MM + spot.y * 10.0
    place_phi = (
        BAG_PLACE_PHI_DEG
        if BAG_PLACE_PHI_DEG is not None
        else _candidate_phi_or_current(robot, candidate)
    )
    place_z, item_height_mm, stack_top_mm, place_debug = _compute_place_robot_z(
        item, bag_state, spot
    )
    hover_z = float(Z_MAX_MM)
    cx, cy, cz, _ = robot.fk()
    travel_z = _choose_safe_travel_z(cz, hover_z)

    print(
        f"[PLACE] BLB spot bag=({spot.x:5.1f},{spot.y:5.1f}) cm  "
        f"-> robot=({place_x:7.1f},{place_y:7.1f}) mm  phi={place_phi:+6.1f}"
    )
    print(f"[PLACE] z plan: {place_debug}")
    print(f"[PLACE] hover_z={hover_z:.1f}  travel_z={travel_z:.1f}")
    print(
        f"[PLACE] bag state before: {len(bag_state.placed_items)} item(s), "
        f"{bag_state.free_area_estimate_cm2():.0f} cm² free"
    )

    # --- Validate ALL Z targets before any motion ---
    for label, z in (("travel", travel_z), ("hover", hover_z), ("place", place_z)):
        reason = _validate_z_command(z, f"[PLACE] {label}")
        if reason is not None:
            print(f"[PLACE] ABORT (no motion issued, item still held): {reason}")
            return False, None

    # --- Execute motion sequence ---
    if not move_cartesian_nonnegative_z(robot, "[PLACE] raise", z_mm=travel_z, move_time_s=COARSE_MOVE_TIME_S):
        return False, None
    _print_motion_verify(robot, "[PLACE] raise verify", target_z=travel_z)

    if not move_cartesian_nonnegative_z(
        robot, "[PLACE] XY+phi",
        x_mm=place_x, y_mm=place_y, phi_deg=place_phi,
        move_time_s=COARSE_MOVE_TIME_S,
    ):
        return False, None
    _print_motion_verify(robot, "[PLACE] XY+phi verify", target_x=place_x, target_y=place_y, target_phi=place_phi)

    if not move_cartesian_nonnegative_z(robot, "[PLACE] hover", z_mm=hover_z, move_time_s=COARSE_MOVE_TIME_S):
        return False, None
    _print_motion_verify(robot, "[PLACE] hover verify", target_z=hover_z)

    if not move_cartesian_nonnegative_z(robot, "[PLACE] lower", z_mm=place_z, move_time_s=PICK_MOVE_TIME_S):
        return False, None
    _print_motion_verify(robot, "[PLACE] lower verify", target_z=place_z)

    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)
    print(f"[PLACE] claw opened ({CLAW_OPEN_DEG}°) — item released")

    if not move_cartesian_nonnegative_z(
        robot, "[PLACE] raise after release",
        z_mm=hover_z, move_time_s=PICK_MOVE_TIME_S,
    ):
        return False, None
    _print_motion_verify(robot, "[PLACE] raise verify", target_z=hover_z)

    placed = bag_state.add(item, spot.x, spot.y)
    print(
        f"[PLACE] OK — bag now {len(bag_state.placed_items)} item(s), "
        f"{bag_state.free_area_estimate_cm2():.0f} cm² free"
    )
    return True, placed


# ============================================================
# Auto-cycle
# ============================================================

def execute_auto_cycle(
    robot: Robot,
    candidates: list[ObjectCandidate],
    bag_state: BagState,
    bundle: dict,
) -> tuple[list[ObjectCandidate], ObjectCandidate | None]:
    """Auto pick-and-place until exhausted or a step fails.

    Returns (remaining_candidates, still_held_candidate). still_held is non-None
    iff the last pick succeeded but its place failed — caller can bind it to
    last_picked_candidate so the user can press 'f' to retry place.
    """
    _hr("AUTO CYCLE", "=")
    if not candidates:
        print("[AUTO] no candidates — survey first")
        return candidates, None

    print(f"[AUTO] starting cycle: {len(candidates)} candidate(s)")
    remaining = list(candidates)
    held: ObjectCandidate | None = None
    placed_count = 0
    fail_count = 0

    while remaining:
        cand = remaining[0]
        print(f"\n[AUTO] iter {placed_count + 1}: picking [{cand.index}] {cand.yolo.class_name}")
        if not execute_pick(robot, cand):
            print(f"[AUTO] pick failed for [{cand.index}], stopping cycle.")
            fail_count += 1
            break
        held = cand

        ok_place, _ = execute_place_blb(robot, cand, bag_state, bundle)
        if not ok_place:
            print(
                "[AUTO] place failed — item still held.\n"
                "       Press 'f' to retry place after fixing the issue."
            )
            fail_count += 1
            break
        held = None
        placed_count += 1
        remaining.pop(0)
        print(f"[AUTO] iter {placed_count} done; {len(remaining)} candidate(s) left.")

    _hr("AUTO CYCLE END", "=")
    print(
        f"[AUTO] summary: placed={placed_count}  failed={fail_count}  "
        f"unplaced={len(remaining)}  holding={held is not None}"
    )
    return remaining, held


# ============================================================
# Bag state display
# ============================================================

def print_bag_state(bag_state: BagState) -> None:
    _hr(f"BAG STATE ({_BAG_WIDTH_CM:.1f} x {_BAG_DEPTH_CM:.1f} cm)", "-")
    print(
        f"[BAG] {len(bag_state.placed_items)} placed  "
        f"free={bag_state.free_area_estimate_cm2():.0f} cm²"
    )
    if not bag_state.placed_items:
        print("[BAG] (empty)")
        return
    for i, p in enumerate(bag_state.placed_items, 1):
        h_mm = _item_height_mm(p.item)
        print(
            f"  [{i:2d}] {p.item.class_name:18s} "
            f"@bag=({p.x:5.1f},{p.y:5.1f}) cm  "
            f"size={p.w:.1f}x{p.d:.1f}cm  h={h_mm:.0f}mm"
        )


# ============================================================
# Drawing helpers
# ============================================================

def _draw_candidates_overhead(
    frame: np.ndarray,
    candidates: list[ObjectCandidate],
    selected_index: int | None,
) -> np.ndarray:
    out = frame.copy()
    for choice_index, cand in enumerate(candidates):
        if cand.overhead_centroid_px is None:
            continue
        px = int(round(cand.overhead_centroid_px[0]))
        py = int(round(cand.overhead_centroid_px[1]))
        is_sel = (choice_index == selected_index)
        color = (0, 255, 0) if is_sel else (0, 200, 255)
        radius = 12 if is_sel else 8
        cv2.circle(out, (px, py), radius, color, 2)
        cv2.putText(out, str(choice_index + 1), (px + 14, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def _draw_bag_overlay(
    frame: np.ndarray,
    bag_state: BagState,
    bundle: dict,
) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"Bag: {len(bag_state.placed_items)} placed",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2, cv2.LINE_AA)
    return out


def _make_status_bar(
    candidates: list[ObjectCandidate],
    selected_index: int | None,
    bag_state: BagState,
    width: int,
) -> np.ndarray:
    lines: list[str] = []
    if not candidates:
        lines.append("No survey yet. Press 's' to survey.")
    else:
        for choice_index, cand in enumerate(candidates):
            is_sel = choice_index == selected_index
            xy_src = cand.target_xy_source_effective
            ovhd = "OH" if cand.overhead_centroid_px is not None else "--"
            marker = ">" if is_sel else " "
            lines.append(
                f"{marker}[{choice_index+1}] {cand.yolo.class_name:16s} "
                f"Z={cand.object_robot_xyz_raw[2]:5.0f}mm "
                f"XY=({cand.target_xy[0]:6.1f},{cand.target_xy[1]:6.1f}) "
                f"src={xy_src:24s} {ovhd} "
                f"pts={cand.valid_point_count}"
            )
    lines.append("")
    lines.append(f"Bag: {len(bag_state.placed_items)} placed / {_BAG_WIDTH_CM:.0f}x{_BAG_DEPTH_CM:.0f} cm")
    lines.append("Keys: s=survey  1-9=select  k=pick  f=place(BLB)  a=auto-cycle  "
                 "g=survey-pose  m=hover  w=bag  x=reset-bag  q=quit")

    row_h = 18
    h = len(lines) * row_h + 10
    bar = np.zeros((h, width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        y = i * row_h + 14
        cv2.putText(bar, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    scale = h / img.shape[0]
    w = max(1, int(img.shape[1] * scale))
    return cv2.resize(img, (w, h))


def build_display(
    overhead_frame: np.ndarray | None,
    left_live: np.ndarray | None,
    right_live: np.ndarray | None,
    candidates: list[ObjectCandidate],
    selected_index: int | None,
    bag_state: BagState,
    bundle: dict,
) -> np.ndarray:
    W = COMBINED_WIDTH_PX

    if overhead_frame is not None:
        oh = overhead_frame.copy()
        oh = _draw_candidates_overhead(oh, candidates, selected_index)
        oh = _draw_bag_overlay(oh, bag_state, bundle)
        scale = W / oh.shape[1]
        oh = cv2.resize(oh, (W, max(1, int(oh.shape[0] * scale))))
        if oh.shape[0] != OVERHEAD_DRAW_H_PX:
            oh = cv2.resize(oh, (W, OVERHEAD_DRAW_H_PX))
    else:
        oh = np.zeros((OVERHEAD_DRAW_H_PX, W, 3), dtype=np.uint8)
        cv2.putText(oh, "Overhead Camera — no frame", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)

    half_w = W // 2
    def _make_stereo_half(img: np.ndarray | None, label: str) -> np.ndarray:
        if img is not None:
            panel = _resize_to_height(img.copy(), STEREO_DRAW_H_PX)
            panel = cv2.resize(panel, (half_w, STEREO_DRAW_H_PX))
        else:
            panel = np.zeros((STEREO_DRAW_H_PX, half_w, 3), dtype=np.uint8)
        cv2.putText(panel, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (158, 200, 255), 1, cv2.LINE_AA)
        return panel

    left_panel = _make_stereo_half(left_live, "LEFT")
    right_panel = _make_stereo_half(right_live, "RIGHT")
    stereo_row = np.hstack([left_panel, right_panel])
    status = _make_status_bar(candidates, selected_index, bag_state, W)
    return np.vstack([oh, stereo_row, status])


# ============================================================
# Main
# ============================================================

def main() -> None:  # noqa: C901
    _hr("run_buildup_pickplace.py", "=")
    if PICK_PHI_MODE not in VALID_PICK_PHI_MODES:
        raise ValueError(
            f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}; "
            f"expected one of {sorted(VALID_PICK_PHI_MODES)}"
        )
    print(f"[MAIN] pick phi mode: {PICK_PHI_MODE}")
    print(f"[MAIN] Z policy: Z_MAX={Z_MAX_MM}  GRIPPER_OFFSET={GRIPPER_OFFSET_MM}  "
          f"BAG_FLOOR_GRIPPER_Z={BAG_FLOOR_GRIPPER_Z_MM}")
    print(f"[MAIN] bag: origin=({BAG_ORIGIN_X_MM},{BAG_ORIGIN_Y_MM}) mm  "
          f"size={BAG_WIDTH_MM}x{BAG_DEPTH_MM} mm")
    print(f"[MAIN] XY blend: overhead_weight={OVERHEAD_XY_BLEND_WEIGHT}  "
          f"warn_disagreement>{XY_DISAGREEMENT_WARN_MM}mm")

    # --- Device ---
    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)
    print(f"[MAIN] device: {device_info.device_str}")

    # --- Calibration ---
    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    print(f"[MAIN] calibration loaded: {BUNDLE_PATH.name}, {STEREO_CALIBRATION_PATH.name}")
    _print_bundle_summary(bundle)
    print_matrix_labeled(
        "A_robot_from_cam_xyz_3x4", bundle.get("A_robot_from_cam_xyz_3x4"),
        ["robot_x", "robot_y", "robot_z"],
        ["cam_x", "cam_y", "cam_z", "1"],
    )

    # --- Vision models ---
    yolo, raft = load_fast_vision(device_info)
    rectifier = StereoRectifier(stereo_calib)
    detector = build_detector()

    # --- Cameras ---
    print(f"[MAIN] opening overhead camera (index {OVERHEAD_INDEX})...")
    overhead = SimpleOverheadCamera(OVERHEAD_INDEX)
    print(f"[MAIN] opening stereo camera (index {STEREO_INDEX})...")
    stereo = SimpleStereoCamera(STEREO_INDEX)

    # --- Robot ---
    print("[MAIN] connecting to robot...")
    robot = Robot(connect=True)
    require_soft_limits(robot)
    robot.enable(True)
    robot.init_drivers()

    # --- Calibration sanity warning ---
    if abs(BAG_FLOOR_GRIPPER_Z_MM) < 1e-6:
        _hr("CALIBRATION WARNING", "!")
        print(
            "[WARN] BAG_FLOOR_GRIPPER_Z_MM is 0 — looks like the placeholder,\n"
            "       not a calibrated value. Place moves will abort until calibrated.\n"
            "       To calibrate: lower the EMPTY gripper tip to the empty bag floor,\n"
            "       read FK Z (press 'p'), and set BAG_FLOOR_GRIPPER_Z_MM to that value."
        )
        _hr("", "!")

    print("\nStartup options:")
    print("  h = run HOME now")
    print("  c = continue from current Teensy step counters, no homing")
    print("  a = assume robot is physically at configured home_pose, no homing")
    choice = input("Choose h/c/a: ").strip().lower()
    if choice == "h":
        if not robot.home():
            print("[MAIN] HOME failed; exiting.")
            return
    elif choice == "c":
        if not robot.sync_estimate_from_teensy_steps():
            print("[MAIN] Teensy sync failed; exiting.")
            return
        robot.print_estimate()
    elif choice == "a":
        robot.assume_homed()
        robot.print_estimate()
    else:
        print("[MAIN] Unknown startup choice; exiting.")
        return

    print("[MAIN] robot connected, soft limits active")
    robot.send('HOMEJ3')

    # --- State ---
    candidates: list[ObjectCandidate] = []
    selected_index: int | None = None
    bag_state = BagState(
        bag_width_cm=_BAG_WIDTH_CM,
        bag_depth_cm=_BAG_DEPTH_CM,
    )
    last_picked_candidate: ObjectCandidate | None = None
    frame_i = 0
    overhead_live_dets: list[YOLODetection] = []
    stereo_left_live_dets: list[YOLODetection] = []
    stereo_right_live_dets: list[YOLODetection] = []
    survey_freeze: SurveyDisplayFreeze | None = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + 120)

    print("[MAIN] ready.  Press 's' to survey, 'q' to quit.")

    running = True
    while running:
        try:
            frame_i += 1

            ok_oh, overhead_frame = overhead.read()
            if not ok_oh or overhead_frame is None:
                overhead_frame = None

            ok_st, _, left_live, right_live = stereo.read_pair()
            if not ok_st:
                left_live = right_live = None

            if (
                survey_freeze is None
                and DEBUG_MASK_OVERLAY
                and (frame_i % max(1, DEBUG_INFER_EVERY_N_FRAMES) == 0)
            ):
                if overhead_frame is not None:
                    overhead_live_dets = yolo.segment(overhead_frame)
                if left_live is not None:
                    stereo_left_live_dets = yolo.segment(left_live)
                if right_live is not None:
                    stereo_right_live_dets = yolo.segment(right_live)

            if survey_freeze is not None:
                overhead_draw = (
                    survey_freeze.overhead_frame.copy()
                    if survey_freeze.overhead_frame is not None
                    else None
                )
                left_draw = (
                    survey_freeze.left_frame.copy()
                    if survey_freeze.left_frame is not None
                    else None
                )
                right_draw = (
                    survey_freeze.right_frame.copy()
                    if survey_freeze.right_frame is not None
                    else None
                )
            else:
                overhead_draw = _overlay_yolo_debug(overhead_frame, overhead_live_dets, "OVERHEAD")
                left_draw = _overlay_yolo_debug(left_live, stereo_left_live_dets, "LEFT")
                right_draw = _overlay_yolo_debug(right_live, stereo_right_live_dets, "RIGHT")

            display = build_display(
                overhead_frame=overhead_draw,
                left_live=left_draw,
                right_live=right_draw,
                candidates=candidates,
                selected_index=selected_index,
                bag_state=bag_state,
                bundle=bundle,
            )
            cv2.imshow(WINDOW, display)

            key = read_command_key(delay_ms=1)
            if key is None:
                continue

            # ---- Survey ----
            if key == "s":
                survey_freeze = None
                result = do_survey(
                    overhead=overhead, stereo=stereo, detector=detector,
                    stereo_calib=stereo_calib, rectifier=rectifier,
                    yolo=yolo, raft=raft, robot=robot, bundle=bundle,
                )
                if result is not None:
                    candidates = result.candidates
                    survey_freeze = result.freeze if candidates else None
                    selected_index = 0 if candidates else None
                    if candidates:
                        _hr("SURVEY RESULTS (scored)", "-")
                        for choice_index, c in enumerate(candidates, start=1):
                            print(_format_candidate_row(c, choice_index))
                        print(f"\n[SURVEY] Selected: [{selected_index + 1}]")
                    if survey_freeze is not None:
                        print("[DISPLAY] survey frame frozen for candidate selection")

            # ---- Candidate selection ----
            elif key.isdigit() and key != "0":
                idx = int(key) - 1
                if idx < len(candidates):
                    selected_index = idx
                    cand = candidates[idx]
                    _hr(f"SELECTED [{idx+1}] {cand.yolo.class_name}", "-")
                    print(_format_candidate_row(cand, idx + 1))
                    print(
                        f"      z plan: z_stereo={cand.object_robot_xyz_raw[2]:.1f}  "
                        f"hover={cand.hover_robot_z:.1f}  grasp={cand.grasp_robot_z:.1f}"
                    )
                else:
                    print(f"[SELECT] no candidate {idx + 1}")

            # ---- Survey pose ----
            elif key == "g":
                survey_freeze = None
                _hr("MOVE TO SURVEY POSE", "-")
                move_cartesian_nonnegative_z(
                    robot, "[POSE]",
                    x_mm=SURVEY_POSE_X_MM, y_mm=SURVEY_POSE_Y_MM,
                    z_mm=SURVEY_POSE_Z_MM, phi_deg=SURVEY_POSE_PHI_DEG,
                    move_time_s=COARSE_MOVE_TIME_S,
                )
                _print_motion_verify(
                    robot, "[POSE] verify",
                    target_x=SURVEY_POSE_X_MM, target_y=SURVEY_POSE_Y_MM,
                    target_z=SURVEY_POSE_Z_MM, target_phi=SURVEY_POSE_PHI_DEG,
                )

            # ---- Hover ----
            elif key == "m":
                survey_freeze = None
                if selected_index is None or selected_index >= len(candidates):
                    print("[HOVER] no candidate selected")
                else:
                    execute_hover_to_object(robot, candidates[selected_index])

            # ---- Pick ----
            elif key == "k":
                survey_freeze = None
                if selected_index is None or selected_index >= len(candidates):
                    print("[PICK] no candidate selected")
                else:
                    cand = candidates[selected_index]
                    ok = execute_pick(robot, cand)
                    if ok:
                        last_picked_candidate = cand
                        candidates = [c for c in candidates if c.index != cand.index]
                        selected_index = 0 if candidates else None
                        print("[PICK] success.  Press 'f' to place with BLB, or 's' to re-survey.")
                    else:
                        print("[PICK] failed.")

            # ---- Place ----
            elif key == "f":
                survey_freeze = None
                if last_picked_candidate is None:
                    print("[PLACE] nothing picked yet")
                else:
                    ok, _ = execute_place_blb(
                        robot=robot, candidate=last_picked_candidate,
                        bag_state=bag_state, bundle=bundle,
                    )
                    if ok:
                        last_picked_candidate = None
                    else:
                        print("[PLACE] failed — item still held.")

            # ---- Auto-cycle ----
            elif key == "a":
                survey_freeze = None
                if not candidates:
                    print("[AUTO] no candidates — survey first")
                else:
                    candidates, held = execute_auto_cycle(
                        robot=robot, candidates=candidates,
                        bag_state=bag_state, bundle=bundle,
                    )
                    selected_index = 0 if candidates else None
                    last_picked_candidate = held  # None unless place failed mid-cycle

            # ---- Bag state ----
            elif key == "w":
                print_bag_state(bag_state)

            elif key == "x":
                n = len(bag_state.placed_items)
                bag_state.placed_items.clear()
                print(f"[BAG] cleared {n} placed item(s).")

            # ---- Geometry sanity ----
            elif key == "b":
                _hr("GEOMETRY SANITY", "-")
                x, y, z, phi = robot.fk()
                print(f"[SANITY] FK: x={x:.1f} y={y:.1f} z={z:.1f} phi={phi:.1f}")
                if selected_index is not None and selected_index < len(candidates):
                    c = candidates[selected_index]
                    print(f"[SANITY] selected [{selected_index + 1}] {c.yolo.class_name}")
                    print(f"  object_robot_xyz_raw  = {np.round(c.object_robot_xyz_raw, 1)}")
                    print(f"  object_robot_xyz_corr = {np.round(c.object_robot_xyz_corrected, 1)}")
                    print(f"  target_xy             = {np.round(c.target_xy, 1)}")
                    print(f"  target_xy_source      = {c.target_xy_source_effective}")
                    print(f"  stereo_z_bias_mm      = {c.stereo_z_bias_mm:.2f}")
                    print(f"  lookup_z_used         = {c.lookup_z_used:.1f}  clamped={c.lookup_z_clamped}")
                    print(f"  support_dist          = {c.support_distance_mm:.1f} mm")
                    print(f"  hover_robot_z         = {c.hover_robot_z:.1f}")
                    print(f"  grasp_robot_z         = {c.grasp_robot_z:.1f}")
                    print(f"  valid_point_count     = {c.valid_point_count}")
                    print(f"  overhead_centroid_px  = {c.overhead_centroid_px}")
                    print(f"  overhead_bbox_px      = {c.overhead_bbox_px}")
                    print(f"  topdown_aabb_cm       = {c.topdown_aabb_cm}")
                    print(f"  topdown_oriented_cm   = {c.topdown_oriented_rect_cm}")
                    print(f"  pointcloud_height_cm  = {c.pointcloud_height_cm}")
                    print(f"  pointcloud_z_range_mm = {c.pointcloud_height_robot_z_range_mm}")
                    _, size_dbg = _grocery_size_debug_from_candidate(c)
                    print(f"  size_estimate         = {size_dbg}")

            # ---- Calibration matrices ----
            elif key == "r":
                _hr("CALIBRATION MATRICES", "-")
                print_matrix_labeled(
                    "A_robot_from_cam_xyz_3x4", bundle.get("A_robot_from_cam_xyz_3x4"),
                    ["robot_x", "robot_y", "robot_z"],
                    ["cam_x", "cam_y", "cam_z", "1"],
                )
                print_matrix_labeled(
                    "overhead_camera_matrix", bundle.get("overhead_camera_matrix"),
                    ["row_0", "row_1", "row_2"],
                    ["col_0", "col_1", "col_2"],
                )
                z_levels = bundle.get("z_levels_mm")
                if z_levels is not None:
                    print(f"[CAL] z_levels_mm: {np.round(z_levels, 1)}")
                _print_bundle_summary(bundle)

            # ---- FK print ----
            elif key == "p":
                _print_fk(robot, "[FK]")

            # ---- Jog Z ----
            elif key == "[":
                jog_nonnegative_z(robot, "[JOG]", dz=-Z_JOG_MM, move_time_s=0.3)
                _print_fk(robot, "[JOG]")
            elif key == "]":
                jog_nonnegative_z(robot, "[JOG]", dz=+Z_JOG_MM, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            # ---- Jog phi ----
            elif key == ",":
                jog_nonnegative_z(robot, "[JOG]", dphi=-PHI_JOG_DEG, move_time_s=0.3)
                _print_fk(robot, "[JOG]")
            elif key == ".":
                jog_nonnegative_z(robot, "[JOG]", dphi=+PHI_JOG_DEG, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            # ---- Claw ----
            elif key == "o":
                robot.servo(CLAW_OPEN_DEG)
                print(f"[CLAW] open ({CLAW_OPEN_DEG}°)")
            elif key == "l":
                robot.servo(CLAW_CLOSED_DEG)
                print(f"[CLAW] closed ({CLAW_CLOSED_DEG}°)")

            # ---- Home / sync ----
            elif key == "h":
                robot.home()
                print("[ROBOT] homing...")
            elif key == "c":
                robot.sync_estimate_from_teensy_steps()
                print("[ROBOT] sync estimate from steps")
            elif key == "u":
                robot.assume_homed()
                print("[ROBOT] assumed homed")

            # ---- Motors ----
            elif key == "e":
                robot.enable(True)
                print("[ROBOT] motors enabled")
            elif key == "d":
                robot.enable(False)
                print("[ROBOT] motors disabled")

            # ---- Quit ----
            elif key in ("q", "\x1b"):
                print("[MAIN] quit requested")
                running = False

        except Exception as exc:
            if not _handle_runtime_error(exc):
                running = False

    cv2.destroyAllWindows()
    print("[MAIN] done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] interrupted by user")
        cv2.destroyAllWindows()
    except Exception:
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
