from __future__ import annotations

"""
Interactive single-object placement repeatability test.

This is intentionally not full bagging. It surveys real objects, picks one
selected candidate, then places the held object into one saved placement zone.
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

SURFACE_ZONE_CONFIG_PATH = Path("config/surface_zones.json")
PLACE_SURFACE_ZONE_NAME = "Test_Zone_Place_V1"

PLACE_ZONE_CONFIG_PATH = Path("config/place_zones.json")
PLACE_ZONE_NAME = "Test_Zone_Place_V1"

YOLO_IMGSZ: int = 640
YOLO_CONF: float = 0.35
YOLO_IOU: float = 0.50
YOLO_RETINA_MASKS: bool = True
TARGET_CLASS_NAMES: list[str] = []

USE_CUDA: bool = True
USE_HALF: bool = True
RAFT_VALID_ITERS: int = 16
RAFT_DOWNSCALE: float = 1.0
RAFT_MIXED_PRECISION: bool = True

MIN_MASK_AREA_PX: int = 500
MIN_DISPARITY_PX: float = 1.0
MIN_VALID_OBJECT_POINTS: int = 300

BURST_COUNT: int = 10
MIN_BURST_HITS: int = 3
BURST_FRAME_DELAY_S: float = 0.05
BURST_CLUSTER_MAX_CENTROID_PX: float = 75.0
BURST_REQUIRE_SAME_CLASS: bool = True

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

Z_MAX_MM: float = 275.0
PLACE_APPROACH_Z_MM: float = Z_MAX_MM
GRIPPER_OFFSET_MM: float = 120.0
HOVER_HEIGHT_MM: float = Z_MAX_MM
GRASP_OFFSET_MM: float = GRIPPER_OFFSET_MM
USE_ROBUST_OBJECT_Z = True
ROBUST_TOP_PERCENTILE = 95.0
ROBUST_BOTTOM_PERCENTILE = 5.0
TOP_SPREAD_LOW_PERCENTILE = 90.0
TOP_SPREAD_HIGH_PERCENTILE = 99.0
Z_UNCERTAINTY_CLEARANCE_GAIN = 1.0
Z_UNCERTAINTY_CLEARANCE_MIN_MM = 0.0
Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
Z_UNCERTAINTY_WARN_MM = 10.0
PICK_EXTRA_CLEARANCE_MM = 0.0
PLACE_RELEASE_GAP_MM = 20.0
PLACE_Z_UNCERTAINTY_GAIN = 0.1
PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 5.0
PICK_Z_UNCERTAINTY_GAIN = 1.0
PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
MIN_OBJECT_HEIGHT_MM = 2.0
MAX_OBJECT_HEIGHT_MM = 180.0

OVERHEAD_MATCH_MAX_DIST_MM: float = 140.0
OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True
OVERHEAD_XY_BLEND_WEIGHT: float = 0.45
OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI: bool = True

USE_CONFIDENCE_PHI_BLEND: bool = True
PHI_DISAGREEMENT_WARN_DEG: float = 25.0
PHI_MIN_CONFIDENCE: float = 0.20
PHI_ASPECT_DECAY: float = 0.8
PHI_STEREO_HEIGHT_DECAY_CM: float = 8.0
PHI_FALLBACK_TO_CURRENT_EE_PHI: bool = False

COARSE_MOVE_TIME_S: float = 1.10
XY_MOVE_TIME_S: float = 1.50
PICK_Z_MOVE_TIME_S: float = 0.60
PLACE_Z_MOVE_TIME_S: float = 0.60
PLACE_RELEASE_DWELL_S: float = 0.3

CONNECT_ROBOT: bool = True
ENABLE_MOTORS_ON_START: bool = True
INIT_DRIVERS_ON_START: bool = True

COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 560
STEREO_DRAW_H_PX: int = 390
STATUS_H_PX: int = 140
WINDOW: str = "Pick One Place One Repeatability"

CLAW_OPEN_DEG: int = 65
CLAW_CLOSED_DEG: int = 0
CLAW_SETTLE_S: float = 0.30

REQUIRE_CONFIRM_BEFORE_PICK = False
REQUIRE_CONFIRM_BEFORE_PLACE = False
USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT = False
USE_PICK_PHI_FOR_PLACE = False
USE_LOCAL_HEIGHT_AWARE_GRASP_XY = True
LOCAL_GRASP_RADIUS_MM = 50.0
LOCAL_GRASP_TOP_REGION_PERCENTILE = 90.0
LOCAL_GRASP_HEIGHT_DELTA_THRESHOLD_MM = 5.0
LOCAL_GRASP_BLEND_WEIGHT = 0.35
LOCAL_GRASP_MAX_SHIFT_MM = 100.0

REQUIRE_OVERHEAD_XY_FOR_PICK: bool = False
REFUSE_PICK_IF_TOO_FEW_POINTS: bool = True

X_SURVEY = 100.0
Y_SURVEY = 100.0
Z_SURVEY = 250.0

# ============================================================

import math
import sys
import time
import traceback

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import motion.pick_validation_motion as _motion_mod
import scripts.pick_validation_display as _display_mod
import vision.burst_tracking as _burst_mod
import vision.object_geometry as _geom_mod
import vision.pick_candidate_builder as _candidate_mod
import vision.pick_phi_resolver as _phi_mod
import vision.pick_survey_pipeline as _survey_mod
import vision.pick_xy_resolver as _xy_mod
import vision.pick_z_resolver as _z_mod
import vision.pointcloud as _pointcloud_mod

from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from config.place_zone_io import get_place_zone
from config.surface_zone_io import get_surface_zone
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from motion.pick_z_policy import compute_pick_z_plan
from motion.place_z_policy import compute_place_z_plan
from motion.pick_validation_motion import _candidate_phi_or_current, _print_fk, startup_robot
from scripts.pick_validation_display import _fmt_xy, _hr, make_display, print_validation
from test_calibration_bundle_live_stereo_z_pickplace import (
    jog_nonnegative_z,
    load_bundle,
    load_stereo_calibration,
    move_cartesian_nonnegative_z,
    print_matrix_labeled,
    read_command_key,
)
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_survey_pipeline import load_vision, run_survey
from vision.pick_z_resolver import validate_z_command
from vision.grasp_xy_policy import compute_grasp_xy_with_local_height
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


def _configure_modules() -> None:
    _pointcloud_mod.MIN_DISPARITY_PX = MIN_DISPARITY_PX

    _geom_mod.USE_EE_FK_Z_BIAS_CORRECTION = False
    _geom_mod.TARGET_XY_SOURCE = "overhead_homography"
    _geom_mod.HOVER_HEIGHT_MM = HOVER_HEIGHT_MM
    _geom_mod.GRASP_OFFSET_MM = GRASP_OFFSET_MM
    _geom_mod.PICK_PHI_MODE = PICK_PHI_MODE

    _burst_mod.BURST_COUNT = BURST_COUNT
    _burst_mod.MIN_BURST_HITS = MIN_BURST_HITS
    _burst_mod.BURST_FRAME_DELAY_S = BURST_FRAME_DELAY_S
    _burst_mod.BURST_CLUSTER_MAX_CENTROID_PX = BURST_CLUSTER_MAX_CENTROID_PX
    _burst_mod.BURST_REQUIRE_SAME_CLASS = BURST_REQUIRE_SAME_CLASS
    _burst_mod.TARGET_CLASS_NAMES = TARGET_CLASS_NAMES

    _candidate_mod.BURST_COUNT = BURST_COUNT
    _candidate_mod.MIN_VALID_OBJECT_POINTS = MIN_VALID_OBJECT_POINTS
    _candidate_mod.OVERHEAD_XY_BLEND_WEIGHT = OVERHEAD_XY_BLEND_WEIGHT

    _z_mod.Z_MAX_MM = Z_MAX_MM
    _z_mod.GRIPPER_OFFSET_MM = GRIPPER_OFFSET_MM
    _z_mod.USE_ROBUST_OBJECT_Z = USE_ROBUST_OBJECT_Z
    _z_mod.ROBUST_TOP_PERCENTILE = ROBUST_TOP_PERCENTILE
    _z_mod.ROBUST_BOTTOM_PERCENTILE = ROBUST_BOTTOM_PERCENTILE
    _z_mod.TOP_SPREAD_LOW_PERCENTILE = TOP_SPREAD_LOW_PERCENTILE
    _z_mod.TOP_SPREAD_HIGH_PERCENTILE = TOP_SPREAD_HIGH_PERCENTILE
    _z_mod.Z_UNCERTAINTY_CLEARANCE_GAIN = PICK_Z_UNCERTAINTY_GAIN
    _z_mod.Z_UNCERTAINTY_CLEARANCE_MIN_MM = Z_UNCERTAINTY_CLEARANCE_MIN_MM
    _z_mod.Z_UNCERTAINTY_CLEARANCE_MAX_MM = PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM
    _z_mod.Z_UNCERTAINTY_WARN_MM = Z_UNCERTAINTY_WARN_MM
    _z_mod.PICK_EXTRA_CLEARANCE_MM = PICK_EXTRA_CLEARANCE_MM
    _z_mod.PLACE_RELEASE_GAP_MM = PLACE_RELEASE_GAP_MM
    _z_mod.MIN_OBJECT_HEIGHT_MM = MIN_OBJECT_HEIGHT_MM
    _z_mod.MAX_OBJECT_HEIGHT_MM = MAX_OBJECT_HEIGHT_MM

    _xy_mod.OVERHEAD_MATCH_MAX_DIST_MM = OVERHEAD_MATCH_MAX_DIST_MM
    _xy_mod.OVERHEAD_MATCH_PREFER_SAME_CLASS = OVERHEAD_MATCH_PREFER_SAME_CLASS

    _phi_mod.PICK_PHI_MODE = PICK_PHI_MODE
    _phi_mod.OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI = OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI
    _phi_mod.USE_CONFIDENCE_PHI_BLEND = USE_CONFIDENCE_PHI_BLEND
    _phi_mod.PHI_DISAGREEMENT_WARN_DEG = PHI_DISAGREEMENT_WARN_DEG
    _phi_mod.PHI_MIN_CONFIDENCE = PHI_MIN_CONFIDENCE
    _phi_mod.PHI_ASPECT_DECAY = PHI_ASPECT_DECAY
    _phi_mod.PHI_STEREO_HEIGHT_DECAY_CM = PHI_STEREO_HEIGHT_DECAY_CM
    _phi_mod.PHI_FALLBACK_TO_CURRENT_EE_PHI = PHI_FALLBACK_TO_CURRENT_EE_PHI

    _survey_mod.YOLO_WEIGHTS_PATH = YOLO_WEIGHTS_PATH
    _survey_mod.YOLO_FALLBACK_WEIGHTS_PATH = YOLO_FALLBACK_WEIGHTS_PATH
    _survey_mod.RAFT_ROOT = RAFT_ROOT
    _survey_mod.RAFT_CHECKPOINT_PATH = RAFT_CHECKPOINT_PATH
    _survey_mod.YOLO_IMGSZ = YOLO_IMGSZ
    _survey_mod.YOLO_CONF = YOLO_CONF
    _survey_mod.YOLO_IOU = YOLO_IOU
    _survey_mod.YOLO_RETINA_MASKS = YOLO_RETINA_MASKS
    _survey_mod.TARGET_CLASS_NAMES = TARGET_CLASS_NAMES
    _survey_mod.RAFT_VALID_ITERS = RAFT_VALID_ITERS
    _survey_mod.RAFT_DOWNSCALE = RAFT_DOWNSCALE
    _survey_mod.RAFT_MIXED_PRECISION = RAFT_MIXED_PRECISION
    _survey_mod.MIN_MASK_AREA_PX = MIN_MASK_AREA_PX

    _display_mod.BURST_COUNT = BURST_COUNT
    _display_mod.COMBINED_WIDTH_PX = COMBINED_WIDTH_PX
    _display_mod.OVERHEAD_DRAW_H_PX = OVERHEAD_DRAW_H_PX
    _display_mod.STEREO_DRAW_H_PX = STEREO_DRAW_H_PX
    _display_mod.STATUS_H_PX = STATUS_H_PX
    _display_mod.Z_MAX_MM = Z_MAX_MM
    _display_mod.GRIPPER_OFFSET_MM = GRIPPER_OFFSET_MM
    _display_mod.PLACE_RELEASE_GAP_MM = PLACE_RELEASE_GAP_MM

    _motion_mod.CONNECT_ROBOT = CONNECT_ROBOT
    _motion_mod.ENABLE_MOTORS_ON_START = ENABLE_MOTORS_ON_START
    _motion_mod.INIT_DRIVERS_ON_START = INIT_DRIVERS_ON_START
    _motion_mod.X_SURVEY = X_SURVEY
    _motion_mod.Y_SURVEY = Y_SURVEY
    _motion_mod.Z_SURVEY = Z_SURVEY
    _motion_mod.Z_MAX_MM = Z_MAX_MM
    _motion_mod.MIN_VALID_OBJECT_POINTS = MIN_VALID_OBJECT_POINTS
    _motion_mod.CLAW_OPEN_DEG = CLAW_OPEN_DEG


def _confirm(prompt: str, enabled: bool) -> bool:
    if not enabled:
        return True
    answer = input(f"{prompt}\nType YES to continue: ").strip()
    return answer == "YES"


def _check_robot_pose_safe(robot, x: float, y: float, z: float, label: str) -> bool:
    if hasattr(robot, "check_cartesian_pose_safe"):
        ok, reason = robot.check_cartesian_pose_safe(x, y, z)
        if not ok:
            print(f"{label} REFUSED: target pose unsafe: {reason}")
            return False
    return True


def _move_checked(robot, label: str, *, x_mm=None, y_mm=None, z_mm=None, phi_deg=None, move_time_s=None) -> bool:
    x_cur, y_cur, z_cur, phi_cur = robot.fk()
    x = x_cur if x_mm is None else float(x_mm)
    y = y_cur if y_mm is None else float(y_mm)
    z = z_cur if z_mm is None else float(z_mm)
    phi = phi_cur if phi_deg is None else float(phi_deg)
    print(f"{label} command: x={x:.1f} y={y:.1f} z={z:.1f} phi={phi:.1f}")
    if not _check_robot_pose_safe(robot, x, y, z, label):
        return False
    return move_cartesian_nonnegative_z(
        robot,
        label,
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        phi_deg=phi_deg,
        move_time_s=move_time_s,
    )


def _print_object_z_diagnostics(prefix: str, candidate) -> None:
    z_debug = getattr(candidate, "z_debug", None)
    if z_debug is None:
        print(f"{prefix} z diagnostics: unavailable")
        return
    print(
        f"{prefix} z percentiles: "
        f"p50={z_debug.z_p50_mm:.1f} p90={z_debug.z_p90_mm:.1f} "
        f"p95={z_debug.z_p95_mm:.1f} p99={z_debug.z_p99_mm:.1f}"
    )
    print(
        f"{prefix} robust_z: top={z_debug.robust_top_z_mm:.1f} "
        f"bottom={z_debug.robust_bottom_z_mm:.1f} height={z_debug.object_height_mm:.1f}"
    )
    print(
        f"{prefix} z clearance: top_spread={z_debug.top_spread_mm:.1f} "
        f"uncertainty={z_debug.uncertainty_clearance_mm:.1f} grasp={z_debug.grasp_z_mm:.1f}"
    )
    print(f"{prefix} z warnings: {z_debug.warnings}")


def execute_pick_selected(robot, dbg: CandidateDebug, bundle: dict | None = None) -> bool:
    if robot is None:
        print("[PICK] robot is not connected.")
        return False

    c = dbg.candidate
    if REQUIRE_OVERHEAD_XY_FOR_PICK and dbg.overhead_xy_mm is None:
        print("[PICK] refused: overhead XY unavailable and REQUIRE_OVERHEAD_XY_FOR_PICK=True")
        return False
    if REFUSE_PICK_IF_TOO_FEW_POINTS and c.valid_point_count < MIN_VALID_OBJECT_POINTS:
        print(f"[PICK] refused: points {c.valid_point_count} < {MIN_VALID_OBJECT_POINTS}")
        return False

    x = float(c.target_xy[0])
    y = float(c.target_xy[1])

    if USE_LOCAL_HEIGHT_AWARE_GRASP_XY:
        try:
            if bundle is None:
                raise RuntimeError("bundle unavailable")
            points_robot_xyz = _pointcloud_mod.cam_points_to_robot_xyz(dbg.points_cam, bundle)
            grasp_xy_plan = compute_grasp_xy_with_local_height(
                points_robot_xyz=points_robot_xyz,
                default_centroid_xy_mm=np.asarray(c.object_robot_xyz_raw[:2], dtype=np.float64),
                default_target_xy_mm=np.asarray(c.target_xy, dtype=np.float64),
                local_radius_mm=LOCAL_GRASP_RADIUS_MM,
                top_region_percentile=LOCAL_GRASP_TOP_REGION_PERCENTILE,
                height_delta_threshold_mm=LOCAL_GRASP_HEIGHT_DELTA_THRESHOLD_MM,
                top_region_blend_weight=LOCAL_GRASP_BLEND_WEIGHT,
                max_xy_shift_mm=LOCAL_GRASP_MAX_SHIFT_MM,
            )
            x = float(grasp_xy_plan.final_grasp_xy_mm[0])
            y = float(grasp_xy_plan.final_grasp_xy_mm[1])
            print(
                "[GRASP XY PLAN] "
                f"mode={grasp_xy_plan.mode} "
                f"centroid=({grasp_xy_plan.centroid_xy_mm[0]:.1f},{grasp_xy_plan.centroid_xy_mm[1]:.1f}) "
                f"top_region=({grasp_xy_plan.top_region_xy_mm[0]:.1f},{grasp_xy_plan.top_region_xy_mm[1]:.1f}) "
                f"final=({x:.1f},{y:.1f}) "
                f"local_top_z={grasp_xy_plan.local_top_z_mm:.1f} "
                f"global_top_z={grasp_xy_plan.global_top_z_mm:.1f} "
                f"height_delta={grasp_xy_plan.height_delta_mm:.1f} "
                f"warnings={grasp_xy_plan.warnings}"
            )
        except Exception as exc:
            print(f"[GRASP XY PLAN] fallback to default target XY due to error: {exc}")

    z_travel = float(Z_MAX_MM)
    z_grasp = float(c.grasp_robot_z)
    z_debug = getattr(c, "z_debug", None)
    if z_debug is not None:
        pick_plan = compute_pick_z_plan(
            z_result=z_debug,
            gripper_offset_mm=GRIPPER_OFFSET_MM,
            z_max_mm=Z_MAX_MM,
            pick_extra_clearance_mm=PICK_EXTRA_CLEARANCE_MM,
            pick_uncertainty_gain=PICK_Z_UNCERTAINTY_GAIN,
            pick_uncertainty_clearance_max_mm=PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
        )
        z_travel = float(pick_plan.travel_z_mm)
        z_grasp = float(pick_plan.final_grasp_z_mm)

    phi = _candidate_phi_or_current(robot, c)

    for label, z in (("travel", z_travel), ("grasp", z_grasp)):
        reason = validate_z_command(z, f"[PICK] {label}")
        if reason:
            print(f"[PICK] ABORT: {reason}")
            return False

    _hr("PICK SELECTED OBJECT", "-")
    print(f"[PICK] candidate [{c.index}] {c.yolo.class_name}")
    print(f"[PICK] XY={_fmt_xy(np.array([x, y], dtype=np.float64))} source={c.target_xy_source_effective}")
    print(f"[PICK] phi={phi:.2f} deg source={c.pick_phi_source}")
    print(f"[PICK] z plan: travel={z_travel:.1f}, grasp={z_grasp:.1f}")
    _print_object_z_diagnostics("[PICK]", c)

    if not _confirm("[PICK] Real pick motion will open claw, descend, close claw, and retract.", REQUIRE_CONFIRM_BEFORE_PICK):
        print("[PICK] canceled by user.")
        return False

    print(f"[PICK] opening claw servo={CLAW_OPEN_DEG}")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)

    if not _move_checked(robot, "[PICK] raise", z_mm=z_travel, move_time_s=COARSE_MOVE_TIME_S):
        return False
    if not _move_checked(robot, "[PICK] XY+phi", x_mm=x, y_mm=y, z_mm=z_travel, phi_deg=phi, move_time_s=XY_MOVE_TIME_S):
        return False
    if not _move_checked(robot, "[PICK] grasp", z_mm=z_grasp, move_time_s=PICK_Z_MOVE_TIME_S):
        return False

    print(f"[PICK] closing claw servo={CLAW_CLOSED_DEG}")
    robot.servo(CLAW_CLOSED_DEG)
    time.sleep(CLAW_SETTLE_S)

    if not _move_checked(robot, "[PICK] retract", z_mm=z_travel, move_time_s=COARSE_MOVE_TIME_S):
        return False

    print("[PICK] OK - item should be held.")
    return True


def _load_place_surface_zone() -> dict:
    try:
        zone = get_surface_zone(PLACE_SURFACE_ZONE_NAME, SURFACE_ZONE_CONFIG_PATH)
        out = {
            "name": zone["name"],
            "center_xy_mm": list(zone.get("center_xy_mm", [450.0, 250.0])),
            "surface_z_mm": float(zone["surface_z_mm"]),
            "default_phi_deg": float(zone.get("default_phi_deg", 0.0)),
            "width_mm": float(zone.get("width_mm", 120.0)),
            "depth_mm": float(zone.get("depth_mm", 120.0)),
            "notes": str(zone.get("notes", "")),
            "source": "surface_zones",
            "legacy_place_z_mm": None,
        }
    except Exception as exc:
        print(f"[SURFACE WARN] failed loading {PLACE_SURFACE_ZONE_NAME!r} from {SURFACE_ZONE_CONFIG_PATH}: {exc}")
        legacy = get_place_zone(PLACE_ZONE_NAME, PLACE_ZONE_CONFIG_PATH)
        legacy_place = legacy.get("place_z_mm", None)
        if legacy_place is not None:
            print("[SURFACE WARN] old place_z_mm detected; treating as destination surface_z_mm for backward compatibility.")
        out = {
            "name": legacy["name"],
            "center_xy_mm": list(legacy["center_xy_mm"]),
            "surface_z_mm": float(legacy.get("floor_z_mm", legacy_place or 0.0)),
            "default_phi_deg": float(legacy.get("phi_deg", 0.0)),
            "width_mm": float(legacy.get("width_mm", 120.0)),
            "depth_mm": float(legacy.get("depth_mm", 120.0)),
            "notes": str(legacy.get("notes", "")),
            "source": "place_zones_legacy",
            "legacy_place_z_mm": None if legacy_place is None else float(legacy_place),
        }

    _display_mod.PLACE_ZONE_FLOOR_Z_MM = float(out["surface_z_mm"])
    print("[ZONE] loaded destination surface zone:")
    print(f"  source      = {out['source']}")
    print(f"  name        = {out['name']}")
    print(f"  center_xy   = ({out['center_xy_mm'][0]:.1f}, {out['center_xy_mm'][1]:.1f}) mm")
    print(f"  surface_z   = {out['surface_z_mm']:.1f} mm")
    if out["legacy_place_z_mm"] is not None:
        print(f"  place_z_mm  = {out['legacy_place_z_mm']:.1f} mm (legacy) ")
    print(f"  phi         = {out['default_phi_deg']:.1f} deg")
    print(f"  size        = {out['width_mm']:.1f} x {out['depth_mm']:.1f} mm")
    print(f"  notes       = {out.get('notes', '')}")
    return out


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


def execute_place_zone(robot, held_object: CandidateDebug | None, *, allow_manual_override: bool = False) -> bool:
    if robot is None:
        print("[PLACE] robot is not connected.")
        return False
    if held_object is None and not allow_manual_override:
        print("[PLACE] refused: held_object is None. Use f again and confirm manual override if you are holding an item.")
        return False

    zone = _load_place_surface_zone()
    x = float(zone["center_xy_mm"][0])
    y = float(zone["center_xy_mm"][1])

    destination_surface_z_mm = float(zone["surface_z_mm"])
    object_height_mm, object_uncertainty_clearance_mm, object_warnings = _resolve_held_object_height_and_uncertainty(held_object)

    place_plan = compute_place_z_plan(
        destination_surface_z_mm=destination_surface_z_mm,
        object_height_mm=object_height_mm,
        z_max_mm=Z_MAX_MM,
        release_gap_mm=PLACE_RELEASE_GAP_MM,
        object_uncertainty_clearance_mm=object_uncertainty_clearance_mm,
        place_uncertainty_gain=PLACE_Z_UNCERTAINTY_GAIN,
        place_uncertainty_clearance_max_mm=PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
    )

    if USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT:
        place_z = float(place_plan.final_release_z_mm)
        place_source = "dynamic_surface_plus_object_height"
    else:
        if zone.get("legacy_place_z_mm") is not None:
            place_z = float(zone["legacy_place_z_mm"])
        else:
            place_z = float(destination_surface_z_mm)
        place_source = "fixed_place_z"
        print("[PLACE] dynamic place Z disabled; using fixed place Z from zone config.")

    place_phi = float(zone["default_phi_deg"])
    if USE_PICK_PHI_FOR_PLACE and held_object is not None:
        c = held_object.candidate
        if c.pick_phi_deg is not None and np.isfinite(float(c.pick_phi_deg)):
            place_phi = float(c.pick_phi_deg)

    travel_z = float(place_plan.approach_z_mm)
    for label, z in (("travel", travel_z), ("place", place_z), ("retract", float(place_plan.retract_z_mm))):
        reason = validate_z_command(z, f"[PLACE] {label}")
        if reason:
            print(f"[PLACE] ABORT: {reason}")
            return False

    if not _check_robot_pose_safe(robot, x, y, travel_z, "[PLACE] approach"):
        return False
    if not _check_robot_pose_safe(robot, x, y, place_z, "[PLACE] lower"):
        return False

    _hr("PLACE HELD OBJECT INTO SAVED ZONE", "-")
    print(f"[PLACE] zone={zone['name']}")
    print(f"[PLACE] target x={x:.1f} y={y:.1f} place_z={place_z:.1f} phi={place_phi:.1f}")
    print(f"[PLACE] z source={place_source}")
    print("[PLACE Z PLAN]")
    print(f"destination_surface_z_mm = {destination_surface_z_mm:.3f}")
    print(f"object_height_mm = {object_height_mm:.3f}")
    print(f"release_gap_mm = {PLACE_RELEASE_GAP_MM:.3f}")
    print(f"object_uncertainty_clearance_mm = {object_uncertainty_clearance_mm:.3f}")
    print(f"place_uncertainty_gain = {PLACE_Z_UNCERTAINTY_GAIN:.3f}")
    print(f"place_uncertainty_clearance_mm = {place_plan.place_uncertainty_clearance_mm:.3f}")
    print(f"final_release_z_mm = {place_plan.final_release_z_mm:.3f}")
    print(f"approach/retract_z_mm = {place_plan.approach_z_mm:.3f}/{place_plan.retract_z_mm:.3f}")
    print(f"warnings = {place_plan.warnings + object_warnings}")
    print("[PLACE] sequence: raise -> XY/phi at safe Z -> descend -> open claw -> dwell -> retract")

    if not _confirm("[PLACE] Real place motion will move to the saved zone and open the claw.", REQUIRE_CONFIRM_BEFORE_PLACE):
        print("[PLACE] canceled by user.")
        return False

    if not _move_checked(robot, "[PLACE] raise", z_mm=travel_z, move_time_s=COARSE_MOVE_TIME_S):
        return False
    if not _move_checked(robot, "[PLACE] XY+phi", x_mm=x, y_mm=y, z_mm=travel_z, phi_deg=place_phi, move_time_s=XY_MOVE_TIME_S):
        return False
    if not _move_checked(robot, "[PLACE] descend", z_mm=place_z, move_time_s=PLACE_Z_MOVE_TIME_S):
        return False

    print(f"[PLACE] opening claw servo={CLAW_OPEN_DEG}")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(float(PLACE_RELEASE_DWELL_S))

    if not _move_checked(robot, "[PLACE] retract", z_mm=float(place_plan.retract_z_mm), move_time_s=COARSE_MOVE_TIME_S):
        return False

    print("[PLACE] OK - item released and robot retracted.")
    return True


def main() -> int:
    _configure_modules()
    if PICK_PHI_MODE not in VALID_PICK_PHI_MODES:
        raise ValueError(f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}; expected one of {sorted(VALID_PICK_PHI_MODES)}")

    _hr("PICK ONE PLACE ONE REPEATABILITY", "=")
    print("[MAIN] This is a single-object placement repeatability test, not full bagging.")
    print(f"[MAIN] place surface zone: {PLACE_SURFACE_ZONE_NAME} from {SURFACE_ZONE_CONFIG_PATH}")
    print(f"[MAIN] Z_MAX={Z_MAX_MM:.1f} PLACE_APPROACH_Z={PLACE_APPROACH_Z_MM:.1f}")
    print(
        f"[MAIN] robust Z={USE_ROBUST_OBJECT_Z} dynamic place Z={USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT} "
        f"release_gap={PLACE_RELEASE_GAP_MM:.1f} place_uncertainty_gain={PLACE_Z_UNCERTAINTY_GAIN:.2f}"
    )
    print(
        f"[MAIN] pick_uncertainty_gain={PICK_Z_UNCERTAINTY_GAIN:.2f} "
        f"pick_uncertainty_cap={PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM:.1f} "
        f"place_uncertainty_cap={PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM:.1f}"
    )
    print(f"[MAIN] local_height_aware_grasp_xy={USE_LOCAL_HEIGHT_AWARE_GRASP_XY}")
    print(f"[MAIN] confirmations: pick={REQUIRE_CONFIRM_BEFORE_PICK} place={REQUIRE_CONFIRM_BEFORE_PLACE}")
    _load_place_surface_zone()
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
    held_object: CandidateDebug | None = None

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

            cv2.imshow(WINDOW, make_display(state, live_overhead, live_left, live_right))
            key = read_command_key(delay_ms=1)
            if key is None:
                continue

            if key in ("q", "escape", "\x1b"):
                print("[MAIN] quit requested")
                running = False

            elif key == "s":
                state = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
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
                    print(f"[ROTATE] selected [{state.selected_index + 1}/{len(state.candidates)}] {dbg.candidate.yolo.class_name}")

            elif key == "v":
                print_validation(state, 0 if state is None else state.selected_index)

            elif key == "k":
                if state is None or not state.candidates:
                    print("[PICK] no candidates. Press s first.")
                else:
                    dbg = state.candidates[state.selected_index]
                    print_validation(state, state.selected_index)
                    if execute_pick_selected(robot, dbg, bundle=bundle):
                        held_object = dbg

            elif key == "f":
                if held_object is None:
                    if not _confirm("[PLACE] No held object is tracked. Type YES only if you manually confirm an item is in the gripper.", True):
                        print("[PLACE] manual override canceled.")
                        continue
                    if execute_place_zone(robot, None, allow_manual_override=True):
                        held_object = None
                else:
                    if execute_place_zone(robot, held_object):
                        held_object = None

            elif key == "a":
                if state is None or not state.candidates:
                    print("[AUTO] no candidates. Press s first.")
                else:
                    dbg = state.candidates[state.selected_index]
                    print_validation(state, state.selected_index)
                    if execute_pick_selected(robot, dbg, bundle=bundle):
                        held_object = dbg
                        if execute_place_zone(robot, held_object):
                            held_object = None
            elif key == "n" and robot is not None:
                robot.send('HOMEJ3')

            elif key == "w":
                _load_place_surface_zone()
                print(f"[STATE] held_object={'yes' if held_object is not None else 'no'}")

            elif key == "o" and robot is not None:
                robot.servo(CLAW_OPEN_DEG)
                print(f"[CLAW] open ({CLAW_OPEN_DEG} deg)")

            elif key == "l" and robot is not None:
                robot.servo(CLAW_CLOSED_DEG)
                print(f"[CLAW] closed ({CLAW_CLOSED_DEG} deg)")

            elif key == "p" and robot is not None:
                _print_fk(robot, "[FK]")

            elif key == "[" and robot is not None:
                jog_nonnegative_z(robot, "[JOG]", dz=-5.0, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            elif key == "]" and robot is not None:
                jog_nonnegative_z(robot, "[JOG]", dz=+5.0, move_time_s=0.3)
                _print_fk(robot, "[JOG]")

            elif key == "g" and robot is not None:
                robot.move_cartesian(X_SURVEY, Y_SURVEY, Z_SURVEY, 0, move_time_s=3.0)
                print("[ROBOT] moved to survey pose")

            elif key == "e" and robot is not None:
                robot.enable(True)
                robot.init_drivers()
                print("[ROBOT] enabled")

            elif key == "d" and robot is not None:
                robot.enable(False)
                print("[ROBOT] disabled")

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
