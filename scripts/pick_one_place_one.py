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
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion.z_safety_config import (
    DEFAULT_Z_SAFETY,
    GRIPPER_OFFSET_MM as SHARED_GRIPPER_OFFSET_MM,
    PLACE_RELEASE_GAP_MM as SHARED_PLACE_RELEASE_GAP_MM,
    Z_MAX_MM as SHARED_Z_MAX_MM,
    print_z_safety_settings,
    validate_z_command,
)

BUNDLE_PATH = Path("robot_calibration_bundle.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")

YOLO_WEIGHTS_PATH = Path("full_data.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("yolo_weights/validate_V2.pt")

RAFT_ROOT = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")

SURFACE_ZONE_CONFIG_PATH = Path("config/surface_zones.json")
PLACE_SURFACE_ZONE_NAME = "New Bag Test"

PLACE_ZONE_CONFIG_PATH = Path("config/place_zones.json")
PLACE_ZONE_NAME = "New Bag Test"

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
    "overhead_semi_minor_projected",
    "pointcloud_shortest_path",
    "overhead_minor_axis",
    "mask_minor_axis_pointcloud",
    "triangulated_short_side",
    "current_fk",
}

Z_MAX_MM: float = SHARED_Z_MAX_MM
MIN_PICK_GRASP_Z_MM: float = DEFAULT_Z_SAFETY.MIN_PICK_GRASP_Z_MM
PLACE_APPROACH_Z_MM: float = Z_MAX_MM
GRIPPER_OFFSET_MM: float = SHARED_GRIPPER_OFFSET_MM
HOVER_HEIGHT_MM: float = Z_MAX_MM
GRASP_OFFSET_MM: float = GRIPPER_OFFSET_MM
USE_ROBUST_OBJECT_Z = True
ROBUST_TOP_PERCENTILE = 95.0
ROBUST_BOTTOM_PERCENTILE = 5.0
TOP_SPREAD_LOW_PERCENTILE = 90.0
TOP_SPREAD_HIGH_PERCENTILE = 99.0
Z_UNCERTAINTY_CLEARANCE_GAIN = 1.0
Z_UNCERTAINTY_CLEARANCE_MIN_MM = 5.0
Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
Z_UNCERTAINTY_WARN_MM = 10.0
PICK_EXTRA_CLEARANCE_MM = 0.0
PLACE_RELEASE_GAP_MM = SHARED_PLACE_RELEASE_GAP_MM
PLACE_Z_UNCERTAINTY_GAIN = 0.30
PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 10.0
PLACE_RELEASE_ANGLE_MODE = "empirical_plus_offset"
PLACE_RELEASE_EMPIRICAL_OFFSET_DEG = 5.0
PLACE_RELEASE_INITIAL_PADDING_DEG = 2.5
PICK_Z_UNCERTAINTY_GAIN = .25
PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
MIN_OBJECT_HEIGHT_MM = 2.0
MAX_OBJECT_HEIGHT_MM = 180.0

OVERHEAD_MATCH_MAX_DIST_MM: float = 140.0
OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True
OVERHEAD_XY_BLEND_WEIGHT: float = 0.45
OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI: bool = True

USE_CONFIDENCE_PHI_BLEND: bool = False
PHI_DISAGREEMENT_WARN_DEG: float = 25.0
PHI_MIN_CONFIDENCE: float = 0.20
PHI_ASPECT_DECAY: float = 0.8
PHI_STEREO_HEIGHT_DECAY_CM: float = 8.0
PHI_FALLBACK_TO_CURRENT_EE_PHI: bool = False

COARSE_MOVE_TIME_S: float = 1.10
XY_MOVE_TIME_S: float = 1.50
PICK_Z_MOVE_TIME_S: float = 0.60
PLACE_Z_MOVE_TIME_S: float = 0.60

CONNECT_ROBOT: bool = True
ENABLE_MOTORS_ON_START: bool = True
INIT_DRIVERS_ON_START: bool = True

COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 560
STEREO_DRAW_H_PX: int = 390
STATUS_H_PX: int = 140
WINDOW: str = "Pick One Place One Repeatability"

CLAW_OPEN_DEG: int = 60
CLAW_CLOSED_DEG: int = 0
ENABLE_DYNAMIC_PICK = True
USE_DYNAMIC_PICK_HEIGHT = False
USE_DYNAMIC_PICK_GRIP_ANGLE = True
DYNAMIC_PICK_FALLBACK_TO_FIXED = False
DYNAMIC_LOWER_CLEARANCE_MM = 20.0
DYNAMIC_SERVO_MARGIN_DEG = 10.0
GRIPPER_GEOMETRY_L_MM = 70.0
GRIPPER_SERVO_MIN_DEG = 0.0
GRIPPER_SERVO_MAX_DEG = 70.0
DYNAMIC_PICK_DEFAULT_SERVO_DEG = 55.0
DYNAMIC_LOWER_DERIV_THRESH_MA = 30.0
DYNAMIC_GRIP_DERIV_THRESH_MA = 500.0
DYNAMIC_CONTACT_LOOKBACK_COUNT = 3
DYNAMIC_CONTACT_NONZERO_EPS_MA = 1.0
DYNAMIC_CONTACT_SUM_GRIP_MA = 1500.0
DYNAMIC_CONTACT_SUM_LOWER_MA = 50.0
DEFAULT_OBJECT_RIGIDITY = "squishable"
SQUISHABLE_POST_CONTACT_EXTRA_CLOSE_DEG = 5.0
DYNAMIC_PICK_TRACE_DEBUG = False
DYNAMIC_PICK_TRACE_QUERY_POS = False

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

# Optional platform-height compensation for pick grasp Z.
# When enabled, pick surface Z is estimated as:
#   z_surface = z_ground_model(x_pick, y_pick) + estimated_object_height
# where z_ground_model is generated by calibrate_platform_z_from_apriltag.py.
USE_Z_GROUND_MODEL_FOR_PICK_SURFACE = True
Z_GROUND_MODEL_PATH = Path("data/z_ground_calibration/z_ground_model_latest.json")

REQUIRE_OVERHEAD_XY_FOR_PICK: bool = False
REFUSE_PICK_IF_TOO_FEW_POINTS: bool = True

X_SURVEY = 500.0
Y_SURVEY = -50.0
Z_SURVEY = 270.0

# ============================================================

import json
import math
import time
import traceback

import cv2
import numpy as np

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
from motion.pick_place_sequence import (
    PickSequenceSettings,
    PlaceSequenceSettings,
    execute_pick_sequence,
    execute_place_sequence,
)
from motion.dynamic_grasp_policy import build_dynamic_pick_plan
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
from vision.grasp_xy_policy import compute_grasp_xy_with_local_height
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


# Internal dynamic command defaults. The user-facing contact trigger knobs live
# in the USER SETTINGS block above; keep these stable unless we're deliberately
# changing the firmware command semantics.
_PUSH_DYNAMIC_FIRMWARE_SETTINGS_ON_START = True
_DYNAMIC_LOWER_N_STEPS = 1
_DYNAMIC_LOWER_SIGNED_ONLY = False
_DYNAMIC_GRIP_N_STEPS = 3
_DYNAMIC_GRIP_SIGNED_ONLY = False

_Z_GROUND_MODEL_CACHE: dict | None = None
_Z_GROUND_MODEL_LOAD_ATTEMPTED = False


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


def _load_z_ground_model_cached(*, ignore_toggle: bool = False) -> dict | None:
    global _Z_GROUND_MODEL_CACHE, _Z_GROUND_MODEL_LOAD_ATTEMPTED

    if _Z_GROUND_MODEL_LOAD_ATTEMPTED:
        return _Z_GROUND_MODEL_CACHE

    _Z_GROUND_MODEL_LOAD_ATTEMPTED = True
    if (not ignore_toggle) and (not USE_Z_GROUND_MODEL_FOR_PICK_SURFACE):
        return None

    if not Z_GROUND_MODEL_PATH.exists():
        print(f"[PICK ZGROUND] model not found: {Z_GROUND_MODEL_PATH.resolve()}")
        print("[PICK ZGROUND] fallback: using stereo-only pick surface Z policy")
        return None

    try:
        with Z_GROUND_MODEL_PATH.open("r", encoding="utf-8") as f:
            model = json.load(f)
        _Z_GROUND_MODEL_CACHE = model
        print(
            "[PICK ZGROUND] loaded "
            f"{Z_GROUND_MODEL_PATH.resolve()} "
            f"type={model.get('model_type')} "
            f"n={model.get('num_samples')} "
            f"rmse={float(model.get('rmse_mm', float('nan'))):.2f} mm"
        )
        return _Z_GROUND_MODEL_CACHE
    except Exception as exc:
        print(f"[PICK ZGROUND] failed loading model: {exc}")
        print("[PICK ZGROUND] fallback: using stereo-only pick surface Z policy")
        _Z_GROUND_MODEL_CACHE = None
        return None


def _predict_z_ground_mm(model: dict | None, x_robot_mm: float, y_robot_mm: float) -> float | None:
    if model is None:
        return None

    try:
        dx = float(x_robot_mm) - float(model["x_center_mm"])
        dy = float(y_robot_mm) - float(model["y_center_mm"])

        term_values = {
            "1": 1.0,
            "dx": dx,
            "dy": dy,
            "dx*dy": dx * dy,
            "dx^2": dx * dx,
            "dy^2": dy * dy,
        }

        z = 0.0
        for coeff, term in zip(model["coefficients_mm"], model["terms"]):
            if term not in term_values:
                raise ValueError(f"Unknown z_ground model term: {term!r}")
            z += float(coeff) * term_values[term]
        return float(z)
    except Exception as exc:
        print(f"[PICK ZGROUND] predict failed: {exc}")
        return None


def _candidate_object_height_mm_for_zground(candidate) -> float | None:
    z_debug = getattr(candidate, "z_debug", None)
    if z_debug is not None:
        try:
            h = float(getattr(z_debug, "object_height_mm", None))
            if np.isfinite(h):
                return max(0.0, h)
        except (TypeError, ValueError):
            pass

    for attr, scale in (
        ("pointcloud_height_mm", 1.0),
        ("height_mm_from_pointcloud", 1.0),
        ("pointcloud_height_cm", 10.0),
        ("height_cm_from_pointcloud", 10.0),
    ):
        try:
            v = float(getattr(candidate, attr, None))
            if np.isfinite(v):
                return max(0.0, v * scale)
        except (TypeError, ValueError):
            continue

    return None


def set_use_z_ground_model_for_pick_surface(enabled: bool) -> bool:
    """Runtime toggle for z_ground pick-surface compensation."""
    global USE_Z_GROUND_MODEL_FOR_PICK_SURFACE, _Z_GROUND_MODEL_CACHE, _Z_GROUND_MODEL_LOAD_ATTEMPTED
    USE_Z_GROUND_MODEL_FOR_PICK_SURFACE = bool(enabled)

    # Reset cache state so a later enable re-reads disk and a disable keeps behavior explicit.
    _Z_GROUND_MODEL_CACHE = None
    _Z_GROUND_MODEL_LOAD_ATTEMPTED = False
    print(f"[PICK ZGROUND] runtime toggle -> {'ON' if USE_Z_GROUND_MODEL_FOR_PICK_SURFACE else 'OFF'}")
    return USE_Z_GROUND_MODEL_FOR_PICK_SURFACE


def get_use_z_ground_model_for_pick_surface() -> bool:
    return bool(USE_Z_GROUND_MODEL_FOR_PICK_SURFACE)


def preview_pick_grasp_z_with_and_without_zground(
    dbg: CandidateDebug,
    *,
    pick_xy_mm: np.ndarray | None = None,
) -> dict[str, float | str | None]:
    """Compute pick grasp Z both with and without z_ground compensation for comparison."""
    c = dbg.candidate
    if pick_xy_mm is None:
        x = float(c.target_xy[0])
        y = float(c.target_xy[1])
    else:
        xy = np.asarray(pick_xy_mm, dtype=np.float64).reshape(2)
        x = float(xy[0])
        y = float(xy[1])

    z_debug = getattr(c, "z_debug", None)
    base_plan = compute_pick_z_plan(
        z_result=z_debug,
        object_surface_z_mm=None,
        candidate=c,
        gripper_offset_mm=GRIPPER_OFFSET_MM,
        z_max_mm=Z_MAX_MM,
        pick_extra_clearance_mm=PICK_EXTRA_CLEARANCE_MM,
        pick_uncertainty_gain=PICK_Z_UNCERTAINTY_GAIN,
        pick_uncertainty_clearance_max_mm=PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
        config=DEFAULT_Z_SAFETY,
    )

    model = _load_z_ground_model_cached(ignore_toggle=True)
    z_ground_mm = _predict_z_ground_mm(model, x, y)
    object_height_mm = _candidate_object_height_mm_for_zground(c)
    surface_override_mm = None
    if z_ground_mm is not None and object_height_mm is not None:
        surface_override_mm = float(z_ground_mm + object_height_mm)

    zground_plan = compute_pick_z_plan(
        z_result=z_debug,
        object_surface_z_mm=surface_override_mm,
        candidate=c,
        gripper_offset_mm=GRIPPER_OFFSET_MM,
        z_max_mm=Z_MAX_MM,
        pick_extra_clearance_mm=PICK_EXTRA_CLEARANCE_MM,
        pick_uncertainty_gain=PICK_Z_UNCERTAINTY_GAIN,
        pick_uncertainty_clearance_max_mm=PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
        config=DEFAULT_Z_SAFETY,
    )

    return {
        "x_mm": x,
        "y_mm": y,
        "base_grasp_z_mm": float(base_plan.final_grasp_z_mm),
        "zground_grasp_z_mm": float(zground_plan.final_grasp_z_mm),
        "delta_grasp_z_mm": float(zground_plan.final_grasp_z_mm - base_plan.final_grasp_z_mm),
        "z_ground_mm": None if z_ground_mm is None else float(z_ground_mm),
        "object_height_mm": None if object_height_mm is None else float(object_height_mm),
        "surface_override_mm": None if surface_override_mm is None else float(surface_override_mm),
        "zground_model_available": bool(model is not None),
        "active_mode": "zground_on" if USE_Z_GROUND_MODEL_FOR_PICK_SURFACE else "zground_off",
    }


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


def _candidate_rigidity_type(candidate) -> str:
    rigidity = str(DEFAULT_OBJECT_RIGIDITY).strip().lower()
    return rigidity if rigidity else "rigid"


def _apply_dynamic_object_grip_settings(robot, candidate) -> None:
    if robot is None:
        return
    rigidity = _candidate_rigidity_type(candidate)
    is_squishable = rigidity == "squishable"
    extra_close_deg = float(SQUISHABLE_POST_CONTACT_EXTRA_CLOSE_DEG) if is_squishable else 0.0
    class_name = getattr(getattr(candidate, "yolo", None), "class_name", "unknown")

    settings = [
        ("grip_object_squishable", bool(is_squishable)),
        ("grip_post_contact_extra_close_deg", float(extra_close_deg)),
    ]
    print(
        "[DYNSET] object grip profile: "
        f"class={class_name} rigidity={rigidity} extra_close_deg={extra_close_deg:.1f}"
    )
    for key, value in settings:
        ok = False
        if hasattr(robot, "dynset"):
            ok = bool(robot.dynset(key, value))
        else:
            value_str = "1" if isinstance(value, bool) and value else "0" if isinstance(value, bool) else f"{float(value):.3f}"
            robot.send(f"dynset {key} {value_str}")
            ok = bool(robot.read_until("dynset OK", 3.0))
            time.sleep(0.05)
            robot.flush()
        if ok:
            if isinstance(value, bool):
                print(f"[DYNSET] {key} = {1 if value else 0}")
            else:
                print(f"[DYNSET] {key} = {float(value):.3f}")
        else:
            if isinstance(value, bool):
                print(f"[DYNSET] WARN failed to apply {key} = {1 if value else 0}")
            else:
                print(f"[DYNSET] WARN failed to apply {key} = {float(value):.3f}")


def _apply_dynamic_firmware_settings(robot) -> None:
    if robot is None:
        return
    if not _PUSH_DYNAMIC_FIRMWARE_SETTINGS_ON_START:
        print("[DYNSET] startup push disabled.")
        return

    settings = [
        ("deriv_nonzero_eps_ma", float(DYNAMIC_CONTACT_NONZERO_EPS_MA)),
        ("pattern_nonzero_n", int(DYNAMIC_CONTACT_LOOKBACK_COUNT)),
        ("pattern_sum_grip_ma", float(DYNAMIC_CONTACT_SUM_GRIP_MA)),
        ("pattern_sum_lower_ma", float(DYNAMIC_CONTACT_SUM_LOWER_MA)),
    ]
    print("[DYNSET] applying dynamic contact-trigger settings...")
    for key, value in settings:
        ok = False
        if hasattr(robot, "dynset"):
            ok = bool(robot.dynset(key, value))
        else:
            value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
            robot.send(f"dynset {key} {value_str}")
            ok = bool(robot.read_until("dynset OK", 3.0))
            time.sleep(0.05)
            robot.flush()
        if ok:
            if isinstance(value, float):
                print(f"[DYNSET] {key} = {value:.3f}")
            else:
                print(f"[DYNSET] {key} = {value}")
        else:
            if isinstance(value, float):
                print(f"[DYNSET] WARN failed to apply {key} = {value:.3f}")
            else:
                print(f"[DYNSET] WARN failed to apply {key} = {value}")


def _print_dynamic_pick_plan(plan) -> None:
    print("[DYNAMIC PICK PLAN]")
    print(f"class = {plan.object_class}")
    print(f"target_xy_mm = ({plan.target_xy_mm[0]:.1f}, {plan.target_xy_mm[1]:.1f})")
    print(f"phi_deg = {plan.phi_deg:.2f}")
    print(f"phi_source = {plan.phi_source}")
    print(f"measured_grip_width_mm = {plan.measured_grip_width_mm}")
    print(f"grip_width_source = {plan.grip_width_source}")
    print(f"use_dynamic_pick_height = {USE_DYNAMIC_PICK_HEIGHT}")
    print(f"use_dynamic_pick_grip_angle = {USE_DYNAMIC_PICK_GRIP_ANGLE}")
    print(f"initial_servo_angle_deg = {plan.initial_servo_angle_deg:.2f}")
    print(f"servo_angle_margin_deg = {plan.servo_angle_margin_deg:.2f}")
    print(f"z_pregrasp_robot_z_mm = {plan.dynamic_pregrasp_robot_z_mm:.2f}")
    print(f"z_probe_start_mm = {plan.dynamic_lower_start_z_mm:.2f}")
    print(f"dynamic_grip_start_angle_deg = {plan.dynamic_grip_start_angle_deg:.2f}")
    print(f"lower_deriv_thresh_ma = {float(DYNAMIC_LOWER_DERIV_THRESH_MA):.1f}")
    print(f"grip_deriv_thresh_ma = {float(DYNAMIC_GRIP_DERIV_THRESH_MA):.1f}")
    print(f"contact_lookback_count = {int(DYNAMIC_CONTACT_LOOKBACK_COUNT)}")
    print(f"contact_nonzero_eps_ma = {float(DYNAMIC_CONTACT_NONZERO_EPS_MA):.1f}")
    print(f"contact_sum_grip_ma = {float(DYNAMIC_CONTACT_SUM_GRIP_MA):.1f}")
    print(f"contact_sum_lower_ma = {float(DYNAMIC_CONTACT_SUM_LOWER_MA):.1f}")
    print(f"warnings = {plan.warnings}")


def _print_dynamic_pick_trace(robot, label: str, *, target_z_mm: float | None = None, note: str | None = None) -> None:
    if not DYNAMIC_PICK_TRACE_DEBUG:
        return
    print(f"[DYNAMIC TRACE] {label}")
    x_fk, y_fk, z_fk, phi_fk = robot.fk()
    print(f"  fk_xyzphi = ({x_fk:.2f}, {y_fk:.2f}, {z_fk:.2f}, {phi_fk:.2f})")
    print(f"  q_est.z_mm = {float(robot.q_est.z_mm):.2f}")
    est_steps = robot.joints_to_steps(robot.q_est)
    print(f"  est_steps = {est_steps}")
    if target_z_mm is not None and hasattr(robot, "z_command_trace"):
        trace = robot.z_command_trace(float(target_z_mm), include_actual_pos=DYNAMIC_PICK_TRACE_QUERY_POS)
        print(f"  target_robot_z_mm = {trace.target_robot_z_mm:.2f}")
        print(f"  current_fk_z_mm = {trace.current_fk_z_mm:.2f}")
        print(f"  current_q_est_z_mm = {trace.current_q_est_z_mm:.2f}")
        print(f"  current_est_j3_steps = {trace.current_est_j3_steps}")
        print(f"  current_est_j3_mm = {trace.current_est_j3_mm:.2f}")
        print(f"  current_est_direct_j3_mm = {trace.current_est_direct_j3_mm:.2f}")
        print(f"  target_est_j3_steps = {trace.target_est_j3_steps}")
        print(f"  target_est_j3_mm = {trace.target_est_j3_mm:.2f}")
        print(f"  target_direct_j3_mm = {trace.target_direct_j3_mm:.2f}")
        if trace.actual_steps is not None:
            print(f"  actual_steps = {trace.actual_steps}")
        if trace.actual_j3_mm is not None:
            print(f"  actual_j3_mm = {trace.actual_j3_mm:.2f}")
        if trace.actual_direct_j3_mm is not None:
            print(f"  actual_direct_j3_mm = {trace.actual_direct_j3_mm:.2f}")
        if trace.inferred_robot_to_direct_offset_mm is not None:
            print(f"  inferred_robot_to_direct_offset_mm = {trace.inferred_robot_to_direct_offset_mm:.2f}")
        if trace.warnings:
            print(f"  warnings = {trace.warnings}")
    if note:
        print(f"  note = {note}")


def _store_dynamic_pick_results(
    candidate,
    plan,
    lower_result=None,
    grip_result=None,
    *,
    z_empirical_robot_frame_mm: float | None = None,
    actual_initial_servo_angle_deg: float | None = None,
) -> None:
    candidate.measured_grip_width_mm = plan.measured_grip_width_mm
    candidate.grip_width_source = plan.grip_width_source
    candidate.initial_servo_angle_deg = (
        plan.initial_servo_angle_deg
        if actual_initial_servo_angle_deg is None
        else float(actual_initial_servo_angle_deg)
    )
    candidate.dynamic_lower_start_z_mm = plan.dynamic_lower_start_z_mm
    candidate.dynamic_pregrasp_robot_z_mm = plan.dynamic_pregrasp_robot_z_mm
    candidate.dynamic_grip_start_angle_deg = plan.dynamic_grip_start_angle_deg
    candidate.servo_empirical_deg = None if grip_result is None else grip_result.servo_empirical_deg
    candidate.z_empirical_mm = None if lower_result is None else lower_result.z_empirical_mm
    candidate.z_empirical_robot_frame_mm = z_empirical_robot_frame_mm
    candidate.dynamic_pick_warnings = list(plan.warnings)


def _resolve_release_servo_angle_deg(held_object: CandidateDebug | None, *, fallback_deg: float) -> tuple[float, str]:
    if held_object is not None:
        candidate = held_object.candidate
        mode = str(PLACE_RELEASE_ANGLE_MODE).strip().lower()

        empirical_angle = getattr(candidate, "servo_empirical_deg", None)
        try:
            empirical_f = float(empirical_angle)
        except (TypeError, ValueError):
            empirical_f = None

        initial_angle = getattr(candidate, "initial_servo_angle_deg", None)
        try:
            initial_f = float(initial_angle)
        except (TypeError, ValueError):
            initial_f = None

        if mode == "initial_minus_padding":
            if initial_f is not None and np.isfinite(initial_f):
                release_angle = float(np.clip(initial_f - PLACE_RELEASE_INITIAL_PADDING_DEG, 0.0, 70.0))
                return release_angle, "held_object.initial_servo_angle_deg-padding"
            if empirical_f is not None and np.isfinite(empirical_f):
                release_angle = float(np.clip(empirical_f + PLACE_RELEASE_EMPIRICAL_OFFSET_DEG, 0.0, 70.0))
                return release_angle, "held_object.servo_empirical_deg+offset(fallback)"

        if empirical_f is not None and np.isfinite(empirical_f):
            release_angle = float(np.clip(empirical_f + PLACE_RELEASE_EMPIRICAL_OFFSET_DEG, 0.0, 70.0))
            return release_angle, "held_object.servo_empirical_deg+offset"
        if initial_f is not None and np.isfinite(initial_f):
            release_angle = float(np.clip(initial_f - PLACE_RELEASE_INITIAL_PADDING_DEG, 0.0, 70.0))
            return release_angle, "held_object.initial_servo_angle_deg-padding(fallback)"

    return float(fallback_deg), "fallback_default"


def _command_servo_angle(robot, angle_deg: float) -> bool:
    if hasattr(robot, "set_servo_fractional"):
        return bool(robot.set_servo_fractional(float(angle_deg)))
    return bool(robot.servo(int(round(float(angle_deg)))))


def _execute_fixed_pick_path(robot, x: float, y: float, phi: float, pick_plan) -> bool:
    return execute_pick_sequence(
        robot,
        None,
        target_xy_mm=np.array([x, y], dtype=np.float64),
        target_phi_deg=phi,
        pick_z_plan=pick_plan,
        settings=PickSequenceSettings(
            claw_open_deg=CLAW_OPEN_DEG,
            claw_closed_deg=CLAW_CLOSED_DEG,
            coarse_move_time_s=COARSE_MOVE_TIME_S,
            xy_move_time_s=XY_MOVE_TIME_S,
            z_move_time_s=PICK_Z_MOVE_TIME_S,
        ),
        config=DEFAULT_Z_SAFETY,
        check_pose_safe_fn=_check_robot_pose_safe,
        label_prefix="[PICK]",
    )


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

    z_debug = getattr(c, "z_debug", None)
    pick_surface_override_mm: float | None = None
    pick_surface_source = "stereo_robust_top"

    if USE_Z_GROUND_MODEL_FOR_PICK_SURFACE:
        model = _load_z_ground_model_cached()
        z_ground_mm = _predict_z_ground_mm(model, x, y)
        object_height_mm = _candidate_object_height_mm_for_zground(c)
        if z_ground_mm is not None and object_height_mm is not None:
            pick_surface_override_mm = float(z_ground_mm + object_height_mm)
            pick_surface_source = "z_ground_model_plus_object_height"
            print(
                "[PICK ZGROUND] "
                f"xy=({x:.1f},{y:.1f}) "
                f"z_ground={z_ground_mm:.2f} "
                f"obj_h={object_height_mm:.2f} "
                f"surface={pick_surface_override_mm:.2f}"
            )
        else:
            if z_ground_mm is None:
                print("[PICK ZGROUND] no z_ground prediction at this XY; fallback to stereo surface")
            if object_height_mm is None:
                print("[PICK ZGROUND] object height unavailable; fallback to stereo surface")

    pick_plan = compute_pick_z_plan(
        z_result=z_debug,
        object_surface_z_mm=pick_surface_override_mm,
        candidate=c,
        gripper_offset_mm=GRIPPER_OFFSET_MM,
        z_max_mm=Z_MAX_MM,
        pick_extra_clearance_mm=PICK_EXTRA_CLEARANCE_MM,
        pick_uncertainty_gain=PICK_Z_UNCERTAINTY_GAIN,
        pick_uncertainty_clearance_max_mm=PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM,
        config=DEFAULT_Z_SAFETY,
    )
    z_travel = float(pick_plan.approach_z_mm)
    z_grasp = float(pick_plan.final_grasp_z_mm)

    phi = _candidate_phi_or_current(robot, c)

    for label, z in (("approach", z_travel), ("grasp", z_grasp), ("retract", pick_plan.retract_z_mm)):
        reason = validate_z_command(z, f"[PICK] {label}", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[PICK] ABORT: {reason}")
            return False

    _hr("PICK SELECTED OBJECT", "-")
    print(f"[PICK] candidate [{c.index}] {c.yolo.class_name}")
    print(f"[PICK] XY={_fmt_xy(np.array([x, y], dtype=np.float64))} source={c.target_xy_source_effective}")
    print(f"[PICK] phi={phi:.2f} deg source={c.pick_phi_source}")
    print(
        f"[PICK] z plan: approach={z_travel:.1f}, grasp={z_grasp:.1f}, "
        f"retract={pick_plan.retract_z_mm:.1f}"
    )
    print(f"[PICK] surface source={pick_surface_source}")
    _print_object_z_diagnostics("[PICK]", c)

    if not _confirm("[PICK] Real pick motion will execute the configured pick sequence and retract.", REQUIRE_CONFIRM_BEFORE_PICK):
        print("[PICK] canceled by user.")
        return False

    if not ENABLE_DYNAMIC_PICK:
        return _execute_fixed_pick_path(robot, x, y, phi, pick_plan)

    if not hasattr(robot, "dynamic_lower_robot_z") or not hasattr(robot, "dynamic_grip") or not hasattr(robot, "set_servo_fractional"):
        print("[PICK] dynamic pick helpers unavailable on Robot.")
        if DYNAMIC_PICK_FALLBACK_TO_FIXED:
            print("[PICK] falling back to fixed open-descend-close path.")
            return _execute_fixed_pick_path(robot, x, y, phi, pick_plan)
        return False

    try:
        dynamic_plan = build_dynamic_pick_plan(
            c,
            dbg,
            robot,
            bundle,
            z_grasp_mm=z_grasp,
            gripper_offset_mm=GRIPPER_OFFSET_MM,
            z_max_mm=Z_MAX_MM,
            dynamic_lower_clearance_mm=DYNAMIC_LOWER_CLEARANCE_MM,
            servo_angle_margin_deg=DYNAMIC_SERVO_MARGIN_DEG,
            gripper_geometry_l_mm=GRIPPER_GEOMETRY_L_MM,
            gripper_servo_min_deg=GRIPPER_SERVO_MIN_DEG,
            gripper_servo_max_deg=GRIPPER_SERVO_MAX_DEG,
            fallback_initial_servo_angle_deg=DYNAMIC_PICK_DEFAULT_SERVO_DEG,
        )
    except Exception as exc:
        print(f"[PICK] dynamic plan build failed: {exc}")
        if DYNAMIC_PICK_FALLBACK_TO_FIXED:
            print("[PICK] falling back to fixed open-descend-close path.")
            return _execute_fixed_pick_path(robot, x, y, phi, pick_plan)
        return False

    probe_reason = validate_z_command(dynamic_plan.dynamic_lower_start_z_mm, "[PICK] probe_start", config=DEFAULT_Z_SAFETY)
    if probe_reason:
        print(f"[PICK] ABORT: {probe_reason}")
        return False

    descent_target_z_mm = float(dynamic_plan.dynamic_pregrasp_robot_z_mm if USE_DYNAMIC_PICK_HEIGHT else z_grasp)
    descent_reason = validate_z_command(descent_target_z_mm, "[PICK] descent_target", config=DEFAULT_Z_SAFETY)
    if descent_reason:
        print(f"[PICK] ABORT: {descent_reason}")
        return False

    preset_servo_angle_deg = (
        float(dynamic_plan.initial_servo_angle_deg)
        if USE_DYNAMIC_PICK_GRIP_ANGLE
        else float(CLAW_OPEN_DEG)
    )

    _apply_dynamic_object_grip_settings(robot, c)
    _print_dynamic_pick_plan(dynamic_plan)
    _print_dynamic_pick_trace(
        robot,
        "plan built",
        target_z_mm=descent_target_z_mm,
        note=(
            "Normal robot moves convert robot z -> target J3 steps via joints_to_steps. "
            "DL sends the direct J3 mm value shown as direct_dynamiclower_j3_mm."
        ),
    )

    print(f"[PICK] servo preset angle={preset_servo_angle_deg:.2f} source={'geometry' if USE_DYNAMIC_PICK_GRIP_ANGLE else 'fixed_open'}")
    if not robot.set_servo_fractional(preset_servo_angle_deg):
        print("[PICK] failed to set initial dynamic servo angle.")
        return False
    _print_dynamic_pick_trace(
        robot,
        "after servo preset",
        target_z_mm=descent_target_z_mm,
        note="Servo preset only; no J3 motion yet.",
    )

    _print_dynamic_pick_trace(robot, "before raise", target_z_mm=z_travel)
    if not _move_checked(robot, "[PICK] raise", z_mm=z_travel, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_dynamic_pick_trace(robot, "after raise", target_z_mm=z_travel)

    _print_dynamic_pick_trace(robot, "before XY+phi", target_z_mm=z_travel)
    if not _move_checked(robot, "[PICK] XY+phi", x_mm=x, y_mm=y, z_mm=z_travel, phi_deg=phi, move_time_s=XY_MOVE_TIME_S):
        return False
    _print_dynamic_pick_trace(robot, "after XY+phi", target_z_mm=descent_target_z_mm)

    if descent_target_z_mm < z_travel - 1e-6:
        _print_dynamic_pick_trace(
            robot,
            "before descent-target move",
            target_z_mm=descent_target_z_mm,
            note=(
                "This is still a normal Cartesian move. "
                "When USE_DYNAMIC_PICK_HEIGHT=False, this is the feedforward grasp Z descent."
            ),
        )
        if not _move_checked(
            robot,
            "[PICK] descent target",
            z_mm=descent_target_z_mm,
            move_time_s=PICK_Z_MOVE_TIME_S,
        ):
            return False
        _print_dynamic_pick_trace(
            robot,
            "after descent-target move",
            target_z_mm=descent_target_z_mm,
            note=(
                "If the slam already happened, it was before DL. "
                "With dynamic height off, the next stage is DG from this feedforward Z."
            ),
        )

    z_empirical_robot_frame_mm = None
    lower_result = None
    if USE_DYNAMIC_PICK_HEIGHT:
        _print_dynamic_pick_trace(
            robot,
            "before DL command",
            target_z_mm=dynamic_plan.dynamic_lower_start_z_mm,
            note=(
                f"About to send: DLR {dynamic_plan.dynamic_lower_start_z_mm:.3f} "
                f"{float(DYNAMIC_LOWER_DERIV_THRESH_MA):.3f} {int(_DYNAMIC_LOWER_N_STEPS)} "
                f"{preset_servo_angle_deg:.3f} {1 if _DYNAMIC_LOWER_SIGNED_ONLY else 0}"
            ),
        )
        lower_result = robot.dynamic_lower_robot_z(
            robot_z_mm=dynamic_plan.dynamic_lower_start_z_mm,
            deriv_thresh_ma=DYNAMIC_LOWER_DERIV_THRESH_MA,
            n_steps=_DYNAMIC_LOWER_N_STEPS,
            servo_deg=preset_servo_angle_deg,
            signed_only=_DYNAMIC_LOWER_SIGNED_ONLY,
        )
        print(
            "[PICK] dynamic lower result: "
            f"ok={lower_result.ok} z_empirical_mm={lower_result.z_empirical_mm} "
            f"servo_empirical_deg={lower_result.servo_empirical_deg} error={lower_result.error}"
        )
        if not lower_result.ok:
            print("[PICK] ABORT: dynamic lower failed.")
            return False

        if lower_result.z_empirical_mm is not None and np.isfinite(float(lower_result.z_empirical_mm)):
            z_empirical_robot_frame_mm = float(robot.teensy_direct_j3_mm_to_robot_z(float(lower_result.z_empirical_mm)))
            empirical_reason = validate_z_command(
                z_empirical_robot_frame_mm,
                "[PICK] z_empirical_robot_frame",
                config=DEFAULT_Z_SAFETY,
            )
            if empirical_reason:
                print(f"[PICK] ABORT: {empirical_reason}")
                return False
            print(
                "[PICK] empirical Z conversion: "
                f"direct_j3_mm={float(lower_result.z_empirical_mm):.2f} "
                f"-> robot_z_mm={z_empirical_robot_frame_mm:.2f}"
            )
            _print_dynamic_pick_trace(
                robot,
                "before empirical-z move",
                target_z_mm=z_empirical_robot_frame_mm,
                note="Move to compensated empirical robot-frame Z before dynamic grip.",
            )
            if not _move_checked(
                robot,
                "[PICK] empirical z",
                z_mm=z_empirical_robot_frame_mm,
                move_time_s=PICK_Z_MOVE_TIME_S,
            ):
                return False
            _print_dynamic_pick_trace(
                robot,
                "after empirical-z move",
                target_z_mm=z_empirical_robot_frame_mm,
                note="At compensated empirical robot-frame Z, ready for dynamic grip.",
            )

        if hasattr(robot, "sync_estimate_from_teensy_steps"):
            if not robot.sync_estimate_from_teensy_steps():
                print("[PICK] WARN: failed to sync pose from Teensy after DL; estimate may be stale.")
        _print_dynamic_pick_trace(
            robot,
            "after DL result",
            target_z_mm=dynamic_plan.dynamic_lower_start_z_mm,
            note="q_est is re-synced from Teensy POS after DL. z_empirical remains grasp metadata only.",
        )
    else:
        print("[PICK] dynamic height disabled; using feedforward grasp Z descent and skipping DLR / empirical-z move.")

    _print_dynamic_pick_trace(
        robot,
        "before DG command",
        note=(
            f"About to send: DG {dynamic_plan.dynamic_grip_start_angle_deg:.3f} "
            f"{float(DYNAMIC_GRIP_DERIV_THRESH_MA):.3f} {int(_DYNAMIC_GRIP_N_STEPS)} "
            f"{1 if _DYNAMIC_GRIP_SIGNED_ONLY else 0}"
        ),
    )
    grip_result = robot.dynamic_grip(
        angle_start_deg=dynamic_plan.dynamic_grip_start_angle_deg,
        deriv_thresh_ma=DYNAMIC_GRIP_DERIV_THRESH_MA,
        n_steps=_DYNAMIC_GRIP_N_STEPS,
        signed_only=_DYNAMIC_GRIP_SIGNED_ONLY,
    )
    print(
        "[PICK] dynamic grip result: "
        f"ok={grip_result.ok} z_empirical_mm={grip_result.z_empirical_mm} "
        f"servo_empirical_deg={grip_result.servo_empirical_deg} error={grip_result.error}"
    )
    if not grip_result.ok:
        print("[PICK] ABORT: dynamic grip failed.")
        return False

    _store_dynamic_pick_results(
        c,
        dynamic_plan,
        lower_result,
        grip_result,
        z_empirical_robot_frame_mm=z_empirical_robot_frame_mm,
        actual_initial_servo_angle_deg=preset_servo_angle_deg,
    )

    _print_dynamic_pick_trace(robot, "before retract", target_z_mm=z_travel)
    if not _move_checked(robot, "[PICK] retract", z_mm=z_travel, move_time_s=COARSE_MOVE_TIME_S):
        return False
    _print_dynamic_pick_trace(robot, "after retract", target_z_mm=z_travel)

    print("[PICK] OK - dynamic lower/grip succeeded and item should be held.")
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
        config=DEFAULT_Z_SAFETY,
    )

    place_z = float(place_plan.final_release_z_mm)
    place_source = "shared_place_z_policy"
    if not USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT:
        print("[PLACE] dynamic place correction disabled; shared Z safety policy still controls release height.")

    place_phi = float(zone["default_phi_deg"])
    if USE_PICK_PHI_FOR_PLACE and held_object is not None:
        c = held_object.candidate
        if c.pick_phi_deg is not None and np.isfinite(float(c.pick_phi_deg)):
            place_phi = float(c.pick_phi_deg)

    travel_z = float(place_plan.approach_z_mm)
    release_servo_angle_deg, release_servo_source = _resolve_release_servo_angle_deg(
        held_object,
        fallback_deg=CLAW_OPEN_DEG,
    )
    for label, z in (("travel", travel_z), ("place", place_z), ("retract", float(place_plan.retract_z_mm))):
        reason = validate_z_command(z, f"[PLACE] {label}", config=DEFAULT_Z_SAFETY)
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
    print(f"[PLACE] release_servo_angle_deg = {release_servo_angle_deg:.2f} source={release_servo_source}")
    print("[PLACE Z PLAN]")
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
    print("[PLACE] sequence: raise -> XY/phi at safe Z -> descend -> open claw -> retract")

    if not _confirm("[PLACE] Real place motion will move to the saved zone and open the claw.", REQUIRE_CONFIRM_BEFORE_PLACE):
        print("[PLACE] canceled by user.")
        return False

    return execute_place_sequence(
        robot,
        held_object,
        target_xy_mm=np.array([x, y], dtype=np.float64),
        target_phi_deg=place_phi,
        place_z_plan=place_plan,
        settings=PlaceSequenceSettings(
            release_servo_deg=release_servo_angle_deg,
            coarse_move_time_s=COARSE_MOVE_TIME_S,
            xy_move_time_s=XY_MOVE_TIME_S,
            z_move_time_s=PLACE_Z_MOVE_TIME_S,
        ),
        config=DEFAULT_Z_SAFETY,
        check_pose_safe_fn=_check_robot_pose_safe,
        label_prefix="[PLACE]",
    )


def main() -> int:
    _configure_modules()
    if PICK_PHI_MODE not in VALID_PICK_PHI_MODES:
        raise ValueError(f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}; expected one of {sorted(VALID_PICK_PHI_MODES)}")

    _hr("PICK ONE PLACE ONE REPEATABILITY", "=")
    print("[MAIN] This is a single-object placement repeatability test, not full bagging.")
    print(f"[MAIN] place surface zone: {PLACE_SURFACE_ZONE_NAME} from {SURFACE_ZONE_CONFIG_PATH}")
    print(f"[MAIN] Z_MAX={Z_MAX_MM:.1f} PLACE_APPROACH_Z={PLACE_APPROACH_Z_MM:.1f}")
    print_z_safety_settings("[MAIN] Z safety", config=DEFAULT_Z_SAFETY)
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
    print(
        f"[MAIN] dynamic contact gate: "
        f"lookback={int(DYNAMIC_CONTACT_LOOKBACK_COUNT)} "
        f"eps={DYNAMIC_CONTACT_NONZERO_EPS_MA:.1f} "
        f"grip_sum={DYNAMIC_CONTACT_SUM_GRIP_MA:.1f} "
        f"lower_sum={DYNAMIC_CONTACT_SUM_LOWER_MA:.1f} "
        f"push_on_start={_PUSH_DYNAMIC_FIRMWARE_SETTINGS_ON_START}"
    )
    print(
        f"[MAIN] object rigidity default={DEFAULT_OBJECT_RIGIDITY} "
        f"squishable_extra_close_deg={SQUISHABLE_POST_CONTACT_EXTRA_CLOSE_DEG:.1f}"
    )
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
    _apply_dynamic_firmware_settings(robot)

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
