from __future__ import annotations

"""No-motion live validation for two-object adjacent placement planning."""

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

PAD_X_MM = 10.0
PAD_Y_MM = 10.0
PAD_Z_MM = 0.0
ADJACENT_DIRECTION = "right"

USE_FIXED_PLACE_PHI = True
FIXED_PLACE_PHI_DEG = 0.0

CONNECT_ROBOT = False
NO_MOTION = True

# Additional survey/display knobs kept aligned with pick_one_place_one defaults.
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

Z_MAX_MM: float = 275.0
GRIPPER_OFFSET_MM: float = 120.0
HOVER_HEIGHT_MM: float = Z_MAX_MM
GRASP_OFFSET_MM: float = GRIPPER_OFFSET_MM
USE_ROBUST_OBJECT_Z = True
ROBUST_TOP_PERCENTILE = 95.0
ROBUST_BOTTOM_PERCENTILE = 5.0
TOP_SPREAD_LOW_PERCENTILE = 90.0
TOP_SPREAD_HIGH_PERCENTILE = 99.0
PICK_Z_UNCERTAINTY_GAIN = 1.0
Z_UNCERTAINTY_CLEARANCE_MIN_MM = 0.0
PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 20.0
Z_UNCERTAINTY_WARN_MM = 10.0
PICK_EXTRA_CLEARANCE_MM = 0.0
PLACE_RELEASE_GAP_MM = 8.0
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

COMBINED_WIDTH_PX: int = 1280
OVERHEAD_DRAW_H_PX: int = 560
STEREO_DRAW_H_PX: int = 390
STATUS_H_PX: int = 140
WINDOW: str = "Validate Two-Object Placement Plan"

X_SURVEY = 100.0
Y_SURVEY = 100.0
Z_SURVEY = 250.0

# ============================================================

import sys
import traceback
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from planning.aabb_utils import make_aabb_from_center_size, aabb_from_object_candidate, pad_aabb
from planning.adjacent_placement import AdjacentPlacementPlan, compute_adjacent_placement
from scripts.aabb_visualization_helpers import corners_from_box, draw_box, draw_robot_axes, set_axes_equal
from scripts.pick_one_place_one import _load_place_scene
from scripts.pick_validation_display import _hr, make_display, print_validation
from test_calibration_bundle_live_stereo_z_pickplace import (
    load_bundle,
    load_stereo_calibration,
    print_matrix_labeled,
    read_command_key,
)
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_survey_pipeline import load_vision, run_survey
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


@dataclass
class PlacementValidationResult:
    object1_raw: Any
    object1_padded: Any
    object2_raw: Any
    object2_padded: Any
    object2_raw_at_target: Any
    object2_padded_at_target: Any
    plan: AdjacentPlacementPlan
    surface_z_mm: float
    place_phi_deg: float


class NoMotionFKRobot:
    """Minimal robot-like object for candidate construction without hardware."""

    def fk(self):
        return float(X_SURVEY), float(Y_SURVEY), float(Z_SURVEY), 0.0


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


def _print_candidate_selection(tag: str, dbg: CandidateDebug, selected_index: int, total: int) -> None:
    c = dbg.candidate
    box = aabb_from_object_candidate(c, default_label=tag)
    print(f"[{tag}] selected candidate [{selected_index + 1}/{total}] class={c.yolo.class_name}")
    print(f"[{tag}] target_xy_mm = ({float(c.target_xy[0]):.1f}, {float(c.target_xy[1]):.1f})")
    print(f"[{tag}] estimated_raw_size_mm = {box.size_xyz_mm.tolist()}")
    z_debug = getattr(c, "z_debug", None)
    if z_debug is None:
        print(f"[{tag}] z diagnostics: unavailable")
    else:
        print(
            f"[{tag}] z diagnostics: object_height_mm={z_debug.object_height_mm:.1f} "
            f"top_spread_mm={z_debug.top_spread_mm:.1f} uncertainty_clearance_mm={z_debug.uncertainty_clearance_mm:.1f}"
        )
        print(f"[{tag}] z warnings: {z_debug.warnings}")


def _compute_result(
    object1_dbg: CandidateDebug,
    object2_dbg: CandidateDebug,
    zone: dict,
) -> PlacementValidationResult:
    object1_raw = aabb_from_object_candidate(object1_dbg, default_label="object1_raw")
    object2_raw = aabb_from_object_candidate(object2_dbg, default_label="object2_raw")

    surface_z_mm = float(zone["surface_z_mm"])
    x1 = float(zone["center_xy_mm"][0])
    y1 = float(zone["center_xy_mm"][1])
    z1 = surface_z_mm + 0.5 * float(object1_raw.size_xyz_mm[2])

    object1_raw_placed = make_aabb_from_center_size(
        center_xyz_mm=np.array([x1, y1, z1], dtype=np.float64),
        size_xyz_mm=object1_raw.size_xyz_mm,
        label=f"{object1_raw.label}_placed_raw",
    )
    object1_padded = pad_aabb(object1_raw_placed, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
    object2_padded = pad_aabb(object2_raw, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)

    if USE_FIXED_PLACE_PHI:
        place_phi_deg = float(FIXED_PLACE_PHI_DEG)
    else:
        place_phi_deg = float(zone["default_phi_deg"])

    plan = compute_adjacent_placement(
        reference_padded_box=object1_padded,
        moving_padded_box=object2_padded,
        direction=ADJACENT_DIRECTION,
        surface_z_mm=surface_z_mm,
        place_phi_deg=place_phi_deg,
    )

    object2_raw_at_target = make_aabb_from_center_size(
        center_xyz_mm=plan.target_center_xyz_mm,
        size_xyz_mm=object2_raw.size_xyz_mm,
        label=f"{object2_raw.label}_target_raw",
    )
    object2_padded_at_target = pad_aabb(object2_raw_at_target, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)

    return PlacementValidationResult(
        object1_raw=object1_raw_placed,
        object1_padded=object1_padded,
        object2_raw=object2_raw,
        object2_padded=object2_padded,
        object2_raw_at_target=object2_raw_at_target,
        object2_padded_at_target=object2_padded_at_target,
        plan=plan,
        surface_z_mm=surface_z_mm,
        place_phi_deg=place_phi_deg,
    )


def _print_plan(result: PlacementValidationResult) -> None:
    print("[PLACEMENT PLAN]")
    print(f"object 1 raw size mm = {result.object1_raw.size_xyz_mm.tolist()}")
    print(f"object 1 padded size mm = {result.object1_padded.padded_box.size_xyz_mm.tolist()}")
    print(f"object 1 placed center xyz mm = {result.object1_raw.center_xyz_mm.tolist()}")
    print(f"object 2 raw size mm = {result.object2_raw.size_xyz_mm.tolist()}")
    print(f"object 2 padded size mm = {result.object2_padded.padded_box.size_xyz_mm.tolist()}")
    print(f"adjacent direction = {result.plan.direction}")
    print(f"object 2 target center xyz mm = {result.plan.target_center_xyz_mm.tolist()}")
    print(f"object 2 target xy mm = {result.plan.target_center_xy_mm.tolist()}")
    print(f"surface_z_mm = {result.surface_z_mm:.3f}")
    print(f"fixed_place_phi_deg = {result.place_phi_deg:.3f}")


def _show_plot(result: PlacementValidationResult) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR] matplotlib is required for visualization: {exc}")
        return False

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_box(ax, result.object1_raw, color="tab:blue", linestyle="-")
    draw_box(ax, result.object1_padded.padded_box, color="tab:blue", linestyle="--")
    draw_box(ax, result.object2_raw_at_target, color="tab:orange", linestyle="-")
    draw_box(ax, result.object2_padded_at_target.padded_box, color="tab:orange", linestyle="--")

    centers = np.vstack([
        result.object1_raw.center_xyz_mm,
        result.plan.target_center_xyz_mm,
    ])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=["blue", "orange"], s=55)
    ax.text(float(centers[0, 0]), float(centers[0, 1]), float(centers[0, 2]), "object1 placed")
    ax.text(float(centers[1, 0]), float(centers[1, 1]), float(centers[1, 2]), "object2 target")

    points = np.vstack([
        corners_from_box(result.object1_padded.padded_box),
        corners_from_box(result.object2_padded_at_target.padded_box),
        centers,
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    ])
    set_axes_equal(ax, points)
    draw_robot_axes(ax)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    tx = result.plan.target_center_xy_mm
    ax.set_title(
        "Two-Object Adjacent Placement (No Motion)\n"
        f"direction={result.plan.direction} target_xy=({tx[0]:.1f}, {tx[1]:.1f})"
    )
    plt.tight_layout()
    plt.show()
    return True


def main() -> int:
    if not NO_MOTION:
        print("[WARN] NO_MOTION=False in settings; this script never commands motion anyway.")
    print(f"[MAIN] CONNECT_ROBOT={CONNECT_ROBOT} NO_MOTION={NO_MOTION}")

    _configure_modules()
    _hr("VALIDATE TWO OBJECT PLACEMENT PLAN (NO MOTION)", "=")
    print("Controls: s=survey, r=rotate, 1=set object1, 2=set object2, p=plan, v=visualize, q/ESC=quit")

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
    robot_stub = NoMotionFKRobot()

    zone = _load_place_scene()
    state: SurveyState | None = None
    object1_dbg: CandidateDebug | None = None
    object2_dbg: CandidateDebug | None = None
    last_result: PlacementValidationResult | None = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + STATUS_H_PX)

    try:
        while True:
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
                break

            if key == "s":
                state = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot_stub, bundle)
                if state.candidates:
                    state.selected_index = 0
                    print("[SURVEY] selected candidate 1")
                else:
                    print("[SURVEY] no candidates")
                continue

            if key == "r":
                if state is None or not state.candidates:
                    print("[ROTATE] no candidates")
                else:
                    state.selected_index = (state.selected_index + 1) % len(state.candidates)
                    dbg = state.candidates[state.selected_index]
                    print(f"[ROTATE] selected [{state.selected_index + 1}/{len(state.candidates)}] {dbg.candidate.yolo.class_name}")
                continue

            if key == "1":
                if state is None or not state.candidates:
                    print("[OBJECT1] no candidates")
                else:
                    object1_dbg = state.candidates[state.selected_index]
                    _print_candidate_selection("OBJECT1", object1_dbg, state.selected_index, len(state.candidates))
                continue

            if key == "2":
                if state is None or not state.candidates:
                    print("[OBJECT2] no candidates")
                else:
                    object2_dbg = state.candidates[state.selected_index]
                    _print_candidate_selection("OBJECT2", object2_dbg, state.selected_index, len(state.candidates))
                continue

            if key == "v":
                if last_result is None:
                    if object1_dbg is None or object2_dbg is None:
                        print("[VIS] select object 1 and object 2 first")
                        continue
                    last_result = _compute_result(object1_dbg, object2_dbg, zone)
                _show_plot(last_result)
                continue

            if key == "p":
                if object1_dbg is None or object2_dbg is None:
                    print("[PLAN] select object 1 and object 2 first")
                    continue
                last_result = _compute_result(object1_dbg, object2_dbg, zone)
                _print_plan(last_result)
                continue

    finally:
        try:
            overhead.release()
        except Exception:
            pass
        try:
            stereo.release()
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
