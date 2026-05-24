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
PLACE_RELEASE_GAP_MM = 8.0
MIN_OBJECT_HEIGHT_MM = 5.0
MAX_OBJECT_HEIGHT_MM = 180.0

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

import sys
import traceback

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import vision.burst_tracking as _burst_mod
import vision.object_geometry as _geom_mod
import vision.pick_candidate_builder as _candidate_mod
import vision.pick_phi_resolver as _phi_mod
import vision.pick_survey_pipeline as _survey_mod
import vision.pick_xy_resolver as _xy_mod
import vision.pick_z_resolver as _z_mod
import vision.pointcloud as _pointcloud_mod
import scripts.pick_validation_display as _display_mod
import motion.pick_validation_motion as _motion_mod

from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from motion.pick_validation_motion import _print_fk, execute_test_descent_no_claw, startup_robot
from scripts.pick_validation_display import _hr, make_display, print_validation
from test_calibration_bundle_live_stereo_z_pickplace import (
    jog_nonnegative_z,
    load_bundle,
    load_stereo_calibration,
    print_matrix_labeled,
    read_command_key,
)
from vision.pick_candidate_builder import SurveyState
from vision.pick_survey_pipeline import load_vision, run_survey
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
    _z_mod.Z_UNCERTAINTY_CLEARANCE_GAIN = Z_UNCERTAINTY_CLEARANCE_GAIN
    _z_mod.Z_UNCERTAINTY_CLEARANCE_MIN_MM = Z_UNCERTAINTY_CLEARANCE_MIN_MM
    _z_mod.Z_UNCERTAINTY_CLEARANCE_MAX_MM = Z_UNCERTAINTY_CLEARANCE_MAX_MM
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
    _motion_mod.FINE_TUNE_BLEND_STEP = FINE_TUNE_BLEND_STEP
    _motion_mod.RAISE_MOVE_TIME_S = RAISE_MOVE_TIME_S
    _motion_mod.XY_MOVE_TIME_S = XY_MOVE_TIME_S
    _motion_mod.DESCENT_STEP_MM = DESCENT_STEP_MM
    _motion_mod.DESCENT_STEP_TIME_S = DESCENT_STEP_TIME_S
    _motion_mod.FINE_TUNE_XY_MOVE_TIME_S = FINE_TUNE_XY_MOVE_TIME_S
    _motion_mod.CLAW_OPEN_DEG = CLAW_OPEN_DEG
    _motion_mod.REQUIRE_OVERHEAD_XY_FOR_TEST = REQUIRE_OVERHEAD_XY_FOR_TEST
    _motion_mod.REFUSE_TEST_IF_TOO_FEW_POINTS = REFUSE_TEST_IF_TOO_FEW_POINTS


def main() -> int:
    _configure_modules()
    if PICK_PHI_MODE not in VALID_PICK_PHI_MODES:
        raise ValueError(
            f"Unknown PICK_PHI_MODE={PICK_PHI_MODE!r}; "
            f"expected one of {sorted(VALID_PICK_PHI_MODES)}"
        )

    _hr("REAL OBJECT PICK VALIDATION BUILDUP", "=")
    print(f"[MAIN] burst: {BURST_COUNT} frames, keep if hits >= {MIN_BURST_HITS}")
    print(
        f"[MAIN] Z policy: robust={USE_ROBUST_OBJECT_Z} "
        f"top_p={ROBUST_TOP_PERCENTILE:.1f} bottom_p={ROBUST_BOTTOM_PERCENTILE:.1f} "
        f"travel=hover=Z_MAX={Z_MAX_MM:.1f}, grasp=top+offset+clearance"
    )
    print(f"[MAIN] XY blend: overhead weight={OVERHEAD_XY_BLEND_WEIGHT:.2f}, stereo={1.0 - OVERHEAD_XY_BLEND_WEIGHT:.2f}")
    print(f"[MAIN] phi from overhead semiminor after match: {OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI}")
    print(
        f"[MAIN] phi confidence blend: USE={USE_CONFIDENCE_PHI_BLEND}  "
        f"min_conf={PHI_MIN_CONFIDENCE:.2f}  disagree_warn={PHI_DISAGREEMENT_WARN_DEG:.1f} deg  "
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
                print(f"[CLAW] open ({CLAW_OPEN_DEG} deg)")

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
