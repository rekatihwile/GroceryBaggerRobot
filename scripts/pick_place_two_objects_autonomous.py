from __future__ import annotations

"""Autonomous adjacent placement demo using largest-volume candidate selection."""

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

# How many objects to place. Set RUN_UNTIL_NO_VALID_CANDIDATE=True for a feeder loop.
TARGET_OBJECT_COUNT = 10
RUN_UNTIL_NO_VALID_CANDIDATE = False
MAX_OBJECT_COUNT_SAFETY = 12
NO_CANDIDATE_RETRY_COUNT = 1
NO_CANDIDATE_RETRY_DELAY_S = 1.0

# Placement geometry. Object 1 goes at the saved zone center. Object 2 is adjacent.
# Object 3 stacks above object 1, object 4 stacks above object 2, and so on.
PAD_X_MM = 10.0
PAD_Y_MM = 10.0
PAD_Z_MM = 0.0
ADJACENT_DIRECTION = "left"

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

# Best-candidate filters. These are intentionally conservative and easy to tune.
BEST_REQUIRE_POSITIVE_PLATFORM_XY = True
BEST_PLATFORM_MIN_X_MM = 0.0
BEST_PLATFORM_MIN_Y_MM = 0.0
BEST_WORKSPACE_X_MIN_MM = 0.0
BEST_WORKSPACE_X_MAX_MM = 600.0
BEST_WORKSPACE_Y_MIN_MM = 0.0
BEST_WORKSPACE_Y_MAX_MM = 600.0
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
USE_DYNAMIC_RELEASE_FOR_PLACE = True
DYNAMIC_RELEASE_TIMEOUT_S = 45.0

# Placement should only crack the claw open so it does not hit the object already placed.
PLACE_CLAW_OPEN_DEG = 45
COARSE_MOVE_TIME_S = 1.10
XY_MOVE_TIME_S = 1.50
PLACE_Z_MOVE_TIME_S = 0.60

WINDOW = "Autonomous Pick Place - Largest Volume"

# ============================================================

import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from motion.place_z_policy import compute_place_z_plan
from motion.pick_place_sequence import PlaceSequenceSettings, execute_place_sequence
from planning.aabb_utils import aabb_from_object_candidate, make_aabb_from_center_size, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement
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
from vision.pick_survey_pipeline import load_vision, run_survey
import vision.pick_survey_pipeline as _survey_pipeline_mod
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device
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


def _best_candidate_config() -> BestCandidateConfig:
    return BestCandidateConfig(
        require_positive_platform_xy=BEST_REQUIRE_POSITIVE_PLATFORM_XY,
        platform_min_x_mm=BEST_PLATFORM_MIN_X_MM,
        platform_min_y_mm=BEST_PLATFORM_MIN_Y_MM,
        workspace_x_min_mm=BEST_WORKSPACE_X_MIN_MM,
        workspace_x_max_mm=BEST_WORKSPACE_X_MAX_MM,
        workspace_y_min_mm=BEST_WORKSPACE_Y_MIN_MM,
        workspace_y_max_mm=BEST_WORKSPACE_Y_MAX_MM,
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


def _execute_place_at_target(
    robot,
    held_object: CandidateDebug,
    *,
    target_xy_mm: np.ndarray,
    target_phi_deg: float,
    destination_surface_z_mm: float,
    on_start_place_descent: Callable[[], None] | None = None,
) -> bool:
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
    print("[PLACE AUTO] place-descent callback will start the next survey." if on_start_place_descent else "[PLACE AUTO] no next survey callback for this placement.")

    if not _confirm("[PLACE AUTO] Real place motion will execute.", False):
        print("[PLACE AUTO] canceled by user")
        return False

    return execute_place_sequence(
        robot,
        held_object,
        target_xy_mm=np.array([x, y], dtype=np.float64),
        target_phi_deg=phi,
        place_z_plan=place_plan,
        settings=PlaceSequenceSettings(
            release_servo_deg=PLACE_CLAW_OPEN_DEG,
            coarse_move_time_s=COARSE_MOVE_TIME_S,
            xy_move_time_s=XY_MOVE_TIME_S,
            z_move_time_s=PLACE_Z_MOVE_TIME_S,
            use_dynamic_release=USE_DYNAMIC_RELEASE_FOR_PLACE,
            dynamic_release_timeout_s=DYNAMIC_RELEASE_TIMEOUT_S,
            dynamic_release_max_open_deg=release_max_open_deg,
        ),
        config=DEFAULT_Z_SAFETY,
        on_start_place_descent=on_start_place_descent,
        label_prefix="[PLACE AUTO]",
    )


def _placed_occupancy_from_plan(center_xyz_mm: np.ndarray, size_xyz_mm: np.ndarray, label: str):
    raw = make_aabb_from_center_size(center_xyz_mm=center_xyz_mm, size_xyz_mm=size_xyz_mm, label=label)
    return pad_aabb(raw, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)


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
) -> np.ndarray:
    frame = make_display(state, live_overhead, live_left, live_right)
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
) -> None:
    if read_live:
        live_overhead, live_left, live_right = _read_live_frames(overhead, stereo)
    else:
        live_overhead = live_left = live_right = None
    cv2.imshow(WINDOW, _make_autonomous_display(state, live_overhead, live_left, live_right, status_lines))
    _check_abort_key()


def _check_abort_key(delay_ms: int = 1) -> None:
    key = read_command_key(delay_ms=delay_ms)
    if key in ("q", "escape", "\x1b"):
        raise UserAbort("quit requested")
    if key == "c":
        raise ClearBoxRequest("clear box requested")


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


def main() -> int:
    _configure_modules()
    _survey_pipeline_mod.OVERHEAD_FRESH_READ_DISCARD_FRAMES = OVERHEAD_FRESH_READ_DISCARD_FRAMES
    _survey_pipeline_mod.OVERHEAD_FRESH_READ_DELAY_S = OVERHEAD_FRESH_READ_DELAY_S
    _hr("AUTONOMOUS PICK PLACE - LARGEST VOLUME", "=")
    print("[MAIN] No manual survey step. The script surveys, filters candidates, and picks the largest valid volume.")
    print(f"[MAIN] target_object_count={TARGET_OBJECT_COUNT} run_until_no_valid={RUN_UNTIL_NO_VALID_CANDIDATE}")
    print(f"[MAIN] placement direction={ADJACENT_DIRECTION} padding=({PAD_X_MM},{PAD_Y_MM},{PAD_Z_MM}) mm")
    print(f"[MAIN] prefetch_on_place_descent={PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT} delay={PREFETCH_PLACE_DESCENT_DELAY_S:.2f}s")
    print(f"[MAIN] overhead_fresh_read discard={OVERHEAD_FRESH_READ_DISCARD_FRAMES} delay={OVERHEAD_FRESH_READ_DELAY_S:.3f}s")
    print(f"[MAIN] pick_z_ground_compensation={get_use_z_ground_model_for_pick_surface()}")
    print(
        "[MAIN] best filters: "
        f"platform_xy>=({BEST_PLATFORM_MIN_X_MM:.1f},{BEST_PLATFORM_MIN_Y_MM:.1f}) "
        f"workspace_x=[{BEST_WORKSPACE_X_MIN_MM},{BEST_WORKSPACE_X_MAX_MM}] "
        f"workspace_y=[{BEST_WORKSPACE_Y_MIN_MM},{BEST_WORKSPACE_Y_MAX_MM}] "
        f"center_gate={'on' if BEST_CENTER_GATE_ENABLED else 'off'}"
        f"({BEST_MAX_IMAGE_CENTER_NORM_RADIUS:.2f}) "
        f"cluster_gate={'on' if BEST_CLUSTER_GATE_ENABLED else 'off'}"
        f"({BEST_MAX_CLUSTER_DISTANCE_MM:.1f}mm) "
        f"max_volume={BEST_MAX_VOLUME_CM3:.0f}cm3"
    )
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

    survey_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="survey_place_descent")
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

    def handle_clear_box_request() -> None:
        nonlocal placed_boxes, state, held, column_xy_primary, column_xy_secondary, prefetched_future
        print("[CLEAR BOX] requested from camera window.")
        set_status([
            "CLEAR BOX: raising arm",
            "Empty the box when motion completes.",
            "Then press Enter in the terminal.",
            "After clear: continuous mode, no object count limit.",
        ])
        _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)

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
        _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)
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
        _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)
        surveyed = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
        state = surveyed
        print(f"[SURVEY AUTO] completed blocking survey for {label}: candidates={len(surveyed.candidates)}")
        return surveyed

    def prefetch_worker(next_object_i: int) -> SurveyState:
        if PREFETCH_PLACE_DESCENT_DELAY_S > 0.0:
            time.sleep(float(PREFETCH_PLACE_DESCENT_DELAY_S))
        print(f"[PREFETCH] object {next_object_i}: running survey in background from place descent")
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
            _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=False)
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

    try:
        _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)

        i = 1
        while run_until_no_valid_active() or i <= target_limit:
            try:
                print(f"\n[FLOW] Object {i}/{'continuous' if run_until_no_valid_active() else target_limit}")
                survey_state = get_survey_for_object(i)

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
                        _show_display(state=survey_state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)
                        time.sleep(0.05)
                    survey_state = run_survey_now(f"{flow_label(i)} retry {retry_i}")
                    selection = _select_best_for_state(
                        survey_state,
                        robot=robot,
                        placed_boxes=placed_boxes,
                        config=selector_config,
                    )

                set_status(selection.display_lines(max_lines=5))
                _show_display(state=survey_state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=False)
                if SELECTION_DISPLAY_HOLD_S > 0.0:
                    time.sleep(float(SELECTION_DISPLAY_HOLD_S))

                if selection.selected is None:
                    print(f"[FLOW] stopping: no valid candidate for object {i}")
                    if run_until_no_valid_active():
                        break
                    run_ok = False
                    break

                cand_dbg = selection.selected
                c = cand_dbg.candidate
                print(
                    f"[FLOW] picking object {i}: candidate [{c.index}] {c.yolo.class_name} "
                    f"xy=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f})"
                )

                if not execute_pick_selected(robot, cand_dbg, bundle=bundle):
                    print(f"[FLOW] object {i} pick failed")
                    run_ok = False
                    break
                held = cand_dbg

                raw_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=f"object{i}")
                object_height_mm = float(raw_box.size_xyz_mm[2])

                destination_surface_for_call = float(surface_z)
                below_top_z_mm: float | None = None
                target_xy: np.ndarray
                target_phi: float
                adjacent_plan = None

                if i == 1:
                    target_xy = base_xy
                    target_phi = place_phi
                    column_xy_primary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                    print(f"[FLOW] object1 target = base zone xy=({target_xy[0]:.1f},{target_xy[1]:.1f})")
                elif i == 2:
                    moving_padded = pad_aabb(raw_box, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
                    adjacent_plan = compute_adjacent_placement(
                        reference_padded_box=placed_boxes[-1],
                        moving_padded_box=moving_padded,
                        direction=ADJACENT_DIRECTION,
                        surface_z_mm=surface_z,
                        place_phi_deg=place_phi,
                    )
                    target_xy = adjacent_plan.target_center_xy_mm
                    target_phi = adjacent_plan.target_phi_deg
                    column_xy_secondary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                    print(f"[FLOW] object2 adjacent target xyz = {adjacent_plan.target_center_xyz_mm.tolist()}")
                else:
                    if column_xy_primary is None or column_xy_secondary is None:
                        print(f"[FLOW] missing column XY anchors for object {i}")
                        run_ok = False
                        break

                    below_idx = i - 2
                    below_box = placed_boxes[below_idx - 1]
                    below_top_z_mm = float(below_box.raw_box.max_xyz_mm[2])

                    if i % 2 == 1:
                        target_xy = column_xy_primary
                    else:
                        target_xy = column_xy_secondary
                    target_phi = place_phi
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

                if not _execute_place_at_target(
                    robot,
                    held,
                    target_xy_mm=target_xy,
                    target_phi_deg=target_phi,
                    destination_surface_z_mm=destination_surface_for_call,
                    on_start_place_descent=place_descent_cb,
                ):
                    print(f"[FLOW] object {i} place failed")
                    held = None
                    run_ok = False
                    break

                if i == 1:
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array([base_xy[0], base_xy[1], surface_z + 0.5 * object_height_mm], dtype=np.float64),
                        size_xyz_mm=raw_box.size_xyz_mm,
                        label="object1_placed",
                    )
                    placed_boxes.append(placed)
                elif i == 2:
                    if adjacent_plan is None:
                        print("[FLOW] missing adjacent placement plan for object2")
                        run_ok = False
                        break
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array(
                            [
                                float(adjacent_plan.target_center_xy_mm[0]),
                                float(adjacent_plan.target_center_xy_mm[1]),
                                surface_z + 0.5 * object_height_mm,
                            ],
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
                _show_display(state=state, overhead=overhead, stereo=stereo, status_lines=status_lines, read_live=True)
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
