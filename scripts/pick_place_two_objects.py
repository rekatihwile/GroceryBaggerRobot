from __future__ import annotations

"""Simple two-object adjacent placement demo (not full bagging)."""

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
    PLACE_RELEASE_GAP_MM as SHARED_PLACE_RELEASE_GAP_MM,
    print_z_safety_settings,
    validate_z_command,
)

PAD_X_MM = 10.0
PAD_Y_MM = 10.0
PAD_Z_MM = 0.0
ADJACENT_DIRECTION = "left"
MATCH_SECOND_OBJECT_CLASS = True
TARGET_OBJECT_COUNT = 2
AUTO_TARGET_COUNT_FROM_REFS = True

# IMPORTANT:
# This flag may disable optional dynamic correction, but it must not bypass the
# shared minimum height, safety padding, or minimum Z clamp.
USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT = False
PLACE_RELEASE_GAP_MM = SHARED_PLACE_RELEASE_GAP_MM
PLACE_Z_UNCERTAINTY_GAIN = 0.1
PLACE_Z_UNCERTAINTY_CLEARANCE_MAX_MM = 3.0
USE_DYNAMIC_RELEASE_FOR_PLACE = True
DYNAMIC_RELEASE_TIMEOUT_S = 45.0

# Pickup can still use the full open angle imported from pick_one_place_one.
# Placement should only crack the claw open so it does not hit the object already placed.
PLACE_CLAW_OPEN_DEG = 45
COARSE_MOVE_TIME_S = 1.10
XY_MOVE_TIME_S = 1.50
PLACE_Z_MOVE_TIME_S = 0.60

# When True, skip all YES confirmations for hands-off execution.
AUTONOMOUS_MODE = True

# ============================================================

import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

from motion.place_z_policy import compute_place_z_plan
from motion.pick_place_sequence import PlaceSequenceSettings, execute_place_sequence
from planning.aabb_utils import aabb_from_object_candidate, make_aabb_from_center_size, pad_aabb
from planning.adjacent_placement import compute_adjacent_placement
from scripts.pick_one_place_one import (
    BUNDLE_PATH,
    CLAW_OPEN_DEG,
    STEREO_CALIBRATION_PATH,
    USE_CUDA,
    USE_HALF,
    OVERHEAD_INDEX,
    STEREO_INDEX,
    WINDOW,
    COMBINED_WIDTH_PX,
    OVERHEAD_DRAW_H_PX,
    STEREO_DRAW_H_PX,
    STATUS_H_PX,
    Z_MAX_MM,
    _configure_modules,
    _confirm,
    get_use_z_ground_model_for_pick_surface,
    _load_place_surface_zone,
    preview_pick_grasp_z_with_and_without_zground,
    set_use_z_ground_model_for_pick_surface,
    execute_pick_selected,
)
from scripts.pick_validation_display import _hr, make_display, print_validation
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_survey_pipeline import load_vision, run_survey
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
    on_start_place_motion: Callable[[], None] | None = None,
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
        print("[PLACE2] dynamic place correction disabled; shared Z safety policy still controls release height.")

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
        reason = validate_z_command(z, f"[PLACE2] {label}", config=DEFAULT_Z_SAFETY)
        if reason:
            print(f"[PLACE2] ABORT: {reason}")
            return False

    print("[PLACE2 Z PLAN]")
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

    require_confirmation = not bool(AUTONOMOUS_MODE)
    if not _confirm("[PLACE2] Real place motion will move to computed adjacent target and open claw.", require_confirmation):
        print("[PLACE2] canceled by user")
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
        on_start_place_motion=on_start_place_motion,
        label_prefix="[PLACE2]",
    )


def _placed_occupancy_from_plan(center_xyz_mm: np.ndarray, size_xyz_mm: np.ndarray, label: str):
    raw = make_aabb_from_center_size(center_xyz_mm=center_xyz_mm, size_xyz_mm=size_xyz_mm, label=label)
    return pad_aabb(raw, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)


def _reset_runtime_state() -> tuple[list, SurveyState | None, CandidateDebug | None]:
    print("[RESET] clearing survey state, held object state, and placed occupancy list")
    return [], None, None


def _select_closest_candidate_to_reference(
    candidates: list[CandidateDebug],
    ref_dbg: CandidateDebug,
    *,
    require_same_class: bool,
) -> CandidateDebug | None:
    if not candidates:
        return None

    ref_xy = np.asarray(ref_dbg.candidate.target_xy, dtype=np.float64).reshape(2)
    ref_cls = str(ref_dbg.candidate.yolo.class_name)

    pool = candidates
    if require_same_class:
        same_cls = [c for c in candidates if str(c.candidate.yolo.class_name) == ref_cls]
        if same_cls:
            pool = same_cls
            print(f"[SELECT2] class match enabled: using {len(pool)} candidates with class={ref_cls}")
        else:
            print(f"[SELECT2 WARN] no same-class candidates for class={ref_cls}; falling back to all classes")

    best = None
    best_dist = float("inf")
    for cand in pool:
        xy = np.asarray(cand.candidate.target_xy, dtype=np.float64).reshape(2)
        d = float(np.linalg.norm(xy - ref_xy))
        if d < best_dist:
            best_dist = d
            best = cand

    if best is not None:
        print(
            "[SELECT2] nearest candidate selected "
            f"class={best.candidate.yolo.class_name} "
            f"distance_mm={best_dist:.1f} "
            f"ref_xy=({ref_xy[0]:.1f},{ref_xy[1]:.1f}) "
            f"cand_xy=({best.candidate.target_xy[0]:.1f},{best.candidate.target_xy[1]:.1f})"
        )
    return best


def main() -> int:
    _configure_modules()
    _hr("PICK PLACE TWO OBJECTS - ADJACENT DEMO", "=")
    print("[MAIN] Not full bagging: this demo places object2 adjacent to object1 only.")
    print(f"[MAIN] direction={ADJACENT_DIRECTION} padding=({PAD_X_MM},{PAD_Y_MM},{PAD_Z_MM}) mm")
    print(f"[MAIN] target_object_count={TARGET_OBJECT_COUNT}")
    print(f"[MAIN] auto_target_count_from_refs={AUTO_TARGET_COUNT_FROM_REFS}")
    print(f"[MAIN] autonomous_mode={AUTONOMOUS_MODE}")
    print(f"[MAIN] pick_z_ground_compensation={get_use_z_ground_model_for_pick_surface()}")
    print_z_safety_settings("[MAIN] Z safety", config=DEFAULT_Z_SAFETY)
    print("[MAIN] controls: s=survey, r=rotate, 1..9=set ref slot, c=set next ref slot, l=list refs, v=validate+grasp compare, t=toggle z_ground pick compensation, a=run N-object flow, x=reset state, q=quit")

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

    placed_boxes: list = []
    state: SurveyState | None = None
    held: CandidateDebug | None = None
    if AUTO_TARGET_COUNT_FROM_REFS:
        object_refs: list[CandidateDebug | None] = []
    else:
        object_refs = [None for _ in range(int(max(1, TARGET_OBJECT_COUNT)))]
    survey_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="survey_prefetch")
    prefetched_future: Future | None = None

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
                state = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
                if state.candidates:
                    state.selected_index = 0
                    print("[SURVEY] selected candidate 1")
                else:
                    print("[SURVEY] no candidates")
                continue
            if key == "n":
                robot.send("HOMEJ3")
            if key == "g":
                robot.move_cartesian(200,200,100,0, move_time_s=2.0)

            if key == "x":
                placed_boxes, state, held = _reset_runtime_state()
                if AUTO_TARGET_COUNT_FROM_REFS:
                    object_refs = []
                else:
                    object_refs = [None for _ in range(int(max(1, TARGET_OBJECT_COUNT)))]
                continue


            if key == "r":
                if state is None or not state.candidates:
                    print("[ROTATE] no candidates")
                else:
                    state.selected_index = (state.selected_index + 1) % len(state.candidates)
                    dbg = state.candidates[state.selected_index]
                    print(f"[ROTATE] selected [{state.selected_index + 1}/{len(state.candidates)}] {dbg.candidate.yolo.class_name}")
                continue

            if key == "l":
                print("[REFS] current reference slots")
                for idx, ref in enumerate(object_refs, start=1):
                    if ref is None:
                        print(f"  {idx}: <unset>")
                    else:
                        c = ref.candidate
                        print(f"  {idx}: {c.yolo.class_name} xy=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f})")
                continue

            if key == "c":
                if state is None or not state.candidates:
                    print("[SET] no candidates; press s first")
                    continue
                next_slot = None
                for i_slot, ref in enumerate(object_refs):
                    if ref is None:
                        next_slot = i_slot
                        break
                if next_slot is None:
                    if AUTO_TARGET_COUNT_FROM_REFS:
                        object_refs.append(None)
                        next_slot = len(object_refs) - 1
                    else:
                        print("[SET] all reference slots already filled")
                        continue
                object_refs[next_slot] = state.candidates[state.selected_index]
                c = object_refs[next_slot].candidate
                print(
                    f"[SET] object{next_slot + 1} reference = {c.yolo.class_name} "
                    f"xy=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f})"
                )
                continue

            if key.isdigit() and key != "0":
                slot = int(key) - 1
                if slot >= len(object_refs):
                    if AUTO_TARGET_COUNT_FROM_REFS:
                        object_refs.extend([None for _ in range(slot + 1 - len(object_refs))])
                    else:
                        print(f"[SET] slot {slot + 1} out of range; TARGET_OBJECT_COUNT={len(object_refs)}")
                        continue
                if state is None or not state.candidates:
                    print("[SET] no candidates; press s first")
                    continue
                object_refs[slot] = state.candidates[state.selected_index]
                c = object_refs[slot].candidate
                print(
                    f"[SET] object{slot + 1} reference = {c.yolo.class_name} "
                    f"xy=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f})"
                )
                continue

            if key == "v":
                print_validation(state, 0 if state is None else state.selected_index)
                if state is None or not state.candidates:
                    print("[COMPARE] no candidate selected; press s first")
                    continue
                try:
                    dbg = state.candidates[state.selected_index]
                    cmp = preview_pick_grasp_z_with_and_without_zground(dbg)
                    print("[COMPARE PICK Z]")
                    print(
                        f"candidate={dbg.candidate.yolo.class_name} "
                        f"xy=({cmp['x_mm']:.1f},{cmp['y_mm']:.1f}) "
                        f"active_mode={cmp['active_mode']}"
                    )
                    print(
                        f"grasp_z_stereo_only={cmp['base_grasp_z_mm']:.2f} "
                        f"grasp_z_zground={cmp['zground_grasp_z_mm']:.2f} "
                        f"delta={cmp['delta_grasp_z_mm']:+.2f}"
                    )
                    print(
                        f"z_ground_mm={cmp['z_ground_mm']} "
                        f"object_height_mm={cmp['object_height_mm']} "
                        f"surface_override_mm={cmp['surface_override_mm']} "
                        f"model_available={cmp['zground_model_available']}"
                    )
                except Exception as exc:
                    print(f"[COMPARE] failed: {exc}")
                continue

            if key == "t":
                new_state = set_use_z_ground_model_for_pick_surface(
                    not get_use_z_ground_model_for_pick_surface()
                )
                print(f"[MAIN] pick_z_ground_compensation={new_state}")
                continue

            if key != "a":
                continue

            if AUTO_TARGET_COUNT_FROM_REFS:
                target_count = int(sum(1 for r in object_refs if r is not None))
                if target_count <= 0:
                    print("[FLOW] no references assigned. Use c or number keys after survey.")
                    continue
                if any(ref is None for ref in object_refs[:target_count]):
                    print("[FLOW] references must be contiguous from slot 1..N in auto mode")
                    continue
            else:
                if any(ref is None for ref in object_refs):
                    print("[FLOW] set references for all slots 1..N before running (use c or number keys)")
                    continue
                target_count = int(max(1, TARGET_OBJECT_COUNT))
            placed_boxes = []
            held = None
            column_xy_primary: np.ndarray | None = None
            column_xy_secondary: np.ndarray | None = None

            def _prefetch_next_survey_async() -> None:
                nonlocal prefetched_future
                if prefetched_future is not None and not prefetched_future.done():
                    return
                prefetched_future = survey_executor.submit(
                    run_survey,
                    overhead,
                    stereo,
                    detector,
                    stereo_calib,
                    rectifier,
                    yolo,
                    raft,
                    robot,
                    bundle,
                )
                print("[PREFETCH] async survey submitted")

            run_ok = True
            for i in range(1, target_count + 1):
                print(f"[FLOW] Object {i}/{target_count}")
                ref_dbg = object_refs[i - 1]
                if ref_dbg is None:
                    print(f"[FLOW] object {i} reference is unset")
                    run_ok = False
                    break

                if i == 1:
                    cand_dbg = ref_dbg
                else:
                    state = None
                    if prefetched_future is not None:
                        if not prefetched_future.done():
                            print(f"[FLOW] waiting for async prefetched survey for object {i}")
                        try:
                            state = prefetched_future.result()
                            print(f"[FLOW] using prefetched survey for object {i}")
                        except Exception as exc:
                            print(f"[FLOW WARN] async prefetch survey failed: {exc}")
                        finally:
                            prefetched_future = None

                    if state is None:
                        state = run_survey(overhead, stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)

                    if not state.candidates:
                        print(f"[FLOW] no candidates available for object {i}")
                        run_ok = False
                        break

                    cand_dbg = _select_closest_candidate_to_reference(
                        state.candidates,
                        ref_dbg,
                        require_same_class=MATCH_SECOND_OBJECT_CLASS,
                    )
                    if cand_dbg is None:
                        print(f"[FLOW] unable to choose candidate for object {i}")
                        run_ok = False
                        break
                    state.selected_index = state.candidates.index(cand_dbg)

                if not execute_pick_selected(robot, cand_dbg, bundle=bundle):
                    print(f"[FLOW] object {i} pick failed")
                    run_ok = False
                    break
                held = cand_dbg

                raw_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=f"object{i}")
                object_height_mm = float(raw_box.size_xyz_mm[2])

                next_prefetch_cb = _prefetch_next_survey_async if i < target_count else None
                destination_surface_for_call = float(surface_z)
                below_top_z_mm: float | None = None

                if i == 1:
                    target_xy = base_xy
                    target_phi = place_phi
                    column_xy_primary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                elif i == 2:
                    moving_padded = pad_aabb(raw_box, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
                    plan = compute_adjacent_placement(
                        reference_padded_box=placed_boxes[-1],
                        moving_padded_box=moving_padded,
                        direction=ADJACENT_DIRECTION,
                        surface_z_mm=surface_z,
                        place_phi_deg=place_phi,
                    )
                    print(f"[FLOW] object{i} adjacent target xyz = {plan.target_center_xyz_mm.tolist()}")
                    target_xy = plan.target_center_xy_mm
                    target_phi = plan.target_phi_deg
                    column_xy_secondary = np.asarray(target_xy, dtype=np.float64).reshape(2)
                else:
                    # 2-per-layer stacking: object i is placed above object i-2.
                    # Odd indices stay in object1 column, even indices in object2 column.
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

                if not _execute_place_at_target(
                    robot,
                    held,
                    target_xy_mm=target_xy,
                    target_phi_deg=target_phi,
                    destination_surface_z_mm=destination_surface_for_call,
                    on_start_place_motion=next_prefetch_cb,
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
                    placed = _placed_occupancy_from_plan(
                        center_xyz_mm=np.array([
                            float(plan.target_center_xy_mm[0]),
                            float(plan.target_center_xy_mm[1]),
                            surface_z + 0.5 * object_height_mm,
                        ], dtype=np.float64),
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

            if run_ok:
                print(f"[FLOW] success: placed {len(placed_boxes)} objects")

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
