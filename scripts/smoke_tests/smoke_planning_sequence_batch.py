from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import time
import traceback

import cv2
import numpy as np

import scripts.autonomous_missed_pick_recovery as wet_mod
import scripts.dry_run_autonomous as dry_mod
from planning.autonomous_planning_sequences import (
    list_planning_sequences,
    validate_planning_sequence_name,
)
from planning.bag_local_3d_aabb_planner import aabbs_intersect_3d
from test_calibration_bundle_live_stereo_z_pickplace import load_bundle
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter


@dataclass
class PairValidationResult:
    pair_index: int
    candidate_count: int
    placed_count: int
    expected_count: int
    success: bool
    reason: str
    planning_step_times_s: list[float]
    total_planning_time_s: float


def _load_models_and_calibration():
    dry_mod._configure_vision_modules()
    calib_path = REPO_ROOT / dry_mod._SURVEY.STEREO_CALIBRATION_PATH
    stereo_data = np.load(calib_path)
    stereo_calib = {k: np.asarray(stereo_data[k]) for k in stereo_data.files}

    bundle_path = REPO_ROOT / dry_mod._SURVEY.BUNDLE_PATH
    if not bundle_path.exists():
        bundle_path = dry_mod._SURVEY.BUNDLE_PATH
    bundle = load_bundle(bundle_path) if Path(bundle_path).exists() else {}

    device_info = select_torch_device(use_cuda=dry_mod._SURVEY.USE_CUDA, use_half=dry_mod._SURVEY.USE_HALF)

    weights = REPO_ROOT / dry_mod._SURVEY.YOLO_WEIGHTS_PATH
    if not weights.exists():
        weights = REPO_ROOT / dry_mod._SURVEY.YOLO_FALLBACK_WEIGHTS_PATH
    yolo = YOLOSegmenter(
        weights_path=str(weights),
        device_info=device_info,
        imgsz=dry_mod._SURVEY.YOLO_IMGSZ,
        conf=dry_mod._SURVEY.YOLO_CONF,
        iou=dry_mod._SURVEY.YOLO_IOU,
        retina_masks=dry_mod._SURVEY.YOLO_RETINA_MASKS,
        min_mask_area_px=dry_mod._SURVEY.MIN_MASK_AREA_PX,
        target_class_names=list(dry_mod._SURVEY.TARGET_CLASS_NAMES) if dry_mod._SURVEY.TARGET_CLASS_NAMES else None,
    )
    yolo.warmup()

    raft = RAFTStereoRunner(
        raft_root=str(REPO_ROOT / dry_mod._SURVEY.RAFT_ROOT),
        checkpoint_path=str(REPO_ROOT / dry_mod._SURVEY.RAFT_CHECKPOINT_PATH),
        device_info=device_info,
        valid_iters=dry_mod._SURVEY.RAFT_VALID_ITERS,
        downscale=dry_mod._SURVEY.RAFT_DOWNSCALE,
        mixed_precision=dry_mod._SURVEY.RAFT_MIXED_PRECISION,
    )
    raft.warmup()

    rectifier = StereoRectifier(stereo_calib)
    return stereo_calib, bundle, yolo, raft, rectifier


def _simulate_pair(
    *,
    survey_state,
    sequence_name: str,
    expected_count: int,
) -> PairValidationResult:
    state = replace(survey_state, candidates=list(survey_state.candidates), selected_index=0)
    selector_config = dry_mod._best_candidate_config()
    surface_zone = wet_mod._load_place_scene()
    surface_z = float(surface_zone["surface_z_mm"])
    base_xy = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    place_phi = float(surface_zone["default_phi_deg"])

    old_sequence_name = wet_mod.PLACE_PLANNING_SEQUENCE_NAME
    wet_mod.PLACE_PLANNING_SEQUENCE_NAME = sequence_name
    planning_step_times_s: list[float] = []
    placed_boxes: list = []
    reason = "ok"

    try:
        for object_i in range(1, expected_count + 1):
            t0 = time.perf_counter()
            selection, _overlay = wet_mod._select_best_for_state_by_slot_fit(
                state,
                object_i=object_i,
                robot=None,
                placed_boxes=placed_boxes,
                config=selector_config,
                surface_zone=surface_zone,
                base_xy=base_xy,
                base_phi_deg=place_phi,
                target_limit=expected_count,
            )
            planning_step_times_s.append(time.perf_counter() - t0)
            cand_dbg = selection.selected
            if cand_dbg is None:
                reason = f"no_selection_object_{object_i}"
                break

            try:
                optimized_target = wet_mod._compute_candidate_place_target_for_sequence(
                    cand_dbg,
                    state=state,
                    object_i=object_i,
                    robot=None,
                    placed_boxes=placed_boxes,
                    config=selector_config,
                    surface_zone=surface_zone,
                    base_xy=base_xy,
                    base_phi_deg=place_phi,
                    target_limit=expected_count,
                )
            except Exception as exc:
                reason = f"target_error_object_{object_i}:{exc}"
                break
            if not bool(optimized_target.can_place):
                reason = f"target_rejected_object_{object_i}:{optimized_target.reason}"
                break

            target_xy = np.asarray(optimized_target.target_xy_mm, dtype=np.float64).reshape(2).copy()
            target_phi = float(optimized_target.target_phi_deg)
            raw_box = wet_mod._aabb_from_object_candidate_quiet(cand_dbg.candidate, default_label=f"pair_obj{object_i}")
            object_height_mm = float(raw_box.size_xyz_mm[2])

            if not wet_mod._validate_gripper_footprint_inside_bag(
                cand_dbg,
                target_xy_mm=target_xy,
                target_phi_deg=target_phi,
                surface_zone=surface_zone,
                label=f"pair_obj{object_i}",
            ):
                reason = f"footprint_gate_object_{object_i}"
                break

            if object_i == 1:
                center_z = surface_z + 0.5 * object_height_mm
            elif object_i == 2:
                center_z = surface_z + 0.5 * object_height_mm
            else:
                below_box = placed_boxes[object_i - 3]
                center_z = float(below_box.raw_box.max_xyz_mm[2]) + 0.5 * object_height_mm

            placed = wet_mod._placed_occupancy_from_plan(
                center_xyz_mm=np.array([float(target_xy[0]), float(target_xy[1]), float(center_z)], dtype=np.float64),
                size_xyz_mm=np.asarray(raw_box.size_xyz_mm, dtype=np.float64),
                label=f"pair_obj{object_i}_placed",
            )
            placed_boxes.append(placed)
            state.candidates = [dbg for dbg in state.candidates if dbg is not cand_dbg]

        if reason == "ok" and len(placed_boxes) != expected_count:
            reason = f"placed_{len(placed_boxes)}_of_{expected_count}"

        if reason == "ok":
            for i, box_a in enumerate(placed_boxes):
                for j, box_b in enumerate(placed_boxes[i + 1 :], start=i + 1):
                    if aabbs_intersect_3d(box_a.raw_box, box_b.raw_box):
                        reason = f"raw_intersection_{i + 1}_{j + 1}"
                        break
                    if aabbs_intersect_3d(box_a.padded_box, box_b.padded_box):
                        reason = f"padded_intersection_{i + 1}_{j + 1}"
                        break
                if reason != "ok":
                    break
    finally:
        wet_mod.PLACE_PLANNING_SEQUENCE_NAME = old_sequence_name

    return PairValidationResult(
        pair_index=-1,
        candidate_count=int(len(survey_state.candidates)),
        placed_count=len(placed_boxes),
        expected_count=expected_count,
        success=(reason == "ok"),
        reason=reason,
        planning_step_times_s=planning_step_times_s,
        total_planning_time_s=float(sum(planning_step_times_s)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-validate wet planning sequences on saved stereo pairs")
    parser.add_argument("--images", default=str(REPO_ROOT / "Training_Images"))
    parser.add_argument("--strategy", action="append", default=None, help="Planning sequence to validate")
    parser.add_argument("--max-pairs", type=int, default=5, help="Max stereo pairs to inspect")
    parser.add_argument("--start-index", type=int, default=None, help="Only consider pairs with index >= this value")
    parser.add_argument("--require-candidates", type=int, default=9, help="Only validate scenes with this many detected candidates")
    parser.add_argument("--expected-count", type=int, default=9, help="How many objects must be planned into the bag")
    args = parser.parse_args(argv)

    strategies = args.strategy or [wet_mod.PLACE_PLANNING_SEQUENCE_NAME]
    strategies = [validate_planning_sequence_name(name) for name in strategies]

    stereo_calib, bundle, yolo, raft, rectifier = _load_models_and_calibration()
    training_dir = Path(args.images)
    pairs = dry_mod.find_stereo_pairs(training_dir)
    if args.start_index is not None:
        pairs = [pair for pair in pairs if pair.index >= int(args.start_index)]

    print(f"[BATCH] strategies={strategies}")
    print(f"[BATCH] available={list(list_planning_sequences())}")
    print(f"[BATCH] scanning up to {args.max_pairs} pair(s) with require_candidates={args.require_candidates}")

    inspected = 0
    validated_pairs = 0
    overall_failures = 0
    for pair in pairs:
        if inspected >= int(args.max_pairs):
            break
        left = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(f"[PAIR {pair.index:04d}] skipped: failed to read images")
            continue

        survey_t0 = time.perf_counter()
        survey = dry_mod.survey_from_images(
            left,
            right,
            yolo=yolo,
            raft=raft,
            rectifier=rectifier,
            stereo_calib=stereo_calib,
            bundle=bundle,
        )
        survey_elapsed = time.perf_counter() - survey_t0
        inspected += 1
        candidate_count = int(len(survey.candidates))
        print(f"\n[PAIR {pair.index:04d}] candidates={candidate_count} survey={survey_elapsed:.3f}s")
        if candidate_count != int(args.require_candidates):
            print(f"[PAIR {pair.index:04d}] skipped: require_candidates={args.require_candidates}")
            continue

        validated_pairs += 1
        for strategy in strategies:
            result = _simulate_pair(
                survey_state=survey,
                sequence_name=strategy,
                expected_count=int(args.expected_count),
            )
            result.pair_index = pair.index
            steps = ", ".join(f"{dt:.3f}" for dt in result.planning_step_times_s) or "none"
            print(
                f"[PAIR {pair.index:04d}] strategy={strategy} success={result.success} "
                f"placed={result.placed_count}/{result.expected_count} planning_total={result.total_planning_time_s:.3f}s "
                f"steps=[{steps}] reason={result.reason}"
            )
            if not result.success:
                overall_failures += 1

    print(
        f"\n[BATCH DONE] inspected={inspected} validated_pairs={validated_pairs} failures={overall_failures}"
    )
    return 0 if overall_failures == 0 and validated_pairs > 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[BATCH] interrupted")
    except Exception:
        traceback.print_exc()
        raise
