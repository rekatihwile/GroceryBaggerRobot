from __future__ import annotations

"""Interactive capstone packing demo with a joint 2.5D packing planner."""

from pathlib import Path
import argparse
import json as _json
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.gripper.gripper_geometry_config import DEFAULT_GRIPPER_GEOMETRY
from config.place import DEFAULT_PLACE, load_place_scene
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    output_dir_for_images,
    save_figure_bundle,
)

# VS Code IDE defaults.
# Copy/paste your image directory here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\Training_Images"
IDE_DEFAULT_IMAGES_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
TRAINING_IMAGES_DIR = Path(IDE_DEFAULT_IMAGES_DIR_STR) if IDE_DEFAULT_IMAGES_DIR_STR else (_REPO_ROOT / "Training_Images")

# Optional explicit output override. None uses paper_figure_sources.
IDE_DEFAULT_SAVE_OUTPUT_DIR_STR: str | None = None

STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
YOLO_WEIGHTS_PATH = _REPO_ROOT / "yolo_weights/full_data.pt"
YOLO_FALLBACK_PATH = _REPO_ROOT / "yolo_weights/validate_V2.pt"
RAFT_ROOT = _REPO_ROOT / "RAFT-Stereo"
RAFT_CKPT_PATH = _REPO_ROOT / "RAFT-Stereo/models/raftstereo-middlebury.pth"

YOLO_CONF = 0.35
YOLO_IMGSZ = 640
PAD_X_MM = float(DEFAULT_PLACE.PAD_X_MM)
PAD_Y_MM = float(DEFAULT_PLACE.PAD_Y_MM)
PAD_Z_MM = float(DEFAULT_PLACE.PAD_Z_MM)

IMAGES_ALREADY_RECTIFIED = True
MAX_PTS_DISP = 3000

_PLACE_SCENE = load_place_scene(DEFAULT_PLACE, verbose=False)
BAG_CENTER_XY_MM = list(_PLACE_SCENE.get("center_xy_mm", [190.0, 270.0]))
BAG_WIDTH_MM = float(_PLACE_SCENE.get("width_mm", 290.0))
BAG_DEPTH_MM = float(_PLACE_SCENE.get("depth_mm", 175.0))
BAG_HEIGHT_MM = float(DEFAULT_PLACE.PLACE_BAG_LOCAL_HEIGHT_MM)
BAG_SURFACE_Z_MM = float(_PLACE_SCENE.get("surface_z_mm", 0.0))

USE_CUDA = True
USE_HALF = True

PLANNER_LAYER_ACCEPT_SCORE = 0.10
SUPPORT_RATIO_THRESHOLD = 0.72
MAX_OVERHANG_RATIO = 0.28

SAVE_OUTPUT_DIR = Path(IDE_DEFAULT_SAVE_OUTPUT_DIR_STR) if IDE_DEFAULT_SAVE_OUTPUT_DIR_STR else (_REPO_ROOT / "paper_figure_sources/global/interactive_packing")
DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "interactive_packing_demo_debug"
ENABLE_GRIPPER_COLLISION_CHECK = False
GRIPPER_FINGER_LENGTH_MM = float(DEFAULT_GRIPPER_GEOMETRY.FINGER_LENGTH_MM)
GRIPPER_FINGER_WIDTH_MM = float(DEFAULT_GRIPPER_GEOMETRY.FINGER_WIDTH_MM)
GRIPPER_FINGER_HEIGHT_MM = float(DEFAULT_GRIPPER_GEOMETRY.FINGER_DEPTH_MM)
GRIPPER_PALM_WIDTH_MM = float(DEFAULT_GRIPPER_GEOMETRY.PALM_WIDTH_MM)
GRIPPER_PALM_HEIGHT_MM = float(DEFAULT_GRIPPER_GEOMETRY.PALM_HEIGHT_MM)
GRIPPER_PALM_THICKNESS_MM = 24.0
SIM_PICK_HOVER_CLEARANCE_MM = 55.0
SIM_TRAVEL_CLEARANCE_MM = 70.0
SIM_RELEASE_OPENING_EXTRA_MM = 30.0
SIM_CLOSED_OPENING_EXTRA_MM = 8.0
SIM_PHASE_FRAMES = {
    "hover_pick": 1,
    "descend_pick": 1,
    "attach": 1,
    "lift": 1,
    "move_bag": 1,
    "descend_place": 1,
    "open_release": 1,
    "detach": 1,
    "retract": 1,
}

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planning.aabb_utils import AxisAlignedBox3D, aabb_from_object_candidate, make_aabb_from_min_max
from planning.bag_local_3d_aabb_planner import (
    CandidatePlacement2p5D,
    FutureItemSpec,
    PlacementPlan3D,
    PlannerWeights,
    aabbs_intersect_3d,
    plan_bag_local_aabb_placement,
)
from vision.object_geometry import build_object_candidate
from vision.pick_candidate_builder import CandidateDebug, SurveyState, colorize_disparity
from vision.pick_phi_resolver import resolve_pick_phi
from vision.pick_xy_resolver import apply_xy_blend
from vision.pick_z_resolver import resolve_robust_object_z
from vision.pointcloud import cam_points_to_robot_xyz, masked_disparity_to_pointcloud
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLODetection, YOLOSegmenter

_PALETTE = [
    "#2e8bcb", "#e8882a", "#2dc96e", "#d83f4a",
    "#8854cc", "#e8c025", "#4abbc9", "#e06090",
]
_BG = "#1a1a2e"
_PANEL_BG = "#0d0d1a"

_GROCERY_SPEC_PATH = _REPO_ROOT / "config" / "grocery_spec.json"
try:
    with _GROCERY_SPEC_PATH.open("r", encoding="utf-8") as _f:
        GROCERY_SPEC: dict[str, dict[str, Any]] = {k: v for k, v in _json.load(_f).items() if not k.startswith("_")}
    print(f"[SPEC] loaded {len(GROCERY_SPEC)} class entries from {_GROCERY_SPEC_PATH.name}")
except Exception as _e:
    GROCERY_SPEC = {}
    print(f"[SPEC] warn: could not load grocery_spec.json: {_e}")


@dataclass
class DebugOptions:
    debug_pngs: bool = False
    debug_summary_only: bool = False

    @property
    def enabled(self) -> bool:
        return self.debug_pngs or self.debug_summary_only


@dataclass
class SimulationOptions:
    simulate_place: bool = False
    save_frames: bool = False
    step_delay_s: float = 0.03


@dataclass
class ObjectInfo:
    det_index: int
    class_name: str
    confidence: float
    color: str
    det: YOLODetection
    points_cam: np.ndarray
    points_robot: np.ndarray
    point_colors_rgb: np.ndarray
    cand_dbg: CandidateDebug
    raw_source_box: AxisAlignedBox3D
    padded_source_box: AxisAlignedBox3D
    object_props: dict[str, float | bool]


@dataclass
class PlacedObject:
    info: ObjectInfo
    target_xy: np.ndarray
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D
    object_i: int
    target_z_mm: float
    yaw_deg: float
    orientation_label: str
    placement_score: float
    support_ratio: float
    layer_z_mm: float
    future_placeable_count: int
    future_total_count: int
    stranded_labels: tuple[str, ...]
    top_clip_margin_mm: float
    future_feasibility_used: bool
    planner_notes: str


@dataclass
class PlacementComputation:
    selected_info: ObjectInfo
    target_xy: np.ndarray
    target_z_mm: float
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D
    yaw_deg: float
    orientation_label: str
    score: float
    support_ratio: float
    layer_z_mm: float
    future_placeable_count: int
    future_total_count: int
    stranded_labels: tuple[str, ...]
    top_clip_margin_mm: float
    future_feasibility_used: bool
    object_candidates_evaluated: int
    object_valid_candidates: int
    total_candidates_evaluated: int
    planning_time_s: float
    planner_notes: str
    debug_candidates: list[CandidatePlacement2p5D] = field(default_factory=list)


@dataclass
class PreviewState:
    moving_info: ObjectInfo | None = None
    moving_raw_box: AxisAlignedBox3D | None = None
    moving_padded_box: AxisAlignedBox3D | None = None
    gripper_boxes: tuple[AxisAlignedBox3D, ...] = ()
    gripper_open: bool = False
    swept_volume: AxisAlignedBox3D | None = None
    status_text: str = ""


@dataclass
class DemoState:
    objects: list[ObjectInfo]
    placed: list[PlacedObject] = field(default_factory=list)
    next_target: CandidateDebug | None = None
    next_plan: PlacementComputation | None = None


def _default_object_props(class_name: str) -> dict[str, float | bool]:
    raw = GROCERY_SPEC.get(class_name, {})
    return {
        "weight": float(raw.get("weight", 0.1)),
        "fragility": float(raw.get("fragility", 0.3)),
        "compliance": float(raw.get("compliance", 0.3)),
        "pin_slot": bool(raw.get("pin_slot", False)),
    }


def _bag_origin_xy_mm() -> np.ndarray:
    return np.array(
        [
            float(BAG_CENTER_XY_MM[0] - BAG_WIDTH_MM / 2.0),
            float(BAG_CENTER_XY_MM[1] - BAG_DEPTH_MM / 2.0),
        ],
        dtype=np.float64,
    )


def _bag_size_xyz_mm() -> np.ndarray:
    return np.array([float(BAG_WIDTH_MM), float(BAG_DEPTH_MM), float(BAG_HEIGHT_MM)], dtype=np.float64)


def _bag_local_to_robot_box(box_local: AxisAlignedBox3D) -> AxisAlignedBox3D:
    delta = np.array([*_bag_origin_xy_mm(), float(BAG_SURFACE_Z_MM)], dtype=np.float64)
    return make_aabb_from_min_max(box_local.min_xyz_mm + delta, box_local.max_xyz_mm + delta, label=box_local.label)


def _robot_box_to_bag_local(box_robot: AxisAlignedBox3D) -> AxisAlignedBox3D:
    delta = np.array([*_bag_origin_xy_mm(), float(BAG_SURFACE_Z_MM)], dtype=np.float64)
    return make_aabb_from_min_max(box_robot.min_xyz_mm - delta, box_robot.max_xyz_mm - delta, label=box_robot.label)


def _make_padded_box_from_raw_box(raw_box: AxisAlignedBox3D) -> AxisAlignedBox3D:
    return make_aabb_from_min_max(
        raw_box.min_xyz_mm - np.array([PAD_X_MM, PAD_Y_MM, 0.0], dtype=np.float64),
        raw_box.max_xyz_mm + np.array([PAD_X_MM, PAD_Y_MM, 0.0], dtype=np.float64),
        label=f"{raw_box.label}_padded",
    )


def _make_object_box_at_pose(
    reference_box: AxisAlignedBox3D,
    *,
    center_xy_mm: np.ndarray,
    min_z_mm: float,
    label: str | None = None,
) -> AxisAlignedBox3D:
    center_xy = np.asarray(center_xy_mm, dtype=np.float64).reshape(2)
    size = np.asarray(reference_box.size_xyz_mm, dtype=np.float64).reshape(3)
    min_xyz = np.array([center_xy[0] - 0.5 * size[0], center_xy[1] - 0.5 * size[1], float(min_z_mm)], dtype=np.float64)
    max_xyz = min_xyz + size
    return make_aabb_from_min_max(min_xyz, max_xyz, label=reference_box.label if label is None else label)


def _union_boxes(boxes: list[AxisAlignedBox3D] | tuple[AxisAlignedBox3D, ...], *, label: str) -> AxisAlignedBox3D:
    if not boxes:
        raise ValueError("boxes is required")
    min_xyz = np.min(np.stack([b.min_xyz_mm for b in boxes], axis=0), axis=0)
    max_xyz = np.max(np.stack([b.max_xyz_mm for b in boxes], axis=0), axis=0)
    return make_aabb_from_min_max(min_xyz, max_xyz, label=label)


def _gripper_center_z_for_object_box(box: AxisAlignedBox3D, *, clearance_mm: float = 0.0) -> float:
    grip_depth = min(float(box.size_xyz_mm[2]) * 0.25, 18.0)
    return float(box.max_xyz_mm[2] - grip_depth + 0.5 * GRIPPER_FINGER_HEIGHT_MM + clearance_mm)


def _opening_width_mm_for_object_box(box: AxisAlignedBox3D, yaw_deg: float, *, extra_mm: float) -> float:
    close_axis = 1 if abs(float(yaw_deg) % 180.0 - 90.0) <= 1e-3 else 0
    object_span = float(box.size_xyz_mm[close_axis])
    return max(22.0, object_span + float(extra_mm))


def _make_gripper_boxes_at_pose(
    center_xyz_mm: np.ndarray,
    *,
    yaw_deg: float,
    opening_width_mm: float,
    label_prefix: str = "gripper",
) -> tuple[AxisAlignedBox3D, AxisAlignedBox3D, AxisAlignedBox3D]:
    center = np.asarray(center_xyz_mm, dtype=np.float64).reshape(3)
    spread = max(18.0, float(opening_width_mm))
    is_yaw_90 = abs(float(yaw_deg) % 180.0 - 90.0) <= 1e-3

    if is_yaw_90:
        finger_size = np.array([GRIPPER_FINGER_LENGTH_MM, GRIPPER_FINGER_WIDTH_MM, GRIPPER_FINGER_HEIGHT_MM], dtype=np.float64)
        left_center = center + np.array([0.0, -0.5 * (spread + GRIPPER_FINGER_WIDTH_MM), 0.0], dtype=np.float64)
        right_center = center + np.array([0.0, 0.5 * (spread + GRIPPER_FINGER_WIDTH_MM), 0.0], dtype=np.float64)
        palm_size = np.array([GRIPPER_PALM_WIDTH_MM, spread + 2.0 * GRIPPER_FINGER_WIDTH_MM + 16.0, GRIPPER_PALM_HEIGHT_MM], dtype=np.float64)
        palm_center = center + np.array([-0.5 * (GRIPPER_FINGER_LENGTH_MM + GRIPPER_PALM_WIDTH_MM) + 8.0, 0.0, 0.5 * (GRIPPER_PALM_HEIGHT_MM - GRIPPER_FINGER_HEIGHT_MM)], dtype=np.float64)
    else:
        finger_size = np.array([GRIPPER_FINGER_WIDTH_MM, GRIPPER_FINGER_LENGTH_MM, GRIPPER_FINGER_HEIGHT_MM], dtype=np.float64)
        left_center = center + np.array([-0.5 * (spread + GRIPPER_FINGER_WIDTH_MM), 0.0, 0.0], dtype=np.float64)
        right_center = center + np.array([0.5 * (spread + GRIPPER_FINGER_WIDTH_MM), 0.0, 0.0], dtype=np.float64)
        palm_size = np.array([spread + 2.0 * GRIPPER_FINGER_WIDTH_MM + 16.0, GRIPPER_PALM_WIDTH_MM, GRIPPER_PALM_HEIGHT_MM], dtype=np.float64)
        palm_center = center + np.array([0.0, -0.5 * (GRIPPER_FINGER_LENGTH_MM + GRIPPER_PALM_WIDTH_MM) + 8.0, 0.5 * (GRIPPER_PALM_HEIGHT_MM - GRIPPER_FINGER_HEIGHT_MM)], dtype=np.float64)

    left_box = make_aabb_from_min_max(left_center - 0.5 * finger_size, left_center + 0.5 * finger_size, label=f"{label_prefix}_finger_l")
    right_box = make_aabb_from_min_max(right_center - 0.5 * finger_size, right_center + 0.5 * finger_size, label=f"{label_prefix}_finger_r")
    palm_box = make_aabb_from_min_max(palm_center - 0.5 * palm_size, palm_center + 0.5 * palm_size, label=f"{label_prefix}_palm")
    return left_box, right_box, palm_box


def _make_vertical_swept_volume(
    start_box: AxisAlignedBox3D,
    end_box: AxisAlignedBox3D,
    *,
    label: str = "swept_volume",
) -> AxisAlignedBox3D:
    min_xyz = np.minimum(start_box.min_xyz_mm, end_box.min_xyz_mm)
    max_xyz = np.maximum(start_box.max_xyz_mm, end_box.max_xyz_mm)
    return make_aabb_from_min_max(min_xyz, max_xyz, label=label)


def _draw_gripper_3d(ax, gripper_boxes: tuple[AxisAlignedBox3D, ...], *, is_open: bool) -> None:
    face = "#7ad7f0" if is_open else "#f0bc5f"
    edge = "#c8f6ff" if is_open else "#ffe3a2"
    for box in gripper_boxes:
        _draw_box_3d(ax, box, edge, alpha_edge=0.95, alpha_face=0.28, ls="-")
        fc = Poly3DCollection(_box_faces(_corners(box)), alpha=0.22)
        fc.set_facecolor(face)
        ax.add_collection3d(fc)


def _placed_boxes_local(state: DemoState) -> tuple[list[AxisAlignedBox3D], list[AxisAlignedBox3D]]:
    return (
        [_robot_box_to_bag_local(p.raw_box) for p in state.placed],
        [_robot_box_to_bag_local(p.padded_box) for p in state.placed],
    )


def _support_props_by_label(state: DemoState) -> dict[str, dict[str, Any]]:
    return {p.raw_box.label: p.info.object_props for p in state.placed}


def _future_item_spec(info: ObjectInfo) -> FutureItemSpec:
    return FutureItemSpec(
        label=info.class_name,
        raw_size_xyz_mm=info.raw_source_box.size_xyz_mm.copy(),
        padding_min_xyz_mm=info.raw_source_box.min_xyz_mm - info.padded_source_box.min_xyz_mm,
        padding_max_xyz_mm=info.padded_source_box.max_xyz_mm - info.raw_source_box.max_xyz_mm,
        object_props=dict(info.object_props),
    )


def _joint_selection_bonus(info: ObjectInfo, plan: PlacementPlan3D) -> float:
    bag_size = _bag_size_xyz_mm()
    area_fraction = float(info.raw_source_box.size_xyz_mm[0] * info.raw_source_box.size_xyz_mm[1]) / max(1.0, float(bag_size[0] * bag_size[1]))
    height_fraction = float(info.raw_source_box.size_xyz_mm[2]) / max(1.0, float(bag_size[2]))
    fragility = float(info.object_props["fragility"])
    compliance = float(info.object_props["compliance"])
    weight = float(info.object_props["weight"])
    z_norm = float(plan.layer_z_mm) / max(1.0, float(bag_size[2]))
    if plan.layer_z_mm <= 1e-6:
        return 12.0 * area_fraction - 1.5 * fragility - 0.8 * compliance - 0.4 * height_fraction + 0.6 * weight
    return 1.5 * (fragility + compliance) * z_norm - 0.4 * weight * z_norm


def _candidate_sort_key(result: PlacementComputation) -> tuple[float, float, float, int]:
    return (
        result.score,
        result.support_ratio,
        -result.layer_z_mm,
        -result.selected_info.det_index,
    )


def _choose_next_object_and_place_target(state: DemoState, debug: DebugOptions) -> tuple[ObjectInfo | None, PlacementComputation | None]:
    remaining = [obj for obj in state.objects if obj.det_index not in {p.info.det_index for p in state.placed}]
    if not remaining:
        return None, None

    placed_raw_local, placed_padded_local = _placed_boxes_local(state)
    support_props = _support_props_by_label(state)
    planner_weights = PlannerWeights()
    plan_start = time.perf_counter()
    results: list[PlacementComputation] = []
    total_candidates = 0

    for info in remaining:
        future_specs = [_future_item_spec(other) for other in remaining if other.det_index != info.det_index]
        plan: PlacementPlan3D = plan_bag_local_aabb_placement(
            item_label=info.class_name,
            raw_size_xyz_mm=info.raw_source_box.size_xyz_mm,
            padding_min_xyz_mm=info.raw_source_box.min_xyz_mm - info.padded_source_box.min_xyz_mm,
            padding_max_xyz_mm=info.padded_source_box.max_xyz_mm - info.raw_source_box.max_xyz_mm,
            bag_size_xyz_mm=_bag_size_xyz_mm(),
            placed_raw_boxes_local=placed_raw_local,
            placed_padded_boxes_local=placed_padded_local,
            object_properties=info.object_props,
            support_properties_by_label=support_props,
            weights=planner_weights,
            lower_layer_score_threshold=PLANNER_LAYER_ACCEPT_SCORE,
            support_min_overlap_ratio=SUPPORT_RATIO_THRESHOLD,
            max_overhang_ratio=MAX_OVERHANG_RATIO,
            remaining_item_specs=future_specs,
            collect_debug_attempts=debug.enabled,
        )
        total_candidates += int(plan.candidates_evaluated)
        final_score = float(plan.score) + _joint_selection_bonus(info, plan)
        raw_box_robot = _bag_local_to_robot_box(plan.raw_box_local)
        padded_box_robot = _bag_local_to_robot_box(plan.padded_box_local)
        results.append(
            PlacementComputation(
                selected_info=info,
                target_xy=raw_box_robot.center_xyz_mm[:2].copy(),
                target_z_mm=float(raw_box_robot.min_xyz_mm[2]),
                raw_box=raw_box_robot,
                padded_box=padded_box_robot,
                yaw_deg=float(plan.yaw_deg),
                orientation_label=plan.orientation_label,
                score=final_score,
                support_ratio=float(plan.support_ratio),
                layer_z_mm=float(plan.layer_z_mm),
                future_placeable_count=int(plan.future_placeable_count),
                future_total_count=int(plan.future_total_count),
                stranded_labels=tuple(plan.stranded_labels),
                top_clip_margin_mm=float(plan.top_clip_margin_mm),
                future_feasibility_used=bool(plan.future_feasibility_used),
                object_candidates_evaluated=int(plan.candidates_evaluated),
                object_valid_candidates=int(plan.valid_candidates),
                total_candidates_evaluated=0,
                planning_time_s=0.0,
                planner_notes=f"{plan.notes}; joint_bonus={final_score - float(plan.score):.3f}",
                debug_candidates=list(plan.attempts),
            )
        )

    lowest_layers = sorted({round(r.layer_z_mm, 6) for r in results})
    shortlist: list[PlacementComputation] = results
    for layer_z in lowest_layers:
        layer_results = [r for r in results if abs(r.layer_z_mm - layer_z) <= 1e-6]
        if layer_z <= 1e-6 or any(r.score >= PLANNER_LAYER_ACCEPT_SCORE for r in layer_results):
            shortlist = layer_results
            break

    if shortlist and abs(shortlist[0].layer_z_mm) <= 1e-6:
        max_area = max(float(r.selected_info.raw_source_box.size_xyz_mm[0] * r.selected_info.raw_source_box.size_xyz_mm[1]) for r in shortlist)
        foundation_shortlist = [
            r
            for r in shortlist
            if float(r.selected_info.raw_source_box.size_xyz_mm[0] * r.selected_info.raw_source_box.size_xyz_mm[1]) >= 0.85 * max_area
        ]
        if foundation_shortlist:
            shortlist = foundation_shortlist

    best = max(shortlist, key=_candidate_sort_key)
    elapsed = time.perf_counter() - plan_start
    best.total_candidates_evaluated = total_candidates
    best.planning_time_s = elapsed
    print(
        f"[PLAN] candidates={total_candidates} chosen={best.selected_info.class_name} "
        f"z=[{best.target_z_mm:.1f},{best.raw_box.max_xyz_mm[2]:.1f}] "
        f"top_margin={best.top_clip_margin_mm:.1f}mm "
        f"future_ok={best.future_placeable_count}/{best.future_total_count} "
        f"yaw={best.yaw_deg:.0f} score={best.score:.3f} "
        f"support={best.support_ratio:.2f} elapsed={elapsed:.3f}s"
    )
    if best.stranded_labels:
        print(f"[PLAN] reject/penalize summary for {best.selected_info.class_name}: stranded={list(best.stranded_labels)}")
    return best.selected_info, best


def _refresh_next_choice(state: DemoState, debug: DebugOptions) -> None:
    info, plan = _choose_next_object_and_place_target(state, debug)
    state.next_target = info.cand_dbg if info is not None else None
    state.next_plan = plan


def _corners(box: AxisAlignedBox3D) -> np.ndarray:
    mn, mx = box.min_xyz_mm, box.max_xyz_mm
    return np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def _box_faces(c: np.ndarray):
    return [[c[i] for i in face] for face in [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]]


def _draw_box_3d(ax, box: AxisAlignedBox3D, color: str, alpha_edge: float = 0.9, alpha_face: float = 0.10, ls: str = "-"):
    c = _corners(box)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], [c[a, 2], c[b, 2]], color=color, alpha=alpha_edge, linewidth=1.1, linestyle=ls)
    if alpha_face > 0:
        fc = Poly3DCollection(_box_faces(c), alpha=alpha_face)
        fc.set_facecolor(color)
        ax.add_collection3d(fc)


def _ensure_debug_dir(debug: DebugOptions) -> Path | None:
    if not debug.enabled:
        return None
    DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_OUTPUT_DIR


def _save_summary_debug_png(*, pair_index: int, step_i: int, plan: PlacementComputation, placed_before: list[PlacedObject], out_dir: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor=_BG)
    ax.set_facecolor(_PANEL_BG)
    bag_size = _bag_size_xyz_mm()
    ax.add_patch(mpatches.Rectangle((0, 0), bag_size[0], bag_size[1], edgecolor="white", facecolor="none", lw=1.5))

    for placed in placed_before:
        raw_local = _robot_box_to_bag_local(placed.raw_box)
        ax.add_patch(
            mpatches.Rectangle(
                raw_local.min_xyz_mm[:2],
                raw_local.size_xyz_mm[0],
                raw_local.size_xyz_mm[1],
                facecolor=placed.info.color,
                edgecolor="white",
                alpha=0.35,
                lw=1.0,
            )
        )

    for cand in plan.debug_candidates:
        box = cand.raw_box_local
        ax.add_patch(
            mpatches.Rectangle(
                box.min_xyz_mm[:2],
                box.size_xyz_mm[0],
                box.size_xyz_mm[1],
                facecolor="#54c46a" if cand.accepted and abs(cand.score - plan.score) <= 1e-6 and abs(cand.yaw_deg - plan.yaw_deg) <= 1e-6 and abs(cand.layer_z_mm - plan.layer_z_mm) <= 1e-6 and cand.orientation_label == plan.orientation_label else "#e25c5c",
                edgecolor="none",
                alpha=0.18 if cand.accepted else 0.10,
            )
        )

    chosen_local = _robot_box_to_bag_local(plan.raw_box)
    ax.add_patch(
        mpatches.Rectangle(
            chosen_local.min_xyz_mm[:2],
            chosen_local.size_xyz_mm[0],
            chosen_local.size_xyz_mm[1],
            facecolor="none",
            edgecolor="#7dff7d",
            lw=2.0,
        )
    )
    ax.text(
        chosen_local.center_xyz_mm[0],
        chosen_local.center_xyz_mm[1],
        f"{plan.selected_info.class_name}\nscore={plan.score:.2f}\nyaw={plan.yaw_deg:.0f}\nz={plan.layer_z_mm:.0f}",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="#1b2d1b", ec="#7dff7d", alpha=0.85),
    )

    ax.set_title(f"Pair {pair_index:04d} step {step_i} planner summary", color="#aaddff", fontsize=10)
    ax.set_xlim(-10, bag_size[0] + 10)
    ax.set_ylim(-10, bag_size[1] + 10)
    ax.set_xlabel("Bag-local X (mm)")
    ax.set_ylabel("Bag-local Y (mm)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(out_dir / f"pair_{pair_index:04d}_step_{step_i:02d}_summary.png"), dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_attempt_debug_pngs(*, pair_index: int, step_i: int, plan: PlacementComputation, placed_before: list[PlacedObject], out_dir: Path) -> None:
    bag_size = _bag_size_xyz_mm()
    for i, cand in enumerate(plan.debug_candidates, start=1):
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
        for ax in (ax_xy, ax_xz):
            ax.set_facecolor(_PANEL_BG)

        ax_xy.add_patch(mpatches.Rectangle((0, 0), bag_size[0], bag_size[1], edgecolor="white", facecolor="none", lw=1.5))
        ax_xz.add_patch(mpatches.Rectangle((0, -5), bag_size[0], 5, edgecolor="white", facecolor="#334", lw=1.0))

        for placed in placed_before:
            raw_local = _robot_box_to_bag_local(placed.raw_box)
            ax_xy.add_patch(mpatches.Rectangle(raw_local.min_xyz_mm[:2], raw_local.size_xyz_mm[0], raw_local.size_xyz_mm[1], facecolor=placed.info.color, edgecolor="white", alpha=0.30, lw=1.0))
            ax_xz.add_patch(mpatches.Rectangle((raw_local.min_xyz_mm[0], raw_local.min_xyz_mm[2]), raw_local.size_xyz_mm[0], raw_local.size_xyz_mm[2], facecolor=placed.info.color, edgecolor="white", alpha=0.30, lw=1.0))

        color = "#7dff7d" if cand.accepted else "#ff6b6b"
        ax_xy.add_patch(mpatches.Rectangle(cand.raw_box_local.min_xyz_mm[:2], cand.raw_box_local.size_xyz_mm[0], cand.raw_box_local.size_xyz_mm[1], facecolor=color, edgecolor="white", alpha=0.55, lw=1.2))
        ax_xz.add_patch(mpatches.Rectangle((cand.raw_box_local.min_xyz_mm[0], cand.raw_box_local.min_xyz_mm[2]), cand.raw_box_local.size_xyz_mm[0], cand.raw_box_local.size_xyz_mm[2], facecolor=color, edgecolor="white", alpha=0.55, lw=1.2))
        ax_xy.set_title(
            f"Step {step_i} {plan.selected_info.class_name} cand {i}\n"
            f"{'ACCEPT' if cand.accepted else 'REJECT'} score={cand.score:.2f} yaw={cand.yaw_deg:.0f}\n"
            f"{'; '.join(cand.reasons)}",
            color="#aaddff",
            fontsize=8,
        )
        ax_xz.set_title("Side view", color="#aaddff", fontsize=8)
        ax_xy.set_xlim(-10, bag_size[0] + 10)
        ax_xy.set_ylim(-10, bag_size[1] + 10)
        ax_xz.set_xlim(-10, bag_size[0] + 10)
        ax_xz.set_ylim(-25, bag_size[2] + 25)
        fig.tight_layout()
        fig.savefig(str(out_dir / f"pair_{pair_index:04d}_step_{step_i:02d}_cand_{i:03d}.png"), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def _emit_debug_outputs(*, pair_index: int, step_i: int, plan: PlacementComputation, placed_before: list[PlacedObject], debug: DebugOptions) -> None:
    out_dir = _ensure_debug_dir(debug)
    if out_dir is None:
        return
    _save_summary_debug_png(pair_index=pair_index, step_i=step_i, plan=plan, placed_before=placed_before, out_dir=out_dir)
    if debug.debug_pngs:
        _save_attempt_debug_pngs(pair_index=pair_index, step_i=step_i, plan=plan, placed_before=placed_before, out_dir=out_dir)


def _assert_no_3d_intersections(placed: list[PlacedObject]) -> None:
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            if aabbs_intersect_3d(a.raw_box, b.raw_box):
                raise AssertionError(f"raw-AABB intersection between {a.info.class_name} and {b.info.class_name}")


def _assert_no_padded_intersections(placed: list[PlacedObject]) -> None:
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            if aabbs_intersect_3d(a.padded_box, b.padded_box):
                raise AssertionError(f"padded-AABB intersection between {a.info.class_name} and {b.info.class_name}")


def _assert_support_constraints(placed: list[PlacedObject]) -> None:
    for placed_obj in placed:
        if placed_obj.layer_z_mm <= 1e-6:
            continue
        if placed_obj.support_ratio + 1e-6 < SUPPORT_RATIO_THRESHOLD:
            raise AssertionError(
                f"{placed_obj.info.class_name} support ratio {placed_obj.support_ratio:.3f} "
                f"below threshold {SUPPORT_RATIO_THRESHOLD:.3f}"
            )


def _assert_no_top_clipping(placed: list[PlacedObject]) -> None:
    for placed_obj in placed:
        if float(placed_obj.raw_box.max_xyz_mm[2]) > float(BAG_HEIGHT_MM) + 1e-6:
            raise AssertionError(
                f"{placed_obj.info.class_name} clips bag top: "
                f"z_max={placed_obj.raw_box.max_xyz_mm[2]:.3f} > {BAG_HEIGHT_MM:.3f}"
            )


def _placed_object_from_plan(plan: PlacementComputation, *, object_i: int) -> PlacedObject:
    return PlacedObject(
        info=plan.selected_info,
        target_xy=plan.target_xy,
        raw_box=plan.raw_box,
        padded_box=plan.padded_box,
        object_i=object_i,
        target_z_mm=plan.target_z_mm,
        yaw_deg=plan.yaw_deg,
        orientation_label=plan.orientation_label,
        placement_score=plan.score,
        support_ratio=plan.support_ratio,
        layer_z_mm=plan.layer_z_mm,
        future_placeable_count=plan.future_placeable_count,
        future_total_count=plan.future_total_count,
        stranded_labels=plan.stranded_labels,
        top_clip_margin_mm=plan.top_clip_margin_mm,
        future_feasibility_used=plan.future_feasibility_used,
        planner_notes=plan.planner_notes,
    )


def _boxes_collide_any(candidates: list[AxisAlignedBox3D] | tuple[AxisAlignedBox3D, ...], placed_boxes: list[AxisAlignedBox3D]) -> bool:
    return any(aabbs_intersect_3d(candidate, placed_box) for candidate in candidates for placed_box in placed_boxes)


def _evaluate_gripper_collision_reason(plan: PlacementComputation, placed: list[PlacedObject]) -> str | None:
    if not ENABLE_GRIPPER_COLLISION_CHECK:
        return None

    placed_raw_boxes = [p.raw_box for p in placed]
    placed_padded_boxes = [p.padded_box for p in placed]
    if _boxes_collide_any([plan.raw_box], placed_raw_boxes) or _boxes_collide_any([plan.padded_box], placed_padded_boxes):
        return "gripper_collision"

    open_width = _opening_width_mm_for_object_box(plan.raw_box, plan.yaw_deg, extra_mm=SIM_RELEASE_OPENING_EXTRA_MM)
    release_center = np.array(
        [
            float(plan.target_xy[0]),
            float(plan.target_xy[1]),
            _gripper_center_z_for_object_box(plan.raw_box, clearance_mm=0.0),
        ],
        dtype=np.float64,
    )
    release_gripper_boxes = _make_gripper_boxes_at_pose(release_center, yaw_deg=plan.yaw_deg, opening_width_mm=open_width, label_prefix="release")
    if _boxes_collide_any(release_gripper_boxes, placed_raw_boxes):
        return "gripper_collision"

    hover_center = release_center.copy()
    hover_center[2] += SIM_PICK_HOVER_CLEARANCE_MM
    hover_union = _union_boxes(list(_make_gripper_boxes_at_pose(hover_center, yaw_deg=plan.yaw_deg, opening_width_mm=open_width, label_prefix="hover")), label="hover_union")
    release_union = _union_boxes(list(release_gripper_boxes), label="release_union")
    swept_box = _make_vertical_swept_volume(hover_union, release_union, label="release_swept")
    if _boxes_collide_any([swept_box], placed_raw_boxes):
        return "swept_volume_collision"
    return None


class _NullRobot:
    def fk(self):
        return 0.0, 0.0, 0.0, 0.0

    def check_cartesian_pose_safe(self, x, y, z):
        return True, "no_robot"

    @property
    def cfg(self):
        return None


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

    def best_observation(self):
        return self._obs


def _sample_point_colors_rgb(rect_left_bgr: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    if len(uv_px) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    h, w = rect_left_bgr.shape[:2]
    xs = np.clip(np.rint(uv_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv_px[:, 1]).astype(np.int32), 0, h - 1)
    bgr = rect_left_bgr[ys, xs].astype(np.float32)
    rgb = bgr[:, ::-1] / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _build_objects(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    rectifier: StereoRectifier,
    stereo_calib: dict,
    bundle: dict,
) -> list[ObjectInfo]:
    rect_l, rect_r = (left_bgr, right_bgr) if IMAGES_ALREADY_RECTIFIED else rectifier.rectify(left_bgr, right_bgr)
    dets = yolo.segment(rect_l)
    print(f"[PIPELINE] {len(dets)} detections")
    disp = raft.predict_disparity(rect_l, rect_r, color="BGR")

    objects: list[ObjectInfo] = []
    for idx, det in enumerate(dets, start=1):
        try:
            pts_cam, uv = masked_disparity_to_pointcloud(det.mask, disp, stereo_calib)
        except Exception as exc:
            print(f"  [PC {idx}] {exc}")
            continue
        if len(pts_cam) < 300:
            print(f"  [PC {idx}] too few pts")
            continue

        try:
            pts_robot = cam_points_to_robot_xyz(pts_cam, bundle)
            point_colors_rgb = _sample_point_colors_rgb(rect_l, uv)
        except Exception as exc:
            print(f"  [XF {idx}] {exc}")
            continue

        try:
            cand = build_object_candidate(
                index=idx,
                yolo_det=det,
                points_cam=pts_cam,
                point_uv_px=uv,
                robot=_NullRobot(),
                bundle=bundle,
                stereo_tags={},
                frame_i=0,
            )
            z_res = resolve_robust_object_z(pts_robot, pts_cam, candidate=cand)
            cand.object_robot_xyz_raw[2] = z_res.robust_top_z_mm
            cand.object_robot_xyz_corrected[2] = z_res.robust_top_z_mm
            cand.z_debug = z_res

            mock_obs = _MockObservation(frame_i=0, detection=det)
            mock_track = _MockTrack(track_id=idx, class_name=det.class_name, _obs=mock_obs)
            dbg = CandidateDebug(
                candidate=cand,
                track=mock_track,
                best_frame_i=0,
                best_detection=det,
                disparity=disp,
                disparity_overlay=colorize_disparity(disp),
                left_overlay=rect_l.copy(),
                point_count=len(pts_cam),
                point_uv_px=uv,
                points_cam=pts_cam,
                stereo_xy_mm=np.asarray(cand.object_robot_xyz_raw[:2]).reshape(2),
                overhead_xy_mm=None,
                blend_weight_overhead=0.0,
                xy_disagreement_mm=None,
            )
            resolve_pick_phi(dbg, None, bundle)
            apply_xy_blend(dbg, bundle)

            raw_source_box = aabb_from_object_candidate(cand, default_label=det.class_name)
            padded_source_box = _make_padded_box_from_raw_box(raw_source_box)
            props = _default_object_props(det.class_name)
        except Exception as exc:
            print(f"  [CAND {idx}] {exc}")
            traceback.print_exc()
            continue

        objects.append(
            ObjectInfo(
                det_index=idx,
                class_name=det.class_name,
                confidence=det.confidence,
                color=_PALETTE[(idx - 1) % len(_PALETTE)],
                det=det,
                points_cam=pts_cam,
                points_robot=pts_robot,
                point_colors_rgb=point_colors_rgb,
                cand_dbg=dbg,
                raw_source_box=raw_source_box,
                padded_source_box=padded_source_box,
                object_props=props,
            )
        )
        print(f"  [OK {idx}] {det.class_name} pts={len(pts_cam)}")
    return objects


def _simulate_full_plan(objects: list[ObjectInfo], *, pair_index: int, debug: DebugOptions) -> DemoState:
    state = DemoState(objects=objects)
    _refresh_next_choice(state, debug)
    while state.next_plan is not None and state.next_target is not None:
        plan = state.next_plan
        _emit_debug_outputs(pair_index=pair_index, step_i=len(state.placed) + 1, plan=plan, placed_before=state.placed, debug=debug)
        placed = _placed_object_from_plan(plan, object_i=len(state.placed) + 1)
        state.placed.append(placed)
        _assert_no_3d_intersections(state.placed)
        _assert_no_padded_intersections(state.placed)
        _assert_support_constraints(state.placed)
        _assert_no_top_clipping(state.placed)
        _refresh_next_choice(state, debug)
    return state


class PackingDemo:
    def __init__(self, state: DemoState, pair_index: int, rect_left: np.ndarray, debug: DebugOptions, simulation: SimulationOptions):
        self.state = state
        self.pair_index = pair_index
        self.rect_left = rect_left
        self.debug = debug
        self.simulation = simulation
        self.preview: PreviewState | None = None
        self._sim_frame_counter = 0
        _refresh_next_choice(self.state, self.debug)

        self.fig = plt.figure(figsize=(18, 9), facecolor=_BG)
        gs = gridspec.GridSpec(1, 2, figure=self.fig, wspace=0.04)
        self.ax_plat = self.fig.add_subplot(gs[0, 0])
        self.ax_bag = self.fig.add_subplot(gs[0, 1], projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def _on_key(self, event):
        if event.key in ("right",):
            self._step_forward()
        elif event.key in ("left",):
            self._step_backward()
        elif event.key in ("r", "R"):
            self._reset()
        elif event.key in ("s", "S"):
            self._save()
        elif event.key in ("q", "Q", "escape"):
            plt.close("all")

    def _step_forward(self):
        plan = self.state.next_plan
        if plan is None:
            print("[DEMO] all objects placed")
            return

        try:
            _emit_debug_outputs(pair_index=self.pair_index, step_i=len(self.state.placed) + 1, plan=plan, placed_before=self.state.placed, debug=self.debug)
            collision_reason = _evaluate_gripper_collision_reason(plan, self.state.placed)
            if collision_reason is not None:
                print(f"[DEMO] placement rejected: {collision_reason}")
                return
            if self.simulation.simulate_place:
                self._animate_place_sequence(plan)
            placed = _placed_object_from_plan(plan, object_i=len(self.state.placed) + 1)
            self.state.placed.append(placed)
            _assert_no_3d_intersections(self.state.placed)
            _assert_no_padded_intersections(self.state.placed)
            _assert_support_constraints(self.state.placed)
            _assert_no_top_clipping(self.state.placed)
        except Exception as exc:
            self.preview = None
            print(f"[DEMO] placement failed: {exc}")
            traceback.print_exc()
            if self.state.placed and self.state.placed[-1].object_i == len(self.state.placed):
                pass
            return

        print(
            f"[DEMO] placed #{placed.object_i} {placed.info.class_name} "
            f"at ({placed.target_xy[0]:.1f},{placed.target_xy[1]:.1f},{placed.target_z_mm:.1f}) "
            f"yaw={placed.yaw_deg:.0f} score={placed.placement_score:.3f} "
            f"top_margin={placed.top_clip_margin_mm:.1f}mm "
            f"future_ok={placed.future_placeable_count}/{placed.future_total_count}"
        )
        _refresh_next_choice(self.state, self.debug)
        self._render()

    def _save_sim_frame(self, *, step_i: int, phase_name: str) -> None:
        if not self.simulation.save_frames:
            return
        self._sim_frame_counter += 1
        out = SAVE_OUTPUT_DIR / (
            f"interactive_packing_demo_sim_{self.pair_index:04d}_"
            f"step{step_i:02d}_{self._sim_frame_counter:04d}_{phase_name}.png"
        )
        self.fig.savefig(str(out), dpi=110, bbox_inches="tight", facecolor=self.fig.get_facecolor())

    def _present_preview_frame(self, *, step_i: int, phase_name: str) -> None:
        self._render()
        self._save_sim_frame(step_i=step_i, phase_name=phase_name)
        plt.pause(max(0.001, float(self.simulation.step_delay_s)))

    def _animate_place_sequence(self, plan: PlacementComputation) -> None:
        step_i = len(self.state.placed) + 1
        pick_xy = plan.selected_info.raw_source_box.center_xyz_mm[:2].copy()
        pick_min_z = float(plan.selected_info.raw_source_box.min_xyz_mm[2])
        pick_raw_box = _make_object_box_at_pose(plan.raw_box, center_xy_mm=pick_xy, min_z_mm=pick_min_z, label=f"{plan.selected_info.class_name}_pick_raw")

        max_placed_top = max([float(p.raw_box.max_xyz_mm[2]) for p in self.state.placed], default=float(BAG_SURFACE_Z_MM))
        travel_object_min_z = max(
            pick_raw_box.max_xyz_mm[2] + SIM_TRAVEL_CLEARANCE_MM - plan.raw_box.size_xyz_mm[2],
            float(plan.raw_box.max_xyz_mm[2]) + SIM_TRAVEL_CLEARANCE_MM - plan.raw_box.size_xyz_mm[2],
            max_placed_top + SIM_TRAVEL_CLEARANCE_MM,
            float(BAG_SURFACE_Z_MM + BAG_HEIGHT_MM + 15.0),
        )

        open_width = _opening_width_mm_for_object_box(plan.raw_box, plan.yaw_deg, extra_mm=SIM_RELEASE_OPENING_EXTRA_MM)
        closed_width = _opening_width_mm_for_object_box(plan.raw_box, plan.yaw_deg, extra_mm=SIM_CLOSED_OPENING_EXTRA_MM)

        pick_grasp_z = _gripper_center_z_for_object_box(pick_raw_box, clearance_mm=0.0)
        pick_hover_z = pick_grasp_z + SIM_PICK_HOVER_CLEARANCE_MM
        travel_grasp_z = _gripper_center_z_for_object_box(
            _make_object_box_at_pose(plan.raw_box, center_xy_mm=pick_xy, min_z_mm=travel_object_min_z, label="travel_raw"),
            clearance_mm=0.0,
        )
        release_grasp_z = _gripper_center_z_for_object_box(plan.raw_box, clearance_mm=0.0)
        release_hover_z = max(travel_grasp_z, release_grasp_z + SIM_PICK_HOVER_CLEARANCE_MM)

        release_gripper_boxes = _make_gripper_boxes_at_pose(
            np.array([plan.target_xy[0], plan.target_xy[1], release_grasp_z], dtype=np.float64),
            yaw_deg=plan.yaw_deg,
            opening_width_mm=open_width,
            label_prefix="release_open",
        )
        hover_gripper_boxes = _make_gripper_boxes_at_pose(
            np.array([plan.target_xy[0], plan.target_xy[1], release_hover_z], dtype=np.float64),
            yaw_deg=plan.yaw_deg,
            opening_width_mm=open_width,
            label_prefix="hover_open",
        )
        swept_volume = _make_vertical_swept_volume(
            _union_boxes(list(hover_gripper_boxes), label="hover_gripper_union"),
            _union_boxes(list(release_gripper_boxes), label="release_gripper_union"),
            label="release_swept_volume",
        )

        def _lerp(a: float, b: float, t: float) -> float:
            return (1.0 - t) * float(a) + t * float(b)

        def _lerp_xy(a_xy: np.ndarray, b_xy: np.ndarray, t: float) -> np.ndarray:
            return (1.0 - t) * np.asarray(a_xy, dtype=np.float64) + t * np.asarray(b_xy, dtype=np.float64)

        def _phase_steps(name: str) -> np.ndarray:
            count = max(1, int(SIM_PHASE_FRAMES.get(name, 1)))
            if count == 1:
                return np.array([1.0], dtype=np.float64)
            return np.linspace(0.0, 1.0, count, dtype=np.float64)

        def _emit(
            *,
            phase_name: str,
            object_xy: np.ndarray,
            object_min_z: float,
            gripper_xy: np.ndarray,
            gripper_z: float,
            gripper_open: bool,
            opening_width_mm: float,
            show_swept: bool,
            status_text: str,
        ) -> None:
            moving_raw_box = _make_object_box_at_pose(plan.raw_box, center_xy_mm=object_xy, min_z_mm=object_min_z, label=f"{plan.selected_info.class_name}_preview_raw")
            moving_padded_box = _make_object_box_at_pose(plan.padded_box, center_xy_mm=object_xy, min_z_mm=object_min_z, label=f"{plan.selected_info.class_name}_preview_padded")
            gripper_boxes = _make_gripper_boxes_at_pose(
                np.array([gripper_xy[0], gripper_xy[1], gripper_z], dtype=np.float64),
                yaw_deg=plan.yaw_deg,
                opening_width_mm=opening_width_mm,
                label_prefix=f"preview_{phase_name}",
            )
            self.preview = PreviewState(
                moving_info=plan.selected_info,
                moving_raw_box=moving_raw_box,
                moving_padded_box=moving_padded_box,
                gripper_boxes=gripper_boxes,
                gripper_open=gripper_open,
                swept_volume=swept_volume if show_swept else None,
                status_text=status_text,
            )
            self._present_preview_frame(step_i=step_i, phase_name=phase_name)

        for t in _phase_steps("hover_pick"):
            _emit(
                phase_name="hover_pick",
                object_xy=pick_xy,
                object_min_z=pick_min_z,
                gripper_xy=pick_xy,
                gripper_z=pick_hover_z,
                gripper_open=True,
                opening_width_mm=open_width,
                show_swept=False,
                status_text="1. Hover above pick",
            )
        for t in _phase_steps("descend_pick"):
            _emit(
                phase_name="descend_pick",
                object_xy=pick_xy,
                object_min_z=pick_min_z,
                gripper_xy=pick_xy,
                gripper_z=_lerp(pick_hover_z, pick_grasp_z, t),
                gripper_open=True,
                opening_width_mm=open_width,
                show_swept=False,
                status_text="2. Descend to pick",
            )
        for t in _phase_steps("attach"):
            _emit(
                phase_name="attach",
                object_xy=pick_xy,
                object_min_z=pick_min_z,
                gripper_xy=pick_xy,
                gripper_z=pick_grasp_z,
                gripper_open=t < 0.5,
                opening_width_mm=_lerp(open_width, closed_width, t),
                show_swept=False,
                status_text="3. Attach object to gripper",
            )
        for t in _phase_steps("lift"):
            object_min_z = _lerp(pick_min_z, travel_object_min_z, t)
            _emit(
                phase_name="lift",
                object_xy=pick_xy,
                object_min_z=object_min_z,
                gripper_xy=pick_xy,
                gripper_z=_gripper_center_z_for_object_box(
                    _make_object_box_at_pose(plan.raw_box, center_xy_mm=pick_xy, min_z_mm=object_min_z, label="lift_raw"),
                    clearance_mm=0.0,
                ),
                gripper_open=False,
                opening_width_mm=closed_width,
                show_swept=False,
                status_text="4. Lift to travel height",
            )
        for t in _phase_steps("move_bag"):
            object_xy = _lerp_xy(pick_xy, plan.target_xy, t)
            _emit(
                phase_name="move_bag",
                object_xy=object_xy,
                object_min_z=travel_object_min_z,
                gripper_xy=object_xy,
                gripper_z=travel_grasp_z,
                gripper_open=False,
                opening_width_mm=closed_width,
                show_swept=False,
                status_text="5. Move above bag target",
            )
        for t in _phase_steps("descend_place"):
            object_min_z = _lerp(travel_object_min_z, plan.target_z_mm, t)
            object_xy = np.asarray(plan.target_xy, dtype=np.float64)
            _emit(
                phase_name="descend_place",
                object_xy=object_xy,
                object_min_z=object_min_z,
                gripper_xy=object_xy,
                gripper_z=_lerp(release_hover_z, release_grasp_z, t),
                gripper_open=False,
                opening_width_mm=closed_width,
                show_swept=True,
                status_text="6. Descend to place target",
            )
        for t in _phase_steps("open_release"):
            _emit(
                phase_name="open_release",
                object_xy=plan.target_xy,
                object_min_z=plan.target_z_mm,
                gripper_xy=plan.target_xy,
                gripper_z=release_grasp_z,
                gripper_open=True,
                opening_width_mm=_lerp(closed_width, open_width, t),
                show_swept=True,
                status_text="7. Open gripper",
            )
        for _ in _phase_steps("detach"):
            _emit(
                phase_name="detach",
                object_xy=plan.target_xy,
                object_min_z=plan.target_z_mm,
                gripper_xy=plan.target_xy,
                gripper_z=release_grasp_z,
                gripper_open=True,
                opening_width_mm=open_width,
                show_swept=True,
                status_text="8. Detach into final placed AABB",
            )
        for t in _phase_steps("retract"):
            _emit(
                phase_name="retract",
                object_xy=plan.target_xy,
                object_min_z=plan.target_z_mm,
                gripper_xy=plan.target_xy,
                gripper_z=_lerp(release_grasp_z, release_hover_z, t),
                gripper_open=True,
                opening_width_mm=open_width,
                show_swept=True,
                status_text="9. Retract upward",
            )
        self.preview = None

    def _step_backward(self):
        if not self.state.placed:
            return
        self.preview = None
        removed = self.state.placed.pop()
        _refresh_next_choice(self.state, self.debug)
        self._render()
        print(f"[DEMO] undo: removed object #{removed.object_i} ({removed.info.class_name})")

    def _reset(self):
        self.preview = None
        self.state.placed.clear()
        _refresh_next_choice(self.state, self.debug)
        self._render()
        print("[DEMO] reset")

    def _save(self):
        stem = SAVE_OUTPUT_DIR / (
            f"interactive_packing_state_pair_{self.pair_index:04d}_step_{len(self.state.placed):02d}"
        )
        for path in save_figure_bundle(
            self.fig,
            stem,
            dpi=DEFAULT_PUBLICATION_DPI,
            facecolor=self.fig.get_facecolor(),
        ):
            print(f"[SAVE] {path}")

    def _render(self):
        for ax in (self.ax_plat, self.ax_bag):
            ax.cla()

        picked_indices = {p.info.det_index for p in self.state.placed}
        next_info = self.state.next_plan.selected_info if self.state.next_plan is not None else None

        ax = self.ax_plat
        ax.set_facecolor(_PANEL_BG)
        ax.set_aspect("equal")
        ax.set_title(
            f"Platform - pair {self.pair_index:04d} ({len(self.state.placed)}/{len(self.state.objects)} placed)\n"
            f"-> place  <- undo  R reset  S save  Q quit",
            color="#aaddff",
            fontsize=9,
            pad=6,
        )
        h, w = self.rect_left.shape[:2]
        ax.imshow(cv2.cvtColor(self.rect_left, cv2.COLOR_BGR2RGB), extent=[0, w, h, 0], alpha=0.30, aspect="auto")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        for obj in self.state.objects:
            x1, y1, x2, y2 = [int(v) for v in obj.det.bbox]
            is_picked = obj.det_index in picked_indices
            is_next = next_info is not None and obj.det_index == next_info.det_index
            alpha = 0.25 if is_picked else 0.85
            lw = 3.5 if is_next else 1.5
            ls = "-" if not is_picked else "--"
            rect = mpatches.FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=2", edgecolor=obj.color, facecolor=obj.color, linewidth=lw, linestyle=ls, alpha=0.18 if not is_picked else 0.06)
            border = mpatches.FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=2", edgecolor=obj.color, facecolor="none", linewidth=lw, linestyle=ls, alpha=alpha)
            ax.add_patch(rect)
            ax.add_patch(border)
            ax.text(x1 + 2, max(14, y1 - 4), f"#{obj.det_index} {obj.class_name} {obj.confidence:.2f}", color=obj.color, fontsize=7, fontweight="bold", alpha=0.4 if is_picked else 1.0)
            if is_picked:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = min(x2 - x1, y2 - y1) * 0.40
                ax.plot([cx - d, cx + d], [cy - d, cy + d], color="#ff4444", lw=2.5, alpha=0.8)
                ax.plot([cx - d, cx + d], [cy + d, cy - d], color="#ff4444", lw=2.5, alpha=0.8)
            if is_next:
                cx, cy = (x1 + x2) / 2, y1 - 10
                ax.text(cx, cy, "NEXT", color="#ffff00", fontsize=9, fontweight="bold", ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="#222244", ec="#ffff00", alpha=0.85))

        status = (
            f"Placed: {len(self.state.placed)} | "
            f"Next: {next_info.class_name if next_info else 'none'} | "
            f"Remaining: {len(self.state.objects) - len(self.state.placed)}"
        )
        if self.state.next_plan is not None:
            status += f" | score={self.state.next_plan.score:.2f} yaw={self.state.next_plan.yaw_deg:.0f}"
        if self.simulation.simulate_place:
            status += " | sim=on"
        ax.text(0.5, -0.04, status, transform=ax.transAxes, color="#cccccc", fontsize=8, ha="center", va="top")
        ax.tick_params(colors="#666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        ax3 = self.ax_bag
        ax3.set_facecolor(_PANEL_BG)
        title = f"Bag ({len(self.state.placed)} placed)\nsolid = raw AABB  dashed = padded AABB"
        if self.preview is not None and self.preview.status_text:
            title += f"\n{self.preview.status_text}"
        ax3.set_title(title, color="#aaddff", fontsize=9, pad=4)
        show_static_point_clouds = self.preview is None
        bx, by = BAG_CENTER_XY_MM
        bw2, bd2 = BAG_WIDTH_MM / 2, BAG_DEPTH_MM / 2
        bz0, bz1 = BAG_SURFACE_Z_MM, BAG_SURFACE_Z_MM + BAG_HEIGHT_MM
        bag_corners = np.array(
            [
                [bx - bw2, by - bd2, bz0],
                [bx + bw2, by - bd2, bz0],
                [bx + bw2, by + bd2, bz0],
                [bx - bw2, by + bd2, bz0],
                [bx - bw2, by - bd2, bz1],
                [bx + bw2, by - bd2, bz1],
                [bx + bw2, by + bd2, bz1],
                [bx - bw2, by + bd2, bz1],
            ]
        )
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax3.plot([bag_corners[a, 0], bag_corners[b, 0]], [bag_corners[a, 1], bag_corners[b, 1]], [bag_corners[a, 2], bag_corners[b, 2]], color="#3a3a5a", lw=1.2, alpha=0.9)

        all_pts_for_scale = list(bag_corners)
        for placed in self.state.placed:
            if show_static_point_clouds and len(placed.info.points_robot) > 0:
                sample_idx = None
                if len(placed.info.points_robot) <= MAX_PTS_DISP:
                    samp = placed.info.points_robot
                else:
                    sample_idx = np.random.choice(len(placed.info.points_robot), MAX_PTS_DISP, replace=False)
                    samp = placed.info.points_robot[sample_idx]
                centroid_xy = np.mean(samp[:, :2], axis=0)
                z_min = float(samp[:, 2].min())
                shift = np.array([placed.target_xy[0] - centroid_xy[0], placed.target_xy[1] - centroid_xy[1], placed.target_z_mm - z_min], dtype=np.float64)
                shifted = samp + shift
                if len(placed.info.point_colors_rgb) == len(placed.info.points_robot):
                    samp_colors = placed.info.point_colors_rgb if sample_idx is None else placed.info.point_colors_rgb[sample_idx]
                    ax3.scatter(shifted[:, 0], shifted[:, 1], shifted[:, 2], c=samp_colors, s=0.6, alpha=0.45)
            _draw_box_3d(ax3, placed.raw_box, placed.info.color, alpha_face=0.15, ls="-")
            _draw_box_3d(ax3, placed.padded_box, placed.info.color, alpha_face=0.0, ls="--", alpha_edge=0.40)
            ax3.text(
                placed.raw_box.center_xyz_mm[0],
                placed.raw_box.center_xyz_mm[1],
                placed.raw_box.max_xyz_mm[2] + 6,
                f"#{placed.object_i} {placed.info.class_name[:7]} y{placed.yaw_deg:.0f}",
                color="white",
                fontsize=6,
                ha="center",
                va="bottom",
            )
            all_pts_for_scale.extend(_corners(placed.padded_box))

        if self.preview is not None:
            if self.preview.swept_volume is not None:
                _draw_box_3d(ax3, self.preview.swept_volume, "#76e0d8", alpha_face=0.05, alpha_edge=0.28, ls=":")
                all_pts_for_scale.extend(_corners(self.preview.swept_volume))
            if self.preview.moving_raw_box is not None:
                preview_color = self.preview.moving_info.color if self.preview.moving_info is not None else "#cccccc"
                _draw_box_3d(ax3, self.preview.moving_raw_box, preview_color, alpha_face=0.22, alpha_edge=0.92, ls="-")
                all_pts_for_scale.extend(_corners(self.preview.moving_raw_box))
            if self.preview.moving_padded_box is not None:
                preview_color = self.preview.moving_info.color if self.preview.moving_info is not None else "#cccccc"
                _draw_box_3d(ax3, self.preview.moving_padded_box, preview_color, alpha_face=0.0, alpha_edge=0.45, ls="--")
                all_pts_for_scale.extend(_corners(self.preview.moving_padded_box))
            if self.preview.gripper_boxes:
                _draw_gripper_3d(ax3, self.preview.gripper_boxes, is_open=self.preview.gripper_open)
                for box in self.preview.gripper_boxes:
                    all_pts_for_scale.extend(_corners(box))

        all_pts_arr = np.array(all_pts_for_scale)
        mn = all_pts_arr.min(axis=0)
        mx = all_pts_arr.max(axis=0)
        ctr = 0.5 * (mn + mx)
        span = max((mx - mn).max() * 0.62, 160.0)
        ax3.set_xlim(ctr[0] - span, ctr[0] + span)
        ax3.set_ylim(ctr[1] - span, ctr[1] + span)
        ax3.set_zlim(-10, max(mx[2] + 50, BAG_HEIGHT_MM))
        ax3.view_init(elev=22, azim=-60)
        ax3.set_xlabel("X (mm)", color="#999", fontsize=7, labelpad=2)
        ax3.set_ylabel("Y (mm)", color="#999", fontsize=7, labelpad=2)
        ax3.set_zlabel("Z up (mm)", color="#999", fontsize=7, labelpad=2)
        ax3.tick_params(colors="#777", labelsize=6)
        for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#2a2a3a")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def _print_final_validation_summary(state: DemoState) -> None:
    print("[VALIDATION] final placement order")
    for placed in state.placed:
        print(
            f"  #{placed.object_i} {placed.info.class_name}: "
            f"z=[{placed.target_z_mm:.1f},{placed.raw_box.max_xyz_mm[2]:.1f}] "
            f"top_margin={placed.top_clip_margin_mm:.1f}mm "
            f"yaw={placed.yaw_deg:.0f} score={placed.placement_score:.3f} "
            f"support={placed.support_ratio:.2f} "
            f"future_ok={placed.future_placeable_count}/{placed.future_total_count} "
            f"stranded={list(placed.stranded_labels)} "
            f"future_used={placed.future_feasibility_used} "
            f"reason={placed.planner_notes}"
        )


def main(argv: list[str] | None = None) -> int:
    global SAVE_OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(TRAINING_IMAGES_DIR))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--validate-all", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--debug-pngs", action="store_true")
    parser.add_argument("--debug-summary-only", action="store_true")
    parser.add_argument("--simulate-place", action="store_true")
    parser.add_argument("--sim-save-frames", action="store_true")
    parser.add_argument("--sim-step-delay", type=float, default=0.03)
    args = parser.parse_args(argv)

    debug = DebugOptions(debug_pngs=bool(args.debug_pngs), debug_summary_only=bool(args.debug_summary_only))
    simulation = SimulationOptions(
        simulate_place=bool(args.simulate_place),
        save_frames=bool(args.sim_save_frames),
        step_delay_s=max(0.0, float(args.sim_step_delay)),
    )

    pairs_dir = Path(args.images)
    SAVE_OUTPUT_DIR = output_dir_for_images(pairs_dir, args.out_dir) / "interactive_packing"
    SAVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lr = re.compile(r"Stereo_Left_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    rr = re.compile(r"Stereo_Right_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    lefts, rights = {}, {}
    for p in pairs_dir.iterdir():
        m = lr.match(p.name)
        if m:
            lefts[int(m.group(1))] = p
            continue
        m = rr.match(p.name)
        if m:
            rights[int(m.group(1))] = p
    common = sorted(set(lefts) & set(rights))
    if not common:
        print(f"[ERROR] no stereo pairs in {pairs_dir}")
        return 1
    if args.index is not None:
        common = [i for i in common if i == args.index]
        if not common:
            print(f"[ERROR] pair {args.index} not found")
            return 1

    idx = common[0]
    lp, rp = lefts[idx], rights[idx]
    print(f"[INIT] pair {idx:04d} left={lp.name}")

    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)
    weights = YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.exists() else YOLO_FALLBACK_PATH
    print("[INIT] loading YOLO ...")
    yolo = YOLOSegmenter(weights_path=str(weights), device_info=device_info, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=0.50, retina_masks=True, min_mask_area_px=500)
    yolo.warmup()
    print("[INIT] loading RAFT ...")
    raft = RAFTStereoRunner(raft_root=str(RAFT_ROOT), checkpoint_path=str(RAFT_CKPT_PATH), device_info=device_info, valid_iters=16)
    raft.warmup()

    calib = {k: np.asarray(v) for k, v in np.load(str(STEREO_CALIB_PATH), allow_pickle=False).items()}
    rectifier = StereoRectifier(calib)
    if BUNDLE_PATH.exists():
        from test_calibration_bundle_live_stereo_z_pickplace import load_bundle

        bundle = load_bundle(BUNDLE_PATH)
    else:
        bundle = {}
    if not bundle:
        print("[WARN] no calibration bundle - robot-frame coordinates will be approximate")

    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    rect_l = left if IMAGES_ALREADY_RECTIFIED else rectifier.rectify(left, right)[0]

    print("[PIPELINE] running YOLO + RAFT + point cloud ...")
    objects = _build_objects(left, right, yolo=yolo, raft=raft, rectifier=rectifier, stereo_calib=calib, bundle=bundle)
    if not objects:
        print("[ERROR] no valid objects detected")
        return 1

    print(f"[DEMO] {len(objects)} object(s) ready")
    if args.validate_all:
        validated = _simulate_full_plan(objects, pair_index=idx, debug=debug)
        _assert_no_3d_intersections(validated.placed)
        _assert_no_padded_intersections(validated.placed)
        _assert_support_constraints(validated.placed)
        _assert_no_top_clipping(validated.placed)
        print(f"[VALIDATION] placed {len(validated.placed)}/{len(objects)} objects with no raw or padded AABB intersections")
        print("[VALIDATION] future feasibility pruning used")
        _print_final_validation_summary(validated)

    if args.no_gui:
        return 0

    print("  -> place next object")
    print("  <- undo last placement")
    print("  R = reset")
    print("  S = save current figure")
    print("  Q = quit")
    if simulation.simulate_place:
        print(f"  simulation preview enabled (delay={simulation.step_delay_s:.2f}s, save_frames={simulation.save_frames})")
    state = DemoState(objects=objects)
    demo = PackingDemo(state, pair_index=idx, rect_left=rect_l, debug=debug, simulation=simulation)
    demo.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
