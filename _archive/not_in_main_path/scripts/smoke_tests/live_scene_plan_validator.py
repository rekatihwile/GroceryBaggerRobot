from __future__ import annotations

"""Standalone survey -> plan -> preview validator.

Default mode is live:
1. Open overhead + stereo cameras
2. Prompt for a survey
3. Run the same survey + next-step planning seam as the wet script
4. Press right-arrow to preview/commit the next virtual place step

This stays motion-free. The robot is represented by dry FK only so we can
inspect what the runtime would do without commanding hardware.

Optional saved-pair mode exists for development and regression checks:
    python scripts/smoke_tests/live_scene_plan_validator.py --images Training_Images --index 1
"""

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import math
import sys
import time
import traceback

if "--no-gui" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from config.place import DEFAULT_PLACE, load_place_scene
from config.runtime_context import resolve_runtime_context
from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from config.robot_config import ROBOT_CONFIG
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from hardware.robot import Robot
from planning.autonomous_planning_sequences import (
    PlanningSelection,
    PlanningSequenceContext,
    PlanningRuntime,
    SequenceTargetPlan,
    select_candidate_for_state,
    validate_planning_sequence_name,
)
from planning.bag_local_3d_aabb_planner import aabbs_intersect_3d
import scripts.autonomous_missed_pick_recovery as wet_mod
from scripts.autonomous_best_candidate import BestCandidateConfig
import scripts.capstone.autonomous_system_wrapper as demo_mod
import scripts.capstone.interactive_packing_demo as packing_demo_mod
import scripts.dry_run_autonomous as dry_mod
from test_calibration_bundle_live_stereo_z_pickplace import load_bundle, load_stereo_calibration
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_survey_pipeline import load_vision, run_survey
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


FIG_BG = "#171728"
PANEL_BG = "#0e0f1a"
PREVIEW_WINDOW = "Live Survey Preview"
SAVE_OUTPUT_DIR = Path(__file__).resolve().parent
CHECK_GRIPPER_COLLISION = False


@dataclass
class PlanPreview:
    selection: PlanningSelection
    selected_dbg: CandidateDebug | None
    selected_scene: demo_mod.SceneObject | None
    target_plan: SequenceTargetPlan | None
    step: demo_mod.PlacementStep | None
    placed_occupancy: object | None
    planning_time_s: float
    status: str
    raw_reason: str


@dataclass
class ValidationResult:
    success: bool
    placed_count: int
    expected_count: int
    reason: str
    planning_step_times_s: list[float]


def _best_candidate_config_for_workspace(workspace, *, use_robot_checks: bool) -> BestCandidateConfig:
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
        require_min_robotframe_centroid_z=bool(use_robot_checks and wet_mod.BEST_REQUIRE_MIN_ROBOTFRAME_CENTROID_Z),
        min_robotframe_centroid_z_mm=wet_mod.BEST_MIN_ROBOTFRAME_CENTROID_Z_MM if use_robot_checks and wet_mod.BEST_REQUIRE_MIN_ROBOTFRAME_CENTROID_Z else None,
        use_robot_reach_check=bool(use_robot_checks and wet_mod.BEST_USE_ROBOT_REACH_CHECK),
        robot_reach_margin_mm=wet_mod.BEST_ROBOT_REACH_MARGIN_MM if use_robot_checks and wet_mod.BEST_USE_ROBOT_REACH_CHECK else None,
        require_soft_pose_safe=bool(use_robot_checks and wet_mod.BEST_REQUIRE_SOFT_POSE_SAFE),
        soft_pose_check_z_mm=wet_mod.BEST_SOFT_POSE_CHECK_Z_MM if use_robot_checks and wet_mod.BEST_REQUIRE_SOFT_POSE_SAFE else None,
        center_gate_enabled=bool(workspace.center_gate_enabled),
        max_image_center_norm_radius=float(workspace.max_image_center_norm_radius),
        cluster_gate_enabled=bool(workspace.cluster_gate_enabled),
        max_cluster_distance_mm=float(workspace.max_cluster_distance_mm),
        reject_placed_overlap=bool(workspace.reject_placed_overlap),
        placed_overlap_margin_mm=float(workspace.placed_overlap_margin_mm),
        min_volume_mm3=float(workspace.min_volume_mm3),
        max_volume_mm3=float(workspace.max_volume_cm3) * 1000.0,
    )


def _build_dry_fk_robot() -> tuple[Robot, object]:
    robot = Robot(config=ROBOT_CONFIG, connect=False)
    q_survey = demo_mod._survey_joint_pose(robot)
    robot.q_est = q_survey
    return robot, q_survey


def _normalize_failure_reason(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if "top_clip" in text:
        return "height_out_of_bag"
    if "vertical_descent_blocked" in text or "swept" in text:
        return "swept_volume_collision"
    if "neighbor_collision" in text or "gripper_collision" in text:
        return "gripper_collision"
    if "bag_wall_clearance" in text:
        return "gripper_collision"
    if "fit_or_footprint_outside_bag" in text:
        return "no_xy_footprint_without_collision"
    if "waiting_for_reference_box" in text:
        return "need_reference_object"
    if "selector_rejected" in text:
        return "no_visible_pick_candidate"
    return text


def _choose_display_disparity(survey: SurveyState, planned_dbg: CandidateDebug | None) -> np.ndarray | None:
    if planned_dbg is not None:
        return np.asarray(planned_dbg.disparity_overlay, dtype=np.uint8)
    if survey.candidates:
        return np.asarray(survey.candidates[0].disparity_overlay, dtype=np.uint8)
    return None


def _choose_display_rect_left(survey: SurveyState) -> np.ndarray | None:
    if survey.burst_frames:
        return np.asarray(survey.burst_frames[0].left_rect, dtype=np.uint8)
    if survey.candidates:
        return np.asarray(survey.candidates[0].left_overlay, dtype=np.uint8)
    return None


def _draw_raw_stereo_pair(ax, left_bgr: np.ndarray | None, right_bgr: np.ndarray | None) -> None:
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Survey Stereo Pair", color="#aaddff", fontsize=10, pad=6)
    if left_bgr is None or right_bgr is None:
        ax.text(0.5, 0.5, "No stereo pair", transform=ax.transAxes, color="white", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    both = np.concatenate([left_bgr, right_bgr], axis=1)
    ax.imshow(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
    h, w = left_bgr.shape[:2]
    ax.axvline(w - 0.5, color="#ffffff", linewidth=1.0, alpha=0.6)
    ax.text(8, 18, "Left", color="#ffffff", fontsize=9, fontweight="bold")
    ax.text(w + 8, 18, "Right", color="#ffffff", fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_yolo_mask_panel(ax, rect_left_bgr: np.ndarray | None, scene_objects: list[demo_mod.SceneObject]) -> None:
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_title("YOLO Mask", color="#aaddff", fontsize=10, pad=6)
    if rect_left_bgr is None:
        ax.text(0.5, 0.5, "No survey image", transform=ax.transAxes, color="white", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    rgb = cv2.cvtColor(rect_left_bgr, cv2.COLOR_BGR2RGB)
    mask_canvas = np.zeros_like(rgb)
    for obj in scene_objects:
        det = obj.candidate_debug.best_detection
        if det.mask is None:
            continue
        mask = np.asarray(det.mask, dtype=bool)
        color_hex = obj.color
        color_rgb = np.array([int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)], dtype=np.uint8)
        mask_canvas[mask] = color_rgb
    composite = np.where(mask_canvas.any(axis=2, keepdims=True), mask_canvas, (0.18 * rgb).astype(np.uint8))
    ax.imshow(composite)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_unbagged_pointcloud(ax, scene_objects: list[demo_mod.SceneObject], selected_scene: demo_mod.SceneObject | None) -> None:
    ax.cla()
    demo_mod._style_3d(ax, "Point Cloud | Unbagged Groceries")
    scale_pts: list[np.ndarray] = []
    for obj in scene_objects:
        pts = np.asarray(obj.points_robot, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue
        if len(pts) <= demo_mod.MAX_PTS_DISP:
            sample = pts
            colors = obj.point_colors_rgb
        else:
            idx = np.linspace(0, len(pts) - 1, demo_mod.MAX_PTS_DISP, dtype=np.int32)
            sample = pts[idx]
            colors = obj.point_colors_rgb[idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=0.8, alpha=0.70 if obj is selected_scene else 0.45)
        demo_mod._draw_box_3d(ax, obj.raw_box, obj.color, alpha_face=0.10 if obj is selected_scene else 0.04, alpha_edge=0.95 if obj is selected_scene else 0.40, ls="-")
        scale_pts.append(obj.raw_box.min_xyz_mm.copy())
        scale_pts.append(obj.raw_box.max_xyz_mm.copy())
    demo_mod._draw_platform_bounds(ax)
    ax.view_init(elev=24, azim=-61)
    if scale_pts:
        demo_mod._set_equal_aspect(ax, np.vstack(scale_pts))


class LiveScenePlanValidator:
    def __init__(
        self,
        *,
        surface_zone: dict[str, object],
        render_surface_zone: dict[str, object],
        selector_config: BestCandidateConfig,
        sequence_name: str,
        bundle: dict,
        stereo_calib: dict,
        rectifier: StereoRectifier,
        yolo,
        raft,
        detector,
        robot: Robot,
        q_survey,
        overhead: SimpleOverheadCamera | None,
        stereo: SimpleStereoCamera | None,
        live_mode: bool,
        save_frames: bool,
        animate_step_delay_s: float,
    ) -> None:
        self.surface_zone = surface_zone
        self.render_surface_zone = render_surface_zone
        self.selector_config = selector_config
        self.sequence_name = validate_planning_sequence_name(sequence_name)
        self.bundle = bundle
        self.stereo_calib = stereo_calib
        self.rectifier = rectifier
        self.yolo = yolo
        self.raft = raft
        self.detector = detector
        self.robot = robot
        self.q_survey = q_survey
        self.overhead = overhead
        self.stereo = stereo
        self.live_mode = bool(live_mode)
        self.save_frames = bool(save_frames)
        self.animate_step_delay_s = float(animate_step_delay_s)

        self.survey: SurveyState | None = None
        self.scene_objects: list[demo_mod.SceneObject] = []
        self.remaining_scene_objects: list[demo_mod.SceneObject] = []
        self.scene_by_det_index: dict[int, demo_mod.SceneObject] = {}
        self.planner_info_by_det_index: dict[int, packing_demo_mod.ObjectInfo] = {}
        self.placed_occupancies: list = []
        self.placed_scene_objects: list[demo_mod.PlacedSceneObject] = []
        self.preview: demo_mod.PreviewState | None = None
        self.plan_preview: PlanPreview | None = None
        self.raw_left_bgr: np.ndarray | None = None
        self.raw_right_bgr: np.ndarray | None = None
        self.display_rect_left_bgr: np.ndarray | None = None
        self.status_lines = [
            f"Sequence: {self.sequence_name}",
            "Press s to survey, right-arrow to preview/commit next place, v to validate all, r to reset, q to quit",
        ]

        self.fig = plt.figure(figsize=(18, 10), facecolor=FIG_BG)
        gs = gridspec.GridSpec(
            2,
            3,
            figure=self.fig,
            width_ratios=[1.25, 1.05, 1.15],
            height_ratios=[1.0, 1.0],
            hspace=0.12,
            wspace=0.08,
        )
        self.ax_raw = self.fig.add_subplot(gs[0, 0:2])
        self.ax_pc = self.fig.add_subplot(gs[0, 2], projection="3d")
        self.ax_disp = self.fig.add_subplot(gs[1, 0])
        self.ax_mask = self.fig.add_subplot(gs[1, 1])
        self.ax_bag = self.fig.add_subplot(gs[1, 2], projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()

    def close(self) -> None:
        if self.overhead is not None:
            self.overhead.release()
        if self.stereo is not None:
            self.stereo.release()
        cv2.destroyAllWindows()

    def _object_i(self) -> int:
        return len(self.placed_occupancies) + 1

    def _planning_runtime(self) -> PlanningRuntime:
        return wet_mod._planning_runtime()

    def _planning_context(self) -> PlanningSequenceContext:
        base_xy = np.asarray(self.surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
        place_phi = float(self.surface_zone["default_phi_deg"])
        return wet_mod._planning_sequence_context(
            object_i=self._object_i(),
            robot=self.robot,
            placed_boxes=self.placed_occupancies,
            config=self.selector_config,
            surface_zone=self.surface_zone,
            base_xy=base_xy,
            base_phi_deg=place_phi,
            target_limit=max(1, len(self.remaining_scene_objects)),
        )

    def _state_for_remaining(self) -> SurveyState | None:
        if self.survey is None:
            return None
        allowed = {int(obj.det_index) for obj in self.remaining_scene_objects}
        filtered = [dbg for dbg in self.survey.candidates if int(getattr(dbg.candidate, "index", -1)) in allowed]
        return replace(self.survey, candidates=filtered, selected_index=0)

    def _top_margin_mm(self, raw_box) -> float:
        bag_top_z = float(self.render_surface_zone["surface_z_mm"]) + float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", wet_mod.PLACE_BAG_LOCAL_HEIGHT_MM))
        return float(bag_top_z - raw_box.max_xyz_mm[2])

    def _placed_center_z_mm(self, target_plan: SequenceTargetPlan, raw_box, *, object_i: int) -> float:
        floor_z = float(self.render_surface_zone["surface_z_mm"])
        object_height_mm = float(raw_box.size_xyz_mm[2])
        planner_layer_z = getattr(target_plan, "planner_layer_z_mm", None)
        if planner_layer_z is not None:
            return float(floor_z + float(planner_layer_z) + 0.5 * object_height_mm)
        if object_i <= 2:
            return float(floor_z + 0.5 * object_height_mm)
        below_box = self.placed_occupancies[object_i - 3]
        return float(below_box.raw_box.max_xyz_mm[2] + 0.5 * object_height_mm)

    def _compute_next_plan(self) -> PlanPreview:
        state = self._state_for_remaining()
        if state is None or not state.candidates:
            selection = PlanningSelection(result=type("EmptyResult", (), {"selected": None})(), overlay={}, target_cache={})
            return PlanPreview(selection, None, None, None, None, None, 0.0, "no_visible_pick_candidate", "no visible candidates")

        ctx = self._planning_context()
        t0 = time.perf_counter()
        selection = select_candidate_for_state(self.sequence_name, state, ctx=ctx, runtime=self._planning_runtime())
        planning_time_s = time.perf_counter() - t0
        selected_dbg = selection.result.selected
        if selected_dbg is None:
            overlay_reasons = [str(entry.reason) for entry in selection.overlay.values() if entry.reason]
            raw_reason = overlay_reasons[0] if overlay_reasons else "no_selection"
            status = _normalize_failure_reason(raw_reason)
            return PlanPreview(selection, None, None, None, None, None, planning_time_s, status, raw_reason)

        selected_index = state.candidates.index(selected_dbg)
        target_plan = selection.target_cache.get(selected_index)
        if target_plan is None:
            target_plan = wet_mod._compute_candidate_place_target_for_sequence(
                selected_dbg,
                state=state,
                object_i=self._object_i(),
                robot=self.robot,
                placed_boxes=self.placed_occupancies,
                config=self.selector_config,
                surface_zone=self.surface_zone,
                base_xy=np.asarray(self.surface_zone["center_xy_mm"], dtype=np.float64).reshape(2),
                base_phi_deg=float(self.surface_zone["default_phi_deg"]),
                target_limit=max(1, len(self.remaining_scene_objects)),
            )

        raw_reason = str(getattr(target_plan, "reason", "ok"))
        if not bool(getattr(target_plan, "can_place", True)):
            status = _normalize_failure_reason(raw_reason)
            selected_scene = self.scene_by_det_index.get(int(getattr(selected_dbg.candidate, "index", -1)))
            return PlanPreview(selection, selected_dbg, selected_scene, target_plan, None, None, planning_time_s, status, raw_reason)

        det_index = int(getattr(selected_dbg.candidate, "index", -1))
        selected_scene = self.scene_by_det_index.get(det_index)
        if selected_scene is None:
            return PlanPreview(selection, selected_dbg, None, target_plan, None, None, planning_time_s, "unknown", "selected scene object missing")

        raw_box = wet_mod._aabb_from_object_candidate_quiet(selected_dbg.candidate, default_label=f"object{self._object_i()}")
        center_z = self._placed_center_z_mm(target_plan, raw_box, object_i=self._object_i())
        placed_occ = wet_mod._placed_occupancy_from_plan(
            center_xyz_mm=np.array([float(target_plan.target_xy_mm[0]), float(target_plan.target_xy_mm[1]), float(center_z)], dtype=np.float64),
            size_xyz_mm=np.asarray(raw_box.size_xyz_mm, dtype=np.float64),
            label=f"object{self._object_i()}_placed",
        )

        servo_angle_deg = demo_mod._servo_angle_deg_for_candidate(selected_scene.candidate_debug)
        pick_yaw_deg = float(getattr(selected_dbg.candidate, "pick_phi_deg", 0.0))
        pick_ok, pick_reason, pick_hold_box, pick_swept_box = demo_mod._audit_pick_clearance(
            selected_scene,
            self.remaining_scene_objects,
            yaw_deg=pick_yaw_deg,
            servo_angle_deg=servo_angle_deg,
        )
        place_ok, place_reason, place_hold_box, place_swept_box = demo_mod._audit_place_clearance(
            placed_occ.raw_box,
            [box.raw_box for box in self.placed_occupancies],
            self.render_surface_zone,
            yaw_deg=float(target_plan.target_phi_deg),
            servo_angle_deg=servo_angle_deg,
        )
        hold_length_mm, hold_width_mm, _half_x, _half_y = demo_mod._held_envelope_xy_extents_mm(
            placed_occ.raw_box,
            yaw_deg=float(target_plan.target_phi_deg),
            servo_angle_deg=servo_angle_deg,
        )
        planner_info = self.planner_info_by_det_index[det_index]
        planner_score = getattr(target_plan, "planner_score", None)
        if planner_score is None:
            planner_score = getattr(target_plan, "fit_clearance_mm", 0.0)
        planner_layer_z = getattr(target_plan, "planner_layer_z_mm", None)
        if planner_layer_z is None:
            planner_layer_z = max(0.0, placed_occ.raw_box.min_xyz_mm[2] - float(self.render_surface_zone["surface_z_mm"]))
        placed_obj = packing_demo_mod.PlacedObject(
            info=planner_info,
            target_xy=np.asarray(target_plan.target_xy_mm, dtype=np.float64).reshape(2).copy(),
            raw_box=placed_occ.raw_box,
            padded_box=placed_occ.padded_box,
            object_i=self._object_i(),
            target_z_mm=float(placed_occ.raw_box.min_xyz_mm[2]),
            yaw_deg=float(target_plan.target_phi_deg),
            orientation_label=str(getattr(target_plan, "planner_yaw_deg", target_plan.target_phi_deg)),
            placement_score=float(planner_score),
            support_ratio=1.0,
            layer_z_mm=float(planner_layer_z),
            future_placeable_count=int(getattr(target_plan, "future_placeable_count", 0) or 0),
            future_total_count=int(getattr(target_plan, "future_total_count", 0) or 0),
            stranded_labels=tuple(),
            top_clip_margin_mm=self._top_margin_mm(placed_occ.raw_box),
            future_feasibility_used=bool(getattr(target_plan, "future_placeable_count", None) is not None),
            planner_notes=raw_reason,
        )
        step = demo_mod.PlacementStep(
            scene_object=selected_scene,
            placed=placed_obj,
            object_i=self._object_i(),
            servo_angle_deg=servo_angle_deg,
            hold_length_mm=hold_length_mm,
            hold_width_mm=hold_width_mm,
            pick_clearance_ok=pick_ok,
            pick_clearance_reason=pick_reason,
            pick_hold_box=pick_hold_box,
            pick_swept_box=pick_swept_box,
            place_clearance_ok=place_ok,
            place_clearance_reason=place_reason,
            place_hold_box=place_hold_box,
            place_swept_box=place_swept_box,
        )
        if CHECK_GRIPPER_COLLISION and not pick_ok:
            status = _normalize_failure_reason(pick_reason)
            raw_reason = pick_reason
        elif CHECK_GRIPPER_COLLISION and not place_ok:
            status = _normalize_failure_reason(place_reason)
            raw_reason = place_reason
        else:
            status = "ok"
        return PlanPreview(selection, selected_dbg, selected_scene, target_plan, step, placed_occ, planning_time_s, status, raw_reason)

    def _commit_current_step(self) -> None:
        if self.plan_preview is None or self.plan_preview.step is None or self.plan_preview.placed_occupancy is None:
            return
        step = self.plan_preview.step
        placed_occ = self.plan_preview.placed_occupancy
        self.placed_occupancies.append(placed_occ)
        self.remaining_scene_objects = [obj for obj in self.remaining_scene_objects if obj is not step.scene_object]
        self.placed_scene_objects.append(
            demo_mod.PlacedSceneObject(
                scene_object=step.scene_object,
                placed_raw_box=step.placed.raw_box,
                placed_padded_box=step.placed.padded_box,
                placed_points_xyz=demo_mod._transform_points_for_box(step.scene_object, step.placed.raw_box),
                object_i=step.object_i,
            )
        )
    def _build_live_preview_frame(self) -> np.ndarray:
        overhead_frame = None
        left = None
        right = None
        if self.overhead is not None:
            ok_oh, overhead_frame = self.overhead.read()
            if not ok_oh:
                overhead_frame = None
        if self.stereo is not None:
            ok_st, _full, left, right = self.stereo.read_pair()
            if not ok_st:
                left = None
                right = None
        if left is not None:
            self.raw_left_bgr = left.copy()
        if right is not None:
            self.raw_right_bgr = right.copy()

        if overhead_frame is None and left is None and right is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        top = overhead_frame if overhead_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        stereo_pair = np.concatenate(
            [
                left if left is not None else np.zeros((top.shape[0], top.shape[1] // 2, 3), dtype=np.uint8),
                right if right is not None else np.zeros((top.shape[0], top.shape[1] // 2, 3), dtype=np.uint8),
            ],
            axis=1,
        )
        top_resized = cv2.resize(top, (1280, 360))
        stereo_resized = cv2.resize(stereo_pair, (1280, 360))
        frame = np.concatenate([top_resized, stereo_resized], axis=0)
        cv2.putText(frame, "Overhead", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Stereo Left | Stereo Right", (12, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press s to survey, q to quit", (12, 706), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (160, 255, 160), 2, cv2.LINE_AA)
        return frame

    def prompt_for_live_survey(self) -> bool:
        if not self.live_mode:
            return True
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW, 1280, 760)
        try:
            while True:
                frame = self._build_live_preview_frame()
                cv2.imshow(PREVIEW_WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return False
                if key in (ord("s"), 13):
                    return True
        finally:
            cv2.destroyWindow(PREVIEW_WINDOW)

    def _apply_survey(self, survey: SurveyState) -> None:
        self.survey = survey
        self.scene_objects = demo_mod._build_scene_objects(survey, self.display_rect_left_bgr, self.bundle)
        self.scene_by_det_index = {int(obj.det_index): obj for obj in self.scene_objects}
        planner_infos = demo_mod._planner_objects_from_scene(self.scene_objects)
        self.planner_info_by_det_index = {int(info.det_index): info for info in planner_infos}
        self.remaining_scene_objects = list(self.scene_objects)
        self.placed_occupancies = []
        self.placed_scene_objects = []
        self.preview = None
        self.plan_preview = self._compute_next_plan()
        self.status_lines = self._status_lines_from_plan(prefix="Survey ready")

    def survey_live(self) -> bool:
        if self.overhead is None or self.stereo is None:
            return False
        ok_st, _full, left_raw, right_raw = self.stereo.read_pair()
        if ok_st:
            self.raw_left_bgr = None if left_raw is None else left_raw.copy()
            self.raw_right_bgr = None if right_raw is None else right_raw.copy()
        survey = run_survey(
            self.overhead,
            self.stereo,
            self.detector,
            self.stereo_calib,
            self.rectifier,
            self.yolo,
            self.raft,
            self.robot,
            self.bundle,
        )
        self.display_rect_left_bgr = _choose_display_rect_left(survey)
        if self.raw_left_bgr is None and survey.burst_frames:
            self.raw_left_bgr = survey.burst_frames[0].left_rect.copy()
            self.raw_right_bgr = survey.burst_frames[0].right_rect.copy()
        self._apply_survey(survey)
        return True

    def survey_saved_pair(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> None:
        self.raw_left_bgr = left_bgr.copy()
        self.raw_right_bgr = right_bgr.copy()
        survey = dry_mod.survey_from_images(
            left_bgr,
            right_bgr,
            yolo=self.yolo,
            raft=self.raft,
            rectifier=self.rectifier,
            stereo_calib=self.stereo_calib,
            bundle=self.bundle,
        )
        self.display_rect_left_bgr = _choose_display_rect_left(survey)
        self._apply_survey(survey)

    def _status_lines_from_plan(self, *, prefix: str) -> list[str]:
        if self.plan_preview is None:
            return [prefix, "No plan computed yet"]
        line1 = (
            f"{prefix} | next step {self._object_i()} | sequence={self.sequence_name} | "
            f"plan_time={1000.0 * self.plan_preview.planning_time_s:.0f} ms | status={self.plan_preview.status}"
        )
        line2 = f"reason={self.plan_preview.raw_reason}"
        if self.plan_preview.step is not None and self.plan_preview.selected_scene is not None:
            line2 = (
                f"#{self.plan_preview.selected_scene.det_index} {self.plan_preview.selected_scene.label} | "
                f"target=({self.plan_preview.step.placed.target_xy[0]:.1f},{self.plan_preview.step.placed.target_xy[1]:.1f},{self.plan_preview.step.placed.target_z_mm:.1f}) | "
                f"yaw={self.plan_preview.step.placed.yaw_deg:.1f}"
            )
        line3 = "Press right-arrow to preview/commit, v to validate all, r to clear bag state, s to resurvey"
        return [line1, line2, line3]

    def _render_disparity_panel(self) -> None:
        self.ax_disp.cla()
        self.ax_disp.set_facecolor(PANEL_BG)
        self.ax_disp.set_title("Disparity", color="#aaddff", fontsize=10, pad=6)
        if self.survey is None:
            self.ax_disp.text(0.5, 0.5, "No survey yet", transform=self.ax_disp.transAxes, color="white", ha="center", va="center")
        else:
            disp = _choose_display_disparity(self.survey, None if self.plan_preview is None else self.plan_preview.selected_dbg)
            if disp is not None:
                self.ax_disp.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        self.ax_disp.set_xticks([])
        self.ax_disp.set_yticks([])

    def _render_bag_panel(self) -> None:
        ax = self.ax_bag
        ax.cla()
        title = "3D Render | Bag Only"
        if self.preview is not None and self.preview.status_text:
            title = f"{title}\n{self.preview.status_text}"
        demo_mod._style_3d(ax, title)
        scale_pts: list[np.ndarray] = []
        scale_pts.extend(demo_mod._draw_place_zone(ax, self.render_surface_zone))
        for placed in self.placed_scene_objects:
            pts = np.asarray(placed.placed_points_xyz, dtype=np.float64).reshape(-1, 3)
            if len(pts) > 0:
                if len(pts) <= demo_mod.MAX_PTS_DISP:
                    sample = pts
                    colors = placed.scene_object.point_colors_rgb
                else:
                    idx = np.linspace(0, len(pts) - 1, demo_mod.MAX_PTS_DISP, dtype=np.int32)
                    sample = pts[idx]
                    colors = placed.scene_object.point_colors_rgb[idx]
                ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=0.8, alpha=0.62)
            demo_mod._draw_box_3d(ax, placed.placed_raw_box, placed.scene_object.color, alpha_face=0.10, alpha_edge=0.90, ls="-")
            demo_mod._draw_box_3d(ax, placed.placed_padded_box, placed.scene_object.color, alpha_face=0.0, alpha_edge=0.35, ls="--")
            scale_pts.append(placed.placed_raw_box.min_xyz_mm.copy())
            scale_pts.append(placed.placed_raw_box.max_xyz_mm.copy())

        if self.plan_preview is not None and self.plan_preview.step is not None and self.preview is None:
            step = self.plan_preview.step
            preview_color = "#7dff9c" if self.plan_preview.status == "ok" else "#ff6d6d"
            demo_mod._draw_box_3d(ax, step.placed.raw_box, preview_color, alpha_face=0.14, alpha_edge=0.95, ls="-")
            demo_mod._draw_box_3d(ax, step.placed.padded_box, preview_color, alpha_face=0.0, alpha_edge=0.42, ls="--")
            clearance_box = step.place_hold_box if self.plan_preview.status != "ok" else step.place_hold_box
            swept_box = step.place_swept_box
            demo_mod._draw_box_3d(ax, clearance_box, preview_color, alpha_face=0.02, alpha_edge=0.24, ls=":")
            demo_mod._draw_box_3d(ax, swept_box, preview_color, alpha_face=0.02, alpha_edge=0.18, ls="--")
            scale_pts.append(step.placed.raw_box.min_xyz_mm.copy())
            scale_pts.append(step.placed.raw_box.max_xyz_mm.copy())

        if self.preview is not None:
            if self.preview.moving_points_xyz is not None and self.preview.moving_point_colors_rgb is not None and len(self.preview.moving_points_xyz) > 0:
                if len(self.preview.moving_points_xyz) <= demo_mod.MAX_PTS_DISP:
                    sample = self.preview.moving_points_xyz
                    colors = self.preview.moving_point_colors_rgb
                else:
                    idx = np.linspace(0, len(self.preview.moving_points_xyz) - 1, demo_mod.MAX_PTS_DISP, dtype=np.int32)
                    sample = self.preview.moving_points_xyz[idx]
                    colors = self.preview.moving_point_colors_rgb[idx]
                ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=0.8, alpha=0.72)
            if self.preview.moving_raw_box is not None and self.preview.moving_object is not None:
                demo_mod._draw_box_3d(ax, self.preview.moving_raw_box, self.preview.moving_object.color, alpha_face=0.22, alpha_edge=0.98, ls="-")
                scale_pts.append(self.preview.moving_raw_box.min_xyz_mm.copy())
                scale_pts.append(self.preview.moving_raw_box.max_xyz_mm.copy())
            if self.preview.moving_padded_box is not None and self.preview.moving_object is not None:
                demo_mod._draw_box_3d(ax, self.preview.moving_padded_box, self.preview.moving_object.color, alpha_face=0.0, alpha_edge=0.40, ls="--")
            if self.preview.gripper_boxes:
                demo_mod._draw_gripper_3d(ax, self.preview.gripper_boxes, is_open=self.preview.gripper_open, color="#8fe7ff" if self.preview.gripper_open else "#ffd36e")
            if self.preview.clearance_box is not None:
                demo_mod._draw_box_3d(ax, self.preview.clearance_box, "#7dff9c" if self.preview.clearance_ok else "#ff6d6d", alpha_face=0.03, alpha_edge=0.28, ls=":")
            if self.preview.swept_box is not None:
                demo_mod._draw_box_3d(ax, self.preview.swept_box, "#7dff9c" if self.preview.clearance_ok else "#ff6d6d", alpha_face=0.02, alpha_edge=0.18, ls="--")

        ax.view_init(elev=26, azim=-53)
        if scale_pts:
            demo_mod._set_equal_aspect(ax, np.vstack(scale_pts))

    def _render(self) -> None:
        _draw_raw_stereo_pair(self.ax_raw, self.raw_left_bgr, self.raw_right_bgr)
        _draw_unbagged_pointcloud(self.ax_pc, self.remaining_scene_objects, None if self.plan_preview is None else self.plan_preview.selected_scene)
        self._render_disparity_panel()
        _draw_yolo_mask_panel(self.ax_mask, self.display_rect_left_bgr, self.remaining_scene_objects)
        self._render_bag_panel()
        self.fig.suptitle("\n".join(self.status_lines[:3]), color="white", fontsize=12, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _present_preview(self, preview: demo_mod.PreviewState) -> None:
        self.preview = preview
        self._render()
        if plt.get_backend().lower() != "agg":
            plt.pause(max(0.001, self.animate_step_delay_s))

    def _animate_current_step(self) -> None:
        if self.plan_preview is None or self.plan_preview.step is None or self.plan_preview.selected_scene is None:
            return
        step = self.plan_preview.step
        scene = self.plan_preview.selected_scene
        pick_phi = float(getattr(scene.candidate_debug.candidate, "pick_phi_deg", 0.0))
        place_phi = float(step.placed.yaw_deg)
        pick_xy = np.asarray(scene.candidate_debug.candidate.target_xy, dtype=np.float64).reshape(2)
        place_xy = np.asarray(step.placed.raw_box.center_xyz_mm[:2], dtype=np.float64).reshape(2)
        source_box = scene.raw_box
        open_width = demo_mod._opening_width_mm_for_box(source_box, pick_phi, extra_mm=demo_mod.ANIM_OPENING_EXTRA_MM)
        closed_width = demo_mod._opening_width_mm_for_box(source_box, pick_phi, extra_mm=demo_mod.ANIM_CLOSED_EXTRA_MM)
        survey_center = np.array(self.robot.fk(self.q_survey)[:3], dtype=np.float64)
        pick_grasp_z = demo_mod._gripper_center_z_for_box(source_box)
        pick_hover_z = pick_grasp_z + demo_mod.ANIM_PICK_HOVER_CLEARANCE_MM
        travel_object_min_z = max(float(source_box.max_xyz_mm[2]), float(step.placed.raw_box.max_xyz_mm[2])) + demo_mod.ANIM_TRAVEL_CLEARANCE_MM
        travel_gripper_z = demo_mod._gripper_center_z_for_box(
            demo_mod._make_object_box_at_pose(source_box, center_xy_mm=pick_xy, min_z_mm=travel_object_min_z, label="travel_raw")
        )
        place_grasp_z = demo_mod._gripper_center_z_for_box(step.placed.raw_box)
        place_hover_z = max(travel_gripper_z, place_grasp_z + demo_mod.ANIM_PICK_HOVER_CLEARANCE_MM)

        def _lerp(a: float, b: float, t: float) -> float:
            return (1.0 - t) * float(a) + t * float(b)

        def _lerp_xy(a_xy: np.ndarray, b_xy: np.ndarray, t: float) -> np.ndarray:
            return (1.0 - t) * np.asarray(a_xy, dtype=np.float64) + t * np.asarray(b_xy, dtype=np.float64)

        phases = [
            ("1. Survey to pick hover", None, pick_xy, pick_hover_z, pick_phi, open_width, True, False),
            ("2. Descend to pick", None, pick_xy, pick_grasp_z, pick_phi, open_width, True, False),
            ("3. Close gripper", None, pick_xy, pick_grasp_z, pick_phi, closed_width, False, False),
            ("4. Lift object", ("pick", source_box, pick_xy, float(source_box.min_xyz_mm[2]), pick_xy, travel_object_min_z), pick_xy, travel_gripper_z, pick_phi, closed_width, False, True),
            ("5. Carry to bag", ("carry", source_box, pick_xy, travel_object_min_z, place_xy, travel_object_min_z), place_xy, travel_gripper_z, place_phi, closed_width, False, True),
            ("6. Descend to place", ("place", source_box, place_xy, travel_object_min_z, place_xy, float(step.placed.raw_box.min_xyz_mm[2])), place_xy, place_grasp_z, place_phi, closed_width, False, True),
            ("7. Open release", ("release", source_box, place_xy, float(step.placed.raw_box.min_xyz_mm[2]), place_xy, float(step.placed.raw_box.min_xyz_mm[2])), place_xy, place_grasp_z, place_phi, open_width, True, True),
            ("8. Retract above place", ("retract", source_box, place_xy, float(step.placed.raw_box.min_xyz_mm[2]), place_xy, float(step.placed.raw_box.min_xyz_mm[2])), place_xy, place_hover_z, place_phi, open_width, True, False),
        ]

        for phase_name, moving_spec, gripper_xy, gripper_z, yaw_deg, width_mm, gripper_open, attached in phases:
            moving_raw_box = None
            moving_padded_box = None
            moving_points_xyz = None
            moving_point_colors_rgb = None
            if moving_spec is not None:
                _label, base_box, start_xy, start_min_z, end_xy, end_min_z = moving_spec
                object_xy = np.asarray(end_xy, dtype=np.float64).reshape(2)
                object_min_z = float(end_min_z)
                moving_raw_box = demo_mod._make_object_box_at_pose(base_box, center_xy_mm=object_xy, min_z_mm=object_min_z, label="moving_raw")
                moving_padded_box = wet_mod._placed_occupancy_from_plan(
                    center_xyz_mm=moving_raw_box.center_xyz_mm,
                    size_xyz_mm=moving_raw_box.size_xyz_mm,
                    label="moving_padded",
                ).padded_box
                moving_points_xyz = demo_mod._transform_points_for_box(scene, moving_raw_box)
                moving_point_colors_rgb = scene.point_colors_rgb
            gripper_boxes = demo_mod._make_gripper_boxes_at_pose(
                np.array([float(gripper_xy[0]), float(gripper_xy[1]), float(gripper_z)], dtype=np.float64),
                yaw_deg=float(yaw_deg),
                opening_width_mm=float(width_mm),
                label_prefix=f"preview_{phase_name.replace(' ', '_')}",
            )
            clearance_ok = step.pick_clearance_ok if phase_name.startswith(("1.", "2.", "3.", "4.")) else step.place_clearance_ok
            clearance_box = step.pick_hold_box if phase_name.startswith(("1.", "2.", "3.", "4.")) else step.place_hold_box
            swept_box = step.pick_swept_box if phase_name.startswith(("1.", "2.", "3.", "4.")) else step.place_swept_box
            self._present_preview(
                demo_mod.PreviewState(
                    moving_object=scene,
                    moving_raw_box=moving_raw_box,
                    moving_padded_box=moving_padded_box,
                    moving_points_xyz=moving_points_xyz,
                    moving_point_colors_rgb=moving_point_colors_rgb,
                    gripper_boxes=gripper_boxes,
                    gripper_open=gripper_open,
                    clearance_box=clearance_box,
                    swept_box=swept_box,
                    clearance_ok=clearance_ok,
                    status_text=f"{phase_name} | pick={step.pick_clearance_reason} place={step.place_clearance_reason}",
                    attached=attached,
                )
            )

        self.preview = None

    def preview_or_commit_next(self) -> None:
        if self.plan_preview is None:
            self.status_lines = ["No survey yet", "Press s to survey"]
            self._render()
            return
        if self.plan_preview.step is None:
            self.status_lines = self._status_lines_from_plan(prefix="Cannot place next object")
            self._render()
            return
        if self.plan_preview.status != "ok":
            self.status_lines = self._status_lines_from_plan(prefix="Placement blocked")
            self._render()
            return
        self._animate_current_step()
        self._commit_current_step()
        self.plan_preview = self._compute_next_plan()
        self.status_lines = self._status_lines_from_plan(prefix="Committed virtual placement")
        self._render()

    def reset_virtual_bag(self) -> None:
        self.remaining_scene_objects = list(self.scene_objects)
        self.placed_occupancies = []
        self.placed_scene_objects = []
        self.preview = None
        self.plan_preview = self._compute_next_plan()
        self.status_lines = self._status_lines_from_plan(prefix="Virtual bag reset")
        self._render()

    def validate_all_visible(self) -> ValidationResult:
        remaining_scene_objects = list(self.remaining_scene_objects)
        placed_occupancies = list(self.placed_occupancies)
        planning_times: list[float] = []
        placed_count = 0
        reason = "ok"

        while remaining_scene_objects:
            state = self._state_for_remaining()
            if state is None:
                reason = "no_survey"
                break
            allowed = {int(obj.det_index) for obj in remaining_scene_objects}
            state = replace(state, candidates=[dbg for dbg in state.candidates if int(getattr(dbg.candidate, "index", -1)) in allowed], selected_index=0)
            if not state.candidates:
                break
            ctx = wet_mod._planning_sequence_context(
                object_i=placed_count + len(self.placed_occupancies) + 1,
                robot=self.robot,
                placed_boxes=placed_occupancies,
                config=self.selector_config,
                surface_zone=self.surface_zone,
                base_xy=np.asarray(self.surface_zone["center_xy_mm"], dtype=np.float64).reshape(2),
                base_phi_deg=float(self.surface_zone["default_phi_deg"]),
                target_limit=max(1, len(remaining_scene_objects)),
            )
            t0 = time.perf_counter()
            selection = select_candidate_for_state(self.sequence_name, state, ctx=ctx, runtime=self._planning_runtime())
            planning_times.append(time.perf_counter() - t0)
            selected_dbg = selection.result.selected
            if selected_dbg is None:
                reason = "no_visible_pick_candidate"
                break
            selected_index = state.candidates.index(selected_dbg)
            target_plan = selection.target_cache.get(selected_index)
            if target_plan is None:
                try:
                    target_plan = wet_mod._compute_candidate_place_target_for_sequence(
                        selected_dbg,
                        state=state,
                        object_i=placed_count + len(self.placed_occupancies) + 1,
                        robot=self.robot,
                        placed_boxes=placed_occupancies,
                        config=self.selector_config,
                        surface_zone=self.surface_zone,
                        base_xy=np.asarray(self.surface_zone["center_xy_mm"], dtype=np.float64).reshape(2),
                        base_phi_deg=float(self.surface_zone["default_phi_deg"]),
                        target_limit=max(1, len(remaining_scene_objects)),
                    )
                except Exception as exc:
                    reason = _normalize_failure_reason(str(exc))
                    break
            if not bool(getattr(target_plan, "can_place", True)):
                reason = _normalize_failure_reason(str(getattr(target_plan, "reason", "rejected")))
                break

            scene_object = next((obj for obj in remaining_scene_objects if int(obj.det_index) == int(getattr(selected_dbg.candidate, "index", -1))), None)
            if scene_object is None:
                reason = "selected_scene_missing"
                break
            raw_box = wet_mod._aabb_from_object_candidate_quiet(selected_dbg.candidate, default_label=f"validate_object_{placed_count+1}")
            floor_z = float(self.render_surface_zone["surface_z_mm"])
            planner_layer_z = getattr(target_plan, "planner_layer_z_mm", None)
            if planner_layer_z is not None:
                center_z = floor_z + float(planner_layer_z) + 0.5 * float(raw_box.size_xyz_mm[2])
            elif placed_count <= 1:
                center_z = floor_z + 0.5 * float(raw_box.size_xyz_mm[2])
            else:
                below_box = placed_occupancies[placed_count - 1]
                center_z = float(below_box.raw_box.max_xyz_mm[2]) + 0.5 * float(raw_box.size_xyz_mm[2])
            placed_occ = wet_mod._placed_occupancy_from_plan(
                center_xyz_mm=np.array([float(target_plan.target_xy_mm[0]), float(target_plan.target_xy_mm[1]), float(center_z)], dtype=np.float64),
                size_xyz_mm=np.asarray(raw_box.size_xyz_mm, dtype=np.float64),
                label=f"validate_object_{placed_count+1}_placed",
            )

            servo_angle_deg = demo_mod._servo_angle_deg_for_candidate(scene_object.candidate_debug)
            pick_ok, pick_reason, _pick_hold, _pick_swept = demo_mod._audit_pick_clearance(
                scene_object,
                remaining_scene_objects,
                yaw_deg=float(getattr(scene_object.candidate_debug.candidate, "pick_phi_deg", 0.0)),
                servo_angle_deg=servo_angle_deg,
            )
            place_ok, place_reason, _place_hold, _place_swept = demo_mod._audit_place_clearance(
                placed_occ.raw_box,
                [box.raw_box for box in placed_occupancies],
                self.render_surface_zone,
                yaw_deg=float(target_plan.target_phi_deg),
                servo_angle_deg=servo_angle_deg,
            )
            if CHECK_GRIPPER_COLLISION and not pick_ok:
                reason = _normalize_failure_reason(pick_reason)
                break
            if CHECK_GRIPPER_COLLISION and not place_ok:
                reason = _normalize_failure_reason(place_reason)
                break

            placed_occupancies.append(placed_occ)
            remaining_scene_objects = [obj for obj in remaining_scene_objects if obj is not scene_object]
            placed_count += 1

        if reason == "ok":
            for i, box_a in enumerate(placed_occupancies):
                for j, box_b in enumerate(placed_occupancies[i + 1 :], start=i + 1):
                    if aabbs_intersect_3d(box_a.raw_box, box_b.raw_box):
                        reason = f"raw_intersection_{i+1}_{j+1}"
                        break
                    if aabbs_intersect_3d(box_a.padded_box, box_b.padded_box):
                        reason = f"padded_intersection_{i+1}_{j+1}"
                        break
                if reason != "ok":
                    break

        return ValidationResult(
            success=(reason == "ok"),
            placed_count=placed_count,
            expected_count=len(self.remaining_scene_objects),
            reason=reason,
            planning_step_times_s=planning_times,
        )

    def save_current_figure(self, stem: str) -> Path:
        out = SAVE_OUTPUT_DIR / f"{stem}.png"
        self.fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        return out

    def _on_key(self, event) -> None:
        if event.key in ("q", "Q", "escape"):
            plt.close(self.fig)
            return
        if event.key in ("r", "R"):
            self.reset_virtual_bag()
            return
        if event.key in ("v", "V", "a", "A"):
            result = self.validate_all_visible()
            self.status_lines = [
                f"Validation {'passed' if result.success else 'failed'} | placed={result.placed_count}/{result.expected_count}",
                f"reason={result.reason}",
                "step_times_ms=" + ", ".join(f"{1000.0 * t:.0f}" for t in result.planning_step_times_s),
            ]
            self._render()
            return
        if event.key in ("s", "S"):
            if self.live_mode:
                self.survey_live()
                self._render()
            return
        if event.key in ("right",):
            self.preview_or_commit_next()


def _load_models_and_calibration():
    dry_mod._configure_vision_modules()
    device_info = select_torch_device(use_cuda=wet_mod.USE_CUDA, use_half=wet_mod.USE_HALF)
    yolo, raft = load_vision(device_info)
    stereo_calib = load_stereo_calibration(wet_mod.STEREO_CALIBRATION_PATH)
    bundle = load_bundle(wet_mod.BUNDLE_PATH)
    rectifier = StereoRectifier(stereo_calib)
    detector = build_detector()
    return yolo, raft, stereo_calib, bundle, rectifier, detector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone live survey -> plan -> preview validator")
    parser.add_argument("--images", default=None, help="Optional saved stereo-pair folder. When omitted, live cameras are used.")
    parser.add_argument("--index", type=int, default=None, help="Saved stereo pair index")
    parser.add_argument("--runtime-context", default=None, choices=["wet_run", "saved_photo_test", "demo"], help="Override workspace/profile behavior")
    parser.add_argument("--sequence", default=wet_mod.PLACE_PLANNING_SEQUENCE_NAME, help="Planning sequence name")
    parser.add_argument("--animate-step-delay", type=float, default=0.03)
    parser.add_argument("--save", action="store_true", help="Save the current figure to scripts/smoke_tests")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--validate-all", action="store_true", help="After survey, validate all visible objects virtually")
    args = parser.parse_args(argv)

    live_mode = args.images is None
    context_name = args.runtime_context or ("wet_run" if live_mode else "saved_photo_test")
    runtime_context = resolve_runtime_context(context_name)
    workspace = get_workspace_filter_config(runtime_context.workspace_profile_name)
    selector_config = _best_candidate_config_for_workspace(workspace, use_robot_checks=live_mode)
    sequence_name = validate_planning_sequence_name(args.sequence)

    yolo, raft, stereo_calib, bundle, rectifier, detector = _load_models_and_calibration()
    robot, q_survey = _build_dry_fk_robot()
    surface_zone = dict(load_place_scene(DEFAULT_PLACE, verbose=False))
    render_surface_zone = demo_mod._load_demo_surface_zone()

    overhead = None
    stereo = None
    if live_mode:
        overhead = SimpleOverheadCamera(wet_mod.OVERHEAD_INDEX)
        stereo = SimpleStereoCamera(wet_mod.STEREO_INDEX)

    validator = LiveScenePlanValidator(
        surface_zone=surface_zone,
        render_surface_zone=render_surface_zone,
        selector_config=selector_config,
        sequence_name=sequence_name,
        bundle=bundle,
        stereo_calib=stereo_calib,
        rectifier=rectifier,
        yolo=yolo,
        raft=raft,
        detector=detector,
        robot=robot,
        q_survey=q_survey,
        overhead=overhead,
        stereo=stereo,
        live_mode=live_mode,
        save_frames=False,
        animate_step_delay_s=args.animate_step_delay,
    )

    try:
        if live_mode:
            if not validator.prompt_for_live_survey():
                return 0
            validator.survey_live()
        else:
            training_dir = Path(args.images)
            pair_index, left_path, right_path = demo_mod._find_stereo_pair(training_dir, args.index)
            print(f"[VALIDATOR] saved pair {pair_index:04d} left={left_path.name}")
            left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if left_bgr is None or right_bgr is None:
                raise RuntimeError("failed to load saved stereo pair")
            validator.survey_saved_pair(left_bgr, right_bgr)

        validator._render()

        if args.validate_all:
            result = validator.validate_all_visible()
            print(
                f"[VALIDATION] success={result.success} placed={result.placed_count}/{result.expected_count} "
                f"reason={result.reason} step_times_ms={[round(1000.0 * t, 1) for t in result.planning_step_times_s]}"
            )
            validator.status_lines = [
                f"Validation {'passed' if result.success else 'failed'} | placed={result.placed_count}/{result.expected_count}",
                f"reason={result.reason}",
                "step_times_ms=" + ", ".join(f"{1000.0 * t:.0f}" for t in result.planning_step_times_s),
            ]
            validator._render()

        if args.save:
            stem = "live_scene_plan_validator"
            if not live_mode and args.index is not None:
                stem = f"{stem}_{int(args.index):04d}"
            out = validator.save_current_figure(stem)
            print(f"[SAVE] {out}")

        if args.no_gui:
            plt.close(validator.fig)
            return 0

        plt.show()
        return 0
    finally:
        validator.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
