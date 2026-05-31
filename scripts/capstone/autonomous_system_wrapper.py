from __future__ import annotations

"""End-to-end autonomous system wrapper and robot-frame reconstruction viewer.

This is a capstone-facing companion to scripts/autonomous_missed_pick_recovery.py.
It reuses the same saved-image survey path as scripts/dry_run_autonomous.py, then
renders one high-level "what the robot sees" view:

1. Rectified left image with YOLO masks / boxes
2. RAFT disparity overlay
3. Robot-frame scene with:
   - RRPR survey-pose FK reconstruction
   - colored per-object point clouds
   - raw and padded object AABBs
   - platform bounds
   - configured bag / place zone
   - pick / place gripper geometry

Run:
    python scripts/capstone/autonomous_system_wrapper.py --index 1
    python scripts/capstone/autonomous_system_wrapper.py --index 1 --save --no-gui
"""

from pathlib import Path
import argparse
import math
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SAVE_OUTPUT_DIR = Path(__file__).resolve().parent
PLACE_ZONE_DISPLAY_HEIGHT_MM = 250.0
MAX_PTS_DISP = 4000
USE_CUDA = True
USE_HALF = True

GRIPPER_FINGER_LENGTH_MM = 70.0
GRIPPER_FINGER_WIDTH_MM = 12.0
GRIPPER_FINGER_HEIGHT_MM = 35.0
GRIPPER_PALM_WIDTH_MM = 40.0
GRIPPER_PALM_HEIGHT_MM = 18.0
ANIM_PICK_HOVER_CLEARANCE_MM = 60.0
ANIM_TRAVEL_CLEARANCE_MM = 80.0
ANIM_OPENING_EXTRA_MM = 30.0
ANIM_CLOSED_EXTRA_MM = 8.0

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config.pick.pick_config import DEFAULT_PICK
from config.pick.servo_config import DEFAULT_SERVO
from config.place.place_config import DEFAULT_PLACE
from config.robot_config import ROBOT_CONFIG
from hardware.robot import JointPose, Robot
from planning.aabb_utils import AxisAlignedBox3D, aabb_from_object_candidate, make_aabb_from_min_max, pad_aabb
import scripts.autonomous_missed_pick_recovery as auto_mod
import scripts.capstone.interactive_packing_demo as packing_demo_mod
import scripts.dry_run_autonomous as dry_mod
from vision.pointcloud import cam_points_to_robot_xyz
from vision.pick_candidate_builder import CandidateDebug, SurveyState
from vision.pick_survey_pipeline import load_vision
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


@dataclass
class SceneObject:
    label: str
    color: str
    det_index: int
    candidate_debug: CandidateDebug
    points_robot: np.ndarray
    point_colors_rgb: np.ndarray
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D


@dataclass
class AnimationOptions:
    animate_pick_place: bool = True
    step_delay_s: float = 0.04


@dataclass
class PreviewState:
    moving_object: SceneObject | None = None
    moving_raw_box: AxisAlignedBox3D | None = None
    moving_padded_box: AxisAlignedBox3D | None = None
    moving_points_xyz: np.ndarray | None = None
    moving_point_colors_rgb: np.ndarray | None = None
    gripper_boxes: tuple[AxisAlignedBox3D, ...] = ()
    gripper_open: bool = True
    status_text: str = ""
    attached: bool = False


@dataclass
class PlacedSceneObject:
    scene_object: SceneObject
    placed_raw_box: AxisAlignedBox3D
    placed_padded_box: AxisAlignedBox3D
    placed_points_xyz: np.ndarray
    object_i: int


@dataclass
class PlacementStep:
    scene_object: SceneObject
    placed: packing_demo_mod.PlacedObject
    object_i: int


_PALETTE = [
    "#2e8bcb",
    "#e8882a",
    "#2dc96e",
    "#d83f4a",
    "#8854cc",
    "#e8c025",
    "#4abbc9",
    "#e06090",
]
_BG = "#1a1a2e"
_PANEL_BG = "#0d0d1a"


def _find_stereo_pair(training_dir: Path, index: int | None) -> tuple[int, Path, Path]:
    pairs = dry_mod.find_stereo_pairs(training_dir)
    if index is not None:
        pairs = [p for p in pairs if p.index == index]
        if not pairs:
            raise FileNotFoundError(f"pair {index} not found in {training_dir}")
    if not pairs:
        raise FileNotFoundError(f"no stereo pairs found in {training_dir}")
    pair = pairs[0]
    return pair.index, pair.left_path, pair.right_path


def _sample_point_colors_rgb(rect_left_bgr: np.ndarray, uv_px: np.ndarray) -> np.ndarray:
    if len(uv_px) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    h, w = rect_left_bgr.shape[:2]
    xs = np.clip(np.rint(uv_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv_px[:, 1]).astype(np.int32), 0, h - 1)
    bgr = rect_left_bgr[ys, xs].astype(np.float32)
    return np.clip(bgr[:, ::-1] / 255.0, 0.0, 1.0)


def _build_scene_objects(survey: SurveyState, rect_left_bgr: np.ndarray, bundle: dict) -> list[SceneObject]:
    objects: list[SceneObject] = []
    for i, dbg in enumerate(survey.candidates, start=1):
        pts_cam = np.asarray(dbg.points_cam, dtype=np.float64).reshape(-1, 3)
        if len(pts_cam) == 0:
            continue
        pts_robot = cam_points_to_robot_xyz(pts_cam, bundle)
        point_colors_rgb = _sample_point_colors_rgb(rect_left_bgr, np.asarray(dbg.point_uv_px, dtype=np.float64).reshape(-1, 2))
        raw_box = aabb_from_object_candidate(dbg.candidate, default_label=dbg.candidate.yolo.class_name)
        padded = make_aabb_from_min_max(
            raw_box.min_xyz_mm - np.array([float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), 0.0], dtype=np.float64),
            raw_box.max_xyz_mm + np.array([float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), 0.0], dtype=np.float64),
            label=f"{raw_box.label}_padded",
        )
        objects.append(
            SceneObject(
                label=str(dbg.candidate.yolo.class_name),
                color=_PALETTE[(i - 1) % len(_PALETTE)],
                det_index=i,
                candidate_debug=dbg,
                points_robot=pts_robot,
                point_colors_rgb=point_colors_rgb,
                raw_box=raw_box,
                padded_box=padded,
            )
        )
    return objects


def _planner_objects_from_scene(scene_objects: list[SceneObject]) -> list[packing_demo_mod.ObjectInfo]:
    planner_objects: list[packing_demo_mod.ObjectInfo] = []
    for obj in scene_objects:
        planner_objects.append(
            packing_demo_mod.ObjectInfo(
                det_index=int(obj.det_index),
                class_name=str(obj.label),
                confidence=float(obj.candidate_debug.best_detection.confidence),
                color=str(obj.color),
                det=obj.candidate_debug.best_detection,
                points_cam=np.asarray(obj.candidate_debug.points_cam, dtype=np.float64).reshape(-1, 3),
                points_robot=np.asarray(obj.points_robot, dtype=np.float64).reshape(-1, 3),
                point_colors_rgb=np.asarray(obj.point_colors_rgb, dtype=np.float32).reshape(-1, 3),
                cand_dbg=obj.candidate_debug,
                raw_source_box=obj.raw_box,
                padded_source_box=obj.padded_box,
                object_props=packing_demo_mod._default_object_props(obj.label),
            )
        )
    return planner_objects


def _configure_packing_planner(surface_zone: dict[str, Any]) -> None:
    packing_demo_mod.PAD_X_MM = float(DEFAULT_PLACE.PAD_X_MM)
    packing_demo_mod.PAD_Y_MM = float(DEFAULT_PLACE.PAD_Y_MM)
    packing_demo_mod.PAD_Z_MM = float(DEFAULT_PLACE.PAD_Z_MM)
    packing_demo_mod.BAG_CENTER_XY_MM = [float(surface_zone["center_xy_mm"][0]), float(surface_zone["center_xy_mm"][1])]
    packing_demo_mod.BAG_WIDTH_MM = float(surface_zone.get("width_mm", 290.0))
    packing_demo_mod.BAG_DEPTH_MM = float(surface_zone.get("depth_mm", 175.0))
    packing_demo_mod.BAG_HEIGHT_MM = PLACE_ZONE_DISPLAY_HEIGHT_MM
    packing_demo_mod.BAG_SURFACE_Z_MM = float(surface_zone.get("surface_z_mm", 0.0))
    packing_demo_mod.PLANNER_LAYER_ACCEPT_SCORE = 0.10
    packing_demo_mod.SUPPORT_RATIO_THRESHOLD = 0.72
    packing_demo_mod.MAX_OVERHANG_RATIO = 0.28


def _compute_packing_sequence(scene_objects: list[SceneObject], surface_zone: dict[str, Any], *, pair_index: int) -> list[PlacementStep]:
    _configure_packing_planner(surface_zone)
    scene_by_det_index = {int(obj.det_index): obj for obj in scene_objects}
    debug = packing_demo_mod.DebugOptions()
    state = packing_demo_mod._simulate_full_plan(_planner_objects_from_scene(scene_objects), pair_index=pair_index, debug=debug)
    steps: list[PlacementStep] = []
    for placed in state.placed:
        scene_object = scene_by_det_index.get(int(placed.info.det_index))
        if scene_object is None:
            raise RuntimeError(f"missing scene object for det_index={placed.info.det_index}")
        steps.append(PlacementStep(scene_object=scene_object, placed=placed, object_i=int(placed.object_i)))
    return steps


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


def _box_faces(corners: np.ndarray) -> list[list[np.ndarray]]:
    faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
    return [[corners[i] for i in face] for face in faces]


def _draw_box_3d(ax, box: AxisAlignedBox3D, color: str, *, alpha_edge: float = 0.9, alpha_face: float = 0.10, ls: str = "-") -> None:
    c = _corners(box)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], [c[a, 2], c[b, 2]], color=color, alpha=alpha_edge, linewidth=1.1, linestyle=ls)
    if alpha_face > 0:
        fc = Poly3DCollection(_box_faces(c), alpha=alpha_face)
        fc.set_facecolor(color)
        ax.add_collection3d(fc)


def _style_3d(ax, title: str) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.set_title(title, color="#aaddff", fontsize=10, pad=6)
    ax.set_xlabel("X (mm)", color="#999", fontsize=7, labelpad=2)
    ax.set_ylabel("Y (mm)", color="#999", fontsize=7, labelpad=2)
    ax.set_zlabel("Z (mm)", color="#999", fontsize=7, labelpad=2)
    ax.tick_params(colors="#777", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#2a2a3a")


def _set_equal_aspect(ax, pts: np.ndarray) -> None:
    if len(pts) == 0:
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ctr = 0.5 * (mn + mx)
    span = max((mx - mn).max() * 0.60, 180.0)
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(min(-40.0, float(mn[2]) - 20.0), ctr[2] + span)


def _draw_detection_overlay(ax, rect_left_bgr: np.ndarray, survey: SurveyState, selected_dbg: CandidateDebug | None) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.set_title("Rectified Left + YOLO", color="#aaddff", fontsize=10, pad=6)
    rgb = cv2.cvtColor(rect_left_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    selected_id = id(selected_dbg) if selected_dbg is not None else None
    overlay = rgb.copy()
    for i, dbg in enumerate(survey.candidates, start=1):
        det = dbg.best_detection
        if det.mask is not None:
            mask = np.asarray(det.mask, dtype=bool)
            overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array([255, 220, 80], dtype=np.float32)).astype(np.uint8)
    ax.imshow(overlay, alpha=0.55)
    for i, dbg in enumerate(survey.candidates, start=1):
        det = dbg.best_detection
        x1, y1, x2, y2 = [float(v) for v in det.bbox]
        is_selected = id(dbg) == selected_id
        color = "#ffff55" if is_selected else _PALETTE[(i - 1) % len(_PALETTE)]
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=3.0 if is_selected else 1.8)
        ax.add_patch(rect)
        label = f"#{i} {det.class_name} {det.confidence:.2f}"
        ax.text(x1 + 3, max(10.0, y1 - 4), label, color=color, fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="#111122", ec=color, alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])


def _parse_teensy_constants(ino_path: Path) -> dict[str, str]:
    text = ino_path.read_text(encoding="utf-8", errors="replace")
    keys = [
        "J1_NORMAL_MAX_SPEED",
        "J1_NORMAL_ACCEL",
        "J2_NORMAL_MAX_SPEED",
        "J2_NORMAL_ACCEL",
        "J3_NORMAL_MAX_SPEED",
        "J3_NORMAL_ACCEL",
        "J4_NORMAL_MAX_SPEED",
        "J4_NORMAL_ACCEL",
        "HOME_J1",
        "HOME_J2",
        "HOME_J3",
        "HOME_J4",
        "J3_PRE_HOME_LIFT_MM",
    ]
    out: dict[str, str] = {}
    for key in keys:
        m = re.search(rf"{re.escape(key)}\s*=\s*([^;]+);", text)
        if m is not None:
            out[key] = str(m.group(1)).strip()
    return out


def _survey_joint_pose(robot: Robot) -> JointPose:
    q = robot.ik(DEFAULT_PICK.X_SURVEY_MM, DEFAULT_PICK.Y_SURVEY_MM, DEFAULT_PICK.Z_SURVEY_MM, 0.0)
    return robot.cfg.home_pose if q is None else q


def _draw_rrpr_arm(ax, robot: Robot, q: JointPose, *, color: str, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1 = math.radians(float(q.q1_deg))
    q2 = math.radians(float(q.q2_deg))
    z = float(q.z_mm)
    p0 = np.array([0.0, 0.0, z], dtype=np.float64)
    p1 = np.array([robot.cfg.L1_mm * math.cos(q1), robot.cfg.L1_mm * math.sin(q1), z], dtype=np.float64)
    p2 = np.array([robot.cfg.L1_mm * math.cos(q1) + robot.cfg.L2_mm * math.cos(q1 + q2), robot.cfg.L1_mm * math.sin(q1) + robot.cfg.L2_mm * math.sin(q1 + q2), z], dtype=np.float64)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, z], color="#6f6f88", linewidth=3.0, alpha=0.9)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=4.0, alpha=0.95)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=4.0, alpha=0.95)
    ax.scatter([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], [p0[2], p1[2], p2[2]], c=[color], s=34, alpha=0.95)
    ax.text(p2[0], p2[1], p2[2] + 12.0, label, color=color, fontsize=8, ha="center")
    return p0, p1, p2


def _make_gripper_boxes_at_pose(center_xyz_mm: np.ndarray, *, yaw_deg: float, opening_width_mm: float, label_prefix: str = "gripper") -> tuple[AxisAlignedBox3D, AxisAlignedBox3D, AxisAlignedBox3D]:
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


def _draw_gripper_3d(ax, gripper_boxes: tuple[AxisAlignedBox3D, ...], *, is_open: bool, color: str) -> None:
    for box in gripper_boxes:
        _draw_box_3d(ax, box, color, alpha_edge=0.95, alpha_face=0.24 if is_open else 0.18, ls="-")


def _opening_width_mm_for_box(box: AxisAlignedBox3D, yaw_deg: float, *, extra_mm: float) -> float:
    close_axis = 1 if abs(float(yaw_deg) % 180.0 - 90.0) <= 1e-3 else 0
    return max(22.0, float(box.size_xyz_mm[close_axis]) + float(extra_mm))


def _gripper_center_z_for_box(box: AxisAlignedBox3D) -> float:
    grip_depth = min(float(box.size_xyz_mm[2]) * 0.25, 18.0)
    return float(box.max_xyz_mm[2] - grip_depth + 0.5 * GRIPPER_FINGER_HEIGHT_MM)


def _draw_place_zone(ax, surface_zone: dict[str, Any]) -> list[np.ndarray]:
    center = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    half_w = 0.5 * float(surface_zone.get("width_mm", 290.0))
    half_d = 0.5 * float(surface_zone.get("depth_mm", 175.0))
    z0 = float(surface_zone.get("surface_z_mm", 0.0))
    z1 = z0 + PLACE_ZONE_DISPLAY_HEIGHT_MM
    box = make_aabb_from_min_max(
        np.array([center[0] - half_w, center[1] - half_d, z0], dtype=np.float64),
        np.array([center[0] + half_w, center[1] + half_d, z1], dtype=np.float64),
        label=str(surface_zone.get("name", "place_zone")),
    )
    _draw_box_3d(ax, box, "#7fd0a8", alpha_edge=0.85, alpha_face=0.04, ls="--")
    ax.text(box.center_xyz_mm[0], box.center_xyz_mm[1], box.max_xyz_mm[2] + 10.0, str(surface_zone.get("name", "Place Zone")), color="#7fd0a8", fontsize=8, ha="center")
    return list(_corners(box))


def _draw_platform_bounds(ax) -> list[np.ndarray]:
    z = 0.0
    min_x = float(auto_mod.PLATFORM_X_MIN_MM)
    max_x = float(auto_mod.PLATFORM_X_MAX_MM)
    min_y = float(auto_mod.PLATFORM_Y_MIN_MM)
    max_y = float(auto_mod.PLATFORM_Y_MAX_MM)
    corners = np.array(
        [
            [min_x, min_y, z],
            [max_x, min_y, z],
            [max_x, max_y, z],
            [min_x, max_y, z],
        ],
        dtype=np.float64,
    )
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], color="#f2c66d", linewidth=2.0, alpha=0.9)
    ax.text(0.5 * (min_x + max_x), min_y - 25.0, z + 5.0, "Platform Bounds", color="#f2c66d", fontsize=8, ha="center")
    return [corners]


def _render_summary_text(ax, *, selected_dbg: CandidateDebug | None, place_target_xy: np.ndarray | None, place_target_phi_deg: float | None, teensy: dict[str, str], robot: Robot, q_survey: JointPose) -> None:
    ax.set_facecolor(_PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_title("System Summary", color="#aaddff", fontsize=10, pad=6)

    lines = [
        "Autonomous path mirrored from:",
        "scripts/autonomous_missed_pick_recovery.py",
        "",
        f"Survey pose XYZ = ({DEFAULT_PICK.X_SURVEY_MM:.1f}, {DEFAULT_PICK.Y_SURVEY_MM:.1f}, {DEFAULT_PICK.Z_SURVEY_MM:.1f}) mm",
        f"Survey pose joints = (q1={q_survey.q1_deg:.1f}, q2={q_survey.q2_deg:.1f}, z={q_survey.z_mm:.1f}, phi={q_survey.phi_deg:.1f})",
        f"RRPR links = L1 {robot.cfg.L1_mm:.0f} mm, L2 {robot.cfg.L2_mm:.0f} mm",
        f"Pick phi mode = {DEFAULT_PICK.PICK_PHI_MODE}",
        f"Pick gripper offset = {DEFAULT_PICK.GRIPPER_OFFSET_MM:.1f} mm",
        f"Place zone = {DEFAULT_PLACE.PLACE_ZONE_NAME}",
        f"Adj direction = {DEFAULT_PLACE.ADJACENT_DIRECTION}  pad_z = {DEFAULT_PLACE.PAD_Z_MM:.1f} mm",
        "Placement planner = bag-local 3D AABB",
        f"Servo geometry L = {DEFAULT_SERVO.GRIPPER_GEOMETRY_L_MM:.1f} mm",
        "",
        f"YOLO imgsz/conf = {dry_mod._SURVEY.YOLO_IMGSZ} / {dry_mod._SURVEY.YOLO_CONF:.2f}",
        f"RAFT iters/downscale = {dry_mod._SURVEY.RAFT_VALID_ITERS} / {dry_mod._SURVEY.RAFT_DOWNSCALE:.2f}",
        f"Burst count / min hits = {dry_mod._SURVEY.BURST_COUNT} / {dry_mod._SURVEY.MIN_BURST_HITS}",
        "",
        "Teensy motion constants:",
        f"J1 speed/accel = {teensy.get('J1_NORMAL_MAX_SPEED', '?')} / {teensy.get('J1_NORMAL_ACCEL', '?')}",
        f"J2 speed/accel = {teensy.get('J2_NORMAL_MAX_SPEED', '?')} / {teensy.get('J2_NORMAL_ACCEL', '?')}",
        f"J3 speed/accel = {teensy.get('J3_NORMAL_MAX_SPEED', '?')} / {teensy.get('J3_NORMAL_ACCEL', '?')}",
        f"J4 speed/accel = {teensy.get('J4_NORMAL_MAX_SPEED', '?')} / {teensy.get('J4_NORMAL_ACCEL', '?')}",
        f"Home J3 / pre-home lift = {teensy.get('HOME_J3', '?')} / {teensy.get('J3_PRE_HOME_LIFT_MM', '?')}",
        "",
        "Runtime note:",
        "Saved-pair demos run RAFT once per launch.",
        "Wet autonomous reruns YOLO + RAFT on each fresh survey.",
    ]
    if selected_dbg is not None:
        c = selected_dbg.candidate
        lines += [
            "",
            f"Selected = {c.yolo.class_name}",
            f"Pick target XY = ({float(c.target_xy[0]):.1f}, {float(c.target_xy[1]):.1f})",
            f"Pick top/grasp Z = {float(c.object_robot_xyz_raw[2]):.1f} / {float(c.grasp_robot_z):.1f}",
            f"Pick phi = {float(getattr(c, 'pick_phi_deg', 0.0)):.1f} deg",
        ]
    if place_target_xy is not None and place_target_phi_deg is not None:
        lines += [
            f"Place target XY = ({float(place_target_xy[0]):.1f}, {float(place_target_xy[1]):.1f})",
            f"Place phi = {float(place_target_phi_deg):.1f} deg",
        ]

    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", color="#dddddd", fontsize=8, family="monospace")


def _make_object_box_at_pose(reference_box: AxisAlignedBox3D, *, center_xy_mm: np.ndarray, min_z_mm: float, label: str) -> AxisAlignedBox3D:
    size = np.asarray(reference_box.size_xyz_mm, dtype=np.float64).reshape(3)
    center_xy = np.asarray(center_xy_mm, dtype=np.float64).reshape(2)
    min_xyz = np.array([center_xy[0] - 0.5 * size[0], center_xy[1] - 0.5 * size[1], float(min_z_mm)], dtype=np.float64)
    return make_aabb_from_min_max(min_xyz, min_xyz + size, label=label)


def _transform_points_for_box(scene_object: SceneObject, target_raw_box: AxisAlignedBox3D) -> np.ndarray:
    pts = np.asarray(scene_object.points_robot, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    centroid_xy = np.mean(pts[:, :2], axis=0)
    z_min = float(pts[:, 2].min())
    shift = np.array(
        [
            float(target_raw_box.center_xyz_mm[0]) - float(centroid_xy[0]),
            float(target_raw_box.center_xyz_mm[1]) - float(centroid_xy[1]),
            float(target_raw_box.min_xyz_mm[2]) - z_min,
        ],
        dtype=np.float64,
    )
    return pts + shift


class SystemViewer:
    def __init__(
        self,
        *,
        pair_index: int,
        rect_left_bgr: np.ndarray,
        survey: SurveyState,
        scene_objects: list[SceneObject],
        placement_steps: list[PlacementStep],
        surface_zone: dict[str, Any],
        teensy_constants: dict[str, str],
        robot: Robot,
        q_survey: JointPose,
        animation: AnimationOptions,
    ) -> None:
        self.pair_index = pair_index
        self.rect_left_bgr = rect_left_bgr
        self.survey = survey
        self.scene_objects = scene_objects
        self.placement_steps = placement_steps
        self.surface_zone = surface_zone
        self.teensy_constants = teensy_constants
        self.robot = robot
        self.q_survey = q_survey
        self.animation = animation
        self.preview: PreviewState | None = None
        self.remaining_scene_objects = list(scene_objects)
        self.placed_scene_objects: list[PlacedSceneObject] = []
        self.selected_dbg: CandidateDebug | None = None
        self.selected_scene: SceneObject | None = None
        self.place_target_xy: np.ndarray | None = None
        self.place_target_phi_deg: float | None = None
        self.place_raw_box: AxisAlignedBox3D | None = None
        self.place_padded_box: AxisAlignedBox3D | None = None
        self.current_object_i: int | None = None
        self.total_objects = len(placement_steps)
        self._set_active_step(self.placement_steps[0] if self.placement_steps else None)

        self.fig = plt.figure(figsize=(19, 11), facecolor=_BG)
        gs = gridspec.GridSpec(2, 3, figure=self.fig, height_ratios=[1.0, 1.25], hspace=0.10, wspace=0.05)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_disp = self.fig.add_subplot(gs[0, 1])
        self.ax_text = self.fig.add_subplot(gs[0, 2])
        self.ax3 = self.fig.add_subplot(gs[1, :], projection="3d")
        self.render()

    def _set_active_step(self, step: PlacementStep | None) -> None:
        if step is None:
            self.selected_scene = None
            self.selected_dbg = None
            self.place_target_xy = None
            self.place_target_phi_deg = None
            self.place_raw_box = None
            self.place_padded_box = None
            self.current_object_i = None
            return
        self.selected_scene = step.scene_object
        self.selected_dbg = step.scene_object.candidate_debug
        self.place_target_xy = np.asarray(step.placed.target_xy, dtype=np.float64).reshape(2)
        self.place_target_phi_deg = float(step.placed.yaw_deg)
        self.place_raw_box = step.placed.raw_box
        self.place_padded_box = step.placed.padded_box
        self.current_object_i = int(step.object_i)

    def _commit_step(self, step: PlacementStep) -> None:
        self.remaining_scene_objects = [obj for obj in self.remaining_scene_objects if obj is not step.scene_object]
        self.placed_scene_objects.append(
            PlacedSceneObject(
                scene_object=step.scene_object,
                placed_raw_box=step.placed.raw_box,
                placed_padded_box=step.placed.padded_box,
                placed_points_xyz=_transform_points_for_box(step.scene_object, step.placed.raw_box),
                object_i=step.object_i,
            )
        )

    def _render_robot_scene(self) -> None:
        ax3 = self.ax3
        active_step = ""
        if self.current_object_i is not None and self.total_objects > 0:
            active_step = f" | step {self.current_object_i}/{self.total_objects}"
        _style_3d(ax3, f"Robot-Frame Reconstruction | pair {self.pair_index:04d}{active_step}" + (f"\n{self.preview.status_text}" if self.preview is not None and self.preview.status_text else ""))
        scale_pts: list[np.ndarray] = []
        scale_pts.extend(_draw_platform_bounds(ax3))
        scale_pts.extend(_draw_place_zone(ax3, self.surface_zone))

        _draw_rrpr_arm(ax3, self.robot, self.q_survey, color="#9ecbff", label="Survey Pose")
        x_fk, y_fk, z_fk, _phi_fk = self.robot.fk(self.q_survey)
        survey_gripper = _make_gripper_boxes_at_pose(
            np.array([x_fk, y_fk, z_fk], dtype=np.float64),
            yaw_deg=self.q_survey.phi_deg,
            opening_width_mm=max(24.0, 2.0 * DEFAULT_SERVO.GRIPPER_GEOMETRY_L_MM * math.sin(math.radians(DEFAULT_SERVO.DYNAMIC_PICK_DEFAULT_SERVO_DEG / 2.0))),
            label_prefix="survey_gripper",
        )
        _draw_gripper_3d(ax3, survey_gripper, is_open=True, color="#9ecbff")
        for box in survey_gripper:
            scale_pts.append(_corners(box))

        for obj in self.remaining_scene_objects:
            skip_static = self.preview is not None and self.preview.attached and self.preview.moving_object is obj
            if skip_static:
                continue
            if len(obj.points_robot) > 0:
                sample_idx = None
                if len(obj.points_robot) <= MAX_PTS_DISP:
                    samp = obj.points_robot
                else:
                    sample_idx = np.random.choice(len(obj.points_robot), MAX_PTS_DISP, replace=False)
                    samp = obj.points_robot[sample_idx]
                samp_colors = obj.point_colors_rgb if sample_idx is None else obj.point_colors_rgb[sample_idx]
                ax3.scatter(samp[:, 0], samp[:, 1], samp[:, 2], c=samp_colors, s=0.7, alpha=0.50)
            _draw_box_3d(ax3, obj.raw_box, obj.color, alpha_face=0.14 if obj is self.selected_scene else 0.09, alpha_edge=0.95 if obj is self.selected_scene else 0.55, ls="-")
            _draw_box_3d(ax3, obj.padded_box, obj.color, alpha_face=0.0, alpha_edge=0.40, ls="--")
            ax3.text(obj.raw_box.center_xyz_mm[0], obj.raw_box.center_xyz_mm[1], obj.raw_box.max_xyz_mm[2] + 8.0, f"#{obj.det_index} {obj.label}", color="white", fontsize=7, ha="center")
            scale_pts.append(_corners(obj.padded_box))

        for placed in self.placed_scene_objects:
            pts = placed.placed_points_xyz
            if len(pts) > 0:
                sample_idx = None
                if len(pts) <= MAX_PTS_DISP:
                    samp = pts
                else:
                    sample_idx = np.random.choice(len(pts), MAX_PTS_DISP, replace=False)
                    samp = pts[sample_idx]
                colors = placed.scene_object.point_colors_rgb if sample_idx is None else placed.scene_object.point_colors_rgb[sample_idx]
                ax3.scatter(samp[:, 0], samp[:, 1], samp[:, 2], c=colors, s=0.7, alpha=0.55)
            _draw_box_3d(ax3, placed.placed_raw_box, placed.scene_object.color, alpha_face=0.15, alpha_edge=0.90, ls="-")
            _draw_box_3d(ax3, placed.placed_padded_box, placed.scene_object.color, alpha_face=0.0, alpha_edge=0.38, ls="--")
            ax3.text(
                placed.placed_raw_box.center_xyz_mm[0],
                placed.placed_raw_box.center_xyz_mm[1],
                placed.placed_raw_box.max_xyz_mm[2] + 8.0,
                f"{placed.object_i}. {placed.scene_object.label}",
                color="#f0fff0",
                fontsize=7,
                ha="center",
            )
            scale_pts.append(_corners(placed.placed_padded_box))

        if self.selected_scene is not None and self.preview is None:
            c = self.selected_scene.candidate_debug.candidate
            pick_phi = float(getattr(c, "pick_phi_deg", 0.0))
            pick_box = self.selected_scene.raw_box
            pick_gripper_center = np.array([float(c.target_xy[0]), float(c.target_xy[1]), _gripper_center_z_for_box(pick_box)], dtype=np.float64)
            pick_gripper = _make_gripper_boxes_at_pose(
                pick_gripper_center,
                yaw_deg=pick_phi,
                opening_width_mm=_opening_width_mm_for_box(pick_box, pick_phi, extra_mm=ANIM_OPENING_EXTRA_MM),
                label_prefix="pick_gripper",
            )
            _draw_gripper_3d(ax3, pick_gripper, is_open=True, color="#ffe28a")
            ax3.text(pick_gripper_center[0], pick_gripper_center[1], pick_gripper_center[2] + 22.0, "Pick Gripper", color="#ffe28a", fontsize=8, ha="center")
            for box in pick_gripper:
                scale_pts.append(_corners(box))

        if self.place_raw_box is not None and self.place_padded_box is not None:
            _draw_box_3d(ax3, self.place_raw_box, "#8dff9e", alpha_face=0.16, alpha_edge=0.95, ls="-")
            _draw_box_3d(ax3, self.place_padded_box, "#8dff9e", alpha_face=0.0, alpha_edge=0.45, ls="--")
            if self.place_target_phi_deg is not None:
                place_gripper = _make_gripper_boxes_at_pose(
                    np.array(
                        [
                            float(self.place_raw_box.center_xyz_mm[0]),
                            float(self.place_raw_box.center_xyz_mm[1]),
                            _gripper_center_z_for_box(self.place_raw_box),
                        ],
                        dtype=np.float64,
                    ),
                    yaw_deg=float(self.place_target_phi_deg),
                    opening_width_mm=_opening_width_mm_for_box(self.place_raw_box, float(self.place_target_phi_deg), extra_mm=ANIM_OPENING_EXTRA_MM),
                    label_prefix="place_gripper",
                )
                if self.preview is None:
                    _draw_gripper_3d(ax3, place_gripper, is_open=True, color="#8dff9e")
                for box in place_gripper:
                    scale_pts.append(_corners(box))
            ax3.text(self.place_raw_box.center_xyz_mm[0], self.place_raw_box.center_xyz_mm[1], self.place_raw_box.max_xyz_mm[2] + 16.0, "Planned Place", color="#8dff9e", fontsize=8, ha="center")
            scale_pts.append(_corners(self.place_padded_box))

        if self.preview is not None:
            if self.preview.moving_points_xyz is not None and self.preview.moving_point_colors_rgb is not None and len(self.preview.moving_points_xyz) > 0:
                sample_idx = None
                if len(self.preview.moving_points_xyz) <= MAX_PTS_DISP:
                    samp = self.preview.moving_points_xyz
                else:
                    sample_idx = np.random.choice(len(self.preview.moving_points_xyz), MAX_PTS_DISP, replace=False)
                    samp = self.preview.moving_points_xyz[sample_idx]
                samp_colors = self.preview.moving_point_colors_rgb if sample_idx is None else self.preview.moving_point_colors_rgb[sample_idx]
                ax3.scatter(samp[:, 0], samp[:, 1], samp[:, 2], c=samp_colors, s=0.8, alpha=0.70)
            if self.preview.moving_raw_box is not None and self.preview.moving_object is not None:
                _draw_box_3d(ax3, self.preview.moving_raw_box, self.preview.moving_object.color, alpha_face=0.24, alpha_edge=0.98, ls="-")
                scale_pts.append(_corners(self.preview.moving_raw_box))
            if self.preview.moving_padded_box is not None and self.preview.moving_object is not None:
                _draw_box_3d(ax3, self.preview.moving_padded_box, self.preview.moving_object.color, alpha_face=0.0, alpha_edge=0.45, ls="--")
                scale_pts.append(_corners(self.preview.moving_padded_box))
            if self.preview.gripper_boxes:
                _draw_gripper_3d(ax3, self.preview.gripper_boxes, is_open=self.preview.gripper_open, color="#ffd36e" if not self.preview.gripper_open else "#8fe7ff")
                for box in self.preview.gripper_boxes:
                    scale_pts.append(_corners(box))

        ax3.view_init(elev=24, azim=-58)
        if scale_pts:
            _set_equal_aspect(ax3, np.vstack(scale_pts))

    def _render_static_panels(self) -> None:
        for ax in (self.ax_img, self.ax_disp, self.ax_text):
            ax.cla()
        _draw_detection_overlay(self.ax_img, self.rect_left_bgr, self.survey, self.selected_dbg)
        self.ax_disp.set_facecolor(_PANEL_BG)
        self.ax_disp.set_title("RAFT Disparity", color="#aaddff", fontsize=10, pad=6)
        disparity_overlay = self.survey.candidates[0].disparity_overlay if self.survey.candidates else np.zeros_like(self.rect_left_bgr)
        self.ax_disp.imshow(cv2.cvtColor(disparity_overlay, cv2.COLOR_BGR2RGB))
        self.ax_disp.set_xticks([])
        self.ax_disp.set_yticks([])
        _render_summary_text(
            self.ax_text,
            selected_dbg=self.selected_dbg,
            place_target_xy=self.place_target_xy,
            place_target_phi_deg=self.place_target_phi_deg,
            teensy=self.teensy_constants,
            robot=self.robot,
            q_survey=self.q_survey,
        )

    def _render_robot_panel(self) -> None:
        self.ax3.cla()
        self._render_robot_scene()

    def render(self) -> None:
        self._render_static_panels()
        self._render_robot_panel()
        self.fig.suptitle("Autonomous Missed-Pick Recovery | What The Robot Sees", color="white", fontsize=14, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _present_preview(self, preview: PreviewState) -> None:
        self.preview = preview
        self._render_robot_panel()
        self.fig.canvas.draw_idle()
        if plt.get_backend().lower() != "agg":
            plt.pause(max(0.001, float(self.animation.step_delay_s)))

    def _animate_single_step(self, step: PlacementStep) -> None:
        self._set_active_step(step)
        self.preview = None
        self.render()
        if self.selected_scene is None or self.place_raw_box is None:
            return
        c = self.selected_scene.candidate_debug.candidate
        pick_phi = float(getattr(c, "pick_phi_deg", 0.0))
        place_phi = float(self.place_target_phi_deg if self.place_target_phi_deg is not None else pick_phi)
        pick_xy = np.asarray(c.target_xy, dtype=np.float64).reshape(2)
        place_xy = np.asarray(self.place_raw_box.center_xyz_mm[:2], dtype=np.float64).reshape(2)
        source_box = self.selected_scene.raw_box
        open_width = _opening_width_mm_for_box(source_box, pick_phi, extra_mm=ANIM_OPENING_EXTRA_MM)
        closed_width = _opening_width_mm_for_box(source_box, pick_phi, extra_mm=ANIM_CLOSED_EXTRA_MM)
        survey_center = np.array(self.robot.fk(self.q_survey)[:3], dtype=np.float64)
        pick_grasp_z = _gripper_center_z_for_box(source_box)
        pick_hover_z = pick_grasp_z + ANIM_PICK_HOVER_CLEARANCE_MM
        travel_object_min_z = max(float(source_box.max_xyz_mm[2]), float(self.place_raw_box.max_xyz_mm[2])) + ANIM_TRAVEL_CLEARANCE_MM
        travel_gripper_z = _gripper_center_z_for_box(_make_object_box_at_pose(source_box, center_xy_mm=pick_xy, min_z_mm=travel_object_min_z, label="travel_raw"))
        place_grasp_z = _gripper_center_z_for_box(self.place_raw_box)
        place_hover_z = max(travel_gripper_z, place_grasp_z + ANIM_PICK_HOVER_CLEARANCE_MM)

        def _lerp(a: float, b: float, t: float) -> float:
            return (1.0 - t) * float(a) + t * float(b)

        def _lerp_xy(a_xy: np.ndarray, b_xy: np.ndarray, t: float) -> np.ndarray:
            return (1.0 - t) * np.asarray(a_xy, dtype=np.float64) + t * np.asarray(b_xy, dtype=np.float64)

        phases = [
            ("1. Survey to pick hover", [1.0], lambda t: (None, _make_gripper_boxes_at_pose(np.array([_lerp(survey_center[0], pick_xy[0], t), _lerp(survey_center[1], pick_xy[1], t), _lerp(survey_center[2], pick_hover_z, t)], dtype=np.float64), yaw_deg=pick_phi, opening_width_mm=open_width, label_prefix="hover_pick"), True)),
            ("2. Descend to pick", [1.0], lambda t: (None, _make_gripper_boxes_at_pose(np.array([pick_xy[0], pick_xy[1], _lerp(pick_hover_z, pick_grasp_z, t)], dtype=np.float64), yaw_deg=pick_phi, opening_width_mm=open_width, label_prefix="desc_pick"), True)),
            ("3. Close gripper", [1.0], lambda t: (None, _make_gripper_boxes_at_pose(np.array([pick_xy[0], pick_xy[1], pick_grasp_z], dtype=np.float64), yaw_deg=pick_phi, opening_width_mm=_lerp(open_width, closed_width, t), label_prefix="close_pick"), False)),
            ("4. Lift object", [1.0], lambda t: (_make_object_box_at_pose(source_box, center_xy_mm=pick_xy, min_z_mm=_lerp(float(source_box.min_xyz_mm[2]), travel_object_min_z, t), label="lift_obj"), _make_gripper_boxes_at_pose(np.array([pick_xy[0], pick_xy[1], _lerp(pick_grasp_z, travel_gripper_z, t)], dtype=np.float64), yaw_deg=pick_phi, opening_width_mm=closed_width, label_prefix="lift_pick"), False)),
            ("5. Carry to bag", [1.0], lambda t: (_make_object_box_at_pose(source_box, center_xy_mm=_lerp_xy(pick_xy, place_xy, t), min_z_mm=travel_object_min_z, label="carry_obj"), _make_gripper_boxes_at_pose(np.array([_lerp(pick_xy[0], place_xy[0], t), _lerp(pick_xy[1], place_xy[1], t), travel_gripper_z], dtype=np.float64), yaw_deg=place_phi, opening_width_mm=closed_width, label_prefix="carry_gripper"), False)),
            ("6. Descend to place", [1.0], lambda t: (_make_object_box_at_pose(source_box, center_xy_mm=place_xy, min_z_mm=_lerp(travel_object_min_z, float(self.place_raw_box.min_xyz_mm[2]), t), label="place_obj"), _make_gripper_boxes_at_pose(np.array([place_xy[0], place_xy[1], _lerp(place_hover_z, place_grasp_z, t)], dtype=np.float64), yaw_deg=place_phi, opening_width_mm=closed_width, label_prefix="desc_place"), False)),
            ("7. Open release", [1.0], lambda t: (_make_object_box_at_pose(source_box, center_xy_mm=place_xy, min_z_mm=float(self.place_raw_box.min_xyz_mm[2]), label="release_obj"), _make_gripper_boxes_at_pose(np.array([place_xy[0], place_xy[1], place_grasp_z], dtype=np.float64), yaw_deg=place_phi, opening_width_mm=_lerp(closed_width, open_width, t), label_prefix="open_place"), True)),
            ("8. Retract above place", [1.0], lambda t: (_make_object_box_at_pose(source_box, center_xy_mm=place_xy, min_z_mm=float(self.place_raw_box.min_xyz_mm[2]), label="placed_obj"), _make_gripper_boxes_at_pose(np.array([place_xy[0], place_xy[1], _lerp(place_grasp_z, place_hover_z, t)], dtype=np.float64), yaw_deg=place_phi, opening_width_mm=open_width, label_prefix="retract_place"), True)),
        ]

        for status, ts, builder in phases:
            for t in ts:
                moving_raw_box, gripper_boxes, gripper_open = builder(float(t))
                moving_padded_box = None
                moving_points = None
                moving_colors = None
                attached = moving_raw_box is not None
                if moving_raw_box is not None:
                    moving_padded_box = pad_aabb(moving_raw_box, DEFAULT_PLACE.PAD_X_MM, DEFAULT_PLACE.PAD_Y_MM, DEFAULT_PLACE.PAD_Z_MM).padded_box
                    moving_points = _transform_points_for_box(self.selected_scene, moving_raw_box)
                    moving_colors = self.selected_scene.point_colors_rgb
                self._present_preview(
                    PreviewState(
                        moving_object=self.selected_scene,
                        moving_raw_box=moving_raw_box,
                        moving_padded_box=moving_padded_box,
                        moving_points_xyz=moving_points,
                        moving_point_colors_rgb=moving_colors,
                        gripper_boxes=gripper_boxes,
                        gripper_open=gripper_open,
                        status_text=f"{step.object_i}/{self.total_objects} {self.selected_scene.label} | {status}",
                        attached=attached,
                    )
                )

        self.preview = None
        self._commit_step(step)

    def animate_pick_place(self) -> None:
        if not self.animation.animate_pick_place or not self.placement_steps:
            return
        for step_i, step in enumerate(self.placement_steps):
            self._animate_single_step(step)
            next_step = self.placement_steps[step_i + 1] if step_i + 1 < len(self.placement_steps) else None
            self._set_active_step(next_step)
            self.render()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(_REPO_ROOT / "Training_Images"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--animate-step-delay", type=float, default=0.04)
    args = parser.parse_args(argv)

    training_dir = Path(args.images)
    pair_index, left_path, right_path = _find_stereo_pair(training_dir, args.index)

    print(f"[WRAPPER] pair {pair_index:04d} left={left_path.name}")
    dry_mod._configure_vision_modules()

    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)
    yolo, raft = load_vision(device_info)

    calib_path = _REPO_ROOT / dry_mod._SURVEY.STEREO_CALIBRATION_PATH
    stereo_data = np.load(str(calib_path), allow_pickle=False)
    stereo_calib = {k: np.asarray(stereo_data[k]) for k in stereo_data.files}
    rectifier = StereoRectifier(stereo_calib)

    bundle_path = _REPO_ROOT / dry_mod._SURVEY.BUNDLE_PATH
    from test_calibration_bundle_live_stereo_z_pickplace import load_bundle

    bundle = load_bundle(bundle_path)

    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        raise RuntimeError("failed to load stereo pair")

    survey = dry_mod.survey_from_images(
        left_bgr,
        right_bgr,
        yolo=yolo,
        raft=raft,
        rectifier=rectifier,
        stereo_calib=stereo_calib,
        bundle=bundle,
    )
    if not survey.candidates:
        raise RuntimeError("no survey candidates")

    surface_zone = dry_mod._load_surface_zone()
    scene_objects = _build_scene_objects(survey, left_bgr, bundle)
    placement_steps = _compute_packing_sequence(scene_objects, surface_zone, pair_index=pair_index)
    print(f"[WRAPPER] computed {len(placement_steps)} planner placement step(s)")

    teensy_constants = _parse_teensy_constants(_REPO_ROOT / "Teensy_Code" / "Teensy_Code.ino")
    robot = Robot(config=ROBOT_CONFIG, connect=False)
    q_survey = _survey_joint_pose(robot)
    viewer = SystemViewer(
        pair_index=pair_index,
        rect_left_bgr=left_bgr,
        survey=survey,
        scene_objects=scene_objects,
        placement_steps=placement_steps,
        surface_zone=surface_zone,
        teensy_constants=teensy_constants,
        robot=robot,
        q_survey=q_survey,
        animation=AnimationOptions(
            animate_pick_place=not bool(args.no_animate) and not bool(args.no_gui),
            step_delay_s=max(0.0, float(args.animate_step_delay)),
        ),
    )

    if args.save:
        out = SAVE_OUTPUT_DIR / f"autonomous_system_wrapper_{pair_index:04d}.png"
        viewer.fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor=viewer.fig.get_facecolor())
        print(f"[SAVE] {out}")

    if args.no_gui:
        plt.close(viewer.fig)
        return 0

    viewer.animate_pick_place()
    plt.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
