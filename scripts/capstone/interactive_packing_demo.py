from __future__ import annotations

"""Interactive capstone packing demo with deterministic 3D bag-local planning."""

from pathlib import Path
import argparse
import json as _json
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TRAINING_IMAGES_DIR = _REPO_ROOT / "Training_Images"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
YOLO_WEIGHTS_PATH = _REPO_ROOT / "yolo_weights/full_data.pt"
YOLO_FALLBACK_PATH = _REPO_ROOT / "yolo_weights/validate_V2.pt"
RAFT_ROOT = _REPO_ROOT / "RAFT-Stereo"
RAFT_CKPT_PATH = _REPO_ROOT / "RAFT-Stereo/models/raftstereo-middlebury.pth"

YOLO_CONF = 0.35
YOLO_IMGSZ = 640
PAD_X_MM = 0.0
PAD_Y_MM = 0.0
PAD_Z_MM = 20.0

IMAGES_ALREADY_RECTIFIED = True
MAX_PTS_DISP = 3000

BAG_CENTER_XY_MM = [190.0, 270.0]
BAG_WIDTH_MM = 290.0
BAG_DEPTH_MM = 175.0
BAG_HEIGHT_MM = 250.0
BAG_SURFACE_Z_MM = 0.0

PLATFORM_X_MIN_MM = 40.0
PLATFORM_X_MAX_MM = 340.0
PLATFORM_Y_MIN_MM = 40.0
PLATFORM_Y_MAX_MM = 490.0

USE_CUDA = True
USE_HALF = True

SAVE_OUTPUT_DIR = _REPO_ROOT / "outputs/capstone/packing_demo"

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from planning.aabb_utils import (
    AxisAlignedBox3D,
    aabb_from_object_candidate,
    make_aabb_from_min_max,
)
from planning.bag_local_3d_aabb_planner import (
    PlacementAttempt3D,
    aabbs_intersect_3d,
    plan_bag_local_aabb_placement,
)
from scripts.autonomous_best_candidate import BestCandidateConfig, choose_best_candidate
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
_DEBUG_PLACED = "#6fa8dc"
_DEBUG_REJECT = "#ff6b6b"
_DEBUG_ACCEPT = "#7dff7d"

_GROCERY_SPEC_PATH = _REPO_ROOT / "config" / "grocery_spec.json"
try:
    with _GROCERY_SPEC_PATH.open("r", encoding="utf-8") as _f:
        GROCERY_SPEC: dict = {k: v for k, v in _json.load(_f).items() if not k.startswith("_")}
    print(f"[SPEC] loaded {len(GROCERY_SPEC)} class entries from {_GROCERY_SPEC_PATH.name}")
except Exception as _e:
    GROCERY_SPEC = {}
    print(f"[SPEC] warn: could not load grocery_spec.json: {_e}")


@dataclass
class ObjectInfo:
    det_index: int
    class_name: str
    confidence: float
    color: str
    det: YOLODetection
    points_cam: np.ndarray
    points_robot: np.ndarray
    cand_dbg: CandidateDebug


@dataclass
class PlacedObject:
    info: ObjectInfo
    target_xy: np.ndarray
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D
    object_i: int
    target_z_mm: float = 0.0
    planner_notes: str = ""


@dataclass
class DemoState:
    objects: list[ObjectInfo]
    placed: list[PlacedObject] = field(default_factory=list)
    next_target: CandidateDebug | None = None


@dataclass
class PlacementComputation:
    target_xy: np.ndarray
    target_z_mm: float
    raw_box: AxisAlignedBox3D
    padded_box: AxisAlignedBox3D
    attempts: list[PlacementAttempt3D]
    planner_notes: str


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
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], [c[a, 2], c[b, 2]], color=color, alpha=alpha_edge, linewidth=1.1, linestyle=ls)
    if alpha_face > 0:
        fc = Poly3DCollection(_box_faces(c), alpha=alpha_face)
        fc.set_facecolor(color)
        ax.add_collection3d(fc)


def _best_candidate_config() -> BestCandidateConfig:
    return BestCandidateConfig(
        require_positive_platform_xy=False,
        platform_min_x_mm=0.0,
        platform_min_y_mm=0.0,
        workspace_x_min_mm=-9999.0,
        workspace_x_max_mm=9999.0,
        workspace_y_min_mm=-9999.0,
        workspace_y_max_mm=9999.0,
        use_robot_reach_check=False,
        require_soft_pose_safe=False,
        center_gate_enabled=False,
        cluster_gate_enabled=False,
        reject_placed_overlap=False,
        placed_overlap_margin_mm=0.0,
        min_volume_mm3=1.0,
        max_volume_mm3=3_000_000.0,
    )


def _select_next(state: DemoState) -> CandidateDebug | None:
    picked_indices = {p.info.det_index for p in state.placed}
    remaining = [o.cand_dbg for o in state.objects if o.det_index not in picked_indices]
    if not remaining:
        return None
    dummy_survey = SurveyState([], [], [], remaining, None, [], 0)
    result = choose_best_candidate(dummy_survey, config=_best_candidate_config(), robot=None, placed_boxes=[p.padded_box for p in state.placed])
    return result.selected


def _bag_origin_xy_mm() -> np.ndarray:
    return np.array([float(BAG_CENTER_XY_MM[0] - BAG_WIDTH_MM / 2.0), float(BAG_CENTER_XY_MM[1] - BAG_DEPTH_MM / 2.0)], dtype=np.float64)


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


def _placed_boxes_local(state: DemoState) -> tuple[list[AxisAlignedBox3D], list[AxisAlignedBox3D]]:
    return (
        [_robot_box_to_bag_local(p.raw_box) for p in state.placed],
        [_robot_box_to_bag_local(p.padded_box) for p in state.placed],
    )


def _compute_place_target(state: DemoState, cand_dbg: CandidateDebug) -> PlacementComputation:
    class_name = getattr(cand_dbg.candidate.yolo, "class_name", "Unknown")
    raw_source_box = aabb_from_object_candidate(cand_dbg.candidate, default_label=class_name)
    padded_source_box = _make_padded_box_from_raw_box(raw_source_box)
    placed_raw_local, placed_padded_local = _placed_boxes_local(state)
    plan = plan_bag_local_aabb_placement(
        item_label=class_name,
        raw_size_xyz_mm=raw_source_box.size_xyz_mm,
        padding_min_xyz_mm=raw_source_box.min_xyz_mm - padded_source_box.min_xyz_mm,
        padding_max_xyz_mm=padded_source_box.max_xyz_mm - raw_source_box.max_xyz_mm,
        bag_size_xyz_mm=_bag_size_xyz_mm(),
        placed_raw_boxes_local=placed_raw_local,
        placed_padded_boxes_local=placed_padded_local,
    )
    raw_box_robot = _bag_local_to_robot_box(plan.raw_box_local)
    padded_box_robot = _bag_local_to_robot_box(plan.padded_box_local)
    return PlacementComputation(
        target_xy=raw_box_robot.center_xyz_mm[:2].copy(),
        target_z_mm=float(raw_box_robot.min_xyz_mm[2]),
        raw_box=raw_box_robot,
        padded_box=padded_box_robot,
        attempts=plan.attempts,
        planner_notes=plan.notes,
    )


def _render_attempt_debug_pngs(*, pair_index: int, step_i: int, item_label: str, placed: list[PlacedObject], attempts: list[PlacementAttempt3D]) -> None:
    out_dir = SAVE_OUTPUT_DIR / f"pair_{pair_index:04d}" / f"step_{step_i:02d}_{item_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bag_size = _bag_size_xyz_mm()
    for attempt in attempts:
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG)
        for ax in (ax_xy, ax_xz):
            ax.set_facecolor(_PANEL_BG)

        ax_xy.set_title(
            f"Step {step_i} {item_label} attempt {attempt.attempt_i}\n"
            f"{'ACCEPT' if attempt.accepted else 'REJECT'} | {'; '.join(attempt.reasons)}",
            color="#aaddff",
            fontsize=8,
        )
        ax_xy.add_patch(mpatches.Rectangle((0, 0), bag_size[0], bag_size[1], edgecolor="white", facecolor="none", lw=1.5))
        ax_xy.set_xlim(-10, bag_size[0] + 10)
        ax_xy.set_ylim(-10, bag_size[1] + 10)
        ax_xy.set_xlabel("Bag-local X (mm)")
        ax_xy.set_ylabel("Bag-local Y (mm)")
        ax_xy.set_aspect("equal")

        ax_xz.set_title("Side view (X x Z)", color="#aaddff", fontsize=8)
        ax_xz.add_patch(mpatches.Rectangle((0, -5), bag_size[0], 5, edgecolor="white", facecolor="#334", lw=1.0))
        ax_xz.set_xlim(-10, bag_size[0] + 10)
        ax_xz.set_ylim(-25, bag_size[2] + 25)
        ax_xz.set_xlabel("Bag-local X (mm)")
        ax_xz.set_ylabel("Z (mm)")

        for placed_obj in placed:
            raw_local = _robot_box_to_bag_local(placed_obj.raw_box)
            pad_local = _robot_box_to_bag_local(placed_obj.padded_box)
            ax_xy.add_patch(
                mpatches.Rectangle(
                    raw_local.min_xyz_mm[:2],
                    raw_local.size_xyz_mm[0],
                    raw_local.size_xyz_mm[1],
                    facecolor=_DEBUG_PLACED,
                    edgecolor="white",
                    alpha=0.35,
                    lw=1.0,
                )
            )
            ax_xy.add_patch(
                mpatches.Rectangle(
                    pad_local.min_xyz_mm[:2],
                    pad_local.size_xyz_mm[0],
                    pad_local.size_xyz_mm[1],
                    facecolor="none",
                    edgecolor=_DEBUG_PLACED,
                    alpha=0.6,
                    lw=0.8,
                    ls="--",
                )
            )
            ax_xz.add_patch(
                mpatches.Rectangle(
                    (raw_local.min_xyz_mm[0], raw_local.min_xyz_mm[2]),
                    raw_local.size_xyz_mm[0],
                    raw_local.size_xyz_mm[2],
                    facecolor=_DEBUG_PLACED,
                    edgecolor="white",
                    alpha=0.35,
                    lw=1.0,
                )
            )

        candidate_color = _DEBUG_ACCEPT if attempt.accepted else _DEBUG_REJECT
        ax_xy.add_patch(
            mpatches.Rectangle(
                attempt.raw_box_local.min_xyz_mm[:2],
                attempt.raw_box_local.size_xyz_mm[0],
                attempt.raw_box_local.size_xyz_mm[1],
                facecolor=candidate_color,
                edgecolor="white",
                alpha=0.55,
                lw=1.2,
            )
        )
        ax_xy.add_patch(
            mpatches.Rectangle(
                attempt.padded_box_local.min_xyz_mm[:2],
                attempt.padded_box_local.size_xyz_mm[0],
                attempt.padded_box_local.size_xyz_mm[1],
                facecolor="none",
                edgecolor=candidate_color,
                alpha=0.9,
                lw=1.0,
                ls="--",
            )
        )
        ax_xz.add_patch(
            mpatches.Rectangle(
                (attempt.raw_box_local.min_xyz_mm[0], attempt.raw_box_local.min_xyz_mm[2]),
                attempt.raw_box_local.size_xyz_mm[0],
                attempt.raw_box_local.size_xyz_mm[2],
                facecolor=candidate_color,
                edgecolor="white",
                alpha=0.55,
                lw=1.2,
            )
        )
        ax_xz.add_patch(
            mpatches.Rectangle(
                (attempt.padded_box_local.min_xyz_mm[0], attempt.padded_box_local.min_xyz_mm[2]),
                attempt.padded_box_local.size_xyz_mm[0],
                attempt.padded_box_local.size_xyz_mm[2],
                facecolor="none",
                edgecolor=candidate_color,
                alpha=0.9,
                lw=1.0,
                ls="--",
            )
        )

        plt.tight_layout()
        out_path = out_dir / f"attempt_{attempt.attempt_i:03d}_{'accept' if attempt.accepted else 'reject'}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def _assert_no_3d_intersections(placed: list[PlacedObject]) -> None:
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            if aabbs_intersect_3d(a.raw_box, b.raw_box):
                raise AssertionError(f"raw-box intersection between {a.info.class_name} and {b.info.class_name}")


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
                cand_dbg=dbg,
            )
        )
        print(f"  [OK {idx}] {det.class_name}  pts={len(pts_cam)}")
    return objects


def _simulate_full_plan(objects: list[ObjectInfo], *, pair_index: int) -> DemoState:
    state = DemoState(objects=objects)
    state.next_target = _select_next(state)
    while state.next_target is not None:
        nxt = state.next_target
        info = next((o for o in state.objects if o.cand_dbg is nxt), None)
        if info is None:
            raise RuntimeError("next target not found in object list")
        placement = _compute_place_target(state, nxt)
        _render_attempt_debug_pngs(
            pair_index=pair_index,
            step_i=len(state.placed) + 1,
            item_label=info.class_name,
            placed=state.placed,
            attempts=placement.attempts,
        )
        placed = PlacedObject(
            info=info,
            target_xy=placement.target_xy,
            raw_box=placement.raw_box,
            padded_box=placement.padded_box,
            object_i=len(state.placed) + 1,
            target_z_mm=placement.target_z_mm,
            planner_notes=placement.planner_notes,
        )
        state.placed.append(placed)
        _assert_no_3d_intersections(state.placed)
        state.next_target = _select_next(state)
    return state


class PackingDemo:
    def __init__(self, state: DemoState, pair_index: int, rect_left: np.ndarray):
        self.state = state
        self.pair_index = pair_index
        self.rect_left = rect_left
        self.state.next_target = _select_next(self.state)

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
        nxt = self.state.next_target
        if nxt is None:
            print("[DEMO] all objects placed")
            return
        info = next((o for o in self.state.objects if o.cand_dbg is nxt), None)
        if info is None:
            return
        try:
            placement = _compute_place_target(self.state, nxt)
            _render_attempt_debug_pngs(
                pair_index=self.pair_index,
                step_i=len(self.state.placed) + 1,
                item_label=info.class_name,
                placed=self.state.placed,
                attempts=placement.attempts,
            )
            placed = PlacedObject(
                info=info,
                target_xy=placement.target_xy,
                raw_box=placement.raw_box,
                padded_box=placement.padded_box,
                object_i=len(self.state.placed) + 1,
                target_z_mm=placement.target_z_mm,
                planner_notes=placement.planner_notes,
            )
        except Exception as exc:
            print(f"[DEMO] placement failed: {exc}")
            traceback.print_exc()
            return

        self.state.placed.append(placed)
        _assert_no_3d_intersections(self.state.placed)
        self.state.next_target = _select_next(self.state)
        self._render()
        print(
            f"[DEMO] placed #{placed.object_i} {info.class_name} "
            f"at ({placement.target_xy[0]:.1f},{placement.target_xy[1]:.1f},{placement.target_z_mm:.1f}) "
            f"attempts={len(placement.attempts)}"
        )

    def _step_backward(self):
        if not self.state.placed:
            return
        removed = self.state.placed.pop()
        self.state.next_target = removed.info.cand_dbg
        self._render()
        print(f"[DEMO] undo: removed object #{removed.object_i} ({removed.info.class_name})")

    def _reset(self):
        self.state.placed.clear()
        self.state.next_target = _select_next(self.state)
        self._render()
        print("[DEMO] reset")

    def _save(self):
        SAVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = SAVE_OUTPUT_DIR / f"packing_demo_{self.pair_index:04d}_step{len(self.state.placed):02d}.png"
        self.fig.savefig(str(out), dpi=110, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        print(f"[SAVE] {out}")

    def _render(self):
        for ax in (self.ax_plat, self.ax_bag):
            ax.cla()

        picked_indices = {p.info.det_index for p in self.state.placed}
        nxt_info = next((o for o in self.state.objects if self.state.next_target is not None and o.cand_dbg is self.state.next_target), None)

        ax = self.ax_plat
        ax.set_facecolor(_PANEL_BG)
        ax.set_aspect("equal")
        ax.set_title(
            f"Platform - pair {self.pair_index:04d}  ({len(self.state.placed)}/{len(self.state.objects)} placed)\n"
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
            is_next = obj is nxt_info
            alpha = 0.25 if is_picked else 0.85
            lw = 3.5 if is_next else 1.5
            ls = "-" if not is_picked else "--"
            rect = mpatches.FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=2", edgecolor=obj.color, facecolor=obj.color, linewidth=lw, linestyle=ls, alpha=0.18 if not is_picked else 0.06)
            ax.add_patch(rect)
            border = mpatches.FancyBboxPatch((x1, y1), x2 - x1, y2 - y1, boxstyle="round,pad=2", edgecolor=obj.color, facecolor="none", linewidth=lw, linestyle=ls, alpha=alpha)
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

        ax.tick_params(colors="#666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.text(0.5, -0.04, f"Placed: {len(self.state.placed)} | Next: {nxt_info.class_name if nxt_info else 'none'} | Remaining: {len(self.state.objects) - len(self.state.placed)}", transform=ax.transAxes, color="#cccccc", fontsize=8, ha="center", va="top")

        ax3 = self.ax_bag
        ax3.set_facecolor(_PANEL_BG)
        ax3.set_title(
            f"Bag ({len(self.state.placed)} placed)\nsolid = raw AABB   dashed = padded AABB",
            color="#aaddff",
            fontsize=9,
            pad=4,
        )
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
            if len(placed.info.points_robot) > 0:
                samp = placed.info.points_robot if len(placed.info.points_robot) <= MAX_PTS_DISP else placed.info.points_robot[np.random.choice(len(placed.info.points_robot), MAX_PTS_DISP, replace=False)]
                centroid_xy = np.mean(samp[:, :2], axis=0)
                z_min = float(samp[:, 2].min())
                shift = np.array([placed.target_xy[0] - centroid_xy[0], placed.target_xy[1] - centroid_xy[1], placed.target_z_mm - z_min])
                shifted = samp + shift
                ax3.scatter(shifted[:, 0], shifted[:, 1], shifted[:, 2], c=[placed.info.color], s=0.6, alpha=0.45)
            _draw_box_3d(ax3, placed.raw_box, placed.info.color, alpha_face=0.15, ls="-")
            _draw_box_3d(ax3, placed.padded_box, placed.info.color, alpha_face=0.0, ls="--", alpha_edge=0.40)
            ax3.text(placed.raw_box.center_xyz_mm[0], placed.raw_box.center_xyz_mm[1], placed.raw_box.max_xyz_mm[2] + 6, f"#{placed.object_i} {placed.info.class_name[:7]}", color="white", fontsize=6, ha="center", va="bottom")
            all_pts_for_scale.extend(_corners(placed.padded_box))

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(TRAINING_IMAGES_DIR))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--validate-all", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)

    pairs_dir = Path(args.images)
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
    print(f"[INIT] pair {idx:04d}  left={lp.name}")

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
        validated = _simulate_full_plan(objects, pair_index=idx)
        _assert_no_3d_intersections(validated.placed)
        print(f"[VALIDATION] placed {len(validated.placed)}/{len(objects)} objects with no 3D raw-AABB intersections")

    if args.no_gui:
        return 0

    print("  -> place next best object")
    print("  <- undo last placement")
    print("  R = reset")
    print("  S = save PNG")
    print("  Q = quit")
    state = DemoState(objects=objects)
    demo = PackingDemo(state, pair_index=idx, rect_left=rect_l)
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
