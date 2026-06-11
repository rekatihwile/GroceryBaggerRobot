from __future__ import annotations

"""Offline per-survey pick/place planner visualizer for saved run snapshots.

For each survey row in a run snapshot, this script reconstructs:
1. the current bag state from all prior placed items,
2. the current platform candidate set from saved stereo + disparity,
3. overhead segmentation + stereo/overhead matching,
4. the deterministic candidate ranking and bag placement target.

It then writes five planner-oriented figures per survey step.
"""

from dataclasses import dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from config.place.place_config import DEFAULT_PLACE
from config.robot_config import ROBOT_CONFIG
from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from planning.aabb_utils import AxisAlignedBox3D, PaddedBox3D, aabb_from_object_candidate, make_aabb_from_center_size, make_aabb_from_min_max, pad_aabb
from planning.autonomous_planning_sequences import select_candidate_for_state as select_candidate_for_sequence
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    run_output_dir,
    save_figure_bundle,
    safe_name,
)
from scripts.autonomous_best_candidate import BestCandidateConfig
from scripts.autonomous_missed_pick_recovery import (
    PLACE_BAG_LOCAL_HEIGHT_MM,
    PLACE_PLANNING_SEQUENCE_NAME,
    _best_candidate_config,
    _planning_runtime,
    _planning_sequence_context,
)
from scripts.pick_one_place_one import _configure_modules, _load_place_scene
from test_calibration_bundle_live_stereo_z_pickplace import load_bundle, load_stereo_calibration
from vision.burst_tracking import BurstFrame, DetectionObservation, DetectionTrack
from vision.object_geometry import build_object_candidate
from vision.pick_candidate_builder import CandidateDebug, SurveyState, colorize_disparity
from vision.pick_phi_resolver import resolve_pick_phi
from vision.pick_xy_resolver import apply_xy_blend, match_overhead_xy_to_candidate
from vision.pick_z_resolver import resolve_robust_object_z
from vision.pointcloud import cam_points_to_robot_xyz
from vision.yolo_segmenter import YOLODetection, YOLOSegmenter


IDE_DEFAULT_RUN_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_185839"
IDE_DEFAULT_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BAG_SCENE_NAME = "New Bag Test"

BG_COLOR = "#080b14"
PANEL_BG = "#0f172a"
GRID_COLOR = "#d1d5db"
PLATFORM_EDGE = "#5ca37e"
PLATFORM_FILL = "#1f3b2f"
BOUNDS_COLOR = "#f2c66d"
BAG_EDGE = "#e5e7eb"
BAG_FILL = "#64748b"
WINNER_COLOR = "#22c55e"
GHOST_COLOR = "#facc15"
INVALID_COLOR = "#ef4444"

VIEW_ELEV_DEG = 24.0
VIEW_AZIM_DEG = -58.0
POINT_SIZE_3D = 1.0
POINT_ALPHA_3D = 0.42
MAX_PTS_OBJECT = 2000
MAX_PTS_OBJECT_TOPDOWN = 1200
CLEAN_EXPORTS = True

PALETTE = [
    "#38bdf8",
    "#fb923c",
    "#f87171",
    "#a78bfa",
    "#facc15",
    "#2dd4bf",
    "#f472b6",
    "#60a5fa",
    "#f59e0b",
    "#cbd5e1",
    "#f43f5e",
]

CLASS_COLOR_OVERRIDES: dict[str, str] = {
    "Bandaid": "#3b82f6",
    "Chex Mix": "#f59e0b",
    "Pringles": "#fb923c",
    "Carmex Lip Balm": "#a78bfa",
    "Burts Bees": "#facc15",
    "Dove Deodorant": "#f87171",
    "Lays Chips": "#38bdf8",
    "Paper Towels": "#94a3b8",
}

HARDCODED_SURVEY_CLASS_SEQUENCE: dict[int, tuple[str, ...]] = {
    1: ("Bandaid", "Chex Mix", "Pringles"),
    2: ("Chex Mix", "Pringles"),
    3: ("Carmex Lip Balm", "Burts Bees", "Pringles", "Dove Deodorant"),
}


@dataclass
class BagVolume:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        return self.y_max - self.y_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min


@dataclass
class ManifestRow:
    manifest_row: int
    object_i: int
    class_name: str
    place_result: str
    pick_xy_mm: np.ndarray | None
    place_xy_mm: np.ndarray | None
    pick_phi_deg: float
    place_phi_deg: float
    destination_surface_z_mm: float | None
    raw_box_center_xyz_mm: np.ndarray | None
    raw_box_size_xyz_mm: np.ndarray | None
    padded_box_center_xyz_mm: np.ndarray | None
    padded_box_size_xyz_mm: np.ndarray | None
    stereo_left: str | None
    stereo_right: str | None
    overhead: str | None
    disparity: str | None
    points_cam: str | None


@dataclass
class PlacedItem:
    manifest_row: int
    order: int
    class_name: str
    color: str
    raw_box_bag: AxisAlignedBox3D
    padded_box_bag: AxisAlignedBox3D
    source_raw_box: AxisAlignedBox3D
    points_pick_robot: np.ndarray
    points_bag_robot: np.ndarray
    colors_rgb: np.ndarray | None


@dataclass
class PlatformCandidate:
    index: int
    class_name: str
    color: str
    dbg: CandidateDebug
    matched_overhead_det: YOLODetection | None
    points_robot: np.ndarray
    colors_rgb: np.ndarray | None
    raw_box: AxisAlignedBox3D
    padded_box: PaddedBox3D
    crop_rgb: np.ndarray
    overlay_box_center: np.ndarray
    overlay_box_size: np.ndarray


@dataclass
class SurveyPlanningStep:
    row: ManifestRow
    placed_before: list[PlacedItem]
    platform_candidates: list[PlatformCandidate]
    survey_state: SurveyState
    selection: Any
    winner_index: int | None
    valid_candidate_indices: list[int]
    planner_target: Any | None
    executed_target: Any | None
    reconciliation_mode: str
    reconciliation_note: str


@dataclass
class ReconciledPlacementTarget:
    target_xy_mm: np.ndarray
    target_phi_deg: float
    footprint_clearance_mm: float
    aabb_clearance_mm: float
    fit_clearance_mm: float
    nudge_xy_mm: np.ndarray
    can_place: bool
    reason: str
    planner_score: float | None
    planner_yaw_deg: float | None
    planner_layer_z_mm: float | None
    future_placeable_count: int | None
    future_total_count: int | None
    support_box_index: int | None
    support_top_z_mm: float | None
    source: str


def _candidate_display_index(candidate: PlatformCandidate) -> int:
    return int(candidate.index)


class _OfflineRobot:
    def fk(self):
        return 0.0, 0.0, 0.0, 0.0

    def check_cartesian_pose_safe(self, x, y, z):
        return True, "offline_snapshot"

    @property
    def cfg(self):
        return ROBOT_CONFIG


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _finite_vec(value: Any, n: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < n:
        return None
    arr = arr[:n]
    return arr if np.all(np.isfinite(arr)) else None


def _resolve_default_run_dir() -> Path:
    if IDE_DEFAULT_RUN_DIR is not None:
        return IDE_DEFAULT_RUN_DIR.resolve()
    runs_root = (_REPO_ROOT / "data" / "run_snapshots").resolve()
    candidates = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run_* directories found in {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_default_out_dir(run_dir: Path) -> Path:
    return run_output_dir(run_dir)


def _parse_pair_index(path_like: str | None) -> int | None:
    if not path_like:
        return None
    m = re.search(r"(\d+)", Path(path_like).name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _class_colors(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        if name not in out:
            out[name] = CLASS_COLOR_OVERRIDES.get(name, PALETTE[len(out) % len(PALETTE)])
    return out


def _survey_class_sequence(row: ManifestRow) -> tuple[str, ...] | None:
    pair_index = _parse_pair_index(row.stereo_left) or row.manifest_row
    return HARDCODED_SURVEY_CLASS_SEQUENCE.get(int(pair_index))


def _survey_class_allowlist(row: ManifestRow) -> set[str] | None:
    seq = _survey_class_sequence(row)
    return set(seq) if seq else None


def _load_image_bgr(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _load_image_rgb(path: Path | None) -> np.ndarray | None:
    img = _load_image_bgr(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _load_disparity(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "disparity" in data.files:
            return np.asarray(data["disparity"], dtype=np.float64)
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2:
                return arr.astype(np.float64)
    return None


def _sample_point_colors_rgb(image_rgb: np.ndarray | None, uv_px: np.ndarray | None, n: int) -> np.ndarray | None:
    if image_rgb is None or uv_px is None or len(uv_px) != n:
        return None
    h, w = image_rgb.shape[:2]
    valid = np.all(np.isfinite(uv_px[:, :2]), axis=1)
    if not np.any(valid):
        return None
    safe_uv = np.where(valid[:, None], uv_px[:, :2], 0.0)
    xs = np.clip(np.rint(safe_uv[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(safe_uv[:, 1]).astype(np.int32), 0, h - 1)
    return np.clip(image_rgb[ys, xs, :3], 0.0, 1.0).astype(np.float32)


def _project_cam_bundle_to_uv(points_cam: np.ndarray, stereo_calib: dict[str, np.ndarray]) -> np.ndarray | None:
    """Back-project saved bundle-convention camera points into rectified-left pixels."""
    if "projection_left_rectified" not in stereo_calib:
        return None
    pts = np.asarray(points_cam, dtype=np.float64).copy().reshape(-1, 3)
    if len(pts) == 0:
        return None

    # Saved run-snapshot points use the calibration-bundle convention from the
    # stereo pipeline: rectified coordinates with Z sign-flipped.
    pts[:, 2] *= -1.0
    if "rectification_left" in stereo_calib:
        r_left = np.asarray(stereo_calib["rectification_left"], dtype=np.float64)
        pts = pts @ r_left.T

    P = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    z = pts[:, 2]
    good = np.isfinite(z) & (z > 1.0)
    z_safe = np.where(good, z, 1.0)
    u = np.where(good, fx * pts[:, 0] / z_safe + cx, np.nan)
    v = np.where(good, fy * pts[:, 1] / z_safe + cy, np.nan)
    return np.column_stack([u, v])


def _sample_points(points_xyz: np.ndarray, colors: np.ndarray | None, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    n = len(points_xyz)
    if n <= max_n:
        return points_xyz, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return points_xyz[idx], (colors[idx] if colors is not None else None)


def _box_corners(box: AxisAlignedBox3D) -> np.ndarray:
    mn, mx = box.min_xyz_mm, box.max_xyz_mm
    return np.array(
        [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def _draw_box_3d(
    ax,
    box: AxisAlignedBox3D,
    *,
    color: str,
    face_alpha: float = 0.0,
    edge_alpha: float = 0.9,
    linewidth: float = 1.2,
    linestyle: str = "-",
) -> None:
    corners = _box_corners(box)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    if face_alpha > 0.0:
        faces = [[corners[i] for i in face] for face in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5])]
        poly = Poly3DCollection(faces, alpha=face_alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def _set_equal_3d(ax, points: np.ndarray, margin_frac: float = 0.08) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ctr = 0.5 * (mn + mx)
    span = max(float(np.max(mx - mn)), 1.0)
    half = 0.5 * span * (1.0 + margin_frac)
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _style_3d(ax, *, minimal: bool = False) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.xaxis.pane.set_facecolor((0.82, 0.84, 0.88, 0.10))
    ax.yaxis.pane.set_facecolor((0.82, 0.84, 0.88, 0.10))
    ax.zaxis.pane.set_facecolor((0.82, 0.84, 0.88, 0.10))
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    if minimal or CLEAN_EXPORTS:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
        ax.set_xlabel("Robot X (mm)")
        ax.set_ylabel("Robot Y (mm)")
        ax.set_zlabel("Robot Z (mm)")
    ax.view_init(elev=VIEW_ELEV_DEG, azim=VIEW_AZIM_DEG)


def _load_manifest_rows(run_dir: Path) -> list[ManifestRow]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("objects") or []
    out: list[ManifestRow] = []
    for i, row in enumerate(rows, start=1):
        out.append(
            ManifestRow(
                manifest_row=i,
                object_i=int(row.get("object_i") or i),
                class_name=str(row.get("class_name") or row.get("detection_class") or "object"),
                place_result=str(row.get("place_result") or ""),
                pick_xy_mm=_finite_vec(row.get("pick_xy_mm"), 2),
                place_xy_mm=_finite_vec(row.get("place_xy_mm"), 2),
                pick_phi_deg=float(row.get("pick_phi_deg") or 0.0),
                place_phi_deg=float(row.get("place_phi_deg") or 0.0),
                destination_surface_z_mm=_finite_float(row.get("destination_surface_z_mm")),
                raw_box_center_xyz_mm=_finite_vec(row.get("raw_box_center_xyz_mm"), 3),
                raw_box_size_xyz_mm=_finite_vec(row.get("raw_box_size_xyz_mm"), 3),
                padded_box_center_xyz_mm=_finite_vec(row.get("padded_box_center_xyz_mm"), 3),
                padded_box_size_xyz_mm=_finite_vec(row.get("padded_box_size_xyz_mm"), 3),
                stereo_left=row.get("stereo_left"),
                stereo_right=row.get("stereo_right"),
                overhead=row.get("overhead"),
                disparity=row.get("disparity"),
                points_cam=row.get("points_cam"),
            )
        )
    return out


def _load_bag() -> BagVolume:
    bag_height = float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", PLACE_BAG_LOCAL_HEIGHT_MM))
    try:
        zones = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
        scene = zones[BAG_SCENE_NAME]
        center = _finite_vec(scene.get("center_xy_mm"), 2)
        width = _finite_float(scene.get("width_mm"))
        depth = _finite_float(scene.get("depth_mm"))
        surface_z = _finite_float(scene.get("surface_z_mm"))
        if center is None or width is None or depth is None or surface_z is None:
            raise ValueError("invalid bag scene")
        return BagVolume(
            x_min=float(center[0] - 0.5 * width),
            x_max=float(center[0] + 0.5 * width),
            y_min=float(center[1] - 0.5 * depth),
            y_max=float(center[1] + 0.5 * depth),
            z_min=float(surface_z),
            z_max=float(surface_z + bag_height),
        )
    except Exception:
        return BagVolume(x_min=80.0, x_max=340.0, y_min=680.0, y_max=830.0, z_min=-175.0, z_max=-175.0 + bag_height)


def _make_bag_raw_box(row: ManifestRow) -> AxisAlignedBox3D:
    if row.raw_box_size_xyz_mm is None:
        raise ValueError("row missing raw box size")
    if row.place_xy_mm is None or row.destination_surface_z_mm is None:
        raise ValueError("row missing placed pose")
    center = np.array(
        [
            float(row.place_xy_mm[0]),
            float(row.place_xy_mm[1]),
            float(row.destination_surface_z_mm) + 0.5 * float(row.raw_box_size_xyz_mm[2]),
        ],
        dtype=np.float64,
    )
    return make_aabb_from_center_size(center, row.raw_box_size_xyz_mm, label=f"{row.class_name}_placed_raw")


def _shift_points_to_box(
    points_robot: np.ndarray,
    target_box: AxisAlignedBox3D,
    colors: np.ndarray | None = None,
    *,
    clip_tol_mm: float | None = 4.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    centroid_xy = np.mean(pts[:, :2], axis=0)
    min_z = float(np.min(pts[:, 2]))
    shift = np.array(
        [
            float(target_box.center_xyz_mm[0] - centroid_xy[0]),
            float(target_box.center_xyz_mm[1] - centroid_xy[1]),
            float(target_box.min_xyz_mm[2] - min_z),
        ],
        dtype=np.float64,
    )
    shifted = pts + shift.reshape(1, 3)
    if clip_tol_mm is None:
        inside = np.ones(len(shifted), dtype=bool)
    else:
        tol = float(clip_tol_mm)
        inside = np.all(
            (shifted >= target_box.min_xyz_mm.reshape(1, 3) - tol)
            & (shifted <= target_box.max_xyz_mm.reshape(1, 3) + tol),
            axis=1,
        )
    out_colors = None
    if colors is not None and len(colors) == len(pts):
        out_colors = colors[inside]
    return shifted[inside], out_colors


def _derive_overlay_box(points_robot: np.ndarray, floor_size: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    size = hi - lo
    if floor_size is not None:
        size = np.maximum(size, np.asarray(floor_size, dtype=np.float64))
    size = np.maximum(size + np.array([4.0, 4.0, 4.0], dtype=np.float64), np.array([8.0, 8.0, 8.0], dtype=np.float64))
    return 0.5 * (lo + hi), size


def _crop_detection_rgb(image_bgr: np.ndarray, det: YOLODetection, pad_px: int = 18) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w, x2 + pad_px)
    y2 = min(h, y2 + pad_px)
    crop = image_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.asarray(det.mask, dtype=np.uint8)
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    crop_mask = mask[y1:y2, x1:x2].astype(bool)
    masked = crop.copy()
    masked[~crop_mask] = (0.35 * masked[~crop_mask]).astype(np.uint8)
    return cv2.cvtColor(masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _segment_overhead_for_row(yolo: YOLOSegmenter, overhead_bgr: np.ndarray | None, row: ManifestRow) -> list[YOLODetection]:
    if overhead_bgr is None:
        return []
    return yolo.segment(overhead_bgr)


def _make_yolo_segmenter() -> YOLOSegmenter:
    weights = _REPO_ROOT / "yolo_weights" / "full_data.pt"
    fallback = _REPO_ROOT / "Validate_Only_100_Training_Best.pt"
    return YOLOSegmenter(
        weights_path=str(weights),
        fallback_weights_path=str(fallback),
        use_half=False,
        warmup_enabled=False,
        retina_masks=True,
        conf=0.30,
    )


def _build_placed_items(
    rows: list[ManifestRow],
    run_dir: Path,
    color_by_class: dict[str, str],
    bundle: dict[str, np.ndarray],
    stereo_calib: dict[str, np.ndarray],
) -> list[PlacedItem]:
    items: list[PlacedItem] = []
    for row in rows:
        if row.place_result.lower().strip() not in ("placed", "success", "ok"):
            continue
        if row.raw_box_center_xyz_mm is None or row.raw_box_size_xyz_mm is None or not row.points_cam:
            continue
        pc_path = run_dir / row.points_cam
        if not pc_path.exists():
            continue
        try:
            with np.load(pc_path, allow_pickle=False) as data:
                points_cam = np.asarray(data["points_cam"], dtype=np.float64).reshape(-1, 3)
            stereo_left_rgb = _load_image_rgb(run_dir / row.stereo_left) if row.stereo_left else None
            point_uv = None
            with np.load(pc_path, allow_pickle=False) as data:
                for key in ("uv", "uv_px", "point_uv_px", "points_uv", "point_uv"):
                    if key in data.files:
                        uv = np.asarray(data[key], dtype=np.float64)
                        if uv.ndim == 2 and uv.shape[1] >= 2:
                            point_uv = uv[:, :2]
                            break
            if point_uv is None:
                point_uv = _project_cam_bundle_to_uv(points_cam, stereo_calib)
            colors = _sample_point_colors_rgb(stereo_left_rgb, point_uv, len(points_cam))
            points_robot = np.asarray(cam_points_to_robot_xyz(points_cam, bundle), dtype=np.float64).reshape(-1, 3)
            finite = np.all(np.isfinite(points_robot), axis=1)
            points_robot = points_robot[finite]
            if colors is not None and len(colors) == len(finite):
                colors = colors[finite]
            source_raw = make_aabb_from_center_size(row.raw_box_center_xyz_mm, row.raw_box_size_xyz_mm, label=f"{row.class_name}_platform")
            placed_raw = _make_bag_raw_box(row)
            if row.padded_box_center_xyz_mm is not None and row.padded_box_size_xyz_mm is not None:
                placed_padded_box = make_aabb_from_center_size(
                    row.padded_box_center_xyz_mm,
                    row.padded_box_size_xyz_mm,
                    label=f"{row.class_name}_placed_padded_manifest",
                )
            else:
                placed_padded_box = pad_aabb(
                    placed_raw,
                    float(DEFAULT_PLACE.PAD_X_MM),
                    float(DEFAULT_PLACE.PAD_Y_MM),
                    float(DEFAULT_PLACE.PAD_Z_MM),
                ).padded_box
            points_pick, colors_pick = _shift_points_to_box(points_robot, source_raw, colors)
            points_bag, colors_bag = _shift_points_to_box(points_robot, placed_raw, colors)
            items.append(
                PlacedItem(
                    manifest_row=row.manifest_row,
                    order=len(items) + 1,
                    class_name=row.class_name,
                    color=color_by_class[row.class_name],
                    raw_box_bag=placed_raw,
                    padded_box_bag=placed_padded_box,
                    source_raw_box=source_raw,
                    points_pick_robot=points_pick,
                    points_bag_robot=points_bag,
                    colors_rgb=colors_bag if colors_bag is not None else colors_pick,
                )
            )
        except Exception as exc:
            print(f"[PLACED WARN] row {row.manifest_row}: {exc}")
    return items


def _build_platform_candidates(
    row: ManifestRow,
    run_dir: Path,
    yolo: YOLOSegmenter,
    disparity: np.ndarray,
    stereo_calib: dict[str, np.ndarray],
    bundle: dict[str, np.ndarray],
) -> tuple[list[PlatformCandidate], SurveyState]:
    left_path = run_dir / row.stereo_left if row.stereo_left else None
    overhead_path = run_dir / row.overhead if row.overhead else None
    left_bgr = _load_image_bgr(left_path)
    if left_bgr is None:
        raise FileNotFoundError(f"missing stereo left image for row {row.manifest_row}")
    overhead_bgr = _load_image_bgr(overhead_path)
    allowed_classes = _survey_class_allowlist(row)
    left_dets = yolo.segment(left_bgr)
    overhead_dets = _segment_overhead_for_row(yolo, overhead_bgr, row)
    if allowed_classes is not None:
        left_dets = [det for det in left_dets if str(det.class_name) in allowed_classes]
        overhead_dets = [det for det in overhead_dets if str(det.class_name) in allowed_classes]
    used_overhead_ids: set[int] = set()

    frame = BurstFrame(frame_i=0, left_rect=left_bgr, right_rect=np.zeros_like(left_bgr), stereo_tags={})
    candidates: list[PlatformCandidate] = []
    debug_candidates: list[CandidateDebug] = []
    color_by_class = _class_colors([d.class_name for d in left_dets] or ["object"])
    null_robot = _OfflineRobot()

    for idx, det in enumerate(left_dets, start=1):
        obs = DetectionObservation(frame_i=0, detection=det)
        track = DetectionTrack(track_id=idx, class_name=str(det.class_name), observations=[obs])
        try:
            from vision.pick_candidate_builder import build_candidate_from_detection

            dbg = build_candidate_from_detection(
                track=track,
                frame=frame,
                det=det,
                disparity=disparity,
                stereo_calib=stereo_calib,
                robot=null_robot,
                bundle=bundle,
                index=idx,
            )
        except Exception as exc:
            print(f"[CAND WARN] row {row.manifest_row} det {idx}: {exc}")
            dbg = None
        if dbg is None:
            continue

        matched_overhead = match_overhead_xy_to_candidate(overhead_dets, dbg, bundle)
        if matched_overhead is None:
            same_name = [
                overhead_det
                for overhead_det in overhead_dets
                if id(overhead_det) not in used_overhead_ids and str(overhead_det.class_name) == str(det.class_name)
            ]
            if len(same_name) == 1:
                matched_overhead = same_name[0]
                print(f"[MATCH FALLBACK] row {row.manifest_row} cand[{idx}] {det.class_name:<14} -> overhead {matched_overhead.class_name:<14} class-only")
        if matched_overhead is None:
            continue
        used_overhead_ids.add(id(matched_overhead))
        resolve_pick_phi(dbg, matched_overhead, bundle)
        apply_xy_blend(dbg, bundle)

        points_robot = np.asarray(cam_points_to_robot_xyz(dbg.points_cam, bundle), dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(points_robot), axis=1)
        points_robot = points_robot[finite]
        colors = _sample_point_colors_rgb(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, dbg.point_uv_px, len(dbg.points_cam))
        if colors is not None and len(colors) == len(finite):
            colors = colors[finite]
        raw_box = aabb_from_object_candidate(dbg.candidate, default_label=str(det.class_name))
        padded_box = pad_aabb(raw_box, float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), float(DEFAULT_PLACE.PAD_Z_MM))
        overlay_center, overlay_size = _derive_overlay_box(points_robot, raw_box.size_xyz_mm)
        crop_rgb = _crop_detection_rgb(left_bgr, det)

        candidate = PlatformCandidate(
            index=idx,
            class_name=str(det.class_name),
            color=color_by_class[str(det.class_name)],
            dbg=dbg,
            matched_overhead_det=matched_overhead,
            points_robot=points_robot,
            colors_rgb=colors,
            raw_box=raw_box,
            padded_box=padded_box,
            crop_rgb=crop_rgb,
            overlay_box_center=overlay_center,
            overlay_box_size=overlay_size,
        )
        candidates.append(candidate)
        debug_candidates.append(dbg)

    desired_sequence = _survey_class_sequence(row)
    if desired_sequence:
        kept_candidates: list[PlatformCandidate] = []
        kept_debug: list[CandidateDebug] = []
        remaining = list(zip(candidates, debug_candidates))
        for desired_name in desired_sequence:
            matches = [(cand, dbg) for cand, dbg in remaining if str(cand.class_name) == desired_name]
            if not matches:
                continue
            best_cand, best_dbg = max(
                matches,
                key=lambda pair: (
                    float(getattr(pair[0].matched_overhead_det, "confidence", 0.0)),
                    float(np.prod(pair[0].raw_box.size_xyz_mm)),
                ),
            )
            kept_candidates.append(best_cand)
            kept_debug.append(best_dbg)
            remaining.remove((best_cand, best_dbg))
        candidates = kept_candidates
        debug_candidates = kept_debug

    survey_state = SurveyState(
        burst_frames=[frame],
        tracks=[cand.dbg.track for cand in candidates],
        kept_tracks=[cand.dbg.track for cand in candidates],
        candidates=debug_candidates,
        overhead_frame=overhead_bgr,
        overhead_detections=overhead_dets,
        selected_index=0,
    )
    return candidates, survey_state


def _placed_box_for_planner(item: PlacedItem) -> PaddedBox3D:
    return PaddedBox3D(raw_box=item.raw_box_bag, padded_box=item.padded_box_bag, padding_xyz_mm=np.array([float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), float(DEFAULT_PLACE.PAD_Z_MM)], dtype=np.float64))


def _manifest_winner_index(row: ManifestRow, platform_candidates: list[PlatformCandidate]) -> int | None:
    if not platform_candidates:
        return None
    target_xy = row.pick_xy_mm if row.pick_xy_mm is not None else None
    best_idx = None
    best_score = None
    row_name = str(row.class_name)
    for idx, cand in enumerate(platform_candidates):
        class_penalty = 0.0 if str(cand.class_name) == row_name else 1e6
        if target_xy is not None:
            dist = float(np.linalg.norm(cand.raw_box.center_xyz_mm[:2] - np.asarray(target_xy, dtype=np.float64).reshape(2)))
        else:
            dist = float(idx)
        score = class_penalty + dist
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _manifest_executed_target(
    row: ManifestRow,
    winner: PlatformCandidate | None,
    placed_before: list[PlacedItem],
    surface_zone: dict[str, Any],
) -> ReconciledPlacementTarget | None:
    if row.place_result.lower().strip() not in ("placed", "success", "ok"):
        return None
    if row.place_xy_mm is None or row.destination_surface_z_mm is None:
        return None

    surface_z = float(surface_zone.get("surface_z_mm", row.destination_surface_z_mm))
    layer_z = float(row.destination_surface_z_mm - surface_z)
    support_idx = None
    support_top = None
    if winner is not None and layer_z > 1.0:
        target_xy = np.asarray(row.place_xy_mm, dtype=np.float64).reshape(2)
        half_xy = 0.5 * np.asarray(winner.raw_box.size_xyz_mm[:2], dtype=np.float64)
        target_min = target_xy - half_xy
        target_max = target_xy + half_xy
        best_overlap = 0.0
        for idx, item in enumerate(placed_before):
            top_z = float(item.raw_box_bag.max_xyz_mm[2])
            if abs(top_z - float(row.destination_surface_z_mm)) > 4.0:
                continue
            overlap_x = max(
                0.0,
                min(float(target_max[0]), float(item.raw_box_bag.max_xyz_mm[0]))
                - max(float(target_min[0]), float(item.raw_box_bag.min_xyz_mm[0])),
            )
            overlap_y = max(
                0.0,
                min(float(target_max[1]), float(item.raw_box_bag.max_xyz_mm[1]))
                - max(float(target_min[1]), float(item.raw_box_bag.min_xyz_mm[1])),
            )
            overlap = overlap_x * overlap_y
            if overlap > best_overlap:
                best_overlap = overlap
                support_idx = idx
                support_top = top_z

    return ReconciledPlacementTarget(
        target_xy_mm=np.asarray(row.place_xy_mm, dtype=np.float64).reshape(2).copy(),
        target_phi_deg=float(row.place_phi_deg),
        footprint_clearance_mm=float("nan"),
        aabb_clearance_mm=float("nan"),
        fit_clearance_mm=float("nan"),
        nudge_xy_mm=np.zeros(2, dtype=np.float64),
        can_place=True,
        reason="manifest_executed_placement",
        planner_score=None,
        planner_yaw_deg=float(row.place_phi_deg),
        planner_layer_z_mm=layer_z,
        future_placeable_count=None,
        future_total_count=None,
        support_box_index=support_idx,
        support_top_z_mm=support_top,
        source="manifest",
    )


def _reconcile_target(
    row: ManifestRow,
    winner: PlatformCandidate | None,
    planner_target: Any | None,
    placed_before: list[PlacedItem],
    surface_zone: dict[str, Any],
) -> tuple[ReconciledPlacementTarget | None, str, str]:
    executed = _manifest_executed_target(row, winner, placed_before, surface_zone)
    if executed is None:
        return None, "planner_only", "manifest row does not record a completed placement"
    if planner_target is None:
        return (
            executed,
            "manifest_fallback_no_replay_target",
            "the current deterministic replay has no target; the executed manifest placement is authoritative",
        )

    xy_error = float(
        np.linalg.norm(
            np.asarray(planner_target.target_xy_mm, dtype=np.float64).reshape(2)
            - executed.target_xy_mm
        )
    )
    planner_layer = float(getattr(planner_target, "planner_layer_z_mm", 0.0) or 0.0)
    layer_error = abs(planner_layer - float(executed.planner_layer_z_mm or 0.0))
    if xy_error <= 5.0 and layer_error <= 4.0:
        return (
            executed,
            "planner_replay_matches_manifest",
            f"replay target agrees with execution within {xy_error:.1f} mm XY and {layer_error:.1f} mm Z",
        )
    return (
        executed,
        "manifest_override_replay_mismatch",
        f"execution overrides replay target; delta={xy_error:.1f} mm XY, {layer_error:.1f} mm layer Z",
    )


def _plan_survey_step(
    row: ManifestRow,
    placed_before: list[PlacedItem],
    platform_candidates: list[PlatformCandidate],
    survey_state: SurveyState,
    surface_zone: dict[str, Any],
) -> SurveyPlanningStep:
    placed_boxes = [_placed_box_for_planner(item) for item in placed_before]
    base_xy = np.asarray(surface_zone["center_xy_mm"], dtype=np.float64).reshape(2)
    base_phi = float(surface_zone.get("default_phi_deg", 0.0))
    cfg: BestCandidateConfig = _best_candidate_config()
    ctx = _planning_sequence_context(
        object_i=int(row.object_i),
        robot=_OfflineRobot(),
        placed_boxes=placed_boxes,
        config=cfg,
        surface_zone=surface_zone,
        base_xy=base_xy,
        base_phi_deg=base_phi,
        target_limit=len(platform_candidates),
    )
    selection = select_candidate_for_sequence(
        str(PLACE_PLANNING_SEQUENCE_NAME),
        survey_state,
        ctx=ctx,
        runtime=_planning_runtime(),
    )

    winner_index = _manifest_winner_index(row, platform_candidates)
    valid_candidate_indices = sorted(int(i) for i, entry in selection.overlay.items() if bool(entry.can_place))
    planner_target = (
        selection.target_cache.get(winner_index)
        if winner_index is not None
        else None
    )
    winner = (
        platform_candidates[winner_index]
        if winner_index is not None and winner_index < len(platform_candidates)
        else None
    )
    executed_target, reconciliation_mode, reconciliation_note = _reconcile_target(
        row,
        winner,
        planner_target,
        placed_before,
        surface_zone,
    )
    if executed_target is not None and winner_index is not None and winner_index not in valid_candidate_indices:
        valid_candidate_indices.append(winner_index)
        valid_candidate_indices.sort()
    return SurveyPlanningStep(
        row=row,
        placed_before=placed_before,
        platform_candidates=platform_candidates,
        survey_state=survey_state,
        selection=selection,
        winner_index=winner_index,
        valid_candidate_indices=valid_candidate_indices,
        planner_target=planner_target,
        executed_target=executed_target,
        reconciliation_mode=reconciliation_mode,
        reconciliation_note=reconciliation_note,
    )


def _winner_target(step: SurveyPlanningStep) -> tuple[PlatformCandidate, Any] | None:
    if step.winner_index is None or step.winner_index >= len(step.platform_candidates):
        return None
    target = step.executed_target or step.planner_target
    if target is None:
        return None
    return step.platform_candidates[step.winner_index], target


def _display_candidate_indices(step: SurveyPlanningStep) -> list[int]:
    display: list[int] = []
    for idx, cand in enumerate(step.platform_candidates):
        decision = next((d for d in getattr(step.selection.result, "decisions", []) if d.dbg is cand.dbg), None)
        if cand.matched_overhead_det is not None or (decision is not None and bool(decision.passed)):
            display.append(idx)
    return display if display else list(range(len(step.platform_candidates)))


def _target_raw_box(candidate: PlatformCandidate, target_xy_mm: np.ndarray, target_layer_z_mm: float | None, bag: BagVolume) -> AxisAlignedBox3D:
    if target_layer_z_mm is None:
        layer_z = float(bag.z_min)
    else:
        layer_z = float(bag.z_min + target_layer_z_mm)
    min_xyz = np.array([float(target_xy_mm[0]) - 0.5 * float(candidate.raw_box.size_xyz_mm[0]), float(target_xy_mm[1]) - 0.5 * float(candidate.raw_box.size_xyz_mm[1]), layer_z], dtype=np.float64)
    max_xyz = np.array([min_xyz[0] + float(candidate.raw_box.size_xyz_mm[0]), min_xyz[1] + float(candidate.raw_box.size_xyz_mm[1]), min_xyz[2] + float(candidate.raw_box.size_xyz_mm[2])], dtype=np.float64)
    return make_aabb_from_min_max(min_xyz, max_xyz, label=f"{candidate.class_name}_planned_raw")


def _all_scene_points(step: SurveyPlanningStep, bag: BagVolume, winner_box: AxisAlignedBox3D | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for item in step.placed_before:
        chunks.append(_box_corners(item.raw_box_bag))
        chunks.append(_box_corners(item.padded_box_bag))
        if len(item.points_bag_robot):
            chunks.append(item.points_bag_robot)
    for idx in _display_candidate_indices(step):
        cand = step.platform_candidates[idx]
        chunks.append(_box_corners(cand.raw_box))
        if len(cand.points_robot):
            chunks.append(cand.points_robot)
    bag_box = make_aabb_from_center_size(
        np.array([0.5 * (bag.x_min + bag.x_max), 0.5 * (bag.y_min + bag.y_max), 0.5 * (bag.z_min + bag.z_max)], dtype=np.float64),
        np.array([bag.width, bag.depth, bag.height], dtype=np.float64),
        label="bag",
    )
    chunks.append(_box_corners(bag_box))
    if winner_box is not None:
        chunks.append(_box_corners(winner_box))
    return np.vstack(chunks) if chunks else np.zeros((1, 3), dtype=np.float64)


def _scene_overview_candidate_visible(cand: PlatformCandidate, *, min_y_mm: float = 30.0) -> bool:
    return float(cand.raw_box.center_xyz_mm[1]) >= float(min_y_mm)


def _draw_platform_bounds(ax, bounds: tuple[float, float, float, float], z_mm: float = -175.0) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    corners = np.array(
        [
            [x_min, y_min, z_mm],
            [x_max, y_min, z_mm],
            [x_max, y_max, z_mm],
            [x_min, y_max, z_mm],
        ],
        dtype=np.float64,
    )
    floor = Poly3DCollection([[corners[i] for i in [0, 1, 2, 3]]], alpha=0.08)
    floor.set_facecolor(PLATFORM_FILL)
    floor.set_edgecolor("none")
    ax.add_collection3d(floor)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], color=BOUNDS_COLOR, linewidth=1.3, alpha=0.9)
    return corners


def _draw_bag(ax, bag: BagVolume) -> AxisAlignedBox3D:
    bag_box = make_aabb_from_center_size(
        np.array([0.5 * (bag.x_min + bag.x_max), 0.5 * (bag.y_min + bag.y_max), 0.5 * (bag.z_min + bag.z_max)], dtype=np.float64),
        np.array([bag.width, bag.depth, bag.height], dtype=np.float64),
        label="bag",
    )
    corners = _box_corners(bag_box)
    floor = Poly3DCollection([[corners[i] for i in [0, 1, 2, 3]]], alpha=0.07)
    floor.set_facecolor(BAG_FILL)
    floor.set_edgecolor("none")
    ax.add_collection3d(floor)
    _draw_box_3d(ax, bag_box, color=BAG_EDGE, face_alpha=0.0, edge_alpha=0.72, linewidth=1.1, linestyle="-")
    return bag_box


def _render_scene_overview(step: SurveyPlanningStep, bag: BagVolume, platform_bounds: tuple[float, float, float, float], out_path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(10.8, 7.6), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    _draw_platform_bounds(ax, platform_bounds, z_mm=bag.z_min)
    _draw_bag(ax, bag)

    winner_info = _winner_target(step)
    winner_box = None
    if winner_info is not None:
        winner_candidate, winner_target = winner_info
        winner_box = _target_raw_box(winner_candidate, winner_target.target_xy_mm, winner_target.planner_layer_z_mm, bag)
        _draw_box_3d(ax, winner_box, color=GHOST_COLOR, face_alpha=0.06, edge_alpha=0.95, linewidth=1.6, linestyle="--")

    for item in step.placed_before:
        pts, cols = _sample_points(item.points_bag_robot, item.colors_rgb, MAX_PTS_OBJECT, 100 + item.order)
        if len(pts) and cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=POINT_SIZE_3D, alpha=POINT_ALPHA_3D, depthshade=False)
        _draw_box_3d(ax, item.raw_box_bag, color=item.color, face_alpha=0.10, edge_alpha=0.65, linewidth=1.0)
        _draw_box_3d(ax, item.padded_box_bag, color=item.color, face_alpha=0.0, edge_alpha=0.45, linewidth=0.9, linestyle="--")

    for idx in _display_candidate_indices(step):
        cand = step.platform_candidates[idx]
        if not _scene_overview_candidate_visible(cand):
            continue
        is_winner = step.winner_index == idx
        pts, cols = _sample_points(cand.points_robot, cand.colors_rgb, MAX_PTS_OBJECT, 2000 + cand.index)
        if len(pts) and cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=POINT_SIZE_3D * (1.2 if is_winner else 1.0), alpha=(0.62 if is_winner else POINT_ALPHA_3D), depthshade=False)
        _draw_box_3d(ax, cand.raw_box, color=(WINNER_COLOR if is_winner else cand.color), face_alpha=0.12 if is_winner else 0.05, edge_alpha=0.95 if is_winner else 0.55, linewidth=1.4 if is_winner else 0.9)

    scene_points = [chunk for chunk in [
        *[_box_corners(item.raw_box_bag) for item in step.placed_before],
        *[_box_corners(item.padded_box_bag) for item in step.placed_before],
        *[item.points_bag_robot for item in step.placed_before if len(item.points_bag_robot)],
        *[
            _box_corners(step.platform_candidates[idx].raw_box)
            for idx in _display_candidate_indices(step)
            if _scene_overview_candidate_visible(step.platform_candidates[idx])
        ],
        *[
            step.platform_candidates[idx].points_robot
            for idx in _display_candidate_indices(step)
            if _scene_overview_candidate_visible(step.platform_candidates[idx]) and len(step.platform_candidates[idx].points_robot)
        ],
        _box_corners(make_aabb_from_center_size(
            np.array([0.5 * (bag.x_min + bag.x_max), 0.5 * (bag.y_min + bag.y_max), 0.5 * (bag.z_min + bag.z_max)], dtype=np.float64),
            np.array([bag.width, bag.depth, bag.height], dtype=np.float64),
            label="bag",
        )),
        (_box_corners(winner_box) if winner_box is not None else None),
    ] if chunk is not None]
    _set_equal_3d(ax, np.vstack(scene_points) if scene_points else np.zeros((1, 3), dtype=np.float64), margin_frac=0.08)
    try:
        _, y_high = ax.get_ylim()
        ax.set_ylim(30.0, float(y_high))
    except Exception:
        pass
    if not CLEAN_EXPORTS:
        ax.set_title("Platform candidates + reconstructed bag state + chosen target", color="white", fontsize=12, pad=10)
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_support_topdown(step: SurveyPlanningStep, bag: BagVolume, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8), facecolor="white")
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_xlabel("Robot X (mm)")
    ax.set_ylabel("Robot Y (mm)")
    ax.add_patch(Rectangle((bag.x_min, bag.y_min), bag.width, bag.depth, facecolor="#f8fafc", edgecolor="#111827", linewidth=2.0, zorder=1))

    winner_info = _winner_target(step)
    support_box = None
    support_intersection = None
    if winner_info is not None:
        winner_candidate, winner_target = winner_info
        winner_raw = _target_raw_box(winner_candidate, winner_target.target_xy_mm, winner_target.planner_layer_z_mm, bag)
        winner_pad = pad_aabb(winner_raw, float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), float(DEFAULT_PLACE.PAD_Z_MM)).padded_box
        if getattr(winner_target, "support_box_index", None) is not None:
            support_idx = int(winner_target.support_box_index)
            if 0 <= support_idx < len(step.placed_before):
                support_box = step.placed_before[support_idx].raw_box_bag
                ix0 = max(float(support_box.min_xyz_mm[0]), float(winner_raw.min_xyz_mm[0]))
                ix1 = min(float(support_box.max_xyz_mm[0]), float(winner_raw.max_xyz_mm[0]))
                iy0 = max(float(support_box.min_xyz_mm[1]), float(winner_raw.min_xyz_mm[1]))
                iy1 = min(float(support_box.max_xyz_mm[1]), float(winner_raw.max_xyz_mm[1]))
                if ix1 > ix0 and iy1 > iy0:
                    support_intersection = Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0)
        for item in step.placed_before:
            rect = Rectangle((item.raw_box_bag.min_xyz_mm[0], item.raw_box_bag.min_xyz_mm[1]), item.raw_box_bag.size_xyz_mm[0], item.raw_box_bag.size_xyz_mm[1], facecolor=item.color, edgecolor=item.color, alpha=0.28, linewidth=1.4, zorder=2)
            ax.add_patch(rect)
        ax.add_patch(Rectangle((winner_pad.min_xyz_mm[0], winner_pad.min_xyz_mm[1]), winner_pad.size_xyz_mm[0], winner_pad.size_xyz_mm[1], facecolor="none", edgecolor=GHOST_COLOR, linewidth=2.0, linestyle=(0, (4, 3)), zorder=5))
        ax.add_patch(Rectangle((winner_raw.min_xyz_mm[0], winner_raw.min_xyz_mm[1]), winner_raw.size_xyz_mm[0], winner_raw.size_xyz_mm[1], facecolor=GHOST_COLOR, edgecolor=GHOST_COLOR, alpha=0.28, linewidth=2.2, zorder=4))
        if support_box is not None:
            ax.add_patch(Rectangle((support_box.min_xyz_mm[0], support_box.min_xyz_mm[1]), support_box.size_xyz_mm[0], support_box.size_xyz_mm[1], facecolor="none", edgecolor=WINNER_COLOR, linewidth=2.2, zorder=6))
        if support_intersection is not None:
            support_intersection.set_facecolor(WINNER_COLOR)
            support_intersection.set_alpha(0.30)
            support_intersection.set_edgecolor("none")
            support_intersection.set_zorder(7)
            ax.add_patch(support_intersection)
        if not CLEAN_EXPORTS:
            note = "Floor placement" if getattr(winner_target, "planner_layer_z_mm", 0.0) in (None, 0.0) else "Stacked placement: overlap footprint shows support"
            ax.set_title(f"Top-down deterministic placement footprint\n{note}", fontsize=12)
    else:
        for item in step.placed_before:
            rect = Rectangle((item.raw_box_bag.min_xyz_mm[0], item.raw_box_bag.min_xyz_mm[1]), item.raw_box_bag.size_xyz_mm[0], item.raw_box_bag.size_xyz_mm[1], facecolor=item.color, edgecolor=item.color, alpha=0.28, linewidth=1.4, zorder=2)
            ax.add_patch(rect)
        if not CLEAN_EXPORTS:
            ax.set_title("Top-down deterministic placement footprint\nNo valid planned placement", fontsize=12)

    ax.set_xlim(bag.x_min - 20.0, bag.x_max + 20.0)
    ax.set_ylim(bag.y_min - 20.0, bag.y_max + 20.0)
    if CLEAN_EXPORTS:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _mask_color_bgr(color_hex: str) -> tuple[int, int, int]:
    color_hex = color_hex.lstrip("#")
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return b, g, r


def _render_overhead_and_targets(step: SurveyPlanningStep, bag: BagVolume, run_dir: Path, out_path: Path, dpi: int) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.2, 6.4), facecolor="white")
    ax_l.set_facecolor("white")
    overhead_bgr = _load_image_bgr(run_dir / step.row.overhead) if step.row.overhead else None
    if overhead_bgr is None:
        overhead_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = overhead_bgr.copy()
    for idx, cand in enumerate(step.platform_candidates):
        det = cand.matched_overhead_det
        if det is None:
            continue
        mask = np.asarray(det.mask, dtype=np.uint8)
        if mask.shape != overlay.shape[:2]:
            mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.array(_mask_color_bgr(cand.color), dtype=np.uint8)
        tint = overlay.copy()
        tint[mask.astype(bool)] = color
        overlay = cv2.addWeighted(tint, 0.35, overlay, 0.65, 0.0)
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        edge = WINNER_COLOR if step.winner_index == idx else cand.color
        ax_l.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=edge, linewidth=2.0))
        if not CLEAN_EXPORTS:
            ax_l.text(x1 + 3, max(18, y1 - 6), f"{idx + 1}", color="white", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", facecolor=edge, edgecolor="none", alpha=0.95))
    ax_l.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    if not CLEAN_EXPORTS:
        ax_l.set_title("Offline overhead segmentation matched to stereo candidates")

    ax_r.set_facecolor("white")
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.grid(True, color="#e5e7eb", linewidth=0.8)
    ax_r.add_patch(Rectangle((bag.x_min, bag.y_min), bag.width, bag.depth, facecolor="#f8fafc", edgecolor="#111827", linewidth=2.0, zorder=1))
    for item in step.placed_before:
        ax_r.add_patch(Rectangle((item.raw_box_bag.min_xyz_mm[0], item.raw_box_bag.min_xyz_mm[1]), item.raw_box_bag.size_xyz_mm[0], item.raw_box_bag.size_xyz_mm[1], facecolor=item.color, edgecolor=item.color, alpha=0.20, linewidth=1.2, zorder=2))
    for idx, cand in enumerate(step.platform_candidates):
        target = (
            step.executed_target
            if step.winner_index == idx and step.executed_target is not None
            else step.selection.target_cache.get(idx)
        )
        if target is None:
            continue
        raw_box = _target_raw_box(cand, target.target_xy_mm, target.planner_layer_z_mm, bag)
        ax_r.add_patch(Rectangle((raw_box.min_xyz_mm[0], raw_box.min_xyz_mm[1]), raw_box.size_xyz_mm[0], raw_box.size_xyz_mm[1], facecolor=cand.color, edgecolor=(WINNER_COLOR if step.winner_index == idx else cand.color), alpha=0.28, linewidth=2.0 if step.winner_index == idx else 1.2, zorder=4))
        if not CLEAN_EXPORTS:
            ax_r.text(raw_box.center_xyz_mm[0], raw_box.center_xyz_mm[1], f"{idx + 1}", ha="center", va="center", color="#111827", fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85), zorder=5)
    ax_r.set_xlim(bag.x_min - 20.0, bag.x_max + 20.0)
    ax_r.set_ylim(bag.y_min - 20.0, bag.y_max + 20.0)
    if CLEAN_EXPORTS:
        ax_r.set_xticks([])
        ax_r.set_yticks([])
    else:
        ax_r.set_xlabel("Robot X (mm)")
        ax_r.set_ylabel("Robot Y (mm)")
        ax_r.set_title("Valid placement target for each current platform item")

    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _ranking_score_text(step: SurveyPlanningStep, idx: int) -> str:
    cand = step.platform_candidates[idx]
    decision = next((d for d in getattr(step.selection.result, "decisions", []) if int(getattr(d.dbg.candidate, "index", -1)) == int(cand.index)), None)
    target = (
        step.executed_target
        if step.winner_index == idx and step.executed_target is not None
        else step.selection.target_cache.get(idx)
    )
    if step.winner_index == idx and step.executed_target is not None:
        if step.reconciliation_mode == "planner_replay_matches_manifest":
            return "EXECUTED\nreplay match"
        return f"EXECUTED FALLBACK\n{step.reconciliation_mode.replace('_', ' ')}"
    if decision is None:
        return "no decision"
    if not bool(decision.passed):
        parts = list(getattr(decision, "reject_reasons", [])[:2])
        reason = "\n".join(parts) if parts else "rejected"
        return f"REJECT\n{reason}"
    if target is None:
        return f"PASS\nvol={float(getattr(decision, 'volume_cm3', 0.0)):.0f} cm3\nno valid placement"
    layer = float(target.planner_layer_z_mm or 0.0)
    floor_flag = "floor" if abs(layer) <= 1e-6 else f"layer {layer:.0f}"
    planner_score = float(target.planner_score or 0.0)
    future = getattr(target, "future_placeable_count", None)
    future_txt = "-" if future is None else str(int(future))
    return (
        f"PASS\n"
        f"{floor_flag}\n"
        f"fit={float(target.fit_clearance_mm):.1f}\n"
        f"planner={planner_score:.1f}\n"
        f"future={future_txt}"
    )


def _render_candidate_placement_thumb(ax, bag: BagVolume, step: SurveyPlanningStep, idx: int) -> None:
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(Rectangle((bag.x_min, bag.y_min), bag.width, bag.depth, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.2, zorder=1))
    for item in step.placed_before:
        ax.add_patch(
            Rectangle(
                (item.raw_box_bag.min_xyz_mm[0], item.raw_box_bag.min_xyz_mm[1]),
                item.raw_box_bag.size_xyz_mm[0],
                item.raw_box_bag.size_xyz_mm[1],
                facecolor="none",
                edgecolor="#cbd5e1",
                alpha=0.95,
                linewidth=1.6,
                zorder=2,
            )
        )
    target = (
        step.executed_target
        if step.winner_index == idx and step.executed_target is not None
        else step.selection.target_cache.get(idx)
    )
    can_place = idx in set(step.valid_candidate_indices)
    if can_place and target is not None and idx < len(step.platform_candidates):
        cand = step.platform_candidates[idx]
        raw_box = _target_raw_box(cand, target.target_xy_mm, target.planner_layer_z_mm, bag)
        ax.add_patch(Rectangle((raw_box.min_xyz_mm[0], raw_box.min_xyz_mm[1]), raw_box.size_xyz_mm[0], raw_box.size_xyz_mm[1], facecolor=cand.color, edgecolor=(WINNER_COLOR if step.winner_index == idx else cand.color), alpha=0.34, linewidth=1.5, zorder=3))
    else:
        if not CLEAN_EXPORTS:
            ax.text(0.5, 0.5, "No valid\nplacement", transform=ax.transAxes, ha="center", va="center", fontsize=10, color=INVALID_COLOR, fontweight="bold")
    ax.set_xlim(bag.x_min - 8.0, bag.x_max + 8.0)
    ax.set_ylim(bag.y_min - 8.0, bag.y_max + 8.0)


def _render_ranking_grid(step: SurveyPlanningStep, bag: BagVolume, out_path: Path, dpi: int) -> None:
    n = max(1, len(step.platform_candidates))
    fig = plt.figure(figsize=(max(10.0, 2.8 * n), 6.8), facecolor="white")
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.05], hspace=0.22, wspace=0.12)

    for idx, cand in enumerate(step.platform_candidates):
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.set_facecolor("white")
        ax_img.imshow(np.clip(cand.crop_rgb, 0.0, 1.0))
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        border_color = WINNER_COLOR if step.winner_index == idx else cand.color
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4.0 if step.winner_index == idx else 2.0)
            spine.set_edgecolor(border_color)
        if not CLEAN_EXPORTS:
            ax_img.set_title(f"{idx + 1}. {cand.class_name}", fontsize=10, color="#111827")
            ax_img.text(0.02, 0.02, _ranking_score_text(step, idx), transform=ax_img.transAxes, ha="left", va="bottom", fontsize=8, color="#111827", bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cbd5e1", alpha=0.90))

        ax_plan = fig.add_subplot(gs[1, idx])
        _render_candidate_placement_thumb(ax_plan, bag, step, idx)
        if not CLEAN_EXPORTS:
            ax_plan.set_title("Best bag target", fontsize=9)

    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_bag_after_pick(step: SurveyPlanningStep, bag: BagVolume, out_path: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(10.2, 7.4), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax)
    _draw_bag(ax, bag)

    scene_pts: list[np.ndarray] = []
    for item in step.placed_before:
        pts, cols = _sample_points(item.points_bag_robot, item.colors_rgb, MAX_PTS_OBJECT, 1000 + item.order)
        if len(pts) and cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=POINT_SIZE_3D, alpha=POINT_ALPHA_3D, depthshade=False)
            scene_pts.append(pts)
        _draw_box_3d(ax, item.raw_box_bag, color=item.color, face_alpha=0.10, edge_alpha=0.62, linewidth=1.0)
        _draw_box_3d(ax, item.padded_box_bag, color=item.color, face_alpha=0.0, edge_alpha=0.45, linewidth=0.9, linestyle="--")
        scene_pts.append(_box_corners(item.padded_box_bag))

    winner_info = _winner_target(step)
    if winner_info is not None:
        winner_candidate, winner_target = winner_info
        winner_raw = _target_raw_box(winner_candidate, winner_target.target_xy_mm, winner_target.planner_layer_z_mm, bag)
        winner_pad = pad_aabb(winner_raw, float(DEFAULT_PLACE.PAD_X_MM), float(DEFAULT_PLACE.PAD_Y_MM), float(DEFAULT_PLACE.PAD_Z_MM)).padded_box
        placed_points, placed_colors = _shift_points_to_box(
            winner_candidate.points_robot,
            winner_raw,
            winner_candidate.colors_rgb,
            clip_tol_mm=14.0,
        )
        pts, cols = _sample_points(placed_points, placed_colors, MAX_PTS_OBJECT, 9000 + winner_candidate.index)
        if len(pts) and cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=POINT_SIZE_3D * 1.15, alpha=0.62, depthshade=False)
            scene_pts.append(pts)
        _draw_box_3d(ax, winner_raw, color=WINNER_COLOR, face_alpha=0.14, edge_alpha=0.98, linewidth=1.4)
        _draw_box_3d(ax, winner_pad, color=GHOST_COLOR, face_alpha=0.0, edge_alpha=0.90, linewidth=1.1, linestyle="--")
        scene_pts.append(_box_corners(winner_pad))

    scene_pts.append(_box_corners(make_aabb_from_center_size(np.array([0.5 * (bag.x_min + bag.x_max), 0.5 * (bag.y_min + bag.y_max), 0.5 * (bag.z_min + bag.z_max)], dtype=np.float64), np.array([bag.width, bag.depth, bag.height], dtype=np.float64), label="bag")))
    _set_equal_3d(ax, np.vstack(scene_pts) if scene_pts else np.zeros((1, 3), dtype=np.float64), margin_frac=0.08)
    if not CLEAN_EXPORTS:
        ax.set_title("Reconstructed bag state after placing the chosen candidate", color="white", fontsize=12, pad=10)
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _write_step_audit(step: SurveyPlanningStep, step_dir: Path) -> None:
    csv_path = step_dir / "planner_candidate_audit.csv"
    fieldnames = [
        "candidate_index",
        "class_name",
        "manifest_selected",
        "selector_passed",
        "selector_reject_reasons",
        "replay_can_place",
        "replay_target_x_mm",
        "replay_target_y_mm",
        "replay_layer_z_mm",
        "executed_target_x_mm",
        "executed_target_y_mm",
        "executed_layer_z_mm",
        "reconciliation_mode",
        "reconciliation_note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, candidate in enumerate(step.platform_candidates):
            decision = next(
                (
                    item
                    for item in getattr(step.selection.result, "decisions", [])
                    if item.dbg is candidate.dbg
                ),
                None,
            )
            replay = step.selection.target_cache.get(idx)
            executed = step.executed_target if step.winner_index == idx else None
            writer.writerow(
                {
                    "candidate_index": idx + 1,
                    "class_name": candidate.class_name,
                    "manifest_selected": step.winner_index == idx,
                    "selector_passed": bool(getattr(decision, "passed", False)),
                    "selector_reject_reasons": " | ".join(
                        str(value) for value in getattr(decision, "reject_reasons", [])
                    ),
                    "replay_can_place": replay is not None,
                    "replay_target_x_mm": (
                        float(replay.target_xy_mm[0]) if replay is not None else ""
                    ),
                    "replay_target_y_mm": (
                        float(replay.target_xy_mm[1]) if replay is not None else ""
                    ),
                    "replay_layer_z_mm": (
                        float(getattr(replay, "planner_layer_z_mm", 0.0) or 0.0)
                        if replay is not None
                        else ""
                    ),
                    "executed_target_x_mm": (
                        float(executed.target_xy_mm[0]) if executed is not None else ""
                    ),
                    "executed_target_y_mm": (
                        float(executed.target_xy_mm[1]) if executed is not None else ""
                    ),
                    "executed_layer_z_mm": (
                        float(executed.planner_layer_z_mm or 0.0)
                        if executed is not None
                        else ""
                    ),
                    "reconciliation_mode": (
                        step.reconciliation_mode if step.winner_index == idx else ""
                    ),
                    "reconciliation_note": (
                        step.reconciliation_note if step.winner_index == idx else ""
                    ),
                }
            )

    summary = {
        "manifest_row": step.row.manifest_row,
        "object_i": step.row.object_i,
        "class_name": step.row.class_name,
        "place_result": step.row.place_result,
        "manifest_candidate_index": (
            step.winner_index + 1 if step.winner_index is not None else None
        ),
        "reconciliation_mode": step.reconciliation_mode,
        "reconciliation_note": step.reconciliation_note,
        "planner_target_available": step.planner_target is not None,
        "executed_target_available": step.executed_target is not None,
    }
    (step_dir / "planner_reconciliation.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _save_step_outputs(
    step: SurveyPlanningStep,
    bag: BagVolume,
    run_dir: Path,
    out_root: Path,
    platform_bounds: tuple[float, float, float, float],
    dpi: int,
) -> Path:
    pair_index = _parse_pair_index(step.row.stereo_left) or step.row.manifest_row
    step_dir = (
        out_root
        / "planner_steps"
        / f"step_{pair_index:04d}_row_{step.row.manifest_row:02d}_{safe_name(step.row.class_name)}"
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    _render_scene_overview(step, bag, platform_bounds, step_dir / "01_scene_platform_and_bag_state.png", dpi)
    _render_support_topdown(step, bag, step_dir / "02_topdown_deterministic_support_logic.png", dpi)
    _render_overhead_and_targets(step, bag, run_dir, step_dir / "03_overhead_segmentation_and_valid_targets.png", dpi)
    _render_ranking_grid(step, bag, step_dir / "04_candidate_ranking_and_best_targets.png", dpi)
    _render_bag_after_pick(step, bag, step_dir / "05_bag_state_after_chosen_pick.png", dpi)
    _write_step_audit(step, step_dir)
    return step_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline per-survey pick/place planner visualizer for run snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run snapshot directory with manifest.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory root")
    parser.add_argument("--workspace-profile", default="wet_run", help="Workspace profile for platform bounds")
    parser.add_argument("--limit", type=int, default=None, help="Optional max unique survey snapshots to render")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="Output image DPI")
    parser.add_argument("--annotated", action="store_true", help="Keep titles, axis labels, and decision text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global CLEAN_EXPORTS

    _configure_modules()

    args = parse_args(argv)
    CLEAN_EXPORTS = not bool(args.annotated)
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve_default_run_dir()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _resolve_default_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows(run_dir)
    color_by_class = _class_colors([row.class_name for row in rows])
    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIB_PATH)
    if bundle is None or stereo_calib is None:
        raise RuntimeError("bundle or stereo calibration failed to load")

    yolo = _make_yolo_segmenter()
    bag = _load_bag()
    surface_zone = _load_place_scene()

    ws_cfg = get_workspace_filter_config(str(args.workspace_profile))
    x_min, x_max, y_min, y_max = workspace_bounds_mm(ws_cfg)
    platform_bounds = (x_min, x_max, y_min, y_max)

    all_placed = _build_placed_items(rows, run_dir, color_by_class, bundle, stereo_calib)
    placed_before_lookup: dict[int, list[PlacedItem]] = {}
    for row in rows:
        placed_before_lookup[row.manifest_row] = [item for item in all_placed if item.manifest_row < row.manifest_row]

    rendered = 0
    seen_pairs: set[int] = set()
    for row in rows:
        pair_index = _parse_pair_index(row.stereo_left) or row.manifest_row
        if pair_index in seen_pairs:
            continue
        if args.limit is not None and rendered >= int(args.limit):
            break
        disparity = _load_disparity(run_dir / row.disparity) if row.disparity else None
        if disparity is None or row.stereo_left is None:
            print(f"[SKIP] row {row.manifest_row}: missing stereo/disparity snapshot")
            continue
        try:
            platform_candidates, survey_state = _build_platform_candidates(row, run_dir, yolo, disparity, stereo_calib, bundle)
            if not platform_candidates:
                print(f"[SKIP] row {row.manifest_row}: no offline candidates")
                continue
            step = _plan_survey_step(row, placed_before_lookup[row.manifest_row], platform_candidates, survey_state, surface_zone)
            step_dir = _save_step_outputs(step, bag, run_dir, out_dir, platform_bounds, max(80, int(args.dpi)))
            seen_pairs.add(pair_index)
            rendered += 1
            winner = _winner_target(step)
            winner_name = winner[0].class_name if winner is not None else "none"
            print(f"[SAVE] row {row.manifest_row}: winner={winner_name} candidates={len(platform_candidates)} -> {step_dir}")
        except Exception as exc:
            print(f"[STEP WARN] row {row.manifest_row}: {exc}")

    print(f"[SUMMARY] rendered={rendered} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
