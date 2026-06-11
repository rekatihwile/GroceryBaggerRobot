from __future__ import annotations

"""Offline storyboard figure maker for saved wet-run snapshot folders.

Modes:
  - perception: multi-panel figure for one selected manifest row/object
  - planning:   platform context + bag-state storyboard
  - tune-view:  interactive 3D planning view with saveable camera config

This script only reads saved files under data/run_snapshots/... and optional
local calibration/config files. It never opens cameras, serial, Teensy, or
robot hardware, and it does not rerun YOLO/RAFT by default.
"""

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _preparse_mode(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("perception", "planning", "tune-view"), default="planning")
    args, _unknown = parser.parse_known_args(argv)
    return str(args.mode)


_MODE = _preparse_mode(sys.argv[1:])

import matplotlib

if _MODE != "tune-view":
    matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    PUBLICATION_GLOBAL_ROOT,
    run_output_dir,
    save_figure_bundle,
)
from scripts.capstone.pointcloud_color import (
    load_stereo_calibration,
    project_bundle_camera_points_to_uv,
)

BAG_SCENE_NAME = "New Bag Test"
SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
DEFAULT_BAG_HEIGHT_MM = 250.0
MAX_POINTS_3D = 1600
MAX_POINTS_2D = 1000
DEFAULT_VIEW_CONFIG_DIR = PUBLICATION_GLOBAL_ROOT / "view_configs"

# VS Code IDE defaults. Leave as None to auto-select newest run snapshot.
# Copy/paste your run folder path here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_184606"
IDE_DEFAULT_RUN_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
IDE_DEFAULT_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None

# Copy/paste your storyboard output root here (or keep None for paper_figure_sources).
IDE_DEFAULT_OUT_DIR: Path | None = None
IDE_DEFAULT_RUNS_ROOT = _REPO_ROOT / "data" / "run_snapshots"

_PALETTE = [
    "#38bdf8",
    "#fb923c",
    "#4ade80",
    "#f87171",
    "#a78bfa",
    "#facc15",
    "#2dd4bf",
    "#f472b6",
    "#cbd5e1",
    "#bef264",
]


def _resolve_default_run_dir() -> Path:
    if IDE_DEFAULT_RUN_DIR is not None:
        return IDE_DEFAULT_RUN_DIR.resolve()
    runs_root = IDE_DEFAULT_RUNS_ROOT.resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"default run root does not exist: {runs_root}")
    candidates = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"no run_* snapshot folders found under: {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_default_out_dir(run_dir: Path) -> Path:
    if IDE_DEFAULT_OUT_DIR is not None:
        return IDE_DEFAULT_OUT_DIR.resolve()
    return run_output_dir(run_dir) / "storyboards"


@dataclass(frozen=True)
class Theme:
    name: str
    figure_bg: str
    axes_bg: str
    fg: str
    muted: str
    grid: str
    bag_fill: str
    bag_edge: str
    annotation_bg: str
    annotation_edge: str
    label_bg: str
    label_fg: str
    accent: str
    next_color: str
    handled_color: str
    reject_color: str


THEMES: dict[str, Theme] = {
    "slide_dark": Theme(
        name="slide_dark",
        figure_bg="#080b14",
        axes_bg="#0f172a",
        fg="#f8fafc",
        muted="#cbd5e1",
        grid="#263244",
        bag_fill="#111827",
        bag_edge="#e5e7eb",
        annotation_bg="#111827",
        annotation_edge="#334155",
        label_bg="#f8fafc",
        label_fg="#111827",
        accent="#38bdf8",
        next_color="#facc15",
        handled_color="#94a3b8",
        reject_color="#f87171",
    ),
    "paper": Theme(
        name="paper",
        figure_bg="#ffffff",
        axes_bg="#ffffff",
        fg="#111827",
        muted="#374151",
        grid="#e5e7eb",
        bag_fill="#f8fafc",
        bag_edge="#111827",
        annotation_bg="#ffffff",
        annotation_edge="#cbd5e1",
        label_bg="#ffffff",
        label_fg="#111827",
        accent="#0f766e",
        next_color="#b45309",
        handled_color="#6b7280",
        reject_color="#b91c1c",
    ),
}


@dataclass
class Box3D:
    center: np.ndarray
    size: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    label: str = ""


@dataclass
class PointCloudOverlay:
    points_xyz: np.ndarray
    colors: np.ndarray | None
    source: str
    keys: list[str]


@dataclass
class SnapshotRow:
    manifest_row: int
    order: int | None
    object_i: Any
    class_name: str
    place_result: str
    manifest_raw_box: Box3D | None
    raw_box: Box3D | None
    padded_box: Box3D | None
    place_xy_mm: np.ndarray | None
    destination_surface_z_mm: float | None
    pick_xy_mm: np.ndarray | None
    pick_z_mm: float | None
    stereo_left: str | None
    left_overlay: str | None
    overhead: str | None
    disparity: str | None
    points_cam: str | None
    color: str
    pointcloud: PointCloudOverlay | None = None


@dataclass
class ManifestInfo:
    path: Path
    run_timestamp: str
    script: str
    planner_sequence: str
    placed_boxes_count: int
    total_rows: int


@dataclass
class BagVolume:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    source: str

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
class ViewConfig:
    azim: float = -54.0
    elev: float = 24.0
    dist: float | None = None
    xlim: list[float] | None = None
    ylim: list[float] | None = None
    zlim: list[float] | None = None
    point_size: float = 1.6
    show_labels: bool = True
    show_padded: bool = True
    show_raw: bool = True
    show_pointclouds: bool = True
    theme: str = "slide_dark"


@dataclass
class StoryboardData:
    info: ManifestInfo
    rows_all: list[SnapshotRow]
    rows_placed: list[SnapshotRow]
    bag: BagVolume
    normalized_notes: list[str]


@dataclass
class RenderOptions:
    show_labels: bool
    show_padded: bool
    show_raw: bool
    show_pointclouds: bool
    point_size: float
    minimal: bool


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _finite_vec(value: Any, n: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < n:
        return None
    out = arr[:n].astype(np.float64)
    return out if np.all(np.isfinite(out)) else None


def _make_box(center_xyz_mm: Any, size_xyz_mm: Any, label: str = "") -> Box3D:
    center = _finite_vec(center_xyz_mm, 3)
    size = _finite_vec(size_xyz_mm, 3)
    if center is None:
        raise ValueError("center_xyz_mm must contain three finite values")
    if size is None:
        raise ValueError("size_xyz_mm must contain three finite values")
    if np.any(size <= 0.0):
        raise ValueError("size_xyz_mm must be positive")
    half = 0.5 * size
    return Box3D(center=center, size=size, min_xyz=center - half, max_xyz=center + half, label=label)


def _box_corners(box: Box3D) -> np.ndarray:
    mn = box.min_xyz
    mx = box.max_xyz
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
    faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ]
    return [[corners[i] for i in face] for face in faces]


def _draw_box_3d(
    ax,
    box: Box3D,
    *,
    color: str,
    face_alpha: float,
    edge_alpha: float,
    linewidth: float,
    linestyle: str,
) -> None:
    corners = _box_corners(box)
    if face_alpha > 0.0:
        poly = Poly3DCollection(_box_faces(corners), alpha=face_alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)
    for a, b in [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            alpha=edge_alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def _class_colors(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        if name not in out:
            out[name] = _PALETTE[len(out) % len(_PALETTE)]
    return out


def _short_label(name: str, max_len: int = 16) -> str:
    clean = " ".join(str(name).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "."


def _bag_height_from_config() -> float:
    try:
        from config.place import DEFAULT_PLACE

        return float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", DEFAULT_BAG_HEIGHT_MM))
    except Exception:
        return DEFAULT_BAG_HEIGHT_MM


def _bag_box(bag: BagVolume) -> Box3D:
    center = np.array(
        [
            0.5 * (bag.x_min + bag.x_max),
            0.5 * (bag.y_min + bag.y_max),
            0.5 * (bag.z_min + bag.z_max),
        ],
        dtype=np.float64,
    )
    size = np.array([bag.width, bag.depth, bag.height], dtype=np.float64)
    return _make_box(center, size, label=bag.name)


def _all_points(rows: list[SnapshotRow], bag: BagVolume | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for row in rows:
        if row.raw_box is not None:
            chunks.append(_box_corners(row.raw_box))
        if row.padded_box is not None:
            chunks.append(_box_corners(row.padded_box))
        if row.pointcloud is not None and len(row.pointcloud.points_xyz):
            chunks.append(row.pointcloud.points_xyz)
    if bag is not None:
        chunks.append(_box_corners(_bag_box(bag)))
    return np.vstack(chunks) if chunks else np.zeros((1, 3), dtype=np.float64)


def _set_equal_3d(ax, points: np.ndarray, margin_frac: float = 0.08) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    span = max(float((mx - mn).max()), 1.0)
    half = 0.5 * span * (1.0 + margin_frac)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def _npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return list(data.files)


def _load_bundle() -> dict[str, np.ndarray] | None:
    if not BUNDLE_PATH.exists():
        return None
    try:
        with np.load(BUNDLE_PATH, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        print(f"[BUNDLE WARN] failed to load {BUNDLE_PATH.name}: {exc}")
        return None


def _pick_point_array(data: Any, preferred: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in preferred:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float64), key
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float64), key
    return None, None


def _pick_uv_array(data: Any) -> np.ndarray | None:
    for key in ("uv", "uv_px", "point_uv_px", "points_uv", "point_uv"):
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                out = arr[:, :2].astype(np.float64)
                return out if np.all(np.isfinite(out)) else None
    return None


def _load_image_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        img = mpimg.imread(path)
    except Exception:
        return None
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.dtype.kind in "ui":
        img = img.astype(np.float32) / 255.0
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return np.asarray(img, dtype=np.float32)


def _sample_point_colors_rgb(image_rgb: np.ndarray | None, uv_px: np.ndarray | None, n: int) -> np.ndarray | None:
    if image_rgb is None or uv_px is None or len(uv_px) != n:
        return None
    h, w = image_rgb.shape[:2]
    xs = np.clip(np.rint(uv_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv_px[:, 1]).astype(np.int32), 0, h - 1)
    rgb = image_rgb[ys, xs, :3]
    return np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)


def _row_color_image(run_dir: Path, row: SnapshotRow) -> np.ndarray | None:
    if row.stereo_left:
        image_rgb = _load_image_rgb(run_dir / row.stereo_left)
        if image_rgb is not None:
            return image_rgb
    if row.left_overlay:
        image_rgb = _load_image_rgb(run_dir / row.left_overlay)
        if image_rgb is not None:
            return image_rgb
    return None


def _load_disparity(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "disparity" in data.files:
                return np.asarray(data["disparity"], dtype=np.float64)
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim == 2:
                    return arr.astype(np.float64)
    except Exception as exc:
        print(f"[DISP WARN] failed to load {path.name}: {exc}")
    return None


def _load_manifest(run_dir: Path) -> StoryboardData:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_json = data.get("objects") or []
    if not isinstance(rows_json, list):
        raise ValueError("manifest field 'objects' is not a list")

    colors = _class_colors([str(r.get("class_name") or r.get("detection_class") or "object") for r in rows_json])
    rows_all: list[SnapshotRow] = []
    rows_placed: list[SnapshotRow] = []

    for row_i, row in enumerate(rows_json, start=1):
        class_name = str(row.get("class_name") or row.get("detection_class") or "object")
        result = str(row.get("place_result") or "")
        raw_box = None
        padded_box = None
        manifest_raw_box = None
        if row.get("raw_box_center_xyz_mm") is not None and row.get("raw_box_size_xyz_mm") is not None:
            try:
                manifest_raw_box = _make_box(row.get("raw_box_center_xyz_mm"), row.get("raw_box_size_xyz_mm"), label=f"{class_name}_raw_manifest")
                raw_box = _make_box(row.get("raw_box_center_xyz_mm"), row.get("raw_box_size_xyz_mm"), label=f"{class_name}_raw")
            except Exception:
                manifest_raw_box = None
                raw_box = None
        if row.get("padded_box_center_xyz_mm") is not None and row.get("padded_box_size_xyz_mm") is not None:
            try:
                padded_box = _make_box(row.get("padded_box_center_xyz_mm"), row.get("padded_box_size_xyz_mm"), label=f"{class_name}_padded")
            except Exception:
                padded_box = None

        snap_row = SnapshotRow(
            manifest_row=row_i,
            order=None,
            object_i=row.get("object_i"),
            class_name=class_name,
            place_result=result,
            manifest_raw_box=manifest_raw_box,
            raw_box=raw_box,
            padded_box=padded_box,
            place_xy_mm=_finite_vec(row.get("place_xy_mm"), 2),
            destination_surface_z_mm=_finite_float(row.get("destination_surface_z_mm")),
            pick_xy_mm=_finite_vec(row.get("pick_xy_mm"), 2),
            pick_z_mm=_finite_float(row.get("pick_z_mm")),
            stereo_left=row.get("stereo_left"),
            left_overlay=row.get("left_overlay"),
            overhead=row.get("overhead"),
            disparity=row.get("disparity"),
            points_cam=row.get("points_cam"),
            color=colors[class_name],
        )
        rows_all.append(snap_row)

        if (
            str(result).lower() == "placed"
            and snap_row.raw_box is not None
            and snap_row.padded_box is not None
        ):
            snap_row.order = len(rows_placed) + 1
            rows_placed.append(snap_row)

    bag = _load_bag_volume(rows_placed)
    normalized_notes = _normalize_raw_boxes_to_bag_pose(rows_placed, bag)

    info = ManifestInfo(
        path=manifest_path,
        run_timestamp=str(data.get("run_timestamp") or run_dir.name.replace("run_", "")),
        script=str(data.get("script") or "unknown"),
        planner_sequence=str(data.get("place_planning_sequence") or "unknown"),
        placed_boxes_count=len(data.get("placed_boxes") or []),
        total_rows=len(rows_json),
    )
    print(
        f"[MANIFEST] loaded {len(rows_placed)} placed row(s) from {manifest_path} "
        f"(total manifest rows={len(rows_all)})"
    )
    return StoryboardData(info=info, rows_all=rows_all, rows_placed=rows_placed, bag=bag, normalized_notes=normalized_notes)


def _load_bag_volume(rows_placed: list[SnapshotRow]) -> BagVolume:
    bag_height = _bag_height_from_config()
    if rows_placed:
        pts = np.vstack([_box_corners(row.padded_box) for row in rows_placed if row.padded_box is not None])
        box_min = pts.min(axis=0)
        box_max = pts.max(axis=0)
    else:
        box_min = np.zeros(3, dtype=np.float64)
        box_max = np.array([250.0, 200.0, bag_height], dtype=np.float64)

    try:
        zones = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
        scene = zones[BAG_SCENE_NAME]
        center = _finite_vec(scene.get("center_xy_mm"), 2)
        width = _finite_float(scene.get("width_mm"))
        depth = _finite_float(scene.get("depth_mm"))
        surface_z = _finite_float(scene.get("surface_z_mm"))
        if center is None or width is None or depth is None or width <= 0.0 or depth <= 0.0:
            raise ValueError("invalid center/width/depth")
        z_min = min(float(surface_z if surface_z is not None else box_min[2]), float(box_min[2]))
        z_max = max(z_min + bag_height, float(box_max[2]) + 10.0)
        bag = BagVolume(
            name=BAG_SCENE_NAME,
            x_min=float(center[0] - 0.5 * width),
            x_max=float(center[0] + 0.5 * width),
            y_min=float(center[1] - 0.5 * depth),
            y_max=float(center[1] + 0.5 * depth),
            z_min=float(z_min),
            z_max=float(z_max),
            source=str(SURFACE_ZONES_PATH.relative_to(_REPO_ROOT)),
        )
        print(
            f"[BAG] {bag.name}: x=[{bag.x_min:.1f},{bag.x_max:.1f}] "
            f"y=[{bag.y_min:.1f},{bag.y_max:.1f}] z=[{bag.z_min:.1f},{bag.z_max:.1f}]"
        )
        return bag
    except Exception as exc:
        margin_xy = 35.0
        margin_z = 20.0
        print(f"[BAG WARN] failed to load {BAG_SCENE_NAME!r}: {exc}; inferring bag bounds from placed boxes")
        return BagVolume(
            name="inferred_bag",
            x_min=float(box_min[0] - margin_xy),
            x_max=float(box_max[0] + margin_xy),
            y_min=float(box_min[1] - margin_xy),
            y_max=float(box_max[1] + margin_xy),
            z_min=float(box_min[2] - margin_z),
            z_max=float(box_max[2] + margin_z),
            source="inferred from placed boxes",
        )


def _normalize_raw_boxes_to_bag_pose(rows_placed: list[SnapshotRow], bag: BagVolume) -> list[str]:
    notes: list[str] = []
    for row in rows_placed:
        if row.raw_box is None or row.manifest_raw_box is None or row.padded_box is None:
            continue
        raw_xy = row.manifest_raw_box.center[:2]
        pad_xy = row.padded_box.center[:2]
        raw_inside = bag.x_min - 10.0 <= raw_xy[0] <= bag.x_max + 10.0 and bag.y_min - 10.0 <= raw_xy[1] <= bag.y_max + 10.0
        pad_inside = bag.x_min - 10.0 <= pad_xy[0] <= bag.x_max + 10.0 and bag.y_min - 10.0 <= pad_xy[1] <= bag.y_max + 10.0
        if not raw_inside and pad_inside:
            center = row.padded_box.center.copy()
            if row.place_xy_mm is not None:
                center[:2] = row.place_xy_mm
            row.raw_box = _make_box(center, row.manifest_raw_box.size, label=f"{row.class_name}_raw_placed")
            notes.append(
                f"row {row.manifest_row}: normalized raw AABB center from pick pose "
                f"({raw_xy[0]:.1f},{raw_xy[1]:.1f}) to bag pose ({center[0]:.1f},{center[1]:.1f})"
            )
    if notes:
        print(f"[AABB] normalized {len(notes)} raw AABB center(s) to placed bag pose")
    return notes


def _shift_points_to_box(points_robot: np.ndarray, row: SnapshotRow) -> tuple[np.ndarray, np.ndarray]:
    if row.raw_box is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
    points_robot = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    points_robot = points_robot[np.all(np.isfinite(points_robot), axis=1)]
    source_centroid_xy = np.mean(points_robot[:, :2], axis=0)
    source_min_z = float(np.min(points_robot[:, 2]))
    shift = np.array(
        [
            row.raw_box.center[0] - source_centroid_xy[0],
            row.raw_box.center[1] - source_centroid_xy[1],
            row.raw_box.min_xyz[2] - source_min_z,
        ],
        dtype=np.float64,
    )
    shifted = points_robot + shift.reshape(1, 3)
    tol = 3.0
    inside = np.all(
        (shifted >= row.raw_box.min_xyz.reshape(1, 3) - tol)
        & (shifted <= row.raw_box.max_xyz.reshape(1, 3) + tol),
        axis=1,
    )
    return shifted[inside], inside


def _attach_pointclouds(rows: list[SnapshotRow], run_dir: Path, selected_only: SnapshotRow | None = None) -> tuple[int, list[str]]:
    bundle = _load_bundle()
    stereo_calib = load_stereo_calibration(STEREO_CALIB_PATH)
    transform = None
    if bundle is not None:
        try:
            from vision.pointcloud import cam_points_to_robot_xyz

            transform = cam_points_to_robot_xyz
        except Exception as exc:
            print(f"[POINTCLOUD WARN] import failed: {exc}")

    targets = [selected_only] if selected_only is not None else rows
    loaded = 0
    summary: list[str] = []
    for row in targets:
        if row is None or not row.points_cam:
            continue
        p = run_dir / row.points_cam
        if not p.exists():
            print(f"[POINTCLOUD WARN] row {row.manifest_row}: missing {p.name}")
            summary.append(f"row {row.manifest_row}: missing point cloud")
            continue
        try:
            with np.load(p, allow_pickle=False) as data:
                keys = list(data.files)
                print(f"[POINTCLOUD] row {row.manifest_row} {p.name} keys={keys}")
                robot_pts, robot_key = _pick_point_array(data, ["points_robot", "points_robot_xyz", "robot_points", "points_xyz_robot"])
                cam_pts = None
                source = None
                if robot_key not in {"points_robot", "points_robot_xyz", "robot_points", "points_xyz_robot"}:
                    robot_pts = None
                    robot_key = None
                if robot_pts is not None:
                    source = str(robot_key)
                if robot_pts is None:
                    cam_pts, cam_key = _pick_point_array(data, ["points_cam", "points_camera", "cam_points", "points_xyz_cam"])
                    if cam_pts is None:
                        raise ValueError("no Nx3 point array found")
                    if transform is None or bundle is None:
                        raise ValueError("camera-frame points found but no calibration transform available")
                    robot_pts = transform(cam_pts, bundle)
                    source = f"{cam_key}->robot"
                uv = _pick_uv_array(data)
                if uv is None and cam_pts is not None:
                    uv = project_bundle_camera_points_to_uv(cam_pts, stereo_calib)
                image_rgb = _row_color_image(run_dir, row)
                point_colors = _sample_point_colors_rgb(image_rgb, uv, len(robot_pts))
                if point_colors is None:
                    print(f"[POINTCLOUD WARN] row {row.manifest_row}: original-photo RGB unavailable; skipping cloud")
                    summary.append(f"row {row.manifest_row}: point cloud skipped because photo RGB was unavailable")
                    continue
                shifted, inside = _shift_points_to_box(robot_pts, row)
                if len(shifted) < 20:
                    print(f"[POINTCLOUD WARN] row {row.manifest_row}: fewer than 20 shifted points inside raw AABB")
                    summary.append(f"row {row.manifest_row}: point cloud too sparse after bag-pose shift")
                    continue
                if point_colors is not None and len(point_colors) == len(inside):
                    point_colors = point_colors[inside]
                row.pointcloud = PointCloudOverlay(points_xyz=shifted, colors=point_colors, source=str(source), keys=keys)
                loaded += 1
                summary.append(f"row {row.manifest_row}: point cloud loaded from {source} ({len(shifted)} shifted points)")
        except Exception as exc:
            print(f"[POINTCLOUD WARN] row {row.manifest_row}: failed to load {p.name}: {exc}")
            summary.append(f"row {row.manifest_row}: point cloud load failed ({exc})")
    return loaded, summary


def _load_view_config(path: Path | None) -> ViewConfig | None:
    if path is None:
        return None
    if not path.exists():
        print(f"[VIEW WARN] config not found: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ViewConfig(**data)
    except Exception as exc:
        print(f"[VIEW WARN] failed to load {path}: {exc}")
        return None


def _save_view_config(path: Path, view: ViewConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(view), indent=2) + "\n", encoding="utf-8")
    return path


def _apply_view_config(ax, view: ViewConfig | None) -> None:
    if view is None:
        ax.view_init(elev=24, azim=-54)
        return
    ax.view_init(elev=float(view.elev), azim=float(view.azim))
    if view.xlim is not None:
        ax.set_xlim(*view.xlim)
    if view.ylim is not None:
        ax.set_ylim(*view.ylim)
    if view.zlim is not None:
        ax.set_zlim(*view.zlim)
    if view.dist is not None and hasattr(ax, "dist"):
        try:
            ax.dist = float(view.dist)
        except Exception:
            pass


def _capture_view_config(ax, theme_name: str, opts: RenderOptions) -> ViewConfig:
    dist = None
    if hasattr(ax, "dist"):
        try:
            dist = float(ax.dist)
        except Exception:
            dist = None
    return ViewConfig(
        azim=float(ax.azim),
        elev=float(ax.elev),
        dist=dist,
        xlim=[float(v) for v in ax.get_xlim3d()],
        ylim=[float(v) for v in ax.get_ylim3d()],
        zlim=[float(v) for v in ax.get_zlim3d()],
        point_size=float(opts.point_size),
        show_labels=bool(opts.show_labels),
        show_padded=bool(opts.show_padded),
        show_raw=bool(opts.show_raw),
        show_pointclouds=bool(opts.show_pointclouds),
        theme=theme_name,
    )


def _theme_from_args(name: str, override: ViewConfig | None) -> Theme:
    theme_name = override.theme if override is not None and override.theme in THEMES else name
    return THEMES[theme_name]


def _render_annotation_box(ax, theme: Theme, lines: list[str], loc: tuple[float, float] = (0.985, 0.985), ha: str = "right") -> None:
    ax.text(
        loc[0],
        loc[1],
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=8,
        color=theme.fg,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=theme.annotation_bg, edgecolor=theme.annotation_edge, alpha=0.86),
    )


def _style_3d(ax, theme: Theme, minimal: bool) -> None:
    ax.set_facecolor(theme.axes_bg)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(theme.axes_bg)
        pane.set_edgecolor(theme.grid)
        pane.set_alpha(0.95)
    ax.grid(not minimal, color=theme.grid, linewidth=0.65, alpha=0.70)
    ax.tick_params(colors=theme.muted, labelsize=7)
    if minimal:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    else:
        ax.set_xlabel("Robot X (mm)", color=theme.muted, fontsize=8, labelpad=8)
        ax.set_ylabel("Robot Y (mm)", color=theme.muted, fontsize=8, labelpad=8)
        ax.set_zlabel("Robot Z (mm)", color=theme.muted, fontsize=8, labelpad=8)


def _draw_bag_3d(ax, bag: BagVolume, theme: Theme) -> None:
    bag_box = _bag_box(bag)
    corners = _box_corners(bag_box)
    floor = Poly3DCollection([[corners[i] for i in [0, 1, 2, 3]]], alpha=0.16)
    floor.set_facecolor(theme.bag_fill)
    floor.set_edgecolor("none")
    ax.add_collection3d(floor)
    _draw_box_3d(ax, bag_box, color=theme.bag_edge, face_alpha=0.0, edge_alpha=0.72, linewidth=1.2, linestyle="-")


def _sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def _row_label_xy(row: SnapshotRow) -> tuple[float, float]:
    assert row.raw_box is not None
    offsets = [(0.0, 0.0), (-0.18, 0.10), (0.18, -0.10), (-0.16, -0.16), (0.16, 0.16), (0.0, 0.22), (0.0, -0.22)]
    ox, oy = offsets[((row.order or row.manifest_row) - 1) % len(offsets)]
    return (
        float(row.raw_box.center[0] + ox * row.raw_box.size[0]),
        float(row.raw_box.center[1] + oy * row.raw_box.size[1]),
    )


def _axis_bounds_2d(rows: list[SnapshotRow], bag: BagVolume, margin: float = 25.0) -> tuple[float, float, float, float]:
    pts = _all_points(rows, bag)
    return (
        min(float(pts[:, 0].min()), bag.x_min) - margin,
        max(float(pts[:, 0].max()), bag.x_max) + margin,
        min(float(pts[:, 1].min()), bag.y_min) - margin,
        max(float(pts[:, 1].max()), bag.y_max) + margin,
    )


def _resolve_row_selector(data: StoryboardData, row_value: int | None, object_index: Any | None) -> SnapshotRow:
    if row_value is not None:
        for row in data.rows_all:
            if row.manifest_row == row_value:
                return row
        raise ValueError(f"--row {row_value} did not match any manifest row")
    if object_index is not None:
        matches = [row for row in data.rows_all if str(row.object_i) == str(object_index)]
        if not matches:
            raise ValueError(f"--object-index {object_index!r} did not match any manifest object_i")
        if len(matches) > 1:
            print(f"[SELECT] object_i={object_index!r} matched {len(matches)} rows; using first manifest row {matches[0].manifest_row}")
        return matches[0]
    raise ValueError("one of --row or --object-index is required for this mode")


def _resolve_placed_cutoff(data: StoryboardData, value: int | None) -> SnapshotRow:
    if value is None:
        return data.rows_placed[-1]
    for row in data.rows_placed:
        if row.order == value:
            return row
    for row in data.rows_placed:
        if row.manifest_row == value:
            print(f"[SELECT] using placed row with manifest_row={value} and placement order={row.order}")
            return row
    raise ValueError(f"could not resolve placed cutoff from value {value}")


def _render_perception_storyboard(
    data: StoryboardData,
    run_dir: Path,
    row: SnapshotRow,
    out_dir: Path,
    theme: Theme,
    render_opts: RenderOptions,
    view_cfg: ViewConfig | None,
    *,
    dpi: int,
    save_panel_crops: bool,
) -> list[Path]:
    if row.pointcloud is None:
        _attach_pointclouds(data.rows_placed, run_dir, selected_only=row if row in data.rows_placed else None)

    fig = plt.figure(figsize=(16.0, 8.6), facecolor=theme.figure_bg)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.06], height_ratios=[1, 1], wspace=0.06, hspace=0.08)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_disp = fig.add_subplot(gs[1, 0])
    ax3d = fig.add_subplot(gs[1, 1], projection="3d")

    for ax in (ax_img, ax_overlay, ax_disp):
        ax.set_facecolor(theme.axes_bg)
        ax.set_xticks([])
        ax.set_yticks([])

    img_left = _load_image_rgb(run_dir / row.stereo_left) if row.stereo_left else None
    img_overlay = _load_image_rgb(run_dir / row.left_overlay) if row.left_overlay else None
    img_overhead = _load_image_rgb(run_dir / row.overhead) if row.overhead else None
    disp = _load_disparity(run_dir / row.disparity) if row.disparity else None

    ax_img.imshow(img_left if img_left is not None else np.zeros((480, 640, 3), dtype=np.float32))
    ax_img.set_title("Saved stereo-left view", color=theme.fg, fontsize=10)
    if img_left is None:
        ax_img.text(0.5, 0.5, "missing", ha="center", va="center", color=theme.muted, transform=ax_img.transAxes)

    ax_overlay.imshow(img_overlay if img_overlay is not None else (img_overhead if img_overhead is not None else np.zeros((480, 640, 3), dtype=np.float32)))
    ax_overlay.set_title("Saved detection / mask overlay", color=theme.fg, fontsize=10)
    if img_overlay is None and img_overhead is None:
        ax_overlay.text(0.5, 0.5, "missing", ha="center", va="center", color=theme.muted, transform=ax_overlay.transAxes)

    if disp is not None:
        lo, hi = np.nanpercentile(disp[np.isfinite(disp)], [2.0, 98.0])
        ax_disp.imshow(disp, cmap="magma", vmin=lo, vmax=hi)
    else:
        ax_disp.imshow(np.zeros((480, 640, 3), dtype=np.float32))
        ax_disp.text(0.5, 0.5, "missing", ha="center", va="center", color=theme.muted, transform=ax_disp.transAxes)
    ax_disp.set_title("Saved disparity heatmap", color=theme.fg, fontsize=10)

    _style_3d(ax3d, theme, render_opts.minimal)
    if row.raw_box is not None and render_opts.show_raw:
        _draw_box_3d(ax3d, row.raw_box, color=row.color, face_alpha=0.26, edge_alpha=1.0, linewidth=1.3, linestyle="-")
    if row.padded_box is not None and render_opts.show_padded:
        _draw_box_3d(ax3d, row.padded_box, color=theme.next_color, face_alpha=0.0, edge_alpha=0.88, linewidth=1.1, linestyle="--")
    if row.pointcloud is not None and row.pointcloud.colors is not None and render_opts.show_pointclouds:
        idx = _sample_indices(len(row.pointcloud.points_xyz), MAX_POINTS_3D, 3000 + row.manifest_row)
        colors = row.pointcloud.colors[idx]
        ax3d.scatter(
            row.pointcloud.points_xyz[idx, 0],
            row.pointcloud.points_xyz[idx, 1],
            row.pointcloud.points_xyz[idx, 2],
            s=render_opts.point_size,
            c=colors,
            alpha=0.36,
            depthshade=False,
        )
    if row.raw_box is not None:
        scale_pts = _all_points([row], None)
        _set_equal_3d(ax3d, scale_pts, margin_frac=0.12)
    _apply_view_config(ax3d, view_cfg)
    ax3d.set_title("Point cloud + AABB geometry estimate", color=theme.fg, fontsize=10, pad=8)

    if row.raw_box is not None and render_opts.show_labels:
        ax3d.text(
            row.raw_box.center[0],
            row.raw_box.center[1],
            row.raw_box.max_xyz[2] + 10.0,
            f"{_short_label(row.class_name, 18)}",
            color=theme.fg,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    fig.suptitle(
        f"Perception storyboard | row {row.manifest_row}: {row.class_name}",
        color=theme.fg,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    _render_annotation_box(
        ax_overlay,
        theme,
        [
            f"run: {data.info.run_timestamp}",
            f"script: {data.info.script}",
            f"planner: {data.info.planner_sequence}",
            f"row: {row.manifest_row}",
            f"result: {row.place_result or 'unknown'}",
            f"point cloud: {'yes' if row.pointcloud is not None else 'no'}",
        ],
        loc=(0.985, 0.04),
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.91, bottom=0.05, wspace=0.06, hspace=0.08)

    paths = save_figure_bundle(
        fig,
        out_dir / "perception_storyboard",
        dpi=dpi,
        facecolor=theme.figure_bg,
    )
    if save_panel_crops:
        paths.extend(_save_panel_crops(fig, {"panel_image": ax_img, "panel_overlay": ax_overlay, "panel_disparity": ax_disp, "panel_3d": ax3d}, out_dir, dpi, theme))
    plt.close(fig)
    return paths


def _interpolate_h_for_z(bundle: dict[str, np.ndarray], z_query: float, key: str) -> np.ndarray:
    z_levels = np.asarray(bundle.get("z_levels_mm"), dtype=np.float64).reshape(-1)
    Hs = np.asarray(bundle.get(key), dtype=np.float64)
    if len(z_levels) == 0 or len(Hs) == 0:
        raise RuntimeError(f"bundle missing {key}/z_levels_mm")
    zq = float(z_query)
    if len(z_levels) == 1:
        return Hs[0]
    if zq <= z_levels[0]:
        lo, hi = 0, 1
    elif zq >= z_levels[-1]:
        lo, hi = len(z_levels) - 2, len(z_levels) - 1
    else:
        hi = int(np.searchsorted(z_levels, zq))
        lo = hi - 1
    denom = float(z_levels[hi] - z_levels[lo])
    alpha = 0.0 if abs(denom) < 1e-9 else float((zq - z_levels[lo]) / denom)
    return (1.0 - alpha) * Hs[lo] + alpha * Hs[hi]


def _apply_h(pt_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.array([float(pt_xy[0]), float(pt_xy[1]), 1.0], dtype=np.float64)
    q = np.asarray(H, dtype=np.float64) @ p
    if abs(float(q[2])) < 1e-9:
        raise RuntimeError("homography projection returned w ~= 0")
    return q[:2] / q[2]


def _project_robot_xyz_to_overhead_uv(robot_xyz: np.ndarray, bundle: dict[str, np.ndarray]) -> np.ndarray | None:
    try:
        H = _interpolate_h_for_z(bundle, float(robot_xyz[2]), "H_robot_to_img_by_z")
        return _apply_h(robot_xyz[:2], H)
    except Exception:
        return None


def _render_planning_left_panel(
    ax,
    data: StoryboardData,
    selected: SnapshotRow,
    placed_before: list[SnapshotRow],
    rows_up_to: list[SnapshotRow],
    run_dir: Path,
    theme: Theme,
) -> list[str]:
    notes: list[str] = []
    img = _load_image_rgb(run_dir / selected.overhead) if selected.overhead else None
    if img is None:
        img = _load_image_rgb(run_dir / selected.left_overlay) if selected.left_overlay else None
        notes.append("used left_overlay because overhead image was missing")
    if img is None:
        img = np.zeros((720, 1280, 3), dtype=np.float32)
        notes.append("no saved scene image was available")
    ax.set_facecolor(theme.axes_bg)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Survey context and chosen next object", color=theme.fg, fontsize=10)

    bundle = _load_bundle()
    uv = None
    if bundle is not None and selected.manifest_raw_box is not None:
        uv = _project_robot_xyz_to_overhead_uv(selected.manifest_raw_box.center, bundle)
        if uv is None:
            notes.append("could not project chosen robot XYZ back into overhead image")
    if uv is not None:
        ax.scatter([uv[0]], [uv[1]], s=120, c=theme.next_color, edgecolors="black", linewidths=1.3, zorder=6)
        ax.text(
            uv[0],
            uv[1] - 20,
            f"NEXT: {selected.class_name}",
            color=theme.fg,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=theme.annotation_bg, edgecolor=theme.next_color, alpha=0.90),
            zorder=7,
        )
    else:
        ax.text(
            0.03,
            0.78,
            f"NEXT\nrow {selected.manifest_row}\n{selected.class_name}",
            transform=ax.transAxes,
            color=theme.fg,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.34", facecolor=theme.annotation_bg, edgecolor=theme.next_color, alpha=0.90),
        )

    handled = [f"{row.order}. {_short_label(row.class_name, 14)}" for row in placed_before]
    preview = handled[:6]
    if len(handled) > 6:
        preview.append(f"+{len(handled) - 6} more")
    lines = [
        f"run: {data.info.run_timestamp}",
        f"planner: {data.info.planner_sequence}",
        f"current placement step: {selected.order}",
        f"handled before this step: {len(placed_before)}",
    ]
    if preview:
        lines.append("handled:")
        lines.extend(preview)
    _render_annotation_box(ax, theme, lines, loc=(0.985, 0.985))

    if rows_up_to:
        footer = f"bag state reconstructed through placement {rows_up_to[-1].order}: {rows_up_to[-1].class_name}"
        ax.text(0.015, 0.02, footer, transform=ax.transAxes, color=theme.muted, fontsize=7.5, ha="left", va="bottom")
    return notes


def _render_bag_state_3d(
    ax,
    rows: list[SnapshotRow],
    bag: BagVolume,
    theme: Theme,
    render_opts: RenderOptions,
    view_cfg: ViewConfig | None,
    highlight_row: SnapshotRow | None,
) -> None:
    _style_3d(ax, theme, render_opts.minimal)
    _draw_bag_3d(ax, bag, theme)

    for row in rows:
        is_highlight = highlight_row is not None and row.manifest_row == highlight_row.manifest_row
        color = theme.next_color if is_highlight else row.color
        if render_opts.show_pointclouds and row.pointcloud is not None and row.pointcloud.colors is not None:
            idx = _sample_indices(len(row.pointcloud.points_xyz), MAX_POINTS_3D, 2000 + row.manifest_row)
            colors = row.pointcloud.colors[idx]
            alpha = 0.48 if is_highlight else 0.32
            ax.scatter(
                row.pointcloud.points_xyz[idx, 0],
                row.pointcloud.points_xyz[idx, 1],
                row.pointcloud.points_xyz[idx, 2],
                s=render_opts.point_size * (1.25 if is_highlight else 1.0),
                c=colors,
                alpha=alpha,
                depthshade=False,
            )
        if render_opts.show_raw and row.raw_box is not None:
            _draw_box_3d(ax, row.raw_box, color=color, face_alpha=0.22 if is_highlight else 0.14, edge_alpha=0.98, linewidth=1.35 if is_highlight else 1.0, linestyle="-")
        if render_opts.show_padded and row.padded_box is not None:
            _draw_box_3d(ax, row.padded_box, color=color, face_alpha=0.0, edge_alpha=0.84 if is_highlight else 0.52, linewidth=1.05, linestyle="--")
        if render_opts.show_labels and row.raw_box is not None:
            ax.text(
                row.raw_box.center[0],
                row.raw_box.center[1],
                row.raw_box.max_xyz[2] + 8.0,
                f"{row.order} {_short_label(row.class_name, 13)}",
                color=theme.fg,
                fontsize=7.0,
                ha="center",
                va="bottom",
            )

    _set_equal_3d(ax, _all_points(rows, bag), margin_frac=0.08)
    _apply_view_config(ax, view_cfg)


def _render_planning_storyboard(
    data: StoryboardData,
    run_dir: Path,
    selected: SnapshotRow,
    rows_up_to: list[SnapshotRow],
    out_dir: Path,
    theme: Theme,
    render_opts: RenderOptions,
    view_cfg: ViewConfig | None,
    *,
    dpi: int,
    save_panel_crops: bool,
) -> tuple[list[Path], list[str]]:
    fig = plt.figure(figsize=(16.0, 8.8), facecolor=theme.figure_bg)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.05)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")

    placed_before = [row for row in data.rows_placed if (row.order or 0) < (selected.order or 0)]
    left_notes = _render_planning_left_panel(ax_left, data, selected, placed_before, rows_up_to, run_dir, theme)
    _render_bag_state_3d(ax_right, rows_up_to, data.bag, theme, render_opts, view_cfg, highlight_row=selected)
    ax_right.set_title("Bag-local colored point clouds + AABB abstraction", color=theme.fg, fontsize=10, pad=8)

    fig.suptitle(
        f"Planning storyboard | through placement {selected.order}: {selected.class_name}",
        color=theme.fg,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    ax_right.text2D(
        0.02,
        0.96,
        "point clouds show placed geometry   solid = raw AABB   dashed = padded AABB",
        transform=ax_right.transAxes,
        color=theme.muted,
        fontsize=7.7,
        va="top",
    )
    fig.subplots_adjust(left=0.03, right=0.985, top=0.92, bottom=0.05, wspace=0.05)

    paths = save_figure_bundle(
        fig,
        out_dir / "planning_storyboard",
        dpi=dpi,
        facecolor=theme.figure_bg,
    )
    if save_panel_crops:
        paths.extend(_save_panel_crops(fig, {"planning_left": ax_left, "planning_right": ax_right}, out_dir, dpi, theme))
    plt.close(fig)
    return paths, left_notes


def _save_panel_crops(fig: plt.Figure, axes: dict[str, Any], out_dir: Path, dpi: int, theme: Theme) -> list[Path]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    paths: list[Path] = []
    for name, ax in axes.items():
        extent = ax.get_tightbbox(renderer).expanded(1.03, 1.05)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches=extent.transformed(fig.dpi_scale_trans.inverted()), facecolor=theme.figure_bg)
        paths.append(path)
    return paths


def _compile_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int, loop: int) -> Path:
    if not frame_paths:
        raise RuntimeError("no GIF frames were rendered")
    try:
        from PIL import Image

        frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=int(loop),
            optimize=True,
        )
        for frame in frames:
            frame.close()
    except Exception as pil_exc:
        try:
            import imageio.v2 as imageio

            images = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(gif_path, images, duration=max(1, int(duration_ms)) / 1000.0, loop=int(loop))
        except Exception as imageio_exc:
            raise RuntimeError(f"GIF compile failed via Pillow ({pil_exc}) and imageio ({imageio_exc})")
    return gif_path


def _frame_output_dir(out_dir: Path, name: str, save_frames: bool) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if save_frames:
        frame_dir = out_dir / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir, None
    tmp = tempfile.TemporaryDirectory(prefix=f"{name}_")
    return Path(tmp.name), tmp


def _save_storyboard_gif(
    data: StoryboardData,
    run_dir: Path,
    up_to: SnapshotRow,
    out_dir: Path,
    theme: Theme,
    render_opts: RenderOptions,
    view_cfg: ViewConfig | None,
    *,
    dpi: int,
    save_frames: bool,
    gif_duration_ms: int,
    gif_loop: int,
) -> list[Path]:
    frame_dir, tmp = _frame_output_dir(out_dir, "frames_storyboard", save_frames)
    frame_paths: list[Path] = []
    try:
        for k in range(1, (up_to.order or 0) + 1):
            selected = data.rows_placed[k - 1]
            rows = data.rows_placed[:k]
            fig = plt.figure(figsize=(14.0, 7.4), facecolor=theme.figure_bg)
            gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.05], wspace=0.05)
            ax_l = fig.add_subplot(gs[0, 0])
            ax_r = fig.add_subplot(gs[0, 1], projection="3d")
            _render_planning_left_panel(ax_l, data, selected, rows[:-1], rows, run_dir, theme)
            _render_bag_state_3d(ax_r, rows, data.bag, theme, render_opts, view_cfg, highlight_row=selected)
            fig.suptitle(
                f"Placement {k}/{up_to.order}: {selected.class_name}",
                color=theme.fg,
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )
            frame = frame_dir / f"planning_storyboard_{k:02d}.png"
            fig.savefig(frame, dpi=dpi, bbox_inches="tight", facecolor=theme.figure_bg)
            plt.close(fig)
            frame_paths.append(frame)
        gif = _compile_gif(frame_paths, out_dir / "planning_storyboard.gif", gif_duration_ms, gif_loop)
        return ([*frame_paths] if save_frames else []) + [gif]
    finally:
        if tmp is not None:
            tmp.cleanup()


def _box_with_center(box: Box3D, center: np.ndarray) -> Box3D:
    return _make_box(center, box.size, label=box.label)


def _save_descent_gif(
    data: StoryboardData,
    moving: SnapshotRow,
    out_dir: Path,
    theme: Theme,
    render_opts: RenderOptions,
    view_cfg: ViewConfig | None,
    *,
    dpi: int,
    save_frames: bool,
    gif_duration_ms: int,
    gif_loop: int,
) -> list[Path]:
    frame_dir, tmp = _frame_output_dir(out_dir, "frames_descent", save_frames)
    frame_paths: list[Path] = []
    try:
        rows_before = [row for row in data.rows_placed if (row.order or 0) < (moving.order or 0)]
        if moving.raw_box is None or moving.padded_box is None:
            return []
        final_center = moving.raw_box.center.copy()
        hover_z = max(data.bag.z_max + 0.5 * moving.raw_box.size[2] + 40.0, final_center[2] + 160.0)
        for i, t in enumerate(np.linspace(0.0, 1.0, 20), start=1):
            fig = plt.figure(figsize=(8.4, 7.0), facecolor=theme.figure_bg)
            ax = fig.add_subplot(111, projection="3d")
            _style_3d(ax, theme, render_opts.minimal)
            _draw_bag_3d(ax, data.bag, theme)
            for row in rows_before:
                if render_opts.show_raw and row.raw_box is not None:
                    _draw_box_3d(ax, row.raw_box, color=row.color, face_alpha=0.12, edge_alpha=0.42, linewidth=0.9, linestyle="-")
                if render_opts.show_padded and row.padded_box is not None:
                    _draw_box_3d(ax, row.padded_box, color=row.color, face_alpha=0.0, edge_alpha=0.30, linewidth=0.8, linestyle="--")

            moving_center = final_center.copy()
            moving_center[2] = (1.0 - float(t)) * hover_z + float(t) * final_center[2]
            moving_raw = _box_with_center(moving.raw_box, moving_center)
            moving_pad = _box_with_center(moving.padded_box, moving_center)
            if render_opts.show_raw:
                _draw_box_3d(ax, moving_raw, color=theme.next_color, face_alpha=0.34, edge_alpha=1.0, linewidth=1.8, linestyle="-")
            if render_opts.show_padded:
                _draw_box_3d(ax, moving_pad, color=theme.next_color, face_alpha=0.0, edge_alpha=0.95, linewidth=1.2, linestyle="--")

            pts = np.vstack([_all_points(rows_before + [moving], data.bag), _box_corners(_box_with_center(moving.raw_box, np.array([final_center[0], final_center[1], hover_z], dtype=np.float64)))])
            _set_equal_3d(ax, pts, margin_frac=0.08)
            _apply_view_config(ax, view_cfg)
            ax.text2D(
                0.03,
                0.94,
                f"Vertical descent into planned bag cell\nplacement {moving.order}: {moving.class_name}",
                transform=ax.transAxes,
                color=theme.fg,
                fontsize=9,
                va="top",
                bbox=dict(boxstyle="round,pad=0.38", facecolor=theme.annotation_bg, edgecolor=theme.annotation_edge, alpha=0.86),
            )
            fig.suptitle(f"Descent storyboard | {data.info.run_timestamp}", color=theme.fg, fontsize=13, fontweight="bold")
            frame = frame_dir / f"descent_{i:02d}.png"
            fig.savefig(frame, dpi=dpi, bbox_inches="tight", facecolor=theme.figure_bg)
            plt.close(fig)
            frame_paths.append(frame)
        gif = _compile_gif(frame_paths, out_dir / "planning_descent.gif", gif_duration_ms, gif_loop)
        return ([*frame_paths] if save_frames else []) + [gif]
    finally:
        if tmp is not None:
            tmp.cleanup()


class _TuneViewSession:
    def __init__(
        self,
        *,
        data: StoryboardData,
        rows: list[SnapshotRow],
        highlight: SnapshotRow,
        theme_name: str,
        view_cfg: ViewConfig | None,
        save_path: Path,
    ):
        self.data = data
        self.rows = rows
        self.highlight = highlight
        self.view_cfg = view_cfg or ViewConfig(theme=theme_name)
        self.theme_name = self.view_cfg.theme if self.view_cfg.theme in THEMES else theme_name
        self.theme = THEMES[self.theme_name]
        self.save_path = save_path
        self.render_opts = RenderOptions(
            show_labels=bool(self.view_cfg.show_labels),
            show_padded=bool(self.view_cfg.show_padded),
            show_raw=bool(self.view_cfg.show_raw),
            show_pointclouds=bool(self.view_cfg.show_pointclouds),
            point_size=float(self.view_cfg.point_size),
            minimal=False,
        )

        self.fig = plt.figure(figsize=(9.0, 7.8), facecolor=self.theme.figure_bg)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._last_saved: Path | None = None
        self._redraw()

    def _status_lines(self) -> list[str]:
        return [
            "tune-view keys: v save view  p pointclouds  a raw  d padded  l labels",
            "theme n swap theme  =/- point size  q close",
            f"save target: {self.save_path}",
            f"rows shown: {len(self.rows)}   highlight: placement {self.highlight.order} {self.highlight.class_name}",
        ]

    def _redraw(self) -> None:
        self.ax.clear()
        _render_bag_state_3d(self.ax, self.rows, self.data.bag, self.theme, self.render_opts, self.view_cfg, self.highlight)
        self.ax.set_title("Interactive planning 3D view", color=self.theme.fg, fontsize=11, pad=8)
        self.fig.suptitle(f"Tune view | {self.data.info.run_timestamp}", color=self.theme.fg, fontsize=15, fontweight="bold")
        self.ax.text2D(
            0.02,
            0.97,
            "\n".join(self._status_lines()),
            transform=self.ax.transAxes,
            color=self.theme.fg,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=self.theme.annotation_bg, edgecolor=self.theme.annotation_edge, alpha=0.86),
        )
        if self._last_saved is not None:
            self.ax.text2D(0.02, 0.05, f"saved: {self._last_saved}", transform=self.ax.transAxes, color=self.theme.next_color, fontsize=8, va="bottom")
        self.fig.canvas.draw_idle()

    def _current_view(self) -> ViewConfig:
        return _capture_view_config(self.ax, self.theme_name, self.render_opts)

    def _save(self) -> None:
        current = self._current_view()
        self._last_saved = _save_view_config(self.save_path, current)
        print(f"[VIEW] saved {self._last_saved}")
        self.view_cfg = current
        self._redraw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "p":
            self.render_opts.show_pointclouds = not self.render_opts.show_pointclouds
        elif key == "a":
            self.render_opts.show_raw = not self.render_opts.show_raw
        elif key == "d":
            self.render_opts.show_padded = not self.render_opts.show_padded
        elif key == "l":
            self.render_opts.show_labels = not self.render_opts.show_labels
        elif key == "n":
            self.theme_name = "paper" if self.theme_name == "slide_dark" else "slide_dark"
            self.theme = THEMES[self.theme_name]
        elif key in ("=", "+"):
            self.render_opts.point_size = min(8.0, self.render_opts.point_size + 0.2)
        elif key in ("-", "_"):
            self.render_opts.point_size = max(0.4, self.render_opts.point_size - 0.2)
        elif key == "v":
            self._save()
            return
        elif key in ("q", "escape"):
            plt.close(self.fig)
            return
        else:
            return
        self.view_cfg = self._current_view()
        self._redraw()

    def show(self) -> None:
        plt.show()


def _default_view_save_path(run_dir: Path, selected: SnapshotRow, out_dir: Path) -> Path:
    name = f"{run_dir.name}_placement_{selected.order or selected.manifest_row}.json"
    return DEFAULT_VIEW_CONFIG_DIR / name


def _print_summary(
    *,
    mode: str,
    data: StoryboardData,
    pointcloud_summary: list[str],
    extra_notes: list[str],
    output_paths: list[Path],
    view_config_path: Path | None,
) -> None:
    print("[SUMMARY]")
    print(f"  mode: {mode}")
    print(f"  run: {data.info.run_timestamp}")
    print(f"  script: {data.info.script}")
    print(f"  planner: {data.info.planner_sequence}")
    print(f"  total manifest rows: {data.info.total_rows}")
    print(f"  placed rows reconstructed: {len(data.rows_placed)}")
    print(f"  manifest placed_boxes count: {data.info.placed_boxes_count}")
    print(f"  bag source: {data.bag.source}")
    if data.normalized_notes:
        print(f"  normalized raw AABB centers: {len(data.normalized_notes)}")
    if pointcloud_summary:
        ok = sum(1 for line in pointcloud_summary if "loaded" in line)
        print(f"  point clouds loaded: {ok}")
    if view_config_path is not None:
        print(f"  view config: {view_config_path}")
    if extra_notes:
        for note in extra_notes:
            print(f"  note: {note}")
    print("[OUTPUTS]")
    for path in output_paths:
        print(f"  {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline storyboard figure maker for saved wet-run snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None, type=Path, help="Run snapshot directory containing manifest.json")
    parser.add_argument("--mode", choices=("perception", "planning", "tune-view"), default="planning", help="Storyboard mode")
    parser.add_argument("--out-dir", default=None, type=Path, help="Output directory")
    parser.add_argument("--row", type=int, default=None, help="Manifest row selector for perception mode")
    parser.add_argument("--object-index", default=None, help="Manifest object_i selector for perception mode")
    parser.add_argument("--up-to-row", type=int, default=None, help="Placed cut-off row/order for planning and tune-view")
    parser.add_argument("--placement-step", type=int, default=None, help="Alias for --up-to-row")
    parser.add_argument("--include-pointcloud", action="store_true", help="Load and render saved point clouds when available")
    parser.add_argument("--view-config", type=Path, default=None, help="Load a saved view configuration JSON")
    parser.add_argument("--theme", choices=sorted(THEMES), default="slide_dark", help="Visual theme")
    parser.add_argument("--save-panel-crops", action="store_true", help="Save panel crops in addition to the combined figure")
    parser.add_argument("--make-gif", action="store_true", help="Also render planning_storyboard.gif in planning mode")
    parser.add_argument("--make-descent-gif", action="store_true", help="Also render planning_descent.gif in planning mode")
    parser.add_argument("--save-frames", action="store_true", help="Keep GIF frame directories")
    parser.add_argument("--gif-duration-ms", type=int, default=750, help="GIF frame duration in milliseconds")
    parser.add_argument("--gif-loop", type=int, default=0, help="GIF loop count; 0 means loop forever")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="Static PNG and GIF frame DPI")
    parser.add_argument("--minimal", action="store_true", help="Reduce titles and labels")
    parser.add_argument("--no-labels", action="store_true", help="Hide object labels in 3D scenes")
    parser.add_argument("--annotated", action="store_true", help="Keep titles and labels in static exports")
    parser.add_argument("--view-name", default=None, help="Optional name for tune-view saved config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.annotated and args.mode != "tune-view":
        args.minimal = True
        args.no_labels = True
    args.dpi = max(50, int(args.dpi))
    args.gif_duration_ms = max(20, int(args.gif_duration_ms))
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve_default_run_dir()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _resolve_default_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_manifest(run_dir)
    view_cfg = _load_view_config(args.view_config)
    theme = _theme_from_args(args.theme, view_cfg)
    render_opts = RenderOptions(
        show_labels=not args.no_labels and not (view_cfg is not None and not view_cfg.show_labels),
        show_padded=True if view_cfg is None else bool(view_cfg.show_padded),
        show_raw=True if view_cfg is None else bool(view_cfg.show_raw),
        show_pointclouds=bool(args.include_pointcloud) if view_cfg is None else bool(view_cfg.show_pointclouds and args.include_pointcloud),
        point_size=1.6 if view_cfg is None else float(view_cfg.point_size),
        minimal=bool(args.minimal),
    )

    pointcloud_summary: list[str] = []
    if args.include_pointcloud:
        loaded, pointcloud_summary = _attach_pointclouds(data.rows_placed, run_dir)
        print(f"[POINTCLOUD] loaded overlays for {loaded} placed row(s)")

    output_paths: list[Path] = []
    extra_notes: list[str] = []

    if args.mode == "perception":
        row = _resolve_row_selector(data, args.row, args.object_index)
        output_paths.extend(
            _render_perception_storyboard(
                data,
                run_dir,
                row,
                out_dir,
                theme,
                render_opts,
                view_cfg,
                dpi=args.dpi,
                save_panel_crops=bool(args.save_panel_crops),
            )
        )
        if row.pointcloud is None:
            extra_notes.append("perception mode rendered without point cloud overlay")
        _print_summary(
            mode=args.mode,
            data=data,
            pointcloud_summary=pointcloud_summary,
            extra_notes=extra_notes,
            output_paths=output_paths,
            view_config_path=args.view_config.resolve() if args.view_config else None,
        )
        return 0

    up_to_value = args.up_to_row if args.up_to_row is not None else args.placement_step
    selected = _resolve_placed_cutoff(data, up_to_value)
    rows_up_to = data.rows_placed[: int(selected.order or 0)]

    if args.mode == "planning":
        paths, planning_notes = _render_planning_storyboard(
            data,
            run_dir,
            selected,
            rows_up_to,
            out_dir,
            theme,
            render_opts,
            view_cfg,
            dpi=args.dpi,
            save_panel_crops=bool(args.save_panel_crops),
        )
        output_paths.extend(paths)
        extra_notes.extend(planning_notes)
        if args.make_gif:
            output_paths.extend(
                _save_storyboard_gif(
                    data,
                    run_dir,
                    selected,
                    out_dir,
                    theme,
                    render_opts,
                    view_cfg,
                    dpi=args.dpi,
                    save_frames=bool(args.save_frames),
                    gif_duration_ms=args.gif_duration_ms,
                    gif_loop=args.gif_loop,
                )
            )
        if args.make_descent_gif:
            output_paths.extend(
                _save_descent_gif(
                    data,
                    selected,
                    out_dir,
                    theme,
                    render_opts,
                    view_cfg,
                    dpi=args.dpi,
                    save_frames=bool(args.save_frames),
                    gif_duration_ms=args.gif_duration_ms,
                    gif_loop=args.gif_loop,
                )
            )
        _print_summary(
            mode=args.mode,
            data=data,
            pointcloud_summary=pointcloud_summary,
            extra_notes=extra_notes,
            output_paths=output_paths,
            view_config_path=args.view_config.resolve() if args.view_config else None,
        )
        return 0

    # tune-view
    save_path = args.view_config.resolve() if args.view_config is not None else _default_view_save_path(run_dir, selected, out_dir)
    if args.view_name:
        save_path = DEFAULT_VIEW_CONFIG_DIR / f"{args.view_name}.json"
    session = _TuneViewSession(
        data=data,
        rows=rows_up_to,
        highlight=selected,
        theme_name=theme.name,
        view_cfg=view_cfg,
        save_path=save_path,
    )
    session.show()
    _print_summary(
        mode=args.mode,
        data=data,
        pointcloud_summary=pointcloud_summary,
        extra_notes=["interactive window closed; save with key 'v' inside tune-view"],
        output_paths=[],
        view_config_path=save_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
