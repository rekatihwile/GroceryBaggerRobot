from __future__ import annotations

"""Offline presentation visualizer for wet-run grocery bagger snapshots.

This script only reads saved run snapshot data. It does not open cameras,
serial ports, Teensy connections, YOLO, RAFT, or robot hardware.
"""

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    run_output_dir,
    save_figure_bundle,
)
from scripts.capstone.pointcloud_color import (
    load_stereo_calibration,
    project_bundle_camera_points_to_uv,
)

from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm

_PACKING_VIS_WORKSPACE = get_workspace_filter_config("wet_run")
_PACKING_VIS_X_MIN, _PACKING_VIS_X_MAX, _PACKING_VIS_Y_MIN, _PACKING_VIS_Y_MAX = workspace_bounds_mm(_PACKING_VIS_WORKSPACE)
_PACKING_VIS_MIN_VOL_MM3: float = 1.0
_PACKING_VIS_MAX_VOL_MM3: float = 3_000_000.0
_PACKING_VIS_GRIPPER_OFFSET_MM: float = 130.0
_PACKING_VIS_Z_MAX_MM: float = 275.0

BAG_SCENE_NAME = "New Bag Test"
SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
DEFAULT_BAG_HEIGHT_MM = 250.0
MAX_POINTS_2D = 900
MAX_POINTS_3D = 1400
DEFAULT_DESCENT_FRAMES = 22

# VS Code IDE defaults. Leave as None to auto-select newest run snapshot.
# Copy/paste your run folder path here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_184606"
IDE_DEFAULT_RUN_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
IDE_DEFAULT_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None

# Copy/paste your output root here (or keep None for paper_figure_sources).
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
    return run_output_dir(run_dir) / "packing"


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
    newest: str


THEMES = {
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
        newest="#facc15",
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
        newest="#dc2626",
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
class PlacedRow:
    order: int
    manifest_row: int
    object_i: Any
    class_name: str
    place_result: str
    manifest_raw_box: Box3D
    raw_box: Box3D
    padded_box: Box3D
    place_xy_mm: np.ndarray | None
    destination_surface_z_mm: float | None
    points_cam: str | None
    stereo_left: str | None
    left_overlay: str | None
    overhead: str | None
    disparity: str | None
    color: str
    pointcloud: PointCloudOverlay | None = None


@dataclass
class ManifestInfo:
    path: Path
    run_timestamp: str
    script: str
    place_planning_sequence: str
    placed_boxes_count: int
    total_object_rows: int
    skipped_rows: list[dict[str, Any]]


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

    try:
        from planning.aabb_utils import make_aabb_from_center_size

        box = make_aabb_from_center_size(center, size, label=label)
        return Box3D(
            center=np.asarray(box.center_xyz_mm, dtype=np.float64),
            size=np.asarray(box.size_xyz_mm, dtype=np.float64),
            min_xyz=np.asarray(box.min_xyz_mm, dtype=np.float64),
            max_xyz=np.asarray(box.max_xyz_mm, dtype=np.float64),
            label=str(getattr(box, "label", label)),
        )
    except Exception:
        half = 0.5 * size
        return Box3D(center=center, size=size, min_xyz=center - half, max_xyz=center + half, label=label)


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
        faces = Poly3DCollection(_box_faces(corners), alpha=face_alpha)
        faces.set_facecolor(color)
        faces.set_edgecolor("none")
        ax.add_collection3d(faces)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
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


def _all_points(rows: list[PlacedRow], bag: BagVolume | None = None) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for row in rows:
        chunks.append(_box_corners(row.raw_box))
        chunks.append(_box_corners(row.padded_box))
        if row.pointcloud is not None and len(row.pointcloud.points_xyz):
            chunks.append(row.pointcloud.points_xyz)
    if bag is not None:
        chunks.append(_box_corners(_bag_box(bag)))
    return np.vstack(chunks) if chunks else np.zeros((1, 3), dtype=np.float64)


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


def _set_equal_3d(ax, pts: np.ndarray, margin_frac: float = 0.10) -> None:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
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


def load_manifest(run_dir: Path) -> tuple[ManifestInfo, list[PlacedRow]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("objects") or []
    if not isinstance(rows, list):
        raise ValueError("manifest field 'objects' is not a list")

    names = [str(row.get("class_name") or row.get("detection_class") or "object") for row in rows]
    colors = _class_colors(names)
    placed: list[PlacedRow] = []
    skipped: list[dict[str, Any]] = []

    for row_i, row in enumerate(rows, start=1):
        result = str(row.get("place_result", "")).lower()
        class_name = str(row.get("class_name") or row.get("detection_class") or "object")
        reason: str | None = None
        if result != "placed":
            reason = f"place_result={result or 'missing'}"
        required = [
            "raw_box_center_xyz_mm",
            "raw_box_size_xyz_mm",
            "padded_box_center_xyz_mm",
            "padded_box_size_xyz_mm",
        ]
        missing = [key for key in required if row.get(key) is None]
        if missing and reason is None:
            reason = "missing " + ", ".join(missing)

        # Robot-side candidate gates: volume range and stereo phantom Z check.
        # Platform XY bounds are implicitly guaranteed by place_result=="placed".
        if reason is None and row.get("raw_box_size_xyz_mm") is not None:
            try:
                sz = [float(v) for v in row["raw_box_size_xyz_mm"]]
                vol = sz[0] * sz[1] * sz[2]
                if vol < _PACKING_VIS_MIN_VOL_MM3 or vol > _PACKING_VIS_MAX_VOL_MM3:
                    reason = f"volume {vol:.0f} mm3 outside robot-side range"
            except Exception:
                pass
        if reason is None and row.get("raw_box_center_xyz_mm") is not None and row.get("raw_box_size_xyz_mm") is not None:
            try:
                cz = float(row["raw_box_center_xyz_mm"][2])
                sz_z = float(row["raw_box_size_xyz_mm"][2])
                top_z = cz + 0.5 * sz_z
                if top_z + _PACKING_VIS_GRIPPER_OFFSET_MM > _PACKING_VIS_Z_MAX_MM:
                    reason = f"stereo phantom: top_z {top_z:.0f}+offset>{_PACKING_VIS_Z_MAX_MM:.0f}"
            except Exception:
                pass

        if reason is not None:
            skipped.append(
                {
                    "manifest_row": row_i,
                    "object_i": row.get("object_i"),
                    "class_name": class_name,
                    "place_result": row.get("place_result"),
                    "reason": reason,
                }
            )
            continue

        try:
            manifest_raw = _make_box(
                row["raw_box_center_xyz_mm"],
                row["raw_box_size_xyz_mm"],
                label=f"{class_name}_raw_manifest",
            )
            padded = _make_box(
                row["padded_box_center_xyz_mm"],
                row["padded_box_size_xyz_mm"],
                label=f"{class_name}_padded",
            )
        except Exception as exc:
            skipped.append(
                {
                    "manifest_row": row_i,
                    "object_i": row.get("object_i"),
                    "class_name": class_name,
                    "place_result": row.get("place_result"),
                    "reason": f"invalid box: {exc}",
                }
            )
            continue

        place_xy = _finite_vec(row.get("place_xy_mm"), 2)
        # Visualization target: in saved wet-run manifests, padded boxes are in
        # the bag pose while raw_box_center_xyz_mm can be the original pick pose.
        # Start with the manifest raw box and normalize to bag pose after loading
        # the active bag scene.
        raw_visual = _make_box(manifest_raw.center.copy(), manifest_raw.size.copy(), label=f"{class_name}_raw")
        placed.append(
            PlacedRow(
                order=len(placed) + 1,
                manifest_row=row_i,
                object_i=row.get("object_i"),
                class_name=class_name,
                place_result=str(row.get("place_result")),
                manifest_raw_box=manifest_raw,
                raw_box=raw_visual,
                padded_box=padded,
                place_xy_mm=None if place_xy is None else place_xy.copy(),
                destination_surface_z_mm=_finite_float(row.get("destination_surface_z_mm")),
                points_cam=row.get("points_cam"),
                stereo_left=row.get("stereo_left"),
                left_overlay=row.get("left_overlay"),
                overhead=row.get("overhead"),
                disparity=row.get("disparity"),
                color=colors[class_name],
            )
        )

    info = ManifestInfo(
        path=manifest_path,
        run_timestamp=str(data.get("run_timestamp") or run_dir.name.replace("run_", "")),
        script=str(data.get("script") or "unknown"),
        place_planning_sequence=str(data.get("place_planning_sequence") or "unknown"),
        placed_boxes_count=len(data.get("placed_boxes") or []),
        total_object_rows=len(rows),
        skipped_rows=skipped,
    )
    print(f"[MANIFEST] {len(placed)} placed row(s) visualized from {manifest_path}")
    print(f"[MANIFEST] skipped/audit row(s): {len(skipped)}")
    return info, placed


def _bag_height_from_config() -> float:
    try:
        from config.place import DEFAULT_PLACE

        return float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", DEFAULT_BAG_HEIGHT_MM))
    except Exception:
        return DEFAULT_BAG_HEIGHT_MM


def load_bag_volume(rows: list[PlacedRow]) -> BagVolume:
    pts = np.vstack([_box_corners(row.padded_box) for row in rows]) if rows else np.zeros((1, 3))
    box_min = pts.min(axis=0)
    box_max = pts.max(axis=0)
    bag_height = _bag_height_from_config()
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
        print(f"[BAG WARN] failed to load {BAG_SCENE_NAME!r}: {exc}; inferring bounds from padded boxes")
        return BagVolume(
            name="inferred_bag",
            x_min=float(box_min[0] - margin_xy),
            x_max=float(box_max[0] + margin_xy),
            y_min=float(box_min[1] - margin_xy),
            y_max=float(box_max[1] + margin_xy),
            z_min=float(box_min[2] - margin_z),
            z_max=float(box_max[2] + margin_z),
            source="inferred from manifest boxes",
        )


def normalize_raw_boxes_to_bag_pose(rows: list[PlacedRow], bag: BagVolume) -> list[str]:
    notes: list[str] = []
    for row in rows:
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
                f"row {row.manifest_row}: raw AABB center normalized from pick pose "
                f"({raw_xy[0]:.1f},{raw_xy[1]:.1f}) to placed pose ({center[0]:.1f},{center[1]:.1f})"
            )
    if notes:
        print(f"[AABB] normalized {len(notes)} raw box center(s) to placed bag pose for visualization")
    return notes


def _npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        return list(data.files)


def _load_bundle() -> dict[str, np.ndarray] | None:
    if not BUNDLE_PATH.exists():
        print(f"[POINTCLOUD WARN] calibration bundle not found: {BUNDLE_PATH}")
        return None
    try:
        with np.load(BUNDLE_PATH, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        print(f"[POINTCLOUD WARN] failed to load calibration bundle: {exc}")
        return None


def _pick_point_array(data: Any, keys: list[str]) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
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


def _load_image_rgb(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        import matplotlib.image as mpimg

        img = mpimg.imread(path)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.dtype.kind in "ui":
            img = img.astype(np.float32) / 255.0
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return np.asarray(img, dtype=np.float32)
    except Exception:
        return None


def _colors_from_uv(image_rgb: np.ndarray | None, uv: np.ndarray | None, n: int) -> np.ndarray | None:
    if image_rgb is None or uv is None or len(uv) != n:
        return None
    h, w = image_rgb.shape[:2]
    xy = np.rint(uv).astype(int)
    ok = (0 <= xy[:, 0]) & (xy[:, 0] < w) & (0 <= xy[:, 1]) & (xy[:, 1] < h)
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, :] = 0.8
    colors[ok] = image_rgb[xy[ok, 1], xy[ok, 0], :3]
    return colors


def _row_color_image(run_dir: Path, row: PlacedRow) -> np.ndarray | None:
    if row.stereo_left:
        image_rgb = _load_image_rgb(run_dir / row.stereo_left)
        if image_rgb is not None:
            return image_rgb
    if row.left_overlay:
        image_rgb = _load_image_rgb(run_dir / row.left_overlay)
        if image_rgb is not None:
            return image_rgb
    return None


def _shift_points_to_box(points_robot: np.ndarray, row: PlacedRow) -> tuple[np.ndarray, np.ndarray]:
    points_robot = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(points_robot), axis=1)
    points_robot = points_robot[finite]
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


def attach_pointclouds(rows: list[PlacedRow], run_dir: Path, *, selected_only: PlacedRow | None = None) -> None:
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
    for row in targets:
        if row is None or not row.points_cam:
            continue
        p = run_dir / row.points_cam
        if not p.exists():
            print(f"[POINTCLOUD WARN] row {row.manifest_row}: missing {p.name}")
            continue
        try:
            with np.load(p, allow_pickle=False) as data:
                keys = list(data.files)
                print(f"[POINTCLOUD] row {row.manifest_row} {p.name} keys={keys}")
                robot_pts, robot_key = _pick_point_array(data, ["points_robot", "points_robot_xyz", "robot_points", "points_xyz_robot"])
                cam_pts = None
                cam_key = None
                if robot_key not in ("points_robot", "points_robot_xyz", "robot_points", "points_xyz_robot"):
                    cam_pts = robot_pts
                    cam_key = robot_key
                    robot_pts = None
                    robot_key = None
                if robot_pts is None:
                    cam_pts, cam_key = _pick_point_array(data, ["points_cam", "points_camera", "cam_points", "points_xyz_cam"])
                    if cam_pts is None:
                        raise ValueError("no Nx3 points found")
                    if transform is None or bundle is None:
                        raise ValueError(f"{cam_key} is camera-frame and no calibration transform is available")
                    robot_pts = transform(cam_pts, bundle)
                    source = f"{cam_key}->robot"
                else:
                    source = str(robot_key)

                uv = _pick_uv_array(data)
                if uv is None and cam_pts is not None:
                    uv = project_bundle_camera_points_to_uv(cam_pts, stereo_calib)
                image_rgb = _row_color_image(run_dir, row)
                point_colors = _colors_from_uv(image_rgb, uv, len(robot_pts))
                if point_colors is None:
                    print(f"[POINTCLOUD WARN] row {row.manifest_row}: original-photo RGB unavailable; skipping cloud")
                    continue
                shifted, inside = _shift_points_to_box(robot_pts, row)
                if len(shifted) < 20:
                    print(f"[POINTCLOUD WARN] row {row.manifest_row}: fewer than 20 shifted points inside raw AABB")
                    continue
                if point_colors is not None and len(point_colors) == len(inside):
                    point_colors = point_colors[inside]
                row.pointcloud = PointCloudOverlay(points_xyz=shifted, colors=point_colors, source=source, keys=keys)
                loaded += 1
        except Exception as exc:
            print(f"[POINTCLOUD WARN] row {row.manifest_row}: failed to load {p.name}: {exc}")
    if loaded:
        print(f"[POINTCLOUD] attached overlays for {loaded} row(s)")


def _sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def _axis_bounds_2d(rows: list[PlacedRow], bag: BagVolume, margin: float = 25.0) -> tuple[float, float, float, float]:
    pts = _all_points(rows, bag)
    return (
        min(float(pts[:, 0].min()), bag.x_min) - margin,
        max(float(pts[:, 0].max()), bag.x_max) + margin,
        min(float(pts[:, 1].min()), bag.y_min) - margin,
        max(float(pts[:, 1].max()), bag.y_max) + margin,
    )


def _label_xy(row: PlacedRow) -> tuple[float, float]:
    offsets = [(0.0, 0.0), (-0.18, 0.10), (0.18, -0.10), (-0.16, -0.16), (0.16, 0.16), (0.0, 0.22), (0.0, -0.22)]
    ox, oy = offsets[(row.order - 1) % len(offsets)]
    return (
        float(row.raw_box.center[0] + ox * row.raw_box.size[0]),
        float(row.raw_box.center[1] + oy * row.raw_box.size[1]),
    )


def _format_annotation(info: ManifestInfo, n_placed: int) -> str:
    return "\n".join(
        [
            f"run: {info.run_timestamp}",
            f"script: {info.script}",
            f"planner: {info.place_planning_sequence}",
            f"placed rows visualized: {n_placed}",
        ]
    )


def render_topdown(
    rows: list[PlacedRow],
    bag: BagVolume,
    info: ManifestInfo,
    theme: Theme,
    *,
    title: str | None = None,
    highlight_order: int | None = None,
    no_labels: bool = False,
    minimal: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor=theme.figure_bg)
    ax.set_facecolor(theme.axes_bg)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(not minimal, color=theme.grid, linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)

    bag_rect = Rectangle(
        (bag.x_min, bag.y_min),
        bag.width,
        bag.depth,
        facecolor=theme.bag_fill,
        edgecolor=theme.bag_edge,
        linewidth=2.2,
        alpha=0.85,
        zorder=1,
    )
    ax.add_patch(bag_rect)

    for row in rows:
        is_newest = highlight_order is not None and row.order == highlight_order
        alpha = 0.50 if is_newest else 0.32
        edge_lw = 3.0 if is_newest else 1.55
        pad_lw = 2.4 if is_newest else 1.25
        edge_color = theme.newest if is_newest else row.color

        if row.pointcloud is not None and row.pointcloud.colors is not None:
            idx = _sample_indices(len(row.pointcloud.points_xyz), MAX_POINTS_2D, 1000 + row.order)
            colors = row.pointcloud.colors[idx]
            ax.scatter(
                row.pointcloud.points_xyz[idx, 0],
                row.pointcloud.points_xyz[idx, 1],
                s=3.0 if is_newest else 2.0,
                c=colors,
                alpha=0.34,
                linewidths=0,
                zorder=2,
            )

        raw = row.raw_box
        pad = row.padded_box
        ax.add_patch(
            Rectangle(
                (raw.min_xyz[0], raw.min_xyz[1]),
                raw.size[0],
                raw.size[1],
                facecolor=row.color,
                edgecolor=edge_color,
                linewidth=edge_lw,
                alpha=alpha,
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (pad.min_xyz[0], pad.min_xyz[1]),
                pad.size[0],
                pad.size[1],
                facecolor="none",
                edgecolor=edge_color,
                linewidth=pad_lw,
                linestyle=(0, (4, 3)),
                alpha=0.95,
                zorder=4,
            )
        )
        if not no_labels and not minimal:
            x, y = _label_xy(row)
            ax.text(
                x,
                y,
                f"{row.order}\n{_short_label(row.class_name)}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=theme.label_fg,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=theme.label_bg, edgecolor="none", alpha=0.86),
                zorder=5,
            )

    x0, x1, y0, y1 = _axis_bounds_2d(rows, bag)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    if not minimal:
        ax.set_title(title or f"Bag-local AABB packing | {info.run_timestamp}", color=theme.fg, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Robot X (mm)", color=theme.muted, fontsize=9)
        ax.set_ylabel("Robot Y (mm)", color=theme.muted, fontsize=9)
        ax.text(
            0.985,
            0.985,
            _format_annotation(info, len(rows)),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            color=theme.fg,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=theme.annotation_bg, edgecolor=theme.annotation_edge, alpha=0.86),
        )
        ax.text(
            0.015,
            0.02,
            "filled = raw AABB   dashed = padded AABB",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=theme.muted,
        )
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    ax.tick_params(colors=theme.muted, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(theme.grid)
    fig.tight_layout()
    return fig


def _style_3d(ax, theme: Theme, minimal: bool) -> None:
    ax.set_facecolor(theme.axes_bg)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(theme.axes_bg)
        pane.set_edgecolor(theme.grid)
        pane.set_alpha(0.95)
    ax.grid(not minimal, color=theme.grid, linewidth=0.65, alpha=0.7)
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


def render_3d(
    rows: list[PlacedRow],
    bag: BagVolume,
    info: ManifestInfo,
    theme: Theme,
    *,
    title: str | None = None,
    no_labels: bool = False,
    minimal: bool = False,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.8, 7.2), facecolor=theme.figure_bg)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax, theme, minimal)
    _draw_bag_3d(ax, bag, theme)

    for row in rows:
        if row.pointcloud is not None and row.pointcloud.colors is not None:
            idx = _sample_indices(len(row.pointcloud.points_xyz), MAX_POINTS_3D, 2000 + row.order)
            colors = row.pointcloud.colors[idx]
            ax.scatter(
                row.pointcloud.points_xyz[idx, 0],
                row.pointcloud.points_xyz[idx, 1],
                row.pointcloud.points_xyz[idx, 2],
                s=1.5,
                c=colors,
                alpha=0.30,
                depthshade=False,
            )
        _draw_box_3d(ax, row.raw_box, color=row.color, face_alpha=0.25, edge_alpha=0.95, linewidth=1.25, linestyle="-")
        _draw_box_3d(ax, row.padded_box, color=row.color, face_alpha=0.0, edge_alpha=0.72, linewidth=1.0, linestyle="--")
        if not no_labels and not minimal:
            ax.text(
                row.raw_box.center[0],
                row.raw_box.center[1],
                row.raw_box.max_xyz[2] + 8.0,
                f"{row.order} {_short_label(row.class_name, 13)}",
                color=theme.fg,
                fontsize=6.8,
                ha="center",
                va="bottom",
            )

    _set_equal_3d(ax, _all_points(rows, bag), margin_frac=0.08)
    ax.view_init(elev=24, azim=-54)
    if not minimal:
        fig.suptitle(title or f"3D bag AABB reconstruction | {info.run_timestamp}", color=theme.fg, fontsize=14, fontweight="bold")
        ax.text2D(
            0.02,
            0.96,
            "solid/translucent = raw AABB    dashed = padded AABB",
            transform=ax.transAxes,
            fontsize=7.7,
            color=theme.muted,
            va="top",
        )
    fig.tight_layout()
    return fig


def _box_with_center(box: Box3D, center: np.ndarray, label: str | None = None) -> Box3D:
    return _make_box(center, box.size, label=box.label if label is None else label)


def render_descent_frame(
    rows_before: list[PlacedRow],
    moving: PlacedRow,
    bag: BagVolume,
    info: ManifestInfo,
    theme: Theme,
    *,
    t: float,
    no_labels: bool,
    minimal: bool,
) -> plt.Figure:
    fig = plt.figure(figsize=(8.4, 7.0), facecolor=theme.figure_bg)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax, theme, minimal)
    _draw_bag_3d(ax, bag, theme)

    for row in rows_before:
        _draw_box_3d(ax, row.raw_box, color=row.color, face_alpha=0.16, edge_alpha=0.48, linewidth=0.9, linestyle="-")
        _draw_box_3d(ax, row.padded_box, color=row.color, face_alpha=0.0, edge_alpha=0.34, linewidth=0.8, linestyle="--")

    final_center = moving.raw_box.center
    hover_z = max(bag.z_max + 0.5 * moving.raw_box.size[2] + 40.0, final_center[2] + 160.0)
    moving_center = final_center.copy()
    moving_center[2] = (1.0 - t) * hover_z + t * final_center[2]
    moving_raw = _box_with_center(moving.raw_box, moving_center, label=moving.raw_box.label)
    moving_pad = _box_with_center(moving.padded_box, moving_center, label=moving.padded_box.label)

    _draw_box_3d(ax, moving_raw, color=theme.newest, face_alpha=0.34, edge_alpha=1.0, linewidth=1.8, linestyle="-")
    _draw_box_3d(ax, moving_pad, color=theme.newest, face_alpha=0.0, edge_alpha=0.95, linewidth=1.25, linestyle="--")

    if not no_labels and not minimal:
        ax.text(
            moving_raw.center[0],
            moving_raw.center[1],
            moving_raw.max_xyz[2] + 10.0,
            f"{moving.order} {_short_label(moving.class_name, 14)}",
            color=theme.fg,
            fontsize=8,
            ha="center",
            va="bottom",
        )
        ax.text2D(
            0.03,
            0.94,
            f"Vertical descent into planned cell\nPlacement {moving.order}: {moving.class_name}",
            transform=ax.transAxes,
            color=theme.fg,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.38", facecolor=theme.annotation_bg, edgecolor=theme.annotation_edge, alpha=0.86),
        )

    scale_rows = rows_before + [moving]
    pts = np.vstack([_all_points(scale_rows, bag), _box_corners(_box_with_center(moving.raw_box, np.array([final_center[0], final_center[1], hover_z])))])
    _set_equal_3d(ax, pts, margin_frac=0.08)
    ax.view_init(elev=24, azim=-54)
    if not minimal:
        fig.suptitle(f"Descent plan | {info.run_timestamp}", color=theme.fg, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _save_fig(fig: plt.Figure, path: Path, dpi: int, theme: Theme) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.figure_bg)
    return path


def save_topdown(rows: list[PlacedRow], bag: BagVolume, info: ManifestInfo, theme: Theme, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    fig = render_topdown(
        rows,
        bag,
        info,
        theme,
        highlight_order=rows[-1].order if args.highlight_last else None,
        no_labels=args.no_labels,
        minimal=args.minimal,
    )
    paths = save_figure_bundle(
        fig,
        out_dir / "packing_executed_aabb_topdown",
        dpi=args.dpi,
        facecolor=theme.figure_bg,
    )
    plt.close(fig)
    return paths


def save_3d(rows: list[PlacedRow], bag: BagVolume, info: ManifestInfo, theme: Theme, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    fig = render_3d(rows, bag, info, theme, no_labels=args.no_labels, minimal=args.minimal)
    paths = save_figure_bundle(
        fig,
        out_dir / "packing_executed_aabb_3d",
        dpi=args.dpi,
        facecolor=theme.figure_bg,
    )
    plt.close(fig)
    return paths


def _compile_gif(frame_paths: list[Path], gif_path: Path, *, duration_ms: int, loop: int) -> Path:
    if not frame_paths:
        raise RuntimeError("No GIF frames were rendered")
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


def save_sequence_gif(rows: list[PlacedRow], bag: BagVolume, info: ManifestInfo, theme: Theme, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    frame_dir, tmp = _frame_output_dir(out_dir, "frames_sequence", args.save_frames)
    frame_paths: list[Path] = []
    try:
        for k in range(1, len(rows) + 1):
            title = f"Placement {k}/{len(rows)}: {rows[k - 1].class_name}"
            fig = render_topdown(
                rows[:k],
                bag,
                info,
                theme,
                title=title,
                highlight_order=k,
                no_labels=args.no_labels,
                minimal=args.minimal,
            )
            frame = frame_dir / f"bag_aabb_sequence_{k:02d}.png"
            _save_fig(fig, frame, args.dpi, theme)
            plt.close(fig)
            frame_paths.append(frame)
        gif = _compile_gif(frame_paths, out_dir / "bag_aabb_sequence.gif", duration_ms=args.gif_duration_ms, loop=args.gif_loop)
        return ([*frame_paths] if args.save_frames else []) + [gif]
    finally:
        if tmp is not None:
            tmp.cleanup()


def _select_descent_row(rows: list[PlacedRow], descent_row: int | None) -> PlacedRow:
    if descent_row is None:
        return rows[-1]
    for row in rows:
        if row.order == descent_row:
            return row
    for row in rows:
        if row.manifest_row == descent_row:
            print(f"[DESCENT] --descent-row matched manifest row {descent_row}; displayed order is {row.order}")
            return row
    raise ValueError(f"--descent-row {descent_row} did not match a placement order or manifest row")


def save_descent_gif(rows: list[PlacedRow], bag: BagVolume, info: ManifestInfo, theme: Theme, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    moving = _select_descent_row(rows, args.descent_row)
    rows_before = [row for row in rows if row.order < moving.order]
    frame_dir, tmp = _frame_output_dir(out_dir, "frames_descent", args.save_frames)
    frame_paths: list[Path] = []
    try:
        for i, t in enumerate(np.linspace(0.0, 1.0, DEFAULT_DESCENT_FRAMES), start=1):
            fig = render_descent_frame(
                rows_before,
                moving,
                bag,
                info,
                theme,
                t=float(t),
                no_labels=args.no_labels,
                minimal=args.minimal,
            )
            frame = frame_dir / f"bag_aabb_descent_{i:02d}.png"
            _save_fig(fig, frame, args.dpi, theme)
            plt.close(fig)
            frame_paths.append(frame)
        gif = _compile_gif(frame_paths, out_dir / "bag_aabb_descent.gif", duration_ms=args.gif_duration_ms, loop=args.gif_loop)
        return ([*frame_paths] if args.save_frames else []) + [gif]
    finally:
        if tmp is not None:
            tmp.cleanup()


def _load_disparity(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "disparity" in data.files:
                arr = np.asarray(data["disparity"], dtype=np.float64)
            else:
                arr = None
                for key in data.files:
                    candidate = np.asarray(data[key])
                    if candidate.ndim == 2:
                        arr = candidate.astype(np.float64)
                        break
                if arr is None:
                    return None
        arr = arr[np.isfinite(arr)]
    except Exception:
        return None
    # Re-open to return full shaped array after validating at least some finite values.
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["disparity"] if "disparity" in data.files else data[data.files[0]], dtype=np.float64)
    except Exception:
        return None


def _imshow_or_placeholder(ax, path: Path | None, title: str, theme: Theme) -> None:
    ax.set_title(title, color=theme.fg, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if path is None or not path.exists():
        ax.text(0.5, 0.5, "missing", ha="center", va="center", color=theme.muted, transform=ax.transAxes)
        ax.set_facecolor(theme.axes_bg)
        return
    img = _load_image_rgb(path)
    if img is None:
        ax.text(0.5, 0.5, "unreadable", ha="center", va="center", color=theme.muted, transform=ax.transAxes)
        ax.set_facecolor(theme.axes_bg)
        return
    ax.imshow(img)


def _select_panel_row(rows: list[PlacedRow], value: int | None) -> PlacedRow:
    if value is None:
        return rows[-1]
    return _select_descent_row(rows, value)


def save_perception_panel(rows: list[PlacedRow], bag: BagVolume, info: ManifestInfo, theme: Theme, run_dir: Path, out_dir: Path, args: argparse.Namespace) -> list[Path]:
    row = _select_panel_row(rows, args.perception_row)
    if row.pointcloud is None:
        attach_pointclouds(rows, run_dir, selected_only=row)

    fig = plt.figure(figsize=(15.6, 4.4), facecolor=theme.figure_bg)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax3d = fig.add_subplot(gs[0, 3], projection="3d")
    for ax in axes:
        ax.set_facecolor(theme.axes_bg)

    _imshow_or_placeholder(axes[0], run_dir / row.stereo_left if row.stereo_left else None, "Stereo left", theme)
    _imshow_or_placeholder(axes[1], run_dir / row.left_overlay if row.left_overlay else None, "Saved YOLO overlay", theme)
    axes[2].set_title("Saved disparity", color=theme.fg, fontsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    disp = _load_disparity(run_dir / row.disparity) if row.disparity else None
    if disp is None:
        axes[2].text(0.5, 0.5, "missing", ha="center", va="center", color=theme.muted, transform=axes[2].transAxes)
        axes[2].set_facecolor(theme.axes_bg)
    else:
        lo, hi = np.nanpercentile(disp, [2.0, 98.0])
        axes[2].imshow(disp, cmap="magma", vmin=lo, vmax=hi)

    _style_3d(ax3d, theme, minimal=True)
    _draw_bag_3d(ax3d, bag, theme)
    if row.pointcloud is not None and row.pointcloud.colors is not None:
        idx = _sample_indices(len(row.pointcloud.points_xyz), MAX_POINTS_3D, 3000 + row.order)
        colors = row.pointcloud.colors[idx]
        ax3d.scatter(
            row.pointcloud.points_xyz[idx, 0],
            row.pointcloud.points_xyz[idx, 1],
            row.pointcloud.points_xyz[idx, 2],
            s=1.8,
            c=colors,
            alpha=0.35,
            depthshade=False,
        )
    _draw_box_3d(ax3d, row.raw_box, color=row.color, face_alpha=0.26, edge_alpha=1.0, linewidth=1.25, linestyle="-")
    _draw_box_3d(ax3d, row.padded_box, color=theme.newest, face_alpha=0.0, edge_alpha=0.85, linewidth=1.0, linestyle="--")
    _set_equal_3d(ax3d, np.vstack([_all_points([row], bag)]), margin_frac=0.10)
    ax3d.view_init(elev=24, azim=-54)
    ax3d.set_title("Point cloud + AABB", color=theme.fg, fontsize=9)

    fig.suptitle(f"Perception snapshot | placement {row.order}: {row.class_name}", color=theme.fg, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.84, bottom=0.06, wspace=0.08)
    paths = [
        _save_fig(fig, out_dir / "perception_panel.png", args.dpi, theme),
        _save_fig(fig, out_dir / "perception_panel.svg", args.dpi, theme),
    ]
    plt.close(fig)
    return paths


def write_summary(
    out_dir: Path,
    info: ManifestInfo,
    rows: list[PlacedRow],
    bag: BagVolume,
    normalized_notes: list[str],
    outputs: list[Path],
) -> list[Path]:
    summary = out_dir / "run_snapshot_visualizer_summary.txt"
    lines = [
        "RUN SNAPSHOT VISUALIZER SUMMARY",
        "=" * 40,
        f"manifest: {info.path}",
        f"run_timestamp: {info.run_timestamp}",
        f"script: {info.script}",
        f"place_planning_sequence: {info.place_planning_sequence}",
        f"total manifest object rows: {info.total_object_rows}",
        f"placed rows visualized: {len(rows)}",
        f"placed_boxes count in manifest: {info.placed_boxes_count}",
        f"bag: {bag.name}",
        f"bag source: {bag.source}",
        f"bag bounds x=[{bag.x_min:.1f},{bag.x_max:.1f}] y=[{bag.y_min:.1f},{bag.y_max:.1f}] z=[{bag.z_min:.1f},{bag.z_max:.1f}]",
        "",
        "Placed rows:",
    ]
    for row in rows:
        pc = "yes" if row.pointcloud is not None else "no"
        lines.append(
            f"  placement {row.order:02d} manifest_row={row.manifest_row:02d} "
            f"object_i={row.object_i!r} class={row.class_name} pointcloud={pc}"
        )
    lines.append("")
    lines.append("Normalization notes:")
    lines.extend([f"  {note}" for note in normalized_notes] or ["  none"])
    lines.append("")
    lines.append("Skipped/audit rows:")
    if info.skipped_rows:
        for row in info.skipped_rows:
            lines.append(
                f"  manifest_row={row['manifest_row']} object_i={row.get('object_i')!r} "
                f"class={row.get('class_name')} result={row.get('place_result')!r} reason={row.get('reason')}"
            )
    else:
        lines.append("  none")
    lines.append("")
    lines.append("Outputs:")
    for path in outputs:
        lines.append(f"  {path}")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    written = [summary]
    if info.skipped_rows:
        audit = out_dir / "missed_pending_audit.txt"
        audit_lines = ["MISSED/PENDING/SKIPPED MANIFEST ROWS", "=" * 40]
        for row in info.skipped_rows:
            audit_lines.append(
                f"manifest_row={row['manifest_row']} object_i={row.get('object_i')!r} "
                f"class={row.get('class_name')} result={row.get('place_result')!r} reason={row.get('reason')}"
            )
        audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
        written.append(audit)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create offline presentation figures/GIFs from a saved wet-run snapshot manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None, type=Path, help="Run snapshot directory containing manifest.json")
    parser.add_argument("--out-dir", default=None, type=Path, help="Output directory for presentation files")
    parser.add_argument("--theme", choices=sorted(THEMES), default="slide_dark", help="Presentation style")
    parser.add_argument("--make-topdown", action="store_true", help="Render bag_aabb_topdown.png/svg")
    parser.add_argument("--make-3d", action="store_true", help="Render bag_aabb_3d.png/svg")
    parser.add_argument("--make-sequence-gif", action="store_true", help="Render bag_aabb_sequence.gif")
    parser.add_argument("--make-descent-gif", action="store_true", help="Render bag_aabb_descent.gif")
    parser.add_argument("--make-perception-panel", action="store_true", help="Render stereo/overlay/disparity/AABB panel for one row")
    parser.add_argument("--include-pointcloud", action="store_true", help="Overlay saved point clouds when available")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="DPI for static PNGs and rendered GIF frames")
    parser.add_argument("--highlight-last", action="store_true", help="Highlight the newest/final object in the static top-down figure")
    parser.add_argument("--no-labels", action="store_true", help="Hide per-object labels")
    parser.add_argument("--minimal", action="store_true", help="Hide titles, annotations, and most axis text")
    parser.add_argument("--annotated", action="store_true", help="Keep labels and titles in static exports")
    parser.add_argument("--gif-duration-ms", type=int, default=750, help="Frame duration for sequence/descent GIFs")
    parser.add_argument("--gif-loop", type=int, default=0, help="GIF loop count; 0 means loop forever")
    parser.add_argument("--save-frames", action="store_true", help="Keep PNG frame directories for GIFs")
    parser.add_argument("--descent-row", type=int, default=None, help="Placement order for descent GIF; falls back to manifest row if needed")
    parser.add_argument("--perception-row", type=int, default=None, help="Placement order for perception panel; falls back to manifest row if needed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dpi = max(50, int(args.dpi))
    args.gif_duration_ms = max(20, int(args.gif_duration_ms))
    if not args.annotated:
        args.minimal = True
        args.no_labels = True

    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve_default_run_dir()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _resolve_default_out_dir(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    info, rows = load_manifest(run_dir)
    if not rows:
        raise RuntimeError("No placed rows with complete AABB fields were found")
    bag = load_bag_volume(rows)
    normalized_notes = normalize_raw_boxes_to_bag_pose(rows, bag)
    theme = THEMES[args.theme]

    if args.include_pointcloud:
        attach_pointclouds(rows, run_dir)

    if not any((args.make_topdown, args.make_3d, args.make_sequence_gif, args.make_descent_gif, args.make_perception_panel)):
        print("[ARGS] no figure flags specified; defaulting to --make-topdown")
        args.make_topdown = True

    outputs: list[Path] = []
    if args.make_topdown:
        outputs.extend(save_topdown(rows, bag, info, theme, out_dir, args))
    if args.make_3d:
        outputs.extend(save_3d(rows, bag, info, theme, out_dir, args))
    if args.make_sequence_gif:
        outputs.extend(save_sequence_gif(rows, bag, info, theme, out_dir, args))
    if args.make_descent_gif:
        outputs.extend(save_descent_gif(rows, bag, info, theme, out_dir, args))
    if args.make_perception_panel:
        outputs.extend(save_perception_panel(rows, bag, info, theme, run_dir, out_dir, args))

    outputs.extend(write_summary(out_dir, info, rows, bag, normalized_notes, outputs))

    print("[OUTPUTS]")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
