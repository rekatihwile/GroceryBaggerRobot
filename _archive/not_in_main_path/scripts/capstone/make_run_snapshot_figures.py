from __future__ import annotations

"""Generate presentation figures from an offline run snapshot manifest.

This script never opens cameras, serial ports, Teensy connections, YOLO, or RAFT.
It only reads saved manifest/NPZ files and writes new figures to --out-dir.

Example:
    python scripts/capstone/make_run_snapshot_figures.py ^
        --run-dir data/run_snapshots/run_20260531_184606 ^
        --out-dir paper_figure_sources/runs/run_20260531_184606/packing ^
        --view both --make-gif --dpi 300
"""

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
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


BAG_SCENE_NAME = "New Bag Test"
SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
DEFAULT_BAG_HEIGHT_MM = 250.0
MAX_POINTS_PER_OBJECT_2D = 900
MAX_POINTS_PER_OBJECT_3D = 2000
CLEAN_EXPORTS = True

# VS Code IDE defaults. Leave as None to auto-select the newest run folder.
# Copy/paste your run folder path here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_184606"
IDE_DEFAULT_RUN_DIR_STR = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
IDE_DEFAULT_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None
IDE_DEFAULT_OUT_DIR: Path | None = None
IDE_DEFAULT_RUNS_ROOT = _REPO_ROOT / "data" / "run_snapshots"

_PALETTE = [
    "#2563eb",  # blue
    "#f97316",  # orange
    "#16a34a",  # green
    "#dc2626",  # red
    "#7c3aed",  # violet
    "#ca8a04",  # amber
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#4b5563",  # gray
    "#65a30d",  # lime
]


@dataclass
class AABB:
    center: np.ndarray
    size: np.ndarray
    min_xyz: np.ndarray
    max_xyz: np.ndarray


@dataclass
class PlacedObject:
    order: int
    manifest_row: int
    object_i: Any
    class_name: str
    raw_box: AABB
    padded_box: AABB
    place_xy_mm: np.ndarray | None
    destination_surface_z_mm: float | None
    points_cam_name: str | None
    stereo_left_name: str | None
    left_overlay_name: str | None
    overhead_name: str | None
    color: str
    points_robot_placed: np.ndarray | None = None
    points_rgb_placed: np.ndarray | None = None


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


def _finite_vec(value: Any, n: int, label: str) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < n:
        return None
    out = arr[:n].astype(np.float64)
    if not np.all(np.isfinite(out)):
        return None
    return out


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _make_aabb(center: np.ndarray, size: np.ndarray) -> AABB:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    if np.any(size <= 0.0):
        raise ValueError("AABB size must be positive")
    half = 0.5 * size
    return AABB(center=center, size=size, min_xyz=center - half, max_xyz=center + half)


def _class_colors(class_names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in class_names:
        if name not in out:
            out[name] = _PALETTE[len(out) % len(_PALETTE)]
    return out


def _short_label(name: str, max_len: int = 18) -> str:
    clean = " ".join(str(name).split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "."


def _label_xy_for_object(obj: PlacedObject) -> tuple[float, float]:
    offsets = [
        (0.00, 0.00),
        (-0.18, 0.10),
        (0.18, -0.10),
        (-0.16, -0.16),
        (0.16, 0.16),
        (0.00, 0.20),
        (0.00, -0.20),
    ]
    ox, oy = offsets[(obj.order - 1) % len(offsets)]
    raw = obj.raw_box
    return (
        float(raw.center[0] + ox * raw.size[0]),
        float(raw.center[1] + oy * raw.size[1]),
    )


def load_placed_objects(run_dir: Path) -> list[PlacedObject]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data.get("objects") or []
    if not isinstance(rows, list):
        raise ValueError("manifest.json field 'objects' is not a list")

    class_names = [str(row.get("class_name") or row.get("detection_class") or "object") for row in rows]
    color_by_class = _class_colors(class_names)

    placed: list[PlacedObject] = []
    skipped = 0
    for row_i, row in enumerate(rows, start=1):
        if str(row.get("place_result", "")).lower() != "placed":
            continue

        raw_center_manifest = _finite_vec(row.get("raw_box_center_xyz_mm"), 3, "raw_box_center_xyz_mm")
        raw_size = _finite_vec(row.get("raw_box_size_xyz_mm"), 3, "raw_box_size_xyz_mm")
        padded_center = _finite_vec(row.get("padded_box_center_xyz_mm"), 3, "padded_box_center_xyz_mm")
        padded_size = _finite_vec(row.get("padded_box_size_xyz_mm"), 3, "padded_box_size_xyz_mm")
        place_xy = _finite_vec(row.get("place_xy_mm"), 2, "place_xy_mm")

        if raw_center_manifest is None or raw_size is None or padded_center is None or padded_size is None:
            skipped += 1
            print(f"[WARN] row {row_i}: placed object skipped because one or more box fields are missing")
            continue

        # Some saved manifests keep raw_box_center_xyz_mm at the original pick
        # pose while padded_box_center_xyz_mm/place_xy_mm are the placed bag pose.
        # For a packing-state figure, draw the raw box at the placed center with
        # the raw size.
        placed_raw_center = padded_center.copy()
        if place_xy is not None:
            placed_raw_center[:2] = place_xy

        try:
            raw_box = _make_aabb(placed_raw_center, raw_size)
            padded_box = _make_aabb(padded_center, padded_size)
        except ValueError as exc:
            skipped += 1
            print(f"[WARN] row {row_i}: invalid AABB skipped: {exc}")
            continue

        class_name = str(row.get("class_name") or row.get("detection_class") or "object")
        placed.append(
            PlacedObject(
                order=len(placed) + 1,
                manifest_row=row_i,
                object_i=row.get("object_i"),
                class_name=class_name,
                raw_box=raw_box,
                padded_box=padded_box,
                place_xy_mm=None if place_xy is None else place_xy.copy(),
                destination_surface_z_mm=_finite_float(row.get("destination_surface_z_mm")),
                points_cam_name=row.get("points_cam"),
                stereo_left_name=row.get("stereo_left"),
                left_overlay_name=row.get("left_overlay"),
                overhead_name=row.get("overhead"),
                color=color_by_class[class_name],
            )
        )

    print(f"[MANIFEST] loaded {len(placed)} placed object(s) from {manifest_path}")
    if skipped:
        print(f"[MANIFEST] skipped {skipped} placed row(s) with incomplete/invalid box data")
    if not placed:
        raise RuntimeError("No placed objects with complete AABB fields were found in the manifest")
    return placed


def _bag_height_from_config() -> float:
    try:
        from config.place import DEFAULT_PLACE

        return float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", DEFAULT_BAG_HEIGHT_MM))
    except Exception:
        return DEFAULT_BAG_HEIGHT_MM


def load_bag_volume(objects: list[PlacedObject]) -> BagVolume:
    box_points = _all_box_points(objects, include_bag=None)
    box_min = box_points.min(axis=0)
    box_max = box_points.max(axis=0)
    bag_height = _bag_height_from_config()

    try:
        zones = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
        scene = zones[BAG_SCENE_NAME]
        center = _finite_vec(scene.get("center_xy_mm"), 2, "center_xy_mm")
        width = _finite_float(scene.get("width_mm"))
        depth = _finite_float(scene.get("depth_mm"))
        surface_z = _finite_float(scene.get("surface_z_mm"))
        if center is None or width is None or depth is None or width <= 0.0 or depth <= 0.0:
            raise ValueError("scene has invalid center/width/depth")
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
            "[BAG] using scene "
            f"{bag.name!r} from {bag.source}: "
            f"x=[{bag.x_min:.1f},{bag.x_max:.1f}] y=[{bag.y_min:.1f},{bag.y_max:.1f}] "
            f"z=[{bag.z_min:.1f},{bag.z_max:.1f}]"
        )
        return bag
    except Exception as exc:
        margin_xy = 35.0
        margin_z = 20.0
        bag = BagVolume(
            name="inferred_bag_bounds",
            x_min=float(box_min[0] - margin_xy),
            x_max=float(box_max[0] + margin_xy),
            y_min=float(box_min[1] - margin_xy),
            y_max=float(box_max[1] + margin_xy),
            z_min=float(box_min[2] - margin_z),
            z_max=float(box_max[2] + margin_z),
            source=f"inferred from manifest boxes ({exc})",
        )
        print(
            "[BAG WARN] could not load New Bag Test scene; "
            f"using inferred bounds from boxes. Reason: {exc}"
        )
        return bag


def _load_bundle_minimal(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        print(f"[POINTCLOUD WARN] calibration bundle not found: {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:
        print(f"[POINTCLOUD WARN] failed to load calibration bundle {path}: {exc}")
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
                uv = arr[:, :2].astype(np.float64)
                return uv if np.all(np.isfinite(uv)) else None
    return None


def _sample_point_colors_rgb(image_rgb: np.ndarray | None, uv_px: np.ndarray | None, n: int) -> np.ndarray | None:
    if image_rgb is None or uv_px is None or len(uv_px) != n:
        return None
    h, w = image_rgb.shape[:2]
    xs = np.clip(np.rint(uv_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv_px[:, 1]).astype(np.int32), 0, h - 1)
    rgb = image_rgb[ys, xs, :3]
    return np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)


def _load_points_cam(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        pts, key = _pick_point_array(data, ["points_cam", "points_camera", "cam_points", "points_xyz_cam"])
        if pts is None:
            raise ValueError(f"no Nx3 point array found in {path.name}")
        if key not in {"points_cam", "points_camera", "cam_points", "points_xyz_cam"}:
            raise ValueError(f"point array '{key}' in {path.name} does not look camera-frame")
        uv = _pick_uv_array(data)
    pts = pts.reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        raise ValueError("point cloud has no finite points")
    if uv is not None and len(uv) == len(finite):
        uv = uv[finite]
    elif uv is not None and len(uv) != len(pts):
        uv = None
    return pts, uv


def attach_optional_pointclouds(objects: list[PlacedObject], run_dir: Path) -> None:
    any_points = any(obj.points_cam_name for obj in objects)
    if not any_points:
        return

    bundle = _load_bundle_minimal(BUNDLE_PATH)
    if bundle is None:
        return
    stereo_calib = load_stereo_calibration(STEREO_CALIB_PATH)

    try:
        from vision.pointcloud import cam_points_to_robot_xyz
    except Exception as exc:
        print(f"[POINTCLOUD WARN] could not import vision.pointcloud.cam_points_to_robot_xyz: {exc}")
        return

    loaded = 0
    for obj in objects:
        if not obj.points_cam_name:
            continue
        p = run_dir / obj.points_cam_name
        if not p.exists():
            print(f"[POINTCLOUD WARN] row {obj.manifest_row}: missing {p.name}; AABB-only for this object")
            continue
        try:
            points_cam, uv_px = _load_points_cam(p)
            if uv_px is None:
                uv_px = project_bundle_camera_points_to_uv(points_cam, stereo_calib)
            points_robot = cam_points_to_robot_xyz(points_cam, bundle)
            points_robot = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
            finite = np.all(np.isfinite(points_robot), axis=1)
            points_robot = points_robot[finite]
            if len(points_robot) == 0:
                raise ValueError("robot-frame point cloud has no finite points")

            # Prefer stereo-left pixel colors and fall back to saved overlay.
            image_rgb = _load_image_rgb(run_dir / obj.stereo_left_name) if obj.stereo_left_name else None
            if image_rgb is None and obj.left_overlay_name:
                image_rgb = _load_image_rgb(run_dir / obj.left_overlay_name)
            point_colors = _sample_point_colors_rgb(image_rgb, uv_px, len(points_cam))
            if point_colors is None:
                print(f"[POINTCLOUD WARN] row {obj.manifest_row}: original-photo RGB unavailable; skipping cloud")
                continue
            if point_colors is not None and len(point_colors) == len(finite):
                point_colors = point_colors[finite]
            else:
                point_colors = None

            source_centroid_xy = np.mean(points_robot[:, :2], axis=0)
            source_min_z = float(np.min(points_robot[:, 2]))
            shift = np.array(
                [
                    obj.raw_box.center[0] - source_centroid_xy[0],
                    obj.raw_box.center[1] - source_centroid_xy[1],
                    obj.raw_box.min_xyz[2] - source_min_z,
                ],
                dtype=np.float64,
            )
            shifted = points_robot + shift.reshape(1, 3)

            tol = 2.0
            inside = np.all(
                (shifted >= obj.raw_box.min_xyz.reshape(1, 3) - tol)
                & (shifted <= obj.raw_box.max_xyz.reshape(1, 3) + tol),
                axis=1,
            )
            clipped = shifted[inside]
            if len(clipped) < 20:
                print(
                    f"[POINTCLOUD WARN] row {obj.manifest_row}: fewer than 20 shifted points "
                    "fell inside the raw AABB; skipping point cloud overlay"
                )
                continue
            obj.points_robot_placed = clipped
            if point_colors is not None and len(point_colors) == len(inside):
                obj.points_rgb_placed = point_colors[inside]
            loaded += 1
        except Exception as exc:
            print(f"[POINTCLOUD WARN] row {obj.manifest_row}: failed to overlay {p.name}: {exc}")

    if loaded:
        print(f"[POINTCLOUD] overlaid shifted point clouds for {loaded}/{len(objects)} placed object(s)")


def _sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def _box_corners(box: AABB) -> np.ndarray:
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
    box: AABB,
    *,
    color: str,
    face_alpha: float,
    edge_alpha: float,
    linewidth: float,
    linestyle: str,
) -> None:
    corners = _box_corners(box)
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
    if face_alpha > 0.0:
        faces = Poly3DCollection(_box_faces(corners), alpha=face_alpha)
        faces.set_facecolor(color)
        faces.set_edgecolor("none")
        ax.add_collection3d(faces)
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


def _bag_aabb(bag: BagVolume) -> AABB:
    center = np.array(
        [
            0.5 * (bag.x_min + bag.x_max),
            0.5 * (bag.y_min + bag.y_max),
            0.5 * (bag.z_min + bag.z_max),
        ],
        dtype=np.float64,
    )
    size = np.array([bag.width, bag.depth, bag.height], dtype=np.float64)
    return _make_aabb(center, size)


def _draw_bag_volume_3d(ax, bag: BagVolume) -> None:
    bag_box = _bag_aabb(bag)
    corners = _box_corners(bag_box)

    floor = [[corners[i] for i in [0, 1, 2, 3]]]
    floor_poly = Poly3DCollection(floor, alpha=0.07)
    floor_poly.set_facecolor("#64748b")
    floor_poly.set_edgecolor("none")
    ax.add_collection3d(floor_poly)

    _draw_box_3d(
        ax,
        bag_box,
        color="#111827",
        face_alpha=0.0,
        edge_alpha=0.75,
        linewidth=1.2,
        linestyle="-",
    )


def _all_box_points(objects: list[PlacedObject], include_bag: BagVolume | None) -> np.ndarray:
    chunks = []
    for obj in objects:
        chunks.append(_box_corners(obj.raw_box))
        chunks.append(_box_corners(obj.padded_box))
        if obj.points_robot_placed is not None and len(obj.points_robot_placed):
            chunks.append(obj.points_robot_placed)
    if include_bag is not None:
        chunks.append(_box_corners(_bag_aabb(include_bag)))
    if not chunks:
        return np.zeros((1, 3), dtype=np.float64)
    return np.vstack(chunks)


def _axis_bounds_2d(objects: list[PlacedObject], bag: BagVolume, margin: float = 25.0) -> tuple[float, float, float, float]:
    pts = _all_box_points(objects, include_bag=bag)
    x_min = min(float(pts[:, 0].min()), bag.x_min) - margin
    x_max = max(float(pts[:, 0].max()), bag.x_max) + margin
    y_min = min(float(pts[:, 1].min()), bag.y_min) - margin
    y_max = max(float(pts[:, 1].max()), bag.y_max) + margin
    return x_min, x_max, y_min, y_max


def _set_equal_3d(ax, points: np.ndarray, margin_frac: float = 0.08) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if len(points) == 0:
        return
    mn = points.min(axis=0)
    mx = points.max(axis=0)
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


def _style_topdown(ax, run_name: str, bag: BagVolume) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Robot X (mm)", fontsize=10)
    ax.set_ylabel("Robot Y (mm)", fontsize=10)
    ax.set_title(f"Bag-local AABB packing state | {run_name}", fontsize=13, fontweight="bold", pad=10)
    ax.text(
        0.01,
        0.01,
        f"Bag: {bag.name} ({bag.source})   dashed = padded AABB   filled = raw AABB",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#374151",
    )


def render_topdown(
    objects: list[PlacedObject],
    bag: BagVolume,
    run_name: str,
    *,
    highlight_order: int | None = None,
    title_suffix: str = "",
):
    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor="white")
    _style_topdown(ax, run_name + title_suffix, bag)

    bag_rect = Rectangle(
        (bag.x_min, bag.y_min),
        bag.width,
        bag.depth,
        facecolor="#f8fafc",
        edgecolor="#111827",
        linewidth=2.2,
        zorder=1,
    )
    ax.add_patch(bag_rect)

    for obj in objects:
        is_highlight = highlight_order is not None and obj.order == highlight_order
        raw = obj.raw_box
        pad = obj.padded_box
        color = obj.color

        if obj.points_robot_placed is not None and obj.points_rgb_placed is not None:
            pts = _sample_points(obj.points_robot_placed, MAX_POINTS_PER_OBJECT_2D, seed=1000 + obj.order)
            rgb = _sample_points(obj.points_rgb_placed, MAX_POINTS_PER_OBJECT_2D, seed=1000 + obj.order)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=2.0 if not is_highlight else 3.2,
                c=rgb,
                alpha=0.30,
                linewidths=0,
                zorder=2,
            )

        pad_rect = Rectangle(
            (pad.min_xyz[0], pad.min_xyz[1]),
            pad.size[0],
            pad.size[1],
            facecolor="none",
            edgecolor=color,
            linewidth=2.0 if is_highlight else 1.25,
            linestyle=(0, (4, 3)),
            zorder=4,
        )
        raw_rect = Rectangle(
            (raw.min_xyz[0], raw.min_xyz[1]),
            raw.size[0],
            raw.size[1],
            facecolor=color,
            edgecolor=color,
            linewidth=2.7 if is_highlight else 1.6,
            alpha=0.30 if not is_highlight else 0.45,
            zorder=3,
        )
        ax.add_patch(pad_rect)
        ax.add_patch(raw_rect)

        if not CLEAN_EXPORTS:
            label = f"{obj.order}\n{_short_label(obj.class_name)}"
            label_x, label_y = _label_xy_for_object(obj)
            ax.text(
                label_x,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if is_highlight else "normal",
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.78),
                zorder=5,
            )

    x_min, x_max, y_min, y_max = _axis_bounds_2d(objects, bag)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if CLEAN_EXPORTS:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for text in ax.texts:
            text.set_visible(False)
    fig.tight_layout()
    return fig


def render_3d(objects: list[PlacedObject], bag: BagVolume, run_name: str):
    fig = plt.figure(figsize=(8.6, 7.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    if not CLEAN_EXPORTS:
        fig.suptitle(f"3D AABB packing state | {run_name}", fontsize=13, fontweight="bold")

    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor("#e5e7eb")
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.set_xlabel("Robot X (mm)", labelpad=8)
    ax.set_ylabel("Robot Y (mm)", labelpad=8)
    ax.set_zlabel("Robot Z (mm)", labelpad=8)

    _draw_bag_volume_3d(ax, bag)

    for obj in objects:
        color = obj.color
        if obj.points_robot_placed is not None and obj.points_rgb_placed is not None:
            pts = _sample_points(obj.points_robot_placed, MAX_POINTS_PER_OBJECT_3D, seed=2000 + obj.order)
            rgb = _sample_points(obj.points_rgb_placed, MAX_POINTS_PER_OBJECT_3D, seed=2000 + obj.order)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.4, c=rgb, alpha=0.22, depthshade=False)

        _draw_box_3d(
            ax,
            obj.raw_box,
            color=color,
            face_alpha=0.22,
            edge_alpha=0.95,
            linewidth=1.2,
            linestyle="-",
        )
        _draw_box_3d(
            ax,
            obj.padded_box,
            color=color,
            face_alpha=0.0,
            edge_alpha=0.65,
            linewidth=1.0,
            linestyle="--",
        )
        top = obj.raw_box.max_xyz[2]
        if not CLEAN_EXPORTS:
            ax.text(
                obj.raw_box.center[0],
                obj.raw_box.center[1],
                top + 8.0,
                f"{obj.order} {_short_label(obj.class_name, 13)}",
                color="#111827",
                fontsize=7,
                ha="center",
                va="bottom",
            )

    _set_equal_3d(ax, _all_box_points(objects, include_bag=bag))
    ax.view_init(elev=24, azim=-54)
    if CLEAN_EXPORTS:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        ax.text2D(
            0.02,
            0.96,
            "solid/translucent = raw AABB    dashed = padded AABB",
            transform=ax.transAxes,
            fontsize=8,
            color="#374151",
            va="top",
        )
    fig.tight_layout()
    return fig


def save_topdown(objects: list[PlacedObject], bag: BagVolume, run_name: str, out_dir: Path, dpi: int) -> list[Path]:
    fig = render_topdown(objects, bag, run_name)
    paths = save_figure_bundle(
        fig,
        out_dir / "packing_manifest_aabb_topdown",
        dpi=dpi,
        facecolor="white",
    )
    plt.close(fig)
    return paths


def save_3d(objects: list[PlacedObject], bag: BagVolume, run_name: str, out_dir: Path, dpi: int) -> list[Path]:
    fig = render_3d(objects, bag, run_name)
    paths = save_figure_bundle(
        fig,
        out_dir / "packing_manifest_aabb_3d",
        dpi=dpi,
        facecolor="white",
    )
    plt.close(fig)
    return paths


def _compile_gif(frame_paths: list[Path], gif_path: Path) -> None:
    if not frame_paths:
        raise RuntimeError("No frames to compile")
    try:
        from PIL import Image

        frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=750,
            loop=0,
            optimize=True,
        )
        for frame in frames:
            frame.close()
        return
    except Exception as pil_exc:
        try:
            import imageio.v2 as imageio

            images = [imageio.imread(path) for path in frame_paths]
            imageio.mimsave(gif_path, images, duration=0.75)
            return
        except Exception as imageio_exc:
            raise RuntimeError(f"GIF compilation failed via Pillow ({pil_exc}) and imageio ({imageio_exc})")


def save_sequence_gif(objects: list[PlacedObject], bag: BagVolume, run_name: str, out_dir: Path, dpi: int) -> list[Path]:
    frame_dir = out_dir / "bag_aabb_sequence_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for k in range(1, len(objects) + 1):
        fig = render_topdown(
            objects[:k],
            bag,
            run_name,
            highlight_order=k,
            title_suffix=f" | step {k:02d}/{len(objects):02d}",
        )
        frame_path = frame_dir / f"bag_aabb_sequence_{k:02d}.png"
        fig.savefig(frame_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = out_dir / "bag_aabb_sequence.gif"
    _compile_gif(frame_paths, gif_path)
    return frame_paths + [gif_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline presentation figures from a run snapshot manifest.",
    )
    parser.add_argument("--run-dir", default=None, type=Path, help="Path to data/run_snapshots/run_YYYYMMDD_HHMMSS")
    parser.add_argument("--out-dir", default=None, type=Path, help="Directory where figures will be written")
    parser.add_argument("--make-gif", action="store_true", help="Also render top-down sequence frames and GIF")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="PNG/frame output DPI")
    parser.add_argument("--view", choices=("topdown", "3d", "both"), default="both", help="Which static views to render")
    parser.add_argument("--annotated", action="store_true", help="Keep titles, labels, and axis text")
    return parser.parse_args()


def main() -> int:
    global CLEAN_EXPORTS

    args = parse_args()
    CLEAN_EXPORTS = not bool(args.annotated)
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve_default_run_dir()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _resolve_default_out_dir(run_dir)
    dpi = max(50, int(args.dpi))

    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_placed_objects(run_dir)
    bag = load_bag_volume(objects)
    attach_optional_pointclouds(objects, run_dir)

    run_name = run_dir.name
    outputs: list[Path] = []
    if args.view in ("topdown", "both"):
        outputs.extend(save_topdown(objects, bag, run_name, out_dir, dpi))
    if args.view in ("3d", "both"):
        outputs.extend(save_3d(objects, bag, run_name, out_dir, dpi))
    if args.make_gif:
        outputs.extend(save_sequence_gif(objects, bag, run_name, out_dir, dpi))

    print("[OUTPUTS]")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
