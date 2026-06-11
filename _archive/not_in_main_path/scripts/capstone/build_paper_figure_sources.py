from __future__ import annotations

"""Build one publication-oriented source tree from saved run snapshots.

This exporter intentionally treats manifest placements as execution ground truth.
Planner replays can be regenerated separately, but they must not erase a placement
that the robot actually executed.
"""

from dataclasses import dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    PUBLICATION_FIGURES_ROOT,
    PUBLICATION_GLOBAL_ROOT,
    PUBLICATION_OUTPUT_ROOT,
    PUBLICATION_RUNS_ROOT,
    RUN_SNAPSHOTS_ROOT,
    safe_name,
    save_figure_bundle,
)


SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"
BAG_SCENE_NAME = "New Bag Test"
PAIR_RE = re.compile(r"(\d+)")
IMAGE_FIELDS = (
    ("stereo_left", "01_stereo_left_rectified"),
    ("stereo_right", "02_stereo_right_rectified"),
    ("overhead", "03_overhead_raw"),
    ("left_overlay", "04_yolo_overlay_saved"),
)
CLASS_COLORS = {
    "Bandaid": "#3b82f6",
    "Chex Mix": "#f59e0b",
    "Pringles": "#fb923c",
    "Carmex Lip Balm": "#a78bfa",
    "Burts Bees": "#eab308",
    "Dove Deodorant": "#ef4444",
    "Lays Chips": "#0ea5e9",
    "Paper Towels": "#94a3b8",
}
PALETTE = (
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#dc2626",
    "#7c3aed",
    "#ca8a04",
    "#0891b2",
    "#be185d",
)


@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


@dataclass
class PlacementAudit:
    manifest_row: int
    object_i: int
    class_name: str
    result: str
    box: Box | None
    target_xy: np.ndarray | None
    destination_z: float | None
    support_row: int | None
    overlap_rows: list[int]
    inside_bag: bool | None
    execution_classification: str


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except Exception:
        return None
    return output if math.isfinite(output) else None


def _finite_vec(value: Any, count: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)[:count]
    except Exception:
        return None
    if len(array) != count or not np.all(np.isfinite(array)):
        return None
    return array


def _pair_index(value: str | None, fallback: int) -> int:
    if value:
        match = PAIR_RE.search(Path(value).name)
        if match:
            return int(match.group(1))
    return int(fallback)


def _load_bag() -> dict[str, float]:
    fallback = {
        "x_min": 80.0,
        "x_max": 340.0,
        "y_min": 690.0,
        "y_max": 820.0,
        "z_min": -155.0,
        "z_max": 95.0,
    }
    try:
        data = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
        scene = data[BAG_SCENE_NAME]
        center = _finite_vec(scene.get("center_xy_mm"), 2)
        width = _finite_float(scene.get("width_mm"))
        depth = _finite_float(scene.get("depth_mm"))
        surface = _finite_float(scene.get("surface_z_mm"))
        if center is None or width is None or depth is None or surface is None:
            return fallback
        return {
            "x_min": float(center[0] - width / 2.0),
            "x_max": float(center[0] + width / 2.0),
            "y_min": float(center[1] - depth / 2.0),
            "y_max": float(center[1] + depth / 2.0),
            "z_min": float(surface),
            "z_max": float(surface + 250.0),
        }
    except Exception:
        return fallback


def _box(center: np.ndarray, size: np.ndarray) -> Box:
    center = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size, dtype=np.float64).reshape(3)
    half = 0.5 * size
    return Box(center=center, size=size, minimum=center - half, maximum=center + half)


def _executed_box(row: dict[str, Any]) -> Box | None:
    size = _finite_vec(row.get("raw_box_size_xyz_mm"), 3)
    xy = _finite_vec(row.get("place_xy_mm"), 2)
    destination_z = _finite_float(row.get("destination_surface_z_mm"))
    if size is None or xy is None or destination_z is None:
        return None
    center = np.array(
        [xy[0], xy[1], destination_z + 0.5 * size[2]],
        dtype=np.float64,
    )
    return _box(center, size)


def _overlap_volume(a: Box, b: Box, tolerance_mm: float = 1.0) -> float:
    overlap = np.minimum(a.maximum, b.maximum) - np.maximum(a.minimum, b.minimum)
    if np.any(overlap <= tolerance_mm):
        return 0.0
    return float(np.prod(overlap))


def _xy_overlap_area(a: Box, b: Box) -> float:
    overlap = np.minimum(a.maximum[:2], b.maximum[:2]) - np.maximum(a.minimum[:2], b.minimum[:2])
    if np.any(overlap <= 0.0):
        return 0.0
    return float(np.prod(overlap))


def _audit_manifest(rows: list[dict[str, Any]], bag: dict[str, float]) -> list[PlacementAudit]:
    audits: list[PlacementAudit] = []
    placed: list[tuple[int, Box]] = []
    for manifest_row, row in enumerate(rows, start=1):
        result = str(row.get("place_result") or "")
        class_name = str(row.get("class_name") or row.get("detection_class") or "object")
        object_i = int(row.get("object_i") or manifest_row)
        box = _executed_box(row) if result.lower() in ("placed", "success", "ok") else None
        target_xy = _finite_vec(row.get("place_xy_mm"), 2)
        destination_z = _finite_float(row.get("destination_surface_z_mm"))
        overlap_rows: list[int] = []
        support_row = None
        inside_bag = None
        classification = f"not_placed:{result or 'missing'}"

        if box is not None:
            inside_bag = bool(
                box.minimum[0] >= bag["x_min"] - 1.0
                and box.maximum[0] <= bag["x_max"] + 1.0
                and box.minimum[1] >= bag["y_min"] - 1.0
                and box.maximum[1] <= bag["y_max"] + 1.0
                and box.maximum[2] <= bag["z_max"] + 1.0
            )
            for earlier_row, earlier_box in placed:
                if _overlap_volume(box, earlier_box) > 0.0:
                    overlap_rows.append(earlier_row)

            if destination_z is not None and abs(destination_z - bag["z_min"]) <= 3.0:
                classification = "manifest_executed_floor"
            else:
                best_area = 0.0
                for earlier_row, earlier_box in placed:
                    if destination_z is None or abs(float(earlier_box.maximum[2]) - destination_z) > 4.0:
                        continue
                    area = _xy_overlap_area(box, earlier_box)
                    if area > best_area:
                        best_area = area
                        support_row = earlier_row
                if support_row is not None:
                    classification = "manifest_executed_supported_stack"
                else:
                    classification = "manifest_executed_unmodeled_fallback"

            if overlap_rows:
                classification += "_with_aabb_overlap"
            if not inside_bag:
                classification += "_with_boundary_overflow"
            placed.append((manifest_row, box))

        audits.append(
            PlacementAudit(
                manifest_row=manifest_row,
                object_i=object_i,
                class_name=class_name,
                result=result,
                box=box,
                target_xy=target_xy,
                destination_z=destination_z,
                support_row=support_row,
                overlap_rows=overlap_rows,
                inside_bag=inside_bag,
                execution_classification=classification,
            )
        )
    return audits


def _write_audit_files(
    run_out: Path,
    manifest: dict[str, Any] | None,
    audits: list[PlacementAudit],
) -> None:
    rows = manifest.get("objects", []) if manifest else []
    with (run_out / "placement_reconciliation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifest_row",
                "object_i",
                "class_name",
                "place_result",
                "execution_classification",
                "support_manifest_row",
                "overlap_manifest_rows",
                "inside_bag",
                "target_x_mm",
                "target_y_mm",
                "destination_surface_z_mm",
                "box_center_x_mm",
                "box_center_y_mm",
                "box_center_z_mm",
                "box_size_x_mm",
                "box_size_y_mm",
                "box_size_z_mm",
            ]
        )
        for audit in audits:
            box = audit.box
            writer.writerow(
                [
                    audit.manifest_row,
                    audit.object_i,
                    audit.class_name,
                    audit.result,
                    audit.execution_classification,
                    audit.support_row or "",
                    "|".join(map(str, audit.overlap_rows)),
                    "" if audit.inside_bag is None else audit.inside_bag,
                    "" if audit.target_xy is None else float(audit.target_xy[0]),
                    "" if audit.target_xy is None else float(audit.target_xy[1]),
                    "" if audit.destination_z is None else audit.destination_z,
                    "" if box is None else float(box.center[0]),
                    "" if box is None else float(box.center[1]),
                    "" if box is None else float(box.center[2]),
                    "" if box is None else float(box.size[0]),
                    "" if box is None else float(box.size[1]),
                    "" if box is None else float(box.size[2]),
                ]
            )

    summary = {
        "manifest_present": manifest is not None,
        "total_manifest_rows": len(rows),
        "placed_rows": sum(audit.box is not None for audit in audits),
        "rows_with_aabb_overlap": sum(bool(audit.overlap_rows) for audit in audits),
        "rows_with_boundary_overflow": sum(audit.inside_bag is False for audit in audits),
        "rows_with_unmodeled_fallback": sum(
            "unmodeled_fallback" in audit.execution_classification for audit in audits
        ),
        "planner_sequence_recorded": (manifest or {}).get("place_planning_sequence"),
        "authority": "manifest execution is ground truth; offline planner replay is explanatory only",
    }
    (run_out / "run_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _copy_image_bundle(source: Path, stem: Path, make_pdf: bool) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    native = stem.with_suffix(source.suffix.lower())
    shutil.copy2(source, native)
    written.append(native)
    try:
        image = Image.open(source).convert("RGB")
        jpg = stem.with_suffix(".jpg")
        image.save(jpg, quality=96, subsampling=0)
        written.append(jpg)
        if make_pdf:
            pdf = stem.with_suffix(".pdf")
            image.save(pdf, "PDF", resolution=300.0)
            written.append(pdf)
        image.close()
    except Exception:
        pass
    return written


def _load_disparity(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "disparity" in data.files:
                return np.asarray(data["disparity"], dtype=np.float64)
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim == 2:
                    return value.astype(np.float64)
    except Exception:
        return None
    return None


def _disparity_heatmap(disparity: np.ndarray) -> np.ndarray:
    value = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(value) & (value > 1.0)
    if not np.any(valid):
        return np.zeros((*value.shape, 3), dtype=np.uint8)
    low, high = np.percentile(value[valid], [2.0, 98.0])
    normalized = np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    heatmap = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    heatmap[~valid] = 0
    return heatmap


def _save_raster_plot(image_bgr: np.ndarray, stem: Path, dpi: int) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 6.0), facecolor="white")
    ax.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outputs = save_figure_bundle(fig, stem, dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    cv2.imwrite(str(stem.with_suffix(".jpg")), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return outputs + [stem.with_suffix(".jpg")]


def _load_point_data(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        return None, None
    try:
        with np.load(path, allow_pickle=False) as data:
            points = None
            for key in ("points_cam", "points", "xyz", "points_xyz"):
                if key in data.files:
                    points = np.asarray(data[key], dtype=np.float64).reshape(-1, 3)
                    break
            if points is None:
                return None, None
            uv_px = None
            for key in ("uv", "uv_px", "point_uv_px", "points_uv", "point_uv"):
                if key in data.files:
                    candidate = np.asarray(data[key], dtype=np.float64)
                    if candidate.ndim == 2 and candidate.shape[1] >= 2 and len(candidate) == len(points):
                        uv_px = candidate[:, :2]
                        break
            finite = np.all(np.isfinite(points), axis=1)
            points = points[finite]
            if uv_px is not None:
                uv_px = uv_px[finite]
            return (points if len(points) else None), uv_px
    except Exception:
        return None, None


def _load_stereo_calib() -> dict[str, np.ndarray] | None:
    if not STEREO_CALIB_PATH.exists():
        return None
    try:
        with np.load(STEREO_CALIB_PATH, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception:
        return None


def _load_image_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _project_cam_bundle_to_uv(
    points_cam: np.ndarray,
    stereo_calib: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    if stereo_calib is None or "projection_left_rectified" not in stereo_calib:
        return None
    points = np.asarray(points_cam, dtype=np.float64).copy().reshape(-1, 3)
    points[:, 2] *= -1.0
    if "rectification_left" in stereo_calib:
        points = points @ np.asarray(stereo_calib["rectification_left"], dtype=np.float64).T
    projection = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 1.0)
    safe_z = np.where(valid, z, 1.0)
    u = np.where(valid, projection[0, 0] * points[:, 0] / safe_z + projection[0, 2], np.nan)
    v = np.where(valid, projection[1, 1] * points[:, 1] / safe_z + projection[1, 2], np.nan)
    return np.column_stack([u, v])


def _sample_point_colors(
    image_rgb: np.ndarray | None,
    uv_px: np.ndarray | None,
    point_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if image_rgb is None or uv_px is None or len(uv_px) != point_count:
        return None, None
    height, width = image_rgb.shape[:2]
    valid = (
        np.all(np.isfinite(uv_px[:, :2]), axis=1)
        & (uv_px[:, 0] >= 0.0)
        & (uv_px[:, 0] <= width - 1)
        & (uv_px[:, 1] >= 0.0)
        & (uv_px[:, 1] <= height - 1)
    )
    if not np.any(valid):
        return None, valid
    x = np.rint(uv_px[valid, 0]).astype(np.int32)
    y = np.rint(uv_px[valid, 1]).astype(np.int32)
    colors = np.clip(image_rgb[y, x, :3], 0.0, 1.0).astype(np.float32)
    return colors, valid


def _load_bundle() -> dict[str, np.ndarray] | None:
    if not BUNDLE_PATH.exists():
        return None
    try:
        from test_calibration_bundle_live_stereo_z_pickplace import load_bundle

        return load_bundle(BUNDLE_PATH)
    except Exception:
        return None


def _to_robot(points_cam: np.ndarray, bundle: dict[str, np.ndarray] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if bundle is None:
        return None, None
    try:
        from vision.pointcloud import cam_points_to_robot_xyz

        points = np.asarray(cam_points_to_robot_xyz(points_cam, bundle), dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        return points[finite], finite
    except Exception:
        return None, None


def _sample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), size=limit, replace=False))]


def _save_pointcloud_sources(
    points_cam: np.ndarray,
    points_robot: np.ndarray | None,
    colors_rgb: np.ndarray,
    uv_px: np.ndarray,
    step_dir: Path,
    dpi: int,
    point_limit: int,
    seed: int,
    color_source: str,
) -> list[Path]:
    sample_count = min(len(points_cam), point_limit)
    if len(points_cam) <= sample_count:
        indices = np.arange(len(points_cam))
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(points_cam), size=sample_count, replace=False))
    sampled_cam = points_cam[indices]
    sampled_robot = points_robot[indices] if points_robot is not None else None
    sampled_colors = colors_rgb[indices]
    sampled_uv = uv_px[indices]

    colorized_npz = step_dir / "08_point_cloud_colorized.npz"
    np.savez_compressed(
        colorized_npz,
        points_cam=points_cam.astype(np.float32),
        points_robot=(points_robot.astype(np.float32) if points_robot is not None else np.empty((0, 3), dtype=np.float32)),
        colors_rgb=colors_rgb.astype(np.float32),
        uv_px=uv_px.astype(np.float32),
        color_source=np.asarray(color_source),
    )
    csv_path = step_dir / "07_point_cloud_coordinates_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "camera_x_mm",
                "camera_y_mm",
                "camera_z_mm",
                "robot_x_mm",
                "robot_y_mm",
                "robot_z_mm",
                "u_px",
                "v_px",
                "red",
                "green",
                "blue",
                "color_source",
            ]
        )
        for index, camera_point in enumerate(sampled_cam):
            robot_point = sampled_robot[index] if sampled_robot is not None else (math.nan, math.nan, math.nan)
            writer.writerow(
                [
                    *map(float, camera_point),
                    *map(float, robot_point),
                    *map(float, sampled_uv[index]),
                    *map(float, sampled_colors[index]),
                    color_source,
                ]
            )

    fig = plt.figure(figsize=(12.0, 5.8), facecolor="white")
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
    for ax, points in zip(axes, (sampled_cam, sampled_robot)):
        ax.set_axis_off()
        if points is None or len(points) == 0:
            continue
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.7,
            alpha=0.75,
            c=sampled_colors,
            depthshade=False,
        )
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = 0.5 * (minimum + maximum)
        half = 0.55 * max(float(np.max(maximum - minimum)), 1.0)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=22, azim=-56)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01)
    outputs = save_figure_bundle(
        fig,
        step_dir / "08_point_cloud_camera_and_robot_frames",
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig)
    audit_path = step_dir / "point_cloud_color_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "point_count": len(points_cam),
                "sample_count": len(sampled_cam),
                "color_source": color_source,
                "rgb_embedded_in_npz": True,
                "rgb_columns_in_csv": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [csv_path, colorized_npz, audit_path, *outputs]


def _color_for(class_name: str, index: int) -> str:
    return CLASS_COLORS.get(class_name, PALETTE[index % len(PALETTE)])


def _box_corners(box: Box) -> np.ndarray:
    mn = box.minimum
    mx = box.maximum
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


def _draw_box_3d(ax, box: Box, color: str) -> None:
    corners = _box_corners(box)
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    for start, end in edges:
        ax.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color=color,
            linewidth=1.2,
            alpha=0.9,
        )
    faces = Poly3DCollection(
        [[corners[index] for index in face] for face in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5))],
        alpha=0.16,
    )
    faces.set_facecolor(color)
    faces.set_edgecolor("none")
    ax.add_collection3d(faces)


def _save_packing_figures(
    audits: list[PlacementAudit],
    bag: dict[str, float],
    packing_dir: Path,
    dpi: int,
) -> list[Path]:
    placed = [audit for audit in audits if audit.box is not None]
    if not placed:
        return []
    packing_dir.mkdir(parents=True, exist_ok=True)

    fig2d, ax2d = plt.subplots(figsize=(9.0, 5.8), facecolor="white")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.add_patch(
        Rectangle(
            (bag["x_min"], bag["y_min"]),
            bag["x_max"] - bag["x_min"],
            bag["y_max"] - bag["y_min"],
            facecolor="#f8fafc",
            edgecolor="#111827",
            linewidth=2.0,
        )
    )
    for index, audit in enumerate(placed):
        box = audit.box
        color = _color_for(audit.class_name, index)
        ax2d.add_patch(
            Rectangle(
                (box.minimum[0], box.minimum[1]),
                box.size[0],
                box.size[1],
                facecolor=color,
                edgecolor=color,
                linewidth=1.5,
                alpha=0.32,
            )
        )
    ax2d.set_xlim(bag["x_min"] - 10.0, bag["x_max"] + 10.0)
    ax2d.set_ylim(bag["y_min"] - 10.0, bag["y_max"] + 10.0)
    ax2d.axis("off")
    fig2d.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outputs = save_figure_bundle(
        fig2d,
        packing_dir / "packing_executed_manifest_topdown",
        dpi=dpi,
        facecolor="white",
        bbox_inches=None,
    )
    plt.close(fig2d)

    fig3d = plt.figure(figsize=(8.0, 7.0), facecolor="white")
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.set_axis_off()
    bag_box = _box(
        np.array(
            [
                0.5 * (bag["x_min"] + bag["x_max"]),
                0.5 * (bag["y_min"] + bag["y_max"]),
                0.5 * (bag["z_min"] + bag["z_max"]),
            ]
        ),
        np.array(
            [
                bag["x_max"] - bag["x_min"],
                bag["y_max"] - bag["y_min"],
                bag["z_max"] - bag["z_min"],
            ]
        ),
    )
    corners = _box_corners(bag_box)
    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        ax3d.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color="#475569",
            linewidth=0.9,
            alpha=0.65,
        )
    for index, audit in enumerate(placed):
        _draw_box_3d(ax3d, audit.box, _color_for(audit.class_name, index))
    all_points = np.vstack([corners, *[_box_corners(audit.box) for audit in placed]])
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half = 0.55 * max(float(np.max(maximum - minimum)), 1.0)
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.view_init(elev=26, azim=-58)
    fig3d.subplots_adjust(left=0, right=1, bottom=0, top=1)
    outputs.extend(
        save_figure_bundle(
            fig3d,
            packing_dir / "packing_executed_manifest_3d",
            dpi=dpi,
            facecolor="white",
            bbox_inches=None,
        )
    )
    plt.close(fig3d)
    return outputs


def _rows_from_files(run_dir: Path) -> list[dict[str, Any]]:
    indices: set[int] = set()
    for path in run_dir.iterdir():
        match = PAIR_RE.search(path.name)
        if match and path.name.lower().endswith((".png", ".npz")):
            indices.add(int(match.group(1)))
    rows = []
    for index in sorted(indices):
        rows.append(
            {
                "object_i": index,
                "class_name": "unknown",
                "stereo_left": f"Stereo_Left_{index:04d}.png",
                "stereo_right": f"Stereo_Right_{index:04d}.png",
                "overhead": f"Overhead_{index:04d}.png",
                "left_overlay": f"left_overlay_{index:04d}.png",
                "disparity": f"disparity_{index:04d}.npz",
                "points_cam": f"points_cam_{index:04d}.npz",
            }
        )
    return rows


def _export_run(
    run_dir: Path,
    *,
    dpi: int,
    point_limit: int,
    make_image_pdfs: bool,
    bundle: dict[str, np.ndarray] | None,
    stereo_calib: dict[str, np.ndarray] | None,
    bag: dict[str, float],
) -> dict[str, Any]:
    run_out = PUBLICATION_RUNS_ROOT / run_dir.name
    run_out.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest_path.exists():
        shutil.copy2(manifest_path, run_out / "manifest.json")
    rows = list((manifest or {}).get("objects") or _rows_from_files(run_dir))
    audits = _audit_manifest(rows, bag)
    _write_audit_files(run_out, manifest, audits)

    index_rows: list[list[Any]] = []
    perception_root = run_out / "perception_steps"
    for manifest_row, row in enumerate(rows, start=1):
        pair_index = _pair_index(row.get("stereo_left"), manifest_row)
        class_name = str(row.get("class_name") or row.get("detection_class") or "unknown")
        step_dir = perception_root / f"step_{pair_index:04d}_row_{manifest_row:02d}_{safe_name(class_name)}"
        source_dir = step_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        for field, output_name in IMAGE_FIELDS:
            filename = row.get(field)
            if not filename:
                continue
            source = run_dir / str(filename)
            if source.exists():
                for written in _copy_image_bundle(source, source_dir / output_name, make_image_pdfs):
                    index_rows.append([manifest_row, pair_index, class_name, field, source.name, written.relative_to(run_out)])

        disparity_name = row.get("disparity")
        if disparity_name:
            source = run_dir / str(disparity_name)
            if source.exists():
                copied = source_dir / "05_disparity_raw.npz"
                shutil.copy2(source, copied)
                index_rows.append([manifest_row, pair_index, class_name, "disparity", source.name, copied.relative_to(run_out)])
                disparity = _load_disparity(source)
                if disparity is not None:
                    heatmap = _disparity_heatmap(disparity)
                    for written in _save_raster_plot(heatmap, source_dir / "06_disparity_heatmap", dpi):
                        index_rows.append([manifest_row, pair_index, class_name, "disparity_heatmap", source.name, written.relative_to(run_out)])

        points_name = row.get("points_cam")
        if points_name:
            source = run_dir / str(points_name)
            if source.exists():
                copied = source_dir / "07_point_cloud_camera_raw.npz"
                shutil.copy2(source, copied)
                index_rows.append([manifest_row, pair_index, class_name, "points_cam", source.name, copied.relative_to(run_out)])
                points_cam, stored_uv = _load_point_data(source)
                if points_cam is not None:
                    stereo_left_name = row.get("stereo_left")
                    stereo_left_path = run_dir / str(stereo_left_name) if stereo_left_name else None
                    image_rgb = _load_image_rgb(stereo_left_path)
                    uv_px = stored_uv if stored_uv is not None else _project_cam_bundle_to_uv(points_cam, stereo_calib)
                    color_source = "npz_uv_plus_stereo_left" if stored_uv is not None else "calibrated_uv_reprojection_plus_stereo_left"
                    colors_rgb, color_valid = _sample_point_colors(image_rgb, uv_px, len(points_cam))
                    if colors_rgb is None or color_valid is None or uv_px is None:
                        audit_path = source_dir / "point_cloud_color_audit.json"
                        audit_path.write_text(
                            json.dumps(
                                {
                                    "point_count": len(points_cam),
                                    "color_source": color_source,
                                    "rgb_embedded_in_npz": False,
                                    "status": "not_exported_without_original_photo_color",
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        index_rows.append([manifest_row, pair_index, class_name, "pointcloud_color_audit", source.name, audit_path.relative_to(run_out)])
                        print(f"[COLOR WARN] {run_dir.name} row {manifest_row}: unable to reconstruct photo RGB; skipped derived point-cloud plot")
                    else:
                        points_cam = points_cam[color_valid]
                        uv_px = uv_px[color_valid]
                        points_robot, robot_finite = _to_robot(points_cam, bundle)
                        if points_robot is not None and robot_finite is not None:
                            points_cam = points_cam[robot_finite]
                            uv_px = uv_px[robot_finite]
                            colors_rgb = colors_rgb[robot_finite]
                        for written in _save_pointcloud_sources(
                            points_cam,
                            points_robot,
                            colors_rgb,
                            uv_px,
                            source_dir,
                            dpi,
                            point_limit,
                            seed=pair_index,
                            color_source=color_source,
                        ):
                            index_rows.append([manifest_row, pair_index, class_name, "pointcloud_derived_colorized", source.name, written.relative_to(run_out)])

        (source_dir / "manifest_row.json").write_text(
            json.dumps(row, indent=2),
            encoding="utf-8",
        )

    with (run_out / "run_source_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["manifest_row", "pair_index", "class_name", "source_type", "original_filename", "publication_path"])
        writer.writerows(index_rows)

    packing_outputs = _save_packing_figures(audits, bag, run_out / "packing", dpi)
    return {
        "run": run_dir.name,
        "manifest": manifest is not None,
        "rows": len(rows),
        "placed": sum(audit.box is not None for audit in audits),
        "packing_outputs": [str(path) for path in packing_outputs],
    }


def _extract_pdf_image(pdf_path: Path, page_number: int, image_index: int, destination: Path) -> Path | None:
    try:
        from pypdf import PdfReader

        image = list(PdfReader(str(pdf_path)).pages[page_number - 1].images)[image_index - 1]
        suffix = Path(image.name).suffix or ".bin"
        output = destination.with_suffix(suffix)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(image.data)
        return output
    except Exception as exc:
        print(f"[PDF WARN] failed to extract {pdf_path.name} page {page_number} image {image_index}: {exc}")
        return None


def _make_recovery_diagram(figure_dir: Path, dpi: int) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    nodes = [
        ("survey", 0.05, 0.68, 0.13, 0.16, "Survey"),
        ("select", 0.24, 0.68, 0.16, 0.16, "Select + plan"),
        ("pick", 0.46, 0.68, 0.11, 0.16, "Pick"),
        ("watchdog", 0.63, 0.68, 0.22, 0.16, "Continuity check"),
        ("place", 0.65, 0.25, 0.20, 0.16, "Release + update"),
        ("recover", 0.27, 0.25, 0.25, 0.16, "Cancel release\nRehome Z + retry"),
    ]
    edges = [
        ("survey", "select", "scene"),
        ("select", "pick", "target"),
        ("pick", "watchdog", "pre-release"),
        ("watchdog", "place", "item absent"),
        ("watchdog", "recover", "item still present"),
        ("recover", "survey", "retry"),
        ("place", "survey", "next item"),
    ]
    positions = {name: (x, y, width, height) for name, x, y, width, height, _ in nodes}

    edge_routes = {
        ("survey", "select"): [(0.18, 0.76), (0.24, 0.76)],
        ("select", "pick"): [(0.40, 0.76), (0.46, 0.76)],
        ("pick", "watchdog"): [(0.57, 0.76), (0.63, 0.76)],
        ("watchdog", "place"): [(0.74, 0.68), (0.74, 0.41)],
        ("watchdog", "recover"): [(0.63, 0.72), (0.58, 0.58), (0.50, 0.41)],
        ("recover", "survey"): [(0.27, 0.33), (0.115, 0.33), (0.115, 0.68)],
        ("place", "survey"): [(0.75, 0.25), (0.75, 0.08), (0.02, 0.08), (0.02, 0.76), (0.05, 0.76)],
    }
    label_positions = {
        ("survey", "select"): (0.21, 0.79),
        ("select", "pick"): (0.43, 0.79),
        ("pick", "watchdog"): (0.60, 0.79),
        ("watchdog", "place"): (0.79, 0.54),
        ("watchdog", "recover"): (0.53, 0.57),
        ("recover", "survey"): (0.18, 0.30),
        ("place", "survey"): (0.48, 0.055),
    }

    fig, ax = plt.subplots(figsize=(12.0, 6.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for name, x, y, width, height, label in nodes:
        color = "#dbeafe" if name not in ("recover", "place") else ("#fee2e2" if name == "recover" else "#dcfce7")
        ax.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="#1f2937", linewidth=1.6))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)

    for source, target, label in edges:
        route = edge_routes[(source, target)]
        if len(route) > 2:
            xs, ys = zip(*route[:-1])
            ax.plot(xs, ys, color="#334155", linewidth=1.4, solid_capstyle="round")
        arrow = FancyArrowPatch(
            route[-2],
            route[-1],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#334155",
            connectionstyle="arc3,rad=0",
        )
        ax.add_patch(arrow)
        label_x, label_y = label_positions[(source, target)]
        ax.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#475569",
            backgroundcolor="white",
        )

    outputs = save_figure_bundle(fig, figure_dir / "missed_pick_recovery_control_flow", dpi=dpi, facecolor="white")
    plt.close(fig)
    with (figure_dir / "recovery_flow_nodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x", "y", "width", "height", "label"])
        writer.writerows(nodes)
    with (figure_dir / "recovery_flow_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "condition"])
        writer.writerows(edges)
    return outputs


def _copy_tree_items(source: Path, destination: Path, patterns: tuple[str, ...]) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for pattern in patterns:
        for path in source.glob(pattern):
            if not path.is_file():
                continue
            target = destination / path.name
            shutil.copy2(path, target)
            outputs.append(target)
    return outputs


def _write_folder_manifest(destination_root: Path, rows: list[list[str]], *, summary: str) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    with (destination_root / "figure_source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "paper_role", "source_scope", "source_path", "notes"])
        writer.writerows(rows)
    (destination_root / "README.md").write_text(
        "\n".join(
            [
                "# Curated Figure Pack",
                "",
                summary,
                "",
                "See `figure_source_manifest.csv` for exact provenance.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _curate_figures(
    *,
    destination_root: Path,
    source_scope_label: str,
    curated_run: str,
    planning_run: str,
    paper_pdf: Path | None,
    presentation_pdf: Path | None,
    dpi: int,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    figure_rows: list[list[str]] = []

    figure1 = destination_root / "figure_01_robot_hero"
    hero = None
    if paper_pdf and paper_pdf.exists():
        hero = _extract_pdf_image(paper_pdf, 1, 1, figure1 / "robot_hero_extracted_from_paper")
    if hero is None and presentation_pdf and presentation_pdf.exists():
        hero = _extract_pdf_image(presentation_pdf, 11, 1, figure1 / "robot_hero_extracted_from_presentation")
    if hero is not None:
        _copy_image_bundle(hero, figure1 / "robot_hero_source", make_pdf=True)
        figure_rows.append(["1", "Robot hero", source_scope_label, str(hero), "Use the original camera file if it becomes available; this is the highest-resolution embedded paper source."])

    curated_root = PUBLICATION_RUNS_ROOT / curated_run
    perception_steps = sorted((curated_root / "perception_steps").glob("step_*")) if curated_root.exists() else []
    if perception_steps:
        preferred = perception_steps[min(1, len(perception_steps) - 1)]
        figure2 = destination_root / "figure_02_perception_pipeline"
        copied = _copy_tree_items(preferred / "source", figure2, ("*.png", "*.jpg", "*.pdf", "*.csv", "*.npz", "*.json"))
        copied.extend(_copy_tree_items(preferred, figure2, ("01_*", "02_*", "03_*", "04_*", "05_*")))
        figure_rows.append(["2", "Perception pipeline", source_scope_label, str(preferred), f"{len(copied)} source files from one manifest survey step."])

    planning_root = PUBLICATION_RUNS_ROOT / planning_run
    figure3 = destination_root / "figure_03_topdown_packing"
    copied3 = _copy_tree_items(planning_root / "packing", figure3, ("*topdown*",))
    copied3.extend(_copy_tree_items(planning_root, figure3, ("placement_reconciliation.csv", "run_audit.json")))
    if copied3:
        figure_rows.append(["3", "Top-down packing", source_scope_label, str(planning_root), "Executed manifest geometry plus reconciliation audit."])

    figure4 = destination_root / "figure_04_3d_planning_scene"
    copied4 = _copy_tree_items(curated_root / "packing", figure4, ("*3d*",))
    planner_steps = sorted((curated_root / "planner_steps").glob("step_*")) if curated_root.exists() else []
    if planner_steps:
        copied4.extend(_copy_tree_items(planner_steps[min(2, len(planner_steps) - 1)], figure4, ("01_scene*", "05_bag_state*", "planner_*")))
    if copied4:
        figure_rows.append(["4", "3D planning scene", source_scope_label, str(curated_root), "Prefer the planner-step scene when present; packing-only fallback is also included."])

    figure5 = destination_root / "figure_05_missed_pick_recovery"
    _make_recovery_diagram(figure5, dpi)
    figure_rows.append(["5", "Missed-pick recovery", source_scope_label, "generated from active control path", "Vector PDF text and CSV node/edge source included."])

    figure6 = destination_root / "figure_06_calibration"
    calibration_root = PUBLICATION_GLOBAL_ROOT / "calibration"
    copied6 = _copy_tree_items(calibration_root, figure6, ("*",)) if calibration_root.exists() else []
    if copied6:
        figure_rows.append(["6", "Calibration", source_scope_label, str(calibration_root), "Rendered PDFs plus support-point and camera-pose CSVs."])
    _write_folder_manifest(destination_root, figure_rows, summary=source_scope_label)


def _write_output_readme() -> None:
    text = """# Paper Figure Sources

This directory is generated by `scripts/capstone/build_paper_figure_sources.py`.

## Layout

- `runs/<run_name>/perception_steps/`: raw camera frames, saved overlays, disparity,
  point-cloud data, CSV samples, and generated perception panels for each survey step.
- `runs/<run_name>/packing/`: clean top-down and 3D views of executed placements.
- `runs/<run_name>/placement_reconciliation.csv`: per-object execution audit.
- `runs/<run_name>/run_source_index.csv`: map from original snapshot files to exports.
- `global/calibration/`: calibration visualizations and their CSV source data.
- `paper_figures/`: curated source folders for the six figures in the draft paper.

## Execution Ground Truth

`manifest.json` records what the runtime executed, including fallback and override
placements. The publication exports preserve those placements even when a replay of
the current planner would reject or choose a different target. Planner replay outputs
are diagnostic comparisons, not replacements for the manifest record.

PDF plots retain vector geometry and editable text. PNG files are high-resolution
raster exports. Raw image and NPZ files are retained beside derived JPG, PDF, and CSV
sources where applicable. Point clouds are colorized from the original rectified-left
photo through stored or calibrated UV coordinates; RGB is embedded in the colorized
NPZ and repeated in the CSV source table.
"""
    (PUBLICATION_OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def _preferred_perception_step(run_root: Path) -> Path | None:
    steps = sorted((run_root / "perception_steps").glob("step_*")) if run_root.exists() else []
    return steps[min(1, len(steps) - 1)] if steps else None


def _preferred_planner_step(run_root: Path) -> Path | None:
    steps = sorted((run_root / "planner_steps").glob("step_*")) if run_root.exists() else []
    return steps[min(2, len(steps) - 1)] if steps else None


def _curate_figures_by_run(run_names: list[str]) -> None:
    by_run_root = PUBLICATION_OUTPUT_ROOT / "paper_figures_by_run"
    by_run_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[list[str]] = []

    for run_name in run_names:
        run_root = PUBLICATION_RUNS_ROOT / run_name
        dest_root = by_run_root / run_name
        dest_root.mkdir(parents=True, exist_ok=True)
        rows: list[list[str]] = []

        perception_step = _preferred_perception_step(run_root)
        if perception_step is not None:
            figure2 = dest_root / "figure_02_perception_pipeline"
            _copy_tree_items(perception_step, figure2, ("01_*", "02_*", "03_*", "04_*", "05_*"))
            _copy_tree_items(
                perception_step / "source",
                figure2 / "source",
                ("*.png", "*.jpg", "*.pdf", "*.csv", "*.npz", "*.json"),
            )
            rows.append(["2", "Perception pipeline", str(perception_step), "copied from this run"])

        figure3 = dest_root / "figure_03_topdown_packing"
        copied3 = _copy_tree_items(run_root / "packing", figure3, ("*topdown*",))
        copied3.extend(_copy_tree_items(run_root, figure3, ("placement_reconciliation.csv", "run_audit.json")))
        if copied3:
            rows.append(["3", "Top-down packing", str(run_root / 'packing'), "copied from this run"])

        figure4 = dest_root / "figure_04_3d_planning_scene"
        copied4 = _copy_tree_items(run_root / "packing", figure4, ("*3d*",))
        planner_step = _preferred_planner_step(run_root)
        if planner_step is not None:
            copied4.extend(_copy_tree_items(planner_step, figure4, ("01_scene*", "05_bag_state*", "planner_*")))
        if copied4:
            rows.append(["4", "3D planning scene", str(run_root), "copied from this run"])

        with (dest_root / "figure_source_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["figure", "paper_role", "source_path", "notes"])
            writer.writerows(rows if rows else [["", "no curated figure candidates", str(run_root), "run has no exported perception/planning figure set"]])

        readme_lines = [
            f"# {run_name}",
            "",
            "This folder contains only figure candidates copied from this run.",
            "",
            "See `figure_source_manifest.csv` for exact provenance.",
        ]
        (dest_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
        summary_rows.append([run_name, str(len(rows)), str(dest_root)])

    with (by_run_root / "run_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "figure_groups", "folder"])
        writer.writerows(summary_rows)
    (by_run_root / "README.md").write_text(
        "\n".join(
            [
                "# Figures By Run",
                "",
                "Each run folder contains only figure candidates copied from that run.",
                "This is separate from `paper_figures/`, which is a mixed-source best-of-45 curation.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package run snapshots and paper figure sources into one publication tree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", action="append", type=Path, default=None, help="Specific run directory; repeatable")
    parser.add_argument("--date-prefix", default="20260608", help="Run timestamp date prefix used when --run is omitted")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="Rendered PNG/PDF DPI")
    parser.add_argument("--point-limit", type=int, default=8000, help="Maximum points per frame in CSV and plots")
    parser.add_argument("--no-image-pdfs", action="store_true", help="Skip PDF wrappers for raw camera images")
    parser.add_argument("--curated-run", default="run_20260608_122527GOODONE_ENDOFDAY")
    parser.add_argument("--planning-run", default="run_20260608_124212")
    parser.add_argument("--paper-pdf", type=Path, default=Path(r"C:\Users\elipp\Downloads\Grocery_Bagging_Write_Up_Draft.pdf"))
    parser.add_argument("--presentation-pdf", type=Path, default=Path(r"C:\Users\elipp\Downloads\MAE 162E Final Presentation (11).pdf"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    PUBLICATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PUBLICATION_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    PUBLICATION_GLOBAL_ROOT.mkdir(parents=True, exist_ok=True)

    if args.run:
        run_dirs = [path.expanduser().resolve() for path in args.run]
    else:
        run_dirs = sorted(
            path
            for path in RUN_SNAPSHOTS_ROOT.glob(f"run_{args.date_prefix}_*")
            if path.is_dir()
        )
    if not run_dirs:
        raise FileNotFoundError("no run directories matched the requested selection")

    bag = _load_bag()
    bundle = _load_bundle()
    stereo_calib = _load_stereo_calib()
    if stereo_calib is None:
        print("[COLOR WARN] stereo calibration unavailable; no derived point-cloud plots will be exported")
    summaries = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[RUN {index}/{len(run_dirs)}] {run_dir.name}")
        summaries.append(
            _export_run(
                run_dir,
                dpi=max(100, int(args.dpi)),
                point_limit=max(100, int(args.point_limit)),
                make_image_pdfs=not bool(args.no_image_pdfs),
                bundle=bundle,
                stereo_calib=stereo_calib,
                bag=bag,
            )
        )

    with (PUBLICATION_OUTPUT_ROOT / "run_export_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "manifest_present", "rows", "placed"])
        for summary in summaries:
            writer.writerow([summary["run"], summary["manifest"], summary["rows"], summary["placed"]])

    _curate_figures(
        destination_root=PUBLICATION_FIGURES_ROOT,
        source_scope_label="Mixed-source best-of-45 curation. Figures 2 and 4 come from the curated run; Figure 3 comes from the planning run; Figures 1, 5, and 6 are global/generated.",
        curated_run=args.curated_run,
        planning_run=args.planning_run,
        paper_pdf=args.paper_pdf,
        presentation_pdf=args.presentation_pdf,
        dpi=max(100, int(args.dpi)),
    )
    _curate_figures(
        destination_root=PUBLICATION_OUTPUT_ROOT / "paper_figures_single_run" / args.curated_run,
        source_scope_label=f"Single-run coherent curation. Figures 2, 3, and 4 all come from {args.curated_run}; Figures 1, 5, and 6 are global/generated.",
        curated_run=args.curated_run,
        planning_run=args.curated_run,
        paper_pdf=args.paper_pdf,
        presentation_pdf=args.presentation_pdf,
        dpi=max(100, int(args.dpi)),
    )
    _curate_figures_by_run([run_dir.name for run_dir in run_dirs])
    _write_output_readme()
    print(f"[DONE] publication sources: {PUBLICATION_OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
