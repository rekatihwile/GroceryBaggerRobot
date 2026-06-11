from __future__ import annotations

"""Headless survey-by-survey storyboard generator from a saved run snapshot.

This script reads one manifest row per survey pair and writes five clean images
for each kept survey step:

1) raw rectified stereo pair with detection bbox
2) YOLO overlay + YOLO-masked disparity
3) camera-frame point cloud + robot-frame point cloud
4) detected AABB in robot frame + planned bag placement
5) robot-frame grocery point cloud with padded AABB only

Colors are derived by back-projecting the saved camera-frame point cloud through
the stereo calibration intrinsics to obtain UV pixel coordinates, then sampling
the rectified-left stereo image.

Filtering:
- Uses workspace platform bounds (wet_run profile by default)
- Drops detections outside configured platform XY bounds

Previously placed items are drawn cumulatively in the plan image so each survey
step's 3D view shows the bag filling up.

No live hardware, no camera capture, and no GUI windows are used.
"""

from dataclasses import dataclass
import argparse
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    run_output_dir,
    save_figure_bundle,
    safe_name,
)


# VS Code IDE defaults.
IDE_DEFAULT_RUN_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260608_101557"
IDE_DEFAULT_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None

SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BAG_SCENE_NAME = "New Bag Test"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
DEFAULT_BAG_HEIGHT_MM = 250.0
MAX_POINTS_CAM = 8000
MAX_POINTS_ROBOT = 8000
MAX_POINTS_PLANNED = 5000


@dataclass
class BagVolume:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class SurveyRow:
    manifest_row: int
    class_name: str
    place_result: str
    pick_xy_mm: np.ndarray | None
    pick_phi_deg: float | None
    place_xy_mm: np.ndarray | None
    place_phi_deg: float | None
    destination_surface_z_mm: float | None
    raw_box_center_xyz_mm: np.ndarray | None
    raw_box_size_xyz_mm: np.ndarray | None
    padded_box_center_xyz_mm: np.ndarray | None
    padded_box_size_xyz_mm: np.ndarray | None
    detection_bbox_xyxy: np.ndarray | None
    stereo_left: str | None
    stereo_right: str | None
    left_overlay: str | None
    disparity: str | None
    points_cam: str | None


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


def _load_image_bgr_as_rgb(path: Path | None) -> np.ndarray | None:
    """Load an image saved by OpenCV (BGR byte order) and return float32 RGB [0,1]."""
    if path is None or not path.exists():
        return None
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0
    except Exception:
        return None


def _load_disparity(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "disparity" in data.files:
                arr = np.asarray(data["disparity"], dtype=np.float64)
                return arr if arr.ndim == 2 else None
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim == 2:
                    return arr.astype(np.float64)
    except Exception:
        return None
    return None


def _load_stereo_calib() -> dict[str, np.ndarray] | None:
    """Load stereo calibration needed for UV back-projection."""
    path = _REPO_ROOT / "stereo_calibration.npz"
    if not path.exists():
        print(f"[CALIB WARN] stereo_calibration.npz not found at {path}; point colors will be flat")
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: np.asarray(d[k]) for k in d.files}
    except Exception as exc:
        print(f"[CALIB WARN] failed to load stereo_calibration.npz: {exc}")
        return None


def _project_cam_bundle_to_uv(points_cam: np.ndarray, stereo_calib: dict[str, np.ndarray]) -> np.ndarray | None:
    """Back-project bundle-convention camera-frame points to rectified-left UV pixel coords.

    The saved points_cam in the run-snapshot NPZ files are in the calibration-bundle
    camera convention: xyz_rect @ R_left, with Z then sign-flipped.  To recover UV:
      1. Undo the Z sign-flip.
      2. Undo the rectification rotation (multiply by R_left.T).
      3. Project with P_left_rectified intrinsics.
    """
    if "projection_left_rectified" not in stereo_calib:
        return None
    pts = np.asarray(points_cam, dtype=np.float64).copy().reshape(-1, 3)
    pts[:, 2] *= -1.0  # undo Z sign-flip
    if "rectification_left" in stereo_calib:
        r_left = np.asarray(stereo_calib["rectification_left"], dtype=np.float64)
        pts = pts @ r_left.T  # undo rectification (orthogonal → inverse = transpose)
    P = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    z = pts[:, 2]
    good = z > 1.0
    z_safe = np.where(good, z, 1.0)
    u = np.where(good, fx * pts[:, 0] / z_safe + cx, np.nan)
    v = np.where(good, fy * pts[:, 1] / z_safe + cy, np.nan)
    return np.column_stack([u, v])


def _pick_points_cam(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load camera-frame 3-D points from a run-snapshot NPZ.  No UV saved."""
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        pts = None
        for key in ("points_cam", "points_camera", "cam_points", "points_xyz_cam"):
            if key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    pts = arr[:, :3].astype(np.float64)
                    break
        if pts is None:
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    pts = arr[:, :3].astype(np.float64)
                    break
        if pts is None:
            raise ValueError(f"no Nx3 point array found in {path.name}")
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        raise ValueError(f"points in {path.name} are empty/non-finite")
    return pts, keys


def _sample_colors_from_uv(image_rgb: np.ndarray | None, uv_px: np.ndarray | None, n: int) -> np.ndarray | None:
    if image_rgb is None or uv_px is None or len(uv_px) != n:
        return None
    h, w = image_rgb.shape[:2]
    xs = np.clip(np.rint(uv_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv_px[:, 1]).astype(np.int32), 0, h - 1)
    rgb = image_rgb[ys, xs, :3]
    return np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _load_yolo_segmenter():
    """Load YOLOSegmenter once; returns None if unavailable."""
    try:
        from vision.yolo_segmenter import YOLOSegmenter
        weights = _REPO_ROOT / "yolo_weights" / "full_data.pt"
        fallback = _REPO_ROOT / "Validate_Only_100_Training_Best.pt"
        seg = YOLOSegmenter(
            weights_path=str(weights),
            fallback_weights_path=str(fallback),
            use_half=False,
            warmup_enabled=False,
            retina_masks=True,
            conf=0.30,
        )
        return seg
    except Exception as exc:
        print(f"[YOLO WARN] Could not load segmenter: {exc}")
        return None


def _yolo_segment_row(
    segmenter,
    stereo_left_path: Path | None,
    bbox_xyxy: np.ndarray | None,
    class_name: str,
) -> tuple[Any | None, list[Any]]:
    """Run YOLO on the saved stereo image; return (best_matching_detection, all_detections).

    The best detection is chosen by highest bbox IoU against the saved manifest bbox.
    Returns (None, []) if YOLO is unavailable or no match found above threshold.
    """
    if segmenter is None or stereo_left_path is None or not stereo_left_path.exists():
        return None, []
    bgr = cv2.imread(str(stereo_left_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, []
    try:
        detections = segmenter.segment(bgr)
    except Exception as exc:
        print(f"[YOLO WARN] inference failed on {stereo_left_path.name}: {exc}")
        return None, []
    if not detections:
        return None, []
    ref_bbox = np.asarray(bbox_xyxy, dtype=np.float64) if bbox_xyxy is not None else None
    best_det: Any | None = None
    best_iou = -1.0
    for det in detections:
        iou = _bbox_iou(ref_bbox, np.asarray(det.bbox, dtype=np.float64)) if ref_bbox is not None else 0.0
        if iou > best_iou:
            best_iou = iou
            best_det = det
    if best_det is None or best_iou < 0.10:
        return None, detections
    return best_det, detections


def _render_yolo_overlay_on_image(stereo_rgb: np.ndarray | None, det: Any) -> np.ndarray | None:
    """Draw YOLO segmentation mask + bbox onto the stereo image (float RGB [0,1]).

    Uses OpenCV for drawing so the result matches what the live overlay pipeline produces.
    """
    if stereo_rgb is None:
        return None
    img_u8 = (np.clip(stereo_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    if det is not None:
        mask = getattr(det, "mask", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape == bgr.shape[:2]:
                overlay = bgr.copy()
                overlay[mask] = [0, 180, 60]  # green fill
                cv2.addWeighted(overlay, 0.42, bgr, 0.58, 0, bgr)
        bbox = np.asarray(det.bbox, dtype=np.float64)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 210, 255), 2)
        label = str(getattr(det, "class_name", "") or "")
        conf = float(getattr(det, "confidence", 0.0) or 0.0)
        text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
        cv2.putText(bgr, text, (x1 + 3, max(y1 - 6, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


@dataclass
class _PlacedItem:
    """Accumulates placed items so subsequent survey steps show the bag filling up."""
    class_name: str
    order: int
    center: np.ndarray   # xyz in robot frame (placed position)
    size: np.ndarray     # xyz size mm
    points_robot: np.ndarray  # Nx3 robot-frame cloud at placed position
    colors: np.ndarray | None  # Nx3 RGB float32 (None → flat cyan)


def _disparity_heatmap_bgr(disp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disp)
    if not np.any(valid):
        return np.zeros((*disp.shape, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(disp[valid], [2.0, 98.0])
    norm = np.clip((disp - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def _load_manifest_rows(run_dir: Path) -> list[SurveyRow]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("objects") or []
    if not isinstance(rows, list):
        raise ValueError("manifest objects field is not a list")

    out: list[SurveyRow] = []
    for i, row in enumerate(rows, start=1):
        out.append(
            SurveyRow(
                manifest_row=i,
                class_name=str(row.get("class_name") or row.get("detection_class") or "object"),
                place_result=str(row.get("place_result") or ""),
                pick_xy_mm=_finite_vec(row.get("pick_xy_mm"), 2),
                pick_phi_deg=_finite_float(row.get("pick_phi_deg")),
                place_xy_mm=_finite_vec(row.get("place_xy_mm"), 2),
                place_phi_deg=_finite_float(row.get("place_phi_deg")),
                destination_surface_z_mm=_finite_float(row.get("destination_surface_z_mm")),
                raw_box_center_xyz_mm=_finite_vec(row.get("raw_box_center_xyz_mm"), 3),
                raw_box_size_xyz_mm=_finite_vec(row.get("raw_box_size_xyz_mm"), 3),
                padded_box_center_xyz_mm=_finite_vec(row.get("padded_box_center_xyz_mm"), 3),
                padded_box_size_xyz_mm=_finite_vec(row.get("padded_box_size_xyz_mm"), 3),
                detection_bbox_xyxy=_finite_vec(row.get("detection_bbox_xyxy"), 4),
                stereo_left=row.get("stereo_left"),
                stereo_right=row.get("stereo_right"),
                left_overlay=row.get("left_overlay"),
                disparity=row.get("disparity"),
                points_cam=row.get("points_cam"),
            )
        )
    return out


def _load_bundle() -> dict[str, np.ndarray] | None:
    if not BUNDLE_PATH.exists():
        return None
    try:
        with np.load(BUNDLE_PATH, allow_pickle=False) as data:
            return {k: np.asarray(data[k]) for k in data.files}
    except Exception:
        return None


def _load_bag() -> BagVolume:
    bag_height = DEFAULT_BAG_HEIGHT_MM
    try:
        from config.place import DEFAULT_PLACE

        bag_height = float(getattr(DEFAULT_PLACE, "PLACE_BAG_LOCAL_HEIGHT_MM", DEFAULT_BAG_HEIGHT_MM))
    except Exception:
        pass

    try:
        zones = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
        scene = zones[BAG_SCENE_NAME]
        center = _finite_vec(scene.get("center_xy_mm"), 2)
        width = _finite_float(scene.get("width_mm"))
        depth = _finite_float(scene.get("depth_mm"))
        surface_z = _finite_float(scene.get("surface_z_mm"))
        if center is None or width is None or depth is None:
            raise ValueError("invalid New Bag Test scene")
        z_min = float(surface_z if surface_z is not None else -155.0)
        return BagVolume(
            x_min=float(center[0] - 0.5 * width),
            x_max=float(center[0] + 0.5 * width),
            y_min=float(center[1] - 0.5 * depth),
            y_max=float(center[1] + 0.5 * depth),
            z_min=z_min,
            z_max=float(z_min + bag_height),
        )
    except Exception:
        return BagVolume(x_min=80.0, x_max=340.0, y_min=680.0, y_max=830.0, z_min=-155.0, z_max=-155.0 + bag_height)


# ── Robot-side candidate filter (applied offline from manifest data) ─────────
# Mirrors the gates in scripts/autonomous_best_candidate.py BestCandidateConfig.
# Only gates that can be computed from manifest fields are applied; robot-reach
# and soft-pose checks need a live robot so they are skipped here.

_ROBOT_SIDE_MIN_VOLUME_MM3: float = 1.0
_ROBOT_SIDE_MAX_VOLUME_MM3: float = 3_000_000.0
# Reject stereo phantoms: if top of raw box would push gripper above this Z,
# the detection was noise (matches BestCandidateConfig defaults).
_ROBOT_SIDE_GRIPPER_OFFSET_MM: float = 130.0
_ROBOT_SIDE_Z_MAX_MM: float = 275.0


def _robot_side_pass(row: SurveyRow) -> tuple[bool, str]:
    """Apply offline robot-side candidate gates.  Returns (pass, reason_if_rejected)."""
    # Volume gate
    if row.raw_box_size_xyz_mm is not None:
        size = row.raw_box_size_xyz_mm
        vol = float(size[0]) * float(size[1]) * float(size[2])
        if vol < _ROBOT_SIDE_MIN_VOLUME_MM3:
            return False, f"volume {vol:.0f} mm3 < min {_ROBOT_SIDE_MIN_VOLUME_MM3:.0f}"
        if vol > _ROBOT_SIDE_MAX_VOLUME_MM3:
            return False, f"volume {vol:.0f} mm3 > max {_ROBOT_SIDE_MAX_VOLUME_MM3:.0f}"

    # Stereo Z phantom gate: top of object + gripper offset must be below Z_MAX
    if row.raw_box_center_xyz_mm is not None and row.raw_box_size_xyz_mm is not None:
        top_z = float(row.raw_box_center_xyz_mm[2]) + 0.5 * float(row.raw_box_size_xyz_mm[2])
        if top_z + _ROBOT_SIDE_GRIPPER_OFFSET_MM > _ROBOT_SIDE_Z_MAX_MM:
            return False, (f"stereo top_z {top_z:.0f} + gripper_offset {_ROBOT_SIDE_GRIPPER_OFFSET_MM:.0f} "
                           f"> Z_MAX {_ROBOT_SIDE_Z_MAX_MM:.0f} (phantom rejection)")

    return True, ""


def _platform_pass(row: SurveyRow, x_min: float, x_max: float, y_min: float, y_max: float, require_positive: bool) -> bool:
    xy = row.pick_xy_mm
    if xy is None and row.raw_box_center_xyz_mm is not None:
        xy = row.raw_box_center_xyz_mm[:2]
    if xy is None:
        return False
    x, y = float(xy[0]), float(xy[1])
    if require_positive and (x <= 0.0 or y <= 0.0):
        return False
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)


def _sample_points(points_xyz: np.ndarray, colors: np.ndarray | None, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    n = len(points_xyz)
    if n <= max_n:
        return points_xyz, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return points_xyz[idx], (colors[idx] if colors is not None else None)


def _planned_place_position(row: SurveyRow) -> tuple[np.ndarray, float] | None:
    """Return (target_xy_mm, destination_z_mm) for bag placement, or None if not determinable.

    target_xy_mm     — where the robot places the XY centroid (place_xy_mm with fallbacks).
    destination_z_mm — Z of the surface the item rests on in the bag
                       (destination_surface_z_mm with fallbacks).
    """
    if row.place_xy_mm is not None:
        target_xy = row.place_xy_mm.copy()
    elif row.padded_box_center_xyz_mm is not None:
        target_xy = row.padded_box_center_xyz_mm[:2].copy()
    elif row.raw_box_center_xyz_mm is not None:
        target_xy = row.raw_box_center_xyz_mm[:2].copy()
    else:
        return None

    if row.destination_surface_z_mm is not None:
        dest_z = float(row.destination_surface_z_mm)
    elif row.padded_box_center_xyz_mm is not None and row.padded_box_size_xyz_mm is not None:
        dest_z = float(row.padded_box_center_xyz_mm[2]) - 0.5 * float(row.padded_box_size_xyz_mm[2])
    elif row.raw_box_center_xyz_mm is not None and row.raw_box_size_xyz_mm is not None:
        dest_z = float(row.raw_box_center_xyz_mm[2]) - 0.5 * float(row.raw_box_size_xyz_mm[2])
    else:
        return None

    return target_xy, dest_z


def _rotate_pointcloud_z_deg(
    points_xyz: np.ndarray,
    *,
    pick_phi_deg: float | None,
    place_phi_deg: float | None,
) -> np.ndarray:
    """Rotate the survey-frame cloud about Z by (place_phi - pick_phi) through its centroid.

    The stereo survey captures the object at pick_phi orientation.  The robot
    then picks and places at place_phi, physically rotating the object by that
    delta.  Replicating the exact rotation here makes the bag-space cloud match
    the actual placed orientation — same path the wet-run script takes.

    Z is never modified (4-DOF SCARA always places items right-side-up).
    """
    if pick_phi_deg is None:
        return np.asarray(points_xyz, dtype=np.float64)

    p_place = float(place_phi_deg) if place_phi_deg is not None else 0.0
    delta_deg = p_place - float(pick_phi_deg)
    delta_deg = ((delta_deg + 180.0) % 360.0) - 180.0  # normalise to [-180, 180]

    if abs(delta_deg) < 0.1:
        return np.asarray(points_xyz, dtype=np.float64)

    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    theta = np.deg2rad(delta_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    rotated = pts.copy()
    rotated[:, 0] = cx + cos_t * dx - sin_t * dy
    rotated[:, 1] = cy + sin_t * dx + cos_t * dy
    return rotated


def _translate_cloud_to_bag_position(
    points_xyz: np.ndarray,
    target_center_xy: np.ndarray,
    destination_z_mm: float,
) -> np.ndarray:
    """Translate the cloud so its XY centroid lands at target_center_xy and its
    Z minimum aligns with destination_z_mm (bag surface or top of existing stack).

    No points are filtered — full cloud is preserved so Z extent (e.g. the top
    face of a Pringles lid) is not clipped.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    centroid_xy = np.mean(pts[:, :2], axis=0)
    min_z = float(np.min(pts[:, 2]))
    shift = np.array([
        float(target_center_xy[0]) - centroid_xy[0],
        float(target_center_xy[1]) - centroid_xy[1],
        destination_z_mm - min_z,
    ], dtype=np.float64)
    return pts + shift.reshape(1, 3)


def _aabb_from_cloud(points_xyz: np.ndarray, *, margin_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute AABB center and size from the 2–98th percentile of the cloud plus a margin."""
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    size = np.maximum(hi - lo + 2.0 * margin_mm, np.array([4.0, 4.0, 4.0], dtype=np.float64))
    center = 0.5 * (lo + hi)
    return center.astype(np.float64), size.astype(np.float64)


def _save_source_materials(
    source_dir: Path,
    *,
    run_dir: Path,
    row: "SurveyRow",
    stereo_left_rgb: "np.ndarray | None",
    stereo_right_rgb: "np.ndarray | None",
    yolo_overlay_rgb: "np.ndarray | None",
    disparity: "np.ndarray | None",
    yolo_mask: "np.ndarray | None",
) -> None:
    """Write raw source files into a per-survey source/ subfolder.

    Saves:
      stereo_left_full.png      — raw left stereo image (full quality)
      stereo_right_full.png     — raw right stereo image
      yolo_seg.png              — YOLO segmentation overlay on stereo left
      disparity_heatmap.png     — RAFT disparity as TURBO colour map (no mask)
      yolo_seg_over_heatmap.png — disparity heatmap masked to the YOLO region
      manifest_row.json         — this row's manifest data (raw JSON)
      packing_aabb.json         — AABB box sizes and centres for this item
    """
    import shutil

    source_dir.mkdir(parents=True, exist_ok=True)

    # ── Raw stereo images (copy original files, preserving full quality) ──────
    for fname, dest in [
        (row.stereo_left, "stereo_left_full.png"),
        (row.stereo_right, "stereo_right_full.png"),
    ]:
        if fname:
            src = run_dir / fname
            if src.exists():
                shutil.copy2(src, source_dir / dest)

    # ── YOLO segmentation overlay ─────────────────────────────────────────────
    if yolo_overlay_rgb is not None:
        u8 = (np.clip(yolo_overlay_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(source_dir / "yolo_seg.png"), bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # ── Disparity heatmap (unmasked) ─────────────────────────────────────────
    if disparity is not None:
        heat = _disparity_heatmap_bgr(disparity)
        cv2.imwrite(str(source_dir / "disparity_heatmap.png"), heat,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])

        # YOLO-masked heatmap (only the detected grocery region lit up)
        if yolo_mask is not None and yolo_mask.shape == disparity.shape:
            masked = np.zeros_like(heat)
            masked[yolo_mask] = heat[yolo_mask]
        else:
            masked = heat
        cv2.imwrite(str(source_dir / "yolo_seg_over_heatmap.png"), masked,
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])

    # ── Manifest row JSON ─────────────────────────────────────────────────────
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        full = json.loads(manifest_path.read_text(encoding="utf-8"))
        objs = full.get("objects") or []
        row_data: dict = {}
        if 0 <= row.manifest_row - 1 < len(objs):
            row_data = objs[row.manifest_row - 1]
        (source_dir / "manifest_row.json").write_text(
            json.dumps(row_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Packing / AABB JSON ───────────────────────────────────────────────────
    aabb_info: dict = {
        "class_name": row.class_name,
        "place_result": row.place_result,
        "pick_xy_mm": row.pick_xy_mm.tolist() if row.pick_xy_mm is not None else None,
        "pick_phi_deg": row.pick_phi_deg,
        "place_xy_mm": row.place_xy_mm.tolist() if row.place_xy_mm is not None else None,
        "place_phi_deg": row.place_phi_deg,
        "destination_surface_z_mm": row.destination_surface_z_mm,
        "raw_box_center_xyz_mm": row.raw_box_center_xyz_mm.tolist() if row.raw_box_center_xyz_mm is not None else None,
        "raw_box_size_xyz_mm": row.raw_box_size_xyz_mm.tolist() if row.raw_box_size_xyz_mm is not None else None,
        "padded_box_center_xyz_mm": row.padded_box_center_xyz_mm.tolist() if row.padded_box_center_xyz_mm is not None else None,
        "padded_box_size_xyz_mm": row.padded_box_size_xyz_mm.tolist() if row.padded_box_size_xyz_mm is not None else None,
    }
    (source_dir / "packing_aabb.json").write_text(
        json.dumps(aabb_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _bag_corners(bag: BagVolume) -> np.ndarray:
    mn = np.array([bag.x_min, bag.y_min, bag.z_min], dtype=np.float64)
    mx = np.array([bag.x_max, bag.y_max, bag.z_max], dtype=np.float64)
    return np.array(
        [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )


def _draw_box(ax, center: np.ndarray, size: np.ndarray, color: str = "#facc15") -> None:
    mn = center - 0.5 * size
    mx = center + 0.5 * size
    c = np.array(
        [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float64,
    )
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], [c[a, 2], c[b, 2]], color=color, linewidth=1.2, alpha=0.95)
    faces = Poly3DCollection(
        [[c[i] for i in face] for face in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5])],
        alpha=0.08,
    )
    faces.set_facecolor(color)
    faces.set_edgecolor("none")
    ax.add_collection3d(faces)


def _derive_pointcloud_overlay_box(
    points_robot: np.ndarray | None,
    row: SurveyRow,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a single object-local overlay box around the survey-frame grocery cloud."""
    if points_robot is None or len(points_robot) == 0:
        return None

    pts = np.asarray(points_robot, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return None

    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    size = hi - lo

    raw_size = row.raw_box_size_xyz_mm if row.raw_box_size_xyz_mm is not None else None
    if raw_size is not None:
        size = np.maximum(size, np.asarray(raw_size, dtype=np.float64))

    size = np.maximum(size, np.array([8.0, 8.0, 8.0], dtype=np.float64))
    margin = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    size = size + margin

    center = 0.5 * (lo + hi)
    return center.astype(np.float64), size.astype(np.float64)


def _save_stereo_pair(
    out_path: Path,
    left_rgb: np.ndarray | None,
    right_rgb: np.ndarray | None,
    row: SurveyRow,
    dpi: int,
) -> None:
    """Output 1: raw rectified stereo pair (left + right side-by-side), bbox on left."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 5.2), facecolor="#080b14")
    blank = np.zeros((480, 640, 3), dtype=np.float32)
    for ax, img, side in [(ax_l, left_rgb, "Left"), (ax_r, right_rgb, "Right")]:
        ax.set_facecolor("#0f172a")
        ax.imshow(img if img is not None else blank)
        ax.set_xticks([])
        ax.set_yticks([])
    if row.detection_bbox_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in row.detection_bbox_xyxy]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#facc15", linewidth=2.0)
        ax_l.add_patch(rect)
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_yolo_and_disparity(
    out_path: Path,
    yolo_overlay_rgb: np.ndarray | None,
    disparity: np.ndarray | None,
    yolo_mask: np.ndarray | None,
    row: SurveyRow,
    dpi: int,
) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.float32)

    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(7.0, 5.2), facecolor="#080b14")
    ax_overlay.set_facecolor("#0f172a")
    ax_overlay.imshow(yolo_overlay_rgb if yolo_overlay_rgb is not None else blank)
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])
    fig_overlay.tight_layout()
    save_figure_bundle(
        fig_overlay,
        out_path.with_name("02_yolo_overlay"),
        dpi=dpi,
        facecolor=fig_overlay.get_facecolor(),
    )
    plt.close(fig_overlay)

    fig_disp, ax_disp = plt.subplots(1, 1, figsize=(7.0, 5.2), facecolor="#080b14")
    ax_disp.set_facecolor("#0f172a")
    if disparity is None:
        ax_disp.imshow(blank.astype(np.uint8))
    else:
        heat = _disparity_heatmap_bgr(disparity)
        if yolo_mask is not None and yolo_mask.shape == disparity.shape:
            masked = np.zeros_like(heat)
            masked[yolo_mask] = heat[yolo_mask]
            heat = masked
        ax_disp.imshow(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))
    ax_disp.set_xticks([])
    ax_disp.set_yticks([])
    fig_disp.tight_layout()
    save_figure_bundle(
        fig_disp,
        out_path.with_name("02_masked_disparity"),
        dpi=dpi,
        facecolor=fig_disp.get_facecolor(),
    )
    plt.close(fig_disp)


def _save_dual_pointclouds(
    out_path: Path,
    points_cam: np.ndarray,
    colors_cam: np.ndarray | None,
    points_robot: np.ndarray | None,
    colors_robot: np.ndarray | None,
    row: SurveyRow,
    dpi: int,
) -> None:
    """Output 3: camera-frame point cloud (left) + robot-frame point cloud (right)."""
    fig = plt.figure(figsize=(14.0, 6.8), facecolor="#080b14")
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")

    def _draw_pc(ax, pts: np.ndarray, cols: np.ndarray | None, seed: int) -> None:
        ax.set_facecolor("#0f172a")
        if pts is None or len(pts) == 0:
            return
        p, c = _sample_points(pts, cols, MAX_POINTS_CAM, seed)
        if c is not None:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                       c=c, s=1.0, alpha=0.68, depthshade=False)
        mn, mx = np.min(p, axis=0), np.max(p, axis=0)
        ctr = 0.5 * (mn + mx)
        span = max(float(np.max(mx - mn)), 1.0)
        half = 0.55 * span
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=-56)

    _draw_pc(ax_l, points_cam, colors_cam, 2000 + row.manifest_row)
    if points_robot is not None and len(points_robot) > 0:
        _draw_pc(ax_r, points_robot, colors_robot, 3000 + row.manifest_row)
    else:
        ax_r.set_facecolor("#0f172a")
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_aabb_and_plan(
    out_path: Path,
    row: SurveyRow,
    points_robot: np.ndarray | None,
    colors_robot: np.ndarray | None,
    planned_center: np.ndarray | None,
    planned_size: np.ndarray | None,
    planned_points_robot: np.ndarray | None,
    planned_colors: np.ndarray | None,
    bag: BagVolume,
    placed_items: list[_PlacedItem],
    dpi: int,
) -> None:
    """Output 4: detected AABB in robot frame (left) + planned bag placement (right)."""
    fig = plt.figure(figsize=(14.0, 6.8), facecolor="#080b14")
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")

    # ── Left: AABB of detected grocery in robot frame ──────────────────────
    ax_l.set_facecolor("#0f172a")
    scale_l: list[np.ndarray] = []
    if points_robot is not None and len(points_robot) > 0:
        pts, cols = _sample_points(points_robot, colors_robot, MAX_POINTS_ROBOT, 4000 + row.manifest_row)
        if cols is not None:
            ax_l.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=cols, s=1.0, alpha=0.68, depthshade=False)
        scale_l.append(pts)
    if row.raw_box_center_xyz_mm is not None and row.raw_box_size_xyz_mm is not None:
        _draw_box(ax_l, row.raw_box_center_xyz_mm, row.raw_box_size_xyz_mm, color="#facc15")

        scale_l.append(np.array([
            row.raw_box_center_xyz_mm - 0.5 * row.raw_box_size_xyz_mm,
            row.raw_box_center_xyz_mm + 0.5 * row.raw_box_size_xyz_mm,
        ], dtype=np.float64))
    if row.padded_box_center_xyz_mm is not None and row.padded_box_size_xyz_mm is not None:
        _draw_box(ax_l, row.padded_box_center_xyz_mm, row.padded_box_size_xyz_mm, color="#60a5fa")
        scale_l.append(np.array([
            row.padded_box_center_xyz_mm - 0.5 * row.padded_box_size_xyz_mm,
            row.padded_box_center_xyz_mm + 0.5 * row.padded_box_size_xyz_mm,
        ], dtype=np.float64))
    if scale_l:
        all_l = np.vstack(scale_l)
        mn, mx = np.min(all_l, axis=0), np.max(all_l, axis=0)
        ctr = 0.5 * (mn + mx)
        span = max(float(np.max(mx - mn)), 100.0)
        half = 0.60 * span
        ax_l.set_xlim(ctr[0] - half, ctr[0] + half)
        ax_l.set_ylim(ctr[1] - half, ctr[1] + half)
        ax_l.set_zlim(ctr[2] - half, ctr[2] + half)
    ax_l.view_init(elev=24, azim=-58)
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    ax_l.set_zticks([])

    # ── Right: Planned placement in bag ────────────────────────────────────
    ax_r.set_facecolor("#0f172a")
    bag_c = _bag_corners(bag)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax_r.plot([bag_c[a, 0], bag_c[b, 0]], [bag_c[a, 1], bag_c[b, 1]], [bag_c[a, 2], bag_c[b, 2]],
                  color="#e5e7eb", linewidth=1.0, alpha=0.7)
    for item in placed_items:
        if len(item.points_robot) > 0:
            pts, cols = _sample_points(item.points_robot, item.colors, MAX_POINTS_PLANNED, 500 + item.order)
            if cols is not None:
                ax_r.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                             c=cols, s=1.0, alpha=0.62, depthshade=False)
        _draw_box(ax_r, item.center, item.size, color="#64748b")
    if planned_points_robot is not None and len(planned_points_robot) > 0:
        pts, cols = _sample_points(planned_points_robot, planned_colors, MAX_POINTS_PLANNED, 1000 + row.manifest_row)
        if cols is not None:
            ax_r.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=cols, s=1.5, alpha=0.72, depthshade=False)
    if planned_center is not None and planned_size is not None:
        _draw_box(ax_r, planned_center, planned_size, color="#facc15")
    scale_r: list[np.ndarray] = [bag_c]
    if planned_points_robot is not None and len(planned_points_robot) > 0:
        scale_r.append(planned_points_robot)
    for item in placed_items:
        if len(item.points_robot) > 0:
            scale_r.append(item.points_robot)
    if planned_center is not None and planned_size is not None:
        scale_r.append(np.array([planned_center - 0.5 * planned_size,
                                  planned_center + 0.5 * planned_size], dtype=np.float64))
    all_r = np.vstack(scale_r)
    mn, mx = np.min(all_r, axis=0), np.max(all_r, axis=0)
    ctr = 0.5 * (mn + mx)
    span = max(float(np.max(mx - mn)), 1.0)
    half = 0.55 * span
    ax_r.set_xlim(ctr[0] - half, ctr[0] + half)
    ax_r.set_ylim(ctr[1] - half, ctr[1] + half)
    ax_r.set_zlim(ctr[2] - half, ctr[2] + half)
    ax_r.set_xticks([])
    ax_r.set_yticks([])
    ax_r.set_zticks([])
    ax_r.view_init(elev=24, azim=-58)
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_robot_pointcloud_padded_box(
    out_path: Path,
    row: SurveyRow,
    points_robot: np.ndarray | None,
    colors_robot: np.ndarray | None,
    dpi: int,
) -> None:
    """Output 5: robot-frame grocery point cloud with a single overlay box only."""
    fig = plt.figure(figsize=(7.2, 6.8), facecolor="#080b14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f172a")
    overlay_box = _derive_pointcloud_overlay_box(points_robot, row)
    scale_chunks: list[np.ndarray] = []
    if points_robot is not None and len(points_robot) > 0:
        pts, cols = _sample_points(points_robot, colors_robot, MAX_POINTS_ROBOT, 7000 + row.manifest_row)
        if cols is not None:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=cols,
                s=1.0,
                alpha=0.70,
                depthshade=False,
            )
        scale_chunks.append(pts)

    if overlay_box is not None:
        box_center, box_size = overlay_box
        _draw_box(ax, box_center, box_size, color="#facc15")
        scale_chunks.append(
            np.array(
                [
                    box_center - 0.5 * box_size,
                    box_center + 0.5 * box_size,
                ],
                dtype=np.float64,
            )
        )

    if scale_chunks:
        all_pts = np.vstack(scale_chunks)
        mn, mx = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        ctr = 0.5 * (mn + mx)
        span = max(float(np.max(mx - mn)), 1.0)
        half = 0.55 * span
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_zlim(ctr[2] - half, ctr[2] + half)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    ax.view_init(elev=22, azim=-56)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    fig.tight_layout()
    save_figure_bundle(fig, out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean headless survey images per manifest row from a saved run snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run snapshot directory with manifest.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory root")
    parser.add_argument("--workspace-profile", default="wet_run", help="Workspace profile for platform filtering")
    parser.add_argument("--limit", type=int, default=None, help="Optional max kept surveys")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="Output image DPI")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _resolve_default_run_dir()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else _resolve_default_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest_rows(run_dir)
    workspace_cfg = get_workspace_filter_config(str(args.workspace_profile))
    x_min, x_max, y_min, y_max = workspace_bounds_mm(workspace_cfg)

    bundle = _load_bundle()
    stereo_calib = _load_stereo_calib()
    transform = None
    if bundle is not None:
        try:
            from vision.pointcloud import cam_points_to_robot_xyz

            transform = cam_points_to_robot_xyz
        except Exception:
            transform = None

    print("[YOLO] Loading segmenter for offline mask generation...")
    yolo_segmenter = _load_yolo_segmenter()
    if yolo_segmenter is None:
        print("[YOLO WARN] Segmenter unavailable — disparity mask will be empty.")

    bag = _load_bag()

    placed_items: list[_PlacedItem] = []
    kept = 0
    skipped = 0
    for row in rows:
        if args.limit is not None and kept >= int(args.limit):
            break

        if not _platform_pass(row, x_min, x_max, y_min, y_max, workspace_cfg.require_positive_platform_xy):
            skipped += 1
            print(f"[SKIP] row {row.manifest_row}: outside platform bounds")
            continue

        robot_ok, robot_reason = _robot_side_pass(row)
        if not robot_ok:
            skipped += 1
            print(f"[SKIP] row {row.manifest_row}: robot-side filter — {robot_reason}")
            continue

        if not row.points_cam or not row.disparity:
            skipped += 1
            print(f"[SKIP] row {row.manifest_row}: missing points_cam/disparity")
            continue

        points_cam_path = run_dir / row.points_cam
        disparity_path = run_dir / row.disparity
        stereo_left_path = run_dir / row.stereo_left if row.stereo_left else None
        overlay_path = run_dir / row.left_overlay if row.left_overlay else None

        try:
            points_cam, _keys = _pick_points_cam(points_cam_path)
        except Exception as exc:
            skipped += 1
            print(f"[SKIP] row {row.manifest_row}: cannot load points ({exc})")
            continue

        # Back-project camera points → UV pixel coords using stereo intrinsics
        uv_px = _project_cam_bundle_to_uv(points_cam, stereo_calib) if stereo_calib is not None else None

        # Load images using OpenCV (BGR saved by cv2.imwrite) and convert to float RGB
        stereo_left_rgb = _load_image_bgr_as_rgb(stereo_left_path)
        stereo_right_path = run_dir / row.stereo_right if row.stereo_right else None
        stereo_right_rgb = _load_image_bgr_as_rgb(stereo_right_path)
        colors_cam = _sample_colors_from_uv(stereo_left_rgb, uv_px, len(points_cam))

        disparity = _load_disparity(disparity_path)

        # Re-run YOLO offline on the raw stereo image for fresh overlay + segmentation mask
        matched_det, _all_dets = _yolo_segment_row(
            yolo_segmenter,
            stereo_left_path,
            row.detection_bbox_xyxy,
            row.class_name,
        )
        if matched_det is None:
            print(f"[MASK WARN] row {row.manifest_row}: YOLO found no match — disparity mask will be empty")
        yolo_mask = np.asarray(matched_det.mask, dtype=bool) if (matched_det is not None and getattr(matched_det, "mask", None) is not None) else None
        yolo_overlay_rgb = _render_yolo_overlay_on_image(stereo_left_rgb, matched_det)

        # Camera → robot transform; keep separate color arrays for cam and robot frames
        colors_robot: np.ndarray | None = None
        points_robot: np.ndarray | None = None
        if transform is not None and bundle is not None:
            try:
                pts_r_all = np.asarray(transform(points_cam, bundle), dtype=np.float64).reshape(-1, 3)
                finite = np.all(np.isfinite(pts_r_all), axis=1)
                points_robot = pts_r_all[finite]
                if colors_cam is not None and len(colors_cam) == len(finite):
                    colors_robot = colors_cam[finite]
            except Exception as exc:
                print(f"[WARN] row {row.manifest_row}: robot transform failed ({exc})")
                points_robot = None

        planned_center = None
        planned_size = None
        planned_points_robot = None
        planned_colors = None
        place_pos = _planned_place_position(row)
        if place_pos is not None and points_robot is not None and len(points_robot) > 0:
            target_xy, dest_z = place_pos
            # 1. Rotate to match placement orientation: apply exact (place_phi - pick_phi)
            #    rotation about Z through the cloud centroid — same transform the robot does.
            rotated = _rotate_pointcloud_z_deg(
                points_robot, pick_phi_deg=row.pick_phi_deg, place_phi_deg=row.place_phi_deg
            )
            # 2. Translate: XY centroid → place_xy, Z-min → destination surface.
            #    No inside-filter so the full cloud height (lid, top face, etc.) is kept.
            planned_points_robot = _translate_cloud_to_bag_position(rotated, target_xy, dest_z)
            # 3. Derive AABB from the actual placed cloud — box naturally fits the
            #    rotated footprint rather than the survey-frame axis-aligned box.
            if len(planned_points_robot) > 0:
                planned_center, planned_size = _aabb_from_cloud(planned_points_robot)
            planned_colors = colors_robot  # 1-to-1 with points_robot; no filtering

        pair_index = _parse_pair_index(row.stereo_left) or row.manifest_row
        step_dir = (
            out_dir
            / "perception_steps"
            / f"step_{pair_index:04d}_row_{row.manifest_row:02d}_{safe_name(row.class_name)}"
        )
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save raw source materials into a source/ subfolder
        _save_source_materials(
            step_dir / "source",
            run_dir=run_dir,
            row=row,
            stereo_left_rgb=stereo_left_rgb,
            stereo_right_rgb=stereo_right_rgb,
            yolo_overlay_rgb=yolo_overlay_rgb,
            disparity=disparity,
            yolo_mask=yolo_mask,
        )

        # Output 1: raw rectified stereo pair
        _save_stereo_pair(
            step_dir / "01_stereo_raw_pair.png",
            stereo_left_rgb,
            stereo_right_rgb,
            row,
            dpi=max(80, int(args.dpi)),
        )

        # Output 2: YOLO overlay (re-run on raw stereo, not saved overlay) + masked disparity
        _save_yolo_and_disparity(
            step_dir / "02_yolo_overlay_and_masked_disparity.png",
            yolo_overlay_rgb,
            disparity,
            yolo_mask,
            row,
            dpi=max(80, int(args.dpi)),
        )

        # Output 3: camera-frame point cloud + robot-frame point cloud
        _save_dual_pointclouds(
            step_dir / "03_stereo_and_robot_pointclouds.png",
            points_cam,
            colors_cam,
            points_robot,
            colors_robot,
            row,
            dpi=max(80, int(args.dpi)),
        )

        # Output 4: detected AABB in robot frame + planned placement in bag
        _save_aabb_and_plan(
            step_dir / "04_aabb_and_planned_placement.png",
            row,
            points_robot,
            colors_robot,
            planned_center,
            planned_size,
            planned_points_robot,
            planned_colors,
            bag,
            placed_items=list(placed_items),
            dpi=max(80, int(args.dpi)),
        )

        # Output 5: clean robot-frame grocery point cloud with padded AABB only
        _save_robot_pointcloud_padded_box(
            step_dir / "05_robot_pointcloud_with_padded_box.png",
            row,
            points_robot,
            colors_robot,
            dpi=max(80, int(args.dpi)),
        )

        # Accumulate this item for subsequent survey step plan images
        if row.place_result.lower().strip() in ("placed", "success", "ok") and planned_points_robot is not None and planned_center is not None and planned_size is not None:
            placed_items.append(
                _PlacedItem(
                    class_name=row.class_name,
                    order=kept + 1,
                    center=np.asarray(planned_center, dtype=np.float64),
                    size=np.asarray(planned_size, dtype=np.float64),
                    points_robot=np.asarray(planned_points_robot, dtype=np.float64),
                    colors=(np.asarray(planned_colors, dtype=np.float32) if planned_colors is not None else None),
                )
            )

        kept += 1
        print(f"[SAVE] row {row.manifest_row}: wrote 5 survey images to {step_dir}")

    print(f"[SUMMARY] kept={kept} skipped={skipped} placed_accumulated={len(placed_items)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
