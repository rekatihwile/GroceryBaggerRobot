from __future__ import annotations

"""Offline placement-animation GIF generator from a saved run snapshot.

Reads the manifest.json from a run snapshot directory, reconstructs the full
pick-and-place animation for every successfully placed object, and writes a
single animated GIF to the output directory.

The visual style mirrors autonomous_system_wrapper.py:
  - same 3D gripper geometry (finger L/R + palm)
  - same bag platform / place zone outline
  - same platform bounds rectangle
  - colored point clouds via UV back-projection from the saved stereo image

Edit the configuration block below to tune the viewer angle, point size,
frame rate, and other visual parameters.
"""

# ─── VIEWER CONFIGURATION ───────────────────────────────────────────────────
# Edit these constants to change the appearance without touching the logic.
# Override any constant by creating view_config.json next to this script
# (use scripts/capstone/run_snapshot_view_config.py to generate it interactively).

VIEW_ELEV_DEG        = 24          # 3D view elevation angle (degrees)
VIEW_AZIM_DEG        = -58         # 3D view azimuth angle (degrees)

POINT_SIZE           = 1.0         # scatter plot marker size for point clouds
POINT_ALPHA          = 0.55        # scatter plot transparency
MAX_PTS_PER_OBJECT   = 600         # max points per object (subsampled if larger)

FIG_W_IN             = 7.0         # figure width in inches
FIG_H_IN             = 5.5         # figure height in inches
GIF_DPI              = 72          # DPI for each GIF frame (lower = smaller file)
GIF_FRAME_MS         = 80          # delay between frames in milliseconds
N_ANIM_STEPS         = 8           # interpolation steps per animation phase

# Axis limits  (None = auto-fit each frame to all scene points)
X_LIM_MM             = None        # (xmin, xmax) or None for auto
Y_LIM_MM             = None        # (ymin, ymax) or None for auto
Z_LIM_MM             = None        # (zmin, zmax) or None for auto

# Gripper geometry (mm) — kept in sync with DEFAULT_GRIPPER_GEOMETRY
FINGER_LENGTH_MM     = 70.0
FINGER_WIDTH_MM      = 12.0
FINGER_DEPTH_MM      = 35.0
PALM_WIDTH_MM        = 40.0
PALM_HEIGHT_MM       = 18.0

# Animation clearance (mm)
HOVER_CLEARANCE_MM   = 60.0
TRAVEL_CLEARANCE_MM  = 80.0
OPEN_EXTRA_MM        = 30.0
CLOSE_EXTRA_MM       = 8.0

# Survey pose (mm) — default survey XYZ from pick config
SURVEY_X_MM          = 80.0        # override if DEFAULT_PICK differs
SURVEY_Y_MM          = -50.0
SURVEY_Z_MM          = 270.0

# Colors
BG_COLOR             = "#1a1a2e"
PANEL_BG_COLOR       = "#0d0d1a"
BAG_COLOR            = "#7fd0a8"
PLATFORM_COLOR       = "#5ca37e"
BOUNDS_COLOR         = "#f2c66d"
GRIPPER_OPEN_COLOR   = "#8fe7ff"
GRIPPER_HOLD_COLOR   = "#ffd36e"
PLACED_LABEL_COLOR   = "#f0fff0"

# ─── END CONFIGURATION ──────────────────────────────────────────────────────

import argparse
import json
import math
from dataclasses import dataclass
from io import BytesIO
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

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from scripts.capstone.publication_config import run_output_dir
from scripts.capstone.pointcloud_color import photo_colors_for_points

# ── IDE defaults ─────────────────────────────────────────────────────────────
IDE_DEFAULT_RUN_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260608_101557"

_RUN_DIR: Path | None = Path(IDE_DEFAULT_RUN_DIR_STR) if IDE_DEFAULT_RUN_DIR_STR else None

SURFACE_ZONES_PATH = _REPO_ROOT / "config" / "surface_zones.json"
BAG_SCENE_NAME = "New Bag Test"
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
BAG_PLATFORM_THICKNESS_MM = 8.0
PLACE_ZONE_DISPLAY_HEIGHT_MM = 260.0  # visual bag height

# view_config.json written by scripts/capstone/run_snapshot_view_config.py
VIEW_CONFIG_PATH = Path(__file__).resolve().parent / "view_config.json"


def _apply_view_config() -> None:
    """Load view_config.json and overwrite module-level config constants if present."""
    if not VIEW_CONFIG_PATH.exists():
        return
    try:
        cfg = json.loads(VIEW_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[CONFIG] Could not parse {VIEW_CONFIG_PATH}: {exc}")
        return
    g = globals()
    _map = {
        "view": {"elev_deg": "VIEW_ELEV_DEG", "azim_deg": "VIEW_AZIM_DEG"},
        "style": {
            "point_size": "POINT_SIZE", "point_alpha": "POINT_ALPHA",
            "gif_dpi": "GIF_DPI", "gif_frame_ms": "GIF_FRAME_MS",
            "n_anim_steps": "N_ANIM_STEPS", "fig_w_in": "FIG_W_IN", "fig_h_in": "FIG_H_IN",
        },
    }
    for section, keys in _map.items():
        sec = cfg.get(section, {})
        for json_key, py_name in keys.items():
            if json_key in sec and sec[json_key] is not None:
                g[py_name] = type(g[py_name])(sec[json_key])
    lims = cfg.get("limits", {})
    for axis, py_name in [("x", "X_LIM_MM"), ("y", "Y_LIM_MM"), ("z", "Z_LIM_MM")]:
        if axis in lims:
            val = lims[axis]
            g[py_name] = (
                tuple(val) if (isinstance(val, list) and len(val) == 2
                               and all(v is not None for v in val))
                else None
            )
    print(f"[CONFIG] Loaded view overrides from {VIEW_CONFIG_PATH}")


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class _Box:
    """Minimal stand-in for AxisAlignedBox3D."""
    label: str
    min_xyz: np.ndarray   # shape (3,) mm
    max_xyz: np.ndarray   # shape (3,) mm

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min_xyz + self.max_xyz)

    @property
    def size(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz


def _make_box(center: np.ndarray, size: np.ndarray, label: str = "") -> _Box:
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = np.asarray(size, dtype=np.float64).reshape(3)
    return _Box(label=label, min_xyz=c - 0.5 * s, max_xyz=c + 0.5 * s)


def _make_box_from_min_max(mn: np.ndarray, mx: np.ndarray, label: str = "") -> _Box:
    return _Box(label=label, min_xyz=np.asarray(mn, dtype=np.float64).reshape(3),
                max_xyz=np.asarray(mx, dtype=np.float64).reshape(3))


@dataclass
class ManifestRow:
    order: int
    class_name: str
    place_result: str
    pick_xy_mm: np.ndarray | None
    place_xy_mm: np.ndarray | None
    pick_phi_deg: float
    place_phi_deg: float
    destination_surface_z_mm: float | None
    raw_box_center_xyz_mm: np.ndarray | None
    raw_box_size_xyz_mm: np.ndarray | None
    detection_bbox_xyxy: np.ndarray | None
    stereo_left: str | None
    points_cam: str | None


@dataclass
class PlacedItem:
    order: int
    class_name: str
    color: str
    raw_box: _Box        # at pick location
    placed_box: _Box     # in bag
    points_pick: np.ndarray    # Nx3 robot-frame at pick location
    points_placed: np.ndarray  # Nx3 robot-frame at placed location
    colors_rgb: np.ndarray | None


# ─── Colour palette ──────────────────────────────────────────────────────────

_PALETTE = [
    "#2e8bcb", "#e8882a", "#2dc96e", "#d83f4a",
    "#8854cc", "#e8c025", "#4abbc9", "#e06090",
]

def _item_color(order: int) -> str:
    return _PALETTE[(order - 1) % len(_PALETTE)]


# ─── Data loaders ────────────────────────────────────────────────────────────

def _resolve_run_dir() -> Path:
    if _RUN_DIR is not None:
        return _RUN_DIR.resolve()
    runs_root = (_REPO_ROOT / "data" / "run_snapshots").resolve()
    candidates = [p for p in runs_root.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_stereo_calib() -> dict[str, np.ndarray] | None:
    path = _REPO_ROOT / "stereo_calibration.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _load_bundle() -> dict[str, np.ndarray] | None:
    if not BUNDLE_PATH.exists():
        return None
    with np.load(BUNDLE_PATH, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _load_surface_zone() -> dict[str, Any]:
    zones = json.loads(SURFACE_ZONES_PATH.read_text(encoding="utf-8"))
    return zones[BAG_SCENE_NAME]


def _parse_manifest(run_dir: Path) -> list[ManifestRow]:
    path = run_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows_raw = data.get("objects") or []
    out: list[ManifestRow] = []
    for i, r in enumerate(rows_raw, start=1):
        def _v2(key: str) -> np.ndarray | None:
            v = r.get(key)
            try:
                a = np.asarray(v, dtype=np.float64).reshape(-1)
                return a[:2] if len(a) >= 2 and np.all(np.isfinite(a[:2])) else None
            except Exception:
                return None

        def _v3(key: str) -> np.ndarray | None:
            v = r.get(key)
            try:
                a = np.asarray(v, dtype=np.float64).reshape(-1)
                return a[:3] if len(a) >= 3 and np.all(np.isfinite(a[:3])) else None
            except Exception:
                return None

        def _v4(key: str) -> np.ndarray | None:
            v = r.get(key)
            try:
                a = np.asarray(v, dtype=np.float64).reshape(-1)
                return a[:4] if len(a) >= 4 and np.all(np.isfinite(a[:4])) else None
            except Exception:
                return None

        def _f(key: str, default: float = 0.0) -> float:
            v = r.get(key)
            try:
                return float(v) if v is not None else default
            except Exception:
                return default

        out.append(ManifestRow(
            order=i,
            class_name=str(r.get("class_name") or "object"),
            place_result=str(r.get("place_result") or ""),
            pick_xy_mm=_v2("pick_xy_mm"),
            place_xy_mm=_v2("place_xy_mm"),
            pick_phi_deg=_f("pick_phi_deg", 0.0),
            place_phi_deg=_f("place_phi_deg", 0.0),
            destination_surface_z_mm=_f("destination_surface_z_mm") if r.get("destination_surface_z_mm") is not None else None,
            raw_box_center_xyz_mm=_v3("raw_box_center_xyz_mm"),
            raw_box_size_xyz_mm=_v3("raw_box_size_xyz_mm"),
            detection_bbox_xyxy=_v4("detection_bbox_xyxy"),
            stereo_left=r.get("stereo_left"),
            points_cam=r.get("points_cam"),
        ))
    return out


# ─── Camera ↔ robot transforms ───────────────────────────────────────────────

def _project_cam_to_uv(points_cam: np.ndarray, stereo_calib: dict[str, np.ndarray]) -> np.ndarray | None:
    """Back-project bundle-convention camera points to rectified-left UV pixels."""
    if "projection_left_rectified" not in stereo_calib:
        return None
    pts = np.asarray(points_cam, dtype=np.float64).copy().reshape(-1, 3)
    pts[:, 2] *= -1.0  # undo Z sign-flip
    if "rectification_left" in stereo_calib:
        r = np.asarray(stereo_calib["rectification_left"], dtype=np.float64)
        pts = pts @ r.T
    P = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    z = pts[:, 2]
    good = z > 1.0
    z_safe = np.where(good, z, 1.0)
    u = np.where(good, fx * pts[:, 0] / z_safe + cx, np.nan)
    v = np.where(good, fy * pts[:, 1] / z_safe + cy, np.nan)
    return np.column_stack([u, v])


def _cam_to_robot(points_cam: np.ndarray, bundle: dict[str, np.ndarray]) -> np.ndarray:
    try:
        from vision.pointcloud import cam_points_to_robot_xyz
        out = np.asarray(cam_points_to_robot_xyz(points_cam, bundle), dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(out), axis=1)
        return out[finite]
    except Exception:
        A = np.asarray(bundle["A_robot_from_cam_xyz_3x4"], dtype=np.float64)
        n = len(points_cam)
        h = np.hstack([points_cam, np.ones((n, 1), dtype=np.float64)])
        out = (A @ h.T).T
        finite = np.all(np.isfinite(out), axis=1)
        return out[finite]


def _sample_colors(img_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    xs = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, h - 1)
    return np.clip(img_rgb[ys, xs, :3].astype(np.float32), 0.0, 1.0)


def _load_image_rgb(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _shift_points_to_box(
    pts: np.ndarray,
    target_box: _Box,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Translate a point cloud so its centroid-XY and min-Z align with target_box.
    Returns (filtered_pts, filtered_colors) where filtered_colors may be None.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    centroid_xy = p[:, :2].mean(axis=0)
    min_z = p[:, 2].min()
    shift = np.array([
        target_box.center[0] - centroid_xy[0],
        target_box.center[1] - centroid_xy[1],
        target_box.min_xyz[2] - min_z,
    ], dtype=np.float64)
    shifted = p + shift
    tol = 4.0
    inside = np.all(
        (shifted >= (target_box.min_xyz - tol)) & (shifted <= (target_box.max_xyz + tol)),
        axis=1,
    )
    c_out = None
    if colors is not None and len(colors) == len(p):
        c_out = colors[inside]
    return shifted[inside], c_out


def _subsample(pts: np.ndarray, colors: np.ndarray | None, n: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(pts) <= n:
        return pts, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx], (colors[idx] if colors is not None else None)


# ─── 3D drawing primitives ───────────────────────────────────────────────────

def _box_corners(box: _Box) -> np.ndarray:
    mn, mx = box.min_xyz, box.max_xyz
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)


_BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
_BOX_FACES = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]


def _draw_box(ax, box: _Box, color: str, *, alpha_edge: float = 0.9, alpha_face: float = 0.10, ls: str = "-") -> None:
    c = _box_corners(box)
    for a, b in _BOX_EDGES:
        ax.plot([c[a,0], c[b,0]], [c[a,1], c[b,1]], [c[a,2], c[b,2]],
                color=color, alpha=alpha_edge, linewidth=1.1, linestyle=ls)
    if alpha_face > 0:
        faces = [[c[i] for i in face] for face in _BOX_FACES]
        fc = Poly3DCollection(faces, alpha=alpha_face)
        fc.set_facecolor(color)
        fc.set_edgecolor("none")
        ax.add_collection3d(fc)


def _gripper_opening_for_box(box: _Box, yaw_deg: float, extra_mm: float) -> float:
    axis = 1 if abs(float(yaw_deg) % 180.0 - 90.0) <= 1e-3 else 0
    return max(22.0, float(box.size[axis]) + float(extra_mm))


def _gripper_center_z(box: _Box) -> float:
    grip_depth = min(float(box.size[2]) * 0.25, 18.0)
    return float(box.max_xyz[2] - grip_depth + 0.5 * FINGER_LENGTH_MM)


def _make_gripper_boxes(center_xyz: np.ndarray, yaw_deg: float, opening_mm: float) -> tuple[_Box, _Box, _Box]:
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    spread = max(18.0, float(opening_mm))
    is_90 = abs(float(yaw_deg) % 180.0 - 90.0) <= 1e-3
    if is_90:
        f_size = np.array([FINGER_DEPTH_MM, FINGER_WIDTH_MM, FINGER_LENGTH_MM])
        l_ctr = c + np.array([0.0, -0.5 * (spread + FINGER_WIDTH_MM), 0.0])
        r_ctr = c + np.array([0.0,  0.5 * (spread + FINGER_WIDTH_MM), 0.0])
        p_size = np.array([PALM_WIDTH_MM, spread + 2.0 * FINGER_WIDTH_MM + 16.0, PALM_HEIGHT_MM])
    else:
        f_size = np.array([FINGER_WIDTH_MM, FINGER_DEPTH_MM, FINGER_LENGTH_MM])
        l_ctr = c + np.array([-0.5 * (spread + FINGER_WIDTH_MM), 0.0, 0.0])
        r_ctr = c + np.array([ 0.5 * (spread + FINGER_WIDTH_MM), 0.0, 0.0])
        p_size = np.array([spread + 2.0 * FINGER_WIDTH_MM + 16.0, PALM_WIDTH_MM, PALM_HEIGHT_MM])
    p_ctr = c + np.array([0.0, 0.0, 0.5 * (FINGER_LENGTH_MM + PALM_HEIGHT_MM) - 6.0])
    return _make_box(l_ctr, f_size, "finger_l"), _make_box(r_ctr, f_size, "finger_r"), _make_box(p_ctr, p_size, "palm")


def _draw_gripper(ax, boxes: tuple[_Box, _Box, _Box], *, is_open: bool, color: str) -> None:
    for box in boxes:
        _draw_box(ax, box, color, alpha_edge=0.95, alpha_face=0.22 if is_open else 0.16)


def _draw_place_zone(ax, zone: dict[str, Any]) -> list[np.ndarray]:
    center = np.asarray(zone["center_xy_mm"], dtype=np.float64)
    hw = 0.5 * float(zone.get("width_mm", 290.0))
    hd = 0.5 * float(zone.get("depth_mm", 175.0))
    z0 = float(zone.get("surface_z_mm", 0.0))
    z1 = z0 + PLACE_ZONE_DISPLAY_HEIGHT_MM
    platform = _make_box_from_min_max(
        np.array([center[0] - hw, center[1] - hd, z0 - BAG_PLATFORM_THICKNESS_MM]),
        np.array([center[0] + hw, center[1] + hd, z0]),
        label="platform",
    )
    bag_outline = _make_box_from_min_max(
        np.array([center[0] - hw, center[1] - hd, z0]),
        np.array([center[0] + hw, center[1] + hd, z1]),
        label="bag",
    )
    _draw_box(ax, platform, PLATFORM_COLOR, alpha_edge=0.95, alpha_face=0.12)
    _draw_box(ax, bag_outline, BAG_COLOR, alpha_edge=0.7, alpha_face=0.03, ls="--")
    ax.text(float(center[0]), float(center[1]), z1 + 12.0,
            zone.get("name", "Bag"), color=BAG_COLOR, fontsize=8, ha="center")
    return [_box_corners(platform), _box_corners(bag_outline)]


def _draw_platform_bounds(ax, x_min: float, x_max: float, y_min: float, y_max: float) -> list[np.ndarray]:
    z = 0.0
    pts = np.array([[x_min,y_min,z],[x_max,y_min,z],[x_max,y_max,z],[x_min,y_max,z]], dtype=np.float64)
    for a, b in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]], [pts[a,2], pts[b,2]],
                color=BOUNDS_COLOR, linewidth=1.8, alpha=0.85)
    ax.text(0.5*(x_min+x_max), y_min - 20.0, z + 4.0, "Platform", color=BOUNDS_COLOR, fontsize=7, ha="center")
    return [pts]


def _style_ax(ax) -> None:
    ax.set_facecolor(PANEL_BG_COLOR)
    ax.tick_params(colors="#666", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222")


def _set_limits(ax, scale_pts: list[np.ndarray]) -> None:
    if X_LIM_MM is not None:
        ax.set_xlim(*X_LIM_MM)
    if Y_LIM_MM is not None:
        ax.set_ylim(*Y_LIM_MM)
    if Z_LIM_MM is not None:
        ax.set_zlim(*Z_LIM_MM)
    if X_LIM_MM is None or Y_LIM_MM is None or Z_LIM_MM is None:
        if not scale_pts:
            return
        all_pts = np.vstack(scale_pts)
        mn = all_pts.min(axis=0)
        mx = all_pts.max(axis=0)
        ctr = 0.5 * (mn + mx)
        span = max(float(np.max(mx - mn)) * 0.60, 200.0)
        if X_LIM_MM is None:
            ax.set_xlim(ctr[0] - span, ctr[0] + span)
        if Y_LIM_MM is None:
            ax.set_ylim(ctr[1] - span, ctr[1] + span)
        if Z_LIM_MM is None:
            ax.set_zlim(min(-40.0, float(mn[2]) - 20.0), ctr[2] + span)


# ─── Frame rendering ─────────────────────────────────────────────────────────

def _render_frame(
    ax,
    *,
    zone: dict[str, Any],
    platform_bounds: tuple[float, float, float, float],
    placed_items: list[PlacedItem],
    active_item: PlacedItem | None,
    moving_box: _Box | None,
    moving_pts: np.ndarray | None,
    moving_colors: np.ndarray | None,
    gripper_boxes: tuple[_Box, _Box, _Box] | None,
    gripper_open: bool,
    phase_label: str,
    step_label: str,
) -> None:
    ax.cla()
    _style_ax(ax)
    scale_pts: list[np.ndarray] = []

    # Platform + bag
    scale_pts.extend(_draw_platform_bounds(ax, *platform_bounds))
    scale_pts.extend(_draw_place_zone(ax, zone))

    # Previously placed items
    for item in placed_items:
        pts, cols = _subsample(item.points_placed, item.colors_rgb, MAX_PTS_PER_OBJECT, seed=100 + item.order)
        if cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=cols, s=POINT_SIZE, alpha=POINT_ALPHA, depthshade=False)
        _draw_box(ax, item.placed_box, item.color, alpha_face=0.12, alpha_edge=0.85)
        scale_pts.append(_box_corners(item.placed_box))

    # Active item at pick location (before it starts moving)
    if active_item is not None and moving_box is None:
        pts, cols = _subsample(active_item.points_pick, active_item.colors_rgb, MAX_PTS_PER_OBJECT, seed=200 + active_item.order)
        if cols is not None:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=cols, s=POINT_SIZE, alpha=0.55, depthshade=False)
        _draw_box(ax, active_item.raw_box, active_item.color, alpha_face=0.10, alpha_edge=0.80)
        scale_pts.append(_box_corners(active_item.raw_box))

    # Moving object
    if moving_box is not None:
        if moving_pts is not None and len(moving_pts) > 0:
            pts, cols = _subsample(moving_pts, moving_colors, MAX_PTS_PER_OBJECT, seed=300)
            if cols is not None:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           c=cols, s=POINT_SIZE * 1.3, alpha=0.75, depthshade=False)
        if active_item is not None:
            _draw_box(ax, moving_box, active_item.color, alpha_face=0.20, alpha_edge=0.98)
        scale_pts.append(_box_corners(moving_box))

    # Gripper
    if gripper_boxes is not None:
        g_color = GRIPPER_OPEN_COLOR if gripper_open else GRIPPER_HOLD_COLOR
        _draw_gripper(ax, gripper_boxes, is_open=gripper_open, color=g_color)
        for gb in gripper_boxes:
            scale_pts.append(_box_corners(gb))

    ax.view_init(elev=VIEW_ELEV_DEG, azim=VIEW_AZIM_DEG)
    _set_limits(ax, scale_pts)


def _fig_to_pil(fig: plt.Figure) -> "Image.Image":
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=GIF_DPI, facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


# ─── Animation phases ─────────────────────────────────────────────────────────

def _lerp_xyz(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)


def _object_box_at_pose(src_box: _Box, center_xy: np.ndarray, min_z: float) -> _Box:
    s = src_box.size
    cxy = np.asarray(center_xy, dtype=np.float64).reshape(2)
    return _make_box(
        np.array([cxy[0], cxy[1], min_z + 0.5 * s[2]]),
        s,
        label="moving",
    )


def _iter_ts(n: int):
    for i in range(n + 1):
        yield float(i) / max(n, 1)


def _build_frames_for_item(
    item: PlacedItem,
    pick_phi_deg: float,
    place_phi_deg: float,
    ax,
    fig: plt.Figure,
    zone: dict[str, Any],
    platform_bounds: tuple[float, float, float, float],
    placed_so_far: list[PlacedItem],
    survey_xyz: np.ndarray,
    n_steps: int,
) -> list["Image.Image"]:
    frames: list[Image.Image] = []

    src = item.raw_box
    dst = item.placed_box
    pick_xy = src.center[:2]
    place_xy = dst.center[:2]

    open_w  = _gripper_opening_for_box(src, pick_phi_deg, OPEN_EXTRA_MM)
    close_w = _gripper_opening_for_box(src, pick_phi_deg, CLOSE_EXTRA_MM)

    pick_grasp_z = _gripper_center_z(src)
    pick_hover_z = pick_grasp_z + HOVER_CLEARANCE_MM

    travel_z_obj = max(float(src.max_xyz[2]), float(dst.max_xyz[2])) + TRAVEL_CLEARANCE_MM
    travel_gripper_z = _gripper_center_z(_object_box_at_pose(src, pick_xy, travel_z_obj))

    place_grasp_z = _gripper_center_z(dst)
    place_hover_z = max(travel_gripper_z, place_grasp_z + HOVER_CLEARANCE_MM)

    step_label = f"Item {item.order}: {item.class_name}"

    def _pts_at_box(box: _Box) -> tuple[np.ndarray, np.ndarray | None]:
        if len(item.points_pick) == 0:
            return np.empty((0, 3)), None
        return _shift_points_to_box(item.points_pick, box, item.colors_rgb)

    def _emit(phase: str, moving_box: _Box | None, gripper: tuple[_Box,_Box,_Box] | None,
              is_open: bool, show_static: bool) -> None:
        if moving_box is not None and len(item.points_pick) > 0:
            m_pts, m_cols = _pts_at_box(moving_box)
        else:
            m_pts, m_cols = None, None
        _render_frame(
            ax,
            zone=zone,
            platform_bounds=platform_bounds,
            placed_items=placed_so_far,
            active_item=item if show_static else None,
            moving_box=moving_box,
            moving_pts=m_pts,
            moving_colors=m_cols,
            gripper_boxes=gripper,
            gripper_open=is_open,
            phase_label=phase,
            step_label=step_label,
        )
        frames.append(_fig_to_pil(fig))

    # Phase 1: survey → pick hover (gripper moves, object static at pick)
    for t in _iter_ts(n_steps):
        g_ctr = _lerp_xyz(
            np.array([survey_xyz[0], survey_xyz[1], survey_xyz[2]]),
            np.array([pick_xy[0], pick_xy[1], pick_hover_z]),
            t,
        )
        g = _make_gripper_boxes(g_ctr, pick_phi_deg, open_w)
        _emit(f"1/8 Survey → hover above {item.class_name}", None, g, True, True)

    # Phase 2: descend to pick
    for t in _iter_ts(n_steps):
        gz = (1 - t) * pick_hover_z + t * pick_grasp_z
        g = _make_gripper_boxes(np.array([pick_xy[0], pick_xy[1], gz]), pick_phi_deg, open_w)
        _emit("2/8 Descend to pick", None, g, True, True)

    # Phase 3: close gripper
    for t in _iter_ts(max(n_steps // 2, 4)):
        ow = (1 - t) * open_w + t * close_w
        g = _make_gripper_boxes(np.array([pick_xy[0], pick_xy[1], pick_grasp_z]), pick_phi_deg, ow)
        _emit("3/8 Close gripper", None, g, False, True)

    # Phase 4: lift object
    for t in _iter_ts(n_steps):
        obj_min_z = (1 - t) * float(src.min_xyz[2]) + t * travel_z_obj
        mbox = _object_box_at_pose(src, pick_xy, obj_min_z)
        gz = (1 - t) * pick_grasp_z + t * travel_gripper_z
        g = _make_gripper_boxes(np.array([pick_xy[0], pick_xy[1], gz]), pick_phi_deg, close_w)
        _emit("4/8 Lift object", mbox, g, False, False)

    # Phase 5: carry to place
    for t in _iter_ts(n_steps):
        cxy = _lerp_xyz(pick_xy, place_xy, t)
        mbox = _object_box_at_pose(src, cxy, travel_z_obj)
        gx = (1 - t) * pick_xy[0] + t * place_xy[0]
        gy = (1 - t) * pick_xy[1] + t * place_xy[1]
        g = _make_gripper_boxes(np.array([gx, gy, travel_gripper_z]), place_phi_deg, close_w)
        _emit("5/8 Carry to bag", mbox, g, False, False)

    # Phase 6: descend to place
    for t in _iter_ts(n_steps):
        obj_min_z = (1 - t) * travel_z_obj + t * float(dst.min_xyz[2])
        mbox = _object_box_at_pose(src, place_xy, obj_min_z)
        gz = (1 - t) * place_hover_z + t * place_grasp_z
        g = _make_gripper_boxes(np.array([place_xy[0], place_xy[1], gz]), place_phi_deg, close_w)
        _emit("6/8 Descend to place", mbox, g, False, False)

    # Phase 7: open / release
    for t in _iter_ts(max(n_steps // 2, 4)):
        ow = (1 - t) * close_w + t * open_w
        mbox = _object_box_at_pose(src, place_xy, float(dst.min_xyz[2]))
        g = _make_gripper_boxes(np.array([place_xy[0], place_xy[1], place_grasp_z]), place_phi_deg, ow)
        _emit("7/8 Release object", mbox, g, True, False)

    # Phase 8: retract above place
    for t in _iter_ts(n_steps):
        gz = (1 - t) * place_grasp_z + t * place_hover_z
        g = _make_gripper_boxes(np.array([place_xy[0], place_xy[1], gz]), place_phi_deg, open_w)
        _emit("8/8 Retract gripper", None, g, True, False)

    return frames


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a single placement animation GIF from a saved run snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, default=None, help="Run snapshot directory with manifest.json")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory root")
    p.add_argument("--workspace-profile", default="wet_run")
    p.add_argument("--limit", type=int, default=None, help="Max items to animate")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _apply_view_config()  # load view_config.json overrides if present

    if not _PIL_AVAILABLE:
        print("[ERROR] Pillow (PIL) is required — install it with: pip install Pillow")
        return 1

    args = parse_args(argv)
    run_dir = args.run_dir.resolve() if args.run_dir else _resolve_run_dir()
    out_root = args.out_dir.resolve() if args.out_dir else (run_output_dir(run_dir) / "animations")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] run_dir  = {run_dir}")
    print(f"[INFO] out_dir  = {out_root}")

    rows = _parse_manifest(run_dir)
    zone = _load_surface_zone()
    bundle = _load_bundle()
    stereo_calib = _load_stereo_calib()

    ws_cfg = get_workspace_filter_config(str(args.workspace_profile))
    x_min, x_max, y_min, y_max = workspace_bounds_mm(ws_cfg)
    platform_bounds = (x_min, x_max, y_min, y_max)

    # Try to load survey XYZ from config
    survey_xyz = np.array([SURVEY_X_MM, SURVEY_Y_MM, SURVEY_Z_MM], dtype=np.float64)
    try:
        from config.pick.pick_config import DEFAULT_PICK
        survey_xyz = np.array([
            float(DEFAULT_PICK.X_SURVEY_MM),
            float(DEFAULT_PICK.Y_SURVEY_MM),
            float(DEFAULT_PICK.Z_SURVEY_MM),
        ], dtype=np.float64)
    except Exception:
        pass

    # Build PlacedItem list from manifest
    placed_items: list[PlacedItem] = []
    skipped = 0
    for row in rows:
        if args.limit is not None and len(placed_items) >= args.limit:
            break

        if row.place_result.lower().strip() not in ("placed", "success", "ok"):
            print(f"[SKIP] row {row.order} ({row.class_name}): place_result='{row.place_result}'")
            skipped += 1
            continue

        if row.raw_box_center_xyz_mm is None or row.raw_box_size_xyz_mm is None:
            print(f"[SKIP] row {row.order}: missing raw box")
            skipped += 1
            continue

        if row.place_xy_mm is None or row.destination_surface_z_mm is None:
            print(f"[SKIP] row {row.order}: missing place_xy or destination_surface_z")
            skipped += 1
            continue

        # Build boxes
        raw_box = _make_box(
            np.asarray(row.raw_box_center_xyz_mm, dtype=np.float64),
            np.asarray(row.raw_box_size_xyz_mm, dtype=np.float64),
            label=f"raw_{row.class_name}",
        )
        placed_center = np.array([
            float(row.place_xy_mm[0]),
            float(row.place_xy_mm[1]),
            float(row.destination_surface_z_mm) + 0.5 * float(row.raw_box_size_xyz_mm[2]),
        ], dtype=np.float64)
        placed_box = _make_box(placed_center, np.asarray(row.raw_box_size_xyz_mm, dtype=np.float64),
                               label=f"placed_{row.class_name}")

        # Load point cloud
        pts_cam: np.ndarray | None = None
        colors_rgb: np.ndarray | None = None
        pts_pick: np.ndarray = np.empty((0, 3), dtype=np.float64)
        pts_placed: np.ndarray = np.empty((0, 3), dtype=np.float64)

        if row.points_cam and bundle is not None:
            pc_path = run_dir / row.points_cam
            if pc_path.exists():
                try:
                    with np.load(pc_path, allow_pickle=False) as d:
                        pts_cam = np.asarray(d["points_cam"], dtype=np.float64).reshape(-1, 3)
                    # Colors: sample from stereo image at UV coords
                    if stereo_calib is not None and row.stereo_left:
                        img = _load_image_rgb(run_dir / row.stereo_left)
                        if img is not None:
                            uv = _project_cam_to_uv(pts_cam, stereo_calib)
                            if uv is not None:
                                colors_rgb, color_valid = photo_colors_for_points(img, uv, len(pts_cam))
                                if colors_rgb is not None and color_valid is not None:
                                    pts_cam = pts_cam[color_valid]
                    # Robot-frame transform; keep color array aligned via finite mask
                    pts_robot = _cam_to_robot(pts_cam, bundle)
                    if len(pts_robot) > 0:
                        # Align colors_rgb with pts_robot via finite mask before shifting
                        colors_robot: np.ndarray | None = None
                        if colors_rgb is not None and len(colors_rgb) == len(pts_cam):
                            try:
                                A = np.asarray(bundle["A_robot_from_cam_xyz_3x4"], dtype=np.float64)
                                h = np.hstack([pts_cam, np.ones((len(pts_cam), 1), dtype=np.float64)])
                                finite = np.all(np.isfinite((A @ h.T).T), axis=1)
                                colors_robot = colors_rgb[finite]
                            except Exception:
                                pass
                        # _shift_points_to_box applies spatial filter and aligns colors
                        pts_pick, colors_pick = _shift_points_to_box(pts_robot, raw_box, colors_robot)
                        pts_placed, _ = _shift_points_to_box(pts_robot, placed_box)
                        colors_rgb = colors_pick
                except Exception as exc:
                    print(f"[WARN] row {row.order}: failed to load point cloud ({exc})")

        color = _item_color(row.order)
        placed_items.append(PlacedItem(
            order=row.order,
            class_name=row.class_name,
            color=color,
            raw_box=raw_box,
            placed_box=placed_box,
            points_pick=pts_pick,
            points_placed=pts_placed,
            colors_rgb=colors_rgb,
        ))
        print(f"[LOAD] row {row.order} ({row.class_name}): pts_pick={len(pts_pick)} pts_placed={len(pts_placed)}")

    if not placed_items:
        print("[ERROR] No placed items found in manifest.")
        return 1

    print(f"[INFO] Building GIF for {len(placed_items)} items ({skipped} skipped)...")

    # Compute pick_phi / place_phi per item from manifest rows indexed by order
    phi_map: dict[int, tuple[float, float]] = {}
    for row in rows:
        phi_map[row.order] = (float(row.pick_phi_deg), float(row.place_phi_deg))

    # Build animation frames
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor=BG_COLOR)
    ax = fig.add_subplot(111, projection="3d")

    all_frames: list[Image.Image] = []
    current_placed: list[PlacedItem] = []

    for item in placed_items:
        pick_phi, place_phi = phi_map.get(item.order, (0.0, 0.0))
        print(f"  Animating {item.class_name} (order={item.order}, phi={pick_phi:.0f}->{place_phi:.0f})...")

        # Seam frame: static plan view matching output-4's right panel —
        # shows the item at its planned placement location before the pick animation begins.
        _render_frame(
            ax,
            zone=zone,
            platform_bounds=platform_bounds,
            placed_items=list(current_placed),
            active_item=item,
            moving_box=None,
            moving_pts=None,
            moving_colors=None,
            gripper_boxes=None,
            gripper_open=True,
            phase_label="Plan view",
            step_label=f"Item {item.order}: {item.class_name}",
        )
        seam_img = _fig_to_pil(fig)
        SEAM_HOLD_FRAMES = max(1, int(800 / max(GIF_FRAME_MS, 1)))
        all_frames.extend([seam_img] * SEAM_HOLD_FRAMES)

        frames = _build_frames_for_item(
            item=item,
            pick_phi_deg=pick_phi,
            place_phi_deg=place_phi,
            ax=ax,
            fig=fig,
            zone=zone,
            platform_bounds=platform_bounds,
            placed_so_far=list(current_placed),
            survey_xyz=survey_xyz,
            n_steps=N_ANIM_STEPS,
        )
        all_frames.extend(frames)
        current_placed.append(item)
        print(f"  -> {len(frames)} frames (+{SEAM_HOLD_FRAMES} seam)")

    plt.close(fig)

    # Final frame: all items placed, no gripper
    fig2 = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor=BG_COLOR)
    ax2 = fig2.add_subplot(111, projection="3d")
    _render_frame(
        ax2,
        zone=zone,
        platform_bounds=platform_bounds,
        placed_items=current_placed,
        active_item=None,
        moving_box=None,
        moving_pts=None,
        moving_colors=None,
        gripper_boxes=None,
        gripper_open=True,
        phase_label="Complete",
        step_label=f"All {len(current_placed)} items placed",
    )
    final_img = _fig_to_pil(fig2)
    plt.close(fig2)
    # Hold the final frame for 2 seconds
    all_frames.extend([final_img] * max(1, int(2000 / max(GIF_FRAME_MS, 1))))

    # Save GIF
    out_path = out_root / "placement_pick_to_bag_animation.gif"
    durations = [GIF_FRAME_MS] * (len(all_frames) - int(2000 / max(GIF_FRAME_MS, 1))) + \
                [2000] * int(2000 / max(GIF_FRAME_MS, 1))
    if len(durations) != len(all_frames):
        durations = [GIF_FRAME_MS] * len(all_frames)
    all_frames[0].save(
        out_path,
        save_all=True,
        append_images=all_frames[1:],
        optimize=False,
        duration=durations,
        loop=0,
    )
    print(f"[SAVE] GIF ({len(all_frames)} frames) → {out_path}")
    print(f"[SUMMARY] items={len(placed_items)} skipped={skipped} frames={len(all_frames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
