"""Generate two custom 3D point cloud plots for the Pringles item from run_20260531_165144.

Plot 1: Robot-frame point cloud only, dark background, no axes/ticks/labels.
Plot 2: Robot-frame point cloud + AABB padded box.

RGB colors come from the rectified stereo-left image sampled at each point's projected UV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.place.place_config import DEFAULT_PLACE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data" / "run_snapshots" / "run_20260531_165144"
OUT_DIR = REPO_ROOT / "presentation_outputs" / "pringles_custom_plots"

# Z padding used by pad_aabb for placement occupancy (+PAD_Z on each side)
PAD_Z_MM: float = float(DEFAULT_PLACE.PAD_Z_MM)

POINTS_NPZ = RUN_DIR / "points_cam_0001.npz"
STEREO_LEFT = RUN_DIR / "Stereo_Left_0001.png"
STEREO_CALIB = REPO_ROOT / "stereo_calibration.npz"
ROBOT_BUNDLE = REPO_ROOT / "robot_calibration_bundle.npz"

# From manifest: Pringles raw box size for the overlay box computation
RAW_BOX_SIZE_XYZ_MM = np.array([65.49901913663473, 50.831631349222775, 99.04494747492441])

MAX_POINTS = 8000
RNG_SEED = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_points_cam() -> np.ndarray:
    with np.load(str(POINTS_NPZ), allow_pickle=False) as d:
        for key in ("points_cam", "points_camera", "cam_points", "points_xyz_cam"):
            if key in d.files:
                arr = np.asarray(d[key], dtype=np.float64).reshape(-1, 3)
                finite = np.all(np.isfinite(arr), axis=1)
                return arr[finite]
        for key in d.files:
            arr = np.asarray(d[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                arr = arr[:, :3].astype(np.float64)
                finite = np.all(np.isfinite(arr), axis=1)
                return arr[finite]
    raise ValueError(f"No Nx3 array found in {POINTS_NPZ}")


def load_stereo_calib() -> dict[str, np.ndarray]:
    with np.load(str(STEREO_CALIB), allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_robot_bundle() -> dict[str, np.ndarray]:
    with np.load(str(ROBOT_BUNDLE), allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_image_rgb() -> np.ndarray:
    img_bgr = cv2.imread(str(STEREO_LEFT))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load {STEREO_LEFT}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ── Coordinate transforms (mirrors run_snapshot_manifest_survey_story.py) ────

def project_cam_bundle_to_uv(points_cam: np.ndarray, calib: dict) -> np.ndarray:
    """Bundle-convention camera-frame points → rectified-left UV pixel coords."""
    pts = points_cam.copy().reshape(-1, 3)
    pts[:, 2] *= -1.0  # undo Z sign-flip
    if "rectification_left" in calib:
        r_left = np.asarray(calib["rectification_left"], dtype=np.float64)
        pts = pts @ r_left.T
    P = np.asarray(calib["projection_left_rectified"], dtype=np.float64)
    fx, fy = float(P[0, 0]), float(P[1, 1])
    cx, cy = float(P[0, 2]), float(P[1, 2])
    z = pts[:, 2]
    good = z > 1.0
    z_safe = np.where(good, z, 1.0)
    u = np.where(good, fx * pts[:, 0] / z_safe + cx, np.nan)
    v = np.where(good, fy * pts[:, 1] / z_safe + cy, np.nan)
    return np.column_stack([u, v])


def cam_to_robot(points_cam: np.ndarray, bundle: dict) -> np.ndarray:
    """Nx3 camera-frame → Nx3 robot-frame using A_robot_from_cam_xyz_3x4."""
    a = bundle["A_robot_from_cam_xyz_3x4"]
    pts_h = np.column_stack([points_cam, np.ones(len(points_cam))])
    return pts_h @ np.asarray(a, dtype=np.float64).T


def sample_colors(image_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    xs = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, h - 1)
    return np.clip(image_rgb[ys, xs, :3].astype(np.float32), 0.0, 1.0)


# ── AABB box (mirrors _derive_pointcloud_overlay_box) ────────────────────────

def derive_padded_box(points_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = points_robot[np.all(np.isfinite(points_robot), axis=1)]
    lo = np.percentile(pts, 2.0, axis=0)
    hi = np.percentile(pts, 98.0, axis=0)
    size = hi - lo
    size = np.maximum(size, RAW_BOX_SIZE_XYZ_MM)
    size = np.maximum(size, np.array([8.0, 8.0, 8.0]))
    size = size + np.array([4.0, 4.0, 4.0])
    center = 0.5 * (lo + hi)
    return center, size


# ── Plotting helpers ──────────────────────────────────────────────────────────

def draw_wireframe_box(ax, center: np.ndarray, size: np.ndarray, color: str = "#facc15") -> None:
    mn = center - 0.5 * size
    mx = center + 0.5 * size
    c = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot([c[a,0], c[b,0]], [c[a,1], c[b,1]], [c[a,2], c[b,2]],
                color=color, linewidth=1.5, alpha=0.95)
    faces = Poly3DCollection(
        [[c[i] for i in face] for face in (
            [0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]
        )],
        alpha=0.08,
    )
    faces.set_facecolor(color)
    faces.set_edgecolor("none")
    ax.add_collection3d(faces)


def strip_axes(ax) -> None:
    """Remove all axes decorations — ticks, labels, panes, grid."""
    ax.set_axis_off()
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")


def make_dark_ax(fig) -> Axes3D:
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f172a")
    return ax


def set_equal_aspect(ax, pts: np.ndarray) -> None:
    """Force equal aspect ratio by computing a cubic bounding box."""
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    mid = 0.5 * (lo + hi)
    half = 0.55 * (hi - lo).max()
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


def subsample(pts: np.ndarray, colors: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(pts) <= n:
        return pts, colors
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx], colors[idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    points_cam = load_points_cam()
    calib = load_stereo_calib()
    bundle = load_robot_bundle()
    image_rgb = load_image_rgb()

    print(f"  {len(points_cam)} camera-frame points")

    uv = project_cam_bundle_to_uv(points_cam, calib)
    colors = sample_colors(image_rgb, uv)

    points_robot = cam_to_robot(points_cam, bundle)
    finite = np.all(np.isfinite(points_robot), axis=1)
    points_robot = points_robot[finite]
    colors = colors[finite]
    print(f"  {len(points_robot)} robot-frame points after finite filter")

    pts_sub, cols_sub = subsample(points_robot, colors, MAX_POINTS)

    box_center, box_size = derive_padded_box(points_robot)
    # Z-padded placement occupancy box: same center, +PAD_Z on each Z side
    packing_size = box_size.copy()
    packing_size[2] += 2.0 * PAD_Z_MM

    print(f"  AABB center:        {box_center.round(1)} mm")
    print(f"  AABB size:          {box_size.round(1)} mm")
    print(f"  Packing box size:   {packing_size.round(1)} mm  (PAD_Z={PAD_Z_MM} mm each side)")

    # ── Plot 1: point cloud only, no axes ────────────────────────────────────
    print("Rendering plot 1 (cloud only)...")
    fig1 = plt.figure(figsize=(8, 8), facecolor="#080b14")
    ax1 = make_dark_ax(fig1)
    ax1.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                c=cols_sub, s=1.5, alpha=0.85, depthshade=False)
    set_equal_aspect(ax1, pts_sub)
    strip_axes(ax1)
    fig1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out1 = OUT_DIR / "pringles_cloud_only.png"
    fig1.savefig(str(out1), dpi=200, facecolor=fig1.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig1)
    print(f"  Saved: {out1}")

    # ── Plot 2: point cloud + AABB padded box ────────────────────────────────
    print("Rendering plot 2 (cloud + AABB box)...")
    fig2 = plt.figure(figsize=(8, 8), facecolor="#080b14")
    ax2 = make_dark_ax(fig2)
    ax2.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                c=cols_sub, s=1.5, alpha=0.85, depthshade=False)
    draw_wireframe_box(ax2, box_center, box_size, color="#facc15")
    # Expand view to encompass box
    box_corners = np.array([box_center - 0.5 * box_size, box_center + 0.5 * box_size])
    view_pts = np.vstack([pts_sub, box_corners])
    set_equal_aspect(ax2, view_pts)
    strip_axes(ax2)
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out2 = OUT_DIR / "pringles_cloud_with_aabb.png"
    fig2.savefig(str(out2), dpi=200, facecolor=fig2.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ── Plot 3: cloud + AABB + Z-padded packing occupancy box ────────────────
    print("Rendering plot 3 (cloud + AABB + Z-padded packing box)...")
    fig3 = plt.figure(figsize=(8, 8), facecolor="#080b14")
    ax3 = make_dark_ax(fig3)
    ax3.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2],
                c=cols_sub, s=1.5, alpha=0.85, depthshade=False)
    draw_wireframe_box(ax3, box_center, box_size, color="#facc15")         # survey AABB
    draw_wireframe_box(ax3, box_center, packing_size, color="#f87171")     # Z-padded packing box
    pack_corners = np.array([box_center - 0.5 * packing_size, box_center + 0.5 * packing_size])
    set_equal_aspect(ax3, np.vstack([pts_sub, pack_corners]))
    strip_axes(ax3)
    fig3.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out3 = OUT_DIR / "pringles_cloud_with_packing_box.png"
    fig3.savefig(str(out3), dpi=200, facecolor=fig3.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig3)
    print(f"  Saved: {out3}")

    print("Done.")


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
