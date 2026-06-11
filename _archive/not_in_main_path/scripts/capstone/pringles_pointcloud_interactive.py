"""Interactive point cloud viewer — rotate to desired angle, press Enter to save.

Saves both plots (cloud only + AABB box) at the chosen azimuth/elevation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.place.place_config import DEFAULT_PLACE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "data" / "run_snapshots" / "run_20260531_165144"
OUT_DIR = REPO_ROOT / "presentation_outputs" / "pringles_custom_plots"

PAD_Z_MM: float = float(DEFAULT_PLACE.PAD_Z_MM)

POINTS_NPZ = RUN_DIR / "points_cam_0001.npz"
STEREO_LEFT = RUN_DIR / "Stereo_Left_0001.png"
STEREO_CALIB = REPO_ROOT / "stereo_calibration.npz"
ROBOT_BUNDLE = REPO_ROOT / "robot_calibration_bundle.npz"

RAW_BOX_SIZE_XYZ_MM = np.array([65.49901913663473, 50.831631349222775, 99.04494747492441])

PREVIEW_POINTS = 5000
SAVE_POINTS = 8000
RNG_SEED = 42
SAVE_DPI = 300


# ── Data loading (same as pringles_pointcloud_plots.py) ──────────────────────

def load_points_cam() -> np.ndarray:
    with np.load(str(POINTS_NPZ), allow_pickle=False) as d:
        for key in ("points_cam", "points_camera", "cam_points", "points_xyz_cam"):
            if key in d.files:
                arr = np.asarray(d[key], dtype=np.float64).reshape(-1, 3)
                return arr[np.all(np.isfinite(arr), axis=1)]
        for key in d.files:
            arr = np.asarray(d[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                arr = arr[:, :3].astype(np.float64)
                return arr[np.all(np.isfinite(arr), axis=1)]
    raise ValueError(f"No Nx3 array found in {POINTS_NPZ}")


def load_stereo_calib() -> dict:
    with np.load(str(STEREO_CALIB), allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_robot_bundle() -> dict:
    with np.load(str(ROBOT_BUNDLE), allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_image_rgb() -> np.ndarray:
    img = cv2.imread(str(STEREO_LEFT))
    if img is None:
        raise FileNotFoundError(str(STEREO_LEFT))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def project_cam_bundle_to_uv(pts: np.ndarray, calib: dict) -> np.ndarray:
    pts = pts.copy()
    pts[:, 2] *= -1.0
    if "rectification_left" in calib:
        pts = pts @ np.asarray(calib["rectification_left"], dtype=np.float64).T
    P = np.asarray(calib["projection_left_rectified"], dtype=np.float64)
    fx, fy, cx, cy = P[0,0], P[1,1], P[0,2], P[1,2]
    z = pts[:, 2]
    zs = np.where(z > 1.0, z, 1.0)
    u = np.where(z > 1.0, fx * pts[:,0] / zs + cx, np.nan)
    v = np.where(z > 1.0, fy * pts[:,1] / zs + cy, np.nan)
    return np.column_stack([u, v])


def cam_to_robot(pts: np.ndarray, bundle: dict) -> np.ndarray:
    A = np.asarray(bundle["A_robot_from_cam_xyz_3x4"], dtype=np.float64)
    return np.column_stack([pts, np.ones(len(pts))]) @ A.T


def sample_colors(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.clip(np.rint(uv[:,0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(uv[:,1]).astype(np.int32), 0, h - 1)
    return np.clip(img[ys, xs, :3].astype(np.float32), 0.0, 1.0)


def derive_padded_box(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = pts[np.all(np.isfinite(pts), axis=1)]
    lo, hi = np.percentile(p, 2.0, axis=0), np.percentile(p, 98.0, axis=0)
    size = np.maximum(hi - lo, RAW_BOX_SIZE_XYZ_MM)
    size = np.maximum(size, 8.0) + 4.0
    return 0.5 * (lo + hi), size


def subsample(pts, cols, n):
    if len(pts) <= n:
        return pts, cols
    idx = np.random.default_rng(RNG_SEED).choice(len(pts), size=n, replace=False)
    return pts[idx], cols[idx]


def draw_wireframe_box(ax, center, size, color="#facc15"):
    mn, mx = center - 0.5 * size, center + 0.5 * size
    c = np.array([[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],[mx[0],mx[1],mn[2]],[mn[0],mx[1],mn[2]],
                  [mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],[mx[0],mx[1],mx[2]],[mn[0],mx[1],mx[2]]])
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot([c[a,0],c[b,0]], [c[a,1],c[b,1]], [c[a,2],c[b,2]], color=color, lw=1.5, alpha=0.95)
    faces = Poly3DCollection([[c[i] for i in f] for f in
        ([0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5])], alpha=0.08)
    faces.set_facecolor(color)
    faces.set_edgecolor("none")
    ax.add_collection3d(faces)


def strip_axes(ax):
    ax.set_axis_off()
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")


def set_equal_aspect(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = 0.5 * (lo + hi)
    half = 0.55 * (hi - lo).max()
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)


def save_plots(pts_robot, colors, box_center, box_size, azim, elev):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pts_s, cols_s = subsample(pts_robot, colors, SAVE_POINTS)
    box_corners = np.array([box_center - 0.5*box_size, box_center + 0.5*box_size])

    packing_size = box_size.copy()
    packing_size[2] += 2.0 * PAD_Z_MM
    pack_corners = np.array([box_center - 0.5*packing_size, box_center + 0.5*packing_size])

    renders = [
        ("cloud_only",           False, False),
        ("cloud_with_aabb",      True,  False),
        ("cloud_with_packing_box", True, True),
    ]
    for suffix, draw_aabb, draw_packing in renders:
        fig = plt.figure(figsize=(10, 10), facecolor="#080b14")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0f172a")
        ax.scatter(pts_s[:,0], pts_s[:,1], pts_s[:,2],
                   c=cols_s, s=2.0, alpha=0.85, depthshade=False)
        if draw_aabb:
            draw_wireframe_box(ax, box_center, box_size, color="#facc15")
        if draw_packing:
            draw_wireframe_box(ax, box_center, packing_size, color="#f87171")
        if draw_packing:
            set_equal_aspect(ax, np.vstack([pts_s, pack_corners]))
        elif draw_aabb:
            set_equal_aspect(ax, np.vstack([pts_s, box_corners]))
        else:
            set_equal_aspect(ax, pts_s)
        ax.view_init(elev=elev, azim=azim)
        strip_axes(ax)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        out = OUT_DIR / f"pringles_{suffix}_az{azim:.0f}_el{elev:.0f}.png"
        fig.savefig(str(out), dpi=SAVE_DPI, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    pts_cam = load_points_cam()
    calib = load_stereo_calib()
    bundle = load_robot_bundle()
    img_rgb = load_image_rgb()

    uv = project_cam_bundle_to_uv(pts_cam, calib)
    colors_cam = sample_colors(img_rgb, uv)
    pts_robot = cam_to_robot(pts_cam, bundle)

    finite = np.all(np.isfinite(pts_robot), axis=1)
    pts_robot = pts_robot[finite]
    colors = colors_cam[finite]

    box_center, box_size = derive_padded_box(pts_robot)
    packing_size = box_size.copy()
    packing_size[2] += 2.0 * PAD_Z_MM
    pts_prev, cols_prev = subsample(pts_robot, colors, PREVIEW_POINTS)

    print(f"  {len(pts_robot)} points | AABB {box_size.round(1)} mm | packing Z pad ±{PAD_Z_MM} mm")
    print("\nRotate the view, then press  Enter  to save high-res images.")
    print("Close the window to exit without saving.\n")

    fig = plt.figure(figsize=(8, 8), facecolor="#080b14")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f172a")
    ax.scatter(pts_prev[:,0], pts_prev[:,1], pts_prev[:,2],
               c=cols_prev, s=1.5, alpha=0.85, depthshade=False)
    draw_wireframe_box(ax, box_center, box_size, color="#facc15")       # survey AABB
    draw_wireframe_box(ax, box_center, packing_size, color="#f87171")   # Z-padded packing box
    pack_corners = np.array([box_center - 0.5*packing_size, box_center + 0.5*packing_size])
    set_equal_aspect(ax, np.vstack([pts_prev, pack_corners]))
    strip_axes(ax)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.text(0.5, 0.02, "Rotate to desired angle, then press  Enter  to save",
             ha="center", color="#94a3b8", fontsize=9)

    def on_key(event):
        if event.key == "enter":
            azim = ax.azim
            elev = ax.elev
            print(f"\nCapturing: azim={azim:.1f}°  elev={elev:.1f}°")
            save_plots(pts_robot, colors, box_center, box_size, azim, elev)
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
