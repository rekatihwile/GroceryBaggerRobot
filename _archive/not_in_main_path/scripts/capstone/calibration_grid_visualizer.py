from __future__ import annotations

"""scripts/capstone/calibration_grid_visualizer.py

Visualize the stereo-camera calibration bundle in two complementary views:

  Figure 1 — Camera-Space Calibration Grid
    3D scatter of the known calibration support points as the stereo camera
    sees them (camera-frame XYZ).  Each point is annotated with the true
    robot-frame coordinate in parentheses so you can see the distortion /
    error between "what the image sees" and "what the robot knows".

  Figure 2 — Camera Pose in Robot Frame
    The calibration grid shown in robot-frame coordinates, plus the estimated
    camera position and optical axis drawn as arrows so you can see where the
    camera is physically mounted and what direction it is looking.

All computation is done post-hoc from the saved calibration bundle and stereo
calibration NPZ.  No live hardware, YOLO, or RAFT is used.

Run:
    cd C:\\Users\\elipp\\OneDrive\\Documents\\Grocery_Buildup
    python scripts/capstone/calibration_grid_visualizer.py
    python scripts/capstone/calibration_grid_visualizer.py --save
"""

from pathlib import Path
import argparse
import csv
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("TkAgg" if sys.stdout.isatty() else "Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for projection='3d'
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    PUBLICATION_GLOBAL_ROOT,
    save_figure_bundle,
)

# ── VS Code IDE defaults ────────────────────────────────────────────────────
BUNDLE_PATH = _REPO_ROOT / "robot_calibration_bundle.npz"
STEREO_CALIB_PATH = _REPO_ROOT / "stereo_calibration.npz"

# Annotation font size for per-point robot-coordinate labels
ANNOT_FONTSIZE = 6
# Arrow scale for camera-axis visualization (mm)
AXIS_ARROW_SCALE_MM = 80.0
# Points to subsample from the support set for clarity (None = all)
MAX_LABEL_POINTS: int | None = None

_FIG_BG = "#080b14"
_AX_BG = "#0f172a"
_FG = "#f8fafc"
_MUTED = "#94a3b8"
_GRID_COLOR = "#1e293b"
CLEAN_EXPORTS = True


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_bundle() -> dict[str, np.ndarray]:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"robot_calibration_bundle.npz not found: {BUNDLE_PATH}")
    with np.load(BUNDLE_PATH, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _load_stereo_calib() -> dict[str, np.ndarray] | None:
    if not STEREO_CALIB_PATH.exists():
        print(f"[CALIB WARN] stereo_calibration.npz not found at {STEREO_CALIB_PATH}")
        return None
    with np.load(STEREO_CALIB_PATH, allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def _project_robot_to_cam(robot_xyz_mm: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Project robot-frame XYZ points to camera-frame XYZ using B_cam_from_robot_xyz_3x4.

    cam_xyz = B[:, :3] @ robot_xyz + B[:, 3]
    """
    pts = np.asarray(robot_xyz_mm, dtype=np.float64).reshape(-1, 3)
    R = B[:, :3]
    t = B[:, 3]
    return (pts @ R.T) + t.reshape(1, 3)


def _camera_position_in_robot_frame(B: np.ndarray) -> np.ndarray:
    """Recover the camera's physical origin in robot-frame coordinates.

    Solves B[:,:3] @ cam_pos_robot + B[:,3] = [0,0,0] for cam_pos_robot.
    Since B[:,:3] is orthonormal: cam_pos_robot = -R^T @ t.
    """
    R = B[:, :3]
    t = B[:, 3]
    return -(R.T @ t)


def _camera_view_direction_in_robot_frame(A: np.ndarray, *, sign_flip_z: bool = True) -> np.ndarray:
    """Return the unit vector in robot frame that the camera looks toward.

    A = A_robot_from_cam_xyz_3x4 converts camera-frame deltas to robot-frame deltas.
    Camera optical axis is camera +Z.  The bundle convention sign-flips Z, so the
    physical viewing direction is -cam_Z in bundle convention = -A[:,2].
    """
    cam_z_in_robot = A[:, :3] @ np.array([0.0, 0.0, 1.0])
    if sign_flip_z:
        cam_z_in_robot = -cam_z_in_robot
    norm = np.linalg.norm(cam_z_in_robot)
    return cam_z_in_robot / max(norm, 1e-12)


def _camera_axes_in_robot_frame(A: np.ndarray, *, sign_flip_z: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_axis, y_axis, z_axis) camera axes in robot frame.

    z_axis is the physical viewing direction (with bundle Z sign-flip applied).
    """
    def _unit(v):
        n = np.linalg.norm(v)
        return v / max(n, 1e-12)

    R = A[:, :3]
    x_cam_robot = _unit(R @ np.array([1.0, 0.0, 0.0]))
    y_cam_robot = _unit(R @ np.array([0.0, 1.0, 0.0]))
    z_cam_robot = _unit(R @ np.array([0.0, 0.0, 1.0]))
    if sign_flip_z:
        z_cam_robot = -z_cam_robot
    return x_cam_robot, y_cam_robot, z_cam_robot


def _style_3d(ax, *, title: str = "") -> None:
    ax.set_facecolor(_AX_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(_AX_BG)
        pane.set_edgecolor(_GRID_COLOR)
        pane.set_alpha(1.0)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.65, alpha=0.7)
    ax.tick_params(colors=_MUTED, labelsize=7)
    if CLEAN_EXPORTS:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if title and not CLEAN_EXPORTS:
        ax.set_title(title, color=_FG, fontsize=10, pad=8)


def _draw_arrow(ax, origin: np.ndarray, direction: np.ndarray, length: float, color: str, label: str = "") -> None:
    end = origin + direction * length
    ax.quiver(
        float(origin[0]), float(origin[1]), float(origin[2]),
        float(direction[0] * length), float(direction[1] * length), float(direction[2] * length),
        color=color, linewidth=2.0, arrow_length_ratio=0.18,
    )
    if label:
        tip = origin + direction * length * 1.15
        ax.text(float(tip[0]), float(tip[1]), float(tip[2]), label,
                color=color, fontsize=8, ha="center", va="center")


def _draw_frustum(ax, cam_pos: np.ndarray, view_dir: np.ndarray, right_dir: np.ndarray, up_dir: np.ndarray,
                  *, focal_dist: float = 220.0, half_fov_h: float = 30.0, half_fov_v: float = 22.0) -> None:
    """Draw a simple camera frustum wireframe."""
    import math
    focal = view_dir * focal_dist
    rw = math.tan(math.radians(half_fov_h)) * focal_dist
    rh = math.tan(math.radians(half_fov_v)) * focal_dist
    tl = cam_pos + focal + right_dir * (-rw) + up_dir * rh
    tr = cam_pos + focal + right_dir * rw + up_dir * rh
    br = cam_pos + focal + right_dir * rw + up_dir * (-rh)
    bl = cam_pos + focal + right_dir * (-rw) + up_dir * (-rh)
    corners = [tl, tr, br, bl]
    for c in corners:
        ax.plot([cam_pos[0], c[0]], [cam_pos[1], c[1]], [cam_pos[2], c[2]],
                color="#64748b", linewidth=0.9, alpha=0.7)
    for a, b in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="#64748b", linewidth=0.9, alpha=0.7)


def _set_equal_axes(ax, pts: np.ndarray, margin_frac: float = 0.15) -> None:
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


# ── Figure 1: Camera-Space Grid ─────────────────────────────────────────────

def make_camera_space_figure(
    robot_xyz_mm: np.ndarray,
    cam_xyz_mm: np.ndarray,
    *,
    max_labels: int | None = None,
) -> plt.Figure:
    """3D scatter of calibration grid in camera frame, labeled with robot coords."""
    fig = plt.figure(figsize=(13.0, 8.0), facecolor=_FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax, title="Calibration Grid — Camera Frame\n(each point labeled with robot-frame coords)")

    n = len(cam_xyz_mm)
    ax.scatter(
        cam_xyz_mm[:, 0], cam_xyz_mm[:, 1], cam_xyz_mm[:, 2],
        c="#38bdf8", s=28, alpha=0.85, depthshade=False, zorder=5,
    )

    label_indices = (
        np.array([], dtype=int)
        if CLEAN_EXPORTS
        else np.arange(n) if (max_labels is None or n <= max_labels)
        else np.linspace(0, n - 1, max_labels, dtype=int)
    )

    for idx in label_indices:
        cx, cy, cz = cam_xyz_mm[idx]
        rx, ry, rz = robot_xyz_mm[idx]
        # Top line: what the stereo frame shows (camera XYZ)
        stereo_label = f"stereo ({cx:.0f}, {cy:.0f}, {cz:.0f})"
        robot_label = f"({rx:.0f}, {ry:.0f}, {rz:.0f})"
        ax.text(cx, cy, cz + 8,
                stereo_label,
                color="#f8fafc", fontsize=ANNOT_FONTSIZE, ha="center", va="bottom",
                zorder=10)
        ax.text(cx, cy, cz - 5,
                robot_label,
                color="#facc15", fontsize=ANNOT_FONTSIZE, ha="center", va="top",
                style="italic", zorder=10)

    if not CLEAN_EXPORTS:
        ax.set_xlabel("X_cam  (mm)", color=_MUTED, fontsize=8, labelpad=6)
        ax.set_ylabel("Y_cam  (mm)", color=_MUTED, fontsize=8, labelpad=6)
        ax.set_zlabel("Z_cam  (mm)", color=_MUTED, fontsize=8, labelpad=6)
        ax.text2D(0.02, 0.96,
                  "White: stereo/camera-frame XYZ\nYellow (in parens): robot-frame XYZ  (mm)",
                  transform=ax.transAxes, color=_FG, fontsize=8, va="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#111827", edgecolor="#334155", alpha=0.88))

    _set_equal_axes(ax, cam_xyz_mm, margin_frac=0.20)
    ax.view_init(elev=18, azim=-52)
    fig.tight_layout()
    return fig


# ── Figure 2: Camera Pose in Robot Frame ────────────────────────────────────

def make_camera_pose_figure(
    robot_xyz_mm: np.ndarray,
    cam_pos_robot: np.ndarray,
    view_dir_robot: np.ndarray,
    cam_x_robot: np.ndarray,
    cam_y_robot: np.ndarray,
    bundle: dict[str, np.ndarray],
) -> plt.Figure:
    """Robot-frame grid + camera position, axes, and viewing frustum."""
    fig = plt.figure(figsize=(13.0, 8.0), facecolor=_FIG_BG)
    ax = fig.add_subplot(111, projection="3d")
    _style_3d(ax, title="Camera Pose — Robot Frame\n(camera position, axes, and viewing direction)")

    # Support grid points
    ax.scatter(
        robot_xyz_mm[:, 0], robot_xyz_mm[:, 1], robot_xyz_mm[:, 2],
        c="#38bdf8", s=20, alpha=0.75, depthshade=False, label="Calib grid (robot frame)",
    )

    # Platform bounding box sketch
    x_min, x_max = float(bundle["x_grid_mm"].min()), float(bundle["x_grid_mm"].max())
    y_min, y_max = float(bundle["y_grid_mm"].min()), float(bundle["y_grid_mm"].max())
    z_levels = np.asarray(bundle["z_levels_mm"], dtype=np.float64)
    z_min, z_max = float(z_levels.min()), float(z_levels.max())
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
    ], dtype=np.float64)
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot([corners[a,0], corners[b,0]], [corners[a,1], corners[b,1]], [corners[a,2], corners[b,2]],
                color="#334155", linewidth=0.8, alpha=0.7)

    # Camera position
    ax.scatter(*cam_pos_robot, c="#f43f5e", s=90, zorder=10, label="Camera origin")
    ax.text(float(cam_pos_robot[0]), float(cam_pos_robot[1]), float(cam_pos_robot[2]) + 20,
            f"Camera\n({cam_pos_robot[0]:.0f}, {cam_pos_robot[1]:.0f}, {cam_pos_robot[2]:.0f}) mm",
            color="#f43f5e", fontsize=8, ha="center", va="bottom")

    # Camera axes
    _draw_arrow(ax, cam_pos_robot, cam_x_robot, AXIS_ARROW_SCALE_MM, "#ef4444", "X_cam")
    _draw_arrow(ax, cam_pos_robot, cam_y_robot, AXIS_ARROW_SCALE_MM, "#22c55e", "Y_cam")
    _draw_arrow(ax, cam_pos_robot, view_dir_robot, AXIS_ARROW_SCALE_MM * 1.6, "#facc15", "view →")

    # Viewing frustum sketch
    _draw_frustum(ax, cam_pos_robot, view_dir_robot, cam_x_robot, cam_y_robot,
                  focal_dist=180.0, half_fov_h=28.0, half_fov_v=20.0)

    # Robot world axes at origin
    orig = np.zeros(3)
    for v, col, lbl in [
        (np.array([AXIS_ARROW_SCALE_MM, 0, 0]), "#f87171", "X_robot"),
        (np.array([0, AXIS_ARROW_SCALE_MM, 0]), "#4ade80", "Y_robot"),
        (np.array([0, 0, AXIS_ARROW_SCALE_MM]), "#60a5fa", "Z_robot"),
    ]:
        ax.quiver(*orig, *v, color=col, linewidth=1.5, arrow_length_ratio=0.18)
        tip = orig + v * 1.2
        ax.text(*tip, lbl, color=col, fontsize=7)

    ax.set_xlabel("X_robot  (mm)", color=_MUTED, fontsize=8, labelpad=6)
    ax.set_ylabel("Y_robot  (mm)", color=_MUTED, fontsize=8, labelpad=6)
    ax.set_zlabel("Z_robot  (mm)", color=_MUTED, fontsize=8, labelpad=6)

    # Annotation: camera viewing direction vector
    ax.text2D(0.02, 0.96,
              f"Camera position: ({cam_pos_robot[0]:.1f}, {cam_pos_robot[1]:.1f}, {cam_pos_robot[2]:.1f}) mm\n"
              f"View direction: ({view_dir_robot[0]:.3f}, {view_dir_robot[1]:.3f}, {view_dir_robot[2]:.3f})  [unit vec]\n"
              f"Yellow arrow = optical axis (where camera looks)",
              transform=ax.transAxes, color=_FG, fontsize=8, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#111827", edgecolor="#334155", alpha=0.88))

    all_pts = np.vstack([robot_xyz_mm, cam_pos_robot.reshape(1, 3),
                          cam_pos_robot.reshape(1, 3) + view_dir_robot.reshape(1, 3) * 200])
    _set_equal_axes(ax, all_pts, margin_frac=0.15)
    ax.view_init(elev=22, azim=-50)
    if CLEAN_EXPORTS:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        for text in ax.texts:
            text.set_visible(False)
    else:
        ax.legend(loc="upper right", fontsize=8, labelcolor=_FG, framealpha=0.4, facecolor=_AX_BG)
    fig.tight_layout()
    return fig


# ── main ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize stereo camera calibration grid and pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save", action="store_true", help="Save publication figures")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for saved figures")
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI, help="Output DPI")
    parser.add_argument("--max-labels", type=int, default=None, help="Max support points to label (default: all)")
    parser.add_argument("--no-gui", action="store_true", help="Suppress plt.show() (useful with --save)")
    parser.add_argument("--annotated", action="store_true", help="Keep titles, labels, and annotations")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global CLEAN_EXPORTS

    args = parse_args(argv)
    CLEAN_EXPORTS = not bool(args.annotated)

    print("[CALIB VIZ] Loading calibration bundle...")
    bundle = _load_bundle()

    # Prefer support points that span the full calibration grid
    robot_xyz_mm: np.ndarray = np.asarray(bundle["support_robot_xyz_mm"], dtype=np.float64)
    if len(robot_xyz_mm) == 0:
        raise ValueError("support_robot_xyz_mm is empty in calibration bundle")

    B = np.asarray(bundle["B_cam_from_robot_xyz_3x4"], dtype=np.float64)
    A = np.asarray(bundle["A_robot_from_cam_xyz_3x4"], dtype=np.float64)

    # Project robot grid to camera frame
    cam_xyz_mm = _project_robot_to_cam(robot_xyz_mm, B)
    finite = np.all(np.isfinite(cam_xyz_mm), axis=1)
    robot_xyz_mm = robot_xyz_mm[finite]
    cam_xyz_mm = cam_xyz_mm[finite]

    print(f"[CALIB VIZ] {len(robot_xyz_mm)} calibration support points loaded")
    print(f"[CALIB VIZ] Robot frame: x=[{robot_xyz_mm[:,0].min():.0f},{robot_xyz_mm[:,0].max():.0f}]  "
          f"y=[{robot_xyz_mm[:,1].min():.0f},{robot_xyz_mm[:,1].max():.0f}]  "
          f"z=[{robot_xyz_mm[:,2].min():.0f},{robot_xyz_mm[:,2].max():.0f}]")
    print(f"[CALIB VIZ] Camera frame: x=[{cam_xyz_mm[:,0].min():.0f},{cam_xyz_mm[:,0].max():.0f}]  "
          f"y=[{cam_xyz_mm[:,1].min():.0f},{cam_xyz_mm[:,1].max():.0f}]  "
          f"z=[{cam_xyz_mm[:,2].min():.0f},{cam_xyz_mm[:,2].max():.0f}]")

    # Derive camera pose
    cam_pos_robot = _camera_position_in_robot_frame(B)
    view_dir_robot = _camera_view_direction_in_robot_frame(A, sign_flip_z=True)
    cam_x_robot, cam_y_robot, _ = _camera_axes_in_robot_frame(A, sign_flip_z=True)

    print(f"[CALIB VIZ] Camera position in robot frame: "
          f"({cam_pos_robot[0]:.1f}, {cam_pos_robot[1]:.1f}, {cam_pos_robot[2]:.1f}) mm")
    print(f"[CALIB VIZ] Camera viewing direction (robot frame): "
          f"({view_dir_robot[0]:.3f}, {view_dir_robot[1]:.3f}, {view_dir_robot[2]:.3f})")

    max_labels = args.max_labels if args.max_labels is not None else MAX_LABEL_POINTS
    fig1 = make_camera_space_figure(robot_xyz_mm, cam_xyz_mm, max_labels=max_labels)
    fig2 = make_camera_pose_figure(robot_xyz_mm, cam_pos_robot, view_dir_robot,
                                    cam_x_robot, cam_y_robot, bundle)

    if args.save:
        out_dir = args.out_dir.resolve() if args.out_dir is not None else PUBLICATION_GLOBAL_ROOT
        calib_dir = out_dir / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        outputs.extend(
            save_figure_bundle(
                fig1,
                calib_dir / "calibration_camera_space_grid",
                dpi=args.dpi,
                facecolor=fig1.get_facecolor(),
            )
        )
        outputs.extend(
            save_figure_bundle(
                fig2,
                calib_dir / "calibration_camera_pose_robot_frame",
                dpi=args.dpi,
                facecolor=fig2.get_facecolor(),
            )
        )
        with (calib_dir / "calibration_support_points.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["robot_x_mm", "robot_y_mm", "robot_z_mm", "camera_x_mm", "camera_y_mm", "camera_z_mm"])
            for robot_xyz, cam_xyz in zip(robot_xyz_mm, cam_xyz_mm):
                writer.writerow([*map(float, robot_xyz), *map(float, cam_xyz)])
        with (calib_dir / "calibration_camera_pose.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["quantity", "x", "y", "z"])
            writer.writerow(["camera_position_robot_mm", *map(float, cam_pos_robot)])
            writer.writerow(["camera_view_direction_unit", *map(float, view_dir_robot)])
            writer.writerow(["camera_x_axis_unit", *map(float, cam_x_robot)])
            writer.writerow(["camera_y_axis_unit", *map(float, cam_y_robot)])
        for path in outputs:
            print(f"[SAVE] {path}")

    if not args.no_gui:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
