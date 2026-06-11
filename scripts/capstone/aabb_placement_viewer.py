from __future__ import annotations

"""scripts/capstone/aabb_placement_viewer.py

Capstone Visualization 2 — Point Cloud → Padded AABB

Shows three stages after the vision pipeline runs on a saved stereo pair:

  Panel A  Point cloud in stereo / camera frame
           (raw triangulated XYZ — what the sensor sees)

  Panel B  Point cloud in robot frame after transform
           (applies the A_robot_from_cam_xyz_3x4 calibration matrix)

  Panel C  Per-object robot-frame point clouds with padded AABB hit-boxes
           overlaid — the exact boxes the placement planner operates on.

Run:
    cd C:\\Users\\elipp\\OneDrive\\Documents\\Grocery_Buildup
    python scripts/capstone/aabb_placement_viewer.py
    python scripts/capstone/aabb_placement_viewer.py --index 2 --save
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.place import DEFAULT_PLACE

# VS Code IDE defaults.
# Copy/paste your image directory here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\Training_Images"
IDE_DEFAULT_IMAGES_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
TRAINING_IMAGES_DIR = Path(IDE_DEFAULT_IMAGES_DIR_STR) if IDE_DEFAULT_IMAGES_DIR_STR else (_REPO_ROOT / "Training_Images")

# Optional explicit output override. None uses paper_figure_sources.
IDE_DEFAULT_SAVE_OUTPUT_DIR_STR: str | None = None

STEREO_CALIB_PATH   = _REPO_ROOT / "stereo_calibration.npz"
BUNDLE_PATH         = _REPO_ROOT / "robot_calibration_bundle.npz"
YOLO_WEIGHTS_PATH   = _REPO_ROOT / "yolo_weights/full_data.pt"
YOLO_FALLBACK_PATH  = _REPO_ROOT / "yolo_weights/validate_V2.pt"
RAFT_ROOT           = _REPO_ROOT / "RAFT-Stereo"
RAFT_CKPT_PATH      = _REPO_ROOT / "RAFT-Stereo/models/raftstereo-middlebury.pth"

YOLO_CONF    = 0.35
YOLO_IMGSZ   = 640
PAD_X_MM     = float(DEFAULT_PLACE.PAD_X_MM)
PAD_Y_MM     = float(DEFAULT_PLACE.PAD_Y_MM)
PAD_Z_MM     = float(DEFAULT_PLACE.PAD_Z_MM)
MAX_PTS_DISP = 5000   # max scatter points per object in the viewer
USE_CUDA     = True
USE_HALF     = True

# Set True when Training_Images are already rectified (the default capture
# script saves pre-rectified pairs).  Set False only if you pass raw images.
IMAGES_ALREADY_RECTIFIED = True

SAVE_OUTPUT_DIR = Path(IDE_DEFAULT_SAVE_OUTPUT_DIR_STR) if IDE_DEFAULT_SAVE_OUTPUT_DIR_STR else (_REPO_ROOT / "paper_figure_sources/global/diagnostics")

# ============================================================

import argparse
import re
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import masked_disparity_to_pointcloud, cam_points_to_robot_xyz
from scripts.capstone.pointcloud_color import photo_colors_for_points
from planning.aabb_utils import aabb_from_object_candidate, pad_aabb, AxisAlignedBox3D
from vision.object_geometry import build_object_candidate
from vision.pick_z_resolver import resolve_robust_object_z
from config.workspace.workspace_config import get_workspace_filter_config, workspace_bounds_mm
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    output_dir_for_images,
    save_figure_bundle,
)

# Robot-side candidate gates (mirrors BestCandidateConfig defaults)
_WORKSPACE_CFG = get_workspace_filter_config("wet_run")
_WS_X_MIN, _WS_X_MAX, _WS_Y_MIN, _WS_Y_MAX = workspace_bounds_mm(_WORKSPACE_CFG)
_MIN_VOLUME_MM3: float = 1.0
_MAX_VOLUME_MM3: float = 3_000_000.0
_GRIPPER_OFFSET_MM: float = 130.0
_Z_MAX_MM: float = 275.0
CLEAN_EXPORTS = True


class _NullRobot:
    """Minimal stand-in so build_object_candidate works without a real robot.

    refresh_candidate_z_bias() always calls robot.fk() to record the EE pose
    at survey time.  With no AprilTag EE in the static image (stereo_tags={}),
    the Z-bias correction path is skipped entirely — only fk_xyz_at_update and
    fk_phi_at_update are set, which are unused in the capstone scripts.
    """
    def fk(self):
        return 0.0, 0.0, 0.0, 0.0

    def check_cartesian_pose_safe(self, x, y, z):
        return True, "no_robot"

    @property
    def cfg(self):
        return None

_PALETTE = [
    (0.18, 0.55, 0.90),
    (0.95, 0.55, 0.15),
    (0.18, 0.78, 0.45),
    (0.88, 0.25, 0.30),
    (0.60, 0.35, 0.80),
    (0.95, 0.75, 0.20),
]


def _find_stereo_pairs(d: Path) -> list[tuple[int, Path, Path]]:
    lr = re.compile(r"Stereo_Left_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    rr = re.compile(r"Stereo_Right_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    lefts, rights = {}, {}
    for p in d.iterdir():
        m = lr.match(p.name)
        if m: lefts[int(m.group(1))] = p; continue
        m = rr.match(p.name)
        if m: rights[int(m.group(1))] = p
    common = sorted(set(lefts) & set(rights))
    return [(i, lefts[i], rights[i]) for i in common]


def _corners(box: AxisAlignedBox3D) -> np.ndarray:
    mn, mx = box.min_xyz_mm, box.max_xyz_mm
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float64)


def _box_faces(corners: np.ndarray) -> list[list[np.ndarray]]:
    i = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
    return [[corners[v] for v in face] for face in i]


def _draw_box_3d(ax, box: AxisAlignedBox3D, color, alpha_edge=0.9, alpha_face=0.08, linestyle="-"):
    c = _corners(box)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot([c[a,0],c[b,0]], [c[a,1],c[b,1]], [c[a,2],c[b,2]],
                color=color, alpha=alpha_edge, linewidth=1.2, linestyle=linestyle)
    if alpha_face > 0:
        faces = Poly3DCollection(_box_faces(c), alpha=alpha_face)
        faces.set_facecolor(color)
        ax.add_collection3d(faces)


def _set_equal_aspect(ax, pts: np.ndarray) -> None:
    if len(pts) == 0:
        return
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    span = max((mx - mn).max() * 0.6, 50.0)
    ax.set_xlim(center[0]-span, center[0]+span)
    ax.set_ylim(center[1]-span, center[1]+span)
    ax.set_zlim(center[2]-span, center[2]+span)


def _style_3d(ax, title: str, xlabel="X", ylabel="Y", zlabel="Z") -> None:
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="#aaddff", fontsize=9, pad=4)
    ax.set_xlabel(xlabel + " (mm)", color="#999", fontsize=7, labelpad=2)
    ax.set_ylabel(ylabel + " (mm)", color="#999", fontsize=7, labelpad=2)
    ax.set_zlabel(zlabel + " (mm)", color="#999", fontsize=7, labelpad=2)
    ax.tick_params(colors="#777", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#2a2a3a")


def run_and_plot(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    rectifier: StereoRectifier,
    stereo_calib: dict,
    bundle: dict,
    pair_index: int,
    save: bool = False,
) -> plt.Figure:

    # ── Pipeline ──────────────────────────────────────────────────────────
    if IMAGES_ALREADY_RECTIFIED:
        rect_l, rect_r = left_bgr, right_bgr
    else:
        rect_l, rect_r = rectifier.rectify(left_bgr, right_bgr)
    dets = yolo.segment(rect_l)
    print(f"[AABB VIZ] {len(dets)} detections")
    disp = raft.predict_disparity(rect_l, rect_r, color="BGR")

    cam_pts_all:   list[np.ndarray] = []  # camera frame
    robot_pts_all: list[np.ndarray] = []  # robot frame
    point_colors_all: list[np.ndarray] = []
    raw_boxes:     list[AxisAlignedBox3D] = []
    pad_boxes:     list[AxisAlignedBox3D] = []
    det_labels:    list[str] = []
    rect_l_rgb = cv2.cvtColor(rect_l, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    for idx, det in enumerate(dets, start=1):
        try:
            pts_cam, uv = masked_disparity_to_pointcloud(det.mask, disp, stereo_calib)
        except Exception as exc:
            print(f"  [PC {idx}] failed: {exc}"); continue
        if len(pts_cam) < 300:
            print(f"  [PC {idx}] too few points ({len(pts_cam)})"); continue
        point_colors, color_valid = photo_colors_for_points(rect_l_rgb, uv, len(pts_cam))
        if point_colors is None or color_valid is None:
            print(f"  [PC {idx}] original-photo colors unavailable"); continue
        pts_cam = pts_cam[color_valid]

        try:
            pts_robot = cam_points_to_robot_xyz(pts_cam, bundle)
        except Exception as exc:
            print(f"  [XF {idx}] robot transform failed: {exc}"); continue

        # Build candidate for AABB
        try:
            cand = build_object_candidate(
                index=idx, yolo_det=det, points_cam=pts_cam,
                point_uv_px=uv, robot=_NullRobot(), bundle=bundle,
                stereo_tags={}, frame_i=0,
            )
            z_res = resolve_robust_object_z(pts_robot, pts_cam, candidate=cand)
            cand.object_robot_xyz_raw[2]       = z_res.robust_top_z_mm
            cand.object_robot_xyz_corrected[2] = z_res.robust_top_z_mm
            cand.z_debug = z_res

            raw = aabb_from_object_candidate(cand, default_label=det.class_name)
            pad = pad_aabb(raw, PAD_X_MM, PAD_Y_MM, PAD_Z_MM)
        except Exception as exc:
            print(f"  [AABB {idx}] failed: {exc}"); continue

        # ── Robot-side candidate gates (same logic the robot applies at runtime) ──
        cand_xy = np.asarray(cand.object_robot_xyz_raw[:2], dtype=np.float64)
        if _WORKSPACE_CFG.require_positive_platform_xy and (cand_xy[0] <= 0.0 or cand_xy[1] <= 0.0):
            print(f"  [FILTER {idx}] {det.class_name}: negative XY ({cand_xy[0]:.0f},{cand_xy[1]:.0f}), robot would reject")
            continue
        if not (_WS_X_MIN <= cand_xy[0] <= _WS_X_MAX and _WS_Y_MIN <= cand_xy[1] <= _WS_Y_MAX):
            print(f"  [FILTER {idx}] {det.class_name}: outside workspace xy ({cand_xy[0]:.0f},{cand_xy[1]:.0f}) "
                  f"bounds=[{_WS_X_MIN:.0f}-{_WS_X_MAX:.0f},{_WS_Y_MIN:.0f}-{_WS_Y_MAX:.0f}], robot would reject")
            continue
        vol = float(raw.size_xyz_mm[0]) * float(raw.size_xyz_mm[1]) * float(raw.size_xyz_mm[2])
        if vol < _MIN_VOLUME_MM3 or vol > _MAX_VOLUME_MM3:
            print(f"  [FILTER {idx}] {det.class_name}: volume {vol:.0f} mm3 out of range, robot would reject")
            continue
        top_z = float(raw.max_xyz_mm[2])
        if top_z + _GRIPPER_OFFSET_MM > _Z_MAX_MM:
            print(f"  [FILTER {idx}] {det.class_name}: top_z {top_z:.0f}+offset>{_Z_MAX_MM:.0f} (stereo phantom), robot would reject")
            continue

        cam_pts_all.append(pts_cam)
        robot_pts_all.append(pts_robot)
        point_colors_all.append(point_colors)
        raw_boxes.append(raw)
        pad_boxes.append(pad.padded_box)
        det_labels.append(f"#{idx} {det.class_name}")
        print(f"  [OK {idx}] {det.class_name}  pts={len(pts_cam)}  "
              f"aabb=({raw.size_xyz_mm[0]:.0f}x{raw.size_xyz_mm[1]:.0f}x{raw.size_xyz_mm[2]:.0f})mm")

    if not cam_pts_all:
        print("[AABB VIZ] no valid objects"); return None

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 7), facecolor="#1a1a2e")
    fig.suptitle(
        f"Stereo Frame → Robot Frame Transform + Padded AABB  |  Pair {pair_index:04d}",
        color="white", fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    # Panel A — camera frame
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    _style_3d(axA, "A  Point Cloud — Camera Frame\n(raw stereo triangulation)")
    all_cam = np.vstack(cam_pts_all)
    for i, (pts, colors, label) in enumerate(zip(cam_pts_all, point_colors_all, det_labels)):
        sample_indices = np.arange(len(pts)) if len(pts) <= MAX_PTS_DISP else np.random.choice(len(pts), MAX_PTS_DISP, False)
        samp = pts[sample_indices]
        axA.scatter(samp[:,0], samp[:,2], -samp[:,1],
                    c=colors[sample_indices], s=0.6, alpha=0.75, label=label)
    _set_equal_aspect(axA, np.column_stack([all_cam[:,0], all_cam[:,2], -all_cam[:,1]]))
    axA.set_xlabel("X_cam (mm)"); axA.set_ylabel("Z_cam (mm)"); axA.set_zlabel("-Y_cam (mm)")
    axA.legend(loc="upper left", fontsize=7, labelcolor="white",
               framealpha=0.3, facecolor="#0d0d1a")

    # Panel B — robot frame (same colours, different axes)
    axB = fig.add_subplot(gs[0, 1], projection="3d")
    _style_3d(axB, "B  Point Cloud — Robot Frame\n(after A_robot_from_cam × [x y z 1]ᵀ)")
    all_robot = np.vstack(robot_pts_all)
    for i, (pts, colors) in enumerate(zip(robot_pts_all, point_colors_all)):
        sample_indices = np.arange(len(pts)) if len(pts) <= MAX_PTS_DISP else np.random.choice(len(pts), MAX_PTS_DISP, False)
        samp = pts[sample_indices]
        axB.scatter(samp[:,0], samp[:,1], samp[:,2],
                    c=colors[sample_indices], s=0.6, alpha=0.75)
    # Draw robot origin axes
    orig = np.zeros(3)
    L = 80.0
    for v, col in [(np.array([L,0,0]),"red"),(np.array([0,L,0]),"lime"),(np.array([0,0,L]),"dodgerblue")]:
        axB.quiver(*orig, *v, color=col, linewidth=1.5, arrow_length_ratio=0.2)
    _set_equal_aspect(axB, all_robot)

    # Panel C — AABB boxes in robot frame
    axC = fig.add_subplot(gs[0, 2], projection="3d")
    _style_3d(axC, "C  AABB Hit-Boxes in Robot Frame\n(solid = raw  |  wire = padded)")
    all_pts_for_scale: list[np.ndarray] = []
    for i, (pts, colors, raw, pad, label) in enumerate(
        zip(robot_pts_all, point_colors_all, raw_boxes, pad_boxes, det_labels)
    ):
        col = _PALETTE[i % len(_PALETTE)]
        sample_indices = np.arange(len(pts)) if len(pts) <= MAX_PTS_DISP else np.random.choice(len(pts), MAX_PTS_DISP, False)
        samp = pts[sample_indices]
        axC.scatter(samp[:,0], samp[:,1], samp[:,2],
                    c=colors[sample_indices], s=0.5, alpha=0.70)
        _draw_box_3d(axC, raw, col, alpha_face=0.12, linestyle="-")
        _draw_box_3d(axC, pad, col, alpha_face=0.0,  linestyle="--", alpha_edge=0.55)
        cx, cy, cz = raw.center_xyz_mm
        axC.text(cx, cy, cz + raw.size_xyz_mm[2]*0.6 + 8,
                 label.replace("#",""), color="white", fontsize=7,
                 ha="center", va="bottom")
        all_pts_for_scale.append(_corners(pad))

    _set_equal_aspect(axC, np.vstack(all_pts_for_scale))
    axC.text2D(0.03, 0.05, "─── raw AABB\n- - - padded AABB",
               transform=axC.transAxes, color="white", fontsize=7, va="bottom")

    return fig


def main(argv: list[str] | None = None) -> int:
    global CLEAN_EXPORTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(TRAINING_IMAGES_DIR))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_PUBLICATION_DPI)
    parser.add_argument("--annotated", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)
    CLEAN_EXPORTS = not bool(args.annotated)

    images_dir = Path(args.images)
    pairs = _find_stereo_pairs(images_dir)
    if not pairs:
        print("[ERROR] no stereo pairs"); return 1
    if args.index is not None:
        pairs = [(i,l,r) for i,l,r in pairs if i == args.index]
        if not pairs:
            print(f"[ERROR] pair {args.index} not found"); return 1

    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)
    weights = YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.exists() else YOLO_FALLBACK_PATH
    yolo = YOLOSegmenter(weights_path=str(weights), device_info=device_info,
                         imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=0.50,
                         retina_masks=True, min_mask_area_px=500)
    yolo.warmup()
    raft = RAFTStereoRunner(raft_root=str(RAFT_ROOT), checkpoint_path=str(RAFT_CKPT_PATH),
                            device_info=device_info, valid_iters=16)
    raft.warmup()

    calib = {k: np.asarray(v) for k, v in np.load(str(STEREO_CALIB_PATH), allow_pickle=False).items()}
    rectifier = StereoRectifier(calib)

    if BUNDLE_PATH.exists():
        from test_calibration_bundle_live_stereo_z_pickplace import load_bundle
        bundle = load_bundle(BUNDLE_PATH)
        print(f"[CALIB] bundle loaded (processed)")
    else:
        print(f"[WARN] bundle not found at {BUNDLE_PATH}; robot-frame panels will be empty")
        bundle = {}

    idx, lp, rp = pairs[0]
    left  = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)

    fig = run_and_plot(left, right, yolo=yolo, raft=raft,
                       rectifier=rectifier, stereo_calib=calib,
                       bundle=bundle, pair_index=idx, save=args.save)
    if fig is not None:
        if CLEAN_EXPORTS:
            fig.suptitle("")
            for ax in fig.axes:
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if hasattr(ax, "set_zlabel"):
                    ax.set_zlabel("")
                for text in ax.texts:
                    text.set_visible(False)
        if args.save:
            save_dir = output_dir_for_images(images_dir, args.out_dir) / "diagnostics"
            outputs = save_figure_bundle(
                fig,
                save_dir / f"pointcloud_to_padded_aabb_pair_{idx:04d}",
                dpi=args.dpi,
                facecolor=fig.get_facecolor(),
            )
            for path in outputs:
                print(f"[SAVE] {path}")
        if args.no_gui:
            plt.close(fig)
        else:
            plt.show()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise
