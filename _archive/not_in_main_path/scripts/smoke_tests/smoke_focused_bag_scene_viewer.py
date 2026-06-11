from __future__ import annotations

"""Focused test viewer for bag-scene storytelling.

Layout:
  Row 1: raw stereo pair | point cloud of only unbagged groceries
  Row 2: disparity | YOLO mask | 3D render of only the bag / packed result
"""

from pathlib import Path
import argparse
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import scripts.capstone.autonomous_system_wrapper as demo_mod
import scripts.dry_run_autonomous as dry_mod
from config.robot_config import ROBOT_CONFIG
from hardware.robot import Robot
from test_calibration_bundle_live_stereo_z_pickplace import load_bundle
from vision.pick_survey_pipeline import load_vision
from vision.stereo_rectifier import StereoRectifier
from vision.torch_device import select_torch_device


SAVE_OUTPUT_DIR = Path(__file__).resolve().parent
FIG_BG = "#171728"
PANEL_BG = "#0e0f1a"


def _draw_raw_stereo_pair(ax, left_bgr: np.ndarray, right_bgr: np.ndarray) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Raw Stereo Pair", color="#aaddff", fontsize=10, pad=6)
    both = np.concatenate([left_bgr, right_bgr], axis=1)
    ax.imshow(cv2.cvtColor(both, cv2.COLOR_BGR2RGB))
    h, w = left_bgr.shape[:2]
    ax.axvline(w - 0.5, color="#ffffff", linewidth=1.0, alpha=0.6)
    ax.text(8, 18, "Left", color="#ffffff", fontsize=9, fontweight="bold")
    ax.text(w + 8, 18, "Right", color="#ffffff", fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_yolo_mask_panel(ax, rect_left_bgr: np.ndarray, survey) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title("YOLO Mask", color="#aaddff", fontsize=10, pad=6)
    rgb = cv2.cvtColor(rect_left_bgr, cv2.COLOR_BGR2RGB)
    mask_canvas = np.zeros_like(rgb)
    for i, dbg in enumerate(survey.candidates, start=1):
        det = dbg.best_detection
        if det.mask is None:
            continue
        mask = np.asarray(det.mask, dtype=bool)
        color_hex = demo_mod._PALETTE[(i - 1) % len(demo_mod._PALETTE)]
        color_rgb = np.array(
            [int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)],
            dtype=np.uint8,
        )
        mask_canvas[mask] = color_rgb
    composite = np.where(mask_canvas.any(axis=2, keepdims=True), mask_canvas, (0.18 * rgb).astype(np.uint8))
    ax.imshow(composite)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_unbagged_pointcloud(ax, scene_objects: list[demo_mod.SceneObject]) -> None:
    demo_mod._style_3d(ax, "Point Cloud | Unbagged Groceries Only")
    scale_pts: list[np.ndarray] = []
    for obj in scene_objects:
        pts = np.asarray(obj.points_robot, dtype=np.float64).reshape(-1, 3)
        if len(pts) == 0:
            continue
        if len(pts) <= demo_mod.MAX_PTS_DISP:
            sample = pts
            colors = obj.point_colors_rgb
        else:
            idx = np.random.choice(len(pts), demo_mod.MAX_PTS_DISP, replace=False)
            sample = pts[idx]
            colors = obj.point_colors_rgb[idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=0.8, alpha=0.65)
        scale_pts.append(obj.raw_box.min_xyz_mm.copy())
        scale_pts.append(obj.raw_box.max_xyz_mm.copy())
    demo_mod._draw_platform_bounds(ax)
    ax.view_init(elev=24, azim=-61)
    if scale_pts:
        demo_mod._set_equal_aspect(ax, np.vstack(scale_pts))


def _draw_bag_only_panel(ax, surface_zone: dict, placement_steps: list[demo_mod.PlacementStep]) -> None:
    demo_mod._style_3d(ax, "3D Render | Bag Only")
    scale_pts: list[np.ndarray] = []
    scale_pts.extend(demo_mod._draw_place_zone(ax, surface_zone))
    for step in placement_steps:
        placed = step.placed
        packed_points = demo_mod._transform_points_for_box(step.scene_object, placed.raw_box)
        if len(packed_points) <= demo_mod.MAX_PTS_DISP:
            sample = packed_points
            colors = step.scene_object.point_colors_rgb
        else:
            idx = np.random.choice(len(packed_points), demo_mod.MAX_PTS_DISP, replace=False)
            sample = packed_points[idx]
            colors = step.scene_object.point_colors_rgb[idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], c=colors, s=0.8, alpha=0.62)
        demo_mod._draw_box_3d(ax, placed.raw_box, step.scene_object.color, alpha_face=0.10, alpha_edge=0.55, ls="-")
        scale_pts.append(placed.raw_box.min_xyz_mm.copy())
        scale_pts.append(placed.raw_box.max_xyz_mm.copy())
    ax.view_init(elev=26, azim=-53)
    if scale_pts:
        demo_mod._set_equal_aspect(ax, np.vstack(scale_pts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(REPO_ROOT / "Training_Images"))
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)

    training_dir = Path(args.images)
    pair_index, left_path, right_path = demo_mod._find_stereo_pair(training_dir, args.index)
    print(f"[FOCUSED VIEW] pair {pair_index:04d} left={left_path.name}")

    dry_mod._configure_vision_modules()
    device_info = select_torch_device(use_cuda=demo_mod.USE_CUDA, use_half=demo_mod.USE_HALF)
    yolo, raft = load_vision(device_info)

    calib_path = REPO_ROOT / dry_mod._SURVEY.STEREO_CALIBRATION_PATH
    stereo_data = np.load(str(calib_path), allow_pickle=False)
    stereo_calib = {k: np.asarray(stereo_data[k]) for k in stereo_data.files}
    rectifier = StereoRectifier(stereo_calib)
    bundle = load_bundle(REPO_ROOT / dry_mod._SURVEY.BUNDLE_PATH)

    left_bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right_bgr = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left_bgr is None or right_bgr is None:
        raise RuntimeError("failed to load stereo pair")

    survey = dry_mod.survey_from_images(
        left_bgr,
        right_bgr,
        yolo=yolo,
        raft=raft,
        rectifier=rectifier,
        stereo_calib=stereo_calib,
        bundle=bundle,
    )
    if not survey.candidates:
        raise RuntimeError("no survey candidates")

    scene_objects = demo_mod._build_scene_objects(survey, left_bgr, bundle)
    surface_zone = demo_mod._load_demo_surface_zone()
    placement_steps = demo_mod._compute_packing_sequence(scene_objects, surface_zone, pair_index=pair_index)

    fig = plt.figure(figsize=(18, 10), facecolor=FIG_BG)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.25, 1.05, 1.15],
        height_ratios=[1.0, 1.0],
        hspace=0.12,
        wspace=0.08,
    )

    ax_raw = fig.add_subplot(gs[0, 0:2])
    ax_pc = fig.add_subplot(gs[0, 2], projection="3d")
    ax_disp = fig.add_subplot(gs[1, 0])
    ax_mask = fig.add_subplot(gs[1, 1])
    ax_bag = fig.add_subplot(gs[1, 2], projection="3d")

    _draw_raw_stereo_pair(ax_raw, left_bgr, right_bgr)
    _draw_unbagged_pointcloud(ax_pc, scene_objects)

    ax_disp.set_facecolor(PANEL_BG)
    ax_disp.set_title("Disparity", color="#aaddff", fontsize=10, pad=6)
    disparity_overlay = survey.candidates[0].disparity_overlay
    ax_disp.imshow(cv2.cvtColor(disparity_overlay, cv2.COLOR_BGR2RGB))
    ax_disp.set_xticks([])
    ax_disp.set_yticks([])

    rect_left_bgr = np.asarray(survey.candidates[0].left_overlay, dtype=np.uint8)
    _draw_yolo_mask_panel(ax_mask, rect_left_bgr, survey)
    _draw_bag_only_panel(ax_bag, surface_zone, placement_steps)

    fig.suptitle(
        f"Focused Bag Scene Viewer | pair {pair_index:04d} | bag floor z={float(surface_zone.get('surface_z_mm', 0.0)):.1f} mm",
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    if args.save:
        out = SAVE_OUTPUT_DIR / f"focused_bag_scene_{pair_index:04d}.png"
        fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[SAVE] {out}")

    if args.no_gui:
        plt.close(fig)
        return 0

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
