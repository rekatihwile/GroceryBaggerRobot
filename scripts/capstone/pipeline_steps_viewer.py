from __future__ import annotations

"""scripts/capstone/pipeline_steps_viewer.py

Capstone Visualization 1 — Stereo Pair → Point Cloud

Shows every stage of the vision pipeline on a single saved stereo pair:

  Step 0: Raw left + right images
  Step 1: Rectified left + right images
  Step 2: YOLO segmentation overlay (on rect-left)
  Step 3: RAFT disparity heat-map
  Step 4: Per-object masked disparity (one subplot per instance)
  Step 5: 3-D point cloud scatter (camera frame) — one object highlighted

Run:
    cd C:\\Users\\elipp\\OneDrive\\Documents\\Grocery_Buildup
    python scripts/capstone/pipeline_steps_viewer.py
    python scripts/capstone/pipeline_steps_viewer.py --index 3
    python scripts/capstone/pipeline_steps_viewer.py --save   # saves PNGs
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# VS Code IDE defaults.
# Copy/paste your image directory here (Windows raw string recommended), e.g.
# r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\Training_Images"
IDE_DEFAULT_IMAGES_DIR_STR: str | None = r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup\data\run_snapshots\run_20260531_165144"
TRAINING_IMAGES_DIR  = Path(IDE_DEFAULT_IMAGES_DIR_STR) if IDE_DEFAULT_IMAGES_DIR_STR else (_REPO_ROOT / "Training_Images")

# Optional explicit output override. None uses paper_figure_sources.
IDE_DEFAULT_SAVE_OUTPUT_DIR_STR: str | None = None

STEREO_CALIB_PATH    = _REPO_ROOT / "stereo_calibration.npz"
YOLO_WEIGHTS_PATH    = _REPO_ROOT / "yolo_weights/full_data.pt"
YOLO_FALLBACK_PATH   = _REPO_ROOT / "yolo_weights/validate_V2.pt"
RAFT_ROOT            = _REPO_ROOT / "RAFT-Stereo"
RAFT_CKPT_PATH       = _REPO_ROOT / "RAFT-Stereo/models/raftstereo-middlebury.pth"

YOLO_CONF        = 0.35
YOLO_IMGSZ       = 640
YOLO_IOU         = 0.50
RAFT_VALID_ITERS = 16
MIN_DISPARITY_PX = 1.0
USE_CUDA         = True
USE_HALF         = True

SAVE_OUTPUT_DIR = Path(IDE_DEFAULT_SAVE_OUTPUT_DIR_STR) if IDE_DEFAULT_SAVE_OUTPUT_DIR_STR else (_REPO_ROOT / "paper_figure_sources/global/diagnostics")

# Set True when Training_Images are already rectified (default from capture script).
IMAGES_ALREADY_RECTIFIED = True

# ============================================================

import argparse
import re
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import masked_disparity_to_pointcloud
from scripts.capstone.pointcloud_color import photo_colors_for_points
from scripts.capstone.publication_config import (
    DEFAULT_PUBLICATION_DPI,
    output_dir_for_images,
    save_figure_bundle,
)

CLEAN_EXPORTS = True

# ── palette for per-object colouring ──────────────────────────────────────
_PALETTE = [
    (0.18, 0.55, 0.90),  # blue
    (0.95, 0.55, 0.15),  # orange
    (0.18, 0.78, 0.45),  # green
    (0.88, 0.25, 0.30),  # red
    (0.60, 0.35, 0.80),  # purple
    (0.95, 0.75, 0.20),  # yellow
]


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _find_stereo_pairs(d: Path) -> list[tuple[int, Path, Path]]:
    lr = re.compile(r"Stereo_Left_(\d+)\.(jpg|jpeg|png)$",  re.IGNORECASE)
    rr = re.compile(r"Stereo_Right_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    lefts, rights = {}, {}
    for p in d.iterdir():
        m = lr.match(p.name)
        if m: lefts[int(m.group(1))] = p; continue
        m = rr.match(p.name)
        if m: rights[int(m.group(1))] = p
    common = sorted(set(lefts) & set(rights))
    return [(i, lefts[i], rights[i]) for i in common]


def _disparity_heatmap(disp: np.ndarray, min_d: float = 1.0) -> np.ndarray:
    d = np.asarray(disp, dtype=np.float32)
    valid = np.isfinite(d) & (d > min_d)
    if not np.any(valid):
        return np.zeros((*d.shape[:2], 3), dtype=np.uint8)
    lo, hi = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    out = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    heat[~valid] = 0
    return heat


def _draw_yolo_overlay(img: np.ndarray, dets: list[YOLODetection]) -> np.ndarray:
    out = img.copy()
    for i, det in enumerate(dets):
        col = tuple(int(c * 255) for c in _PALETTE[i % len(_PALETTE)])
        mask = np.asarray(det.mask, dtype=bool)
        if mask.shape != out.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = out.copy()
        overlay[mask] = col[::-1]  # BGR
        out = cv2.addWeighted(overlay, 0.38, out, 0.62, 0)
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), col[::-1], 2)
        label = f"#{i+1} {det.class_name} {det.confidence:.2f}"
        cv2.putText(out, label, (x1, max(16, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col[::-1], 1, cv2.LINE_AA)
    return out


def run_pipeline(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    rectifier: StereoRectifier,
    stereo_calib: dict,
) -> dict:
    """Run full pipeline and return all intermediate results."""
    if IMAGES_ALREADY_RECTIFIED:
        rect_left, rect_right = left_bgr, right_bgr
    else:
        rect_left, rect_right = rectifier.rectify(left_bgr, right_bgr)
    dets     = yolo.segment(rect_left)
    disparity = raft.predict_disparity(rect_left, rect_right, color="BGR")

    pointclouds: list[tuple[YOLODetection, np.ndarray, np.ndarray]] = []
    rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for det in dets:
        try:
            pts, uv_px = masked_disparity_to_pointcloud(det.mask, disparity, stereo_calib)
            colors_rgb, color_valid = photo_colors_for_points(rect_left_rgb, uv_px, len(pts))
            if colors_rgb is not None and color_valid is not None:
                pointclouds.append((det, pts[color_valid], colors_rgb))
        except Exception:
            pass

    return {
        "left_raw":    left_bgr,
        "right_raw":   right_bgr,
        "rect_left":   rect_left,
        "rect_right":  rect_right,
        "dets":        dets,
        "disparity":   disparity,
        "pointclouds": pointclouds,
    }


def make_figure(data: dict, pair_index: int) -> plt.Figure:
    dets  = data["dets"]
    pcs   = data["pointclouds"]
    n_obj = max(len(dets), 1)

    # Layout: rows depend on object count
    # Row 0: raw left | raw right | rect left | rect right
    # Row 1: YOLO overlay | disparity heatmap | (empty) | (empty)
    # Row 2+: per-object masked disparity + 3-D scatter (pairs)

    n_obj_rows = max(1, (n_obj + 1) // 2)
    total_rows = 2 + n_obj_rows

    fig = plt.figure(figsize=(16, 4 * total_rows), facecolor="#1a1a2e")
    fig.suptitle(
        f"Vision Pipeline  —  Pair {pair_index:04d}  |  "
        f"{len(dets)} object(s) detected",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(total_rows, 4, figure=fig,
                           hspace=0.45, wspace=0.25)

    def _ax(r, c, **kw):
        ax = fig.add_subplot(gs[r, c], **kw)
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        return ax

    def _img_ax(r, c, img_bgr, title):
        ax = _ax(r, c)
        ax.imshow(_bgr_to_rgb(img_bgr))
        ax.set_title(title, color="#aaddff", fontsize=9)
        ax.axis("off")
        return ax

    # Row 0: raw + rectified images
    _img_ax(0, 0, data["left_raw"],   "① Raw Left")
    _img_ax(0, 1, data["right_raw"],  "① Raw Right")
    _img_ax(0, 2, data["rect_left"],  "② Rectified Left")
    _img_ax(0, 3, data["rect_right"], "② Rectified Right")

    # Row 1: YOLO overlay + disparity
    yolo_overlay = _draw_yolo_overlay(data["rect_left"], dets)
    _img_ax(1, 0, yolo_overlay, f"③ YOLO  ({len(dets)} dets)")

    heat = _disparity_heatmap(data["disparity"])
    _img_ax(1, 1, heat, "④ RAFT Disparity Heatmap")

    # Disparity stats text
    d = data["disparity"]
    valid = np.isfinite(d) & (d > MIN_DISPARITY_PX)
    ax_stat = _ax(1, 2)
    ax_stat.axis("off")
    stats_txt = (
        f"Disparity stats\n"
        f"valid px : {int(np.sum(valid)):,}\n"
        f"min      : {float(np.nanmin(d[valid])):.1f} px\n"
        f"max      : {float(np.nanmax(d[valid])):.1f} px\n"
        f"mean     : {float(np.nanmean(d[valid])):.1f} px\n"
    )
    ax_stat.text(0.05, 0.85, stats_txt, color="#ccffcc", fontsize=9,
                 va="top", fontfamily="monospace",
                 transform=ax_stat.transAxes)
    ax_stat.set_title("④ Disparity Info", color="#aaddff", fontsize=9)

    # Object legend
    ax_leg = _ax(1, 3)
    ax_leg.axis("off")
    ax_leg.set_title("Object Legend", color="#aaddff", fontsize=9)
    for i, det in enumerate(dets[:6]):
        col = _PALETTE[i % len(_PALETTE)]
        ax_leg.add_patch(plt.Rectangle((0.03, 0.85 - i * 0.13), 0.08, 0.09,
                                        color=col, transform=ax_leg.transAxes))
        ax_leg.text(0.15, 0.895 - i * 0.13,
                    f"#{i+1}  {det.class_name}  conf={det.confidence:.2f}",
                    color="white", fontsize=8, va="center",
                    transform=ax_leg.transAxes)

    # Row 2+: per-object masked disparity + 3-D scatter
    for obj_i, (det, pts, colors_rgb) in enumerate(pcs[:n_obj_rows * 2]):
        row = 2 + obj_i // 2
        col_offset = (obj_i % 2) * 2
        pal_col = _PALETTE[obj_i % len(_PALETTE)]
        bgr_col = tuple(int(c * 255) for c in reversed(pal_col))

        # Masked disparity
        mask = np.asarray(det.mask, dtype=bool)
        if mask.shape != d.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8),
                              (d.shape[1], d.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        masked_heat = heat.copy()
        masked_heat[~mask] = (masked_heat[~mask] * 0.15).astype(np.uint8)
        ax_md = _ax(row, col_offset)
        ax_md.imshow(_bgr_to_rgb(masked_heat))
        ax_md.set_title(
            f"⑤ #{obj_i+1} {det.class_name}  masked disp  ({len(pts):,} pts)",
            color="#aaddff", fontsize=8,
        )
        ax_md.axis("off")

        # 3-D scatter
        ax3d = fig.add_subplot(gs[row, col_offset + 1], projection="3d")
        ax3d.set_facecolor("#0d0d1a")
        if len(pts) <= 4000:
            sample = pts
            sample_colors = colors_rgb
        else:
            sample_indices = np.random.choice(len(pts), 4000, replace=False)
            sample = pts[sample_indices]
            sample_colors = colors_rgb[sample_indices]
        ax3d.scatter(
            sample[:, 0], sample[:, 2], -sample[:, 1],
            c=sample_colors, s=0.8, alpha=0.75,
        )
        ax3d.set_xlabel("X cam", color="#aaa", fontsize=7, labelpad=2)
        ax3d.set_ylabel("Z cam", color="#aaa", fontsize=7, labelpad=2)
        ax3d.set_zlabel("Y cam", color="#aaa", fontsize=7, labelpad=2)
        ax3d.tick_params(colors="#888", labelsize=6)
        ax3d.set_title(
            f"⑤ #{obj_i+1} 3-D point cloud (cam frame)",
            color="#aaddff", fontsize=8,
        )
        for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333")

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

    training_dir = Path(args.images)
    pairs = _find_stereo_pairs(training_dir)
    if not pairs:
        print(f"[ERROR] no stereo pairs in {training_dir}")
        return 1

    if args.index is not None:
        pairs = [(i, l, r) for i, l, r in pairs if i == args.index]
        if not pairs:
            print(f"[ERROR] pair {args.index} not found")
            return 1

    # Load models once
    print("[INIT] loading torch device …")
    device_info = select_torch_device(use_cuda=USE_CUDA, use_half=USE_HALF)

    weights = YOLO_WEIGHTS_PATH if YOLO_WEIGHTS_PATH.exists() else YOLO_FALLBACK_PATH
    print(f"[INIT] loading YOLO from {weights} …")
    yolo = YOLOSegmenter(
        weights_path=str(weights), device_info=device_info,
        imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU,
        retina_masks=True, min_mask_area_px=500,
    )
    yolo.warmup()

    print(f"[INIT] loading RAFT from {RAFT_CKPT_PATH} …")
    raft = RAFTStereoRunner(
        raft_root=str(RAFT_ROOT), checkpoint_path=str(RAFT_CKPT_PATH),
        device_info=device_info, valid_iters=RAFT_VALID_ITERS,
    )
    raft.warmup()

    calib_data = np.load(str(STEREO_CALIB_PATH), allow_pickle=False)
    stereo_calib = {k: np.asarray(calib_data[k]) for k in calib_data.files}
    rectifier = StereoRectifier(stereo_calib)

    # Process (first pair if no --index given)
    idx, lp, rp = pairs[0]
    print(f"[INFO] processing pair {idx:04d}  left={lp.name}")
    left  = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)

    data = run_pipeline(left, right, yolo=yolo, raft=raft,
                        rectifier=rectifier, stereo_calib=stereo_calib)

    fig = make_figure(data, idx)

    if args.save:
        save_dir = output_dir_for_images(training_dir, args.out_dir) / "diagnostics"
        outputs = save_figure_bundle(
            fig,
            save_dir / f"perception_full_pipeline_pair_{idx:04d}",
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
