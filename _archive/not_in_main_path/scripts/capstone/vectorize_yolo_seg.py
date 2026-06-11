"""Vectorize a YOLO segmentation image into a PDF.

Step 1 (rasterized): raw stereo image + colored segmentation mask overlay.
Step 2 (vectorized):  bounding box rectangle + label text on top.

The mask is extracted from yolo_seg.png using the known OpenCV blend parameters
from _render_yolo_overlay_on_image:
    output_bgr = round(0.42 * [0,180,60] + 0.58 * raw_bgr)
Then the mask-only composite is rendered without any baked-in bbox/label lines,
so the PDF has clean raster underneath and fully editable vector annotations.

Outputs:
  yolo_seg_vector.pdf / .png  — in survey source dir
  04_yolo_overlay_saved.pdf / .jpg  — in paper_figure_sources tree (updated)

Usage:
    python scripts/capstone/vectorize_yolo_seg.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType — editable in Illustrator
matplotlib.rcParams["ps.fonttype"]  = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

SURVEY_DIR = Path(
    r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup"
    r"\presentation_outputs\run_20260608_122527GOODONE_ENDOFDAY_survey_story"
    r"\survey_0002_row_02_Chex_Mix\source"
)
PAPER_FIG_DIR = Path(
    r"C:\Users\elipp\OneDrive\Documents\Grocery_Buildup"
    r"\paper_figure_sources\paper_figures\figure_02_perception_pipeline"
)
MANIFEST_JSON = SURVEY_DIR / "manifest_row.json"
STEREO_LEFT   = SURVEY_DIR / "stereo_left_full.png"
YOLO_SEG      = SURVEY_DIR / "yolo_seg.png"

# Outputs
OUT_PDF       = SURVEY_DIR  / "yolo_seg_vector.pdf"
OUT_PNG       = SURVEY_DIR  / "yolo_seg_vector_preview.png"
OUT_PAPER_PDF = PAPER_FIG_DIR / "04_yolo_overlay_saved.pdf"
OUT_PAPER_JPG = PAPER_FIG_DIR / "04_yolo_overlay_saved.jpg"

# Mask blend params from _render_yolo_overlay_on_image in run_snapshot_manifest_survey_story.py
# overlay[mask] = [0, 180, 60]  (BGR)   cv2.addWeighted(overlay, 0.42, bgr, 0.58, 0, bgr)
MASK_COLOR_BGR = np.array([0, 180, 60], dtype=np.float32)
MASK_ALPHA     = 0.42
MASK_MATCH_TOL = 3   # max per-channel absolute error for mask detection (lossless PNG, only rounding)

BBOX_COLOR   = "#f59e0b"   # amber — Chex Mix class color
LABEL_FG     = "white"
LABEL_FONT   = 11
BBOX_LW      = 2.5
DPI_PREVIEW  = 300


def extract_mask(raw_bgr: np.ndarray, yolo_bgr: np.ndarray) -> np.ndarray:
    """Recover binary mask from a known alpha-blend.

    Returns a (H, W) bool array — True where the green mask was applied.
    """
    predicted = np.clip(
        np.round(MASK_ALPHA * MASK_COLOR_BGR + (1.0 - MASK_ALPHA) * raw_bgr.astype(np.float32)),
        0, 255,
    ).astype(np.uint8)

    diff = np.abs(yolo_bgr.astype(np.int32) - predicted.astype(np.int32))
    mask = np.all(diff <= MASK_MATCH_TOL, axis=2)

    # Morphological cleanup: close tiny gaps in the mask, remove isolated noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE,  kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,   kernel, iterations=1)
    return mask_u8.astype(bool)


def build_composite_rgb(raw_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the green mask overlay onto raw, without any bbox/label lines."""
    composite = raw_bgr.copy()
    overlay   = raw_bgr.copy()
    overlay[mask] = MASK_COLOR_BGR.astype(np.uint8)[[2, 1, 0]]  # BGR stays BGR
    cv2.addWeighted(overlay, MASK_ALPHA, composite, 1.0 - MASK_ALPHA, 0, composite)
    return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)


def render_figure(composite_rgb: np.ndarray, manifest: dict) -> plt.Figure:
    class_name = str(manifest["detection_class"])
    confidence = float(manifest["detection_confidence"])
    x1, y1, x2, y2 = manifest["detection_bbox_xyxy"]
    img_h, img_w = composite_rgb.shape[:2]

    fig_w = 7.0
    fig_h = fig_w * img_h / img_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # ── Rasterized: image + mask ──────────────────────────────────────────────
    ax.imshow(composite_rgb, aspect="auto", interpolation="lanczos")

    # ── Vector bounding box ───────────────────────────────────────────────────
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=BBOX_COLOR, linewidth=BBOX_LW, zorder=3,
    ))

    # ── Vector label chip ────────────────────────────────────────────────────
    label  = f"{class_name}  {confidence:.2f}"
    chip_y = y1 - 2 if y1 > 20 else y2 + 2
    chip_va = "bottom" if y1 > 20 else "top"
    ax.text(
        x1, chip_y, label,
        ha="left", va=chip_va,
        fontsize=LABEL_FONT, fontweight="bold", color=LABEL_FG,
        bbox=dict(boxstyle="square,pad=0.3", facecolor=BBOX_COLOR,
                  edgecolor="none", alpha=1.0),
        zorder=4,
    )

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_axis_off()
    return fig


def main() -> None:
    manifest  = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    raw_bgr   = cv2.imread(str(STEREO_LEFT),  cv2.IMREAD_COLOR)
    yolo_bgr  = cv2.imread(str(YOLO_SEG),     cv2.IMREAD_COLOR)

    if raw_bgr is None:
        raise FileNotFoundError(str(STEREO_LEFT))
    if yolo_bgr is None:
        raise FileNotFoundError(str(YOLO_SEG))

    print("Extracting segmentation mask from yolo_seg.png...")
    mask = extract_mask(raw_bgr, yolo_bgr)
    print(f"  Mask pixels: {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")

    composite_rgb = build_composite_rgb(raw_bgr, mask)

    fig = render_figure(composite_rgb, manifest)

    fig.savefig(str(OUT_PDF), facecolor=fig.get_facecolor())
    print(f"  Survey PDF:  {OUT_PDF}")

    fig.savefig(str(OUT_PNG), dpi=DPI_PREVIEW, facecolor=fig.get_facecolor())
    print(f"  Survey PNG:  {OUT_PNG}")

    fig.savefig(str(OUT_PAPER_PDF), facecolor=fig.get_facecolor())
    print(f"  Paper  PDF:  {OUT_PAPER_PDF}")

    fig.savefig(str(OUT_PAPER_JPG), dpi=DPI_PREVIEW, facecolor=fig.get_facecolor())
    print(f"  Paper  JPG:  {OUT_PAPER_JPG}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
