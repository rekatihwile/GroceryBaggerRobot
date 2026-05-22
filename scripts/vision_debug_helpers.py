from __future__ import annotations

"""scripts/vision_debug_helpers.py

Shared drawing, saving and layout helpers used by the live vision debug scripts.

No camera opens here.  No GPU inference here.  Pure OpenCV + numpy utilities.
"""

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ------------------------------------------------------------------ #
# Colour palette
# ------------------------------------------------------------------ #

_PALETTE: list[tuple[int, int, int]] = [
    (220, 50,  50),  (50, 200,  50),  (50,  50, 220), (180, 180,   0),
    (  0, 180, 180), (180,   0, 180), (255, 128,   0), (  0, 128, 255),
    (128, 255,   0), (255,   0, 128), (100, 200, 200), (200, 100, 200),
    (200, 200, 100), ( 80,  80, 200), (200,  80,  80), ( 80, 200,  80),
    (160, 100,  50), ( 50, 100, 160), (160,  50, 100), (100, 160,  50),
]


def colour_for(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(class_id) % len(_PALETTE)]


# ------------------------------------------------------------------ #
# YOLO overlay drawing
# ------------------------------------------------------------------ #

def draw_yolo_detections(
    image_bgr: np.ndarray,
    detections: list[Any],          # list[YOLODetection]
    *,
    alpha: float = 0.45,
    draw_axes: bool = True,
    draw_bbox: bool = True,
    draw_label: bool = True,
    draw_centroid: bool = True,
    label_offset: int = 3,
    numbered: bool = False,         # prefix label with 1-based detection index
) -> np.ndarray:
    """Draw YOLO masks, boxes, labels, centroids and PCA axes onto a copy of image_bgr.

    Expects each detection to be a YOLODetection (from vision.yolo_segmenter).
    The detections are indexed 1-based for display when numbered=True.
    """
    overlay = image_bgr.copy()
    canvas = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    for idx, det in enumerate(detections):
        colour = colour_for(det.class_id)

        # --- Filled mask on overlay ---
        mask = np.asarray(det.mask, dtype=bool)
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        colour_layer = np.zeros_like(canvas)
        colour_layer[mask] = colour
        overlay[mask] = (
            np.asarray(overlay[mask], dtype=np.float32) * (1 - alpha)
            + np.asarray(colour_layer[mask], dtype=np.float32) * alpha
        ).astype(np.uint8)

        cx = float(det.centroid_px[0]) if np.isfinite(det.centroid_px[0]) else w // 2
        cy = float(det.centroid_px[1]) if np.isfinite(det.centroid_px[1]) else h // 2

        # --- Bounding box ---
        if draw_bbox:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        # --- Centroid dot ---
        if draw_centroid:
            cv2.circle(overlay, (int(cx), int(cy)), 5, colour, -1)
            cv2.circle(overlay, (int(cx), int(cy)), 5, (0, 0, 0), 1)

        # --- PCA axes ---
        if draw_axes:
            _draw_axis(
                overlay,
                cx, cy,
                det.major_axis_length_px,
                det.major_axis_angle_deg,
                colour=colour,
                thickness=2,
            )
            _draw_axis(
                overlay,
                cx, cy,
                det.minor_axis_length_px,
                det.minor_axis_angle_deg,
                colour=(0, 200, 255),
                thickness=1,
            )

        # --- Label ---
        if draw_label:
            prefix = f"{idx + 1}: " if numbered else ""
            label = f"{prefix}{det.class_name} {det.confidence:.2f}"
            x1_l = int(det.bbox[0])
            y1_l = int(det.bbox[1])
            lx = max(x1_l, 2)
            ly = max(y1_l - label_offset, 18)
            _draw_label(overlay, label, lx, ly, colour)

    return overlay


def _draw_axis(
    img: np.ndarray,
    cx: float,
    cy: float,
    half_len: float,
    angle_deg: float,
    *,
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if half_len < 2:
        return
    theta = float(np.radians(angle_deg))
    dx = np.cos(theta) * half_len * 0.5
    dy = np.sin(theta) * half_len * 0.5
    p1 = (int(round(cx - dx)), int(round(cy - dy)))
    p2 = (int(round(cx + dx)), int(round(cy + dy)))
    cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    colour: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    pad = 3
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


# ------------------------------------------------------------------ #
# Disparity preview
# ------------------------------------------------------------------ #

def make_disparity_preview(
    disparity: np.ndarray,
    *,
    min_disparity: float = 1.0,
    colormap: int = cv2.COLORMAP_TURBO,
    invalid_colour: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    disp = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > min_disparity)

    if not np.any(valid):
        preview = np.zeros((*disp.shape, 3), dtype=np.uint8)
        preview[:] = invalid_colour
        return preview

    lo = float(np.percentile(disp[valid], 2))
    hi = float(np.percentile(disp[valid], 98))
    norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(u8, colormap)
    coloured[~valid] = invalid_colour
    return coloured


def make_masked_disparity_preview(
    disparity: np.ndarray,
    mask: np.ndarray,
    *,
    min_disparity: float = 1.0,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    masked_disp = np.where(mask.astype(bool), disparity, np.nan)
    return make_disparity_preview(
        masked_disp, min_disparity=min_disparity, colormap=colormap
    )


# ------------------------------------------------------------------ #
# Debug grid composition
# ------------------------------------------------------------------ #

def compose_debug_grid(
    tiles: list[tuple[str, np.ndarray | None]],
    *,
    cell_w: int = 640,
    cell_h: int = 400,
    cols: int = 2,
    bg_colour: tuple[int, int, int] = (30, 30, 30),
    label_colour: tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """Tile a list of (label, image_or_None) into a grid.

    None entries render as a labelled placeholder.
    """
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * cell_h, cols * cell_w, 3), bg_colour, dtype=np.uint8)

    for idx, (label, img) in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y0, y1 = row * cell_h, (row + 1) * cell_h
        x0, x1 = col * cell_w, (col + 1) * cell_w

        cell = np.full((cell_h, cell_w, 3), bg_colour, dtype=np.uint8)

        if img is not None:
            # Fit inside cell preserving aspect ratio
            img_h, img_w = img.shape[:2]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            scale = min(cell_w / max(img_w, 1), cell_h / max(img_h, 1))
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            off_x = (cell_w - new_w) // 2
            off_y = (cell_h - new_h) // 2
            cell[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        # Label
        if label:
            cv2.putText(
                cell, label, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_colour, 1, cv2.LINE_AA,
            )

        canvas[y0:y1, x0:x1] = cell

    return canvas


# ------------------------------------------------------------------ #
# JSON / file utilities
# ------------------------------------------------------------------ #

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def timestamped_output_dir(base: Path, prefix: str = "run") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = base / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detections_to_json(detections: list[Any]) -> list[dict]:
    """Convert a list of YOLODetection objects to JSON-serialisable dicts."""
    out = []
    for i, det in enumerate(detections):
        cx = float(det.centroid_px[0]) if np.isfinite(det.centroid_px[0]) else None
        cy = float(det.centroid_px[1]) if np.isfinite(det.centroid_px[1]) else None
        out.append(
            {
                "index": i + 1,
                "class_id": int(det.class_id),
                "class_name": str(det.class_name),
                "confidence": round(float(det.confidence), 4),
                "mask_area_px": int(det.mask_area),
                "bbox": [round(float(v), 2) for v in det.bbox],
                "centroid_px": [round(cx, 2), round(cy, 2)] if cx is not None else None,
                "major_axis_length_px": round(float(det.major_axis_length_px), 2),
                "minor_axis_length_px": round(float(det.minor_axis_length_px), 2),
                "major_axis_angle_deg": round(float(det.major_axis_angle_deg), 2),
                "minor_axis_angle_deg": round(float(det.minor_axis_angle_deg), 2),
            }
        )
    return out


def save_disparity_heatmap(path: Path, disparity: np.ndarray, *, min_disparity: float = 1.0) -> None:
    preview = make_disparity_preview(disparity, min_disparity=min_disparity)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview)
