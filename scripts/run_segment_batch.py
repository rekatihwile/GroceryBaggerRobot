"""
run_segment_batch.py

Batch YOLO segmentation over a directory of images.

No argparse.  Edit the USER SETTINGS block below to configure paths and options.

Run:
    python scripts/run_segment_batch.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

IMAGE_DIR   = "data/images"       # input directory
OUTPUT_DIR  = "data/segment_out"  # output directory

YOLO_WEIGHTS_PATH = "yolo_weights/best.pt"

YOLO_IMGSZ = 640
YOLO_CONF  = 0.35
YOLO_IOU   = 0.50

SAVE_MASKS    = True    # save binary mask as .png for each detection
SAVE_OVERLAYS = True    # save BGR overlay image as .png
SAVE_METADATA_JSON = True  # save per-image metadata as .json

OVERWRITE_EXISTING = False
MAX_IMAGES: int | None = None  # None = process all
START_INDEX = 0

USE_CUDA = True

# ============================================================
# End of user settings
# ============================================================

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Random distinct BGR colours for up to 20 classes
_COLOURS = [
    (220, 50, 50), (50, 200, 50), (50, 50, 220), (180, 180, 0),
    (0, 180, 180), (180, 0, 180), (255, 128, 0), (0, 128, 255),
    (128, 255, 0), (255, 0, 128), (100, 200, 200), (200, 100, 200),
    (200, 200, 100), (80, 80, 200), (200, 80, 80), (80, 200, 80),
    (160, 100, 50), (50, 100, 160), (160, 50, 100), (100, 160, 50),
]


def _colour_for(class_id: int) -> tuple[int, int, int]:
    return _COLOURS[int(class_id) % len(_COLOURS)]


def _save_detection_outputs(
    image_path: Path,
    image_bgr: np.ndarray,
    detections: list[YOLODetection],
    out_dir: Path,
) -> dict:
    stem = image_path.stem
    h, w = image_bgr.shape[:2]
    metadata: dict = {
        "source": str(image_path.resolve()),
        "shape_hw": [h, w],
        "detections": [],
    }

    overlay = image_bgr.copy()
    for i, det in enumerate(detections):
        colour = _colour_for(det.class_id)

        # Draw mask
        if SAVE_MASKS:
            mask_path = out_dir / f"{stem}_mask_{i:02d}_{det.class_name}.png"
            cv2.imwrite(str(mask_path), (det.mask.astype(np.uint8) * 255))

        # Overlay on BGR image
        if SAVE_OVERLAYS:
            colour_layer = np.zeros_like(image_bgr)
            colour_layer[det.mask] = colour
            overlay = cv2.addWeighted(overlay, 1.0, colour_layer, 0.4, 0)
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(overlay, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

        cx, cy = float(det.centroid_px[0]), float(det.centroid_px[1])
        metadata["detections"].append({
            "index": i,
            "class_id": det.class_id,
            "class_name": det.class_name,
            "confidence": round(float(det.confidence), 4),
            "mask_area_px": det.mask_area,
            "bbox": [round(float(v), 2) for v in det.bbox],
            "centroid_px": [round(cx, 2), round(cy, 2)],
            "major_axis_length_px": round(det.major_axis_length_px, 2),
            "minor_axis_length_px": round(det.minor_axis_length_px, 2),
            "major_axis_angle_deg": round(det.major_axis_angle_deg, 2),
            "minor_axis_angle_deg": round(det.minor_axis_angle_deg, 2),
        })

    if SAVE_OVERLAYS:
        overlay_path = out_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

    return metadata


def main() -> int:
    image_dir = Path(IMAGE_DIR)
    out_dir   = Path(OUTPUT_DIR)

    if not image_dir.exists():
        print(f"[ERROR] IMAGE_DIR does not exist: {image_dir.resolve()}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        print(f"[ERROR] No images found in {image_dir.resolve()}")
        return 1

    images = images[START_INDEX:]
    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    print(f"[BATCH] {len(images)} image(s) to process")

    device_info = select_torch_device(print_info=True, use_cuda=USE_CUDA)
    yolo = YOLOSegmenter(
        weights_path=YOLO_WEIGHTS_PATH,
        device_info=device_info,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        warmup_enabled=True,
        warmup_iters=2,
    )
    yolo.warmup()

    processed = 0
    skipped = 0
    errors = 0
    all_metadata: list[dict] = []

    for img_path in images:
        json_path = out_dir / f"{img_path.stem}_metadata.json"
        if not OVERWRITE_EXISTING and json_path.exists():
            print(f"[SKIP] {img_path.name}")
            skipped += 1
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Could not load image: {img_path}")
            errors += 1
            continue

        try:
            detections = yolo.segment(image_bgr)
        except Exception as exc:
            print(f"[ERROR] YOLO failed for {img_path.name}: {exc}")
            errors += 1
            continue

        metadata = _save_detection_outputs(img_path, image_bgr, detections, out_dir)

        if SAVE_METADATA_JSON:
            with open(str(json_path), "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)

        all_metadata.append(metadata)
        processed += 1
        print(f"[OK] {img_path.name}  detections={len(detections)}")

    # Summary JSON
    summary_path = out_dir / "_batch_summary.json"
    with open(str(summary_path), "w", encoding="utf-8") as fh:
        json.dump(
            {"processed": processed, "skipped": skipped, "errors": errors,
             "images": all_metadata},
            fh, indent=2,
        )

    print(f"\n[BATCH] Done. processed={processed}, skipped={skipped}, errors={errors}")
    print(f"[BATCH] Summary: {summary_path.resolve()}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
