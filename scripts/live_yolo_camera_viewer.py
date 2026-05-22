from __future__ import annotations

"""
scripts/live_yolo_camera_viewer.py

Live overhead + stereo camera preview with on-demand YOLO segmentation.

No RAFT.  No point cloud.  No robot connection.  No Teensy.

Keys
----
SPACE  — freeze current frames and run YOLO detection
c      — toggle continuous YOLO mode
s      — save current freeze/overlay/detections
r      — resume live preview (discard freeze)
q/ESC  — quit

Run:
    python scripts/live_yolo_camera_viewer.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

DETECT_OVERHEAD    = True
DETECT_STEREO_LEFT = True
DETECT_STEREO_RIGHT = False

CONTINUOUS_YOLO            = False   # False = press SPACE to run detections
RUN_YOLO_ON_STARTUP_FRAME  = False

YOLO_WEIGHTS_PATH          = Path("yolo_weights/Validate_Only_100_Training_Best.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("best.pt")

YOLO_IMGSZ       = 640
YOLO_CONF        = 0.35
YOLO_IOU         = 0.50
YOLO_RETINA_MASKS = True

# Empty list = detect all classes.
TARGET_CLASS_NAMES: list[str] = []

USE_CUDA = True
USE_HALF = True

WINDOW_NAME   = "Live YOLO Camera Viewer"
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 900

SAVE_FREEZE_FRAMES = True
SAVE_DIR = Path("outputs/live_yolo_freezes")

# ============================================================
# End of user settings
# ============================================================

import sys
import time
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from config.camera_config import OVERHEAD_INDEX, STEREO_INDEX
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera
from vision.torch_device import select_torch_device
from vision.yolo_segmenter import YOLOSegmenter, YOLODetection

# Local helpers (in same scripts/ folder)
_SCRIPTS = _Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from vision_debug_helpers import (
    draw_yolo_detections,
    compose_debug_grid,
    save_json,
    timestamped_output_dir,
    detections_to_json,
)


# ------------------------------------------------------------------ #
# Grid layout helpers
# ------------------------------------------------------------------ #

def _build_display(
    overhead_draw: np.ndarray | None,
    stereo_left_draw: np.ndarray | None,
    stereo_right_draw: np.ndarray | None,
    status_text: str,
) -> np.ndarray:
    tiles = []
    if overhead_draw is not None:
        tiles.append(("Overhead", overhead_draw))
    if stereo_left_draw is not None:
        tiles.append(("Stereo Left", stereo_left_draw))
    if stereo_right_draw is not None:
        tiles.append(("Stereo Right (raw)", stereo_right_draw))

    if not tiles:
        canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        cv2.putText(canvas, "No frames", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        return canvas

    n = len(tiles)
    cols = min(n, 2)
    cell_w = DISPLAY_WIDTH // cols
    cell_h = DISPLAY_HEIGHT // max(1, int(np.ceil(n / cols)))

    grid = compose_debug_grid(tiles, cell_w=cell_w, cell_h=cell_h, cols=cols)

    # Stamp status line at bottom
    h = grid.shape[0]
    cv2.putText(
        grid,
        status_text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 220, 180),
        1,
        cv2.LINE_AA,
    )
    return grid


# ------------------------------------------------------------------ #
# Save helpers
# ------------------------------------------------------------------ #

def _save_freeze(
    out_dir: _Path,
    overhead_raw: np.ndarray | None,
    stereo_full: np.ndarray | None,
    stereo_left: np.ndarray | None,
    stereo_right: np.ndarray | None,
    overhead_detections: list[YOLODetection],
    stereo_left_detections: list[YOLODetection],
    stereo_right_detections: list[YOLODetection],
) -> None:
    def _save(name: str, img: np.ndarray | None) -> None:
        if img is not None:
            p = out_dir / name
            cv2.imwrite(str(p), img)
            print(f"  [SAVE] {p}")

    _save("overhead_raw.jpg", overhead_raw)
    _save("stereo_full.jpg", stereo_full)
    _save("stereo_left.jpg", stereo_left)
    _save("stereo_right.jpg", stereo_right)

    if overhead_raw is not None and overhead_detections:
        overlay = draw_yolo_detections(overhead_raw, overhead_detections, numbered=True)
        _save("overhead_yolo_overlay.jpg", overlay)

    if stereo_left is not None and stereo_left_detections:
        overlay = draw_yolo_detections(stereo_left, stereo_left_detections, numbered=True)
        _save("stereo_left_yolo_overlay.jpg", overlay)

    if stereo_right is not None and stereo_right_detections:
        overlay = draw_yolo_detections(stereo_right, stereo_right_detections, numbered=True)
        _save("stereo_right_yolo_overlay.jpg", overlay)

    all_dets: dict[str, list] = {}
    if overhead_detections:
        all_dets["overhead"] = detections_to_json(overhead_detections)
    if stereo_left_detections:
        all_dets["stereo_left"] = detections_to_json(stereo_left_detections)
    if stereo_right_detections:
        all_dets["stereo_right"] = detections_to_json(stereo_right_detections)

    if all_dets:
        save_json(out_dir / "detections.json", all_dets)
        print(f"  [SAVE] {out_dir / 'detections.json'}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> int:
    print("[INIT] Loading YOLO…")
    device_info = select_torch_device(
        print_info=True, use_cuda=USE_CUDA, use_half=USE_HALF
    )
    yolo = YOLOSegmenter(
        weights_path=YOLO_WEIGHTS_PATH,
        device_info=device_info,
        fallback_weights_path=YOLO_FALLBACK_WEIGHTS_PATH,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        retina_masks=YOLO_RETINA_MASKS,
        use_half=USE_HALF,
        target_class_names=TARGET_CLASS_NAMES if TARGET_CLASS_NAMES else None,
        warmup_enabled=True,
        warmup_iters=2,
    )
    yolo.warmup()

    print("[INIT] Opening cameras…")
    overhead = SimpleOverheadCamera(index=OVERHEAD_INDEX)
    stereo   = SimpleStereoCamera(index=STEREO_INDEX)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    # State
    frozen         = False
    continuous     = bool(CONTINUOUS_YOLO)

    overhead_raw:   np.ndarray | None = None
    stereo_full:    np.ndarray | None = None
    stereo_left:    np.ndarray | None = None
    stereo_right:   np.ndarray | None = None

    overhead_draw:         np.ndarray | None = None
    stereo_left_draw:      np.ndarray | None = None
    stereo_right_draw:     np.ndarray | None = None

    overhead_dets:       list[YOLODetection] = []
    stereo_left_dets:    list[YOLODetection] = []
    stereo_right_dets:   list[YOLODetection] = []

    def _read_frames():
        nonlocal overhead_raw, stereo_full, stereo_left, stereo_right
        nonlocal overhead_draw, stereo_left_draw, stereo_right_draw
        nonlocal overhead_dets, stereo_left_dets, stereo_right_dets

        ok_o, frame_o = overhead.read()
        if ok_o and frame_o is not None:
            overhead_raw  = frame_o.copy()
            overhead_draw = overhead_raw.copy()

        ok_s, sfull, sleft, sright = stereo.read_pair()
        if ok_s and sfull is not None:
            stereo_full   = sfull.copy()
            stereo_left   = sleft.copy()
            stereo_right  = sright.copy()
            stereo_left_draw  = stereo_left.copy()
            stereo_right_draw = stereo_right.copy()

        overhead_dets     = []
        stereo_left_dets  = []
        stereo_right_dets = []

    def _run_yolo():
        nonlocal overhead_draw, stereo_left_draw, stereo_right_draw
        nonlocal overhead_dets, stereo_left_dets, stereo_right_dets

        images_to_run: list[np.ndarray] = []
        labels: list[str] = []

        if DETECT_OVERHEAD and overhead_raw is not None:
            images_to_run.append(overhead_raw)
            labels.append("overhead")
        if DETECT_STEREO_LEFT and stereo_left is not None:
            images_to_run.append(stereo_left)
            labels.append("stereo_left")
        if DETECT_STEREO_RIGHT and stereo_right is not None:
            images_to_run.append(stereo_right)
            labels.append("stereo_right")

        if not images_to_run:
            return

        t0 = time.perf_counter()
        results_by_frame = yolo.segment_batch(images_to_run)
        dt = time.perf_counter() - t0
        print(f"[YOLO] {sum(len(r) for r in results_by_frame)} detection(s) in {dt:.3f}s")

        for label, dets in zip(labels, results_by_frame):
            if label == "overhead":
                overhead_dets = dets
                if overhead_raw is not None:
                    overhead_draw = draw_yolo_detections(overhead_raw, dets, numbered=True)
                    for i, det in enumerate(dets):
                        print(f"  [Overhead #{i+1}] {det.class_name} conf={det.confidence:.2f} area={det.mask_area}")
            elif label == "stereo_left":
                stereo_left_dets = dets
                if stereo_left is not None:
                    stereo_left_draw = draw_yolo_detections(stereo_left, dets, numbered=True)
                    for i, det in enumerate(dets):
                        print(f"  [StereoLeft #{i+1}] {det.class_name} conf={det.confidence:.2f} area={det.mask_area}")
            elif label == "stereo_right":
                stereo_right_dets = dets
                if stereo_right is not None:
                    stereo_right_draw = draw_yolo_detections(stereo_right, dets, numbered=True)

    # Initial frame
    _read_frames()
    if RUN_YOLO_ON_STARTUP_FRAME:
        _run_yolo()

    print("\n[READY]")
    print("  SPACE — freeze + run YOLO")
    print("  c     — toggle continuous YOLO")
    print("  s     — save freeze/overlay")
    print("  r     — resume live")
    print("  q/ESC — quit\n")

    while True:
        if not frozen:
            _read_frames()
            if continuous:
                _run_yolo()

        mode_tag = "FROZEN" if frozen else ("LIVE-YOLO" if continuous else "LIVE")
        dets_tag = (
            f"O:{len(overhead_dets)} SL:{len(stereo_left_dets)} SR:{len(stereo_right_dets)}"
        )
        status = f"{mode_tag}  dets: {dets_tag}  [SPACE=freeze  c=toggle  s=save  r=resume  q=quit]"

        grid = _build_display(overhead_draw, stereo_left_draw, stereo_right_draw, status)
        cv2.imshow(WINDOW_NAME, grid)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):   # q or ESC
            break

        elif key == ord(" "):       # SPACE — freeze + detect
            if not frozen:
                print("[FREEZE] Freezing frames…")
                frozen = True
            _run_yolo()

        elif key == ord("c"):       # toggle continuous
            continuous = not continuous
            if continuous:
                frozen = False
            print(f"[MODE] Continuous YOLO = {continuous}")

        elif key == ord("r"):       # resume
            frozen = False
            continuous = False
            overhead_dets = []
            stereo_left_dets = []
            stereo_right_dets = []
            print("[RESUME] Live preview")

        elif key == ord("s"):       # save
            if stereo_full is None and overhead_raw is None:
                print("[SAVE] Nothing to save yet.")
            else:
                out_dir = timestamped_output_dir(SAVE_DIR, prefix="freeze")
                print(f"[SAVE] Writing to {out_dir}")
                _save_freeze(
                    out_dir,
                    overhead_raw,
                    stereo_full,
                    stereo_left,
                    stereo_right,
                    overhead_dets,
                    stereo_left_dets,
                    stereo_right_dets,
                )

    cv2.destroyAllWindows()
    overhead.release()
    stereo.release()
    print("[EXIT] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
