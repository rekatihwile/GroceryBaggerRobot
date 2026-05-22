from __future__ import annotations

"""
scripts/stepwise_yolo_raft_pointcloud_validation.py

Live-camera, step-by-step validation of the full vision pipeline:
  Freeze → Rectify → YOLO → RAFT disparity → Masked point cloud → Robot-frame target estimate.

No robot commands.  No Teensy.  Pure vision validation.

Keyqs
----
SPACE — capture / freeze stereo pair from live stream
y     — run YOLO on frozen rectified-left image
1-9   — select detected object (by 1-based index)
d     — run RAFT disparity on frozen rect-left / rect-right
p     — build masked point cloud for selected object
m     — map point cloud to robot frame and print estimate
a     — run all steps from current freeze (y → 1 → d → p → m)
s     — save all outputs for current run
r     — resume live preview (discard freeze / results)
q/ESC — quit

Run:
    python scripts/stepwise_yolo_raft_pointcloud_validation.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

USE_OVERHEAD = True
LIVE_CAMERA  = True

STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")
BUNDLE_PATH             = Path("robot_calibration_bundle.npz")

YOLO_WEIGHTS_PATH          = Path("yolo_weights/Validate_Only_100_Training_Best.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("best.pt")

RAFT_ROOT            = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")

YOLO_IMGSZ       = 640
YOLO_CONF        = 0.35
YOLO_IOU         = 0.50
YOLO_RETINA_MASKS = True

TARGET_CLASS_NAMES: list[str] = []   # empty = all classes

USE_CUDA = True
USE_HALF = True

RAFT_VALID_ITERS     = 16
RAFT_DOWNSCALE       = 1.0
RAFT_MIXED_PRECISION = True

OBJECT_TARGET_MODE    = "centroid"   # "centroid" | "top_surface"
MIN_MASK_AREA_PX      = 500
MIN_DISPARITY_PX      = 1.0
MIN_VALID_OBJECT_POINTS = 300
POINTCLOUD_MAX_POINTS   = 20000

SAVE_VALIDATION_OUTPUTS = True
OUTPUT_DIR = Path("outputs/stepwise_yolo_raft_pointcloud_validation")

SHOW_MATPLOTLIB_POINTCLOUD = True
SHOW_OPENCV_WINDOWS        = True
DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 900

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
from vision.raft_runner import RAFTStereoRunner
from vision.stereo_rectifier import StereoRectifier
from vision.pointcloud import masked_disparity_to_pointcloud, cam_points_to_robot_xyz, cam_xyz_to_robot_xyz

_SCRIPTS = _Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from vision_debug_helpers import (
    draw_yolo_detections,
    make_disparity_preview,
    make_masked_disparity_preview,
    save_json,
    save_disparity_heatmap,
    compose_debug_grid,
    timestamped_output_dir,
    detections_to_json,
)


# ------------------------------------------------------------------ #
# Calibration helpers
# ------------------------------------------------------------------ #

def _load_stereo_calibration(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stereo calibration: {path}")
    data = np.load(str(path), allow_pickle=False)
    calib = {k: np.asarray(data[k]) for k in data.files}
    print(f"[CALIB] Stereo calibration loaded from {path}")
    return calib


def _load_bundle(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        print(f"[BUNDLE] WARNING — bundle not found at {path}. Robot-frame mapping disabled.")
        return {}
    data = np.load(str(path), allow_pickle=False)
    bundle = dict(data)
    print(f"[BUNDLE] Robot calibration bundle loaded from {path}")
    return bundle


# ------------------------------------------------------------------ #
# Pipeline state container
# ------------------------------------------------------------------ #

class _State:
    frozen:       bool = False
    stereo_full:  np.ndarray | None = None
    stereo_left:  np.ndarray | None = None
    stereo_right: np.ndarray | None = None
    rect_left:    np.ndarray | None = None
    rect_right:   np.ndarray | None = None
    overhead_raw: np.ndarray | None = None

    detections:   list[YOLODetection] | None = None
    selected_idx: int = 0            # 0-based index into detections

    disparity:    np.ndarray | None = None

    points_cam:   np.ndarray | None = None   # Nx3 cam-mm
    points_uv:    np.ndarray | None = None   # Nx2 pixel coords
    robot_xyz:    np.ndarray | None = None   # 3-vector mm

    run_dir:      Path | None = None


# ------------------------------------------------------------------ #
# Display helpers
# ------------------------------------------------------------------ #

def _status_text(state: _State) -> str:
    parts = []
    if state.frozen:
        parts.append("FROZEN")
        if state.detections is not None:
            parts.append(f"y=YOLO({len(state.detections)}dets)")
        else:
            parts.append("y=YOLO?")
        if state.detections:
            parts.append(f"sel={state.selected_idx + 1}/{len(state.detections)}")
        if state.disparity is not None:
            parts.append("d=disp✓")
        if state.points_cam is not None:
            parts.append(f"p=pts({len(state.points_cam)})")
        if state.robot_xyz is not None:
            rx, ry, rz = state.robot_xyz
            parts.append(f"m=({rx:.0f},{ry:.0f},{rz:.0f})mm")
    else:
        parts.append("LIVE")
    parts.append("[SPACE=freeze  y=YOLO  1-9=sel  d=RAFT  p=pts  m=robot  a=all  s=save  r=live  q=quit]")
    return "  ".join(parts)


def _build_display(state: _State) -> np.ndarray:
    tiles: list[tuple[str, np.ndarray | None]] = []

    if state.frozen and state.rect_left is not None:
        if state.detections is not None:
            det_img = draw_yolo_detections(state.rect_left, state.detections, numbered=True)
            # Highlight selected
            if state.detections and 0 <= state.selected_idx < len(state.detections):
                det = state.detections[state.selected_idx]
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            tiles.append(("Rect Left + YOLO", det_img))
        else:
            tiles.append(("Rect Left", state.rect_left.copy()))
    elif state.stereo_left is not None:
        tiles.append(("Live Left", state.stereo_left.copy()))

    if state.frozen and state.rect_right is not None:
        tiles.append(("Rect Right", state.rect_right.copy()))
    elif state.stereo_right is not None:
        tiles.append(("Live Right", state.stereo_right.copy()))

    if state.disparity is not None:
        disp_preview = make_disparity_preview(state.disparity, min_disparity=MIN_DISPARITY_PX)
        if (
            state.detections
            and 0 <= state.selected_idx < len(state.detections)
        ):
            sel_mask = state.detections[state.selected_idx].mask
            masked_preview = make_masked_disparity_preview(
                state.disparity, sel_mask, min_disparity=MIN_DISPARITY_PX
            )
            tiles.append(("Disparity (full)", disp_preview))
            tiles.append(("Disparity (masked)", masked_preview))
        else:
            tiles.append(("Disparity", disp_preview))

    if state.overhead_raw is not None and USE_OVERHEAD:
        tiles.append(("Overhead", state.overhead_raw.copy()))

    if not tiles:
        canvas = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        cv2.putText(canvas, "No frames yet.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        return canvas

    n = len(tiles)
    cols = min(n, 2)
    cell_w = DISPLAY_WIDTH // cols
    cell_h = DISPLAY_HEIGHT // max(1, int(np.ceil(n / cols)))
    grid = compose_debug_grid(tiles, cell_w=cell_w, cell_h=cell_h, cols=cols)

    h = grid.shape[0]
    cv2.putText(grid, _status_text(state), (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 180), 1, cv2.LINE_AA)
    return grid


# ------------------------------------------------------------------ #
# Save helpers
# ------------------------------------------------------------------ #

def _make_run_dir() -> Path:
    d = timestamped_output_dir(OUTPUT_DIR, prefix="run")
    print(f"[SAVE] Output dir: {d}")
    return d


def _save_all(state: _State) -> None:
    if not SAVE_VALIDATION_OUTPUTS:
        print("[SAVE] SAVE_VALIDATION_OUTPUTS=False — skipping.")
        return
    if state.run_dir is None:
        state.run_dir = _make_run_dir()

    out = state.run_dir

    def _img(name: str, img: np.ndarray | None) -> None:
        if img is not None:
            p = out / name
            cv2.imwrite(str(p), img)
            print(f"  [SAVE] {p.name}")

    _img("stereo_full.jpg",  state.stereo_full)
    _img("stereo_left.jpg",  state.stereo_left)
    _img("stereo_right.jpg", state.stereo_right)
    _img("rect_left.jpg",    state.rect_left)
    _img("rect_right.jpg",   state.rect_right)
    _img("overhead_raw.jpg", state.overhead_raw)

    if state.detections is not None and state.rect_left is not None:
        overlay = draw_yolo_detections(state.rect_left, state.detections, numbered=True)
        _img("yolo_overlay.jpg", overlay)
        save_json(out / "detections.json", detections_to_json(state.detections))
        print(f"  [SAVE] detections.json")

    if state.disparity is not None:
        np.save(str(out / "disparity_raw.npy"), state.disparity)
        print(f"  [SAVE] disparity_raw.npy")
        save_disparity_heatmap(out / "disparity_preview.png", state.disparity)
        print(f"  [SAVE] disparity_preview.png")
        if (
            state.detections
            and 0 <= state.selected_idx < len(state.detections)
        ):
            sel_mask = state.detections[state.selected_idx].mask
            masked_prev = make_masked_disparity_preview(
                state.disparity, sel_mask, min_disparity=MIN_DISPARITY_PX
            )
            _img("disparity_masked_preview.png", masked_prev)

    if state.points_cam is not None:
        np.save(str(out / "pointcloud_cam_mm.npy"), state.points_cam)
        print(f"  [SAVE] pointcloud_cam_mm.npy ({len(state.points_cam)} pts)")

    if state.detections and 0 <= state.selected_idx < len(state.detections):
        sel = state.detections[state.selected_idx]
        target_data: dict = {
            "selected_object_index": state.selected_idx + 1,
            "class_name": sel.class_name,
            "class_id": int(sel.class_id),
            "confidence": round(float(sel.confidence), 4),
            "mask_area_px": int(sel.mask_area),
            "valid_point_count": len(state.points_cam) if state.points_cam is not None else 0,
        }
        if state.robot_xyz is not None:
            target_data["robot_xyz_mm"] = [round(float(v), 2) for v in state.robot_xyz]
        save_json(out / "target_estimate.json", target_data)
        print(f"  [SAVE] target_estimate.json")

    print(f"[SAVE] Done — {out}")


# ------------------------------------------------------------------ #
# Matplotlib point cloud view
# ------------------------------------------------------------------ #

def _show_pointcloud_topdown(points_cam: np.ndarray, class_name: str) -> None:
    if not SHOW_MATPLOTLIB_POINTCLOUD:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLT] matplotlib not available — skipping point-cloud plot.")
        return

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(x, y, c=z, cmap="viridis", s=1, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Z cam (mm)")
    ax.set_xlabel("X cam (mm)")
    ax.set_ylabel("Y cam (mm)")
    ax.set_title(f"Top-down point cloud — {class_name} ({len(points_cam)} pts)")
    ax.set_aspect("equal", "box")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.05)
    print("[PLT] Point-cloud plot shown (non-blocking).")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> int:
    print("[INIT] Loading stereo calibration and bundle…")
    stereo_calib = _load_stereo_calibration(STEREO_CALIBRATION_PATH)
    bundle = _load_bundle(BUNDLE_PATH)
    rectifier = StereoRectifier(stereo_calib)

    print("[INIT] Loading YOLO…")
    device_info = select_torch_device(print_info=True, use_cuda=USE_CUDA, use_half=USE_HALF)
    yolo = YOLOSegmenter(
        weights_path=YOLO_WEIGHTS_PATH,
        device_info=device_info,
        fallback_weights_path=YOLO_FALLBACK_WEIGHTS_PATH,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        retina_masks=YOLO_RETINA_MASKS,
        use_half=USE_HALF,
        min_mask_area_px=MIN_MASK_AREA_PX,
        target_class_names=TARGET_CLASS_NAMES if TARGET_CLASS_NAMES else None,
        warmup_enabled=True,
        warmup_iters=2,
    )
    yolo.warmup()

    print("[INIT] Loading RAFT…")
    raft = RAFTStereoRunner(
        raft_root=RAFT_ROOT,
        checkpoint_path=RAFT_CHECKPOINT_PATH,
        device_info=device_info,
        valid_iters=RAFT_VALID_ITERS,
        downscale=RAFT_DOWNSCALE,
        mixed_precision=RAFT_MIXED_PRECISION,
        warmup=True,
    )

    print("[INIT] Opening cameras…")
    overhead = SimpleOverheadCamera(index=OVERHEAD_INDEX) if USE_OVERHEAD else None
    stereo   = SimpleStereoCamera(index=STEREO_INDEX)

    if SHOW_OPENCV_WINDOWS:
        cv2.namedWindow("Stepwise Pipeline", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Stepwise Pipeline", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    state = _State()

    print("\n[READY]")
    print("  SPACE — capture/freeze stereo pair")
    print("  y     — YOLO on rect-left")
    print("  1-9   — select detected object")
    print("  d     — RAFT disparity")
    print("  p     — build masked point cloud")
    print("  m     — map to robot frame")
    print("  a     — run all steps from freeze")
    print("  s     — save outputs")
    print("  r     — resume live")
    print("  q/ESC — quit\n")

    def _read_live():
        if overhead is not None:
            ok_o, frame_o = overhead.read()
            if ok_o and frame_o is not None:
                state.overhead_raw = frame_o.copy()

        ok_s, sfull, sleft, sright = stereo.read_pair()
        if ok_s and sfull is not None:
            state.stereo_full  = sfull.copy()
            state.stereo_left  = sleft.copy()
            state.stereo_right = sright.copy()

    def _do_freeze():
        print("[FREEZE] Capturing stereo pair…")
        if overhead is not None:
            ok_o, frame_o = overhead.read()
            if ok_o and frame_o is not None:
                state.overhead_raw = frame_o.copy()

        ok_s, sfull, sleft, sright = stereo.read_pair()
        if not ok_s or sfull is None:
            print("[FREEZE] ERROR: could not read stereo pair.")
            return

        state.stereo_full  = sfull.copy()
        state.stereo_left  = sleft.copy()
        state.stereo_right = sright.copy()

        print("[RECTIFY] Rectifying stereo pair…")
        state.rect_left, state.rect_right = rectifier.rectify(sleft, sright)
        print(f"[RECTIFY] rect_left={state.rect_left.shape}, rect_right={state.rect_right.shape}")

        state.frozen = True
        state.detections  = None
        state.disparity   = None
        state.points_cam  = None
        state.points_uv   = None
        state.robot_xyz   = None
        state.run_dir     = None

    def _do_yolo():
        if not state.frozen or state.rect_left is None:
            print("[YOLO] Not frozen yet — press SPACE first.")
            return
        print("[YOLO] Running on rect-left…")
        t0 = time.perf_counter()
        state.detections = yolo.segment(state.rect_left)
        dt = time.perf_counter() - t0
        print(f"[YOLO] {len(state.detections)} detection(s) in {dt:.3f}s")
        for i, det in enumerate(state.detections):
            print(
                f"  [{i+1}] {det.class_name} conf={det.confidence:.2f} "
                f"area={det.mask_area} "
                f"major_axis={det.major_axis_length_px:.1f}px / {det.major_axis_angle_deg:.1f}°"
            )
        if state.detections:
            state.selected_idx = 0
            print(f"[YOLO] Auto-selected object 1 = {state.detections[0].class_name}")

    def _do_raft():
        if not state.frozen or state.rect_left is None or state.rect_right is None:
            print("[RAFT] Not frozen yet — press SPACE first.")
            return
        print("[RAFT] Running RAFT disparity…")
        t0 = time.perf_counter()
        state.disparity = raft.predict_disparity(
            state.rect_left, state.rect_right, color="BGR"
        )
        dt = time.perf_counter() - t0
        valid = np.isfinite(state.disparity) & (state.disparity > MIN_DISPARITY_PX)
        print(
            f"[RAFT] Done in {dt:.3f}s  "
            f"disp range=[{np.nanmin(state.disparity):.1f}, {np.nanmax(state.disparity):.1f}]  "
            f"valid={np.sum(valid)}/{state.disparity.size}"
        )

    def _do_pointcloud():
        if state.disparity is None:
            print("[PC] Run RAFT first (press d).")
            return
        if not state.detections:
            print("[PC] Run YOLO first (press y).")
            return
        if not (0 <= state.selected_idx < len(state.detections)):
            print("[PC] Select an object first (press 1-9).")
            return

        det = state.detections[state.selected_idx]
        print(f"[PC] Building point cloud for object {state.selected_idx+1}: {det.class_name}…")
        pts_cam, uv_px = masked_disparity_to_pointcloud(
            mask=det.mask,
            disparity=state.disparity,
            stereo_calib=stereo_calib,
        )
        print(f"[PC] {len(pts_cam)} valid points (min={MIN_VALID_OBJECT_POINTS})")

        if len(pts_cam) < MIN_VALID_OBJECT_POINTS:
            print(f"[PC] WARNING: fewer than {MIN_VALID_OBJECT_POINTS} valid points — results may be noisy.")

        state.points_cam = pts_cam
        state.points_uv  = uv_px

        if SHOW_MATPLOTLIB_POINTCLOUD and len(pts_cam) > 0:
            _show_pointcloud_topdown(pts_cam, det.class_name)

    def _do_robot_map():
        if state.points_cam is None or len(state.points_cam) == 0:
            print("[ROBOT] Build point cloud first (press p).")
            return
        if not bundle:
            print("[ROBOT] No robot calibration bundle loaded — cannot map to robot frame.")
            return
        if not state.detections or not (0 <= state.selected_idx < len(state.detections)):
            print("[ROBOT] No object selected.")
            return

        det = state.detections[state.selected_idx]

        # Compute centroid and top-surface point in camera frame
        if OBJECT_TARGET_MODE == "top_surface":
            from vision.pointcloud import TOP_SURFACE_PERCENTILE
            z_thresh = float(np.percentile(state.points_cam[:, 2], 100 - TOP_SURFACE_PERCENTILE))
            top_mask = state.points_cam[:, 2] <= z_thresh
            top_pts  = state.points_cam[top_mask]
            centroid_cam = np.median(top_pts, axis=0) if len(top_pts) > 0 else np.mean(state.points_cam, axis=0)
        else:
            centroid_cam = np.mean(state.points_cam, axis=0)

        # Map centroid to robot frame
        try:
            robot_xyz = cam_xyz_to_robot_xyz(centroid_cam, bundle)
            state.robot_xyz = robot_xyz
        except Exception as exc:
            print(f"[ROBOT] cam_xyz_to_robot_xyz failed: {exc}")
            # Fallback: try cam_points_to_robot_xyz
            try:
                robot_pts = cam_points_to_robot_xyz(state.points_cam, bundle)
                state.robot_xyz = np.mean(robot_pts, axis=0)
            except Exception as exc2:
                print(f"[ROBOT] cam_points_to_robot_xyz also failed: {exc2}")
                return

        print("\n" + "=" * 50)
        print(f"  Object        : {det.class_name}  (conf={det.confidence:.2f})")
        print(f"  mask_area     : {det.mask_area} px²")
        print(f"  valid points  : {len(state.points_cam)}")
        print(f"  centroid_cam  : ({centroid_cam[0]:.1f}, {centroid_cam[1]:.1f}, {centroid_cam[2]:.1f}) mm")
        print(f"  robot_xyz     : ({state.robot_xyz[0]:.1f}, {state.robot_xyz[1]:.1f}, {state.robot_xyz[2]:.1f}) mm")
        print(f"  major_axis    : {det.major_axis_length_px:.1f} px @ {det.major_axis_angle_deg:.1f}°")
        print(f"  minor_axis    : {det.minor_axis_length_px:.1f} px @ {det.minor_axis_angle_deg:.1f}°")
        print("=" * 50 + "\n")

    def _do_all():
        _do_yolo()
        _do_raft()
        _do_pointcloud()
        _do_robot_map()

    # Main loop
    while True:
        if not state.frozen:
            _read_live()

        if SHOW_OPENCV_WINDOWS:
            grid = _build_display(state)
            cv2.imshow("Stepwise Pipeline", grid)
            key = cv2.waitKey(30) & 0xFF
        else:
            key = ord("q")  # headless — exit immediately (shouldn't happen)

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            _do_freeze()
        elif key == ord("y"):
            _do_yolo()
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if state.detections and 0 <= idx < len(state.detections):
                state.selected_idx = idx
                print(f"[SELECT] Object {idx+1}: {state.detections[idx].class_name}")
            elif state.detections:
                print(f"[SELECT] Index {idx+1} out of range (have {len(state.detections)} dets)")
            else:
                print("[SELECT] No detections yet — press y first.")
        elif key == ord("d"):
            _do_raft()
        elif key == ord("p"):
            _do_pointcloud()
        elif key == ord("m"):
            _do_robot_map()
        elif key == ord("a"):
            _do_all()
        elif key == ord("s"):
            _save_all(state)
        elif key == ord("r"):
            state.frozen      = False
            state.detections  = None
            state.disparity   = None
            state.points_cam  = None
            state.points_uv   = None
            state.robot_xyz   = None
            state.run_dir     = None
            print("[RESUME] Live preview.")

    if SHOW_OPENCV_WINDOWS:
        cv2.destroyAllWindows()
    if overhead is not None:
        overhead.release()
    stereo.release()
    print("[EXIT] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
