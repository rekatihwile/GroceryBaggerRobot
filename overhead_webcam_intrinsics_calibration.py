from __future__ import annotations

"""
overhead_webcam_intrinsics_calibration.py

Live overhead-webcam intrinsic calibration for the grocery bagger.

What it does:
  - Opens the overhead webcam.
  - Detects a printed checkerboard live.
  - Only keeps frames where the checkerboard is detected.
  - Stores calibration data in memory only, not raw photos.
  - Calibrates intrinsics when requested.
  - Shows/prints RMS, camera matrix, distortion coefficients, and per-view errors.
  - Saves overhead_intrinsics.npz only after explicit confirmation.
  - Deletes any optional debug image folder on exit unless KEEP_DEBUG_IMAGES=True.

Controls:
  SPACE  keep current valid checkerboard frame
  c      calibrate using kept frames
  u      toggle undistorted live preview after calibration
  d      delete last kept frame
  r      reset all kept frames
  s      save overhead_intrinsics.npz after calibration
  q/ESC  quit

Important:
  OpenCV wants INNER CORNERS, not number of checker squares.
  If your printed board is "8 x 11 inner corners", leave BOARD_INNER_CORNERS=(8, 11).
  If detection fails constantly, try BOARD_INNER_CORNERS=(7, 10) or swap to (11, 8).
"""

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_config import (
    OVERHEAD_FOURCC as FOURCC,
    OVERHEAD_FPS as FPS,
    OVERHEAD_HEIGHT as FRAME_HEIGHT,
    OVERHEAD_INDEX as CAMERA_INDEX,
    OVERHEAD_INTRINSICS_PATH as OUT_PATH,
    OVERHEAD_WIDTH as FRAME_WIDTH,
)
from overhead_camera import SimpleOverheadCamera


# ============================================================
# USER SETTINGS
# ============================================================

# OpenCV checkerboard size = number of INNER CORNERS per row/column.
# Try (11, 8) if (8, 11) does not detect your board orientation reliably.
BOARD_INNER_CORNERS = (10, 7)

SQUARE_SIZE_MM = 15.0

MIN_VALID_VIEWS_FOR_CALIBRATION = 10
RECOMMENDED_VALID_VIEWS = 20

# Raw photos are NOT saved by default.
SAVE_DEBUG_IMAGES = False
KEEP_DEBUG_IMAGES = False
DEBUG_DIR = Path("_tmp_overhead_calibration_debug")

WINDOW = "Overhead Webcam Intrinsics Calibration"


# ============================================================
# DATA
# ============================================================

@dataclass
class KeptView:
    index: int
    t: float
    corners: np.ndarray          # shape (N,1,2), float32
    image_size: tuple[int, int]  # (w,h)
    sharpness: float


# ============================================================
# HELPERS
# ============================================================

def open_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def make_object_points() -> np.ndarray:
    cols, rows = BOARD_INNER_CORNERS
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(SQUARE_SIZE_MM)
    return objp


def sharpness_metric(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_checkerboard(frame_bgr: np.ndarray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # findChessboardCornersSB is usually better on real webcam images.
    flags_sb = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    ok, corners = cv2.findChessboardCornersSB(gray, BOARD_INNER_CORNERS, flags_sb)

    # Fallback to classic detector if SB fails.
    if not ok:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        ok, corners = cv2.findChessboardCorners(gray, BOARD_INNER_CORNERS, flags)
        if ok:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)

    return ok, corners, gray


def draw_text_lines(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_coverage(frame: np.ndarray, kept: list[KeptView]) -> np.ndarray:
    if not kept:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    for view in kept:
        pts = view.corners.reshape(-1, 2)
        center = pts.mean(axis=0)
        x, y = int(round(center[0])), int(round(center[1]))
        cv2.circle(out, (x, y), 4, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (80, 80, 80), 1)
    return out


def calibrate(kept: list[KeptView], objp: np.ndarray):
    if len(kept) < MIN_VALID_VIEWS_FOR_CALIBRATION:
        print(f"[CAL] Need at least {MIN_VALID_VIEWS_FOR_CALIBRATION} valid views. Have {len(kept)}.")
        return None

    image_size = kept[0].image_size
    objpoints = [objp.copy() for _ in kept]
    imgpoints = [v.corners.astype(np.float32) for v in kept]

    # Standard pinhole + radial/tangential distortion.
    flags = 0

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags,
    )

    # Per-view reprojection errors.
    per_view = []
    for i, (obj, img, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        err = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        per_view.append(float(err))

    per_view = np.asarray(per_view, dtype=np.float64)

    print("\n" + "=" * 72)
    print("[CALIBRATION RESULTS]")
    print(f"Views used: {len(kept)}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"OpenCV RMS reprojection error: {rms:.6f} px")
    print(f"Mean per-view error: {per_view.mean():.6f} px")
    print(f"Max per-view error:  {per_view.max():.6f} px")
    print("\nCamera matrix K:")
    print(K)
    print("\nDistortion coefficients:")
    print(dist.reshape(-1))
    print("\nPer-view errors:")
    for view, err in zip(kept, per_view):
        print(f"  view {view.index:02d}: error={err:.4f} px  sharpness={view.sharpness:.1f}")
    print("=" * 72 + "\n")

    return {
        "rms": float(rms),
        "camera_matrix": K,
        "dist_coeffs": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_view_errors_px": per_view,
        "image_size": np.asarray(image_size, dtype=np.int32),
        "board_inner_corners": np.asarray(BOARD_INNER_CORNERS, dtype=np.int32),
        "square_size_mm": np.asarray([SQUARE_SIZE_MM], dtype=np.float64),
        "num_views": np.asarray([len(kept)], dtype=np.int32),
    }


def save_calibration(result: dict, path: Path) -> None:
    np.savez(
        path,
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
        distortion_coefficients=result["dist_coeffs"],  # alias for compatibility
        rms=np.asarray([result["rms"]], dtype=np.float64),
        per_view_errors_px=result["per_view_errors_px"],
        image_size=result["image_size"],
        board_inner_corners=result["board_inner_corners"],
        square_size_mm=result["square_size_mm"],
        num_views=result["num_views"],
        created_unix_time=np.asarray([time.time()], dtype=np.float64),
    )
    print(f"[SAVE] Wrote {path.resolve()}")


def make_undistorted_preview(frame: np.ndarray, result: dict):
    K = result["camera_matrix"]
    dist = result["dist_coeffs"]
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0, newImgSize=(w, h))
    undist = cv2.undistort(frame, K, dist, None, new_K)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        cv2.rectangle(undist, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
    return undist


def cleanup_debug_dir():
    if DEBUG_DIR.exists() and not KEEP_DEBUG_IMAGES:
        shutil.rmtree(DEBUG_DIR, ignore_errors=True)
        print(f"[CLEANUP] Deleted {DEBUG_DIR.resolve()}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nOverhead Webcam Intrinsics Calibration")
    print("--------------------------------------")
    print(f"Board inner corners: {BOARD_INNER_CORNERS}")
    print(f"Square size: {SQUARE_SIZE_MM:.3f} mm")
    print("Move the board around the image: center, corners, tilted, rotated, near/far.")
    print("Raw photos are stored in RAM only. Invalid frames are not kept.\n")

    cap = open_camera()
    detector_objp = make_object_points()

    kept: list[KeptView] = []
    last_result = None
    undistort_preview = False
    next_index = 1

    if SAVE_DEBUG_IMAGES:
        DEBUG_DIR.mkdir(exist_ok=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 540)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            found, corners, gray = detect_checkerboard(frame)
            sharp = sharpness_metric(gray)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, BOARD_INNER_CORNERS, corners, found)

            display = draw_coverage(display, kept)

            if undistort_preview and last_result is not None:
                undist = make_undistorted_preview(frame, last_result)
                if found:
                    cv2.drawChessboardCorners(undist, BOARD_INNER_CORNERS, corners, found)
                display = np.hstack([
                    cv2.resize(display, (640, 360)),
                    cv2.resize(undist, (640, 360)),
                ])

            status = "VALID - press SPACE to keep" if found else "NO CHECKERBOARD"
            n = len(kept)
            lines = [
                f"{status}",
                f"kept views: {n} / recommended {RECOMMENDED_VALID_VIEWS}   min={MIN_VALID_VIEWS_FOR_CALIBRATION}",
                f"board={BOARD_INNER_CORNERS} inner corners   square={SQUARE_SIZE_MM:.1f} mm",
                f"sharpness={sharp:.1f}",
                "SPACE keep | c calibrate | u undistort preview | d delete last | r reset | s save | q quit",
            ]

            if last_result is not None:
                lines.append(
                    f"last calib: RMS={last_result['rms']:.4f}px  "
                    f"mean view err={last_result['per_view_errors_px'].mean():.4f}px"
                )

            display = draw_text_lines(display, lines)
            cv2.imshow(WINDOW, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == 32:  # SPACE
                if not found:
                    print("[KEEP] Rejected: no valid checkerboard in current frame.")
                    continue

                image_size = (gray.shape[1], gray.shape[0])
                kept.append(
                    KeptView(
                        index=next_index,
                        t=time.time(),
                        corners=corners.copy().astype(np.float32),
                        image_size=image_size,
                        sharpness=sharp,
                    )
                )

                print(f"[KEEP] view {next_index:02d}: valid checkerboard, sharpness={sharp:.1f}, total={len(kept)}")

                if SAVE_DEBUG_IMAGES:
                    p = DEBUG_DIR / f"valid_view_{next_index:02d}.png"
                    cv2.imwrite(str(p), display)

                next_index += 1
                last_result = None

            elif key == ord("d"):
                if kept:
                    removed = kept.pop()
                    print(f"[DELETE] Removed view {removed.index:02d}. total={len(kept)}")
                    last_result = None
                else:
                    print("[DELETE] No kept views.")

            elif key == ord("r"):
                kept.clear()
                last_result = None
                next_index = 1
                print("[RESET] Cleared all kept views.")

            elif key == ord("c"):
                last_result = calibrate(kept, detector_objp)
                if last_result is not None:
                    undistort_preview = True
                    print("[CAL] Undistorted preview ON. Press u to toggle.")

            elif key == ord("u"):
                if last_result is None:
                    print("[UNDISTORT] Calibrate first with c.")
                else:
                    undistort_preview = not undistort_preview
                    print(f"[UNDISTORT] preview {'ON' if undistort_preview else 'OFF'}")

            elif key == ord("s"):
                if last_result is None:
                    print("[SAVE] Calibrate first with c.")
                    continue

                print("\nSave calibration to overhead_intrinsics.npz? Type YES then Enter.")
                ans = input("> ").strip()
                if ans == "YES":
                    save_calibration(last_result, OUT_PATH)
                else:
                    print("[SAVE] Cancelled. Nothing written.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cleanup_debug_dir()
        print("[DONE] Closed camera and cleaned up temporary files.")


if __name__ == "__main__":
    main()
