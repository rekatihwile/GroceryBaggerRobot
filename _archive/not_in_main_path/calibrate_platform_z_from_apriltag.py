from __future__ import annotations

"""
scripts/calibrate_platform_z_from_apriltag.py

Slow-buildup height-reference script for the grocery bagger.

Purpose
-------
This script creates a z_ground(x, y) calibration function using a known 100 mm
reference object with AprilTag/ArUco ID 1 on top.

You move the 100 mm object around the pickup platform. Whenever the tag is
visible in BOTH stereo frames, press SPACE. The script saves one sample:

    stereo pixels:
        (u_L, v_L), (u_R, v_R)

    stereo camera-frame triangulation:
        (x_S, y_S, z_S) mm

    robot-frame mapping through robot_calibration_bundle.npz:
        (x_RB, y_RB, z_RB) mm

    inferred local ground/platform height:
        z_ground = z_RB - REFERENCE_OBJECT_HEIGHT_MM

After each sample, the script refits a simple z_ground(x, y) model and writes:

    data/z_ground_calibration/z_ground_samples_<timestamp>.csv
    data/z_ground_calibration/z_ground_samples_<timestamp>.json
    data/z_ground_calibration/z_ground_model_latest.json

This does NOT move the robot. It does NOT run YOLO/RAFT. It does NOT pick.
It only measures the known 100 mm reference object and fits platform height.
"""

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Make imports work whether this file is run from repo root or scripts/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.camera_config import open_overhead_camera as open_overhead_camera_cap
from hardware.cameras.stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
)
from test_calibration_bundle_live_stereo_z_pickplace import (
    BUNDLE_PATH,
    STEREO_CALIBRATION_PATH,
    TriangulatedTag,
    cam_xyz_to_robot_xyz,
    load_bundle,
    load_stereo_calibration,
    read_stereo_tags_once,
)


# ============================================================
# USER SETTINGS
# ============================================================

# The calibration block tag. You said this is AprilTag ID 1, same physical tag
# size as the EE tag. The physical tag size is not needed for this script
# because we triangulate the tag CENTER from stereo pixel correspondence.
REFERENCE_TAG_ID = 10
REFERENCE_TAG_SIZE_MM = 40.0

# Known physical distance from platform/floor contact surface to the detected
# tag center plane on top of your printed reference object.
# If the tag is stuck on the top face of a 100 mm tall object, use 100.0.
REFERENCE_OBJECT_HEIGHT_MM = 100.0

# Optional overhead image capture. This does not affect the fit yet; it just
# saves overhead pixel coordinates in the CSV/JSON so you can later connect
# z_ground samples to overhead homography debugging if useful.
USE_OVERHEAD_CAMERA = True

# Output folder and file stem.
OUTPUT_DIR = Path("data/z_ground_calibration")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = OUTPUT_DIR / f"z_ground_samples_{RUN_STAMP}.csv"
JSON_PATH = OUTPUT_DIR / f"z_ground_samples_{RUN_STAMP}.json"
LATEST_MODEL_PATH = OUTPUT_DIR / "z_ground_model_latest.json"

# Model selection:
#   constant  = z_ground = c0                       [works with >= 1 sample]
#   plane     = z_ground = c0 + c1*dx + c2*dy       [works with >= 3 samples]
#   quadratic = adds dx*dy, dx^2, dy^2 terms        [works with >= 6 samples]
# AUTO uses the most complex model supported by the number of samples.
MODEL_MODE = "auto"  # "auto", "constant", "plane", or "quadratic"

# Display window names.
STEREO_WINDOW = "Z Ground Calibration - Stereo ID1"
OVERHEAD_WINDOW = "Z Ground Calibration - Overhead ID1"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class HeightSample:
    sample_index: int
    timestamp_s: float

    # Left/right image coordinates of the detected reference tag center.
    u_left_px: float
    v_left_px: float
    u_right_px: float
    v_right_px: float
    disparity_px: float

    # Optional overhead image coordinate of the same tag center. This is not
    # used by the z fit, but saved for later homography sanity/debug.
    overhead_u_px: float | None
    overhead_v_px: float | None

    # Raw triangulated stereo-camera coordinates, in mm. These come directly
    # from cv2.triangulatePoints through the existing project helper.
    x_stereo_mm: float
    y_stereo_mm: float
    z_stereo_mm: float

    # Robot-frame coordinates, in mm. These are computed by applying the fitted
    # A_robot_from_cam_xyz_3x4 matrix from robot_calibration_bundle.npz.
    x_robot_mm: float
    y_robot_mm: float
    z_robot_mm: float

    # Local platform/floor height estimate at this XY point:
    # top tag z minus the known 100 mm reference height.
    z_ground_mm: float

    # Filled in after fitting.
    model_pred_z_ground_mm: float | None = None
    model_residual_mm: float | None = None


# ============================================================
# FITTING HELPERS
# ============================================================

def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(y):
        return None
    return y


def choose_model_type(n: int) -> str:
    """Choose model complexity based only on sample count."""
    if MODEL_MODE != "auto":
        return MODEL_MODE
    if n >= 6:
        return "quadratic"
    if n >= 3:
        return "plane"
    return "constant"


def _design_matrix(x_mm: np.ndarray, y_mm: np.ndarray, model_type: str, x0: float, y0: float) -> tuple[np.ndarray, list[str]]:
    """Build centered polynomial design matrix for z_ground(x,y).

    Centering makes the least-squares fit numerically nicer because robot X/Y
    are hundreds of mm, while z_ground variation is usually only mm-scale.
    """
    dx = np.asarray(x_mm, dtype=np.float64) - float(x0)
    dy = np.asarray(y_mm, dtype=np.float64) - float(y0)

    if model_type == "constant":
        return np.column_stack([np.ones_like(dx)]), ["1"]

    if model_type == "plane":
        return np.column_stack([np.ones_like(dx), dx, dy]), ["1", "dx", "dy"]

    if model_type == "quadratic":
        return (
            np.column_stack([np.ones_like(dx), dx, dy, dx * dy, dx * dx, dy * dy]),
            ["1", "dx", "dy", "dx*dy", "dx^2", "dy^2"],
        )

    raise ValueError(f"Unknown model_type={model_type!r}")


def fit_z_ground_model(samples: list[HeightSample]) -> dict[str, Any] | None:
    """Fit z_ground = f(x_robot, y_robot) from all saved samples."""
    if not samples:
        return None

    x = np.array([s.x_robot_mm for s in samples], dtype=np.float64)
    y = np.array([s.y_robot_mm for s in samples], dtype=np.float64)
    z = np.array([s.z_ground_mm for s in samples], dtype=np.float64)

    model_type = choose_model_type(len(samples))
    required = {"constant": 1, "plane": 3, "quadratic": 6}[model_type]
    if len(samples) < required:
        model_type = choose_model_type(required - 1)

    x0 = float(np.mean(x))
    y0 = float(np.mean(y))
    A, terms = _design_matrix(x, y, model_type, x0, y0)

    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coeff
    residual = z - pred

    for s, p, r in zip(samples, pred, residual):
        s.model_pred_z_ground_mm = float(p)
        s.model_residual_mm = float(r)

    rmse = float(np.sqrt(np.mean(residual ** 2))) if len(samples) else float("nan")
    max_abs = float(np.max(np.abs(residual))) if len(samples) else float("nan")

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference_tag_id": REFERENCE_TAG_ID,
        "reference_tag_size_mm": REFERENCE_TAG_SIZE_MM,
        "reference_object_height_mm": REFERENCE_OBJECT_HEIGHT_MM,
        "model_type": model_type,
        "num_samples": len(samples),
        "x_center_mm": x0,
        "y_center_mm": y0,
        "terms": terms,
        "coefficients_mm": [float(c) for c in coeff],
        "rmse_mm": rmse,
        "max_abs_error_mm": max_abs,
        "equation_note": (
            "Let dx = x_robot_mm - x_center_mm and dy = y_robot_mm - y_center_mm. "
            "Then z_ground_mm = sum(coefficients_mm[i] * term_i)."
        ),
    }


def predict_z_ground(model: dict[str, Any], x_robot_mm: float, y_robot_mm: float) -> float:
    """Evaluate a saved z_ground model dict at one robot-frame XY point."""
    x0 = float(model["x_center_mm"])
    y0 = float(model["y_center_mm"])
    model_type = str(model["model_type"])
    coeff = np.asarray(model["coefficients_mm"], dtype=np.float64)
    A, _ = _design_matrix(
        np.array([float(x_robot_mm)]),
        np.array([float(y_robot_mm)]),
        model_type,
        x0,
        y0,
    )
    return float((A @ coeff)[0])


# ============================================================
# SAVE / PRINT HELPERS
# ============================================================

def save_outputs(samples: list[HeightSample], model: dict[str, Any] | None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV: convenient for Excel/plotting.
    fieldnames = list(HeightSample.__dataclass_fields__.keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))

    # JSON: stores both samples and fit metadata.
    payload = {
        "samples": [asdict(s) for s in samples],
        "model": model,
    }
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if model is not None:
        with LATEST_MODEL_PATH.open("w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)


def print_sample(sample: HeightSample, model: dict[str, Any] | None) -> None:
    print("\n" + "=" * 72)
    print(f"[SAMPLE {sample.sample_index}] saved")
    print(
        "[1] pixels: "
        f"[(u_L,v_L)=({sample.u_left_px:.2f},{sample.v_left_px:.2f}), "
        f"(u_R,v_R)=({sample.u_right_px:.2f},{sample.v_right_px:.2f})]  "
        f"disp={sample.disparity_px:.2f}px"
    )
    if sample.overhead_u_px is not None:
        print(
            "    overhead: "
            f"(u_OH,v_OH)=({sample.overhead_u_px:.2f},{sample.overhead_v_px:.2f})"
        )
    else:
        print("    overhead: not detected / not enabled")

    print(
        "[2] stereo cam XYZ: "
        f"x_S={sample.x_stereo_mm:+.2f} mm, "
        f"y_S={sample.y_stereo_mm:+.2f} mm, "
        f"z_S={sample.z_stereo_mm:+.2f} mm"
    )
    print(
        "[3] robot XYZ:      "
        f"x_RB={sample.x_robot_mm:+.2f} mm, "
        f"y_RB={sample.y_robot_mm:+.2f} mm, "
        f"z_RB={sample.z_robot_mm:+.2f} mm"
    )
    print(
        "[4] inferred ground/platform: "
        f"z_ground = z_RB - {REFERENCE_OBJECT_HEIGHT_MM:.1f} = "
        f"{sample.z_ground_mm:+.2f} mm"
    )

    if model is not None:
        print(
            "[5] fit: "
            f"type={model['model_type']}  n={model['num_samples']}  "
            f"rmse={model['rmse_mm']:.2f} mm  "
            f"max_abs={model['max_abs_error_mm']:.2f} mm"
        )
        if sample.model_pred_z_ground_mm is not None:
            print(
                "    newest residual: "
                f"pred={sample.model_pred_z_ground_mm:+.2f} mm, "
                f"residual={sample.model_residual_mm:+.2f} mm"
            )

    print(f"[SAVE] {CSV_PATH}")
    print(f"[SAVE] {JSON_PATH}")
    if model is not None:
        print(f"[SAVE] {LATEST_MODEL_PATH}")
    print("=" * 72 + "\n")


def _format_model_status(model: dict[str, Any] | None) -> str:
    if model is None:
        return "model: none yet"
    return (
        f"model: {model['model_type']}  "
        f"n={model['num_samples']}  "
        f"rmse={model['rmse_mm']:.1f}mm"
    )


def add_status_bar(image: np.ndarray, lines: list[str]) -> np.ndarray:
    row_h = 22
    pad = 8
    h_bar = pad * 2 + row_h * len(lines)
    bar = np.zeros((h_bar, image.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        y = pad + 16 + i * row_h
        cv2.putText(bar, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


# ============================================================
# SAMPLE CREATION
# ============================================================

def make_sample_from_detection(
    *,
    tag: TriangulatedTag,
    bundle: dict[str, Any],
    overhead_det: Any | None,
    sample_index: int,
) -> HeightSample:
    robot_xyz = cam_xyz_to_robot_xyz(tag.xyz_cam_mm, bundle)
    z_ground = float(robot_xyz[2] - REFERENCE_OBJECT_HEIGHT_MM)

    overhead_u = None
    overhead_v = None
    if overhead_det is not None:
        overhead_u = _safe_float(overhead_det.center[0])
        overhead_v = _safe_float(overhead_det.center[1])

    return HeightSample(
        sample_index=sample_index,
        timestamp_s=time.time(),
        u_left_px=float(tag.left_px[0]),
        v_left_px=float(tag.left_px[1]),
        u_right_px=float(tag.right_px[0]),
        v_right_px=float(tag.right_px[1]),
        disparity_px=float(tag.disparity_px),
        overhead_u_px=overhead_u,
        overhead_v_px=overhead_v,
        x_stereo_mm=float(tag.xyz_cam_mm[0]),
        y_stereo_mm=float(tag.xyz_cam_mm[1]),
        z_stereo_mm=float(tag.xyz_cam_mm[2]),
        x_robot_mm=float(robot_xyz[0]),
        y_robot_mm=float(robot_xyz[1]),
        z_robot_mm=float(robot_xyz[2]),
        z_ground_mm=z_ground,
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> int:
    print("=" * 72)
    print("Z ground/platform calibration from stereo AprilTag ID1")
    print(f"REFERENCE_TAG_ID={REFERENCE_TAG_ID}")
    print(f"REFERENCE_TAG_SIZE_MM={REFERENCE_TAG_SIZE_MM:.1f}  (not used for center triangulation)")
    print(f"REFERENCE_OBJECT_HEIGHT_MM={REFERENCE_OBJECT_HEIGHT_MM:.1f}")
    print("Keys: SPACE=save sample, BACKSPACE=delete last, p=print model, q/ESC=quit")
    print("=" * 72)

    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    detector = build_detector()

    stereo = SimpleStereoCamera()

    overhead_cap = None
    if USE_OVERHEAD_CAMERA:
        try:
            overhead_cap = open_overhead_camera_cap()
        except Exception as exc:
            print(f"[WARN] overhead camera unavailable; continuing stereo-only: {exc}")
            overhead_cap = None

    samples: list[HeightSample] = []
    model: dict[str, Any] | None = None
    latest_tag: TriangulatedTag | None = None
    latest_overhead_det: Any | None = None

    cv2.namedWindow(STEREO_WINDOW, cv2.WINDOW_NORMAL)
    if overhead_cap is not None:
        cv2.namedWindow(OVERHEAD_WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            stereo_tags, left, right, det_l_all, det_r_all = read_stereo_tags_once(
                stereo, detector, stereo_calib
            )

            latest_tag = stereo_tags.get(REFERENCE_TAG_ID)

            overhead_frame = None
            latest_overhead_det = None
            if overhead_cap is not None:
                ok_oh, overhead_frame = overhead_cap.read()
                if ok_oh and overhead_frame is not None:
                    oh_dets = detect_tags(detector, overhead_frame)
                    latest_overhead_det = oh_dets.get(REFERENCE_TAG_ID)

            # Draw stereo preview.
            if left is not None and right is not None:
                left_det = det_l_all.get(REFERENCE_TAG_ID)
                right_det = det_r_all.get(REFERENCE_TAG_ID)
                left_draw = draw_detection(left, left_det, f"LEFT ID {REFERENCE_TAG_ID}")
                right_draw = draw_detection(right, right_det, f"RIGHT ID {REFERENCE_TAG_ID}")
                preview = np.hstack([left_draw, right_draw])

                if latest_tag is not None:
                    robot_xyz = cam_xyz_to_robot_xyz(latest_tag.xyz_cam_mm, bundle)
                    z_ground_live = float(robot_xyz[2] - REFERENCE_OBJECT_HEIGHT_MM)
                    live_line = (
                        f"LIVE: cam=({latest_tag.xyz_cam_mm[0]:+.1f},"
                        f"{latest_tag.xyz_cam_mm[1]:+.1f},{latest_tag.xyz_cam_mm[2]:+.1f})mm  "
                        f"robot=({robot_xyz[0]:+.1f},{robot_xyz[1]:+.1f},{robot_xyz[2]:+.1f})mm  "
                        f"z_ground={z_ground_live:+.1f}mm"
                    )
                else:
                    live_line = f"LIVE: ID {REFERENCE_TAG_ID} not detected in BOTH stereo frames"

                preview = add_status_bar(
                    preview,
                    [
                        live_line,
                        f"samples={len(samples)}  {_format_model_status(model)}",
                        "SPACE=save sample   BACKSPACE=delete last   p=print model   q/ESC=quit",
                    ],
                )
                cv2.imshow(STEREO_WINDOW, preview)

            # Draw overhead preview separately if enabled.
            if overhead_cap is not None and overhead_frame is not None:
                oh_draw = draw_detection(overhead_frame, latest_overhead_det, f"OVERHEAD ID {REFERENCE_TAG_ID}")
                cv2.imshow(OVERHEAD_WINDOW, oh_draw)

            key = cv2.waitKeyEx(1)
            if key < 0:
                continue
            key8 = key & 0xFF

            # Quit: q or ESC.
            if key8 in (ord("q"), 27):
                print("[QUIT] requested")
                break

            # Backspace/delete last sample.
            if key8 in (8, 127):
                if samples:
                    removed = samples.pop()
                    for i, s in enumerate(samples, start=1):
                        s.sample_index = i
                    model = fit_z_ground_model(samples)
                    save_outputs(samples, model)
                    print(f"[DELETE] removed sample {removed.sample_index}; {len(samples)} remain")
                else:
                    print("[DELETE] no samples to remove")
                continue

            # Print model.
            if key8 == ord("p"):
                if model is None:
                    print("[MODEL] no model yet; save at least one sample")
                else:
                    print("\n[MODEL]")
                    print(json.dumps(model, indent=2))
                continue

            # Save sample on SPACE.
            if key8 == ord(" "):
                if latest_tag is None:
                    print(f"[SAMPLE] cannot save: ID {REFERENCE_TAG_ID} not detected in BOTH stereo frames")
                    continue

                sample = make_sample_from_detection(
                    tag=latest_tag,
                    bundle=bundle,
                    overhead_det=latest_overhead_det,
                    sample_index=len(samples) + 1,
                )
                samples.append(sample)
                model = fit_z_ground_model(samples)
                save_outputs(samples, model)
                print_sample(sample, model)
                continue

    finally:
        stereo.release()
        if overhead_cap is not None:
            overhead_cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
