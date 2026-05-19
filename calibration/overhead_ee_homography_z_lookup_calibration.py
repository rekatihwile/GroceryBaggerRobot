from __future__ import annotations

"""
overhead_ee_homography_z_lookup_calibration.py

Build a height-indexed overhead homography lookup table for the grocery bagger.

Why:
  A homography maps pixels to robot XY only for one physical plane. If the tag/object
  is at a different Z height, the same overhead pixel no longer maps to the same XY.
  This script moves the EE AprilTag through the same robot XY calibration points at
  multiple Z levels and saves one homography per Z level:

      overhead pixel (u,v) + selected z_mm -> robot XY (x_mm,y_mm)

Outputs:
  overhead_homography_z_lookup.npz
  overhead_homography_z_lookup_samples.csv

Controls:
  h/c/a at startup = home / sync current Teensy POS / assume configured home pose
  SPACE            = accept current point early
  r                = reset current point samples
  s                = skip current point
  q / ESC          = abort

Required files in same folder:
  robot.py
  stereo_apriltag_viewer.py

Optional:
  overhead_intrinsics.npz containing camera_matrix and dist_coeffs/distortion_coefficients.
  If present, tag centers are undistorted before building the homographies.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "run_pickplace_fast.py").exists()),
    Path(__file__).resolve().parents[1],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from hardware.robot import Robot
from config.robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
    require_robot_soft_limits_loaded,
    require_soft_limits_configured,
)
from config.camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_INDEX,
    OVERHEAD_INTRINSICS_PATH as INTRINSICS_PATH,
    OVERHEAD_WIDTH,
    OVERHEAD_Z_LOOKUP_PATH as OUT_NPZ,
    STEREO_INDEX,
)
from hardware.cameras.overhead_camera import SimpleOverheadCamera
from hardware.cameras.stereo_apriltag_viewer import build_detector, detect_tags, draw_detection

# ============================================================
# USER SETTINGS
# ============================================================

# Same XY points at every Z. Use only points visible to the overhead webcam.
CAL_POINTS = [
    ("P1_front_left",   295, 285.0),
    ("P2_front_right", 465.0, 245.0),
    ("P3_back_right", 450.0, 580.0),
    ("P4_back_left",  168.0, 530.0),
    ("P5_center",     300.0, 300.0),
    ("P6_mid",        300.0, 400.0),
]

# Pick levels that correspond to planes where target tags may be seen.
# Start with 3-4 levels. Add more only if needed.
Z_LEVELS_MM = [0.0, LOW_Z_MM, 50.0, 75.0, HOME_Z_MM]

CAL_PHI_DEG = 0.0
MOVE_TIME_S = 1.15
SETTLE_S = 0.35
SAMPLES_PER_POINT = 20
POINT_TIMEOUT_S = 5.0

OUT_CSV = Path("overhead_homography_z_lookup_samples.csv")
WINDOW = "Overhead Z Lookup Calibration"

# ============================================================
# DATA
# ============================================================

@dataclass
class SampleRow:
    z_level_mm: float
    name: str
    robot_x_mm: float
    robot_y_mm: float
    robot_z_mm: float
    robot_phi_deg: float
    overhead_u_raw_px: float
    overhead_v_raw_px: float
    overhead_u_used_px: float
    overhead_v_used_px: float
    overhead_theta_deg: float

# ============================================================
# CAMERA / INTRINSICS
# ============================================================

def open_overhead_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def load_intrinsics(path: Path):
    if not path.exists():
        print(f"[Intrinsics] No {path}. Using raw distorted pixel centers.")
        return None, None
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)
    K = data["camera_matrix"] if "camera_matrix" in keys else data["K"]
    if "dist_coeffs" in keys:
        dist = data["dist_coeffs"]
    elif "distortion_coefficients" in keys:
        dist = data["distortion_coefficients"]
    elif "dist" in keys:
        dist = data["dist"]
    else:
        raise KeyError(f"Could not find dist coeffs in {path}. Keys={data.files}")
    print(f"[Intrinsics] Loaded {path.resolve()}")
    print(f"[Intrinsics] K=\n{K}")
    print(f"[Intrinsics] dist={dist.reshape(-1)}")
    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def undistort_pixel(pt_uv: np.ndarray, K: np.ndarray | None, dist: np.ndarray | None) -> np.ndarray:
    if K is None or dist is None:
        return np.asarray(pt_uv, dtype=np.float64).reshape(2)
    pt = np.asarray(pt_uv, dtype=np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def read_tag(cap: cv2.VideoCapture, detector):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    dets = detect_tags(detector, frame)
    return frame, dets.get(EE_TAG_ID)


def draw_status(frame: np.ndarray, det, lines: list[str]) -> np.ndarray:
    out = draw_detection(frame, det, "OVERHEAD")
    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# ============================================================
# MEASUREMENT
# ============================================================

def measure_current_point(z_level: float, name: str, robot: Robot, cap, detector, K, dist) -> SampleRow | None:
    raw_samples: list[np.ndarray] = []
    used_samples: list[np.ndarray] = []
    theta_samples: list[float] = []
    start = time.time()

    while time.time() - start < POINT_TIMEOUT_S:
        frame, det = read_tag(cap, detector)
        if frame is None:
            continue

        if det is not None:
            raw_uv = np.asarray(det.center, dtype=np.float64).reshape(2)
            used_uv = undistort_pixel(raw_uv, K, dist)
            raw_samples.append(raw_uv)
            used_samples.append(used_uv)
            theta_samples.append(float(det.theta_deg))

        lines = [
            f"Z lookup calibration | z={z_level:.1f} mm | {name}",
            f"samples {len(raw_samples)}/{SAMPLES_PER_POINT}",
            f"EE tag {EE_TAG_ID} visible: {det is not None}",
            f"intrinsics: {'ON / undistorting centers' if K is not None else 'OFF / raw centers'}",
            "SPACE accept early | r reset | s skip | q abort",
        ]
        if det is not None:
            lines.append(f"raw uv=({raw_uv[0]:.1f}, {raw_uv[1]:.1f}) used uv=({used_uv[0]:.1f}, {used_uv[1]:.1f})")
        cv2.imshow(WINDOW, draw_status(frame, det, lines))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("s"):
            print(f"[SKIP] z={z_level:.1f} {name}")
            return None
        if key == ord("r"):
            print(f"[RESET] z={z_level:.1f} {name}")
            raw_samples.clear(); used_samples.clear(); theta_samples.clear()
        if key == 32 and len(raw_samples) >= 4:
            print(f"[ACCEPT EARLY] z={z_level:.1f} {name} with {len(raw_samples)} samples")
            break
        if len(raw_samples) >= SAMPLES_PER_POINT:
            break

    if len(raw_samples) < 4:
        print(f"[WARN] Not enough samples for z={z_level:.1f} {name}. Got {len(raw_samples)}.")
        return None

    raw_mean = np.mean(np.stack(raw_samples), axis=0)
    used_mean = np.mean(np.stack(used_samples), axis=0)
    theta_mean = float(np.mean(theta_samples)) if theta_samples else float("nan")
    x, y, z, phi = robot.fk()

    row = SampleRow(
        z_level_mm=float(z_level),
        name=name,
        robot_x_mm=float(x),
        robot_y_mm=float(y),
        robot_z_mm=float(z),
        robot_phi_deg=float(phi),
        overhead_u_raw_px=float(raw_mean[0]),
        overhead_v_raw_px=float(raw_mean[1]),
        overhead_u_used_px=float(used_mean[0]),
        overhead_v_used_px=float(used_mean[1]),
        overhead_theta_deg=theta_mean,
    )
    print(f"[OK] z={z_level:.1f} {name}: robot=({x:.1f},{y:.1f},{z:.1f}) used_uv=({used_mean[0]:.1f},{used_mean[1]:.1f})")
    return row

# ============================================================
# SAVE / BUILD
# ============================================================

def save_csv(rows: list[SampleRow]) -> None:
    fields = list(SampleRow.__dataclass_fields__.keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"[SAVE] CSV -> {OUT_CSV.resolve()}")


def build_lookup(rows: list[SampleRow], K, dist):
    z_levels = sorted(set(round(r.z_level_mm, 6) for r in rows))
    Hs = []
    Hs_inv = []
    rms = []
    max_err = []
    names_by_z = []

    print("\nHeight-layer homography residuals:")
    for z in z_levels:
        layer = [r for r in rows if abs(r.z_level_mm - z) < 1e-6]
        if len(layer) < 4:
            print(f"  z={z:.1f}: SKIP, only {len(layer)} points")
            continue

        img_pts = np.array([[r.overhead_u_used_px, r.overhead_v_used_px] for r in layer], dtype=np.float32)
        robot_pts = np.array([[r.robot_x_mm, r.robot_y_mm] for r in layer], dtype=np.float32)

        H, _ = cv2.findHomography(img_pts, robot_pts, method=0)
        H_inv, _ = cv2.findHomography(robot_pts, img_pts, method=0)
        if H is None or H_inv is None:
            print(f"  z={z:.1f}: findHomography failed")
            continue

        pred = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = pred - robot_pts
        err_norm = np.linalg.norm(err, axis=1)
        layer_rms = float(np.sqrt(np.mean(err_norm ** 2)))
        layer_max = float(np.max(err_norm))

        print(f"  z={z:7.2f} mm: RMS={layer_rms:6.2f} mm  MAX={layer_max:6.2f} mm  n={len(layer)}")
        for r, e, en in zip(layer, err, err_norm):
            print(f"      {r.name:15s} ex={e[0]:+7.2f} ey={e[1]:+7.2f} |e|={en:6.2f}")

        Hs.append(H)
        Hs_inv.append(H_inv)
        rms.append(layer_rms)
        max_err.append(layer_max)
        names_by_z.append([r.name for r in layer])

    if not Hs:
        raise RuntimeError("No valid homography layers were built.")

    valid_z = []
    for z in z_levels:
        if len([r for r in rows if abs(r.z_level_mm - z) < 1e-6]) >= 4:
            valid_z.append(z)

    np.savez(
        OUT_NPZ,
        z_levels_mm=np.asarray(valid_z, dtype=np.float64),
        H_img_to_robot_by_z=np.stack(Hs, axis=0),
        H_robot_to_img_by_z=np.stack(Hs_inv, axis=0),
        rms_error_mm=np.asarray(rms, dtype=np.float64),
        max_error_mm=np.asarray(max_err, dtype=np.float64),
        used_intrinsics=np.asarray([K is not None]),
        camera_matrix=np.asarray(K if K is not None else np.eye(3), dtype=np.float64),
        dist_coeffs=np.asarray(dist if dist is not None else np.zeros((1, 5)), dtype=np.float64),
        cal_points_xy_mm=np.asarray([[x, y] for _, x, y in CAL_POINTS], dtype=np.float64),
        ee_tag_id=np.asarray([EE_TAG_ID]),
        overhead_index=np.asarray([OVERHEAD_INDEX]),
    )
    print(f"\n[SAVE] Lookup NPZ -> {OUT_NPZ.resolve()}")
    print("Use the helper function in the test script to interpolate between z layers.")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\nOverhead EE AprilTag Z-Lookup Homography Calibration")
    print("----------------------------------------------------")
    print("This repeats the same overhead calibration at multiple Z heights.")
    print("Keep your hand near power / E-stop.\n")
    print_startup_config("overhead_ee_homography_z_lookup_calibration.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("overhead_ee_homography_z_lookup_calibration.py")

    robot = Robot(ROBOT_CONFIG, connect=True)
    cap = None
    rows: list[SampleRow] = []

    try:
        require_robot_soft_limits_loaded(robot, "overhead_ee_homography_z_lookup_calibration.py")
        robot.enable(True)
        robot.init_drivers()

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()
        if choice == "h":
            if not robot.home():
                print("HOME failed. Aborting."); return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps():
                print("POS sync failed. Aborting."); return
        elif choice == "a":
            robot.assume_homed(); robot.print_estimate()
        else:
            print("Unknown choice. Aborting."); return

        detector = build_detector()
        cap = open_overhead_camera()
        K, dist = load_intrinsics(INTRINSICS_PATH)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 540)

        for z_level in Z_LEVELS_MM:
            print("\n" + "=" * 70)
            print(f"[Z LEVEL] {z_level:.1f} mm")
            print("=" * 70)
            for name, x_cmd, y_cmd in CAL_POINTS:
                print(f"\n[MOVE] z={z_level:.1f} {name}: x={x_cmd:.1f}, y={y_cmd:.1f}")
                ok = robot.move_cartesian(
                    x_mm=x_cmd,
                    y_mm=y_cmd,
                    z_mm=z_level,
                    phi_deg=CAL_PHI_DEG,
                    move_time_s=MOVE_TIME_S,
                )
                if not ok:
                    print(f"[WARN] Move failed for z={z_level:.1f} {name}. Skipping.")
                    continue
                time.sleep(SETTLE_S)
                robot.sync_estimate_from_teensy_steps()

                row = measure_current_point(z_level, name, robot, cap, detector, K, dist)
                if row is not None:
                    rows.append(row)
                    save_csv(rows)

        save_csv(rows)
        build_lookup(rows, K, dist)

    except KeyboardInterrupt:
        print("\n[ABORT] User aborted.")
        if rows:
            save_csv(rows)
            print("Partial CSV saved.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        robot.close()

if __name__ == "__main__":
    main()
