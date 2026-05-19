from __future__ import annotations

"""
overhead_ee_homography_calibration.py

End-effector AprilTag homography calibration for the grocery bagger.

What this does:
  1. Connects to the robot.
  2. Opens the overhead webcam and the side-by-side stereo camera.
  3. Moves the end effector to known robot-frame XY calibration points.
  4. Detects the EE AprilTag in the overhead webcam.
  5. Optionally also detects/triangulates the EE tag in the stereo camera for logging.
  6. Builds a homography:

        overhead pixel (u, v)  -->  robot XY (x_mm, y_mm)

  7. Saves:
        overhead_homography_calibration.npz
        overhead_homography_samples.csv

Controls:
  h/c/a at startup = home / sync current Teensy POS / assume configured home pose
  SPACE            = accept the current point after robot moves there
  r                = re-measure current point
  s                = skip current point
  q / ESC          = abort

Required files in same folder:
  robot.py
  stereo_apriltag_viewer.py
  stereo_calibration.npz    # optional but used for stereo XYZ logging if present
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from robot import Robot
from robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
    require_robot_soft_limits_loaded,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    OVERHEAD_FOURCC,
    OVERHEAD_FPS,
    OVERHEAD_HEIGHT,
    OVERHEAD_HOMOGRAPHY_PATH as OUT_NPZ,
    OVERHEAD_INDEX,
    OVERHEAD_WIDTH,
    STEREO_CALIBRATION_PATH as CALIBRATION_PATH,
    STEREO_FOURCC,
    STEREO_FPS,
    STEREO_HEIGHT,
    STEREO_INDEX,
    STEREO_WIDTH,
)
from overhead_camera import SimpleOverheadCamera
from stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
    make_preview,
    average_angles_deg,
)

# ============================================================
# USER SETTINGS — EDIT THESE FIRST
# ============================================================

OUT_CSV = Path("overhead_homography_samples.csv")

# Robot-frame calibration points in mm.
# Keep these conservative and inside your real reachable/visible workspace.
# The script uses robot.move_cartesian(x, y, CAL_Z_MM, CAL_PHI_DEG).
CAL_Z_MM = LOW_Z_MM
CAL_PHI_DEG = 0.0
MOVE_TIME_S = 1.15
SETTLE_S = 0.35

CAL_POINTS = [
    ("P1_front_left",   75.0, 245.0),
    ("P2_front_right",  465.0, 245.0),
    ("P3_back_right",   450, 580.0),
    ("P4_back_left",    168.0, 530.0),

    # Extra points improve least-squares robustness. Keep them visible.
    ("P5_center",       300.0, 300.0),
    ("P6_mid_front",    300.0, 400.0),
]

SAMPLES_PER_POINT = 20
POINT_TIMEOUT_S = 5.0

WINDOW_OVERHEAD = "Overhead EE Homography Calibration"
WINDOW_STEREO = "Stereo EE Logger"

# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class SampleRow:
    name: str
    robot_x_mm: float
    robot_y_mm: float
    robot_z_mm: float
    robot_phi_deg: float
    overhead_u_px: float
    overhead_v_px: float
    overhead_theta_deg: float
    stereo_left_u_px: float | None = None
    stereo_left_v_px: float | None = None
    stereo_right_u_px: float | None = None
    stereo_right_v_px: float | None = None
    stereo_disparity_px: float | None = None
    stereo_x_mm: float | None = None
    stereo_y_mm: float | None = None
    stereo_z_mm: float | None = None
    stereo_theta_deg: float | None = None

# ============================================================
# CAMERA HELPERS
# ============================================================

def open_overhead_camera() -> cv2.VideoCapture:
    return SimpleOverheadCamera().cap


def read_overhead_detection(cap: cv2.VideoCapture, detector):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, None
    dets = detect_tags(detector, frame)
    return frame, dets.get(EE_TAG_ID)


def draw_overhead(frame: np.ndarray, det, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    out = draw_detection(out, det, "OVERHEAD")
    for i, line in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def load_stereo_calibration(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        print(f"[Stereo] No {path}. Stereo XYZ will be skipped, but left/right pixel logging still works.")
        return None
    data = np.load(path, allow_pickle=False)
    calib = {k: np.asarray(data[k]) for k in data.files}
    print(f"[Stereo] Loaded calibration: {path.resolve()}")
    return calib


def undistort_pixel_to_pixel_coords(point_xy: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    pt = np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2)
    corrected = cv2.undistortPoints(pt, K, dist, P=K)
    return corrected.reshape(2).astype(np.float64)


def triangulate_center_raw(left_xy: np.ndarray, right_xy: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    left_pt = undistort_pixel_to_pixel_coords(
        left_xy,
        np.asarray(calib["left_camera_matrix"], dtype=np.float64),
        np.asarray(calib["left_distortion_coefficients"], dtype=np.float64),
    ).reshape(2, 1)
    right_pt = undistort_pixel_to_pixel_coords(
        right_xy,
        np.asarray(calib["right_camera_matrix"], dtype=np.float64),
        np.asarray(calib["right_distortion_coefficients"], dtype=np.float64),
    ).reshape(2, 1)

    P_left = np.asarray(calib["projection_left_raw"], dtype=np.float64)
    P_right = np.asarray(calib["projection_right_raw"], dtype=np.float64)

    X_h = cv2.triangulatePoints(P_left, P_right, left_pt, right_pt)
    xyz = (X_h[:3] / X_h[3]).reshape(3).astype(np.float64)

    # Match your existing convention from the stereo PD scripts.
    xyz[2] *= -1.0
    return xyz


def read_stereo_once(stereo: SimpleStereoCamera, detector, calib: dict[str, np.ndarray] | None):
    ok, _, left, right = stereo.read_pair()
    if not ok or left is None or right is None:
        return None, None, None, None, None

    det_l_all = detect_tags(detector, left)
    det_r_all = detect_tags(detector, right)
    det_l = det_l_all.get(EE_TAG_ID)
    det_r = det_r_all.get(EE_TAG_ID)

    xyz = None
    if calib is not None and det_l is not None and det_r is not None:
        xyz = triangulate_center_raw(
            np.asarray(det_l.center, dtype=np.float64),
            np.asarray(det_r.center, dtype=np.float64),
            calib,
        )

    return left, right, det_l, det_r, xyz


def draw_stereo(left, right, det_l, det_r, xyz, lines: list[str]) -> np.ndarray:
    left_draw = draw_detection(left, det_l, "LEFT") if left is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    right_draw = draw_detection(right, det_r, "RIGHT") if right is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    if xyz is not None:
        lines = list(lines) + [f"stereo XYZ=({xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f}) mm"]
    return make_preview(left_draw, right_draw, lines)

# ============================================================
# MEASUREMENT
# ============================================================

def measure_current_point(
    name: str,
    robot: Robot,
    overhead: cv2.VideoCapture,
    stereo: SimpleStereoCamera,
    detector,
    stereo_calib: dict[str, np.ndarray] | None,
) -> SampleRow | None:
    """Average overhead tag center. Also logs stereo if visible."""

    overhead_samples = []
    stereo_samples = []
    start = time.time()

    while time.time() - start < POINT_TIMEOUT_S:
        frame, det_o = read_overhead_detection(overhead, detector)
        left, right, det_l, det_r, xyz = read_stereo_once(stereo, detector, stereo_calib)

        if frame is not None and det_o is not None:
            overhead_samples.append([
                float(det_o.center[0]),
                float(det_o.center[1]),
                float(det_o.theta_deg),
            ])

        if det_l is not None and det_r is not None:
            ul, vl = det_l.center
            ur, vr = det_r.center
            theta = average_angles_deg(det_l.theta_deg, det_r.theta_deg)
            row = [float(ul), float(vl), float(ur), float(vr), float(ul - ur), float(theta)]
            if xyz is not None:
                row += [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            else:
                row += [np.nan, np.nan, np.nan]
            stereo_samples.append(row)

        n = len(overhead_samples)
        lines = [
            f"Measuring {name}",
            f"overhead samples {n}/{SAMPLES_PER_POINT}",
            f"overhead EE visible: {det_o is not None}",
            "SPACE accept early | r reset | s skip | q abort",
        ]
        if det_o is not None:
            lines.append(f"overhead uv=({det_o.center[0]:.1f}, {det_o.center[1]:.1f})")

        if frame is not None:
            cv2.imshow(WINDOW_OVERHEAD, draw_overhead(frame, det_o, lines))

        if left is not None and right is not None:
            stereo_lines = [
                f"Stereo logger for {name}",
                f"EE left/right visible: {det_l is not None}/{det_r is not None}",
                f"stereo samples: {len(stereo_samples)}",
            ]
            cv2.imshow(WINDOW_STEREO, draw_stereo(left, right, det_l, det_r, xyz, stereo_lines))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord("s"):
            print(f"[SKIP] {name}")
            return None
        if key == ord("r"):
            print(f"[RESET] {name}")
            overhead_samples.clear()
            stereo_samples.clear()
        if key == 32 and len(overhead_samples) >= 4:
            print(f"[ACCEPT EARLY] {name} with {len(overhead_samples)} overhead samples")
            break
        if len(overhead_samples) >= SAMPLES_PER_POINT:
            break

    if len(overhead_samples) < 4:
        print(f"[WARN] Not enough overhead samples for {name}. Got {len(overhead_samples)}.")
        return None

    o = np.asarray(overhead_samples, dtype=float).mean(axis=0)
    x, y, z, phi = robot.fk()

    row = SampleRow(
        name=name,
        robot_x_mm=float(x),
        robot_y_mm=float(y),
        robot_z_mm=float(z),
        robot_phi_deg=float(phi),
        overhead_u_px=float(o[0]),
        overhead_v_px=float(o[1]),
        overhead_theta_deg=float(o[2]),
    )

    if len(stereo_samples) >= 2:
        s = np.asarray(stereo_samples, dtype=float).mean(axis=0)
        row.stereo_left_u_px = float(s[0])
        row.stereo_left_v_px = float(s[1])
        row.stereo_right_u_px = float(s[2])
        row.stereo_right_v_px = float(s[3])
        row.stereo_disparity_px = float(s[4])
        row.stereo_theta_deg = float(s[5])
        row.stereo_x_mm = None if np.isnan(s[6]) else float(s[6])
        row.stereo_y_mm = None if np.isnan(s[7]) else float(s[7])
        row.stereo_z_mm = None if np.isnan(s[8]) else float(s[8])

    print(f"[OK] {name}: robot=({row.robot_x_mm:.1f},{row.robot_y_mm:.1f},{row.robot_z_mm:.1f}) overhead=({row.overhead_u_px:.1f},{row.overhead_v_px:.1f})")
    return row

# ============================================================
# SAVE / VALIDATE
# ============================================================

def save_csv(rows: list[SampleRow]) -> None:
    fields = list(SampleRow.__dataclass_fields__.keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k) for k in fields})
    print(f"[SAVE] CSV → {OUT_CSV.resolve()}")


def build_and_save_homography(rows: list[SampleRow]) -> None:
    if len(rows) < 4:
        raise RuntimeError("Need at least 4 valid calibration points to compute homography.")

    image_pts = np.array([[r.overhead_u_px, r.overhead_v_px] for r in rows], dtype=np.float32)
    robot_pts = np.array([[r.robot_x_mm, r.robot_y_mm] for r in rows], dtype=np.float32)

    H_img_to_robot, mask = cv2.findHomography(image_pts, robot_pts, method=0)
    H_robot_to_img, _ = cv2.findHomography(robot_pts, image_pts, method=0)

    if H_img_to_robot is None or H_robot_to_img is None:
        raise RuntimeError("cv2.findHomography failed.")

    projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H_img_to_robot).reshape(-1, 2)
    err = projected - robot_pts
    err_norm = np.linalg.norm(err, axis=1)

    print("\nCalibration residuals:")
    for r, e, en in zip(rows, err, err_norm):
        print(f"  {r.name:15s}: ex={e[0]:+7.2f} mm  ey={e[1]:+7.2f} mm  |e|={en:6.2f} mm")
    print(f"  RMS error: {np.sqrt(np.mean(err_norm**2)):.2f} mm")
    print(f"  Max error: {np.max(err_norm):.2f} mm")

    np.savez(
        OUT_NPZ,
        H_img_to_robot=H_img_to_robot,
        H_robot_to_img=H_robot_to_img,
        image_pts_px=image_pts,
        robot_pts_mm=robot_pts,
        residuals_mm=err,
        residual_norms_mm=err_norm,
        names=np.array([r.name for r in rows]),
        overhead_index=np.array([OVERHEAD_INDEX]),
        stereo_index=np.array([STEREO_INDEX]),
        cal_z_mm=np.array([CAL_Z_MM]),
        ee_tag_id=np.array([EE_TAG_ID]),
    )
    print(f"[SAVE] Homography NPZ → {OUT_NPZ.resolve()}")

    print("\nUse this to map a detected overhead pixel to robot XY:")
    print("    pt = np.array([[[u, v]]], dtype=np.float32)")
    print("    xy = cv2.perspectiveTransform(pt, H_img_to_robot)[0,0]")

# ============================================================
# MAIN
# ============================================================

def main():
    print("\nOverhead EE AprilTag Homography Calibration")
    print("-------------------------------------------")
    print("This builds overhead pixel → robot XY using the end-effector tag at known robot poses.")
    print("Keep your hand near power / E-stop. Make sure all points are reachable and visible.\n")
    print_startup_config("overhead_ee_homography_calibration.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("overhead_ee_homography_calibration.py")

    robot = Robot(ROBOT_CONFIG, connect=True)
    overhead = None
    stereo = None

    try:
        require_robot_soft_limits_loaded(robot, "overhead_ee_homography_calibration.py")
        robot.enable(True)
        robot.init_drivers()

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()

        if choice == "h":
            if not robot.home():
                print("HOME failed. Aborting.")
                return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps():
                print("Could not sync from POS. Aborting.")
                return
        elif choice == "a":
            robot.assume_homed()
            robot.print_estimate()
        else:
            print("Unknown choice. Aborting.")
            return

        detector = build_detector()
        overhead = open_overhead_camera()
        stereo = SimpleStereoCamera(
            index=STEREO_INDEX,
            width=STEREO_WIDTH,
            height=STEREO_HEIGHT,
            fps=STEREO_FPS,
            fourcc=STEREO_FOURCC,
        )
        stereo_calib = load_stereo_calibration(CALIBRATION_PATH)

        cv2.namedWindow(WINDOW_OVERHEAD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_OVERHEAD, 960, 540)
        cv2.namedWindow(WINDOW_STEREO, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_STEREO, 1280, 520)

        rows: list[SampleRow] = []

        for name, x_cmd, y_cmd in CAL_POINTS:
            print(f"\n[MOVE] {name}: x={x_cmd:.1f}, y={y_cmd:.1f}, z={CAL_Z_MM:.1f}, phi={CAL_PHI_DEG:.1f}")
            ok = robot.move_cartesian(
                x_mm=x_cmd,
                y_mm=y_cmd,
                z_mm=CAL_Z_MM,
                phi_deg=CAL_PHI_DEG,
                move_time_s=MOVE_TIME_S,
            )
            if not ok:
                print(f"[WARN] Move failed for {name}. Skipping.")
                continue

            time.sleep(SETTLE_S)
            robot.sync_estimate_from_teensy_steps()

            print("Look at the overhead window. Press SPACE to accept early, r to reset, s to skip.")
            row = measure_current_point(name, robot, overhead, stereo, detector, stereo_calib)
            if row is not None:
                rows.append(row)
                save_csv(rows)

        if len(rows) >= 4:
            build_and_save_homography(rows)
        else:
            print(f"[FAIL] Only got {len(rows)} valid points. Need at least 4.")

    except KeyboardInterrupt:
        print("\n[ABORT] User aborted.")
        if 'rows' in locals() and rows:
            save_csv(rows)
            print("Partial CSV saved.")
    finally:
        if overhead is not None:
            overhead.release()
        if stereo is not None:
            stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
