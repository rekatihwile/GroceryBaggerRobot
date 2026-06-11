from __future__ import annotations

"""Calibrate/update a named destination or staging surface Z zone."""

# ============================================================
# USER SETTINGS
# ============================================================

from pathlib import Path

SURFACE_ZONE_NAME = "default_test_place_zone"
KNOWN_TAG_BLOCK_HEIGHT_MM = 100.0
SURFACE_ZONE_CONFIG_PATH = Path("config/surface_zones.json")

REFERENCE_TAG_ID = 10
ENABLE_APRILTAG_SAMPLING = True
APRILTAG_MAX_SAMPLES = 8

CONNECT_ROBOT_FOR_MANUAL_CAPTURE = True
ENABLE_MOTORS_ON_START = True

# ============================================================

import json
import statistics
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "run_pickplace_fast.py").exists()),
    Path(__file__).resolve().parents[2],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.surface_zone_io import get_surface_zone, upsert_surface_zone
from config.robot_config import ROBOT_CONFIG
from hardware.robot import Robot
from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera, build_detector
from test_calibration_bundle_live_stereo_z_pickplace import (
    BUNDLE_PATH,
    STEREO_CALIBRATION_PATH,
    cam_xyz_to_robot_xyz,
    load_bundle,
    load_stereo_calibration,
    read_stereo_tags_once,
)


def _sample_surface_z_from_apriltag() -> list[float]:
    if not ENABLE_APRILTAG_SAMPLING:
        return []

    print("\n[APRILTAG] Sampling mode")
    print("[APRILTAG] Place known-height tag/block on target surface.")
    print("[APRILTAG] Press Enter to sample, q to stop sampling.")

    samples: list[float] = []
    stereo = None
    try:
        bundle = load_bundle(BUNDLE_PATH)
        stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
        detector = build_detector()
        stereo = SimpleStereoCamera()

        while len(samples) < APRILTAG_MAX_SAMPLES:
            cmd = input(f"[APRILTAG] sample {len(samples) + 1}/{APRILTAG_MAX_SAMPLES} (Enter/q): ").strip().lower()
            if cmd == "q":
                break

            tags = read_stereo_tags_once(stereo, detector, stereo_calib)
            tri = tags.get(int(REFERENCE_TAG_ID))
            if tri is None:
                print(f"[APRILTAG] tag id {REFERENCE_TAG_ID} not visible in stereo; try again.")
                continue

            tag_robot_xyz = cam_xyz_to_robot_xyz(tri.xyz_cam_mm, bundle)
            surface_z_mm = float(tag_robot_xyz[2]) - float(KNOWN_TAG_BLOCK_HEIGHT_MM)
            samples.append(surface_z_mm)
            print(
                f"[APRILTAG] tag_z={float(tag_robot_xyz[2]):.3f} mm -> "
                f"surface_z={surface_z_mm:.3f} mm"
            )

    except Exception as exc:
        print(f"[APRILTAG] sampling unavailable: {exc}")
    finally:
        if stereo is not None:
            try:
                stereo.release()
            except Exception:
                pass

    if samples:
        arr = np.asarray(samples, dtype=np.float64)
        print("\n[APRILTAG] summary")
        print(f"mean={float(np.mean(arr)):.3f} mm")
        print(f"std={float(np.std(arr)):.3f} mm")
        print(f"min={float(np.min(arr)):.3f} mm")
        print(f"max={float(np.max(arr)):.3f} mm")
        print(f"count={len(samples)}")

    return samples


def _manual_capture_surface_z(initial_surface_z_mm: float | None) -> float | None:
    if not CONNECT_ROBOT_FOR_MANUAL_CAPTURE:
        return initial_surface_z_mm

    print("\n[MANUAL] fallback capture mode")
    print("[MANUAL] Jog robot so reference point is just touching or just above surface.")
    print("[MANUAL] Commands: c=capture FK z, s=save last capture, q=quit")

    robot = Robot(ROBOT_CONFIG)
    captured_z_mm = initial_surface_z_mm
    try:
        if ENABLE_MOTORS_ON_START:
            robot.enable(True)

        while True:
            cmd = input("[MANUAL] command (c/s/q): ").strip().lower()
            if cmd == "q":
                return None
            if cmd == "c":
                _x, _y, z, _phi = robot.fk()
                captured_z_mm = float(z)
                print(f"[MANUAL] captured surface_z_mm={captured_z_mm:.3f}")
            elif cmd == "s":
                if captured_z_mm is None:
                    print("[MANUAL] no capture yet; press c first.")
                    continue
                return float(captured_z_mm)

    finally:
        try:
            robot.close()
        except Exception:
            pass


def _save_surface_zone(surface_z_mm: float) -> dict[str, Any]:
    try:
        existing = get_surface_zone(SURFACE_ZONE_NAME, SURFACE_ZONE_CONFIG_PATH)
    except Exception:
        existing = {}

    zone = upsert_surface_zone(
        name=SURFACE_ZONE_NAME,
        surface_z_mm=float(surface_z_mm),
        center_xy_mm=existing.get("center_xy_mm", [450.0, 250.0]),
        width_mm=existing.get("width_mm", 120.0),
        depth_mm=existing.get("depth_mm", 120.0),
        default_phi_deg=existing.get("default_phi_deg", 0.0),
        notes=(
            "Calibrated by scripts/manual_control/calibrate_surface_z_from_apriltag.py"
        ),
        path=SURFACE_ZONE_CONFIG_PATH,
    )
    return zone


def main() -> int:
    apriltag_samples = _sample_surface_z_from_apriltag()
    apriltag_mean = None
    if apriltag_samples:
        apriltag_mean = float(statistics.fmean(apriltag_samples))

    selected_surface_z = apriltag_mean
    if apriltag_mean is not None:
        ans = input(
            f"[SAVE] Use AprilTag mean surface_z_mm={apriltag_mean:.3f}? (y/n): "
        ).strip().lower()
        if ans != "y":
            selected_surface_z = None

    if selected_surface_z is None:
        selected_surface_z = _manual_capture_surface_z(initial_surface_z_mm=apriltag_mean)

    if selected_surface_z is None:
        print("[EXIT] no surface z selected; nothing saved.")
        return 1

    zone = _save_surface_zone(selected_surface_z)
    print("\n[SURFACE ZONE] saved:")
    print(json.dumps(zone, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
