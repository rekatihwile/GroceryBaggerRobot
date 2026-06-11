from __future__ import annotations

"""
scripts/smoke_tests/smoke_live_vision_imports.py

Validates that all hardware + vision imports needed by the live camera debug
scripts are importable without opening any cameras or running GPU inference.

Run from repo root:
    python scripts/smoke_tests/smoke_live_vision_imports.py
"""

# ============================================================
# USER SETTINGS
# ============================================================
VERBOSE = True
BUNDLE_PATH_CHECK = "robot_calibration_bundle.npz"
STEREO_CALIB_PATH_CHECK = "stereo_calibration.npz"
# ============================================================

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_PASS = []
_WARN = []
_FAIL = []


def _check(label: str, fn) -> bool:
    try:
        fn()
        _PASS.append(label)
        if VERBOSE:
            print(f"  [PASS] {label}")
        return True
    except Exception as exc:
        _FAIL.append(f"{label}: {exc}")
        if VERBOSE:
            print(f"  [FAIL] {label}: {exc}")
        return False


def _warn(msg: str) -> None:
    _WARN.append(msg)
    if VERBOSE:
        print(f"  [WARN] {msg}")


# ------------------------------------------------------------------ #
# Hardware imports
# ------------------------------------------------------------------ #

_check(
    "SimpleStereoCamera (hardware.cameras.stereo_apriltag_viewer)",
    lambda: __import__("hardware.cameras.stereo_apriltag_viewer", fromlist=["SimpleStereoCamera"]),
)

_check(
    "SimpleOverheadCamera (hardware.cameras.overhead_camera)",
    lambda: __import__("hardware.cameras.overhead_camera", fromlist=["SimpleOverheadCamera"]),
)

_check(
    "config.camera_config (OVERHEAD_INDEX, STEREO_INDEX)",
    lambda: (
        __import__("config.camera_config", fromlist=["OVERHEAD_INDEX", "STEREO_INDEX"])
    ),
)

# ------------------------------------------------------------------ #
# Vision module imports
# ------------------------------------------------------------------ #

_check(
    "YOLOSegmenter (vision.yolo_segmenter)",
    lambda: __import__("vision.yolo_segmenter", fromlist=["YOLOSegmenter"]),
)

_check(
    "RAFTStereoRunner (vision.raft_runner)",
    lambda: __import__("vision.raft_runner", fromlist=["RAFTStereoRunner"]),
)

_check(
    "StereoRectifier (vision.stereo_rectifier)",
    lambda: __import__("vision.stereo_rectifier", fromlist=["StereoRectifier"]),
)

_check(
    "masked_disparity_to_pointcloud (vision.pointcloud)",
    lambda: __import__("vision.pointcloud", fromlist=["masked_disparity_to_pointcloud"]),
)

_check(
    "select_torch_device (vision.torch_device)",
    lambda: __import__("vision.torch_device", fromlist=["select_torch_device"]),
)

# ------------------------------------------------------------------ #
# Local helper import
# ------------------------------------------------------------------ #

_check(
    "vision_debug_helpers (scripts/vision_debug_helpers.py)",
    lambda: __import__("vision_debug_helpers"),
)

# ------------------------------------------------------------------ #
# Calibration file checks (warn only — no hardware needed)
# ------------------------------------------------------------------ #

import numpy as np

stereo_path = _REPO_ROOT / STEREO_CALIB_PATH_CHECK
if stereo_path.exists():
    def _load_stereo():
        data = np.load(str(stereo_path), allow_pickle=False)
        _ = list(data.files)
    _check(f"np.load stereo_calibration.npz ({stereo_path.name})", _load_stereo)
else:
    _warn(f"stereo_calibration.npz not found at {stereo_path} — will fail at runtime.")

bundle_path = _REPO_ROOT / BUNDLE_PATH_CHECK
if bundle_path.exists():
    def _load_bundle():
        data = np.load(str(bundle_path), allow_pickle=False)
        _ = dict(data)
    _check(f"np.load robot_calibration_bundle.npz ({bundle_path.name})", _load_bundle)
else:
    _warn(f"robot_calibration_bundle.npz not found at {bundle_path} — robot mapping will be disabled at runtime.")

# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #

print()
print("=" * 50)
print(f"PASS : {len(_PASS)}")
print(f"WARN : {len(_WARN)}")
print(f"FAIL : {len(_FAIL)}")
print("=" * 50)

if _WARN:
    print("Warnings:")
    for w in _WARN:
        print(f"  ! {w}")

if _FAIL:
    print("Failures:")
    for f in _FAIL:
        print(f"  x {f}")
    print()
    print("RESULT: FAILED")
    sys.exit(1)
else:
    print("RESULT: PASSED")
    sys.exit(0)
