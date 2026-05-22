from __future__ import annotations

from pathlib import Path


OVERHEAD_INDEX = 2
OVERHEAD_WIDTH = 1280
OVERHEAD_HEIGHT = 720
OVERHEAD_FPS = 30
OVERHEAD_FOURCC = "MJPG"
OVERHEAD_AUTO_EXPOSURE = True
OVERHEAD_EXPOSURE = -6

# DirectShow/OpenCV commonly use 0.25 for manual and 0.75 for auto.
OVERHEAD_AUTO_EXPOSURE_MANUAL_VALUE = 0.25
OVERHEAD_AUTO_EXPOSURE_AUTO_VALUE = 0.75

STEREO_INDEX = 1
STEREO_WIDTH = 1280
STEREO_HEIGHT = 480
STEREO_FPS = 30
STEREO_FOURCC = "MJPG"
STEREO_AUTO_EXPOSURE = True
STEREO_EXPOSURE = None

# DirectShow/OpenCV commonly use 0.25 for manual and 0.75 for auto.
STEREO_AUTO_EXPOSURE_MANUAL_VALUE = 0.25
STEREO_AUTO_EXPOSURE_AUTO_VALUE = 0.75

EE_TAG_ID = 0
TARGET_TAG_ID = 2

OVERHEAD_INTRINSICS_PATH = Path("overhead_intrinsics.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")
OVERHEAD_HOMOGRAPHY_PATH = Path("overhead_homography_calibration.npz")
OVERHEAD_Z_LOOKUP_PATH = Path("overhead_homography_z_lookup.npz")
STEREO_PD_JACOBIAN_PATH = Path("stereo_pd_jacobian.npz")


def apply_overhead_camera_settings(cap, *, label: str = "[Overhead]", index: int = OVERHEAD_INDEX) -> None:
    import cv2

    if OVERHEAD_FOURCC:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*OVERHEAD_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(OVERHEAD_WIDTH))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(OVERHEAD_HEIGHT))
    cap.set(cv2.CAP_PROP_FPS, int(OVERHEAD_FPS))

    auto_exposure = (
        OVERHEAD_AUTO_EXPOSURE_AUTO_VALUE
        if OVERHEAD_AUTO_EXPOSURE
        else OVERHEAD_AUTO_EXPOSURE_MANUAL_VALUE
    )
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure))
    if not OVERHEAD_AUTO_EXPOSURE and OVERHEAD_EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(OVERHEAD_EXPOSURE))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    mode = "auto" if OVERHEAD_AUTO_EXPOSURE else "manual"
    print(f"{label} index={index} actual {actual_w}x{actual_h}@{actual_fps:.1f}")
    print(f"{label} exposure mode={mode} value={actual_exposure:.3f}")


def apply_stereo_camera_settings(
    cap,
    *,
    label: str = "[Stereo]",
    index: int = STEREO_INDEX,
    width: int = STEREO_WIDTH,
    height: int = STEREO_HEIGHT,
    fps: int = STEREO_FPS,
    fourcc: str = STEREO_FOURCC,
) -> None:
    import cv2

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    if STEREO_AUTO_EXPOSURE is not None:
        auto_exposure = (
            STEREO_AUTO_EXPOSURE_AUTO_VALUE
            if STEREO_AUTO_EXPOSURE
            else STEREO_AUTO_EXPOSURE_MANUAL_VALUE
        )
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure))
        if not STEREO_AUTO_EXPOSURE and STEREO_EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(STEREO_EXPOSURE))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"{label} index={index} requested {width}x{height}@{fps}")
    print(f"{label} index={index} actual    {actual_w}x{actual_h}@{actual_fps:.1f}")
    if STEREO_AUTO_EXPOSURE is not None:
        actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        mode = "auto" if STEREO_AUTO_EXPOSURE else "manual"
        print(f"{label} exposure mode={mode} value={actual_exposure:.3f}")


def open_overhead_camera():
    from hardware.cameras.overhead_camera import SimpleOverheadCamera

    return SimpleOverheadCamera().cap
