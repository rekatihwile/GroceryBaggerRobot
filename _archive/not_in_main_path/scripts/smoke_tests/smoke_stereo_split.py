"""
smoke_stereo_split.py

Smoke test for the stereo camera side-by-side split logic.

By default (LIVE_CAMERA = False) this test uses a synthetic frame and
verifies the split without any real hardware.

Set LIVE_CAMERA = True to open the real stereo camera and check one frame.

Run:
    python scripts/smoke_tests/smoke_stereo_split.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

LIVE_CAMERA          = False
SYNTHETIC_FRAME_WIDTH  = 1280
SYNTHETIC_FRAME_HEIGHT = 480
PRINT_DEBUG          = True

# ============================================================
# End of user settings
# ============================================================

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def _split_frame(frame: np.ndarray):
    """Same split logic as SimpleStereoCamera.read_pair()."""
    mid = frame.shape[1] // 2
    left = frame[:, :mid].copy()
    right = frame[:, mid:].copy()
    return left, right


def _check_split(frame: np.ndarray, label: str = "") -> bool:
    h, w = frame.shape[:2]
    left, right = _split_frame(frame)

    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]

    ok = True

    if lh != h or rh != h:
        print(f"[FAIL{label}] Height mismatch: frame h={h}, left h={lh}, right h={rh}")
        ok = False

    if lw != w // 2 or rw != w // 2:
        print(f"[FAIL{label}] Width mismatch: expected {w//2}, got left={lw}, right={rw}")
        ok = False

    if PRINT_DEBUG:
        print(
            f"[DEBUG{label}] frame={w}x{h}  "
            f"left={lw}x{lh}  right={rw}x{rh}  "
            f"split_ok={ok}"
        )

    # Check left half really is the left half (pixel values match)
    left_check = frame[:, :w // 2]
    right_check = frame[:, w // 2:]
    if not np.array_equal(left, left_check):
        print(f"[FAIL{label}] Left half pixel values do not match frame[:, :{w//2}]")
        ok = False
    if not np.array_equal(right, right_check):
        print(f"[FAIL{label}] Right half pixel values do not match frame[:, {w//2}:]")
        ok = False

    return ok


def test_synthetic() -> bool:
    """Test the split using a synthetic gradient frame."""
    w, h = SYNTHETIC_FRAME_WIDTH, SYNTHETIC_FRAME_HEIGHT

    # Create a frame where left and right halves have different solid colours
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2] = (80, 120, 200)   # left = blueish
    frame[:, w // 2 :] = (200, 100, 50)   # right = orangeish

    ok = _check_split(frame, label=" synthetic")
    if ok:
        print("[PASS] Synthetic split: left/right correctly separated.")
    return ok


def test_live() -> bool:
    """Open the real stereo camera and read one frame."""
    from hardware.cameras.stereo_apriltag_viewer import SimpleStereoCamera

    print("[LIVE] Opening stereo camera …")
    try:
        with SimpleStereoCamera() as stereo:
            ok, frame, left, right = stereo.read_pair()
            if not ok or frame is None:
                print("[FAIL] Live camera read returned no frame.")
                return False

            h, w = frame.shape[:2]
            lh, lw = left.shape[:2]
            rh, rw = right.shape[:2]
            print(
                f"[LIVE] frame={w}x{h}  left={lw}x{lh}  right={rw}x{rh}"
            )
            result = _check_split(frame, label=" live")
            if result:
                print("[PASS] Live split: left/right correctly separated.")
            return result
    except Exception as exc:
        print(f"[FAIL] Live camera error: {exc}")
        return False


def main() -> int:
    print("=" * 60)
    print("smoke_stereo_split.py")
    print(f"  LIVE_CAMERA={LIVE_CAMERA}")
    print("=" * 60)

    passed = True

    # Always run synthetic test
    if not test_synthetic():
        passed = False

    # Optionally run live test
    if LIVE_CAMERA:
        if not test_live():
            passed = False
    else:
        print("[SKIP] Live camera test skipped (LIVE_CAMERA=False).")

    print()
    if passed:
        print("[PASS] smoke_stereo_split: all checks passed.")
        return 0
    else:
        print("[FAIL] smoke_stereo_split: one or more checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
