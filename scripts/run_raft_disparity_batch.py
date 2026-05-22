"""
run_raft_disparity_batch.py

Batch RAFT-Stereo disparity computation over a directory of left/right image pairs.

No argparse.  Edit the USER SETTINGS block below to configure paths and options.

Run:
    python scripts/run_raft_disparity_batch.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

LEFT_IMAGE_DIR  = "data/left"    # directory containing left-frame images
RIGHT_IMAGE_DIR = "data/right"   # directory containing right-frame images (matched by filename)
OUTPUT_DIR      = "data/disparity_out"

RAFT_ROOT            = "RAFT-Stereo"
RAFT_CHECKPOINT_PATH = "RAFT-Stereo/models/raftstereo-middlebury.pth"

SAVE_DISPARITY_NPY     = True     # save raw float32 disparity as .npy
SAVE_DISPARITY_PREVIEW = True     # save colour-mapped preview as .png
OVERWRITE_EXISTING     = False    # skip pairs whose output already exists

MAX_IMAGE_PAIRS: int | None = None  # None = process all pairs
START_INDEX     = 0                 # skip the first N pairs

USE_CUDA  = True
USE_HALF  = True

# RAFT architecture knobs — must match the checkpoint
RAFT_CORR_IMPLEMENTATION = "alt"
RAFT_CONTEXT_NORM        = "batch"
RAFT_N_DOWNSAMPLE        = 2
RAFT_N_GRU_LAYERS        = 3
RAFT_VALID_ITERS         = 16
RAFT_DOWNSCALE           = 1.0

# ============================================================
# End of user settings
# ============================================================

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from vision.torch_device import select_torch_device
from vision.raft_runner import RAFTStereoRunner

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_pair(left_path: Path, right_path: Path):
    left = cv2.imread(str(left_path))
    right = cv2.imread(str(right_path))
    if left is None:
        print(f"[WARN] Could not load left image: {left_path}")
        return None, None
    if right is None:
        print(f"[WARN] Could not load right image: {right_path}")
        return None, None
    return left, right


def _save_outputs(disparity: np.ndarray, out_stem: Path) -> None:
    if SAVE_DISPARITY_NPY:
        npy_path = out_stem.with_suffix(".npy")
        np.save(str(npy_path), disparity)

    if SAVE_DISPARITY_PREVIEW:
        png_path = out_stem.with_suffix(".png")
        d_min = float(disparity.min())
        d_max = float(disparity.max())
        if d_max > d_min:
            preview = ((disparity - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            preview = np.zeros_like(disparity, dtype=np.uint8)
        coloured = cv2.applyColorMap(preview, cv2.COLORMAP_MAGMA)
        cv2.imwrite(str(png_path), coloured)


def main() -> int:
    left_dir  = Path(LEFT_IMAGE_DIR)
    right_dir = Path(RIGHT_IMAGE_DIR)
    out_dir   = Path(OUTPUT_DIR)

    if not left_dir.exists():
        print(f"[ERROR] LEFT_IMAGE_DIR does not exist: {left_dir.resolve()}")
        return 1
    if not right_dir.exists():
        print(f"[ERROR] RIGHT_IMAGE_DIR does not exist: {right_dir.resolve()}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    left_images = sorted(p for p in left_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not left_images:
        print(f"[ERROR] No images found in {left_dir.resolve()}")
        return 1

    left_images = left_images[START_INDEX:]
    if MAX_IMAGE_PAIRS is not None:
        left_images = left_images[:MAX_IMAGE_PAIRS]

    print(f"[BATCH] {len(left_images)} image pair(s) to process")

    device_info = select_torch_device(print_info=True, use_cuda=USE_CUDA, use_half=USE_HALF)
    raft = RAFTStereoRunner(
        raft_root=RAFT_ROOT,
        checkpoint_path=RAFT_CHECKPOINT_PATH,
        device_info=device_info,
        corr_implementation=RAFT_CORR_IMPLEMENTATION,
        context_norm=RAFT_CONTEXT_NORM,
        n_downsample=RAFT_N_DOWNSAMPLE,
        n_gru_layers=RAFT_N_GRU_LAYERS,
        valid_iters=RAFT_VALID_ITERS,
        downscale=RAFT_DOWNSCALE,
        warmup=True,
        warmup_iters=1,
    )
    raft.warmup()

    processed = 0
    skipped = 0
    errors = 0

    for left_path in left_images:
        right_path = right_dir / left_path.name
        if not right_path.exists():
            print(f"[SKIP] No matching right image for {left_path.name}")
            skipped += 1
            continue

        out_stem = out_dir / left_path.stem
        if not OVERWRITE_EXISTING:
            if (SAVE_DISPARITY_NPY and out_stem.with_suffix(".npy").exists()) or \
               (SAVE_DISPARITY_PREVIEW and out_stem.with_suffix(".png").exists()):
                print(f"[SKIP] Output exists for {left_path.name}")
                skipped += 1
                continue

        left_bgr, right_bgr = _load_pair(left_path, right_path)
        if left_bgr is None:
            errors += 1
            continue

        try:
            disparity = raft.predict_disparity(left_bgr, right_bgr, color="BGR")
        except Exception as exc:
            print(f"[ERROR] RAFT failed for {left_path.name}: {exc}")
            errors += 1
            continue

        _save_outputs(disparity, out_stem)
        processed += 1
        print(f"[OK] {left_path.name}  disp_range=[{disparity.min():.1f}, {disparity.max():.1f}]")

    print(
        f"\n[BATCH] Done. processed={processed}, skipped={skipped}, errors={errors}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
