from __future__ import annotations

"""vision/raft_runner.py

RAFT-Stereo disparity runner.

Extracted from run_pickplace_fast.py.  All knobs that were previously module-level
constants are now constructor kwargs with matching defaults.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from vision.torch_device import (
    TorchDeviceInfo,
    is_cuda_oom,
    print_gpu_memory_hint,
    select_torch_device,
    synchronize_if_cuda,
)

# Defaults matching run_pickplace_fast.py
_DEFAULT_RAFT_ROOT = "RAFT-Stereo"
_DEFAULT_CHECKPOINT = "RAFT-Stereo/models/raftstereo-middlebury.pth"


class RAFTStereoRunner:
    """Thin wrapper around RAFT-Stereo that manages model loading and inference.

    All RAFT architecture knobs are exposed as constructor kwargs so that callers
    can override defaults without touching module-level constants.  The defaults
    exactly match the knobs in run_pickplace_fast.py.
    """

    def __init__(
        self,
        raft_root: str | Path = _DEFAULT_RAFT_ROOT,
        checkpoint_path: str | Path = _DEFAULT_CHECKPOINT,
        device_info: TorchDeviceInfo | None = None,
        *,
        # Architecture knobs (must match the checkpoint's training config)
        corr_implementation: str = "alt",
        context_norm: str = "batch",
        shared_backbone: bool = False,
        n_downsample: int = 2,
        n_gru_layers: int = 3,
        slow_fast_gru: bool = False,
        # Inference knobs
        mixed_precision: bool = True,
        valid_iters: int = 16,
        downscale: float = 1.0,
        # Caching
        cache_last_disparity: bool = True,
        reuse_if_stable: bool = True,
        object_stability_px: float = 15.0,
        # Warmup
        warmup: bool = True,
        warmup_iters: int = 2,
    ) -> None:
        self.raft_root = Path(raft_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.device_info = device_info or select_torch_device(print_info=False)

        # Architecture knobs
        self.corr_implementation = corr_implementation
        self.context_norm = context_norm
        self.shared_backbone = shared_backbone
        self.n_downsample = n_downsample
        self.n_gru_layers = n_gru_layers
        self.slow_fast_gru = slow_fast_gru

        # Inference knobs
        self.mixed_precision = bool(mixed_precision and self.device_info.cuda_selected)
        self.valid_iters = int(valid_iters)
        self.downscale = float(downscale)

        # Caching
        self.cache_last_disparity = bool(cache_last_disparity)
        self.reuse_if_stable = bool(reuse_if_stable)
        self.object_stability_px = float(object_stability_px)
        self.last_disparity: np.ndarray | None = None
        self.last_frame_shape: tuple[int, int] | None = None
        self.last_centroid_px: np.ndarray | None = None

        # Warmup
        self.warmup_enabled = bool(warmup)
        self.warmup_iters = int(warmup_iters)

        self._validate_paths()
        self._add_raft_to_syspath()

        try:
            from core.raft_stereo import RAFTStereo
            from core.utils.utils import InputPadder
        except Exception as exc:
            raise RuntimeError(
                "Could not import RAFT-Stereo.\n"
                "  Expected: from core.raft_stereo import RAFTStereo\n"
                "  Expected: from core.utils.utils import InputPadder\n"
                f"  RAFT root: {self.raft_root.resolve()}\n"
                f"  Original error: {exc}"
            ) from exc

        self.torch = self.device_info.torch
        self.InputPadder = InputPadder
        self.device = self.device_info.device

        args = self._build_args()
        model = RAFTStereo(args)
        state = self._load_checkpoint(self.torch, self.checkpoint_path, self.device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RAFT checkpoint architecture mismatch: {self.checkpoint_path}\n"
                "Adjust the RAFT architecture knobs to match the checkpoint.\n"
                f"Original error: {exc}"
            ) from exc

        self.model = model.to(self.device)
        self.model.eval()

        print("[RAFT] Loaded")
        print(f"  root       = {self.raft_root.resolve()}")
        print(f"  checkpoint = {self.checkpoint_path.resolve()}")
        print(f"  device     = {self.device}")
        print(f"  half/amp   = {self.mixed_precision}")
        print(f"  valid_iters= {self.valid_iters}")
        print(f"  downscale  = {self.downscale}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _validate_paths(self) -> None:
        if not self.raft_root.exists():
            raise FileNotFoundError(
                f"Missing RAFT-Stereo folder: {self.raft_root.resolve()}. "
                "Place the RAFT-Stereo repo in the project root."
            )
        for sub in ("core", ):
            if not (self.raft_root / sub).exists():
                raise FileNotFoundError(
                    f"Missing RAFT-Stereo subfolder: {(self.raft_root / sub).resolve()}"
                )
        if not self.checkpoint_path.exists():
            available = sorted(str(p) for p in (self.raft_root / "models").glob("*.pth"))
            raise FileNotFoundError(
                f"Missing RAFT checkpoint: {self.checkpoint_path.resolve()}\n"
                f"Available: {available}"
            )

    def _add_raft_to_syspath(self) -> None:
        abs_path = str(self.raft_root.resolve())
        if abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    def _build_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            hidden_dims=[128] * 3,
            corr_implementation=self.corr_implementation,
            shared_backbone=self.shared_backbone,
            corr_levels=4,
            corr_radius=4,
            n_downsample=self.n_downsample,
            context_norm=self.context_norm,
            slow_fast_gru=self.slow_fast_gru,
            n_gru_layers=self.n_gru_layers,
            mixed_precision=self.mixed_precision,
        )

    @staticmethod
    def _load_checkpoint(torch_module: Any, path: Path, device: Any) -> dict[str, Any]:
        try:
            state = torch_module.load(path, map_location=device, weights_only=False)
        except TypeError:
            state = torch_module.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected RAFT checkpoint format: {type(state)}")
        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict_disparity(
        self,
        left_bgr_or_rgb: np.ndarray,
        right_bgr_or_rgb: np.ndarray,
        *,
        color: str = "BGR",
    ) -> np.ndarray:
        """Compute dense disparity (pixels) from a rectified stereo pair.

        Returns a float32 HxW array. Positive disparity = left image
        feature is to the right of the corresponding right image feature,
        consistent with the standard stereo convention.
        """
        if left_bgr_or_rgb is None or right_bgr_or_rgb is None:
            raise ValueError("predict_disparity: left and right images must not be None.")
        if left_bgr_or_rgb.shape[:2] != right_bgr_or_rgb.shape[:2]:
            raise ValueError(
                f"Left/right shape mismatch: {left_bgr_or_rgb.shape} vs {right_bgr_or_rgb.shape}"
            )

        h, w = left_bgr_or_rgb.shape[:2]
        scale = float(self.downscale)
        if scale <= 0.0 or scale > 1.0:
            raise ValueError("downscale must be in (0, 1].")

        if scale < 1.0:
            run_w = max(32, int(round(w * scale)))
            run_h = max(32, int(round(h * scale)))
            left_run = cv2.resize(left_bgr_or_rgb, (run_w, run_h), interpolation=cv2.INTER_AREA)
            right_run = cv2.resize(right_bgr_or_rgb, (run_w, run_h), interpolation=cv2.INTER_AREA)
        else:
            run_h, run_w = h, w
            left_run = left_bgr_or_rgb
            right_run = right_bgr_or_rgb

        image1 = self._to_tensor(left_run, color=color)
        image2 = self._to_tensor(right_run, color=color)

        padder = self.InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        try:
            with self.torch.no_grad():
                with self.torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    _, flow_up = self.model(image1, image2, iters=self.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                print_gpu_memory_hint("[RAFT]")
            raise

        disparity = (-flow_up).detach().float().cpu().numpy().astype(np.float32)

        if scale < 1.0:
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
            disparity = disparity / scale
        elif disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)

        return disparity

    def _to_tensor(self, image: np.ndarray, *, color: str) -> Any:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
        if color.upper() == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color.upper() != "RGB":
            raise ValueError("color must be 'BGR' or 'RGB'")
        tensor = self.torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return tensor[None].to(self.device)

    # ------------------------------------------------------------------ #
    # Disparity caching
    # ------------------------------------------------------------------ #

    def can_reuse_cache(
        self, frame_shape: tuple[int, int], centroid_px: np.ndarray
    ) -> bool:
        if not self.cache_last_disparity or not self.reuse_if_stable:
            return False
        if self.last_disparity is None or self.last_frame_shape != tuple(frame_shape):
            return False
        if self.last_centroid_px is None:
            return False
        delta = float(
            np.linalg.norm(
                np.asarray(centroid_px, dtype=np.float64) - self.last_centroid_px
            )
        )
        return delta <= self.object_stability_px

    def remember_cache(
        self,
        disparity: np.ndarray,
        frame_shape: tuple[int, int],
        centroid_px: np.ndarray,
    ) -> None:
        if not self.cache_last_disparity:
            return
        self.last_disparity = disparity
        self.last_frame_shape = tuple(frame_shape)
        self.last_centroid_px = np.asarray(centroid_px, dtype=np.float64).reshape(2)

    # ------------------------------------------------------------------ #
    # Warmup
    # ------------------------------------------------------------------ #

    def warmup(self, size_hw: tuple[int, int] = (320, 320)) -> None:
        if not self.warmup_enabled:
            return
        h, w = int(size_hw[0]), int(size_hw[1])
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        synchronize_if_cuda(self.device_info)
        t0 = time.perf_counter()
        for _ in range(self.warmup_iters):
            _ = self.predict_disparity(dummy, dummy, color="BGR")
        synchronize_if_cuda(self.device_info)
        dt = time.perf_counter() - t0
        print(
            f"[RAFT] Warmup: iters={self.warmup_iters} size={w}x{h} "
            f"time={dt:.3f}s device={self.device}"
        )
