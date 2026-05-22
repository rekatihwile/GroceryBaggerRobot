from __future__ import annotations

"""vision/torch_device.py

PyTorch device selection and CUDA helpers.

Extracted from run_pickplace_fast.py.  The module-level defaults match the
knobs in run_pickplace_fast.py so calling select_torch_device() without
arguments works identically to the original.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TorchDeviceInfo:
    torch: Any
    device: Any
    device_str: str
    cuda_available: bool
    cuda_selected: bool
    half_enabled: bool


def select_torch_device(
    *,
    print_info: bool = True,
    use_cuda: bool = True,
    torch_device: str = "cuda:0",
    use_half: bool = True,
) -> TorchDeviceInfo:
    """Select a torch device based on availability and user preferences.

    Args:
        print_info:    Print a [TORCH DEVICE] summary to stdout.
        use_cuda:      Allow CUDA if available.
        torch_device:  Requested CUDA device string, e.g. "cuda:0".
        use_half:      Enable float16 when CUDA is selected.

    Returns:
        TorchDeviceInfo with the resolved device and flags.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Could not import torch. YOLO/RAFT inference requires PyTorch."
        ) from exc

    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    selected = "cpu"

    if use_cuda and cuda_available:
        requested = str(torch_device or "cuda:0")
        selected = requested if requested.startswith("cuda") else "cuda:0"
        if ":" in selected:
            try:
                index = int(selected.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    print(
                        f"[DEVICE WARN] Requested {selected}, but only "
                        f"{torch.cuda.device_count()} CUDA device(s) visible. Using cuda:0."
                    )
                    selected = "cuda:0"
            except ValueError:
                print(f"[DEVICE WARN] Could not parse torch_device={selected!r}. Using cuda:0.")
                selected = "cuda:0"
    else:
        selected = "cpu"
        if use_cuda and not cuda_available:
            print(
                "[DEVICE WARN] CUDA requested but torch.cuda.is_available() is False."
                " Falling back to CPU."
            )

    device = torch.device(selected)
    cuda_selected = device.type == "cuda"
    half_enabled = bool(use_half and cuda_selected)

    if print_info:
        print("\n[TORCH DEVICE]")
        print(f"  torch version  = {torch.__version__}")
        print(f"  cuda available = {cuda_available}")
        print(f"  cuda version   = {cuda_version}")
        if cuda_available:
            try:
                print(f"  cuda devices   = {torch.cuda.device_count()}")
                print(f"  gpu name       = {torch.cuda.get_device_name(device if cuda_selected else 0)}")
            except Exception as exc:
                print(f"  gpu name       = unavailable ({exc})")
        print(f"  selected       = {device}")
        print(f"  half enabled   = {half_enabled}\n")

    return TorchDeviceInfo(
        torch=torch,
        device=device,
        device_str=str(device),
        cuda_available=cuda_available,
        cuda_selected=cuda_selected,
        half_enabled=half_enabled,
    )


def synchronize_if_cuda(device_info: TorchDeviceInfo) -> None:
    if device_info.cuda_selected:
        device_info.torch.cuda.synchronize(device_info.device)


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and (
        "out of memory" in text or "cublas" in text or "cudnn" in text
    )


def print_gpu_memory_hint(prefix: str = "[GPU]") -> None:
    print(
        f"{prefix} CUDA memory issue. Try lowering YOLO_IMGSZ, setting RAFT_DOWNSCALE < 1, "
        "lowering RAFT_VALID_ITERS, or keeping SURVEY_RAFT_MODE='best_frame_only'."
    )
