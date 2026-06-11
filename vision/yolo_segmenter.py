from __future__ import annotations

"""vision/yolo_segmenter.py

YOLO segmentation wrapper.

Extracted from run_pickplace_fast.py.  All knobs that were previously
module-level constants in run_pickplace_fast.py are now constructor kwargs
with the same default values so existing call sites need no changes.
"""

import time
from dataclasses import dataclass
from pathlib import Path
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

# Default fallback weights path (matches run_pickplace_fast.py YOLO_FALLBACK_WEIGHTS_PATH)
_DEFAULT_FALLBACK = "full_data.pt"


# ------------------------------------------------------------------ #
# Data container
# ------------------------------------------------------------------ #

@dataclass
class YOLODetection:
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    mask_area: int
    centroid_px: np.ndarray
    major_axis_length_px: float
    minor_axis_length_px: float
    major_axis_angle_deg: float
    minor_axis_angle_deg: float


# ------------------------------------------------------------------ #
# Mask helpers
# ------------------------------------------------------------------ #

def largest_mask_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def compute_mask_axis_metrics(mask: np.ndarray, *, min_pixels: int = 10) -> dict[str, Any]:
    """PCA mask axis metrics.

    Preserved verbatim from run_pickplace_fast.py.  The key convention:
        major_phi = atan2(major_vec_y, major_vec_x) % 180
        minor_phi = (major_phi + 90) % 180
    """
    component = largest_mask_component(mask)
    ys, xs = np.nonzero(component)
    if len(xs) < min_pixels:
        return {
            "component_mask": component,
            "mask_area_px": int(len(xs)),
            "centroid_px": np.array([np.nan, np.nan], dtype=np.float64),
            "major_axis_length_px": 0.0,
            "minor_axis_length_px": 0.0,
            "major_axis_angle_deg": 0.0,
            "minor_axis_angle_deg": 90.0,
        }

    moments = cv2.moments(component, binaryImage=True)
    if abs(moments["m00"]) < 1e-9:
        centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    else:
        centroid = np.array(
            [
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            ],
            dtype=np.float64,
        )

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = coords - centroid.reshape(1, 2)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    major_vec = eigenvectors[:, 0]
    projections = centered @ eigenvectors
    major_px = float(np.ptp(projections[:, 0]))
    minor_px = float(np.ptp(projections[:, 1]))

    major_phi = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])) % 180.0)
    minor_phi = float((major_phi + 90.0) % 180.0)

    return {
        "component_mask": component,
        "mask_area_px": int(len(xs)),
        "centroid_px": centroid,
        "major_axis_length_px": major_px,
        "minor_axis_length_px": minor_px,
        "major_axis_angle_deg": major_phi,
        "minor_axis_angle_deg": minor_phi,
    }


# ------------------------------------------------------------------ #
# YOLO segmenter class
# ------------------------------------------------------------------ #

class YOLOSegmenter:
    """YOLO segmentation model wrapper.

    Constructor kwargs mirror the module-level knobs from run_pickplace_fast.py
    so existing call sites (YOLOSegmenter(weights, device_info=...)) work
    without changes.  Pass knobs explicitly to override defaults.
    """

    def __init__(
        self,
        weights_path: str | Path = "yolo_weights/best.pt",
        device_info: TorchDeviceInfo | None = None,
        *,
        fallback_weights_path: str | Path = _DEFAULT_FALLBACK,
        yolo_device_str: str | None = None,    # override device (default = device_info.device)
        use_half: bool = True,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.50,
        retina_masks: bool = True,
        min_mask_area_px: int = 500,
        target_class_names: list[str] | None = None,
        mask_axis_min_pixels: int = 10,
        warmup_enabled: bool = True,
        warmup_imgsz: int = 640,
        warmup_iters: int = 3,
    ) -> None:
        self.device_info = device_info or select_torch_device(print_info=False)
        self.weights_path = self._resolve_weights(
            Path(weights_path), Path(fallback_weights_path)
        )

        # Determine YOLO device string
        configured = yolo_device_str or str(self.device_info.device)
        self.yolo_device = (
            configured
            if self.device_info.cuda_selected and configured.startswith("cuda")
            else str(self.device_info.device)
        )
        self.yolo_half = bool(use_half and self.device_info.cuda_selected)

        # Inference knobs stored as instance vars
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.retina_masks = bool(retina_masks)
        self.min_mask_area_px = int(min_mask_area_px)
        self.target_class_names: list[str] = list(target_class_names) if target_class_names else []
        self.mask_axis_min_pixels = int(mask_axis_min_pixels)
        self.warmup_enabled = bool(warmup_enabled)
        self.warmup_imgsz = int(warmup_imgsz)
        self.warmup_iters = int(warmup_iters)

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Could not import ultralytics.YOLO. "
                "Install ultralytics in this Python environment."
            ) from exc

        self.model = YOLO(str(self.weights_path), task="segment")
        try:
            self.model.to(self.yolo_device)
        except Exception as exc:
            print(f"[YOLO WARN] model.to({self.yolo_device!r}) failed: {exc}")

        self.names = self.model.names or {}

        print("[YOLO] Loaded")
        print(f"  weights = {self.weights_path.resolve()}")
        print(f"  device  = {self.yolo_device}")
        print(f"  half    = {self.yolo_half}")
        print(f"  imgsz   = {self.imgsz}")
        print(f"  conf    = {self.conf:.2f}")
        print(f"  iou     = {self.iou:.2f}")
        print(f"  retina  = {self.retina_masks}")

    @staticmethod
    def _resolve_weights(path: Path, fallback: Path) -> Path:
        if path.exists():
            return path
        if fallback.exists():
            print(
                f"[YOLO] Expected {path} is missing; "
                f"using local fallback {fallback.resolve()}"
            )
            return fallback
        raise FileNotFoundError(
            f"Missing YOLO weights. Tried:\n"
            f"  {path.resolve()}\n"
            f"  {fallback.resolve()}"
        )

    def segment(self, image_bgr: np.ndarray) -> list[YOLODetection]:
        return self.segment_batch([image_bgr])[0]

    def segment_batch(self, images_bgr: list[np.ndarray]) -> list[list[YOLODetection]]:
        if not images_bgr:
            return []

        try:
            results = self.model(
                images_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                retina_masks=self.retina_masks,
                verbose=False,
                device=self.yolo_device,
                half=self.yolo_half,
            )
            synchronize_if_cuda(self.device_info)
        except RuntimeError as exc:
            if is_cuda_oom(exc):
                print_gpu_memory_hint("[YOLO]")
            raise

        if not results:
            return [[] for _ in images_bgr]

        detections_by_frame: list[list[YOLODetection]] = []
        for image, result in zip(images_bgr, results):
            detections_by_frame.append(self._result_to_detections(result, image.shape[:2]))
        while len(detections_by_frame) < len(images_bgr):
            detections_by_frame.append([])
        return detections_by_frame

    def _result_to_detections(
        self, result: Any, image_shape_hw: tuple[int, int]
    ) -> list[YOLODetection]:
        if result.masks is None or result.boxes is None:
            return []

        h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
        masks = result.masks.data.detach().cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)

        out: list[YOLODetection] = []
        for i in range(len(masks)):
            mask = masks[i] > 0.5
            if mask.shape != (h, w):
                mask = (
                    cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                )

            axis_metrics = compute_mask_axis_metrics(
                mask, min_pixels=self.mask_axis_min_pixels
            )
            area = int(axis_metrics["mask_area_px"])
            if area < self.min_mask_area_px:
                continue

            class_id = int(class_ids[i])
            class_name = str(self.names.get(class_id, class_id))
            if self.target_class_names and class_name not in self.target_class_names:
                continue

            bbox = tuple(float(v) for v in boxes_xyxy[i])
            out.append(
                YOLODetection(
                    mask=axis_metrics["component_mask"].astype(bool),
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[i]),
                    mask_area=area,
                    centroid_px=axis_metrics["centroid_px"],
                    major_axis_length_px=float(axis_metrics["major_axis_length_px"]),
                    minor_axis_length_px=float(axis_metrics["minor_axis_length_px"]),
                    major_axis_angle_deg=float(axis_metrics["major_axis_angle_deg"]),
                    minor_axis_angle_deg=float(axis_metrics["minor_axis_angle_deg"]),
                )
            )

        return out

    def warmup(self) -> None:
        if not self.warmup_enabled:
            return
        dummy = np.zeros((self.warmup_imgsz, self.warmup_imgsz, 3), dtype=np.uint8)
        synchronize_if_cuda(self.device_info)
        t0 = time.perf_counter()
        for _ in range(self.warmup_iters):
            _ = self.segment(dummy)
        synchronize_if_cuda(self.device_info)
        dt = time.perf_counter() - t0
        print(
            f"[YOLO] Warmup complete: iters={self.warmup_iters} "
            f"imgsz={self.warmup_imgsz} time={dt:.3f}s device={self.yolo_device}"
        )
