from __future__ import annotations

"""
Random YOLO + RAFT point-cloud validation script.

Run from:
    C:\\Users\\elipp\\OneDrive\\Documents\\Grocery_Buildup

Expected structure:
    Grocery_Buildup/
        random_yolo_raft_pointcloud_validation.py
        Training_Images/
            Stereo_Left_0001.jpg
            Stereo_Right_0001.jpg
            Overhead_Webcam_0001.jpg
            ...
        stereo_calibration.npz
        RAFT-Stereo/
            models/
                raftstereo-middlebury.pth
        yolo_weights/
            best.pt

What it does:
    1. Randomly chooses one Stereo_Left_#### + Stereo_Right_#### pair.
    2. Runs YOLO segmentation on the LEFT image only.
    3. Runs RAFT-Stereo disparity on the stereo pair.
    4. Converts masked disparity pixels into XYZ point clouds.
    5. Saves .ply files for CloudCompare.

Outputs:
    PointCloud_Validation/
        sample_####_YYYYMMDD_HHMMSS/
            left.jpg
            right.jpg
            yolo_overlay.jpg
            disparity_heatmap.jpg
            disparity_raw.npy
            combined_segmented_pointcloud.ply
            instance_00_Class_conf0.95.ply
            instance_01_Class_conf0.88.ply
            summary.txt
"""

import argparse
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


# ============================================================
# DEFAULT PATHS / KNOBS
# ============================================================

ROOT = Path(__file__).resolve().parent

TRAINING_IMAGES_DIR = ROOT / "Training_Images"
STEREO_CALIBRATION_PATH = ROOT / "stereo_calibration.npz"

YOLO_WEIGHTS_CANDIDATES = [
    ROOT / "yolo_weights" / "full_data.pt",
    ROOT / "yolo_weights" / "best.pt",
    ROOT / "weights" / "best.pt",
]

RAFT_ROOT = ROOT / "RAFT-Stereo"
RAFT_CHECKPOINT_PATH = RAFT_ROOT / "models" / "raftstereo-middlebury.pth"

OUTPUT_ROOT = ROOT / "PointCloud_Validation"

YOLO_IMGSZ = 960
YOLO_CONF = 0.25
YOLO_IOU = 0.50
YOLO_RETINA_MASKS = True

RAFT_VALID_ITERS = 16
RAFT_MIXED_PRECISION = True
RAFT_CORR_IMPLEMENTATION = "alt"
RAFT_CONTEXT_NORM = "batch"
RAFT_SHARED_BACKBONE = False
RAFT_N_DOWNSAMPLE = 2
RAFT_N_GRU_LAYERS = 3
RAFT_SLOW_FAST_GRU = False

MIN_DISPARITY_PX = 1.0
MAX_POINTS_PER_INSTANCE = 50000

# For quick validation in CloudCompare, use rectified camera coordinates.
# If Z direction looks flipped, toggle this.
FLIP_Z_FOR_DISPLAY = False

# If your Training_Images are already rectified from the capture script, leave False.
# If you accidentally saved raw split stereo images, set --rectify-inputs.
RECTIFY_INPUTS_DEFAULT = False


# ============================================================
# DATA
# ============================================================

@dataclass
class StereoPair:
    index: int
    left_path: Path
    right_path: Path


@dataclass
class Detection:
    instance_i: int
    class_id: int
    class_name: str
    confidence: float
    mask: np.ndarray  # bool HxW
    bbox_xyxy: tuple[float, float, float, float]


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_stereo_pairs(training_dir: Path) -> list[StereoPair]:
    if not training_dir.exists():
        raise FileNotFoundError(f"Missing Training_Images folder: {training_dir}")

    left_re = re.compile(r"Stereo_Left_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    right_re = re.compile(r"Stereo_Right_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

    lefts: dict[int, Path] = {}
    rights: dict[int, Path] = {}

    for p in training_dir.iterdir():
        m = left_re.match(p.name)
        if m:
            lefts[int(m.group(1))] = p
            continue

        m = right_re.match(p.name)
        if m:
            rights[int(m.group(1))] = p
            continue

    common = sorted(set(lefts.keys()) & set(rights.keys()))

    pairs = [
        StereoPair(index=i, left_path=lefts[i], right_path=rights[i])
        for i in common
    ]

    if not pairs:
        raise FileNotFoundError(
            f"No Stereo_Left_#### / Stereo_Right_#### pairs found in {training_dir}"
        )

    return pairs


def resolve_yolo_weights(path_arg: str | None) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"YOLO weights not found: {p}")
        return p

    for p in YOLO_WEIGHTS_CANDIDATES:
        if p.exists():
            return p

    # Fallback: search project root for likely best.pt files.
    found = sorted(ROOT.glob("**/best.pt"))
    if found:
        print("[YOLO] Auto-found possible weights:")
        for i, p in enumerate(found):
            print(f"  [{i}] {p}")
        return found[0]

    raise FileNotFoundError(
        "Could not find YOLO weights. Put best.pt at one of:\n"
        + "\n".join(f"  {p}" for p in YOLO_WEIGHTS_CANDIDATES)
        + "\nOr run with --weights path\\to\\best.pt"
    )


# ============================================================
# CALIBRATION / RECTIFICATION
# ============================================================

def load_stereo_calibration(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing stereo calibration: {path}")

    data = np.load(path, allow_pickle=False)
    calib = {k: np.asarray(data[k]) for k in data.files}

    print(f"[CALIB] Loaded {path}")
    print("[CALIB] Keys:")
    for k in sorted(calib.keys()):
        print(f"  - {k}: {calib[k].shape}")

    return calib


class StereoRectifier:
    def __init__(self, stereo_calib: dict[str, np.ndarray]) -> None:
        self.calib = stereo_calib
        self._maps_by_size: dict[
            tuple[int, int],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        required = [
            "left_camera_matrix",
            "left_distortion_coefficients",
            "right_camera_matrix",
            "right_distortion_coefficients",
            "rectification_left",
            "rectification_right",
            "projection_left_rectified",
            "projection_right_rectified",
        ]

        missing = [k for k in required if k not in stereo_calib]
        if missing:
            raise KeyError(
                f"Stereo calibration missing keys: {missing}\n"
                f"Available keys: {sorted(stereo_calib.keys())}"
            )

    def rectify(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError(f"Stereo size mismatch: {left_bgr.shape} vs {right_bgr.shape}")

        h, w = left_bgr.shape[:2]
        key = (w, h)

        if key not in self._maps_by_size:
            self._maps_by_size[key] = self._build_maps(w, h)

        map_lx, map_ly, map_rx, map_ry = self._maps_by_size[key]

        left_rect = cv2.remap(left_bgr, map_lx, map_ly, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_bgr, map_rx, map_ry, cv2.INTER_LINEAR)

        return left_rect, right_rect

    def _build_maps(self, w: int, h: int):
        c = self.calib

        map_lx, map_ly = cv2.initUndistortRectifyMap(
            np.asarray(c["left_camera_matrix"], dtype=np.float64),
            np.asarray(c["left_distortion_coefficients"], dtype=np.float64),
            np.asarray(c["rectification_left"], dtype=np.float64),
            np.asarray(c["projection_left_rectified"], dtype=np.float64),
            (int(w), int(h)),
            cv2.CV_32FC1,
        )

        map_rx, map_ry = cv2.initUndistortRectifyMap(
            np.asarray(c["right_camera_matrix"], dtype=np.float64),
            np.asarray(c["right_distortion_coefficients"], dtype=np.float64),
            np.asarray(c["rectification_right"], dtype=np.float64),
            np.asarray(c["projection_right_rectified"], dtype=np.float64),
            (int(w), int(h)),
            cv2.CV_32FC1,
        )

        print(f"[RECTIFY] Built maps for {w}x{h}")
        return map_lx, map_ly, map_rx, map_ry


# ============================================================
# YOLO SEGMENTATION
# ============================================================

def run_yolo_segmentation(
    weights_path: Path,
    left_bgr: np.ndarray,
    conf: float,
    imgsz: int,
) -> tuple[list[Detection], np.ndarray]:
    from ultralytics import YOLO

    print(f"[YOLO] Loading {weights_path}")
    model = YOLO(str(weights_path))

    print("[YOLO] Running segmentation on LEFT image only...")
    results = model.predict(
        source=left_bgr,
        task="segment",
        imgsz=imgsz,
        conf=conf,
        iou=YOLO_IOU,
        retina_masks=YOLO_RETINA_MASKS,
        verbose=False,
    )

    if not results:
        return [], left_bgr.copy()

    result = results[0]
    overlay = result.plot()

    if result.masks is None or result.boxes is None:
        print("[YOLO] No masks detected.")
        return [], overlay

    names = result.names or model.names or {}

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = result.boxes.conf.detach().cpu().numpy().astype(float)

    h, w = left_bgr.shape[:2]

    detections: list[Detection] = []

    for i in range(len(masks)):
        mask = masks[i] > 0.5

        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        cls_id = int(class_ids[i])
        class_name = str(names.get(cls_id, cls_id))
        conf_i = float(confidences[i])
        bbox = tuple(float(v) for v in boxes[i])

        detections.append(
            Detection(
                instance_i=i,
                class_id=cls_id,
                class_name=class_name,
                confidence=conf_i,
                mask=mask,
                bbox_xyxy=bbox,
            )
        )

    print(f"[YOLO] Found {len(detections)} instances:")
    for det in detections:
        print(f"  #{det.instance_i:02d} {det.class_name} conf={det.confidence:.3f}")

    return detections, overlay


# ============================================================
# RAFT-STEREO
# ============================================================

class RAFTStereoRunner:
    def __init__(
        self,
        raft_root: Path,
        checkpoint_path: Path,
        device: str | None = None,
    ) -> None:
        if not raft_root.exists():
            raise FileNotFoundError(f"Missing RAFT-Stereo folder: {raft_root}")
        if not (raft_root / "core").exists():
            raise FileNotFoundError(f"Missing RAFT-Stereo/core folder: {raft_root / 'core'}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing RAFT checkpoint: {checkpoint_path}")

        self.raft_root = raft_root
        self.checkpoint_path = checkpoint_path

        raft_root_abs = str(raft_root.resolve())
        if raft_root_abs not in sys.path:
            sys.path.insert(0, raft_root_abs)

        import torch
        from core.raft_stereo import RAFTStereo
        from core.utils.utils import InputPadder

        self.torch = torch
        self.InputPadder = InputPadder

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        args = SimpleNamespace(
            hidden_dims=[128] * 3,
            corr_implementation=RAFT_CORR_IMPLEMENTATION,
            shared_backbone=RAFT_SHARED_BACKBONE,
            corr_levels=4,
            corr_radius=4,
            n_downsample=RAFT_N_DOWNSAMPLE,
            context_norm=RAFT_CONTEXT_NORM,
            slow_fast_gru=RAFT_SLOW_FAST_GRU,
            n_gru_layers=RAFT_N_GRU_LAYERS,
            mixed_precision=bool(RAFT_MIXED_PRECISION and self.device.type == "cuda"),
        )

        print(f"[RAFT] Loading model from {checkpoint_path}")
        model = RAFTStereo(args)

        try:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

        model.load_state_dict(state, strict=True)
        model = model.to(self.device)
        model.eval()

        self.model = model

        print(f"[RAFT] Loaded on {self.device}")

    def predict_disparity(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError(f"RAFT image size mismatch: {left_bgr.shape} vs {right_bgr.shape}")

        image1 = self._image_to_tensor(left_bgr)
        image2 = self._image_to_tensor(right_bgr)

        padder = self.InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        print("[RAFT] Predicting disparity...")

        with self.torch.no_grad():
            with self.torch.cuda.amp.autocast(
                enabled=bool(RAFT_MIXED_PRECISION and self.device.type == "cuda")
            ):
                _, flow_up = self.model(
                    image1,
                    image2,
                    iters=RAFT_VALID_ITERS,
                    test_mode=True,
                )

            flow_up = padder.unpad(flow_up).squeeze()

        # RAFT-Stereo returns x-flow. Positive disparity is -flow_x.
        disparity = (-flow_up).detach().float().cpu().numpy().astype(np.float32)

        if disparity.shape != left_bgr.shape[:2]:
            disparity = cv2.resize(
                disparity,
                (left_bgr.shape[1], left_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        print(
            f"[RAFT] disparity shape={disparity.shape}, "
            f"min={np.nanmin(disparity):.3f}, max={np.nanmax(disparity):.3f}"
        )

        return disparity

    def _image_to_tensor(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float()
        return tensor[None].to(self.device)


# ============================================================
# DISPARITY → POINT CLOUD
# ============================================================

def disparity_to_xyz_rectified(
    disparity: np.ndarray,
    calib: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Converts entire disparity image into XYZ in rectified-left camera coordinates.
    Invalid disparity becomes NaN.
    """
    h, w = disparity.shape[:2]
    disp = np.asarray(disparity, dtype=np.float64)

    valid = np.isfinite(disp) & (disp > MIN_DISPARITY_PX)

    xyz = np.full((h, w, 3), np.nan, dtype=np.float64)

    ys, xs = np.nonzero(valid)

    if len(xs) == 0:
        return xyz

    d = disp[ys, xs]
    u = xs.astype(np.float64)
    v = ys.astype(np.float64)

    if "disparity_to_depth_Q" in calib:
        Q = np.asarray(calib["disparity_to_depth_Q"], dtype=np.float64)

        pts_h = np.column_stack([u, v, d, np.ones_like(d)]) @ Q.T
        ww = pts_h[:, 3]

        good = np.isfinite(ww) & (np.abs(ww) > 1e-9)

        pts = pts_h[good, :3] / ww[good, None]

        xyz[ys[good], xs[good], :] = pts

    else:
        P_left = np.asarray(calib["projection_left_rectified"], dtype=np.float64)
        fx = float(P_left[0, 0])
        fy = float(P_left[1, 1])
        cx = float(P_left[0, 2])
        cy = float(P_left[1, 2])

        if "baseline_mm" in calib:
            baseline = float(np.asarray(calib["baseline_mm"]).reshape(-1)[0])
        else:
            P_right = np.asarray(calib["projection_right_rectified"], dtype=np.float64)
            baseline = abs(float(P_right[0, 3]) / fx)

        z = fx * baseline / d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.column_stack([x, y, z])

        xyz[ys, xs, :] = pts

    if FLIP_Z_FOR_DISPLAY:
        xyz[..., 2] *= -1.0

    return xyz


def points_from_mask(
    xyz_img: np.ndarray,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != xyz_img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (xyz_img.shape[1], xyz_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    valid_xyz = np.all(np.isfinite(xyz_img), axis=2)
    valid = mask.astype(bool) & valid_xyz

    ys, xs = np.nonzero(valid)

    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    if len(xs) > max_points:
        choice = np.random.choice(len(xs), size=max_points, replace=False)
        xs = xs[choice]
        ys = ys[choice]

    pts = xyz_img[ys, xs, :].astype(np.float64)
    colors_bgr = image_bgr[ys, xs, :]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    return pts, colors_rgb


# ============================================================
# PLY WRITER
# ============================================================

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\\-]+", "_", name).strip("_")


def write_ply_xyzrgb_scalar(
    path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    class_ids: np.ndarray | None = None,
    instance_ids: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    points_xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)

    n = len(points_xyz)

    if class_ids is None:
        class_ids = np.zeros(n, dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)

    if instance_ids is None:
        instance_ids = np.zeros(n, dtype=np.int32)
    else:
        instance_ids = np.asarray(instance_ids, dtype=np.int32).reshape(-1)

    if not (len(colors_rgb) == len(class_ids) == len(instance_ids) == n):
        raise ValueError("PLY arrays have inconsistent lengths.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment Generated by random_yolo_raft_pointcloud_validation.py\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int class_id\n")
        f.write("property int instance_id\n")
        f.write("end_header\n")

        for p, c, cls, inst in zip(points_xyz, colors_rgb, class_ids, instance_ids):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{int(cls)} {int(inst)}\n"
            )

    print(f"[PLY] Wrote {n} points: {path}")


# ============================================================
# VISUALS
# ============================================================

def save_disparity_heatmap(path: Path, disparity: np.ndarray) -> None:
    disp = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > MIN_DISPARITY_PX)

    if not np.any(valid):
        heat = np.zeros((*disp.shape, 3), dtype=np.uint8)
    else:
        lo = float(np.percentile(disp[valid], 2))
        hi = float(np.percentile(disp[valid], 98))
        norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
        heat_u8 = (norm * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        heat[~valid] = 0

    cv2.imwrite(str(path), heat)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default=str(TRAINING_IMAGES_DIR))
    parser.add_argument("--calib", type=str, default=str(STEREO_CALIBRATION_PATH))
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--raft-root", type=str, default=str(RAFT_ROOT))
    parser.add_argument("--raft-ckpt", type=str, default=str(RAFT_CHECKPOINT_PATH))
    parser.add_argument("--index", type=int, default=None, help="Specific #### index to use.")
    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--rectify-inputs", action="store_true", default=RECTIFY_INPUTS_DEFAULT)
    parser.add_argument("--max-points", type=int, default=MAX_POINTS_PER_INSTANCE)
    args = parser.parse_args()

    training_dir = Path(args.images)
    calib_path = Path(args.calib)
    raft_root = Path(args.raft_root)
    raft_ckpt = Path(args.raft_ckpt)

    weights_path = resolve_yolo_weights(args.weights)
    pairs = find_stereo_pairs(training_dir)
    for image in pairs:
        if args.index is None:
            pair = image
        else:
            matches = [p for p in pairs if p.index == args.index]
            if not matches:
                available = [p.index for p in pairs[:10]]
                raise ValueError(f"No pair found for index {args.index}. First available indices: {available}")
            pair = matches[0]

        print("\n================ SELECTED PAIR ================")
        print("index:", pair.index)
        print("left :", pair.left_path)
        print("right:", pair.right_path)
        print("================================================\n")

        left = cv2.imread(str(pair.left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(pair.right_path), cv2.IMREAD_COLOR)

        if left is None:
            raise RuntimeError(f"Could not read left image: {pair.left_path}")
        if right is None:
            raise RuntimeError(f"Could not read right image: {pair.right_path}")

        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"Left/right image size mismatch: {left.shape} vs {right.shape}")

        calib = load_stereo_calibration(calib_path)

        if args.rectify_inputs:
            print("[INFO] --rectify-inputs enabled. Rectifying loaded stereo images.")
            rectifier = StereoRectifier(calib)
            left_proc, right_proc = rectifier.rectify(left, right)
        else:
            print("[INFO] Assuming Training_Images stereo pair is already rectified.")
            left_proc, right_proc = left, right

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"sample_{pair.index:04d}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / "left.jpg"), left_proc)
        cv2.imwrite(str(out_dir / "right.jpg"), right_proc)

        detections, overlay = run_yolo_segmentation(
            weights_path=weights_path,
            left_bgr=left_proc,
            conf=args.conf,
            imgsz=args.imgsz,
        )

        cv2.imwrite(str(out_dir / "yolo_overlay.jpg"), overlay)

        raft = RAFTStereoRunner(
            raft_root=raft_root,
            checkpoint_path=raft_ckpt,
        )

        disparity = raft.predict_disparity(left_proc, right_proc)

        np.save(out_dir / "disparity_raw.npy", disparity)
        save_disparity_heatmap(out_dir / "disparity_heatmap.jpg", disparity)

        xyz_img = disparity_to_xyz_rectified(disparity, calib)

        all_points = []
        all_colors = []
        all_class_ids = []
        all_instance_ids = []

        summary_lines = []
        summary_lines.append(f"Selected pair index: {pair.index}")
        summary_lines.append(f"Left image: {pair.left_path}")
        summary_lines.append(f"Right image: {pair.right_path}")
        summary_lines.append(f"YOLO weights: {weights_path}")
        summary_lines.append(f"RAFT checkpoint: {raft_ckpt}")
        summary_lines.append(f"Stereo calibration: {calib_path}")
        summary_lines.append(f"Rectify inputs: {args.rectify_inputs}")
        summary_lines.append("")
        summary_lines.append(f"Detections: {len(detections)}")
        summary_lines.append("")

        for det in detections:
            pts, colors = points_from_mask(
                xyz_img=xyz_img,
                image_bgr=left_proc,
                mask=det.mask,
                max_points=args.max_points,
            )

            safe_cls = sanitize_name(det.class_name)
            inst_path = out_dir / f"instance_{det.instance_i:02d}_{safe_cls}_conf{det.confidence:.2f}.ply"

            class_ids = np.full(len(pts), det.class_id, dtype=np.int32)
            instance_ids = np.full(len(pts), det.instance_i, dtype=np.int32)

            if len(pts) > 0:
                write_ply_xyzrgb_scalar(
                    inst_path,
                    points_xyz=pts,
                    colors_rgb=colors,
                    class_ids=class_ids,
                    instance_ids=instance_ids,
                )

                all_points.append(pts)
                all_colors.append(colors)
                all_class_ids.append(class_ids)
                all_instance_ids.append(instance_ids)

            bbox = det.bbox_xyxy
            line = (
                f"#{det.instance_i:02d} class={det.class_name} "
                f"class_id={det.class_id} conf={det.confidence:.3f} "
                f"points={len(pts)} "
                f"bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] "
                f"ply={inst_path.name if len(pts) > 0 else 'NO_VALID_POINTS'}"
            )

            print("[INSTANCE]", line)
            summary_lines.append(line)

        if all_points:
            combined_points = np.vstack(all_points)
            combined_colors = np.vstack(all_colors)
            combined_class_ids = np.concatenate(all_class_ids)
            combined_instance_ids = np.concatenate(all_instance_ids)

            combined_path = out_dir / "combined_segmented_pointcloud.ply"

            write_ply_xyzrgb_scalar(
                combined_path,
                points_xyz=combined_points,
                colors_rgb=combined_colors,
                class_ids=combined_class_ids,
                instance_ids=combined_instance_ids,
            )

            summary_lines.append("")
            summary_lines.append(f"Combined pointcloud: {combined_path.name}")
            summary_lines.append(f"Combined points: {len(combined_points)}")
        else:
            summary_lines.append("")
            summary_lines.append("No valid segmented point cloud points produced.")

        with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

        print("\n================ DONE ================")
        print("Output folder:")
        print(out_dir)
        print("")
        print("Open this in CloudCompare:")
        print(out_dir / "combined_segmented_pointcloud.ply")
        print("======================================")


if __name__ == "__main__":
    main()