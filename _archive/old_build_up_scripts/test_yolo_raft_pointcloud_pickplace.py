from __future__ import annotations

"""
test_yolo_raft_pointcloud_pickplace.py

YOLO segmentation + RAFT-Stereo point-cloud pick/place test for the grocery
bagger. This is intentionally based on the working AprilTag pick/place flow in
test_calibration_bundle_live_stereo_z_pickplace.py, but the object target is now:

  1. YOLO segmentation mask on the rectified stereo-left frame.
  2. RAFT-Stereo disparity on the rectified stereo pair.
  3. Masked stereo point cloud centroid/top-point extraction.
  4. A_robot_from_cam_xyz_3x4 maps object cam XYZ into robot XYZ.

EE AprilTag ID0 is still used for stereo/FK sanity and local Z bias correction.
Target AprilTag ID2 is not used by this script.

Keys:
  s       survey workspace with YOLO + RAFT
  1-9     select surveyed object candidate
  m       coarse hover move to selected object
  k       PICK selected object
  n       save current FK as drop zone
  f       PLACE at saved drop zone
  o/l     open/close claw
  [/]     jog robot Z
  ,/.     jog phi/J4
  v       validate overhead EE mapping
  b       print geometry sanity check once
  r       print stereo XYZ matrix / vision model paths
  h/c/a   home/sync/assume home
  e/d     enable/disable motors
  q       quit
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any

import cv2
import numpy as np

from robot import Robot
from robot_config import (
    ROBOT_CONFIG,
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    print_startup_config,
    require_soft_limits_configured,
)
from camera_config import (
    EE_TAG_ID,
    OVERHEAD_INDEX,
    STEREO_INDEX,
)
from stereo_apriltag_viewer import (
    build_detector,
    detect_tags,
    draw_detection,
)

# Reuse the working camera/calibration/key/safety utilities from the current
# AprilTag pick/place script. The object pipeline below is new and local to this file.
from test_calibration_bundle_live_stereo_z_pickplace import (
    open_overhead_camera,
    open_stereo_camera,
    load_bundle,
    load_stereo_calibration,
    read_stereo_tags_once,
    cam_xyz_to_robot_xyz,
    map_uv_z_to_robot_xy,
    nearest_support_distance,
    read_command_key,
    require_soft_limits,
    move_cartesian_nonnegative_z,
    jog_nonnegative_z,
    clamp_lookup_z_to_bundle,
    print_matrix_labeled,
)


# ============================================================
# FILES / WINDOWS
# ============================================================

BUNDLE_PATH = Path("robot_calibration_bundle.npz")
STEREO_CALIBRATION_PATH = Path("stereo_calibration.npz")
WINDOW = "YOLO + RAFT PointCloud PickPlace"


# ============================================================
# RAFT / YOLO KNOBS
# ============================================================

RAFT_ROOT = Path("RAFT-Stereo")
RAFT_CHECKPOINT_PATH = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")
RAFT_VALID_ITERS = 24
RAFT_CORR_IMPLEMENTATION = "alt"
RAFT_MIXED_PRECISION = True
RAFT_CONTEXT_NORM = "batch"
RAFT_SHARED_BACKBONE = False
RAFT_N_DOWNSAMPLE = 2
RAFT_N_GRU_LAYERS = 3
RAFT_SLOW_FAST_GRU = False

YOLO_WEIGHTS_PATH = Path("yolo_weights/best.pt")
YOLO_FALLBACK_WEIGHTS_PATH = Path("best.pt")
YOLO_CONF = 0.35
YOLO_IOU = 0.50
MIN_MASK_AREA_PX = 500
TARGET_CLASS_NAMES: list[str] = []  # Empty means allow all grocery classes.


# ============================================================
# POINT CLOUD / TARGET KNOBS
# ============================================================

MIN_DISPARITY_PX = 1.0
POINTCLOUD_MAX_POINTS = 20000
POINTCLOUD_Z_MIN_MM = -2000.0
POINTCLOUD_Z_MAX_MM = 2000.0
POINTCLOUD_REMOVE_OUTLIERS = True

OBJECT_TARGET_MODE = "centroid"  # "centroid" or "top_surface"
TOP_SURFACE_PERCENTILE = 90.0
TOP_SURFACE_MEDIAN_BAND_MM = 10.0

TARGET_XY_SOURCE = "stereo_xyz"  # Future option: "overhead_homography".

SURVEY_BURST_COUNT = 5
SURVEY_FRAME_DELAY_S = 0.05
SURVEY_MATCH_MAX_CENTROID_PX = 60.0


# ============================================================
# Z GEOMETRY KNOBS
# ============================================================

# EE AprilTag ID0 is mounted above the actual end-effector/tool point.
# ee_tool_z = ee_tag_z + TAG_TO_EE_Z_MM.
TAG_TO_EE_Z_MM = 0.0

# Hover/grasp are relative to the selected object point/surface.
GRASP_OFFSET_MM = 115.0
HOVER_HEIGHT_MM = GRASP_OFFSET_MM + 50.0 + 50.0


# ============================================================
# MOTION / CLAW / SAFETY KNOBS
# ============================================================

MIN_COARSE_TRAVEL_Z_MM = 50.0
COARSE_MOVE_TIME_S = 1.10
PICK_MOVE_TIME_S = 0.75

Z_JOG_MM = 5.0
PHI_JOG_DEG = 5.0

CLAW_OPEN_DEG = 75
CLAW_CLOSED_DEG = 5
CLAW_SETTLE_S = 0.30

USE_EE_FK_Z_BIAS_CORRECTION = True
WARN_STEREO_Z_BIAS_MM = 50.0
MAX_ALLOWED_STEREO_Z_BIAS_MM = 80.0

MIN_VALID_OBJECT_POINTS = 300
REFUSE_PICK_WITHOUT_EE_STEREO = True
REFUSE_PICK_IF_STEREO_Z_BIAS_TOO_LARGE = True
REFUSE_PLACE_IF_NO_DROP_ZONE = True
REFUSE_MOVE_IF_HOMOGRAPHY_TARGET_FAR = True

WARN_EE_ERROR_MM = 15.0
MAX_EE_ERROR_MM = 30.0

PRINT_GEOMETRY_SANITY = False
GEOMETRY_SANITY_PRINT_EVERY_SEC = 0.75


# ============================================================
# DISPLAY
# ============================================================

COMBINED_WIDTH_PX = 1280
OVERHEAD_DRAW_H_PX = 600
STEREO_DRAW_H_PX = 380


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class YOLODetection:
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    mask_area: int
    centroid_px: np.ndarray


@dataclass
class ObjectCandidate:
    index: int
    yolo: YOLODetection
    frame_i: int
    valid_point_count: int
    centroid_cam_xyz: np.ndarray
    top_cam_xyz: np.ndarray
    target_cam_xyz: np.ndarray
    object_robot_xyz_raw: np.ndarray
    object_robot_xyz_corrected: np.ndarray
    target_xy: np.ndarray
    target_xy_source_requested: str
    target_xy_source_effective: str
    overhead_centroid_px: np.ndarray | None
    lookup_z_used: float
    lookup_z_clamped: bool
    support_distance_mm: float
    support_index: int
    hover_robot_z: float
    grasp_robot_z: float
    ee_cam_xyz: np.ndarray | None
    ee_tag_robot_xyz_raw: np.ndarray | None
    ee_tool_z_from_stereo_raw: float | None
    stereo_z_bias_mm: float
    ee_stereo_visible: bool
    fk_xyz_at_update: np.ndarray
    fk_phi_at_update: float


# ============================================================
# RAFT WRAPPER
# ============================================================

class RAFTStereoRunner:
    def __init__(
        self,
        raft_root: Path = RAFT_ROOT,
        checkpoint_path: Path = RAFT_CHECKPOINT_PATH,
    ) -> None:
        self.raft_root = Path(raft_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.import_summary = "(not loaded)"

        if not self.raft_root.exists():
            raise FileNotFoundError(
                f"Missing RAFT-Stereo folder: {self.raft_root.resolve()}. "
                "Place the official/local RAFT-Stereo repo in the project root."
            )
        if not (self.raft_root / "core").exists():
            raise FileNotFoundError(
                f"Missing RAFT-Stereo core folder: {(self.raft_root / 'core').resolve()}"
            )
        if not (self.raft_root / "demo.py").exists():
            raise FileNotFoundError(
                f"Missing RAFT-Stereo demo.py: {(self.raft_root / 'demo.py').resolve()}"
            )
        if not self.checkpoint_path.exists():
            available = sorted(str(p) for p in (self.raft_root / "models").glob("*.pth"))
            raise FileNotFoundError(
                f"Missing RAFT checkpoint: {self.checkpoint_path.resolve()}\n"
                f"Available local checkpoints: {available}"
            )

        raft_root_abs = str(self.raft_root.resolve())
        if raft_root_abs not in sys.path:
            sys.path.insert(0, raft_root_abs)

        try:
            import torch
            from core.raft_stereo import RAFTStereo
            from core.utils.utils import InputPadder
        except Exception as exc:
            raise RuntimeError(
                "Could not import RAFT-Stereo. Expected imports:\n"
                "  from core.raft_stereo import RAFTStereo\n"
                "  from core.utils.utils import InputPadder\n"
                f"RAFT root on sys.path: {raft_root_abs}\n"
                "Check RAFT-Stereo/environment.yaml for dependencies. In this local "
                "checkout, core/update.py requires opt_einsum.\n"
                f"Original import error: {exc}"
            ) from exc

        self.torch = torch
        self.InputPadder = InputPadder
        self.import_summary = (
            "from core.raft_stereo import RAFTStereo; "
            "from core.utils.utils import InputPadder"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = self._build_args()
        self.args = args

        model = RAFTStereo(args)
        state = self._load_checkpoint(torch, self.checkpoint_path, self.device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"RAFT checkpoint did not match the configured architecture: {self.checkpoint_path}\n"
                "If you switch checkpoints, adjust the RAFT_* knobs near the top of this file.\n"
                f"Original load error: {exc}"
            ) from exc

        self.model = model.to(self.device)
        self.model.eval()

        print("[RAFT] Loaded")
        print(f"  root       = {self.raft_root.resolve()}")
        print(f"  checkpoint = {self.checkpoint_path.resolve()}")
        print(f"  device     = {self.device}")
        print(f"  imports    = {self.import_summary}")

    def _build_args(self) -> SimpleNamespace:
        return SimpleNamespace(
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

    def predict_disparity(self, left_bgr_or_rgb: np.ndarray, right_bgr_or_rgb: np.ndarray, *, color: str = "BGR") -> np.ndarray:
        if left_bgr_or_rgb is None or right_bgr_or_rgb is None:
            raise ValueError("RAFT predict_disparity requires left and right images.")
        if left_bgr_or_rgb.shape[:2] != right_bgr_or_rgb.shape[:2]:
            raise ValueError(
                f"RAFT left/right size mismatch: {left_bgr_or_rgb.shape} vs {right_bgr_or_rgb.shape}"
            )

        h, w = left_bgr_or_rgb.shape[:2]
        image1 = self._image_to_tensor(left_bgr_or_rgb, color=color)
        image2 = self._image_to_tensor(right_bgr_or_rgb, color=color)

        padder = self.InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with self.torch.no_grad():
            _, flow_up = self.model(image1, image2, iters=RAFT_VALID_ITERS, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()

        # RAFT-Stereo reports x-flow from left to right. Positive stereo disparity is -flow_x.
        disparity = (-flow_up).detach().float().cpu().numpy().astype(np.float32)
        if disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        return disparity

    def _image_to_tensor(self, image: np.ndarray, *, color: str) -> Any:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image for RAFT, got shape={image.shape}")
        if color.upper() == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color.upper() != "RGB":
            raise ValueError("color must be 'BGR' or 'RGB'")
        tensor = self.torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return tensor[None].to(self.device)


# ============================================================
# YOLO WRAPPER
# ============================================================

class YOLOSegmenter:
    def __init__(self, weights_path: Path = YOLO_WEIGHTS_PATH) -> None:
        self.weights_path = self._resolve_weights(Path(weights_path))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Could not import ultralytics.YOLO. Install ultralytics in this Python environment."
            ) from exc

        self.model = YOLO(str(self.weights_path))
        self.names = self.model.names or {}

        print("[YOLO] Loaded")
        print(f"  weights = {self.weights_path.resolve()}")
        print(f"  conf    = {YOLO_CONF:.2f}")
        print(f"  iou     = {YOLO_IOU:.2f}")

    @staticmethod
    def _resolve_weights(path: Path) -> Path:
        if path.exists():
            return path
        if YOLO_FALLBACK_WEIGHTS_PATH.exists():
            print(
                f"[YOLO] Expected {path} is missing; using local fallback "
                f"{YOLO_FALLBACK_WEIGHTS_PATH.resolve()}"
            )
            return YOLO_FALLBACK_WEIGHTS_PATH
        raise FileNotFoundError(
            f"Missing YOLO weights. Tried:\n"
            f"  {path.resolve()}\n"
            f"  {YOLO_FALLBACK_WEIGHTS_PATH.resolve()}"
        )

    def segment(self, image_bgr: np.ndarray) -> list[YOLODetection]:
        results = self.model.predict(
            source=image_bgr,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        h, w = image_bgr.shape[:2]
        masks = result.masks.data.detach().cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)

        out: list[YOLODetection] = []
        for i in range(len(masks)):
            mask = masks[i] > 0.5
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

            area = int(mask.sum())
            if area < MIN_MASK_AREA_PX:
                continue

            class_id = int(class_ids[i])
            class_name = str(self.names.get(class_id, class_id))
            if TARGET_CLASS_NAMES and class_name not in TARGET_CLASS_NAMES:
                continue

            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            centroid = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
            bbox = tuple(float(v) for v in boxes_xyxy[i])

            out.append(
                YOLODetection(
                    mask=mask,
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[i]),
                    mask_area=area,
                    centroid_px=centroid,
                )
            )

        return out


# ============================================================
# STEREO RECTIFICATION / POINT CLOUD
# ============================================================

class StereoRectifier:
    def __init__(self, stereo_calib: dict[str, np.ndarray]) -> None:
        self.calib = stereo_calib
        self._maps_by_size: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
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
                f"Stereo calibration missing rectification keys: {missing}. "
                f"Available keys={list(stereo_calib.keys())}"
            )

    def rectify(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise ValueError(f"Stereo frame shape mismatch: {left_bgr.shape} vs {right_bgr.shape}")
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
        print(f"[Stereo] Built rectification maps for {w}x{h}")
        return map_lx, map_ly, map_rx, map_ry


def _available_calib_keys_message(stereo_calib: dict[str, np.ndarray]) -> str:
    return "Available stereo calibration keys: " + ", ".join(sorted(stereo_calib.keys()))


def _rectified_xyz_to_bundle_cam_convention(xyz_rect_mm: np.ndarray, stereo_calib: dict[str, np.ndarray]) -> np.ndarray:
    xyz = np.asarray(xyz_rect_mm, dtype=np.float64)
    if "rectification_left" in stereo_calib:
        r_left = np.asarray(stereo_calib["rectification_left"], dtype=np.float64)
        # OpenCV rectification maps original camera rays into the rectified camera.
        # Convert rectified 3D points back toward the original left-camera frame.
        xyz = xyz @ r_left

    # Existing calibration code flips Z after stereo triangulation. Preserve that
    # convention before feeding A_robot_from_cam_xyz_3x4.
    xyz = xyz.copy()
    xyz[:, 2] *= -1.0
    return xyz


def masked_disparity_to_pointcloud(
    mask: np.ndarray,
    disparity: np.ndarray,
    stereo_calib: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Project a mask-filtered rectified disparity map into stereo camera XYZ.

    Returns:
      points_cam_mm - Nx3 points in the same camera convention used by the
                      existing calibration bundle.
      uv_px         - Nx2 rectified stereo-left pixels corresponding to points.
    """
    if mask.shape != disparity.shape[:2]:
        raise ValueError(f"mask/disparity shape mismatch: {mask.shape} vs {disparity.shape}")

    mask_bool = mask.astype(bool)
    disp = np.asarray(disparity, dtype=np.float32)
    valid = mask_bool & np.isfinite(disp) & (disp > MIN_DISPARITY_PX)

    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)

    if len(xs) > POINTCLOUD_MAX_POINTS:
        step = int(np.ceil(len(xs) / POINTCLOUD_MAX_POINTS))
        xs = xs[::step][:POINTCLOUD_MAX_POINTS]
        ys = ys[::step][:POINTCLOUD_MAX_POINTS]

    d = disp[ys, xs].astype(np.float64)
    u = xs.astype(np.float64)
    v = ys.astype(np.float64)

    if "disparity_to_depth_Q" in stereo_calib:
        q = np.asarray(stereo_calib["disparity_to_depth_Q"], dtype=np.float64)
        pts_h = np.column_stack([u, v, d, np.ones_like(d)]) @ q.T
        w = pts_h[:, 3]
        good_w = np.isfinite(w) & (np.abs(w) > 1e-9)
        xyz_rect = pts_h[good_w, :3] / w[good_w, None]
        uv = np.column_stack([u[good_w], v[good_w]])
    else:
        required = ["projection_left_rectified"]
        missing = [k for k in required if k not in stereo_calib]
        if missing:
            print(_available_calib_keys_message(stereo_calib))
            raise KeyError(f"Cannot project disparity without Q or fallback keys: missing {missing}")

        p_left = np.asarray(stereo_calib["projection_left_rectified"], dtype=np.float64)
        fx = float(p_left[0, 0])
        fy = float(p_left[1, 1])
        cx = float(p_left[0, 2])
        cy = float(p_left[1, 2])

        if "baseline_mm" in stereo_calib:
            baseline = float(np.asarray(stereo_calib["baseline_mm"]).reshape(-1)[0])
        elif "projection_right_rectified" in stereo_calib:
            p_right = np.asarray(stereo_calib["projection_right_rectified"], dtype=np.float64)
            baseline = abs(float(p_right[0, 3]) / fx)
        else:
            print(_available_calib_keys_message(stereo_calib))
            raise KeyError("Cannot project disparity: missing baseline_mm and projection_right_rectified")

        z = fx * baseline / d
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        xyz_rect = np.column_stack([x, y, z])
        uv = np.column_stack([u, v])

    points_cam = _rectified_xyz_to_bundle_cam_convention(xyz_rect, stereo_calib)
    finite = np.all(np.isfinite(points_cam), axis=1)
    z_ok = (points_cam[:, 2] >= POINTCLOUD_Z_MIN_MM) & (points_cam[:, 2] <= POINTCLOUD_Z_MAX_MM)
    keep = finite & z_ok
    points_cam = points_cam[keep]
    uv = uv[keep]

    if POINTCLOUD_REMOVE_OUTLIERS and len(points_cam) >= 30:
        points_cam, uv = remove_pointcloud_outliers(points_cam, uv)

    return points_cam, uv


def remove_pointcloud_outliers(points_cam: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.median(points_cam, axis=0)
    dist = np.linalg.norm(points_cam - med.reshape(1, 3), axis=1)
    cutoff = float(np.percentile(dist, 95.0))
    keep = dist <= max(cutoff, 1e-6)
    return points_cam[keep], uv[keep]


def cam_points_to_robot_xyz(points_cam: np.ndarray, bundle) -> np.ndarray:
    a = bundle.get("A_robot_from_cam_xyz_3x4")
    if a is None:
        raise RuntimeError("Bundle has no A_robot_from_cam_xyz_3x4. Re-run XYZ calibration.")
    pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    return pts_h @ np.asarray(a, dtype=np.float64).T


def choose_object_target_point(points_cam: np.ndarray, bundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_cam) == 0:
        raise ValueError("No valid object points.")

    centroid_cam = np.median(points_cam, axis=0).astype(np.float64)
    top_cam = centroid_cam.copy()

    try:
        points_robot = cam_points_to_robot_xyz(points_cam, bundle)
        robot_z = points_robot[:, 2]
        z_cut = float(np.percentile(robot_z, TOP_SURFACE_PERCENTILE))
        band = robot_z >= (z_cut - TOP_SURFACE_MEDIAN_BAND_MM)
        if int(band.sum()) >= 5:
            top_cam = np.median(points_cam[band], axis=0).astype(np.float64)
    except Exception as exc:
        print(f"[POINTCLOUD] Top-surface extraction failed, using centroid for top too: {exc}")

    if OBJECT_TARGET_MODE == "top_surface":
        target_cam = top_cam
    elif OBJECT_TARGET_MODE == "centroid":
        target_cam = centroid_cam
    else:
        raise ValueError("OBJECT_TARGET_MODE must be 'centroid' or 'top_surface'")

    return centroid_cam, top_cam, target_cam.astype(np.float64)


# ============================================================
# OBJECT GEOMETRY
# ============================================================

def build_object_candidate(
    index: int,
    yolo_det: YOLODetection,
    points_cam: np.ndarray,
    robot: Robot,
    bundle,
    stereo_tags: dict[int, Any],
    frame_i: int,
) -> ObjectCandidate:
    centroid_cam, top_cam, target_cam = choose_object_target_point(points_cam, bundle)
    raw_robot = cam_xyz_to_robot_xyz(target_cam, bundle)

    candidate = ObjectCandidate(
        index=index,
        yolo=yolo_det,
        frame_i=frame_i,
        valid_point_count=int(len(points_cam)),
        centroid_cam_xyz=centroid_cam,
        top_cam_xyz=top_cam,
        target_cam_xyz=target_cam,
        object_robot_xyz_raw=raw_robot,
        object_robot_xyz_corrected=raw_robot.copy(),
        target_xy=raw_robot[:2].copy(),
        target_xy_source_requested=TARGET_XY_SOURCE,
        target_xy_source_effective="stereo_xyz",
        overhead_centroid_px=None,
        lookup_z_used=float(raw_robot[2]),
        lookup_z_clamped=False,
        support_distance_mm=np.inf,
        support_index=-1,
        hover_robot_z=float(raw_robot[2] + HOVER_HEIGHT_MM),
        grasp_robot_z=float(raw_robot[2] + GRASP_OFFSET_MM),
        ee_cam_xyz=None,
        ee_tag_robot_xyz_raw=None,
        ee_tool_z_from_stereo_raw=None,
        stereo_z_bias_mm=0.0,
        ee_stereo_visible=False,
        fk_xyz_at_update=np.zeros(3, dtype=np.float64),
        fk_phi_at_update=0.0,
    )
    refresh_candidate_z_bias(candidate, robot, bundle, stereo_tags, warn=False)
    return candidate


def refresh_candidate_z_bias(
    candidate: ObjectCandidate,
    robot: Robot,
    bundle,
    stereo_tags: dict[int, Any] | None,
    *,
    warn: bool = True,
) -> None:
    x_fk, y_fk, z_fk, phi_fk = robot.fk()
    candidate.fk_xyz_at_update = np.array([x_fk, y_fk, z_fk], dtype=np.float64)
    candidate.fk_phi_at_update = float(phi_fk)

    ee_tri = (stereo_tags or {}).get(EE_TAG_ID)
    ee_raw = None
    ee_tool_z_raw = None
    stereo_z_bias = 0.0

    if ee_tri is not None:
        ee_raw = cam_xyz_to_robot_xyz(ee_tri.xyz_cam_mm, bundle)
        ee_tool_z_raw = float(ee_raw[2] + TAG_TO_EE_Z_MM)
        if USE_EE_FK_Z_BIAS_CORRECTION:
            stereo_z_bias = float(z_fk - ee_tool_z_raw)

    corrected = candidate.object_robot_xyz_raw.copy()
    corrected[2] += stereo_z_bias

    lookup_z_used, lookup_z_clamped = clamp_lookup_z_to_bundle(float(corrected[2]), bundle)

    xy_source_effective = TARGET_XY_SOURCE
    if TARGET_XY_SOURCE == "overhead_homography" and candidate.overhead_centroid_px is not None:
        target_xy, _, _, _, _ = map_uv_z_to_robot_xy(candidate.overhead_centroid_px, lookup_z_used, bundle)
    else:
        if TARGET_XY_SOURCE == "overhead_homography":
            xy_source_effective = "stereo_xyz"
        target_xy = corrected[:2].copy()

    support_dist, support_idx = nearest_support_distance(target_xy, lookup_z_used, bundle)

    candidate.object_robot_xyz_corrected = corrected
    candidate.target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    candidate.target_xy_source_requested = TARGET_XY_SOURCE
    candidate.target_xy_source_effective = xy_source_effective
    candidate.lookup_z_used = float(lookup_z_used)
    candidate.lookup_z_clamped = bool(lookup_z_clamped)
    candidate.support_distance_mm = float(support_dist)
    candidate.support_index = int(support_idx)
    candidate.hover_robot_z = float(corrected[2] + HOVER_HEIGHT_MM)
    candidate.grasp_robot_z = float(corrected[2] + GRASP_OFFSET_MM)
    candidate.ee_cam_xyz = None if ee_tri is None else ee_tri.xyz_cam_mm.copy()
    candidate.ee_tag_robot_xyz_raw = ee_raw
    candidate.ee_tool_z_from_stereo_raw = ee_tool_z_raw
    candidate.stereo_z_bias_mm = float(stereo_z_bias)
    candidate.ee_stereo_visible = ee_tri is not None

    if warn and ee_tri is not None and abs(stereo_z_bias) > WARN_STEREO_Z_BIAS_MM:
        print(
            f"[Z BIAS WARN] stereo_z_bias={stereo_z_bias:+.1f} mm "
            f"(warn>{WARN_STEREO_Z_BIAS_MM:.1f}, refuse>{MAX_ALLOWED_STEREO_Z_BIAS_MM:.1f})"
        )


def candidate_score(candidate: ObjectCandidate) -> float:
    return float(candidate.yolo.confidence) * np.log1p(float(candidate.valid_point_count))


def fuse_survey_candidates(raw_candidates: list[ObjectCandidate]) -> list[ObjectCandidate]:
    grouped: list[ObjectCandidate] = []
    for cand in sorted(raw_candidates, key=candidate_score, reverse=True):
        replaced = False
        for i, existing in enumerate(grouped):
            same_class = cand.yolo.class_name == existing.yolo.class_name
            px_dist = float(np.linalg.norm(cand.yolo.centroid_px - existing.yolo.centroid_px))
            if same_class and px_dist <= SURVEY_MATCH_MAX_CENTROID_PX:
                if candidate_score(cand) > candidate_score(existing):
                    grouped[i] = cand
                replaced = True
                break
        if not replaced:
            grouped.append(cand)

    grouped = sorted(grouped, key=candidate_score, reverse=True)[:9]
    for i, cand in enumerate(grouped, start=1):
        cand.index = i
    return grouped


def run_survey_workspace(
    stereo,
    detector,
    stereo_calib: dict[str, np.ndarray],
    rectifier: StereoRectifier,
    yolo: YOLOSegmenter,
    raft: RAFTStereoRunner,
    robot: Robot,
    bundle,
) -> list[ObjectCandidate]:
    print("\n[SURVEY] Starting YOLO + RAFT burst survey")
    print(f"[SURVEY] frames={SURVEY_BURST_COUNT}, target_mode={OBJECT_TARGET_MODE}, xy_source={TARGET_XY_SOURCE}")

    raw_candidates: list[ObjectCandidate] = []
    next_index = 1

    for frame_i in range(SURVEY_BURST_COUNT):
        stereo_tags, left_raw, right_raw, _, _ = read_stereo_tags_once(stereo, detector, stereo_calib)
        if left_raw is None or right_raw is None:
            print(f"[SURVEY] frame {frame_i + 1}: stereo read failed")
            continue

        left_rect, right_rect = rectifier.rectify(left_raw, right_raw)
        detections = yolo.segment(left_rect)
        print(f"[SURVEY] frame {frame_i + 1}: YOLO detections={len(detections)}")

        disparity = raft.predict_disparity(left_rect, right_rect, color="BGR")
        for det in detections:
            points_cam, _uv = masked_disparity_to_pointcloud(det.mask, disparity, stereo_calib)
            if len(points_cam) < MIN_VALID_OBJECT_POINTS:
                print(
                    f"  skip {det.class_name}: valid_points={len(points_cam)} "
                    f"< {MIN_VALID_OBJECT_POINTS}"
                )
                continue

            try:
                cand = build_object_candidate(
                    index=next_index,
                    yolo_det=det,
                    points_cam=points_cam,
                    robot=robot,
                    bundle=bundle,
                    stereo_tags=stereo_tags,
                    frame_i=frame_i,
                )
            except Exception as exc:
                print(f"  skip {det.class_name}: geometry failed: {exc}")
                continue

            raw_candidates.append(cand)
            next_index += 1
            print(
                f"  candidate {det.class_name} conf={det.confidence:.2f} "
                f"pts={cand.valid_point_count} robot=({cand.object_robot_xyz_corrected[0]:.1f},"
                f"{cand.object_robot_xyz_corrected[1]:.1f},{cand.object_robot_xyz_corrected[2]:.1f})"
            )

        time.sleep(SURVEY_FRAME_DELAY_S)

    candidates = fuse_survey_candidates(raw_candidates)
    print_candidate_list(candidates)
    return candidates


# ============================================================
# SAFETY / MOTION
# ============================================================

def choose_safe_travel_z(robot, target_x, target_y, hover_robot_z, object_target_z):
    desired = max(MIN_COARSE_TRAVEL_Z_MM, float(hover_robot_z))
    min_allowed = float(object_target_z) + min(15.0, HOVER_HEIGHT_MM)

    candidates = [
        desired,
        HOME_Z_MM,
        DEFAULT_TRAVEL_Z_MM,
        100.0,
        75.0,
        65.0,
        50.0,
        40.0,
        25.0,
        0.0,
    ]

    clean: list[float] = []
    for z in candidates:
        z = float(z)
        if z < min_allowed:
            continue
        if all(abs(z - old) > 1e-6 for old in clean):
            clean.append(z)

    for z in clean:
        ok_pose, reason_pose = robot.check_cartesian_pose_safe(target_x, target_y, z)
        if not ok_pose:
            print(f"[TRAVEL_Z] Reject robot_z={z:.1f}: target unsafe: {reason_pose}")
            continue

        ok_path, _path, reason_path = robot.plan_cartesian_path(target_x, target_y, z)
        if not ok_path:
            print(f"[TRAVEL_Z] Reject robot_z={z:.1f}: path unsafe: {reason_path}")
            continue

        print(f"[TRAVEL_Z] Selected robot_z={z:.1f} mm")
        return z

    return None


def validate_selected_object(candidate: ObjectCandidate | None, bundle, *, action: str) -> bool:
    if candidate is None:
        print(f"[{action}] REFUSED: no selected object. Press s to survey, then 1-9 to select.")
        return False

    if candidate.valid_point_count < MIN_VALID_OBJECT_POINTS:
        print(
            f"[{action}] REFUSED: selected object has only {candidate.valid_point_count} "
            f"valid points (<{MIN_VALID_OBJECT_POINTS})."
        )
        return False

    if not np.all(np.isfinite(candidate.object_robot_xyz_corrected)):
        print(f"[{action}] REFUSED: selected object robot XYZ is not finite.")
        return False

    if action == "PICK":
        if REFUSE_PICK_WITHOUT_EE_STEREO and not candidate.ee_stereo_visible:
            print(f"[{action}] REFUSED: EE tag ID{EE_TAG_ID} is not stereo-visible for Z bias correction.")
            return False
        if (
            REFUSE_PICK_IF_STEREO_Z_BIAS_TOO_LARGE
            and abs(candidate.stereo_z_bias_mm) > MAX_ALLOWED_STEREO_Z_BIAS_MM
        ):
            print(
                f"[{action}] REFUSED: stereo Z bias {candidate.stereo_z_bias_mm:+.1f} mm "
                f"exceeds {MAX_ALLOWED_STEREO_Z_BIAS_MM:.1f} mm."
            )
            return False

    if (
        REFUSE_MOVE_IF_HOMOGRAPHY_TARGET_FAR
        and candidate.target_xy_source_effective == "overhead_homography"
        and candidate.support_distance_mm > bundle["max_nearest"]
    ):
        print(
            f"[{action}] REFUSED: target too far from calibration support "
            f"({candidate.support_distance_mm:.1f} > {bundle['max_nearest']:.1f})"
        )
        return False

    return True


def execute_hover_to_object(robot: Robot, candidate: ObjectCandidate, bundle) -> bool:
    if not validate_selected_object(candidate, bundle, action="MOVE"):
        return False

    x_t, y_t = float(candidate.target_xy[0]), float(candidate.target_xy[1])
    object_z = float(candidate.object_robot_xyz_corrected[2])
    travel_z = choose_safe_travel_z(robot, x_t, y_t, candidate.hover_robot_z, object_z)
    if travel_z is None:
        print("[MOVE] REFUSED: no soft-limit-safe travel Z found.")
        return False

    print(
        f"[MOVE] Coarse hover to candidate {candidate.index}: "
        f"XY=({x_t:.1f},{y_t:.1f}) travel_z={travel_z:.1f}"
    )
    _, _, z_cur, phi_cur = robot.fk()
    if abs(z_cur - travel_z) > 1.0:
        if not move_cartesian_nonnegative_z(robot, "[MOVE] travel Z", z_mm=travel_z, move_time_s=0.75):
            print("[MOVE] Failed moving to travel Z.")
            return False
        robot.sync_estimate_from_teensy_steps()

    if not move_cartesian_nonnegative_z(
        robot,
        "[MOVE] hover",
        x_mm=x_t,
        y_mm=y_t,
        z_mm=travel_z,
        phi_deg=phi_cur,
        move_time_s=COARSE_MOVE_TIME_S,
    ):
        print("[MOVE] Coarse XY move failed.")
        return False

    robot.sync_estimate_from_teensy_steps()
    robot.print_estimate()
    return True


def execute_pick(robot: Robot, candidate: ObjectCandidate, bundle) -> tuple[bool, float | None]:
    if not validate_selected_object(candidate, bundle, action="PICK"):
        return False, None

    x_t, y_t = float(candidate.target_xy[0]), float(candidate.target_xy[1])
    object_z = float(candidate.object_robot_xyz_corrected[2])
    travel_z = choose_safe_travel_z(robot, x_t, y_t, candidate.hover_robot_z, object_z)
    if travel_z is None:
        print("[PICK] REFUSED: no soft-limit-safe travel Z found.")
        return False, None

    print("\n[PICK] Starting pick sequence")
    print(
        f"[PICK] object={candidate.yolo.class_name} #{candidate.index} "
        f"XY=({x_t:.1f},{y_t:.1f}) object_z={object_z:.1f}"
    )
    print(f"[PICK] TAG_TO_EE={TAG_TO_EE_Z_MM:+.1f}, HOVER={HOVER_HEIGHT_MM:+.1f}, GRASP={GRASP_OFFSET_MM:+.1f}")
    print(f"[PICK] travel_z={travel_z:.1f}, hover_z={candidate.hover_robot_z:.1f}, grasp_z={candidate.grasp_robot_z:.1f}")

    print(f"[PICK] Opening claw (servo={CLAW_OPEN_DEG} deg)")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)

    _, _, z_cur, phi_cur = robot.fk()
    phi = float(phi_cur)

    if abs(z_cur - travel_z) > 1.0:
        print(f"[PICK] Raising/moving Z to travel_z={travel_z:.1f}")
        if not move_cartesian_nonnegative_z(robot, "[PICK] travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
            print("[PICK] Failed moving to travel Z.")
            return False, None
        robot.sync_estimate_from_teensy_steps()

    print(f"[PICK] Approach XY=({x_t:.1f},{y_t:.1f}) phi={phi:.1f} z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(
        robot,
        "[PICK] approach",
        x_mm=x_t,
        y_mm=y_t,
        z_mm=travel_z,
        phi_deg=phi,
        move_time_s=COARSE_MOVE_TIME_S,
    ):
        print("[PICK] Approach move failed.")
        return False, None
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PICK] Lowering to grasp_robot_z={candidate.grasp_robot_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PICK] grasp Z", z_mm=float(candidate.grasp_robot_z), move_time_s=PICK_MOVE_TIME_S):
        print("[PICK] Failed lowering to grasp Z.")
        return False, None
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PICK] Closing claw (servo={CLAW_CLOSED_DEG} deg)")
    robot.servo(CLAW_CLOSED_DEG)
    time.sleep(CLAW_SETTLE_S)

    print(f"[PICK] Raising back to travel_z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PICK] post-pick travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
        print("[PICK] Failed raising after pick, opening claw for safety.")
        robot.servo(CLAW_OPEN_DEG)
        return False, None
    robot.sync_estimate_from_teensy_steps()

    print(f"[PICK] Done. grasp_robot_z={candidate.grasp_robot_z:.1f}, phi={phi:.1f}")
    return True, float(candidate.grasp_robot_z)


def execute_place(robot: Robot, drop_zone_xy, drop_zone_phi, place_robot_z) -> bool:
    x_d, y_d = float(drop_zone_xy[0]), float(drop_zone_xy[1])
    drop_phi = float(drop_zone_phi)

    object_z_for_place = float(place_robot_z - GRASP_OFFSET_MM)
    hover_robot_z = float(object_z_for_place + HOVER_HEIGHT_MM)

    travel_z = choose_safe_travel_z(robot, x_d, y_d, hover_robot_z, object_z_for_place)
    if travel_z is None:
        print("[PLACE] REFUSED: no soft-limit-safe travel Z to drop zone.")
        return False

    print("\n[PLACE] Starting place sequence")
    print(f"[PLACE] drop XY=({x_d:.1f},{y_d:.1f}) phi={drop_phi:.1f} travel_z={travel_z:.1f} place_z={place_robot_z:.1f}")

    _, _, z_cur, _ = robot.fk()
    if abs(z_cur - travel_z) > 1.0:
        print(f"[PLACE] Raising to travel_z={travel_z:.1f}")
        if not move_cartesian_nonnegative_z(robot, "[PLACE] travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
            print("[PLACE] Failed moving to travel Z.")
            return False
        robot.sync_estimate_from_teensy_steps()

    print(f"[PLACE] Moving to drop zone XY=({x_d:.1f},{y_d:.1f}) phi={drop_phi:.1f} z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(
        robot,
        "[PLACE] approach",
        x_mm=x_d,
        y_mm=y_d,
        z_mm=travel_z,
        phi_deg=drop_phi,
        move_time_s=COARSE_MOVE_TIME_S,
    ):
        print("[PLACE] Move to drop zone failed.")
        return False
    robot.sync_estimate_from_teensy_steps()

    print(f"[PLACE] Lowering to place_z={place_robot_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PLACE] place Z", z_mm=float(place_robot_z), move_time_s=PICK_MOVE_TIME_S):
        print("[PLACE] Failed lowering to place Z.")
        return False
    robot.sync_estimate_from_teensy_steps()
    time.sleep(0.1)

    print(f"[PLACE] Opening claw (servo={CLAW_OPEN_DEG} deg)")
    robot.servo(CLAW_OPEN_DEG)
    time.sleep(CLAW_SETTLE_S)

    print(f"[PLACE] Raising back to travel_z={travel_z:.1f}")
    if not move_cartesian_nonnegative_z(robot, "[PLACE] post-place travel Z", z_mm=travel_z, move_time_s=PICK_MOVE_TIME_S):
        print("[PLACE] Failed raising after place.")
        return False
    robot.sync_estimate_from_teensy_steps()

    print("[PLACE] Done.")
    return True


# ============================================================
# DEBUG PRINTS
# ============================================================

def _fmt_xyz(v, decimals: int = 1) -> str:
    if v is None:
        return "(not available)"
    a = np.asarray(v, dtype=np.float64).reshape(3)
    return f"[x={a[0]:+8.{decimals}f}, y={a[1]:+8.{decimals}f}, z={a[2]:+8.{decimals}f}] mm"


def _fmt_uv(v, decimals: int = 1) -> str:
    if v is None:
        return "(not available)"
    a = np.asarray(v, dtype=np.float64).reshape(2)
    return f"[u={a[0]:.{decimals}f}, v={a[1]:.{decimals}f}] px"


def print_geometry_sanity(candidate: ObjectCandidate | None, robot: Robot | None = None, prefix: str = "[GEOM]") -> None:
    if candidate is None:
        print(f"{prefix} No selected object geometry available. Press s, then 1-9.")
        return

    fk_xyz = candidate.fk_xyz_at_update
    fk_phi = candidate.fk_phi_at_update
    if robot is not None:
        x_fk, y_fk, z_fk, phi_fk = robot.fk()
        fk_xyz = np.array([x_fk, y_fk, z_fk], dtype=np.float64)
        fk_phi = float(phi_fk)

    dz_to_grasp = candidate.grasp_robot_z - float(fk_xyz[2])
    direction = "DOWN" if dz_to_grasp < 0 else "UP"

    print("\n" + "=" * 78)
    print(f"{prefix} YOLO/RAFT object geometry sanity check")
    print("-" * 78)
    print("[1] SELECTED OBJECT")
    print(f"    index/class/conf     = #{candidate.index} {candidate.yolo.class_name} {candidate.yolo.confidence:.3f}")
    print(f"    bbox                 = ({candidate.yolo.bbox[0]:.1f}, {candidate.yolo.bbox[1]:.1f}, {candidate.yolo.bbox[2]:.1f}, {candidate.yolo.bbox[3]:.1f})")
    print(f"    mask centroid        = {_fmt_uv(candidate.yolo.centroid_px)}")
    print(f"    mask area / points   = {candidate.yolo.mask_area} px / {candidate.valid_point_count} 3D points")
    print(f"    target mode          = {OBJECT_TARGET_MODE}")
    print()

    print("[2] CAMERA FRAME")
    print(f"    centroid_cam_xyz     = {_fmt_xyz(candidate.centroid_cam_xyz)}")
    print(f"    top_cam_xyz          = {_fmt_xyz(candidate.top_cam_xyz)}")
    print(f"    target_cam_xyz       = {_fmt_xyz(candidate.target_cam_xyz)}")
    print(f"    EE cam xyz           = {_fmt_xyz(candidate.ee_cam_xyz)}")
    print()

    print("[3] ROBOT FRAME")
    print(f"    FK estimate          = {_fmt_xyz(fk_xyz)} phi={fk_phi:+.1f}")
    print(f"    object raw xyz       = {_fmt_xyz(candidate.object_robot_xyz_raw)}")
    print(f"    EE tag raw xyz       = {_fmt_xyz(candidate.ee_tag_robot_xyz_raw)}")
    print(f"    object corrected xyz = {_fmt_xyz(candidate.object_robot_xyz_corrected)}")
    print(f"    target XY            = ({candidate.target_xy[0]:.1f}, {candidate.target_xy[1]:.1f}) source={candidate.target_xy_source_effective}")
    print()

    print("[4] Z OFFSETS / BIAS")
    print(f"    TAG_TO_EE_Z_MM       = {TAG_TO_EE_Z_MM:+8.2f}")
    if candidate.ee_tool_z_from_stereo_raw is None:
        print("    EE tag not visible   = no stereo Z bias correction available")
    else:
        print(f"    EE tool z stereo     = {candidate.ee_tool_z_from_stereo_raw:+8.2f}")
        print(f"    Stereo Z bias        = {candidate.stereo_z_bias_mm:+8.2f}")
    print()

    print("[5] COMMANDS / SUPPORT")
    print(f"    HOVER_HEIGHT_MM      = {HOVER_HEIGHT_MM:+8.2f}")
    print(f"    GRASP_OFFSET_MM      = {GRASP_OFFSET_MM:+8.2f}")
    print(f"    hover_z              = {candidate.hover_robot_z:+8.2f}")
    print(f"    grasp_z              = {candidate.grasp_robot_z:+8.2f}")
    print(f"    dz FK -> grasp       = {dz_to_grasp:+8.2f} -> robot will move {direction} {abs(dz_to_grasp):.1f} mm")
    print(f"    lookup_z             = {candidate.lookup_z_used:+8.2f} {'CLAMPED' if candidate.lookup_z_clamped else ''}")
    print(f"    support distance     = {candidate.support_distance_mm:.1f} mm nearest idx={candidate.support_index}")
    print("=" * 78 + "\n")


def print_candidate_list(candidates: list[ObjectCandidate]) -> None:
    print("\n[SURVEY] Candidate objects")
    if not candidates:
        print("  none")
        return
    for cand in candidates:
        print(
            f"  {cand.index}: {cand.yolo.class_name:<14} conf={cand.yolo.confidence:.2f} "
            f"centroid=({cand.yolo.centroid_px[0]:.1f},{cand.yolo.centroid_px[1]:.1f}) "
            f"robot=({cand.object_robot_xyz_corrected[0]:.1f},"
            f"{cand.object_robot_xyz_corrected[1]:.1f},"
            f"{cand.object_robot_xyz_corrected[2]:.1f}) "
            f"pts={cand.valid_point_count} bias={cand.stereo_z_bias_mm:+.1f}"
        )
    print()


def print_vision_paths(raft: RAFTStereoRunner | None, yolo: YOLOSegmenter | None) -> None:
    print("\n[VISION PATHS]")
    print(f"  RAFT root knob       = {RAFT_ROOT}")
    print(f"  RAFT checkpoint knob = {RAFT_CHECKPOINT_PATH}")
    print(f"  YOLO weights knob    = {YOLO_WEIGHTS_PATH}")
    print(f"  YOLO fallback        = {YOLO_FALLBACK_WEIGHTS_PATH}")
    if raft is not None:
        print(f"  RAFT checkpoint used = {raft.checkpoint_path.resolve()}")
        print(f"  RAFT imports used    = {raft.import_summary}")
    if yolo is not None:
        print(f"  YOLO weights used    = {yolo.weights_path.resolve()}")
    print("  TODO: overhead YOLO masks for homography XY, object-aware gripper phi, temporal object tracking.")


# ============================================================
# DRAWING
# ============================================================

def put_text_outline(img: np.ndarray, text: str, org: tuple[int, int], scale=0.55, color=(255, 255, 255), thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overhead_status(frame: np.ndarray, det_ee, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    out = draw_detection(out, det_ee, f"EE {EE_TAG_ID}")
    for i, line in enumerate(lines):
        put_text_outline(out, line, (14, 30 + i * 27), scale=0.63, color=(255, 255, 255), thickness=2)
    return out


def candidate_color(index: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (255, 160, 0),
        (0, 200, 255),
        (255, 0, 255),
        (120, 255, 120),
        (255, 255, 0),
        (0, 120, 255),
        (220, 220, 220),
        (120, 120, 255),
    ]
    return palette[(max(index, 1) - 1) % len(palette)]


def draw_object_candidates(
    image_bgr: np.ndarray,
    candidates: list[ObjectCandidate],
    selected_index: int | None,
) -> np.ndarray:
    out = image_bgr.copy()
    overlay = out.copy()

    for cand in candidates:
        color = candidate_color(cand.index)
        mask = cand.yolo.mask
        if mask.shape == out.shape[:2]:
            overlay[mask] = color

    out = cv2.addWeighted(overlay, 0.30, out, 0.70, 0.0)

    for cand in candidates:
        color = candidate_color(cand.index)
        x1, y1, x2, y2 = [int(round(v)) for v in cand.yolo.bbox]
        selected = selected_index == cand.index
        thickness = 3 if selected else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        c = tuple(np.round(cand.yolo.centroid_px).astype(int))
        cv2.circle(out, c, 6 if selected else 4, color, -1, cv2.LINE_AA)
        label = f"{cand.index} {cand.yolo.class_name} {cand.yolo.confidence:.2f}"
        put_text_outline(out, label, (x1, max(20, y1 - 8)), scale=0.55, color=color, thickness=2)

        if selected:
            xyz = cand.object_robot_xyz_corrected
            put_text_outline(
                out,
                f"XY=({cand.target_xy[0]:.0f},{cand.target_xy[1]:.0f}) Z={xyz[2]:.0f} H={cand.hover_robot_z:.0f} G={cand.grasp_robot_z:.0f}",
                (max(8, x1), min(out.shape[0] - 14, y2 + 22)),
                scale=0.48,
                color=(255, 255, 255),
                thickness=1,
            )

    return out


def _draw_stereo_pane(
    img: np.ndarray | None,
    det_dict,
    side_label: str,
    candidates: list[ObjectCandidate] | None = None,
    selected_index: int | None = None,
) -> np.ndarray | None:
    if img is None:
        return None
    out = img.copy()
    if candidates:
        out = draw_object_candidates(out, candidates, selected_index)
    for tag_id, det in (det_dict or {}).items():
        label = f"EE {tag_id}" if tag_id == EE_TAG_ID else f"ID{tag_id}"
        out = draw_detection(out, det, label)
    put_text_outline(out, side_label, (10, 28), scale=0.8, color=(255, 255, 255), thickness=2)
    return out


def compose_camera_views(
    overhead_annotated: np.ndarray | None,
    stereo_left: np.ndarray | None,
    stereo_right: np.ndarray | None,
    det_l_all=None,
    det_r_all=None,
    candidates: list[ObjectCandidate] | None = None,
    selected_index: int | None = None,
) -> np.ndarray:
    if overhead_annotated is not None:
        top = cv2.resize(overhead_annotated, (COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX))
    else:
        top = np.zeros((OVERHEAD_DRAW_H_PX, COMBINED_WIDTH_PX, 3), dtype=np.uint8)
        put_text_outline(top, "Overhead camera not available", (20, OVERHEAD_DRAW_H_PX // 2), scale=1.0, color=(0, 0, 255), thickness=2)

    half_w = COMBINED_WIDTH_PX // 2

    if stereo_left is not None:
        left_drawn = _draw_stereo_pane(stereo_left, det_l_all or {}, "STEREO LEFT RECT", candidates, selected_index)
        bot_l = cv2.resize(left_drawn, (half_w, STEREO_DRAW_H_PX))
    else:
        bot_l = np.zeros((STEREO_DRAW_H_PX, half_w, 3), dtype=np.uint8)
        put_text_outline(bot_l, "stereo left N/A", (20, STEREO_DRAW_H_PX // 2), scale=0.7, color=(0, 0, 255), thickness=2)

    if stereo_right is not None:
        right_drawn = _draw_stereo_pane(stereo_right, det_r_all or {}, "STEREO RIGHT RECT")
        bot_r = cv2.resize(right_drawn, (half_w, STEREO_DRAW_H_PX))
    else:
        bot_r = np.zeros((STEREO_DRAW_H_PX, half_w, 3), dtype=np.uint8)
        put_text_outline(bot_r, "stereo right N/A", (20, STEREO_DRAW_H_PX // 2), scale=0.7, color=(0, 0, 255), thickness=2)

    bottom = np.hstack([bot_l, bot_r])
    sep = np.full((2, COMBINED_WIDTH_PX, 3), 80, dtype=np.uint8)
    return np.vstack([top, sep, bottom])


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\nYOLO + RAFT PointCloud PickPlace")
    print("--------------------------------")
    print_startup_config("test_yolo_raft_pointcloud_pickplace.py", OVERHEAD_INDEX, STEREO_INDEX)
    require_soft_limits_configured("test_yolo_raft_pointcloud_pickplace.py")

    print("[Z KNOBS]")
    print(f"  TAG_TO_EE_Z_MM={TAG_TO_EE_Z_MM:+.1f}")
    print(f"  HOVER_HEIGHT_MM={HOVER_HEIGHT_MM:+.1f}")
    print(f"  GRASP_OFFSET_MM={GRASP_OFFSET_MM:+.1f}")
    print(f"  USE_EE_FK_Z_BIAS_CORRECTION={USE_EE_FK_Z_BIAS_CORRECTION}")
    print(f"  TARGET_XY_SOURCE={TARGET_XY_SOURCE}")
    print(f"  OBJECT_TARGET_MODE={OBJECT_TARGET_MODE}")

    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    rectifier = StereoRectifier(stereo_calib)

    try:
        yolo = YOLOSegmenter(YOLO_WEIGHTS_PATH)
        raft = RAFTStereoRunner(RAFT_ROOT, RAFT_CHECKPOINT_PATH)
    except Exception as exc:
        print(f"\n[INIT] Vision initialization failed. Autonomous motion is refused.\n{exc}\n")
        return

    print_vision_paths(raft, yolo)

    detector = build_detector()
    overhead_cap = open_overhead_camera()
    stereo = open_stereo_camera()
    robot = Robot(ROBOT_CONFIG, connect=True)

    candidates: list[ObjectCandidate] = []
    selected_object: ObjectCandidate | None = None
    selected_index: int | None = None

    has_item = False
    drop_zone_xy = None
    drop_zone_phi = None
    last_grasp_robot_z = None
    last_sanity_print = 0.0

    try:
        require_soft_limits(robot)

        robot.enable(True)
        robot.init_drivers()

        print("\nStartup options:")
        print("  h = run HOME now")
        print("  c = continue from current Teensy step counters, no homing")
        print("  a = assume robot is physically at configured home_pose, no homing")
        choice = input("Choose h/c/a: ").strip().lower()
        if choice == "h":
            if not robot.home():
                return
        elif choice == "c":
            if not robot.sync_estimate_from_teensy_steps():
                return
        elif choice == "a":
            robot.assume_homed()
            robot.print_estimate()
        else:
            print("Unknown choice. Aborting.")
            return

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, COMBINED_WIDTH_PX, OVERHEAD_DRAW_H_PX + STEREO_DRAW_H_PX + 2)

        while True:
            ok, frame = overhead_cap.read()
            if not ok or frame is None:
                continue

            dets_overhead = detect_tags(detector, frame)
            det_ee_overhead = dets_overhead.get(EE_TAG_ID)

            stereo_tags, left_raw, right_raw, _, _ = read_stereo_tags_once(stereo, detector, stereo_calib)
            left_display = None
            right_display = None
            det_l_rect = {}
            det_r_rect = {}

            if left_raw is not None and right_raw is not None:
                try:
                    left_display, right_display = rectifier.rectify(left_raw, right_raw)
                    det_l_rect = detect_tags(detector, left_display)
                    det_r_rect = detect_tags(detector, right_display)
                except Exception as exc:
                    print(f"[DISPLAY] Stereo rectification failed: {exc}")
                    left_display, right_display = left_raw, right_raw

            if selected_object is not None:
                refresh_candidate_z_bias(selected_object, robot, bundle, stereo_tags, warn=True)

            x_fk, y_fk, z_fk, phi_fk = robot.fk()
            ee_stereo_visible = EE_TAG_ID in stereo_tags

            lines = [
                "s survey | 1-9 select | m hover | k PICK | f PLACE | n drop | b sanity | r matrix | q quit",
                f"FK=({x_fk:.1f},{y_fk:.1f},z={z_fk:.1f},phi={phi_fk:.1f}) | stereo EE={ee_stereo_visible} | candidates={len(candidates)}",
            ]

            if selected_object is None:
                lines.append("Selected object: none. Press s to survey.")
            else:
                c = selected_object
                lines.append(
                    f"Selected #{c.index} {c.yolo.class_name} conf={c.yolo.confidence:.2f} "
                    f"XY=({c.target_xy[0]:.1f},{c.target_xy[1]:.1f}) "
                    f"Z={c.object_robot_xyz_corrected[2]:.1f} hover={c.hover_robot_z:.1f} grasp={c.grasp_robot_z:.1f}"
                )
                lines.append(
                    f"pts={c.valid_point_count} bias={c.stereo_z_bias_mm:+.1f} "
                    f"xy_source={c.target_xy_source_effective} support={c.support_distance_mm:.1f}"
                )

            if det_ee_overhead is not None:
                ee_xy, _, _, _, _ = map_uv_z_to_robot_xy(det_ee_overhead.center, z_fk, bundle)
                ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                ee_err_norm = float(np.linalg.norm(ee_err))
                status = "OK"
                if ee_err_norm > MAX_EE_ERROR_MM:
                    status = "BAD"
                elif ee_err_norm > WARN_EE_ERROR_MM:
                    status = "WARN"
                lines.append(f"Overhead EE {status}: err=({ee_err[0]:+.1f},{ee_err[1]:+.1f}) |e|={ee_err_norm:.1f}mm")

            lines.append(
                f"PICK: {'HOLDING' if has_item else 'empty'} | drop="
                f"{('(' + f'{drop_zone_xy[0]:.0f},{drop_zone_xy[1]:.0f},phi={drop_zone_phi:.0f}' + ')') if drop_zone_xy is not None else 'NOT SET'} "
                f"| grasp_z={f'{last_grasp_robot_z:.1f}' if last_grasp_robot_z is not None else 'N/A'}"
            )

            if PRINT_GEOMETRY_SANITY and selected_object is not None and (time.time() - last_sanity_print) > GEOMETRY_SANITY_PRINT_EVERY_SEC:
                print_geometry_sanity(selected_object, robot)
                last_sanity_print = time.time()

            overhead_annotated = draw_overhead_status(frame, det_ee_overhead, lines)
            combined = compose_camera_views(
                overhead_annotated,
                left_display,
                right_display,
                det_l_rect,
                det_r_rect,
                candidates,
                selected_index,
            )
            cv2.imshow(WINDOW, combined)

            key = read_command_key(1)

            if key in ("q", "escape"):
                break

            elif key == "s":
                candidates = run_survey_workspace(stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle)
                selected_object = candidates[0] if candidates else None
                selected_index = selected_object.index if selected_object is not None else None
                if selected_object is not None:
                    print(f"[SELECT] Auto-selected #{selected_object.index} {selected_object.yolo.class_name}")

            elif key is not None and key.isdigit() and key != "0":
                idx = int(key)
                match = next((c for c in candidates if c.index == idx), None)
                if match is None:
                    print(f"[SELECT] No candidate #{idx}. Press s to survey.")
                else:
                    selected_object = match
                    selected_index = idx
                    print(f"[SELECT] #{idx} {match.yolo.class_name} conf={match.yolo.confidence:.2f}")

            elif key == "b":
                print_geometry_sanity(selected_object, robot)

            elif key == "m":
                if selected_object is not None:
                    execute_hover_to_object(robot, selected_object, bundle)
                else:
                    print("[MOVE] Need selected object. Press s, then 1-9.")

            elif key == "k":
                if selected_object is None:
                    print("[PICK] Need selected object. Press s, then 1-9.")
                else:
                    print_geometry_sanity(selected_object, robot, prefix="[PICK GEOM]")
                    ok_pick, last_grasp_robot_z = execute_pick(robot, selected_object, bundle)
                    has_item = ok_pick
                    robot.print_estimate()

            elif key == "f":
                if not has_item:
                    print("[PLACE] No item held. Run pick (k) first.")
                elif REFUSE_PLACE_IF_NO_DROP_ZONE and drop_zone_xy is None:
                    print("[PLACE] Drop zone not set. Jog to drop location and press n.")
                elif drop_zone_phi is None:
                    print("[PLACE] Drop zone phi not set. Press n again.")
                elif last_grasp_robot_z is None:
                    print("[PLACE] No grasp height recorded.")
                else:
                    ok_place = execute_place(robot, drop_zone_xy, drop_zone_phi, last_grasp_robot_z)
                    if ok_place:
                        has_item = False
                    robot.print_estimate()

            elif key == "n":
                x_cur, y_cur, _, phi_cur = robot.fk()
                drop_zone_xy = np.array([x_cur, y_cur], dtype=np.float64)
                drop_zone_phi = float(phi_cur)
                print(f"[DROP ZONE] Set to current FK: ({x_cur:.1f}, {y_cur:.1f}, phi={phi_cur:.1f} deg)")

            elif key == "[":
                print(f"[JOG] Z down by {Z_JOG_MM:.1f} mm")
                jog_nonnegative_z(robot, "[JOG] Z down", dz=-Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == "]":
                print(f"[JOG] Z up by {Z_JOG_MM:.1f} mm")
                jog_nonnegative_z(robot, "[JOG] Z up", dz=+Z_JOG_MM, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ",":
                print(f"[JOG] Phi/J4 negative by {PHI_JOG_DEG:.1f} deg")
                jog_nonnegative_z(robot, "[JOG] phi negative", dphi=-PHI_JOG_DEG, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == ".":
                print(f"[JOG] Phi/J4 positive by {PHI_JOG_DEG:.1f} deg")
                jog_nonnegative_z(robot, "[JOG] phi positive", dphi=+PHI_JOG_DEG, move_time_s=0.5)
                robot.sync_estimate_from_teensy_steps()

            elif key == "v":
                if det_ee_overhead is None:
                    print("[VALIDATE] EE tag not visible overhead.")
                else:
                    ee_xy, _, lo, hi, alpha = map_uv_z_to_robot_xy(det_ee_overhead.center, z_fk, bundle)
                    ee_err = ee_xy - np.array([x_fk, y_fk], dtype=np.float64)
                    support_dist, support_idx = nearest_support_distance(ee_xy, z_fk, bundle)
                    print("\n[VALIDATE OVERHEAD EE]")
                    print(f"  FK xy       = ({x_fk:.2f}, {y_fk:.2f}) at z={z_fk:.2f}")
                    print(f"  mapped xy   = ({ee_xy[0]:.2f}, {ee_xy[1]:.2f})")
                    print(f"  error       = ({ee_err[0]:+.2f}, {ee_err[1]:+.2f}) |e|={np.linalg.norm(ee_err):.2f} mm")
                    print(f"  layers      = {lo}/{hi}, alpha={alpha:.3f}")
                    print(f"  support     = {support_dist:.2f} mm nearest idx={support_idx}\n")

            elif key == "r":
                print("\n[STEREO XYZ MODEL]")
                a = bundle.get("A_robot_from_cam_xyz_3x4")
                b = bundle.get("B_cam_from_robot_xyz_3x4")
                a_lin = bundle.get("A_robot_from_cam_xyz_linear_3x3")
                if a is None:
                    print("  No full XYZ stereo model in bundle. Re-run calibration.")
                else:
                    print("  Convention:")
                    print("    robot_xyz = A_robot_from_cam_xyz_3x4 @ [cam_x, cam_y, cam_z, 1]")
                    print("    object_z_corrected = raw_object_z + (FK_z - (raw_ee_tag_z + TAG_TO_EE_Z_MM))")
                    print("    hover_z = object_z_corrected + HOVER_HEIGHT_MM")
                    print("    grasp_z = object_z_corrected + GRASP_OFFSET_MM\n")
                    print(f"  RMSE = {bundle['stereo_robot_xyz_fit_rmse_mm']:.2f} mm")
                    print_matrix_labeled(
                        "A_robot_from_cam_xyz_3x4  (stereo cam -> robot)",
                        a,
                        row_labels=["rob_x", "rob_y", "rob_z"],
                        col_labels=["cam_x", "cam_y", "cam_z", "bias"],
                        units="mm per mm (last col mm)",
                    )
                    if b is not None:
                        print_matrix_labeled(
                            "B_cam_from_robot_xyz_3x4  (robot -> stereo cam)",
                            b,
                            row_labels=["cam_x", "cam_y", "cam_z"],
                            col_labels=["rob_x", "rob_y", "rob_z", "bias"],
                            units="mm per mm (last col mm)",
                        )
                    if a_lin is not None:
                        print_matrix_labeled(
                            "A linear part (3x3) -- how cam delta -> robot delta",
                            a_lin,
                            row_labels=["rob_x", "rob_y", "rob_z"],
                            col_labels=["cam_x", "cam_y", "cam_z"],
                            units="mm per mm",
                        )
                print_vision_paths(raft, yolo)

            elif key == "p":
                robot.print_estimate()

            elif key == "h":
                robot.home()
                robot.print_estimate()

            elif key == "c":
                robot.sync_estimate_from_teensy_steps()
                robot.print_estimate()

            elif key == "a":
                robot.assume_homed()
                robot.print_estimate()

            elif key == "e":
                robot.enable(True)
                robot.init_drivers()

            elif key == "d":
                robot.enable(False)

            elif key == "o":
                print(f"[CLAW] Opening (servo={CLAW_OPEN_DEG} deg)")
                robot.servo(CLAW_OPEN_DEG)

            elif key == "l":
                print(f"[CLAW] Closing (servo={CLAW_CLOSED_DEG} deg)")
                robot.servo(CLAW_CLOSED_DEG)

    finally:
        overhead_cap.release()
        stereo.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
