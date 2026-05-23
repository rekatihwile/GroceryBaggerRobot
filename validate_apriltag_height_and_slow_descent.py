from __future__ import annotations

"""
scripts/validate_apriltag_height_weighted_xy_and_slow_descent.py

Tag-height validation + slow descent test using the same geometry convention as
the fast pick/place script:

    1. Stereo tag ID gives raw stereo camera XYZ.
    2. A_robot_from_cam_xyz_3x4 maps raw stereo XYZ into robot XYZ.
    3. Optional EE/FK stereo-Z bias correction copies fast pick/place:
           stereo_z_bias = FK_z - (raw_ee_tag_z + TAG_TO_EE_Z_MM)
           corrected_target_z = raw_target_z + stereo_z_bias
    4. corrected_target_z is used for height-indexed overhead H(z) lookup.
    5. Final commanded XY is a weighted blend:
           xy_cmd = w_overhead * xy_overhead_Hz + w_stereo * xy_stereo
       normalized by the active weights.
    6. Final commanded Z is:
           z_ee_grip = corrected_target_z + GRIPPER_OFFSET_MM

This does NOT run YOLO, RAFT, BLB, packing, or autonomous pick/place.
It only validates one AprilTag on the known-height reference object and can do
a slow Z descent to the computed grip height.
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Make imports work when run as:
#     python scripts/validate_apriltag_height_weighted_xy_and_slow_descent.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.camera_config import (
    EE_TAG_ID,
    open_overhead_camera as open_overhead_camera_cap,
)
from hardware.cameras.stereo_apriltag_viewer import (
    SimpleStereoCamera,
    build_detector,
    detect_tags,
    draw_detection,
)
from hardware.robot import Robot
from test_calibration_bundle_live_stereo_z_pickplace import (
    BUNDLE_PATH,
    STEREO_CALIBRATION_PATH,
    TriangulatedTag,
    cam_xyz_to_robot_xyz,
    clamp_lookup_z_to_bundle,
    load_bundle,
    load_stereo_calibration,
    map_uv_z_to_robot_xy,
    nearest_support_distance,
    read_stereo_tags_once,
)


# ============================================================
# USER SETTINGS — edit these first
# ============================================================

# Tag on top of the 100 mm printed reference object.
REFERENCE_TAG_ID = 10

# Known physical height from platform surface to tag/top surface.
REFERENCE_OBJECT_HEIGHT_MM = 100.0

# Latest z_ground model produced by:
#     scripts/calibrate_platform_z_from_apriltag.py
Z_GROUND_MODEL_PATH = Path("data/z_ground_calibration/z_ground_model_latest.json")

# Same Z-bias convention as fast pick/place.
TAG_TO_EE_Z_MM = 0.0
USE_EE_FK_Z_BIAS_CORRECTION = True

# For validation, default False so you can still print numbers if EE tag is not
# visible. Set True if you want the "t" test to be stricter like the pick script.
REQUIRE_EE_STEREO_FOR_TEST = False

# Weighted XY fusion.
#
# stereo_xy:
#   from stereo triangulated tag center mapped by A_robot_from_cam_xyz_3x4.
#
# overhead_xy:
#   from overhead tag center mapped through height-indexed H(z), where z is the
#   corrected stereo robot Z above.
#
# final command:
#   xy_cmd = normalized weighted average of available sources.
#
# Practical starting point:
#   use overhead mostly for XY, but keep a little stereo contribution for sanity.
OVERHEAD_XY_WEIGHT = 0.9
STEREO_XY_WEIGHT = 0.1

# Require overhead XY before the "t" test motion. The validation print can still
# run without overhead; motion should not unless you are intentionally debugging
# stereo-only fallback.
REQUIRE_OVERHEAD_XY_FOR_TEST = True

# Robot-Z offset from object top/tag height to commanded end-effector/gripper Z.
#
# Current convention from your robot code:
#     larger robot z_mm = higher EE
#     smaller robot z_mm = lower EE
#
# This script commands:
#     z_ee_grip_robot_mm = corrected_object_top_z_robot_mm + GRIPPER_OFFSET_MM
GRIPPER_OFFSET_MM = 140.0

# Motion heights and speeds.
TRAVEL_Z_MM = 250.0
HOVER_ABOVE_GRIP_MM = 70.0
XY_MOVE_TIME_S = 1.50
RAISE_MOVE_TIME_S = 1.00
HOVER_MOVE_TIME_S = 1.00

# Slow descent is broken into small Z waypoints so you can watch it.
DESCENT_STEP_MM = 5.0
DESCENT_STEP_TIME_S = 0.35

# Robot connection.
CONNECT_ROBOT = True
ENABLE_MOTORS_ON_START = True
INIT_DRIVERS_ON_START = True

STEREO_WINDOW = "Validate Height + Weighted XY - Stereo"
OVERHEAD_WINDOW = "Validate Height + Weighted XY - Overhead"


# ============================================================
# Data containers
# ============================================================

@dataclass
class Measurement:
    tag_id: int

    # Stereo pixel observations.
    u_left_px: float
    v_left_px: float
    u_right_px: float
    v_right_px: float
    disparity_px: float

    # Overhead pixel observation.
    overhead_u_px: float | None
    overhead_v_px: float | None

    # Raw triangulated stereo camera-frame coordinates.
    x_stereo_mm: float
    y_stereo_mm: float
    z_stereo_mm: float

    # Raw target tag mapped through stereo XYZ calibration.
    x_robot_stereo_raw_mm: float
    y_robot_stereo_raw_mm: float
    z_robot_stereo_raw_mm: float

    # EE/FK Z-bias correction, same convention as fast pick/place.
    ee_visible_stereo: bool
    ee_x_robot_raw_mm: float | None
    ee_y_robot_raw_mm: float | None
    ee_z_robot_raw_mm: float | None
    ee_tool_z_from_stereo_raw_mm: float | None
    fk_z_robot_mm: float | None
    stereo_z_bias_mm: float

    # Corrected target Z; XY from stereo is not bias-corrected because this
    # correction is intentionally a Z-only local bias term.
    z_robot_corrected_mm: float

    # Overhead H(z) XY mapped at corrected target Z.
    x_robot_overhead_hz_mm: float | None
    y_robot_overhead_hz_mm: float | None
    overhead_lookup_z_mm: float | None
    overhead_lookup_clamped: bool
    overhead_lookup_lo: int | None
    overhead_lookup_hi: int | None
    overhead_lookup_alpha: float | None
    overhead_support_distance_mm: float | None
    overhead_support_index: int | None

    # Weighted fused command XY.
    x_cmd_robot_mm: float
    y_cmd_robot_mm: float
    xy_source_effective: str
    overhead_weight_used: float
    stereo_weight_used: float

    # Platform / object-height estimate from z_ground model.
    z_ground_pred_mm: float | None
    object_height_pred_mm: float | None

    # Actual robot Z command for the gripper/end-effector.
    z_ee_grip_robot_mm: float


# ============================================================
# z_ground model loading/evaluation
# ============================================================

def load_z_ground_model(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[WARN] z_ground model not found: {path.resolve()}")
        print("[WARN] Height prediction unavailable until you run the z_ground calibration script.")
        return None

    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    print(f"[MODEL] loaded: {path.resolve()}")
    print(
        f"[MODEL] type={model.get('model_type')} "
        f"n={model.get('num_samples')} "
        f"rmse={float(model.get('rmse_mm', float('nan'))):.2f} mm"
    )
    return model


def predict_z_ground_mm(model: dict[str, Any] | None, x_robot_mm: float, y_robot_mm: float) -> float | None:
    if model is None:
        return None

    dx = float(x_robot_mm) - float(model["x_center_mm"])
    dy = float(y_robot_mm) - float(model["y_center_mm"])

    term_values = {
        "1": 1.0,
        "dx": dx,
        "dy": dy,
        "dx*dy": dx * dy,
        "dx^2": dx * dx,
        "dy^2": dy * dy,
    }

    z = 0.0
    for coeff, term in zip(model["coefficients_mm"], model["terms"]):
        if term not in term_values:
            raise ValueError(f"Unknown z_ground model term: {term!r}")
        z += float(coeff) * term_values[term]
    return float(z)


# ============================================================
# Geometry helpers
# ============================================================

def weighted_xy(
    *,
    stereo_xy: np.ndarray,
    overhead_xy: np.ndarray | None,
) -> tuple[np.ndarray, str, float, float]:
    """Fuse stereo XY and overhead H(z) XY with normalized active weights.

    If overhead is missing, returns stereo-only.
    If overhead is present, returns:
        (OVERHEAD_XY_WEIGHT * overhead_xy + STEREO_XY_WEIGHT * stereo_xy)
        / (OVERHEAD_XY_WEIGHT + STEREO_XY_WEIGHT)

    The returned weights are the normalized weights actually used.
    """
    st = np.asarray(stereo_xy, dtype=np.float64).reshape(2)

    if overhead_xy is None:
        return st, "stereo_only_fallback", 0.0, 1.0

    oh = np.asarray(overhead_xy, dtype=np.float64).reshape(2)
    w_oh = max(0.0, float(OVERHEAD_XY_WEIGHT))
    w_st = max(0.0, float(STEREO_XY_WEIGHT))
    denom = w_oh + w_st

    if denom <= 1e-9:
        return oh, "overhead_only_weights_zeroed", 1.0, 0.0

    w_oh_n = w_oh / denom
    w_st_n = w_st / denom
    xy = w_oh_n * oh + w_st_n * st
    return xy.astype(np.float64), "weighted_overhead_hz_plus_stereo", w_oh_n, w_st_n


def compute_measurement(
    *,
    stereo_tags: dict[int, TriangulatedTag],
    overhead_det: Any | None,
    bundle: dict[str, Any],
    z_model: dict[str, Any] | None,
    robot: Robot | None,
) -> Measurement | None:
    """Compute tag validation geometry using fast-pick-style conventions.

    This intentionally mirrors the fast pick/place ObjectCandidate update logic:

      raw_robot = cam_xyz_to_robot_xyz(target_cam_xyz)
      ee_raw = cam_xyz_to_robot_xyz(ee_cam_xyz) if visible
      stereo_z_bias = FK_z - (ee_raw.z + TAG_TO_EE_Z_MM)
      corrected_z = raw_robot.z + stereo_z_bias
      lookup_z_used = clamp_lookup_z_to_bundle(corrected_z)
      overhead_xy = map_uv_z_to_robot_xy(overhead_uv, lookup_z_used)
      target_xy = weighted blend of overhead_xy and stereo raw XY
      grip_z = corrected_z + GRIPPER_OFFSET_MM
    """
    stereo_tag = stereo_tags.get(REFERENCE_TAG_ID)
    if stereo_tag is None:
        return None

    raw_robot = cam_xyz_to_robot_xyz(stereo_tag.xyz_cam_mm, bundle)
    x_st = float(raw_robot[0])
    y_st = float(raw_robot[1])
    z_raw = float(raw_robot[2])

    # Optional EE/FK local Z-bias correction, same sign convention as fast pick/place.
    fk_z: float | None = None
    ee_raw = None
    ee_tool_z_raw: float | None = None
    stereo_z_bias = 0.0

    if robot is not None:
        try:
            _x_fk, _y_fk, z_fk, _phi_fk = robot.fk()
            fk_z = float(z_fk)
        except Exception:
            fk_z = None

    ee_tri = stereo_tags.get(EE_TAG_ID)
    if ee_tri is not None:
        ee_raw = cam_xyz_to_robot_xyz(ee_tri.xyz_cam_mm, bundle)
        ee_tool_z_raw = float(ee_raw[2] + TAG_TO_EE_Z_MM)
        if USE_EE_FK_Z_BIAS_CORRECTION and fk_z is not None:
            stereo_z_bias = float(fk_z - ee_tool_z_raw)

    z_corr = float(z_raw + stereo_z_bias)

    lookup_z_used, lookup_z_clamped = clamp_lookup_z_to_bundle(z_corr, bundle)

    oh_u = None
    oh_v = None
    xy_oh = None
    lookup_lo = lookup_hi = None
    lookup_alpha = None
    support_dist = None
    support_idx = None

    if overhead_det is not None:
        oh_u = float(overhead_det.center[0])
        oh_v = float(overhead_det.center[1])
        try:
            xy_tmp, _uv_undist, lo, hi, alpha = map_uv_z_to_robot_xy(
                np.asarray([oh_u, oh_v], dtype=np.float64),
                lookup_z_used,
                bundle,
            )
            xy_oh = np.asarray(xy_tmp, dtype=np.float64).reshape(2)
            lookup_lo = int(lo)
            lookup_hi = int(hi)
            lookup_alpha = float(alpha)
            support_dist, support_idx = nearest_support_distance(xy_oh, lookup_z_used, bundle)
            support_dist = float(support_dist)
            support_idx = int(support_idx)
        except Exception as exc:
            print(f"[WARN] overhead H(z) mapping failed: {exc}")
            xy_oh = None

    xy_cmd, xy_source, w_oh_used, w_st_used = weighted_xy(
        stereo_xy=np.array([x_st, y_st], dtype=np.float64),
        overhead_xy=xy_oh,
    )

    z_ground = predict_z_ground_mm(z_model, float(xy_cmd[0]), float(xy_cmd[1]))
    object_height = None if z_ground is None else float(z_corr - z_ground)

    z_ee_grip = float(z_corr + GRIPPER_OFFSET_MM)

    return Measurement(
        tag_id=int(stereo_tag.tag_id),
        u_left_px=float(stereo_tag.left_px[0]),
        v_left_px=float(stereo_tag.left_px[1]),
        u_right_px=float(stereo_tag.right_px[0]),
        v_right_px=float(stereo_tag.right_px[1]),
        disparity_px=float(stereo_tag.disparity_px),
        overhead_u_px=oh_u,
        overhead_v_px=oh_v,
        x_stereo_mm=float(stereo_tag.xyz_cam_mm[0]),
        y_stereo_mm=float(stereo_tag.xyz_cam_mm[1]),
        z_stereo_mm=float(stereo_tag.xyz_cam_mm[2]),
        x_robot_stereo_raw_mm=x_st,
        y_robot_stereo_raw_mm=y_st,
        z_robot_stereo_raw_mm=z_raw,
        ee_visible_stereo=ee_tri is not None,
        ee_x_robot_raw_mm=None if ee_raw is None else float(ee_raw[0]),
        ee_y_robot_raw_mm=None if ee_raw is None else float(ee_raw[1]),
        ee_z_robot_raw_mm=None if ee_raw is None else float(ee_raw[2]),
        ee_tool_z_from_stereo_raw_mm=ee_tool_z_raw,
        fk_z_robot_mm=fk_z,
        stereo_z_bias_mm=float(stereo_z_bias),
        z_robot_corrected_mm=z_corr,
        x_robot_overhead_hz_mm=None if xy_oh is None else float(xy_oh[0]),
        y_robot_overhead_hz_mm=None if xy_oh is None else float(xy_oh[1]),
        overhead_lookup_z_mm=float(lookup_z_used),
        overhead_lookup_clamped=bool(lookup_z_clamped),
        overhead_lookup_lo=lookup_lo,
        overhead_lookup_hi=lookup_hi,
        overhead_lookup_alpha=lookup_alpha,
        overhead_support_distance_mm=support_dist,
        overhead_support_index=support_idx,
        x_cmd_robot_mm=float(xy_cmd[0]),
        y_cmd_robot_mm=float(xy_cmd[1]),
        xy_source_effective=xy_source,
        overhead_weight_used=float(w_oh_used),
        stereo_weight_used=float(w_st_used),
        z_ground_pred_mm=z_ground,
        object_height_pred_mm=object_height,
        z_ee_grip_robot_mm=z_ee_grip,
    )


def print_validation(m: Measurement) -> None:
    print("\n" + "=" * 96)
    print(f"[VALIDATE] AprilTag ID {m.tag_id} — fast-pick-style weighted XY + corrected stereo Z")

    print(
        "1) stereo pixels:"
        f"  [(u_L,v_L)=({m.u_left_px:.2f}, {m.v_left_px:.2f}), "
        f"(u_R,v_R)=({m.u_right_px:.2f}, {m.v_right_px:.2f})]  "
        f"disparity={m.disparity_px:.2f} px"
    )

    if m.overhead_u_px is None:
        print("   overhead pixel: unavailable / tag not detected overhead")
    else:
        print(
            "   overhead pixel:"
            f"  (u_OH,v_OH)=({m.overhead_u_px:.2f}, {m.overhead_v_px:.2f})"
        )

    print(
        "2) stereo camera frame:"
        f"  x_S={m.x_stereo_mm:+.2f} mm,"
        f" y_S={m.y_stereo_mm:+.2f} mm,"
        f" z_S={m.z_stereo_mm:+.2f} mm"
    )

    print(
        "3a) raw stereo mapped robot frame:"
        f"  x_st={m.x_robot_stereo_raw_mm:+.2f} mm,"
        f" y_st={m.y_robot_stereo_raw_mm:+.2f} mm,"
        f" z_raw={m.z_robot_stereo_raw_mm:+.2f} mm"
    )

    print("3b) EE/FK Z-bias correction:")
    print(f"    EE stereo visible = {m.ee_visible_stereo}")
    if m.ee_visible_stereo:
        print(
            f"    EE raw robot xyz  = ({m.ee_x_robot_raw_mm:+.2f}, "
            f"{m.ee_y_robot_raw_mm:+.2f}, {m.ee_z_robot_raw_mm:+.2f}) mm"
        )
        print(f"    EE tool z raw     = {m.ee_tool_z_from_stereo_raw_mm:+.2f} mm")
    else:
        print("    EE raw robot xyz  = unavailable")
    print(f"    FK z              = {m.fk_z_robot_mm if m.fk_z_robot_mm is not None else 'unavailable'}")
    print(f"    stereo_z_bias     = {m.stereo_z_bias_mm:+.2f} mm")
    print(f"    z_corrected       = z_raw + bias = {m.z_robot_corrected_mm:+.2f} mm")

    if m.x_robot_overhead_hz_mm is None:
        print("3c) overhead H(z) robot XY: unavailable")
    else:
        clamped = " CLAMPED" if m.overhead_lookup_clamped else ""
        print(
            "3c) overhead H(z) robot XY:"
            f"  x_oh={m.x_robot_overhead_hz_mm:+.2f} mm,"
            f" y_oh={m.y_robot_overhead_hz_mm:+.2f} mm,"
            f" lookup_z={m.overhead_lookup_z_mm:+.2f} mm{clamped},"
            f" H[{m.overhead_lookup_lo}->{m.overhead_lookup_hi}] alpha={m.overhead_lookup_alpha:+.3f}"
        )
        dx = m.x_robot_overhead_hz_mm - m.x_robot_stereo_raw_mm
        dy = m.y_robot_overhead_hz_mm - m.y_robot_stereo_raw_mm
        dxy = math.hypot(dx, dy)
        print(
            "    overhead-vs-stereo XY delta:"
            f"  dx={dx:+.2f} mm, dy={dy:+.2f} mm, norm={dxy:.2f} mm"
        )
        if m.overhead_support_distance_mm is not None:
            print(
                f"    support distance = {m.overhead_support_distance_mm:.2f} mm "
                f"nearest idx={m.overhead_support_index}"
            )

    print(
        "3d) WEIGHTED COMMAND XY:"
        f"  x_cmd={m.x_cmd_robot_mm:+.2f} mm,"
        f" y_cmd={m.y_cmd_robot_mm:+.2f} mm,"
        f" source={m.xy_source_effective},"
        f" w_overhead={m.overhead_weight_used:.3f},"
        f" w_stereo={m.stereo_weight_used:.3f}"
    )

    if m.z_ground_pred_mm is None or m.object_height_pred_mm is None:
        print("4) predicted object height: unavailable because z_ground model is missing")
        print(f"   known reference object height = {REFERENCE_OBJECT_HEIGHT_MM:.2f} mm")
    else:
        err = m.object_height_pred_mm - REFERENCE_OBJECT_HEIGHT_MM
        print(
            "4) predicted object height:"
            f"  z_ground_pred={m.z_ground_pred_mm:+.2f} mm,"
            f" height={m.object_height_pred_mm:.2f} mm,"
            f" error_vs_known={err:+.2f} mm"
        )

    print(
        "5) estimated EE/gripper Z command:"
        f"  z_EE_grip = z_corrected_object_top + GRIPPER_OFFSET"
        f" = {m.z_robot_corrected_mm:+.2f} + ({GRIPPER_OFFSET_MM:+.2f})"
        f" = {m.z_ee_grip_robot_mm:+.2f} mm"
    )
    print("=" * 96 + "\n")


# ============================================================
# Robot motion
# ============================================================

def startup_robot() -> Robot | None:
    if not CONNECT_ROBOT:
        print("[ROBOT] CONNECT_ROBOT=False; validation only, no test motion.")
        return None

    robot = Robot(connect=True)

    if ENABLE_MOTORS_ON_START:
        robot.enable(True)
    if INIT_DRIVERS_ON_START:
        robot.init_drivers()

    print("\nStartup options:")
    print("  h = run HOME now")
    print("  c = continue from current Teensy step counters, no homing")
    print("  a = assume robot is physically at configured home_pose, no homing")
    choice = input("Choose h/c/a: ").strip().lower()

    if choice == "h":
        if not robot.home():
            raise RuntimeError("HOME failed")
    elif choice == "c":
        if not robot.sync_estimate_from_teensy_steps():
            raise RuntimeError("Teensy sync failed")
    elif choice == "a":
        robot.assume_homed()
        robot.print_estimate()
    else:
        raise RuntimeError("Unknown startup choice")

    print("[ROBOT] ready")
    return robot


def _require_safe_z(z_mm: float, label: str) -> None:
    if not math.isfinite(float(z_mm)):
        raise RuntimeError(f"{label}: z is not finite: {z_mm}")
    if float(z_mm) < 0.0:
        raise RuntimeError(f"{label}: refusing to command negative robot Z = {z_mm:.2f} mm")


def run_slow_descent_test(robot: Robot | None, m: Measurement) -> bool:
    """Raise, move above object using weighted XY, hover, then descend slowly."""
    if robot is None:
        print("[TEST] robot is not connected; set CONNECT_ROBOT=True")
        return False

    if REQUIRE_EE_STEREO_FOR_TEST and not m.ee_visible_stereo:
        print("[TEST] refused: EE tag is not stereo-visible and REQUIRE_EE_STEREO_FOR_TEST=True")
        return False

    if REQUIRE_OVERHEAD_XY_FOR_TEST and m.x_robot_overhead_hz_mm is None:
        print("[TEST] refused: overhead H(z) XY unavailable, and REQUIRE_OVERHEAD_XY_FOR_TEST=True")
        return False

    x = float(m.x_cmd_robot_mm)
    y = float(m.y_cmd_robot_mm)
    z_grip = float(m.z_ee_grip_robot_mm)
    z_hover = float(z_grip + HOVER_ABOVE_GRIP_MM)
    z_travel = float(max(TRAVEL_Z_MM, z_hover))

    _require_safe_z(z_grip, "[TEST] grip")
    _require_safe_z(z_hover, "[TEST] hover")
    _require_safe_z(z_travel, "[TEST] travel")

    _, _, _, phi = robot.fk()

    print("\n" + "-" * 96)
    print("[TEST] slow descent plan")
    print(f"  target XY:       x={x:.2f} mm, y={y:.2f} mm  source={m.xy_source_effective}")
    print(f"  weights:         overhead={m.overhead_weight_used:.3f}, stereo={m.stereo_weight_used:.3f}")
    print(f"  raw stereo Z:    {m.z_robot_stereo_raw_mm:.2f} mm")
    print(f"  stereo Z bias:   {m.stereo_z_bias_mm:+.2f} mm")
    print(f"  corrected top Z: {m.z_robot_corrected_mm:.2f} mm")
    print(f"  travel Z:        {z_travel:.2f} mm")
    print(f"  hover Z:         {z_hover:.2f} mm")
    print(f"  grip Z:          {z_grip:.2f} mm")
    print(f"  current phi:     {phi:.2f} deg")
    print("-" * 96)

    print("[TEST] 1/4 raise to travel Z")
    if not robot.move_cartesian(z_mm=z_travel, move_time_s=RAISE_MOVE_TIME_S):
        return False

    print("[TEST] 2/4 move weighted XY over object at travel Z")
    if not robot.move_cartesian(x_mm=x, y_mm=y, z_mm=z_travel, phi_deg=phi, move_time_s=XY_MOVE_TIME_S):
        return False

    print("[TEST] 3/4 descend to hover Z")
    if not robot.move_cartesian(z_mm=z_hover, move_time_s=HOVER_MOVE_TIME_S):
        return False

    print("[TEST] 4/4 slow descent to grip Z")
    total_drop = max(0.0, z_hover - z_grip)
    n_steps = max(1, int(math.ceil(total_drop / max(DESCENT_STEP_MM, 0.1))))
    for i, z in enumerate(np.linspace(z_hover, z_grip, n_steps + 1)[1:], start=1):
        print(f"  descent step {i}/{n_steps}: z={float(z):.2f} mm")
        if not robot.move_cartesian(z_mm=float(z), move_time_s=DESCENT_STEP_TIME_S):
            return False

    print("[TEST] done — robot is at computed grip height. No claw action was commanded.")
    return True


# ============================================================
# Display
# ============================================================

def put_text_outline(img: np.ndarray, text: str, org: tuple[int, int], scale=0.55, color=(255, 255, 255), thickness=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def make_stereo_preview(
    left: np.ndarray,
    right: np.ndarray,
    det_l_ref: Any | None,
    det_r_ref: Any | None,
    det_l_ee: Any | None,
    det_r_ee: Any | None,
    latest: Measurement | None,
) -> np.ndarray:
    left_draw = draw_detection(left, det_l_ref, f"LEFT REF {REFERENCE_TAG_ID}")
    right_draw = draw_detection(right, det_r_ref, f"RIGHT REF {REFERENCE_TAG_ID}")

    # Draw EE tag too if visible, so you can verify whether Z-bias correction is active.
    if det_l_ee is not None and EE_TAG_ID != REFERENCE_TAG_ID:
        left_draw = draw_detection(left_draw, det_l_ee, f"LEFT EE {EE_TAG_ID}")
    if det_r_ee is not None and EE_TAG_ID != REFERENCE_TAG_ID:
        right_draw = draw_detection(right_draw, det_r_ee, f"RIGHT EE {EE_TAG_ID}")

    preview = np.hstack([left_draw, right_draw])

    if latest is None:
        line1 = f"ID {REFERENCE_TAG_ID} not detected in BOTH stereo frames"
    else:
        h_text = "height unavailable" if latest.object_height_pred_mm is None else f"h_pred={latest.object_height_pred_mm:.1f}mm"
        ee_text = "EE=yes" if latest.ee_visible_stereo else "EE=no"
        line1 = (
            f"LIVE weighted_xy=({latest.x_cmd_robot_mm:+.1f}, {latest.y_cmd_robot_mm:+.1f})mm  "
            f"z_raw={latest.z_robot_stereo_raw_mm:+.1f}mm  "
            f"bias={latest.stereo_z_bias_mm:+.1f}mm  "
            f"z_corr={latest.z_robot_corrected_mm:+.1f}mm  "
            f"{h_text}  z_grip={latest.z_ee_grip_robot_mm:+.1f}mm  {ee_text}"
        )
    line2 = "Keys: v=validate print   t=test slow descent   q/ESC=quit"

    row_h = 22
    bar = np.zeros((12 + row_h * 2, preview.shape[1], 3), dtype=np.uint8)
    for i, line in enumerate([line1, line2]):
        put_text_outline(bar, line, (8, 20 + i * row_h), scale=0.55, color=(230, 230, 230), thickness=1)

    return np.vstack([preview, bar])


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("=" * 96)
    print("Validate tag height + fast-pick-style weighted XY + slow descent")
    print(f"REFERENCE_TAG_ID={REFERENCE_TAG_ID}")
    print(f"EE_TAG_ID={EE_TAG_ID}")
    print(f"REFERENCE_OBJECT_HEIGHT_MM={REFERENCE_OBJECT_HEIGHT_MM:.1f}")
    print(f"TAG_TO_EE_Z_MM={TAG_TO_EE_Z_MM:+.1f}")
    print(f"USE_EE_FK_Z_BIAS_CORRECTION={USE_EE_FK_Z_BIAS_CORRECTION}")
    print(f"XY weights: overhead={OVERHEAD_XY_WEIGHT:.3f}, stereo={STEREO_XY_WEIGHT:.3f}")
    print(f"GRIPPER_OFFSET_MM={GRIPPER_OFFSET_MM:+.1f}")
    print("Keys: v=validate, t=test slow descent, q/ESC=quit")
    print("=" * 96)

    bundle = load_bundle(BUNDLE_PATH)
    stereo_calib = load_stereo_calibration(STEREO_CALIBRATION_PATH)
    z_model = load_z_ground_model(Z_GROUND_MODEL_PATH)

    detector = build_detector()
    stereo = SimpleStereoCamera()

    overhead_cap = None
    try:
        overhead_cap = open_overhead_camera_cap()
        cv2.namedWindow(OVERHEAD_WINDOW, cv2.WINDOW_NORMAL)
    except Exception as exc:
        print(f"[WARN] overhead camera unavailable: {exc}")
        print("[WARN] Validation will still show stereo. Test may be refused depending on REQUIRE_OVERHEAD_XY_FOR_TEST.")

    robot = startup_robot()

    cv2.namedWindow(STEREO_WINDOW, cv2.WINDOW_NORMAL)
    latest_measurement: Measurement | None = None

    try:
        while True:
            stereo_tags, left, right, det_l_all, det_r_all = read_stereo_tags_once(stereo, detector, stereo_calib)

            overhead_frame = None
            overhead_det = None
            if overhead_cap is not None:
                ok_oh, overhead_frame = overhead_cap.read()
                if ok_oh and overhead_frame is not None:
                    oh_dets = detect_tags(detector, overhead_frame)
                    overhead_det = oh_dets.get(REFERENCE_TAG_ID)

                    oh_draw = draw_detection(overhead_frame, overhead_det, f"OVERHEAD REF {REFERENCE_TAG_ID}")
                    cv2.imshow(OVERHEAD_WINDOW, oh_draw)

            latest_measurement = compute_measurement(
                stereo_tags=stereo_tags,
                overhead_det=overhead_det,
                bundle=bundle,
                z_model=z_model,
                robot=robot,
            )

            if left is not None and right is not None:
                preview = make_stereo_preview(
                    left,
                    right,
                    det_l_all.get(REFERENCE_TAG_ID),
                    det_r_all.get(REFERENCE_TAG_ID),
                    det_l_all.get(EE_TAG_ID),
                    det_r_all.get(EE_TAG_ID),
                    latest_measurement,
                )
                cv2.imshow(STEREO_WINDOW, preview)

            key = cv2.waitKeyEx(1)
            if key < 0:
                continue
            key8 = key & 0xFF

            if key8 in (ord("q"), 27):
                print("[QUIT] requested")
                break

            if key8 == ord("v"):
                if latest_measurement is None:
                    print(f"[VALIDATE] cannot validate: ID {REFERENCE_TAG_ID} not detected in BOTH stereo frames")
                    continue
                print_validation(latest_measurement)
                continue

            if key8 == ord("t"):
                if latest_measurement is None:
                    print(f"[TEST] cannot test: ID {REFERENCE_TAG_ID} not detected in BOTH stereo frames")
                    continue
                print_validation(latest_measurement)
                ok = run_slow_descent_test(robot, latest_measurement)
                print(f"[TEST] {'success' if ok else 'failed'}")
                continue

    finally:
        stereo.release()
        if overhead_cap is not None:
            overhead_cap.release()
        if robot is not None:
            robot.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
