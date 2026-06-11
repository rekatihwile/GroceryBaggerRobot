from __future__ import annotations

"""config/pick/pick_config.py — Pick perception and motion configuration.

Covers phi resolution, Z estimation, grasp XY policy, and the robot survey pose.
Z safety limits are in config/motion/z_safety_config.py; servo geometry is in
config/pick/servo_config.py.

Usage:
    from config.pick.pick_config import DEFAULT_PICK, PickConfig
    _PICK = DEFAULT_PICK                          # use all defaults
    _PICK = PickConfig(PICK_PHI_MODE="current_fk")  # override one field
"""

from dataclasses import dataclass, replace
from pathlib import Path

from config.motion.z_safety_config import DEFAULT_Z_SAFETY


@dataclass(frozen=True)
class PickConfig:
    # ── Survey / home pose ───────────────────────────────────────────────
    # The robot moves here before surveying.  Recovery pose defaults to this.
    X_SURVEY_MM: float = 500.0
    Y_SURVEY_MM: float = -50.0
    Z_SURVEY_MM: float = 270.0

    # ── Gripper angle (φ) resolution ─────────────────────────────────────
    # Selects how the gripper yaw is chosen from the detection geometry.
    # Options: centroid_shortest_ray_parallel | centroid_longest_ray_perp |
    #          overhead_semi_minor_projected | mask_minor_axis_pointcloud |
    #          triangulated_short_side | overhead_minor_axis | current_fk
    PICK_PHI_MODE: str = "centroid_shortest_ray_parallel"
    OVERHEAD_USE_SEMIMINOR_AXIS_FOR_PHI: bool = True
    USE_CONFIDENCE_PHI_BLEND: bool = False
    PHI_DISAGREEMENT_WARN_DEG: float = 25.0
    PHI_MIN_CONFIDENCE: float = 0.20
    PHI_ASPECT_DECAY: float = 0.8
    PHI_STEREO_HEIGHT_DECAY_CM: float = 8.0
    PHI_FALLBACK_TO_CURRENT_EE_PHI: bool = False

    # ── Z safety bounds (must mirror config/motion/z_safety_config.py) ───
    Z_MAX_MM: float = float(DEFAULT_Z_SAFETY.Z_MAX_MM)
    MIN_PICK_GRASP_Z_MM: float = float(DEFAULT_Z_SAFETY.MIN_PICK_GRASP_Z_MM)
    GRIPPER_OFFSET_MM: float = float(DEFAULT_Z_SAFETY.GRIPPER_OFFSET_MM)

    # ── Robust Z estimation from point cloud ──────────────────────────────
    USE_ROBUST_OBJECT_Z: bool = True
    ROBUST_TOP_PERCENTILE: float = 95.0
    ROBUST_BOTTOM_PERCENTILE: float = 5.0
    TOP_SPREAD_LOW_PERCENTILE: float = 90.0
    TOP_SPREAD_HIGH_PERCENTILE: float = 99.0
    Z_UNCERTAINTY_CLEARANCE_GAIN: float = 1.0
    Z_UNCERTAINTY_CLEARANCE_MIN_MM: float = 2.5
    Z_UNCERTAINTY_CLEARANCE_MAX_MM: float = 20.0
    Z_UNCERTAINTY_WARN_MM: float = 20.0
    PICK_EXTRA_CLEARANCE_MM: float = 0.0
    PICK_Z_UNCERTAINTY_GAIN: float = 0.25
    PICK_Z_UNCERTAINTY_CLEARANCE_MAX_MM: float = 20.0
    MIN_OBJECT_HEIGHT_MM: float = 2.0
    MAX_OBJECT_HEIGHT_MM: float = 180.0

    # ── Platform Z ground model ───────────────────────────────────────────
    # When True, pick grasp Z = z_ground_model(x, y) + estimated_object_height
    # rather than the stereo-derived surface Z.  Turn off if the model is
    # clearly worse than stereo Z at the current platform position.
    USE_Z_GROUND_MODEL_FOR_PICK_SURFACE: bool = True
    Z_GROUND_MODEL_PATH: Path = Path("data/z_ground_calibration/z_ground_model_latest.json")

    # ── Overhead XY ───────────────────────────────────────────────────────
    # Refuse picks where the overhead camera could not confirm an XY position.
    REQUIRE_OVERHEAD_XY_FOR_PICK: bool = True
    REFUSE_PICK_IF_TOO_FEW_POINTS: bool = True

    # ── Local height-aware grasp XY ───────────────────────────────────────
    # Shifts the grasp point slightly toward the local highest region of the
    # object so the gripper lands on the widest/most stable part.
    USE_LOCAL_HEIGHT_AWARE_GRASP_XY: bool = True
    LOCAL_GRASP_RADIUS_MM: float = 1000.0
    LOCAL_GRASP_TOP_REGION_PERCENTILE: float = 95.0
    LOCAL_GRASP_HEIGHT_DELTA_THRESHOLD_MM: float = 1.0
    LOCAL_GRASP_BLEND_WEIGHT: float = 0.2
    LOCAL_GRASP_MAX_SHIFT_MM: float = 100.0

    # ── Display (for pick_one_place_one interactive viewer) ───────────────
    COMBINED_WIDTH_PX: int = 1280
    OVERHEAD_DRAW_H_PX: int = 560
    STEREO_DRAW_H_PX: int = 390
    STATUS_H_PX: int = 140


DEFAULT_PICK = PickConfig()


def pick_config_with_overrides(
    config: PickConfig | None = None,
    **overrides,
) -> PickConfig:
    base = DEFAULT_PICK if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
