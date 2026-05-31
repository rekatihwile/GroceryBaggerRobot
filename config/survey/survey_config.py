from __future__ import annotations

"""config/survey/survey_config.py — Vision pipeline configuration.

Covers YOLO, RAFT-Stereo, burst tracking, and overhead camera settings.
All values here become the default when scripts import DEFAULT_SURVEY.

Usage pattern (mirrors ZSafetyConfig):
    from config.survey.survey_config import DEFAULT_SURVEY, SurveyConfig
    _SURVEY = DEFAULT_SURVEY  # use defaults
    _SURVEY = SurveyConfig(YOLO_CONF=0.50, BURST_COUNT=15)  # custom instance
"""

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class SurveyConfig:
    # ── Calibration / model paths ──────────────────────────────────────────
    STEREO_CALIBRATION_PATH: Path = Path("stereo_calibration.npz")
    BUNDLE_PATH: Path = Path("robot_calibration_bundle.npz")

    YOLO_WEIGHTS_PATH: Path = Path("yolo_weights/full_data.pt")
    YOLO_FALLBACK_WEIGHTS_PATH: Path = Path("yolo_weights/validate_V2.pt")
    RAFT_ROOT: Path = Path("RAFT-Stereo")
    RAFT_CHECKPOINT_PATH: Path = Path("RAFT-Stereo/models/raftstereo-middlebury.pth")

    # ── YOLO segmentation ─────────────────────────────────────────────────
    YOLO_IMGSZ: int = 640
    YOLO_CONF: float = 0.35
    YOLO_IOU: float = 0.50
    YOLO_RETINA_MASKS: bool = True
    # Empty tuple = all classes.  Set e.g. ("Lays_Chips", "Pringles") to filter.
    TARGET_CLASS_NAMES: tuple[str, ...] = ()
    MIN_MASK_AREA_PX: int = 500

    # ── CUDA / half-precision ────────────────────────────────────────────
    USE_CUDA: bool = True
    USE_HALF: bool = True

    # ── RAFT-Stereo disparity ────────────────────────────────────────────
    RAFT_VALID_ITERS: int = 16
    RAFT_DOWNSCALE: float = 1.0
    RAFT_MIXED_PRECISION: bool = True
    MIN_DISPARITY_PX: float = 1.0

    # ── Point-cloud quality ──────────────────────────────────────────────
    MIN_VALID_OBJECT_POINTS: int = 300

    # ── Burst tracking ───────────────────────────────────────────────────
    # How many stereo frames to grab per survey; a candidate must appear in
    # at least MIN_BURST_HITS frames to survive the cluster filter.
    BURST_COUNT: int = 10
    MIN_BURST_HITS: int = 3
    BURST_FRAME_DELAY_S: float = 0.05
    BURST_CLUSTER_MAX_CENTROID_PX: float = 75.0
    BURST_REQUIRE_SAME_CLASS: bool = True

    # ── Overhead camera ──────────────────────────────────────────────────
    # Discard this many queued overhead frames before the survey reads one;
    # prevents stale frames from appearing during RAFT/pointcloud work.
    OVERHEAD_FRESH_READ_DISCARD_FRAMES: int = 6
    OVERHEAD_FRESH_READ_DELAY_S: float = 0.02
    OVERHEAD_MATCH_MAX_DIST_MM: float = 140.0
    OVERHEAD_MATCH_PREFER_SAME_CLASS: bool = True
    # How much the overhead-projected XY pulls the final target_xy toward it.
    # 0.0 = pure stereo, 1.0 = pure overhead.
    OVERHEAD_XY_BLEND_WEIGHT: float = 0.45


DEFAULT_SURVEY = SurveyConfig()


def survey_config_with_overrides(
    config: SurveyConfig | None = None,
    **overrides,
) -> SurveyConfig:
    base = DEFAULT_SURVEY if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
