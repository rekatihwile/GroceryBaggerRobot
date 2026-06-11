from __future__ import annotations

"""config/run/miss_check_config.py — YOLO miss-pick watchdog settings.

After each pick the script runs a cheap overhead YOLO burst to decide whether
the gripper actually caught the item.  These knobs control burst count,
timeout, and the matching/scoring thresholds.

Speed knobs:
- Reduce MISS_BURST_COUNT to 1 for the fastest possible check.
- Reduce MISS_CHECK_PERIOD_S to 0.05 to poll more aggressively.
- Reduce MISS_CHECK_TIMEOUT_S to shorten the maximum wait.

Usage:
    from config.run.miss_check_config import DEFAULT_MISS_CHECK, MissCheckConfig
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MissCheckConfig:
    # ── Enable / camera ───────────────────────────────────────────────────────
    MISS_CHECK_ENABLED: bool = True
    # Camera used for the post-pick verification burst.
    # "overhead" is cheapest; it avoids RAFT and runs YOLO only.
    MISS_CHECK_CAMERA: str = "overhead"

    # ── Burst capture (speed knobs) ───────────────────────────────────────────
    MISS_BURST_COUNT: int = 2
    MISS_MIN_HITS: int = 1
    MISS_CHECK_TIMEOUT_S: float = 1.0
    MISS_CHECK_PERIOD_S: float = 0.15
    # Only run miss-check when at the place hover, not mid-ascent.
    MISS_CONFIRM_AT_PLACE_HOVER_ONLY: bool = True

    # ── Matching gates ────────────────────────────────────────────────────────
    MISS_MATCH_MAX_ROBOT_DIST_MM: float = 35.0
    MISS_MATCH_MIN_IOU: float = 0.20
    MISS_MATCH_AREA_RATIO_MIN: float = 0.50
    MISS_MATCH_AREA_RATIO_MAX: float = 2.00

    # ── Scoring weights (should sum to 1.0) ───────────────────────────────────
    MISS_SCORE_THRESHOLD: float = 0.65
    MISS_CLASS_WEIGHT: float = 0.25
    MISS_ROBOT_DIST_WEIGHT: float = 0.35
    MISS_IOU_WEIGHT: float = 0.20
    MISS_AREA_WEIGHT: float = 0.20


DEFAULT_MISS_CHECK = MissCheckConfig()


def miss_check_config_with_overrides(
    config: MissCheckConfig | None = None,
    **overrides,
) -> MissCheckConfig:
    base = DEFAULT_MISS_CHECK if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
