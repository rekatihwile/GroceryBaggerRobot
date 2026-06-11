from __future__ import annotations

"""config/run/run_config.py — General run control, display, and profiling settings.

Controls loop counts, confirmation gates, prefetch timing, snapshot save,
display hold, and the per-phase survey profiling flag.

Usage:
    from config.run.run_config import DEFAULT_RUN, RunConfig
    _RUN = DEFAULT_RUN
    _RUN = RunConfig(TARGET_OBJECT_COUNT=5, PROFILE_SURVEY_TIMING=False)
"""

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    # ── Loop counts ───────────────────────────────────────────────────────────
    TARGET_OBJECT_COUNT: int = 20
    RUN_UNTIL_NO_VALID_CANDIDATE: bool = False
    MAX_OBJECT_COUNT_SAFETY: int = 120
    REQUIRE_CONFIRM_BEFORE_REAL_MOTION: bool = False

    # ── No-candidate retry behaviour ──────────────────────────────────────────
    NO_CANDIDATE_RETRY_COUNT: int = 1
    NO_CANDIDATE_RETRY_DELAY_S: float = 0.1
    # "continuous" | "wait_for_resume" | "stop"
    NO_CANDIDATE_AFTER_RETRIES_MODE: str = "continuous"
    NO_CANDIDATE_RESUME_KEY: str = "r"
    NO_CANDIDATE_RECOVERY_MOVE_ENABLED: bool = True
    SURVEY_CLEAR_ON_NO_CANDIDATE: bool = True

    # ── Run snapshot / logging ────────────────────────────────────────────────
    SAVE_RUN_SNAPSHOT: bool = True
    RUN_SNAPSHOT_BASE_DIR: Path = Path("data/run_snapshots")
    RUN_LOG_PATH: Path = Path("autonomous_missed_pick_recovery_last_run.txt")

    # ── Prefetch and display timing ───────────────────────────────────────────
    PREFETCH_NEXT_SURVEY_ON_PLACE_DESCENT: bool = True
    PREFETCH_PLACE_DESCENT_DELAY_S: float = 0.0
    SELECTION_DISPLAY_HOLD_S: float = 0.25
    HOLD_WINDOW_AFTER_RUN: bool = True

    # ── Arm-clear box Z/timing ────────────────────────────────────────────────
    CLEAR_BOX_Z_MM: float = 275.0
    CLEAR_BOX_MOVE_TIME_S: float = 1.25

    # ── Per-phase survey profiling ────────────────────────────────────────────
    # When True, the survey pipeline prints elapsed time for each phase
    # (prefresh, burst capture, YOLO, RAFT, overhead, candidate building) so
    # you can identify the bottleneck and tune the speed knobs in SurveyConfig
    # and MissCheckConfig.
    PROFILE_SURVEY_TIMING: bool = True


DEFAULT_RUN = RunConfig()


def run_config_with_overrides(
    config: RunConfig | None = None,
    **overrides,
) -> RunConfig:
    base = DEFAULT_RUN if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
