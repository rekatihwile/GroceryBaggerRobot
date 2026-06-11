from __future__ import annotations

"""config/run/recovery_config.py — Pick recovery pose and retry-grasp settings.

The recovery pose is where the arm moves after a confirmed miss or a failed
retry.  The retry settings control how the re-pick attempt differs from the
original (higher Z, wider claw, optional re-localisation).

Usage:
    from config.run.recovery_config import DEFAULT_RECOVERY, RecoveryConfig
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RecoveryConfig:
    # ── Recovery pose (None = use survey pose from PickConfig) ────────────────
    RECOVERY_POSE_X_MM: float | None = None
    RECOVERY_POSE_Y_MM: float | None = None
    RECOVERY_POSE_Z_MM: float | None = None
    RECOVERY_POSE_PHI_DEG: float = 0.0
    RECOVERY_MOVE_TIME_S: float = 1.50

    # ── J3 rehome ─────────────────────────────────────────────────────────────
    RECOVERY_REHOME_J3_ENABLED: bool = True
    # Drop to this Z before rehoming J3 so the arm doesn't swing wide at height.
    # None = no pre-drop.
    RECOVERY_DROP_Z_BEFORE_REHOME_MM: float | None = None
    RECOVERY_REHOME_TIMEOUT_S: float = 120.0

    # ── Retry-grasp ───────────────────────────────────────────────────────────
    MAX_PICK_RETRIES_PER_OBJECT: int = 1
    RETRY_SAFE_PICK_ENABLED: bool = True
    # Raise the grasp target by this amount so the gripper doesn't dig into
    # the same failed contact point on retry.
    RETRY_GRASP_Z_OFFSET_MM: float = 10.0
    # Open the claw wider than normal on the retry to increase catch probability.
    RETRY_START_CLAW_EXTRA_DEG: float = 8.0
    RETRY_START_CLAW_MAX_DEG: float = 80.0
    # If the retry XY is within this radius of the previous attempt, force the
    # safe pick sequence to avoid re-driving into the same failed position.
    RETRY_XY_SAME_THRESHOLD_MM: float = 15.0
    RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY: bool = True
    # After a miss, re-localise with full RAFT disparity instead of reusing the
    # last stereo centroid, which may have drifted.
    RETRY_RELOCALIZE_WITH_RAFT_ONLY_AFTER_MISS: bool = True
    # "stop" or "skip": what to do when all retries are exhausted.
    ON_RETRY_FAIL: str = "stop"


DEFAULT_RECOVERY = RecoveryConfig()


def recovery_config_with_overrides(
    config: RecoveryConfig | None = None,
    **overrides,
) -> RecoveryConfig:
    base = DEFAULT_RECOVERY if config is None else config
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(base, **clean) if clean else base
