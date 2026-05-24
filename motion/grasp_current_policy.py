from __future__ import annotations

"""Placeholder policy module for future current-controlled grasp handling."""

from dataclasses import dataclass


@dataclass
class GraspCurrentSettings:
    enabled: bool = False
    close_timeout_s: float = 1.0
    target_current_ma: float | None = None
    hold_current_ma: float | None = None
    min_position_delta_deg: float = 0.0
    notes: str = "TODO: wire to robot current feedback API when available."


# TODO: Implement current-based claw close policy once the Robot API exposes
# real-time motor/servo current and a safe close-until-threshold primitive.
# Keep this module import-safe and side-effect free for smoke tests.
