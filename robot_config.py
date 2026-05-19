from __future__ import annotations

from pathlib import Path

from robot import JointPose, RobotConfig


HOME_Z_MM = 100.0
LOW_Z_MM = 0.0
DEFAULT_TRAVEL_Z_MM = HOME_Z_MM

STEPS_PER_MM_J3 = 50.92958
L1_MM = 450.0
L2_MM = 450.0

SOFT_LIMITS_CONFIG_PATH = Path("soft_limits_config.json")

HOME_POSE = JointPose(

)

ROBOT_CONFIG = RobotConfig(
    
)


def print_startup_config(script_name: str, overhead_index: int, stereo_index: int) -> None:
    print(f"[CONFIG] {script_name}")
    print(f"[CONFIG] home_pose.z_mm={ROBOT_CONFIG.home_pose.z_mm}")
    print(f"[CONFIG] steps_per_mm_j3={ROBOT_CONFIG.steps_per_mm_j3}")
    print(f"[CONFIG] soft_limits_enabled={ROBOT_CONFIG.soft_limits_enabled}")
    print(f"[CONFIG] soft_limits_path={ROBOT_CONFIG.soft_limits_path}")
    print(f"[CONFIG] overhead index={overhead_index}")
    print(f"[CONFIG] stereo index={stereo_index}")


def require_soft_limits_configured(script_name: str) -> None:
    if not getattr(ROBOT_CONFIG, "soft_limits_enabled", False):
        raise RuntimeError(f"{script_name} refuses to run because ROBOT_CONFIG.soft_limits_enabled is False.")


def require_robot_soft_limits_loaded(robot, script_name: str) -> None:
    if not getattr(robot.cfg, "soft_limits_enabled", False):
        raise RuntimeError(f"{script_name} refuses to run because robot soft limits are disabled or failed to load.")
