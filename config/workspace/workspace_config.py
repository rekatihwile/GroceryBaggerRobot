from __future__ import annotations

"""Shared workspace and candidate-filter profiles."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class WorkspaceFilterConfig:
    name: str
    platform_grid_x_mm: tuple[float, ...] = (40.0, 140.0, 240.0, 340.0)
    platform_grid_y_mm: tuple[float, ...] = (40.0, 190.0, 340.0, 490.0)
    platform_grid_z_mm: tuple[float, ...] = (0.0, 50.0, 150.0, 200.0)

    require_positive_platform_xy: bool = True
    enable_workspace_bounds: bool = True
    enable_robotframe_centroid_xy_platform_bounds: bool = True

    center_gate_enabled: bool = False
    max_image_center_norm_radius: float = 0.85
    cluster_gate_enabled: bool = False
    max_cluster_distance_mm: float = 600.0

    reject_placed_overlap: bool = True
    placed_overlap_margin_mm: float = 25.0
    min_volume_mm3: float = 1.0
    max_volume_cm3: float = 3000.0


DEFAULT_WET_RUN_WORKSPACE = WorkspaceFilterConfig(
    name="wet_run",
)

DEFAULT_SAVED_PHOTO_TEST_WORKSPACE = WorkspaceFilterConfig(
    name="saved_photo_test",
    require_positive_platform_xy=False,
    enable_workspace_bounds=False,
    enable_robotframe_centroid_xy_platform_bounds=False,
)

KNOWN_WORKSPACE_PROFILES: dict[str, WorkspaceFilterConfig] = {
    DEFAULT_WET_RUN_WORKSPACE.name: DEFAULT_WET_RUN_WORKSPACE,
    DEFAULT_SAVED_PHOTO_TEST_WORKSPACE.name: DEFAULT_SAVED_PHOTO_TEST_WORKSPACE,
    "demo": replace(DEFAULT_SAVED_PHOTO_TEST_WORKSPACE, name="demo"),
}


def get_workspace_filter_config(name: str = "wet_run") -> WorkspaceFilterConfig:
    key = str(name).strip().lower()
    if key not in KNOWN_WORKSPACE_PROFILES:
        known = ", ".join(sorted(KNOWN_WORKSPACE_PROFILES))
        raise ValueError(f"unknown workspace profile {name!r}; expected one of {known}")
    return KNOWN_WORKSPACE_PROFILES[key]


def workspace_bounds_mm(config: WorkspaceFilterConfig) -> tuple[float, float, float, float]:
    xs = [float(v) for v in config.platform_grid_x_mm]
    ys = [float(v) for v in config.platform_grid_y_mm]
    return min(xs), max(xs), min(ys), max(ys)
