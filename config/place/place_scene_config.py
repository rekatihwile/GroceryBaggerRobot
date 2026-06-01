from __future__ import annotations

"""Canonical place-scene loader shared by wet runs, demos, and dry validators."""

from pathlib import Path
from typing import Any

from config.runtime_context import resolve_place_scene_name
from config.surface_zone_io import get_surface_zone

from .place_config import DEFAULT_PLACE, PlaceConfig


def load_place_scene(
    config: PlaceConfig = DEFAULT_PLACE,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    scene_name = resolve_place_scene_name(config.PLACE_SCENE_NAME)
    scene_path = Path(config.PLACE_SCENE_CONFIG_PATH)
    zone = get_surface_zone(scene_name, scene_path)
    out = {
        "name": str(zone["name"]),
        "center_xy_mm": list(zone.get("center_xy_mm", [450.0, 250.0])),
        "surface_z_mm": float(zone["surface_z_mm"]),
        "default_phi_deg": float(zone.get("default_phi_deg", 0.0)),
        "width_mm": float(zone.get("width_mm", 120.0)),
        "depth_mm": float(zone.get("depth_mm", 120.0)),
        "notes": str(zone.get("notes", "")),
        "source": "surface_zones",
        "config_floor_z_mm": float(zone["surface_z_mm"]),
        "config_place_z_mm": float(zone["surface_z_mm"]),
    }

    if verbose:
        print("[ZONE] loaded destination surface zone:")
        print(f"  source      = {out['source']}")
        print(f"  name        = {out['name']}")
        print(f"  center_xy   = ({out['center_xy_mm'][0]:.1f}, {out['center_xy_mm'][1]:.1f}) mm")
        print(f"  surface_z   = {out['surface_z_mm']:.1f} mm")
        print(f"  size        = {out['width_mm']:.1f} x {out['depth_mm']:.1f} mm")
        print(f"  default_phi = {out['default_phi_deg']:.1f} deg")
        if out.get("notes"):
            print(f"  notes       = {out['notes']}")

    return out
