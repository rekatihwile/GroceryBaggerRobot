from __future__ import annotations

"""JSON IO helpers for named staging/place surface Z zones."""

import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SURFACE_ZONE_PATH = Path("config/surface_zones.json")


def _as_finite_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _validate_center_xy(center_xy_mm: Any) -> list[float]:
    if not isinstance(center_xy_mm, (list, tuple)) or len(center_xy_mm) != 2:
        raise ValueError("center_xy_mm must be a 2-element list")
    return [
        _as_finite_float(center_xy_mm[0], "center_xy_mm[0]"),
        _as_finite_float(center_xy_mm[1], "center_xy_mm[1]"),
    ]


def _validate_zone(zone: dict[str, Any], key_name: str | None = None) -> dict[str, Any]:
    if not isinstance(zone, dict):
        raise ValueError("surface zone must be a JSON object")

    name = str(zone.get("name", key_name or "")).strip()
    if not name:
        raise ValueError("surface zone name must be non-empty")

    if "surface_z_mm" not in zone:
        raise ValueError(f"surface zone {name!r} missing required field: surface_z_mm")

    out: dict[str, Any] = {
        "name": name,
        "surface_z_mm": _as_finite_float(zone["surface_z_mm"], "surface_z_mm"),
        "height_reference": str(zone.get("height_reference", "robot_command_frame")),
        "notes": str(zone.get("notes", "")),
    }

    if "center_xy_mm" in zone and zone.get("center_xy_mm") is not None:
        out["center_xy_mm"] = _validate_center_xy(zone["center_xy_mm"])
    if "width_mm" in zone and zone.get("width_mm") is not None:
        out["width_mm"] = _as_finite_float(zone["width_mm"], "width_mm")
    if "depth_mm" in zone and zone.get("depth_mm") is not None:
        out["depth_mm"] = _as_finite_float(zone["depth_mm"], "depth_mm")
    if "default_phi_deg" in zone and zone.get("default_phi_deg") is not None:
        out["default_phi_deg"] = _as_finite_float(zone["default_phi_deg"], "default_phi_deg")

    return out


def load_surface_zones(path: Path = DEFAULT_SURFACE_ZONE_PATH) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}

    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{p} must contain a JSON object of named zones")

    zones: dict[str, dict[str, Any]] = {}
    for key, zone in payload.items():
        valid = _validate_zone(zone, key_name=str(key))
        if valid["name"] != str(key):
            raise ValueError(f"zone key {key!r} does not match zone name {valid['name']!r}")
        zones[valid["name"]] = valid
    return zones


def save_surface_zones(zones: dict[str, dict[str, Any]], path: Path = DEFAULT_SURFACE_ZONE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    valid_zones: dict[str, dict[str, Any]] = {}
    for _name, zone in zones.items():
        valid = _validate_zone(zone)
        valid_zones[valid["name"]] = valid

    p.write_text(json.dumps(valid_zones, indent=2) + "\n", encoding="utf-8")


def get_surface_zone(name: str, path: Path = DEFAULT_SURFACE_ZONE_PATH) -> dict[str, Any]:
    zones = load_surface_zones(path)
    key = str(name)
    if key not in zones:
        raise KeyError(f"surface zone {key!r} not found in {Path(path)}")
    return zones[key]


def upsert_surface_zone(
    name: str,
    surface_z_mm: float,
    center_xy_mm=None,
    width_mm=None,
    depth_mm=None,
    default_phi_deg=None,
    notes: str = "",
    path: Path = DEFAULT_SURFACE_ZONE_PATH,
) -> dict[str, Any]:
    zone_name = str(name).strip()
    if not zone_name:
        raise ValueError("name must be non-empty")

    zones = load_surface_zones(path)
    existing = zones.get(zone_name, {})

    merged: dict[str, Any] = {
        "name": zone_name,
        "surface_z_mm": _as_finite_float(surface_z_mm, "surface_z_mm"),
        "height_reference": str(existing.get("height_reference", "robot_command_frame")),
        "notes": str(notes if notes else existing.get("notes", "")),
    }

    if center_xy_mm is not None:
        merged["center_xy_mm"] = _validate_center_xy(center_xy_mm)
    elif "center_xy_mm" in existing:
        merged["center_xy_mm"] = _validate_center_xy(existing["center_xy_mm"])

    if width_mm is not None:
        merged["width_mm"] = _as_finite_float(width_mm, "width_mm")
    elif "width_mm" in existing:
        merged["width_mm"] = _as_finite_float(existing["width_mm"], "width_mm")

    if depth_mm is not None:
        merged["depth_mm"] = _as_finite_float(depth_mm, "depth_mm")
    elif "depth_mm" in existing:
        merged["depth_mm"] = _as_finite_float(existing["depth_mm"], "depth_mm")

    if default_phi_deg is not None:
        merged["default_phi_deg"] = _as_finite_float(default_phi_deg, "default_phi_deg")
    elif "default_phi_deg" in existing:
        merged["default_phi_deg"] = _as_finite_float(existing["default_phi_deg"], "default_phi_deg")

    valid = _validate_zone(merged)
    zones[zone_name] = valid
    save_surface_zones(zones, path)
    return valid
