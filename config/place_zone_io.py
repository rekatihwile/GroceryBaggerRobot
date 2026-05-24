from __future__ import annotations

"""JSON IO helpers for named placement repeatability zones."""

import json
from pathlib import Path
from typing import Any


DEFAULT_PLACE_ZONE_PATH = Path("config/place_zones.json")
REQUIRED_FIELDS = {
    "name",
    "center_xy_mm",
    "phi_deg",
    "width_mm",
    "depth_mm",
    "notes",
}


def _as_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    return out


def validate_place_zone(zone: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(zone.keys()))
    if missing:
        raise ValueError(f"place zone missing required field(s): {missing}")
    if "floor_z_mm" not in zone and "place_z_mm" not in zone:
        raise ValueError("place zone missing required field: floor_z_mm (or legacy place_z_mm)")

    name = str(zone["name"]).strip()
    if not name:
        raise ValueError("place zone name must be non-empty")

    center = zone["center_xy_mm"]
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("center_xy_mm must be a 2-element list")

    width = _as_float(zone["width_mm"], "width_mm")
    depth = _as_float(zone["depth_mm"], "depth_mm")
    if width <= 0.0:
        raise ValueError("width_mm must be > 0")
    if depth <= 0.0:
        raise ValueError("depth_mm must be > 0")

    if "floor_z_mm" in zone:
        floor_z_mm = _as_float(zone["floor_z_mm"], "floor_z_mm")
        legacy_place_z_mm = _as_float(zone.get("place_z_mm", floor_z_mm), "place_z_mm")
    else:
        floor_z_mm = _as_float(zone["place_z_mm"], "place_z_mm")
        legacy_place_z_mm = floor_z_mm
        print(
            f"[PLACE ZONE WARN] zone {name!r} uses legacy place_z_mm only; "
            "treating it as floor_z_mm."
        )

    return {
        "name": name,
        "center_xy_mm": [
            _as_float(center[0], "center_xy_mm[0]"),
            _as_float(center[1], "center_xy_mm[1]"),
        ],
        "floor_z_mm": floor_z_mm,
        "place_z_mm": legacy_place_z_mm,
        "phi_deg": _as_float(zone["phi_deg"], "phi_deg"),
        "width_mm": width,
        "depth_mm": depth,
        "notes": str(zone.get("notes", "")),
    }


def load_place_zones(path: Path = DEFAULT_PLACE_ZONE_PATH) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}

    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{p} must contain a JSON object of zones")

    zones: dict[str, dict[str, Any]] = {}
    for name, zone in payload.items():
        if not isinstance(zone, dict):
            raise ValueError(f"zone {name!r} must be a JSON object")
        valid = validate_place_zone(zone)
        if valid["name"] != str(name):
            raise ValueError(f"zone key {name!r} does not match zone name {valid['name']!r}")
        zones[valid["name"]] = valid
    return zones


def save_place_zones(
    zones: dict[str, dict[str, Any]],
    path: Path = DEFAULT_PLACE_ZONE_PATH,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    valid_zones = {}
    for _name, zone in zones.items():
        valid = validate_place_zone(zone)
        valid_zones[valid["name"]] = valid
    p.write_text(json.dumps(valid_zones, indent=2) + "\n", encoding="utf-8")


def get_place_zone(
    name: str = "default_test_zone",
    path: Path = DEFAULT_PLACE_ZONE_PATH,
) -> dict[str, Any]:
    zones = load_place_zones(path)
    key = str(name)
    if key not in zones:
        raise KeyError(f"place zone {key!r} not found in {Path(path)}")
    return zones[key]


def upsert_place_zone(
    name: str,
    center_xy_mm,
    place_z_mm: float,
    phi_deg: float,
    width_mm: float,
    depth_mm: float,
    notes: str = "",
    path: Path = DEFAULT_PLACE_ZONE_PATH,
) -> dict[str, Any]:
    zone = validate_place_zone(
        {
            "name": str(name),
            "center_xy_mm": list(center_xy_mm),
            "floor_z_mm": place_z_mm,
            "place_z_mm": place_z_mm,
            "phi_deg": phi_deg,
            "width_mm": width_mm,
            "depth_mm": depth_mm,
            "notes": notes,
        }
    )
    zones = load_place_zones(path)
    zones[zone["name"]] = zone
    save_place_zones(zones, path)
    return zone
