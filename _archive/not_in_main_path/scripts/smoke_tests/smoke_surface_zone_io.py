"""Smoke test for config.surface_zone_io helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.surface_zone_io import get_surface_zone, load_surface_zones, upsert_surface_zone


def main() -> int:
    tmp_path = Path("scripts/smoke_tests/_tmp_surface_zones.json")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path.write_text("{}\n", encoding="utf-8")

    zone = upsert_surface_zone(
        name="smoke_zone",
        surface_z_mm=42.0,
        center_xy_mm=[100.0, 200.0],
        width_mm=120.0,
        depth_mm=90.0,
        default_phi_deg=5.0,
        notes="smoke",
        path=tmp_path,
    )
    assert zone["name"] == "smoke_zone"
    assert abs(float(zone["surface_z_mm"]) - 42.0) < 1e-9

    loaded = load_surface_zones(tmp_path)
    assert "smoke_zone" in loaded

    got = get_surface_zone("smoke_zone", tmp_path)
    assert abs(float(got["surface_z_mm"]) - 42.0) < 1e-9

    payload = json.loads(tmp_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    print("[PASS] smoke_surface_zone_io")
    tmp_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
