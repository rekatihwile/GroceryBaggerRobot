from __future__ import annotations

"""
No-hardware smoke test for config.place_zone_io.

Run:
    python scripts/smoke_tests/smoke_place_zone_io.py
"""

import sys
import tempfile
from pathlib import Path
import json

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.place_zone_io import load_place_zones, upsert_place_zone


def _check_close(label: str, got: float, expected: float) -> None:
    if abs(float(got) - float(expected)) > 1e-6:
        raise AssertionError(f"{label}: got {got!r}, expected {expected!r}")


def main() -> int:
    print("=" * 60)
    print("smoke_place_zone_io.py")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "place_zones.json"
        zone = upsert_place_zone(
            "test_zone",
            center_xy_mm=[123.0, 456.0],
            place_z_mm=78.0,
            phi_deg=12.5,
            width_mm=111.0,
            depth_mm=222.0,
            notes="smoke",
            path=path,
        )
        print(f"[PASS] upserted zone: {zone['name']}")

        zones = load_place_zones(path)
        loaded = zones["test_zone"]
        _check_close("center x", loaded["center_xy_mm"][0], 123.0)
        _check_close("center y", loaded["center_xy_mm"][1], 456.0)
        _check_close("floor_z_mm", loaded["floor_z_mm"], 78.0)
        _check_close("place_z_mm", loaded["place_z_mm"], 78.0)
        _check_close("phi_deg", loaded["phi_deg"], 12.5)
        _check_close("width_mm", loaded["width_mm"], 111.0)
        _check_close("depth_mm", loaded["depth_mm"], 222.0)
        print("[PASS] reloaded zone values match.")

        legacy_path = Path(td) / "legacy_place_zones.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "legacy_zone": {
                        "name": "legacy_zone",
                        "center_xy_mm": [1.0, 2.0],
                        "place_z_mm": 3.0,
                        "phi_deg": 4.0,
                        "width_mm": 5.0,
                        "depth_mm": 6.0,
                        "notes": "legacy",
                    }
                }
            ),
            encoding="utf-8",
        )
        legacy = load_place_zones(legacy_path)["legacy_zone"]
        _check_close("legacy floor_z_mm", legacy["floor_z_mm"], 3.0)
        _check_close("legacy place_z_mm", legacy["place_z_mm"], 3.0)
        print("[PASS] legacy place_z_mm is treated as floor_z_mm.")

    print("\n[PASS] smoke_place_zone_io: all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
