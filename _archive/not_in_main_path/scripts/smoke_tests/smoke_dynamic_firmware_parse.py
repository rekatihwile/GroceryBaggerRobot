from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.robot import parse_dynamic_state_lines


def main() -> int:
    lines = [
        "# some other line",
        "# z_empirical = 142.75 mm",
        "# servo_empirical= 33.5 deg",
    ]
    result = parse_dynamic_state_lines(lines)
    assert result.ok, result
    assert abs(float(result.z_empirical_mm) - 142.75) < 1e-6
    assert abs(float(result.servo_empirical_deg) - 33.5) < 1e-6
    print("[PASS] parsed SHOW output into z_empirical_mm and servo_empirical_deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
