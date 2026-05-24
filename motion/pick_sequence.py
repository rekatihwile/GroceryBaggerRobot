from __future__ import annotations

"""Lightweight container for pick sequence parameters.

This module intentionally avoids behavioral changes. Existing scripts can adopt
it incrementally when mechanical extraction of pick motion is safe.
"""

from dataclasses import dataclass


@dataclass
class PickSequencePlan:
    x_mm: float
    y_mm: float
    phi_deg: float
    travel_z_mm: float
    grasp_z_mm: float
