from __future__ import annotations

"""Lightweight container for place sequence parameters.

This module intentionally avoids behavioral changes. Existing scripts can adopt
it incrementally when place motion extraction is mechanically validated.
"""

from dataclasses import dataclass


@dataclass
class PlaceSequencePlan:
    x_mm: float
    y_mm: float
    phi_deg: float
    approach_z_mm: float
    release_z_mm: float
    retract_z_mm: float
