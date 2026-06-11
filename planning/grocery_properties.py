from __future__ import annotations

"""Per-class grocery packing properties from config/grocery_spec.json."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

_GROCERY_SPEC_PATH = Path("config/grocery_spec.json")
_WEIGHT_MAX_KG = 2.0  # normalization ceiling for foundation strength


@dataclass(frozen=True)
class GroceryProperties:
    weight: float      # kg (approximate)
    fragility: float   # 0–1 (1 = easily crushed)
    compliance: float  # 0–1 (1 = very squishable)
    pin_slot: bool     # True = prefer a corner/wall slot
    source: str        # "spec" or "default"


_DEFAULT_PROPS = GroceryProperties(
    weight=0.10,
    fragility=0.30,
    compliance=0.30,
    pin_slot=False,
    source="default",
)


def load_grocery_specs(path: Path = _GROCERY_SPEC_PATH) -> dict[str, GroceryProperties]:
    """Load config/grocery_spec.json; return {} on any failure (warns instead of crashing)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        warnings.warn(f"[GROCERY] spec not found: {path}; all classes will use defaults")
        return {}
    except Exception as exc:
        warnings.warn(f"[GROCERY] spec load failed ({exc}); using defaults")
        return {}

    specs: dict[str, GroceryProperties] = {}
    for class_name, entry in raw.items():
        if str(class_name).startswith("_"):
            continue
        try:
            specs[str(class_name)] = GroceryProperties(
                weight=float(entry.get("weight", _DEFAULT_PROPS.weight)),
                fragility=float(entry.get("fragility", _DEFAULT_PROPS.fragility)),
                compliance=float(entry.get("compliance", _DEFAULT_PROPS.compliance)),
                pin_slot=bool(entry.get("pin_slot", _DEFAULT_PROPS.pin_slot)),
                source="spec",
            )
        except Exception as exc:
            warnings.warn(f"[GROCERY] skipping {class_name!r}: {exc}")
    return specs


def properties_for_class(
    class_name: str,
    specs: dict[str, GroceryProperties],
    default: GroceryProperties | None = None,
) -> GroceryProperties:
    """Return the GroceryProperties for class_name, falling back to default."""
    if default is None:
        default = _DEFAULT_PROPS
    hit = specs.get(str(class_name))
    if hit is not None:
        return hit
    return GroceryProperties(
        weight=default.weight,
        fragility=default.fragility,
        compliance=default.compliance,
        pin_slot=default.pin_slot,
        source="default",
    )


def foundation_strength_score(props: GroceryProperties) -> float:
    """Scalar strength score: higher = better foundation candidate.

    Range is roughly 0–4.25 (heavier, less fragile, less compliant → higher).
    """
    normalized_weight = min(1.0, float(props.weight) / _WEIGHT_MAX_KG)
    return float(
        2.0 * normalized_weight
        + 1.5 * (1.0 - float(props.fragility))
        + 0.75 * (1.0 - float(props.compliance))
    )
