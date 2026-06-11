from __future__ import annotations

"""planning/planner_2d_blb.py

Bottom-Left-Back (BLB) 2D bag packing planner.

Rules
-----
- 2D only: the bag plane is x-y (width × depth).  No z-stacking yet.
- BLB candidate spots come from the bag origin plus the right and top edges of
  every already-placed item.
- Geometry hard filter: item padded rect must fit inside bag without overlap.
- Crush hard filter: reject placement if the total weight_score of remaining
  items above would exceed the candidate's fragility_score.
  (Since we have no z-layers yet, "above" is approximated as heavier items
   placed before fragile items.)
- Scoring: lower squish_score is better; heavier / larger-footprint items are
  preferred earlier; fragile items are deferred.

This is intentionally conservative — 3D extreme points, multi-layer stacking,
and real load simulation are future work.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planning.grocery_item import GroceryItem
    from planning.bag_state import BagState


# ------------------------------------------------------------------ #
# Candidate spot
# ------------------------------------------------------------------ #

@dataclass
class PlacementSpot:
    x: float   # bottom-left x (cm)
    y: float   # bottom-left y (cm)


# ------------------------------------------------------------------ #
# Spot generation
# ------------------------------------------------------------------ #

def generate_blb_candidate_spots(bag_state: "BagState") -> list[PlacementSpot]:
    """Return candidate placement corners using BLB logic.

    Candidates come from:
    - (0, 0) — bag origin
    - For every placed item: its right edge x, its bottom y (new column)
    - For every placed item: its left x, its top edge y (new row)

    Duplicates are deduplicated to a 1 mm grid.
    """
    spots: list[tuple[float, float]] = [(0.0, 0.0)]
    for px, py, pw, pd in bag_state.occupied_rects():
        spots.append((px + pw, py))       # right edge of item, same row
        spots.append((px, py + pd))       # same column, above item

    # Deduplicate to 1 cm grid for speed
    seen: set[tuple[int, int]] = set()
    unique: list[PlacementSpot] = []
    for x, y in spots:
        key = (round(x * 10), round(y * 10))
        if key not in seen:
            seen.add(key)
            unique.append(PlacementSpot(x=round(x, 3), y=round(y, 3)))

    # Sort BLB order: lowest y first, then leftmost x
    unique.sort(key=lambda s: (s.y, s.x))
    return unique


# ------------------------------------------------------------------ #
# Placement validation
# ------------------------------------------------------------------ #

def choose_placement_spot_2d(
    item: "GroceryItem",
    bag_state: "BagState",
) -> PlacementSpot | None:
    """Return the best BLB spot for item, or None if no valid spot exists."""
    spots = generate_blb_candidate_spots(bag_state)
    for spot in spots:
        if bag_state.can_fit_2d(item, spot.x, spot.y):
            return spot
    return None


# ------------------------------------------------------------------ #
# Crush / squish scoring
# ------------------------------------------------------------------ #

def can_stack(candidate: "GroceryItem", remaining_items: list["GroceryItem"]) -> bool:
    """Return True if candidate can bear the estimated load from remaining items.

    'Remaining items' are those not yet placed that would conceptually sit above
    this item in the bag.  We approximate total load as sum of weight_scores of
    heavier remaining items.

    A fragile item (high fragility_score) should not have heavy items above it.
    Reject if total_weight_above > candidate.fragility_score * max_weight_scale.
    """
    _MAX_WEIGHT_SCALE = 5.0  # max combined weight_score that can sit on top
    total_above = sum(
        r.weight_score for r in remaining_items
        if r.weight_score >= candidate.weight_score
    )
    threshold = candidate.fragility_score * _MAX_WEIGHT_SCALE
    return total_above <= threshold


def squish_score(candidate: "GroceryItem", remaining_items: list["GroceryItem"]) -> float:
    """Squish risk score: lower is better.

    Heavier items sitting on fragile candidates = high squish score.
    Fragile items have high squish risk if placed early/bottom.
    """
    heavy_above = sum(
        r.weight_score for r in remaining_items
        if r.weight_score > candidate.weight_score
    )
    return float(candidate.fragility_score * (1.0 + heavy_above))


# ------------------------------------------------------------------ #
# Candidate scoring
# ------------------------------------------------------------------ #

def _placement_score(
    item: "GroceryItem",
    spot: PlacementSpot,
    bag_state: "BagState",
    remaining_items: list["GroceryItem"],
) -> float:
    """Higher score = better placement choice.

    Components (all normalised to [0,1] range):
    - weight_score:     heavier items earlier
    - footprint_frac:   larger footprint items earlier (packs bottom efficiently)
    - inv_squish:       lower squish risk is better
    - blb_position:     lower y + lower x is better (BLB preference)
    """
    bag_area = bag_state.bag_width_cm * bag_state.bag_depth_cm
    footprint = item.padded_rect_xy_cm[0] * item.padded_rect_xy_cm[1]
    footprint_frac = footprint / max(bag_area, 1.0)

    sq = squish_score(item, remaining_items)
    inv_squish = 1.0 / (1.0 + sq)

    # BLB position: normalise to [0, 1]; lower is better → invert
    blb_raw = (spot.y / max(bag_state.bag_depth_cm, 1.0)) + 0.5 * (spot.x / max(bag_state.bag_width_cm, 1.0))
    inv_blb = 1.0 - min(blb_raw, 1.0)

    return (
        0.35 * float(item.weight_score)
        + 0.25 * footprint_frac
        + 0.25 * inv_squish
        + 0.15 * inv_blb
    )


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

@dataclass
class PickResult:
    item: "GroceryItem"
    spot: PlacementSpot
    score: float


def best_pick(
    candidates: list["GroceryItem"],
    bag_state: "BagState",
) -> PickResult | None:
    """Choose the best item to place next and its BLB spot.

    Returns the (item, spot, score) triple with the highest placement score,
    or None if no item can fit.

    Hard filters applied:
    1. Item padded rect must fit in bag without collision.
    2. Crush filter: can_stack() must return True.
    """
    if not candidates:
        return None

    results: list[PickResult] = []

    for item in candidates:
        spot = choose_placement_spot_2d(item, bag_state)
        if spot is None:
            continue

        remaining = [c for c in candidates if c is not item]
        if not can_stack(item, remaining):
            continue

        score = _placement_score(item, spot, bag_state, remaining)
        results.append(PickResult(item=item, spot=spot, score=score))

    if not results:
        return None

    return max(results, key=lambda r: r.score)
