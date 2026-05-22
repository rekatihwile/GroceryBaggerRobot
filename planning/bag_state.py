from __future__ import annotations

"""planning/bag_state.py

2D bag geometry model for the BLB planner.

All coordinates are in centimetres, measured from the bag origin (0, 0) =
bottom-left corner of the bag plane.  Width runs along +x, depth along +y.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from planning.grocery_item import GroceryItem


@dataclass
class PlacedItem:
    item: "GroceryItem"
    x: float   # left edge of item rectangle in bag (cm)
    y: float   # bottom edge of item rectangle in bag (cm)
    z: float = 0.0  # base layer Z (future use)
    # Resolved footprint at placement time (cm)
    w: float = 0.0  # padded width
    d: float = 0.0  # padded depth
    h: float = 0.0  # padded height


class BagState:
    """Tracks 2D placement of items inside a rectangular bag.

    Collision uses axis-aligned bounding rectangles in the bag x-y plane.
    No 3D stacking or extreme-point geometry yet — that is future work.
    """

    def __init__(self, bag_width_cm: float = 30.0, bag_depth_cm: float = 20.0) -> None:
        if bag_width_cm <= 0 or bag_depth_cm <= 0:
            raise ValueError("Bag dimensions must be positive.")
        self.bag_width_cm = float(bag_width_cm)
        self.bag_depth_cm = float(bag_depth_cm)
        self.placed_items: list[PlacedItem] = []

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def occupied_rects(self) -> list[tuple[float, float, float, float]]:
        """Return list of (x, y, w, d) rectangles for all placed items."""
        return [(p.x, p.y, p.w, p.d) for p in self.placed_items]

    def can_fit_2d(self, item: "GroceryItem", x: float, y: float) -> bool:
        """Return True if item can be placed with its bottom-left at (x, y)
        without leaving the bag and without overlapping any placed item.

        A zero-size padded rect always fails.
        """
        w, d = float(item.padded_rect_xy_cm[0]), float(item.padded_rect_xy_cm[1])
        if w <= 0 or d <= 0:
            return False

        # Must fit within bag bounds (allow small epsilon for floating point)
        eps = 1e-6
        if x < -eps or y < -eps:
            return False
        if x + w > self.bag_width_cm + eps:
            return False
        if y + d > self.bag_depth_cm + eps:
            return False

        # Must not overlap any already-placed item
        for px, py, pw, pd in self.occupied_rects():
            if _rects_overlap(x, y, w, d, px, py, pw, pd):
                return False

        return True

    def free_area_estimate_cm2(self) -> float:
        """Rough free area = bag area minus sum of placed footprints.

        Does not account for fragmentation; use only for loose estimates.
        """
        used = sum(p.w * p.d for p in self.placed_items)
        total = self.bag_width_cm * self.bag_depth_cm
        return max(0.0, total - used)

    def stack_height_at_mm(self, x_cm: float, y_cm: float) -> float:
        """Return current stack height in mm at a bag-plane point.

        The current BLB planner is 2D and rejects overlapping placements, so
        new valid spots normally see a stack height of 0.  This helper keeps
        place-Z calculation compatible with future stacked placements.
        """
        stack_top_cm = 0.0
        for placed in self.placed_items:
            inside_x = placed.x <= x_cm < placed.x + placed.w
            inside_y = placed.y <= y_cm < placed.y + placed.d
            if inside_x and inside_y:
                stack_top_cm = max(stack_top_cm, placed.z + placed.h)
        return stack_top_cm * 10.0

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add(self, item: "GroceryItem", x: float, y: float) -> PlacedItem:
        """Place item with its bottom-left corner at (x, y)."""
        w = float(item.padded_rect_xy_cm[0])
        d = float(item.padded_rect_xy_cm[1])
        h = float(item.padded_box_xyz_cm[2]) if len(item.padded_box_xyz_cm) >= 3 else float(item.height_cm)
        placed = PlacedItem(item=item, x=float(x), y=float(y), w=w, d=d, h=h)
        self.placed_items.append(placed)
        return placed

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"BagState({self.bag_width_cm:.1f}x{self.bag_depth_cm:.1f} cm, "
            f"{len(self.placed_items)} placed, "
            f"free~{self.free_area_estimate_cm2():.1f} cm^2)"
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _rects_overlap(
    ax: float, ay: float, aw: float, ad: float,
    bx: float, by: float, bw: float, bd: float,
    margin: float = 0.0,
) -> bool:
    """Return True if two axis-aligned rectangles overlap (or touch within margin)."""
    return (
        ax < bx + bw + margin
        and ax + aw + margin > bx
        and ay < by + bd + margin
        and ay + ad + margin > by
    )
