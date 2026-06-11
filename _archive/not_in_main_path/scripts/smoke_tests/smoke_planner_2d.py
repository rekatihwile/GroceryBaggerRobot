"""
smoke_planner_2d.py

Smoke test for the 2D BLB bag planner.

No cameras, no YOLO, no RAFT, no robot hardware required.
Creates fake grocery items, packs them into a BagState, and prints the
selected order and placement spots.  Exits nonzero if the planner fails to
place at least one item.

Run:
    python scripts/smoke_tests/smoke_planner_2d.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

BAG_WIDTH_CM  = 30.0
BAG_DEPTH_CM  = 20.0
PRINT_EACH_STEP = True

# ============================================================
# End of user settings
# ============================================================

import sys
from pathlib import Path

# Allow running from repo root or from this file's directory
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.grocery_item import GroceryItem
from planning.bag_state import BagState
from planning.planner_2d_blb import best_pick


def make_item(
    name: str,
    w_cm: float,
    d_cm: float,
    h_cm: float,
    weight: float,
    fragility: float,
) -> GroceryItem:
    area = w_cm * d_cm
    return GroceryItem(
        id=name,
        class_name=name,
        confidence=1.0,
        grasp_xyz_robot_mm=None,
        grasp_xyz_overhead=None,
        grasp_xyz_stereo_cam_mm=None,
        cross_sectional_area_cm2=area,
        height_cm=h_cm,
        weight_score=weight,
        fragility_score=fragility,
        padded_rect_xy_cm=(w_cm, d_cm),
        padded_box_xyz_cm=(w_cm, d_cm, h_cm),
    )


def main() -> int:
    print("=" * 60)
    print("smoke_planner_2d.py")
    print(f"  Bag: {BAG_WIDTH_CM:.1f} x {BAG_DEPTH_CM:.1f} cm")
    print("=" * 60)

    # --- fake grocery items ---
    all_items: list[GroceryItem] = [
        make_item("milk",      w_cm=7.0,  d_cm=7.0,  h_cm=25.0, weight=0.9, fragility=0.1),
        make_item("bread",     w_cm=14.0, d_cm=10.0, h_cm=15.0, weight=0.2, fragility=0.7),
        make_item("eggs",      w_cm=12.0, d_cm=8.0,  h_cm=10.0, weight=0.5, fragility=0.95),
        make_item("cereal",    w_cm=9.0,  d_cm=5.0,  h_cm=30.0, weight=0.3, fragility=0.4),
        make_item("juice",     w_cm=8.0,  d_cm=8.0,  h_cm=18.0, weight=0.8, fragility=0.15),
        make_item("chips",     w_cm=13.0, d_cm=7.0,  h_cm=35.0, weight=0.1, fragility=0.3),
        make_item("yogurt",    w_cm=6.0,  d_cm=6.0,  h_cm=8.0,  weight=0.4, fragility=0.5),
    ]

    bag = BagState(bag_width_cm=BAG_WIDTH_CM, bag_depth_cm=BAG_DEPTH_CM)
    remaining = list(all_items)
    step = 0
    placed_names: list[str] = []

    while remaining:
        result = best_pick(remaining, bag)
        if result is None:
            print(f"\n  No more items fit.  {len(remaining)} item(s) left unplaced.")
            break

        placed = bag.add(result.item, result.spot.x, result.spot.y)
        remaining.remove(result.item)
        step += 1
        placed_names.append(result.item.class_name)

        if PRINT_EACH_STEP:
            print(
                f"  Step {step:2d}: {result.item.class_name:<10}  "
                f"spot=({result.spot.x:.1f},{result.spot.y:.1f})  "
                f"size=({result.item.padded_rect_xy_cm[0]:.1f}x{result.item.padded_rect_xy_cm[1]:.1f})  "
                f"score={result.score:.3f}  "
                f"w={result.item.weight_score:.2f}  frag={result.item.fragility_score:.2f}"
            )

    print()
    print(f"Placed {step}/{len(all_items)} items in order: {placed_names}")
    print(f"Bag free area after packing: {bag.free_area_estimate_cm2():.1f} cm²")
    print(bag)

    # --- validation ---
    if step == 0:
        print("\n[FAIL] Planner placed ZERO items — check bag dimensions and item sizes.")
        return 1

    # Check no collision in placed rects
    rects = bag.occupied_rects()
    from planning.bag_state import _rects_overlap
    collision_found = False
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            ax, ay, aw, ad = rects[i]
            bx, by, bw, bd = rects[j]
            if _rects_overlap(ax, ay, aw, ad, bx, by, bw, bd, margin=-1e-4):
                print(f"\n[FAIL] Collision between placed item {i} and {j}!")
                collision_found = True

    if collision_found:
        return 1

    # Check no item exceeds bag bounds
    for i, (px, py, pw, pd) in enumerate(rects):
        if px < -1e-4 or py < -1e-4 or px + pw > BAG_WIDTH_CM + 1e-4 or py + pd > BAG_DEPTH_CM + 1e-4:
            print(f"\n[FAIL] Placed item {i} is out of bag bounds!")
            return 1

    print("\n[PASS] smoke_planner_2d: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
