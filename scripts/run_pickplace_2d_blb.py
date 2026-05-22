"""
run_pickplace_2d_blb.py

2D BLB bag-packing pick/place runner.

In offline mode (USE_FAKE_CANDIDATES=True, LIVE_CAMERA=False, LIVE_ROBOT=False)
this script simulates the full pack loop with fake grocery items — no cameras,
no YOLO/RAFT, no robot hardware required.

Set LIVE_CAMERA=True to use real YOLO survey candidates.
Set LIVE_ROBOT=True to execute real robot pick/place motion.

No argparse.  Edit the USER SETTINGS block below.

Run (offline demo):
    python scripts/run_pickplace_2d_blb.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

LIVE_CAMERA        = False   # True  → use real YOLO/RAFT survey
LIVE_ROBOT         = False   # True  → execute real robot motion
BAG_WIDTH_CM       = 30.0
BAG_DEPTH_CM       = 20.0
USE_FAKE_CANDIDATES = True   # False → survey from real camera (requires LIVE_CAMERA=True)
MAX_PICKPLACE_STEPS = 10
ENABLE_ROBOT_MOTION = False  # Extra guard: set True to allow robot moves
ENABLE_CLAW         = False  # Extra guard: set True to allow claw open/close

# ============================================================
# End of user settings
# ============================================================

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planning.grocery_item import GroceryItem
from planning.bag_state import BagState
from planning.planner_2d_blb import best_pick, PickResult


# ------------------------------------------------------------------ #
# Fake candidate factory (offline demo mode)
# ------------------------------------------------------------------ #

def _make_fake_grocery_item(
    name: str,
    w_cm: float,
    d_cm: float,
    h_cm: float,
    weight: float,
    fragility: float,
) -> GroceryItem:
    return GroceryItem(
        id=name,
        class_name=name,
        confidence=1.0,
        grasp_xyz_robot_mm=None,
        grasp_xyz_overhead=None,
        grasp_xyz_stereo_cam_mm=None,
        cross_sectional_area_cm2=w_cm * d_cm,
        height_cm=h_cm,
        weight_score=weight,
        fragility_score=fragility,
        padded_rect_xy_cm=(w_cm, d_cm),
        padded_box_xyz_cm=(w_cm, d_cm, h_cm),
    )


def _get_fake_candidates() -> list[GroceryItem]:
    return [
        _make_fake_grocery_item("milk",   7.0, 7.0, 25.0, 0.9, 0.1),
        _make_fake_grocery_item("bread", 14.0, 10.0, 15.0, 0.2, 0.7),
        _make_fake_grocery_item("eggs",  12.0,  8.0, 10.0, 0.5, 0.95),
        _make_fake_grocery_item("cereal", 9.0,  5.0, 30.0, 0.3, 0.4),
        _make_fake_grocery_item("juice",  8.0,  8.0, 18.0, 0.8, 0.15),
    ]


# ------------------------------------------------------------------ #
# Live candidate factory (requires LIVE_CAMERA=True)
# ------------------------------------------------------------------ #

def _get_live_candidates() -> list[GroceryItem]:
    """Run a real YOLO+RAFT survey and convert ObjectCandidates to GroceryItems."""
    from planning.grocery_item import GroceryItem
    # Defer heavy imports so offline mode starts instantly
    from vision.torch_device import select_torch_device
    from vision.yolo_segmenter import YOLOSegmenter
    from vision.raft_runner import RAFTStereoRunner
    from vision.stereo_rectifier import StereoRectifier
    from vision.survey import run_survey_workspace
    from test_calibration_bundle_live_stereo_z_pickplace import (
        open_stereo_camera,
        load_bundle,
        load_stereo_calibration,
    )
    from config.camera_config import STEREO_INDEX
    from hardware.cameras.stereo_apriltag_viewer import build_detector
    from config.robot_config import ROBOT_CONFIG
    from hardware.robot import Robot

    bundle = load_bundle()
    stereo_calib = load_stereo_calibration()
    device_info = select_torch_device(print_info=True)
    rectifier = StereoRectifier(stereo_calib)
    yolo = YOLOSegmenter(device_info=device_info)
    raft = RAFTStereoRunner(device_info=device_info)
    yolo.warmup()
    raft.warmup()

    stereo = open_stereo_camera()
    detector = build_detector()

    robot = Robot(ROBOT_CONFIG)
    candidates_raw = run_survey_workspace(
        stereo, detector, stereo_calib, rectifier, yolo, raft, robot, bundle
    )
    stereo.release()

    items: list[GroceryItem] = []
    for cand in candidates_raw:
        item = GroceryItem.from_object_candidate(cand, attrs_db={}, item_id=cand.yolo.class_name)
        items.append(item)
    return items


# ------------------------------------------------------------------ #
# Pick/place execution (stub — extend for real robot)
# ------------------------------------------------------------------ #

def _execute_pick(item: GroceryItem) -> bool:
    if not LIVE_ROBOT or not ENABLE_ROBOT_MOTION:
        print(f"    [SIM] Pick {item.class_name} at robot xyz={item.grasp_xyz_robot_mm}")
        return True
    # TODO: wire to execute_pick() from run_pickplace_fast.py
    print(f"    [ROBOT] Pick {item.class_name} (LIVE_ROBOT not yet implemented here)")
    return False


def _execute_place(item: GroceryItem, spot) -> bool:
    if not LIVE_ROBOT or not ENABLE_ROBOT_MOTION:
        print(f"    [SIM] Place {item.class_name} at bag spot ({spot.x:.1f},{spot.y:.1f})")
        return True
    print(f"    [ROBOT] Place {item.class_name} (LIVE_ROBOT not yet implemented here)")
    return False


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #

def main() -> int:
    print("=" * 60)
    print("run_pickplace_2d_blb.py")
    print(f"  LIVE_CAMERA={LIVE_CAMERA}  LIVE_ROBOT={LIVE_ROBOT}")
    print(f"  USE_FAKE_CANDIDATES={USE_FAKE_CANDIDATES}")
    print(f"  Bag: {BAG_WIDTH_CM:.1f} x {BAG_DEPTH_CM:.1f} cm")
    print(f"  MAX_PICKPLACE_STEPS={MAX_PICKPLACE_STEPS}")
    print("=" * 60)

    if USE_FAKE_CANDIDATES or not LIVE_CAMERA:
        print("[MODE] Using fake grocery candidates (no hardware).")
        remaining = _get_fake_candidates()
    else:
        print("[MODE] Surveying workspace with YOLO+RAFT…")
        remaining = _get_live_candidates()

    if not remaining:
        print("[WARN] No candidates found.")
        return 0

    bag = BagState(bag_width_cm=BAG_WIDTH_CM, bag_depth_cm=BAG_DEPTH_CM)
    step = 0

    while remaining and step < MAX_PICKPLACE_STEPS:
        result: PickResult | None = best_pick(remaining, bag)
        if result is None:
            print(f"\n  No more items can fit. {len(remaining)} remaining.")
            break

        item = result.item
        spot = result.spot

        print(
            f"\n  Step {step + 1}: PICK  {item.class_name:<12} "
            f"size=({item.padded_rect_xy_cm[0]:.1f}x{item.padded_rect_xy_cm[1]:.1f})  "
            f"score={result.score:.3f}"
        )
        pick_ok = _execute_pick(item)
        if not pick_ok:
            print(f"    Pick failed for {item.class_name}; skipping.")
            remaining.remove(item)
            continue

        print(
            f"  Step {step + 1}: PLACE {item.class_name:<12} "
            f"→ bag spot ({spot.x:.1f},{spot.y:.1f})"
        )
        place_ok = _execute_place(item, spot)
        if not place_ok:
            print(f"    Place failed for {item.class_name}; skipping.")
            remaining.remove(item)
            continue

        bag.add(item, spot.x, spot.y)
        remaining.remove(item)
        step += 1

    print(f"\n[DONE] Placed {step} item(s). Bag free area: {bag.free_area_estimate_cm2():.1f} cm²")
    print(bag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
