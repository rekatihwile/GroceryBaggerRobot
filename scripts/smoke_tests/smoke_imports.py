"""
smoke_imports.py

Smoke test: verifies all new modules can be imported without hardware.

No cameras, no YOLO model on disk, no RAFT model on disk, no robot required.
Just checks that the module-level import machinery works (Python paths, class
definitions, dataclass fields, etc.).

Run:
    python scripts/smoke_tests/smoke_imports.py
"""

# ============================================================
# USER SETTINGS
# ============================================================

VERBOSE = True

# ============================================================
# End of user settings
# ============================================================

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _try_import(label: str, import_fn):
    try:
        result = import_fn()
        if VERBOSE:
            print(f"  [OK]  {label}")
        return True, result
    except Exception as exc:
        print(f"  [FAIL] {label}  --  {type(exc).__name__}: {exc}")
        return False, None


def main() -> int:
    print("=" * 60)
    print("smoke_imports.py")
    print("=" * 60)

    failures = 0

    # ---- hardware ----
    ok, _ = _try_import(
        "hardware.robot.Robot",
        lambda: __import__("hardware.robot", fromlist=["Robot"]).Robot,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "hardware.robot.RobotConfig",
        lambda: __import__("hardware.robot", fromlist=["RobotConfig"]).RobotConfig,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "hardware.robot.JointPose",
        lambda: __import__("hardware.robot", fromlist=["JointPose"]).JointPose,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "hardware.cameras.stereo_apriltag_viewer.SimpleStereoCamera",
        lambda: __import__(
            "hardware.cameras.stereo_apriltag_viewer",
            fromlist=["SimpleStereoCamera"],
        ).SimpleStereoCamera,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "hardware.cameras.overhead_camera.SimpleOverheadCamera",
        lambda: __import__(
            "hardware.cameras.overhead_camera",
            fromlist=["SimpleOverheadCamera"],
        ).SimpleOverheadCamera,
    )
    if not ok:
        failures += 1

    # ---- planning ----
    ok, _ = _try_import(
        "planning.grocery_item.GroceryItem",
        lambda: __import__("planning.grocery_item", fromlist=["GroceryItem"]).GroceryItem,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "planning.bag_state.BagState",
        lambda: __import__("planning.bag_state", fromlist=["BagState"]).BagState,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "planning.planner_2d_blb.best_pick",
        lambda: __import__(
            "planning.planner_2d_blb", fromlist=["best_pick"]
        ).best_pick,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "planning.aabb_utils.AxisAlignedBox3D",
        lambda: __import__("planning.aabb_utils", fromlist=["AxisAlignedBox3D"]).AxisAlignedBox3D,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "planning.adjacent_placement.compute_adjacent_placement",
        lambda: __import__(
            "planning.adjacent_placement", fromlist=["compute_adjacent_placement"]
        ).compute_adjacent_placement,
    )
    if not ok:
        failures += 1

    # ---- vision (no model loading, just module-level definitions) ----
    ok, _ = _try_import(
        "vision.torch_device.TorchDeviceInfo",
        lambda: __import__(
            "vision.torch_device", fromlist=["TorchDeviceInfo"]
        ).TorchDeviceInfo,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.torch_device.select_torch_device (function present)",
        lambda: __import__(
            "vision.torch_device", fromlist=["select_torch_device"]
        ).select_torch_device,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.yolo_segmenter.YOLODetection",
        lambda: __import__(
            "vision.yolo_segmenter", fromlist=["YOLODetection"]
        ).YOLODetection,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.yolo_segmenter.YOLOSegmenter (class definition)",
        lambda: __import__(
            "vision.yolo_segmenter", fromlist=["YOLOSegmenter"]
        ).YOLOSegmenter,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.raft_runner.RAFTStereoRunner (class definition)",
        lambda: __import__(
            "vision.raft_runner", fromlist=["RAFTStereoRunner"]
        ).RAFTStereoRunner,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.stereo_rectifier.StereoRectifier",
        lambda: __import__(
            "vision.stereo_rectifier", fromlist=["StereoRectifier"]
        ).StereoRectifier,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.pointcloud.masked_disparity_to_pointcloud",
        lambda: __import__(
            "vision.pointcloud", fromlist=["masked_disparity_to_pointcloud"]
        ).masked_disparity_to_pointcloud,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.object_geometry.ObjectCandidate",
        lambda: __import__(
            "vision.object_geometry", fromlist=["ObjectCandidate"]
        ).ObjectCandidate,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.survey.run_survey_workspace",
        lambda: __import__(
            "vision.survey", fromlist=["run_survey_workspace"]
        ).run_survey_workspace,
    )
    if not ok:
        failures += 1

    # ---- new placement/policy modules ----
    ok, _ = _try_import(
        "config.surface_zone_io.load_surface_zones",
        lambda: __import__(
            "config.surface_zone_io", fromlist=["load_surface_zones"]
        ).load_surface_zones,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "motion.pick_z_policy.compute_pick_z_plan",
        lambda: __import__(
            "motion.pick_z_policy", fromlist=["compute_pick_z_plan"]
        ).compute_pick_z_plan,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "motion.place_z_policy.compute_place_z_plan",
        lambda: __import__(
            "motion.place_z_policy", fromlist=["compute_place_z_plan"]
        ).compute_place_z_plan,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "vision.grasp_xy_policy.compute_grasp_xy_with_local_height",
        lambda: __import__(
            "vision.grasp_xy_policy", fromlist=["compute_grasp_xy_with_local_height"]
        ).compute_grasp_xy_with_local_height,
    )
    if not ok:
        failures += 1

    ok, _ = _try_import(
        "motion.grasp_current_policy.GraspCurrentSettings",
        lambda: __import__(
            "motion.grasp_current_policy", fromlist=["GraspCurrentSettings"]
        ).GraspCurrentSettings,
    )
    if not ok:
        failures += 1

    print()
    total = 24
    passed = total - failures
    print(f"Result: {passed}/{total} imports OK, {failures} failed.")

    if failures == 0:
        print("\n[PASS] smoke_imports: all imports succeeded.")
        return 0
    else:
        print("\n[FAIL] smoke_imports: some imports failed (see above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
