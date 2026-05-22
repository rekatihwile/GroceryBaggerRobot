# Grocery Buildup Robotics Workspace

This repository contains the local grocery-bagger robotics capstone code: robot control, camera utilities, calibration scripts, and the current YOLO plus RAFT-Stereo pick/place demo.

## Main Commands

Run the organized entrypoint:

```powershell
python run_pickplace_fast.py
```

The old gold-master command still works through a compatibility wrapper:

```powershell
python test_yolo_raft_pointcloud_pickplace_fast.py
```

Useful safe modes:

```powershell
python run_pickplace_fast.py --gpu-check
python run_pickplace_fast.py --survey-only
```

## Current Pipeline

The working demo uses the stereo-left frame for YOLO segmentation masks, RAFT-Stereo disparity for dense depth, masked point-cloud targeting, and the existing camera-to-robot calibration transform to choose robot pick targets. The overhead camera pathway is still preserved for end-effector validation and future localization work.

## Stereo Camera Architecture

**The stereo camera is ONE USB device, not two.**  
`SimpleStereoCamera` in `hardware/cameras/stereo_apriltag_viewer.py` opens a single `cv2.VideoCapture(STEREO_INDEX, cv2.CAP_DSHOW)` capture — the USB stereo module delivers a 1280×480 side-by-side stream. `read_pair()` splits the frame at the mid-column to produce `left` (640×480) and `right` (640×480). Never pass two separate camera indices to this class.

## Intended Future Pipeline

The intended robust localization path is stereo for object Z/height plus the overhead camera for XY through a height-dependent homography. J4 yaw can then align the gripper with the object mask minor axis measured in the undistorted overhead webcam frame.

## New Modules (refactor 2025)

### `planning/`

2D BLB bag-packing planning layer. No camera or robot hardware required.

| File                         | Purpose                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| `planning/grocery_item.py`   | `GroceryItem` dataclass — physical attributes of a scanned item        |
| `planning/bag_state.py`      | `BagState` — tracks placed items, checks geometry, estimates free area |
| `planning/planner_2d_blb.py` | Bottom-Left-Back 2D planner; `best_pick()` selects next item + spot    |

### `vision/`

Extracted vision utilities. All module-level constants match `run_pickplace_fast.py` defaults so imported call sites are unchanged.

| File                         | Purpose                                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| `vision/torch_device.py`     | `TorchDeviceInfo`, `select_torch_device()`, CUDA helpers                                                 |
| `vision/yolo_segmenter.py`   | `YOLODetection`, `YOLOSegmenter` — YOLO seg wrapper with all knobs as constructor kwargs                 |
| `vision/raft_runner.py`      | `RAFTStereoRunner` — RAFT-Stereo wrapper with all knobs as constructor kwargs                            |
| `vision/stereo_rectifier.py` | `StereoRectifier` — lazy rectification map cache                                                         |
| `vision/pointcloud.py`       | `masked_disparity_to_pointcloud`, `choose_object_target_point`, `estimate_pick_phi_from_mask_minor_axis` |
| `vision/object_geometry.py`  | `ObjectCandidate` dataclass, `build_object_candidate`, `fuse_survey_candidates`                          |
| `vision/survey.py`           | `run_survey_workspace` — burst capture + YOLO + RAFT + point-cloud pipeline                              |

### New scripts

```powershell
# 2D BLB offline demo (no hardware)
python scripts/run_pickplace_2d_blb.py

# Batch RAFT disparity over a folder of image pairs
python scripts/run_raft_disparity_batch.py

# Batch YOLO segmentation over a folder of images
python scripts/run_segment_batch.py
```

All new scripts use a **USER SETTINGS block** at the top of the file — no argparse.  
Defaults are always `LIVE_CAMERA = False` / `LIVE_ROBOT = False` so they are safe to open and run in the IDE without hardware.

## Main Pipeline

### `run_buildup_pickplace.py`

Production pick/place pipeline.  Three key differences from `run_pickplace_fast.py`:

1. **No EE AprilTag required** — object Z comes directly from stereo triangulation via
   `A_robot_from_cam_xyz_3x4`.  `stereo_z_bias` is always 0.
2. **Overhead YOLO → H(z) homography** for robot X/Y.  During each survey the overhead
   camera is captured and YOLO detections are matched to stereo candidates; each match
   is projected through the height-indexed homography `H(z)` to give accurate X/Y.
3. **BLB 2D packing planner** for placement.  Bag geometry (origin, width, depth in mm)
   is defined in the USER SETTINGS block.  `f` key calls `choose_placement_spot_2d` and
   executes the computed spot automatically.

```powershell
python run_buildup_pickplace.py
```

Edit the **USER SETTINGS** block at the top of the file to set bag geometry, camera
indices, YOLO weights, and motion parameters.

| Key         | Action                                                |
| ----------- | ----------------------------------------------------- |
| `s`         | Survey workspace (stereo YOLO+RAFT + overhead YOLO)   |
| `1`–`9`     | Select surveyed candidate                             |
| `g`         | Move to survey pose                                   |
| `m`         | Hover to selected object                              |
| `k`         | Pick selected object                                  |
| `f`         | Place using BLB-planned bag spot                      |
| `w`         | Print current bag state                               |
| `x`         | Reset bag state (clear placed items)                  |
| `b`         | Print geometry sanity check                           |
| `r`         | Print calibration matrices                            |
| `[` / `]`   | Jog Z down / up                                       |
| `,` / `.`   | Jog phi down / up                                     |
| `o` / `l`   | Open / close claw                                     |
| `p`         | Print current FK pose                                 |
| `h/c/a`     | Home / sync-home / assume-home                        |
| `e` / `d`   | Enable / disable motors                               |
| `q` / `ESC` | Quit                                                  |

## Vision Debug Scripts

Live-camera interactive debug and validation tools.  
No argparse — all settings are `USER SETTINGS` blocks at the top of each file.  
Neither script commands the robot. Both use the **one-USB stereo split** architecture.

### `scripts/live_yolo_camera_viewer.py`

Live overhead + stereo preview with on-demand YOLO segmentation.

```powershell
python scripts/live_yolo_camera_viewer.py
```

| Key         | Action                                       |
| ----------- | -------------------------------------------- |
| `SPACE`     | Freeze current frames and run YOLO detection |
| `c`         | Toggle continuous YOLO mode                  |
| `s`         | Save freeze / overlay / detections JSON      |
| `r`         | Resume live preview                          |
| `q` / `ESC` | Quit                                         |

Saves to `outputs/live_yolo_freezes/freeze_YYYYMMDD_HHMMSS/` when `SAVE_FREEZE_FRAMES = True`.

### `scripts/stepwise_yolo_raft_pointcloud_validation.py`

Live freeze → rectify → YOLO → RAFT disparity → masked point cloud → robot-frame target estimate.

```powershell
python scripts/stepwise_yolo_raft_pointcloud_validation.py
```

| Key         | Action                                       |
| ----------- | -------------------------------------------- |
| `SPACE`     | Capture / freeze stereo pair and rectify     |
| `y`         | Run YOLO on frozen rectified-left            |
| `1`–`9`     | Select detected object by index              |
| `d`         | Run RAFT disparity on rect pair              |
| `p`         | Build masked point cloud for selected object |
| `m`         | Map point cloud centroid to robot frame      |
| `a`         | Run all steps automatically after freeze     |
| `s`         | Save all outputs                             |
| `r`         | Resume live preview                          |
| `q` / `ESC` | Quit                                         |

Saves to `outputs/stepwise_yolo_raft_pointcloud_validation/run_YYYYMMDD_HHMMSS/` when `SAVE_VALIDATION_OUTPUTS = True`.

### `scripts/vision_debug_helpers.py`

Shared drawing / saving / layout utilities used by both scripts above.  
No cameras, no GPU.

## Smoke Tests

Run without any hardware:

```powershell
python scripts/smoke_tests/smoke_imports.py
python scripts/smoke_tests/smoke_planner_2d.py
python scripts/smoke_tests/smoke_stereo_split.py
python scripts/smoke_tests/smoke_live_vision_imports.py
```

Hardware / camera smoke tests:

```powershell
python scripts/smoke_tests/camera_oop_smoke_test.py
python scripts/smoke_tests/test_robot.py
python scripts/smoke_tests/soft_limit_workspace_test.py
```

## Folder Structure

- `config/` holds robot, camera, and soft-limit configuration code.
- `hardware/` holds robot and camera wrappers.
- `calibration/` holds calibration and homography scripts.
- `planning/` holds 2D bag-packing data models and the BLB planner.
- `scripts/manual_control/` holds direct manual-control and click-control tools.
- `scripts/smoke_tests/` holds quick sanity checks.
- `vision/` holds extracted vision helpers (torch device, YOLO, RAFT, point cloud, survey).
- `_archive/old_build_up_scripts/` holds old duplicate buildup scripts.

## Path Stability

`RAFT-Stereo/`, `yolo_weights/`, calibration `.npz` files, calibration `.csv` files, and root `.txt` data files remain at the repository root for path stability. The gold-master logic still expects those runtime-sensitive paths in their original locations.

## Manual And Smoke Scripts

Manual-control tools live in:

```powershell
python scripts/manual_control/manual_cartesian_jog.py
python scripts/manual_control/workspace_click_jog.py
python scripts/manual_control/workspace_click_pickplace.py
python scripts/manual_control/robot_camera_motion_test.py
```

Smoke and sanity tools live in:

```powershell
python scripts/smoke_tests/camera_oop_smoke_test.py
python scripts/smoke_tests/test_robot.py
python scripts/smoke_tests/soft_limit_workspace_test.py
```

## Future Work

- 3D stacking / multi-layer bin packing (currently 2D only)
- Wire `scripts/run_pickplace_2d_blb.py` to real robot motion (set `LIVE_ROBOT=True`, `ENABLE_ROBOT_MOTION=True`)
- Real-time survey → `GroceryItem` attribute lookup against a product database
- Overhead camera XY integration in the 2D planner
