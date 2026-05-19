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

## Intended Future Pipeline

The intended robust localization path is stereo for object Z/height plus the overhead camera for XY through a height-dependent homography. J4 yaw can then align the gripper with the object mask minor axis measured in the undistorted overhead webcam frame.

## Folder Structure

- `config/` holds robot, camera, and soft-limit configuration code.
- `hardware/` holds robot and camera wrappers.
- `calibration/` holds calibration and homography scripts.
- `scripts/manual_control/` holds direct manual-control and click-control tools.
- `scripts/smoke_tests/` holds quick sanity checks.
- `vision/` is reserved for future low-risk extraction from the gold-master script.
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
