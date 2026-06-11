# Grocery Bagger Robotics Workspace

This repository contains the grocery-bagger robotics capstone code: robot control, camera utilities, vision pipeline, 3D AABB bag packing, and the autonomous pick/place loop.

## Main Entrypoint

The wet-run entrypoint is the autonomous missed-pick recovery script (see `docs/final_demo_code_audit.md` for the full pipeline audit):

```powershell
python scripts/autonomous_missed_pick_recovery.py
```

## Current Pipeline

1. Stereo burst capture from one side-by-side USB stereo device.
2. YOLO segmentation over the stereo-left burst, with burst track filtering.
3. RAFT-Stereo disparity once per selected survey frame.
4. Masked point cloud and robust object geometry (`vision/`).
5. Overhead YOLO match plus height-indexed overhead homography H(z) for XY correction.
6. Deterministic bag-local 3D AABB placement planning (`planning/`).
7. Python kinematics and soft-limit checks (`hardware/robot.py`), Teensy firmware execution with current-aware dynamic grip (`Teensy_Code/`).
8. YOLO-only missed-pick watchdog before release; RAFT recovery only after a confirmed miss.

## Stereo Camera Architecture

**The stereo camera is ONE USB device, not two.**  
`SimpleStereoCamera` in `hardware/cameras/stereo_apriltag_viewer.py` opens a single `cv2.VideoCapture(STEREO_INDEX, cv2.CAP_DSHOW)` capture — the USB stereo module delivers a 1280×480 side-by-side stream. `read_pair()` splits the frame at the mid-column to produce `left` (640×480) and `right` (640×480). Never pass two separate camera indices to this class.

## Folder Structure

- `config/` — robot, camera, pick/place/survey, soft-limit, and runtime configuration.
- `hardware/` — robot serial interface and camera wrappers.
- `motion/` — pick/place motion sequences, Z policies, dynamic grasp, recovery moves.
- `planning/` — bag-local 3D AABB planner and planning sequences.
- `vision/` — survey pipeline: YOLO segmentation, RAFT disparity, point cloud, candidate geometry, XY/Z/phi resolvers.
- `scripts/` — the main entrypoint and its direct helpers.
- `calibration/` — `calibrate_all_safe_grid_xyz_models.py`, the tool that regenerates `robot_calibration_bundle.npz`.
- `Teensy_Code/` — Teensy 4.1 firmware (steppers, homing, dynamic grip via INA219).
- `docs/` — final demo audit, claims table, and visualization usage notes.
- `data/`, `PointCloud_Validation/`, `Training_Images/`, `paper_plots/`, `paper_figure_sources/`, `presentation_outputs/`, `run_data_export/` — data, run snapshots, and generated figures.
- `test_calibration_bundle_live_stereo_z_pickplace.py` stays at the repo root: it is the calibration-bundle loader/geometry module imported throughout the run path.

## Path Stability

`RAFT-Stereo/`, `yolo_weights/`, calibration `.npz` files, calibration `.csv` files, and root `.txt` data files remain at the repository root. The run path expects those runtime-sensitive paths in their original locations. The active calibration files are `robot_calibration_bundle.npz` and `stereo_calibration.npz`; the pick surface model is `data/z_ground_calibration/z_ground_model_latest.json`.

## Archive

Everything not in the import closure of `scripts/autonomous_missed_pick_recovery.py` lives in `_archive/`:

- `_archive/not_in_main_path/` — mirrors the original tree: old entrypoints (`run_pickplace_fast.py`), calibration tools, smoke tests, manual-control tools, capstone presentation/figure scripts, vision debug viewers, and the 2D BLB planner family. Restore any file by moving it back to the mirrored path.
- `_archive/old_entrypoints/` — `run_buildup_pickplace.py` and the two-object pick/place predecessors.
- `_archive/old_build_up_scripts/` — historical buildup and validation scripts.
- `_archive/demo_baseline/` — frozen baseline copy of the capstone demo scripts.
- `_archive/unused_motion_modules/`, `_archive/misc/` — unreferenced motion modules, old README, scratch files.

Note: the remaining archived calibration tools (e.g. `calibrate_platform_z_from_apriltag.py`, which regenerates the z-ground model) only work if moved back to their original paths — they produce artifacts the run path consumes.
