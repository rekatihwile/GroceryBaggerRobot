# Final Demo Code Audit

Static audit date: 2026-06-01

Scope: this audit used static code inspection plus saved logs, calibration bundles, and run snapshots. I did not run any robot, serial, camera, or live hardware scripts.

## Executive Summary

The final wet-run code path is `scripts/autonomous_missed_pick_recovery.py`, not the older README entrypoint. The strongest evidence is `autonomous_missed_pick_recovery_last_run.txt` and the recent manifests in `data/run_snapshots/`, which record `script = autonomous_missed_pick_recovery.py` and `place_planning_sequence = bag_local_aabb_joint`.

The real system pipeline is:

1. Stereo burst capture from one side-by-side USB stereo device.
2. YOLO segmentation over the stereo-left burst.
3. Burst track filtering requiring the object to persist across the burst.
4. RAFT-Stereo disparity once per selected survey frame.
5. Masked point cloud and robust object geometry.
6. Overhead YOLO match plus height-indexed overhead homography for XY correction.
7. Deterministic bag-local 3D AABB placement planning.
8. Python robot kinematics and soft-limit checks.
9. Teensy firmware execution with synchronized stepper moves and INA219 current-aware dynamic grip.
10. YOLO-only missed-pick watchdog before release; RAFT recovery runs only after a confirmed miss.

Best supportable demo claim: the system is an autonomous pick-place loop with learned segmentation, dense stereo depth, height-aware calibration, deterministic 3D packing, current-triggered gripping, and a missed-pick gate before release.

Claims to avoid: global-optimal packing, live use of class-specific fragility from `config/grocery_spec.json` in the wet-run planner, dynamic release being active, dynamic lowering being active in the final pick path, or "20 unique objects placed" from the latest manifest alone.

## Active Entrypoints

Primary wet-run entrypoint:

- `scripts/autonomous_missed_pick_recovery.py`
- It records snapshots under `data/run_snapshots/` and writes manifests in `_write_run_manifest()` (`scripts/autonomous_missed_pick_recovery.py:401`).
- It imports the real pick helper from `scripts/pick_one_place_one.py`, especially `execute_pick_selected()` (`scripts/pick_one_place_one.py:772`).
- The latest run log starts with "AUTONOMOUS PICK PLACE - MISSED PICK RECOVERY" and prints `RUNTIME_CONTEXT='wet_run'`, `WORKSPACE_PROFILE='wet_run'`, `PLACE_PLANNING_SEQUENCE_NAME='bag_local_aabb_joint'`, and `SAVE_RUN_SNAPSHOT=True`.

Useful no-hardware entrypoint:

- `scripts/dry_run_autonomous.py` validates much of the selection/planning flow without robot motion.

Stale or older public entrypoints:

- `README.md` still advertises `run_pickplace_fast.py`.
- `run_buildup_pickplace.py` and older BLB/2D modules remain in the repository, but the latest saved wet runs point to `scripts/autonomous_missed_pick_recovery.py`.

## Saved Run Evidence

Recent manifests show the final planner sequence settled on `bag_local_aabb_joint`.

| Run snapshot | Script | Sequence | Manifest object rows | Rows marked placed | Final `placed_boxes` count | Notes |
|---|---|---:|---:|---:|---:|---|
| `run_20260531_185839` | `autonomous_missed_pick_recovery.py` | `bag_local_aabb_joint` | 20 | 20 | 11 | `object_i` and image names repeat, so this is not 20 unique placements. The final bag-state count is 11. |
| `run_20260531_185559` | `autonomous_missed_pick_recovery.py` | `bag_local_aabb_joint` | 2 | 2 | 2 | Short successful run. |
| `run_20260531_185404` | `autonomous_missed_pick_recovery.py` | `bag_local_aabb_joint` | 3 | 2 | 2 | Third object remained pending. |
| `run_20260531_184606` | `autonomous_missed_pick_recovery.py` | `bag_local_aabb_joint` | 11 | 9 | 2 | Manifest contains more rows than final bag-state boxes. |
| `run_20260531_183430` | `autonomous_missed_pick_recovery.py` | `bag_local_aabb_joint` | 21 | 20 | 7 | Evidence of repeated/continued attempts. |

The latest terminal log tail is more reliable than counting manifest rows by itself. It shows object 11 was released, the manifest was written, and `[FLOW] placed_boxes count = 11`. The next survey for object 12 found no valid candidate, then the run exited by quit request.

## Calibration

Active calibration files:

- `robot_calibration_bundle.npz`
- `stereo_calibration.npz`

The generated calibration report is `robot_calibration_report.txt`, generated on Sunday, May 31, 2026 at 12:38:20. It reports:

- 65 rows scanned.
- 53 overhead-visible samples.
- 34 stereo-visible samples.
- Height-indexed overhead homography levels: `z = [0, 50, 150, 200]` mm.
- Homography RMS by layer: 9.58, 11.51, 6.72, and 4.17 mm.
- Full stereo `robot_xyz_from_cam_xyz` RMSE: 8.01 mm.
- Stereo baseline in `stereo_calibration.npz`: 59.9965 mm.

How calibration is built:

- `calibration/calibrate_all_safe_grid_xyz_models.py` builds one overhead homography per Z layer using `cv2.findHomography()` (`build_homography_layers()` around line 943).
- It fits stereo camera XYZ to robot XYZ with least-squares affine models via `np.linalg.lstsq()` (`fit_stereo_models()` around line 1019). This is an affine fit, not a rigid-body Kabsch/Procrustes calibration.
- `test_calibration_bundle_live_stereo_z_pickplace.py` loads the bundle and maps overhead `(u, v, z)` through interpolated `H(z)` (`map_uv_z_to_robot_xy()` around line 475).
- Active overhead XY projection in `vision/pick_xy_resolver.py` clamps negative object Z to at least 0 before calling the bundle mapper.

There is also an older `overhead_homography_z_lookup.npz` with different Z levels and `overhead_index=2`. The active wet-run code loads `robot_calibration_bundle.npz`.

## Vision Pipeline

The core survey pipeline lives in `vision/pick_survey_pipeline.py`:

- `load_vision()` loads `YOLOSegmenter` and `RAFTStereoRunner`.
- `run_survey()` captures the stereo burst, runs YOLO, clusters burst detections, runs RAFT once on the best shared frame, then matches a fresh overhead YOLO frame to stereo candidates.

Active perception settings from `config/survey/survey_config.py` and the latest run log:

- YOLO weights: `yolo_weights/full_data.pt`, fallback `yolo_weights/validate_V2.pt`.
- YOLO image size: 640.
- YOLO confidence: 0.35.
- YOLO IoU: 0.50.
- Retina masks enabled.
- RAFT checkpoint: `RAFT-Stereo/models/raftstereo-middlebury.pth`.
- RAFT iterations: 16.
- Minimum disparity: 1 px.
- Minimum valid object points: 300.
- Survey burst: 5 frames.
- Required hits: 5 of 5 frames.
- Burst cluster centroid threshold: 75 px.
- Overhead XY blend weight: 0.45.
- Latest log used CUDA on an NVIDIA RTX A1000 Laptop GPU.

Stereo camera architecture:

- `hardware/cameras/stereo_apriltag_viewer.py` opens one USB device at index 2.
- The camera produces a 1280x480 side-by-side frame.
- `read_pair()` splits it into left and right 640x480 images.

Candidate geometry:

- `vision/yolo_segmenter.py` produces masks, bounding boxes, confidence, centroids, and PCA-like mask axes.
- `vision/raft_runner.py` pads images, runs RAFT-Stereo, and returns dense disparity.
- `vision/pointcloud.py` converts masked disparity into camera-frame points using stereo calibration and `Q` when available.
- `vision/pick_candidate_builder.py` rejects candidates with fewer than 300 valid object points, resolves robust Z, and saves point/disparity debug outputs.
- `vision/object_geometry.py` builds `ObjectCandidate` geometry, including robot-frame XYZ, footprints, object height, and pick phi.
- `vision/pick_xy_resolver.py` matches overhead detections to stereo candidates and blends overhead XY with stereo XY.
- `vision/pick_phi_resolver.py` supports the active `centroid_shortest_ray_parallel` phi mode.

## Pick Targeting

Active pick settings:

- `PICK_PHI_MODE = centroid_shortest_ray_parallel`.
- `REQUIRE_OVERHEAD_XY_FOR_PICK = True`.
- `USE_Z_GROUND_MODEL_FOR_PICK_SURFACE = True`.
- `USE_LOCAL_HEIGHT_AWARE_GRASP_XY = True`.
- Local top-region XY blend weight: 0.35.
- `GRIPPER_OFFSET_MM = 140`.
- `Z_MAX_MM = 270`.
- `MIN_PICK_GRASP_Z_MM = 125`.

The pick helper refuses to pick if overhead XY is required and unavailable. Before a pick, it may shift the XY target toward a locally higher top-region point in the point cloud. Pick Z is computed by `motion/pick_z_policy.py` as object surface plus gripper offset, with short-object buffering and minimum-Z clamping.

Active dynamic pick behavior:

- Dynamic pick is enabled.
- Dynamic grip angle from object geometry is enabled.
- Dynamic height probing is disabled (`USE_DYNAMIC_PICK_HEIGHT=False`), so the final path skips `DLR` and descends to feedforward grasp Z.
- The firmware `DG` dynamic grip command is active. It closes the servo until current-change logic indicates grip contact.
- Dynamic pick fallback to the old fixed open-descend-close path is disabled.

## Placement and Planning

Active place scene from `config/surface_zones.json`:

- Scene: `New Bag Test`.
- Center: `(210, 755)` mm.
- Surface Z: `-155` mm.
- Width/depth: `260 x 150` mm.
- Default phi: 0 deg.

Active place settings:

- `PLACE_PLANNING_SEQUENCE_NAME = bag_local_aabb_joint`.
- Bag-local planning height: 250 mm.
- AABB padding: X = 0 mm, Y = 0 mm, Z = 20 mm.
- Efficient packing enabled.
- Gripper footprint must stay inside the bag.
- Rotation candidates include 0 and 90 deg offsets.
- `USE_PICK_PHI_FOR_PLACE = False`.
- Place release mode is fixed servo open, not dynamic release.
- Negative-bin place Z policy is active (`PLACE_Z_POLICY_MODE='negative_bin_hang'`).

Planning flow:

- `scripts/autonomous_best_candidate.py` still filters candidates for workspace bounds, reach, soft-limit safety, overlap, and volume.
- `planning/autonomous_planning_sequences.py` then computes a bag-local target for each candidate that passed filters.
- `planning/aabb_utils.py` converts the selected object geometry into a raw and padded 3D AABB.
- `planning/bag_local_3d_aabb_planner.py` evaluates deterministic 2.5D AABB placements over layers, rotations, and candidate XY starts.

Important selection detail: for `bag_local_aabb_joint`, the final sort explicitly prefers floor placements over stack placements, and among floor placements it prefers smaller-volume floor items before using planner score and future feasibility as tie-breakers. This is different from older "largest volume first" language in some scripts.

Future feasibility:

- The bag-local planner can evaluate remaining items as `FutureItemSpec`s and refines top candidates with future feasibility.
- The active call in `_compute_bag_local_target()` passes future specs generated from current candidates, but does not pass class-specific `object_properties` or `support_properties`.
- Therefore `config/grocery_spec.json` is present and used by capstone visualization/demo scripts, but I did not find evidence that its per-class weight/fragility/compliance values are active in the wet-run `bag_local_aabb_joint` call path.

## Robot Motion Layer

`hardware/robot.py` is the Python robot interface:

- Serial: COM4 at 115200 baud in the latest run.
- Planar two-link FK/IK with L1 = 450 mm and L2 = 450 mm.
- J4 world yaw is handled as wrist compensation: motor J4 = desired phi - `(q1 + q2)`.
- Cartesian moves run through soft-limit planning when enabled.
- Before sending a move, Python samples the actual interpolated motor-step path and checks soft limits.
- Motion commands use `MOVESYNC_T` when a move duration is supplied.
- Dynamic helper commands include `DG`, `DR`, `DL`, and `DLR`.

The latest log shows direct soft-limit path checks and `MOVESYNC_T` commands during place hover, release, and recovery moves.

## Firmware Layer

Active firmware: `Teensy_Code/Teensy_Code.ino`.

Hardware/control stack:

- Teensy 4.1.
- `AccelStepper` for four axes.
- TMC2209 drivers over UART.
- Servo on pin 14.
- INA219 current monitor on I2C.
- J1/J2 limit switches on pins 15 and 16.

Key pin map:

- J1: step/dir/en = 2/3/4.
- J2: step/dir/en = 5/6/7.
- J3: step/dir/en = 8/9/10.
- J4: step/dir/en = 11/12/13.
- Servo: 14.

Firmware capabilities:

- `HOME`, `HOMEJ1`, `HOMEJ2`, `HOMEJ3`.
- `MOVE`, `MOVESYNC`, `MOVESYNC_T`.
- `POS`, `LIMITS`, `SHOW`.
- `servo = ...`, `J3 = ...`.
- `dynset` runtime tuning.
- `DG` dynamic grip.
- `DR` dynamic release.
- `DL` and `DLR` dynamic lower.
- StallGuard streaming/checking for J3 dynamic lower.

Firmware safety boundaries:

- It has homing timeouts, J1/J2 limit switches, servo clamping, INA219 checks for dynamic operations, and J3 dynamic lower bounds.
- Cartesian collision/keep-out logic is Python-side, not firmware-side.
- J4 has no physical homing sensor in the firmware; HOME sets its counter to zero.

## Missed-Pick Recovery

The final script adds a conservative missed-pick gate before release:

- It moves the held object to place hover first.
- It starts a YOLO-only watchdog burst from the configured camera.
- It intentionally skips RAFT/disparity/point cloud in this watchdog.
- If the object still appears at the pick site with enough score and hits, the script cancels release.
- RAFT is run once only after a confirmed miss, during recovery/relocalization.

Active watchdog settings from the latest log:

- Camera: overhead.
- Burst count: 8.
- Minimum hits: 3.
- Timeout: 4.0 s.
- Period: 0.15 s.
- Score threshold: 0.65.
- Scoring weights: class 0.25, robot distance 0.35, IoU 0.20, area 0.20.
- Max robot distance for match: 35 mm.
- Min IoU: 0.2.
- Area ratio window: 0.5 to 2.0.

Latest log example: object 11's watchdog reported no miss, skipped RAFT, then released the object.

## Demo-Safe Caveats

- The latest manifest has 20 rows marked placed, but `object_i` and file names repeat. Use the run log's final `placed_boxes count = 11` for the current bag-state claim.
- The active planner is deterministic and geometry-based, but it is not a global optimizer.
- `config/grocery_spec.json` is real, but per-class fragility/weight does not appear to be passed into the active wet-run planner path.
- Dynamic release and dynamic height probing are implemented in firmware/Python, but the final config disables them.
- Calibration errors are low enough to discuss, but not zero: overhead H(z) RMS ranges from about 4.17 to 11.51 mm, and stereo XYZ RMSE is about 8.01 mm.
- Some log lines show high overhead support-distance warnings near the end of the run. Present the system as robustly instrumented and safety-gated, not as perfectly calibrated in every frame.
