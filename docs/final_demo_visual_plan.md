# Final Demo Visual Plan

Use at most three software slides. The robot itself is the main demo; these visuals should make the autonomy legible without turning the presentation into a code tour.

## Slide 1: Perception -> Robot Coordinates

Purpose: show how the robot turns camera pixels into a pickable 3D target.

Recommended assets:

- `data/run_snapshots/run_20260531_185839/Stereo_Left_0011.png`
- `data/run_snapshots/run_20260531_185839/left_overlay_0011.png`
- `data/run_snapshots/run_20260531_185839/Overhead_0011.png`
- Optional derived visual from `data/run_snapshots/run_20260531_185839/disparity_0011.npz`
- Optional point cloud source: `data/run_snapshots/run_20260531_185839/points_cam_0011.npz`

Layout:

- Left panel: stereo-left camera view.
- Middle panel: YOLO mask overlay.
- Right panel: overhead view with projected/blended pick target.
- Add a small text strip under the panels with: `YOLO mask -> RAFT disparity -> masked point cloud -> H(z) overhead XY blend`.

Talk-over callouts:

- "5-frame burst, 5 required hits."
- "RAFT runs once per survey on the best shared frame."
- "Stereo estimates Z and size; overhead refines XY."
- "Calibration bundle maps camera measurements into robot coordinates."

Avoid:

- Showing only raw camera frames with no labels.
- Claiming the overhead view alone localizes the object.

## Slide 2: Bag-Local AABB Planner

Purpose: show that placement is computed from object geometry and current bag occupancy.

Recommended assets:

- `scripts/smoke_tests/live_scene_plan_validator_0001.png`
- `scripts/smoke_tests/focused_bag_scene_0001.png`
- `scripts/capstone/autonomous_system_wrapper_0001.png`
- If using a generated planner screenshot, base it on the active scene: `New Bag Test`, center `(210, 755)` mm, bag footprint `260 x 150` mm, height `250` mm.

Layout:

- Full slide: top-down bag footprint with placed AABBs and the next target.
- Use one color for already placed boxes and a stronger outline for the planned next box.
- Include a compact side label: `inside bag`, `no padded overlap`, `support checked`, `vertical descent checked`.

Talk-over callouts:

- "The active sequence is `bag_local_aabb_joint`."
- "The planner checks floor and stack layers."
- "The current run uses 0 mm X/Y padding and 20 mm Z padding."
- "The selector is placement-aware, not just largest-object-first."

Avoid:

- Saying the planner uses global optimization.
- Saying class-specific grocery fragility is active in the wet-run planner unless the code is updated to pass `grocery_spec.json` properties into `plan_bag_local_aabb_placement()`.

## Slide 3: Closed Loop Execution and Recovery

Purpose: show the full cycle and the final run evidence.

Recommended assets:

- A simple flow diagram built from the active code path:
  `survey -> select/place-plan -> pick -> place hover -> YOLO miss watchdog -> release or recover -> next survey`
- Small evidence table from `data/run_snapshots/run_20260531_185839/manifest.json`.
- One log excerpt from `autonomous_missed_pick_recovery_last_run.txt`, preferably:
  - `[MISS CHECK AUDIT] RAFT skipped.`
  - `[PLACE AUTO] OK - item released and robot retracted.`
  - `[FLOW] placed_boxes count = 11`

Layout:

- Top two-thirds: flow diagram.
- Bottom third: "latest wet-run evidence" with script name, planner name, and final bag-state count.

Talk-over callouts:

- "The final script is `scripts/autonomous_missed_pick_recovery.py`."
- "The watchdog is YOLO-only until a miss is confirmed, so the normal release path is fast."
- "If a miss is confirmed, the script cancels release and runs recovery/relocalization."
- "The latest run reached 11 placed boxes in the final bag state before the next object had no valid candidate and the run exited."

Avoid:

- Counting the latest manifest's 20 object rows as 20 unique placements. In that file, `object_i` and image names repeat, while the final `placed_boxes` count is 11.

## Optional Backup Visuals

Calibration:

- Use a small table from `robot_calibration_report.txt`:
  - H(z) RMS: 9.58, 11.51, 6.72, 4.17 mm.
  - Stereo XYZ RMSE: 8.01 mm.

Firmware:

- Show a minimal block diagram:
  `Python IK/soft limits -> serial MOVESYNC_T/DG -> Teensy -> TMC2209 steppers + servo + INA219`.

Run snapshots:

- The latest snapshot directory contains paired files for each saved object index:
  - `Stereo_Left_0001.png` through `Stereo_Left_0011.png`.
  - `Stereo_Right_0001.png` through `Stereo_Right_0011.png`.
  - `Overhead_0001.png` through `Overhead_0011.png`.
  - `left_overlay_0001.png` through `left_overlay_0011.png`.
  - `disparity_0001.npz` through `disparity_0011.npz`.
  - `points_cam_0001.npz` through `points_cam_0011.npz`.

## Suggested Slide Titles

1. "Perception: From Pixels to Robot Coordinates"
2. "Planning: Placement-Aware 3D Packing"
3. "Execution: Current-Aware Grip and Miss Recovery"
