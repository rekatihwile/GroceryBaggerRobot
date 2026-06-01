# Missed-Pick Recovery Knobs

This is a practical tuning map for `scripts/autonomous_missed_pick_recovery.py`.

## No-Candidate Resume

- `NO_CANDIDATE_RETRY_COUNT` controls how many extra surveys run after the first no-candidate result. With `1`, the robot tries two surveys total.
- `NO_CANDIDATE_AFTER_RETRIES_MODE="continuous"` keeps surveying forever until `q`/ESC or `c` clear-box. This is useful when you are feeding objects by hand.
- `NO_CANDIDATE_AFTER_RETRIES_MODE="wait_for_resume"` pauses after the retries. Press `NO_CANDIDATE_RESUME_KEY` in the camera window to resume automation.
- `NO_CANDIDATE_AFTER_RETRIES_MODE="stop"` restores the older stop behavior.

## Pick Location

- `PLATFORM_GRID_X_MM`, `PLATFORM_GRID_Y_MM`, and `PLATFORM_GRID_Z_MM` mirror `SCAN_PRESETS["staging_refined"]` from `calibration/calibrate_all_safe_grid_xyz_models.py`.
- The autonomous candidate workspace bounds are derived from that refined grid: X `40..340` mm and Y `40..490` mm.
- The startup survey pose comes directly from `scripts/pick_one_place_one.py`: `X_SURVEY`, `Y_SURVEY`, and `Z_SURVEY`.
- The recovery / pre-`HOMEJ3` pose uses that same survey pose unless you explicitly override `RECOVERY_POSE_X_MM`, `RECOVERY_POSE_Y_MM`, or `RECOVERY_POSE_Z_MM`.
- `PICK_PHI_MODE` comes from `pick_one_place_one.py`. Tune it when the claw is rotated wrong for long or skinny packages.
- `USE_LOCAL_HEIGHT_AWARE_GRASP_XY`, `LOCAL_GRASP_RADIUS_MM`, and `LOCAL_GRASP_BLEND_WEIGHT` affect whether the pick shifts toward the local high/top region. If the claw grabs an edge or a tall corner, reduce the blend or radius.
- `USE_Z_GROUND_MODEL_FOR_PICK_SURFACE` uses the calibrated platform Z model plus object height. Turn this off only when the ground model is clearly worse than stereo Z.
- `REQUIRE_OVERHEAD_XY_FOR_PICK` refuses picks without overhead XY. Enable this when stereo XY is drifting.

## Place Location And Pattern

- `PLACE_SCENE_NAME` selects the bag/surface zone from `config/surface_zones.json`.
- The bag-local AABB planner chooses the next object and target from the current visible scene plus already placed boxes.
- `PLACE_REQUIRE_GRIPPER_FOOTPRINT_INSIDE_BAG` aborts placement if the estimated claw footprint would leave the bag rectangle.
- The claw footprint uses `length = PLACE_GRIPPER_FOOTPRINT_LENGTH_L_MM * sin(servo_angle)` and fixed width `PLACE_GRIPPER_FOOTPRINT_WIDTH_MM`, plus margin.
- Use `scripts/tune_bag_place_settings.py` to audit and jog these target centers at max Z without running autonomy.
- `USE_DYNAMIC_PLACE_Z_FROM_OBJECT_HEIGHT`, `PLACE_RELEASE_GAP_MM`, and `PLACE_Z_UNCERTAINTY_GAIN` control release height. If items clip the bag or another item, increase gap or uncertainty gain.
- `PLACE_Z_POLICY_MODE="negative_bin_hang"` is the experimental box policy for a bin whose bottom is below zero, for example `PLACE_NEGATIVE_BIN_PLATFORM_Z_MM=-200`.
- The negative-bin policy estimates how far the held grocery hangs below the gripper from the pick grasp Z and object bottom. It blends that with a simple object-height estimate, then applies release gap/padding.
- For first-layer testing, `PLACE_NEGATIVE_BIN_USE_EXISTING_STACK=False` prevents an old `surface_z=0` zone from being interpreted as a 200 mm stack above a `-200` bin floor.
- `PLACE_NEGATIVE_BIN_INCLUDE_RELEASE_GAP_PADDING=False` lets any object whose estimated bottom would be below robot Z 0 clamp directly to 0.
- The final Z is still clamped by `PLACE_NEGATIVE_BIN_MIN_RELEASE_Z_MM` and `DEFAULT_Z_SAFETY.MIN_PLACE_Z_MM`, then validated before motion. It will not command illegal negative Z unless the shared safety config is deliberately changed.

## Padding And AABB

- `PAD_X_MM`, `PAD_Y_MM`, and `PAD_Z_MM` inflate the placed object occupancy used by the adjacent/stack planner.
- Increase X/Y padding when objects touch or collide in the bag. Decrease it if the planner leaves too much unused space.
- `BEST_REJECT_PLACED_OVERLAP` and `BEST_PLACED_OVERLAP_MARGIN_MM` prevent re-picking objects already believed to be in the bag.

## Candidate Filters

- Workspace and platform limits reject detections outside the real pickup area.
- Reach and soft-pose checks reject candidates the robot cannot safely reach.
- Center/cluster gates are useful when the detector sees reflections or background objects.
- Volume limits catch bad pointclouds. Lower max volume if phantom pointclouds are being selected.

## Missed-Pick Watchdog

- `MISS_CHECK_ENABLED` turns the YOLO-only watchdog on/off.
- `MISS_BURST_COUNT`, `MISS_MIN_HITS`, `MISS_CHECK_PERIOD_S`, and `MISS_CHECK_TIMEOUT_S` control how much repeated evidence is needed.
- `MISS_MATCH_MAX_ROBOT_DIST_MM` is the pick-site XY tolerance. Increase it if the overhead homography is noisy; decrease it if nearby same-class objects cause false misses.
- `MISS_MATCH_MIN_IOU` and area-ratio limits compare the new YOLO detection to the original pick-site detection.
- `MISS_SCORE_THRESHOLD` is the final miss threshold. Raise it to reduce false recovery trips; lower it if obvious misses are not caught.
- RAFT is intentionally skipped here. RAFT/pointcloud runs once only after YOLO says the object is probably still at the pick site.

## Recovery Pose And J3

- `RECOVERY_POSE_X_MM`, `RECOVERY_POSE_Y_MM`, `RECOVERY_POSE_Z_MM`, and `RECOVERY_POSE_PHI_DEG` define the safe recalibration pose.
- Keep `RECOVERY_POSE_Z_MM` high and inside `DEFAULT_Z_SAFETY`.
- `RECOVERY_REHOME_J3_ENABLED` sends `HOMEJ3` after reaching the recovery pose.
- `RECOVERY_DROP_Z_BEFORE_REHOME_MM` is a checked pre-HOMEJ3 Z command. Keep it conservative; every value still passes `validate_z_command`.
- `RECOVERY_REHOME_TIMEOUT_S` should cover the full J3 homing cycle.

## Retry Grasp

- `MAX_PICK_RETRIES_PER_OBJECT` defaults to one retry.
- `RETRY_GRASP_Z_OFFSET_MM` raises the retry grasp target. Increase if the first attempt digs too low or pushes the item away.
- `RETRY_START_CLAW_EXTRA_DEG` opens the claw wider at retry start.
- `RETRY_START_CLAW_MAX_DEG` caps that wider opening.
- `RETRY_XY_SAME_THRESHOLD_MM` and `RETRY_FORCE_SAFE_SEQUENCE_IF_SAME_XY` force the safer retry path when recovery says the object is still basically at the same XY.

## Camera And Timing

- `OVERHEAD_FRESH_READ_DISCARD_FRAMES` and `OVERHEAD_FRESH_READ_DELAY_S` fight stale overhead frames. Increase discard frames if the miss check sees a ghost from before the pick.
- Survey timing still uses the normal YOLO + RAFT burst path.
- Miss timing uses a cheap YOLO-only burst, then one full recovery survey only after a confirmed miss.
- Prefetch still starts during place descent, after the miss watchdog has cleared the object.

## Bag-State Timing

- `placed_boxes.append(...)` happens only after place descent, release, and retract succeed.
- A confirmed miss cancels the pending place descent and does not update placed state.
- A failed retry also leaves bag state unchanged.
