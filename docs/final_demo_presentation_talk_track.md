# Final Demo Presentation Talk Track

Goal: give a concise software/control explanation for the MAE 162D/E final demo without overclaiming. This is written for three software slides max.

## 20-Second Opening

"Our capstone is an autonomous grocery bagging loop. The software has to answer four questions every cycle: what objects are on the platform, where are they in robot coordinates, which one should be picked next, and where can it fit in the bag without colliding with what is already there. The final demo code uses YOLO segmentation, RAFT-Stereo depth, a height-aware overhead calibration, a 3D AABB bag planner, and current-aware gripper firmware."

## Slide 1: Perception and Calibration

"The perception stack starts with a five-frame stereo burst. YOLO runs on the rectified stereo-left images, and we only keep detections that persist through the burst. That gives us a more stable object candidate than a single frame."

"For depth, we run RAFT-Stereo once on the best shared frame from that survey. We mask the dense disparity with the YOLO segment, convert it into a point cloud, and estimate object height, footprint, and a robust top surface. That gives the robot the Z and size information it needs for picking and packing."

"XY is refined with the overhead camera. The calibration bundle has height-indexed homographies at 0, 50, 150, and 200 mm, so an overhead centroid can be projected into robot coordinates at the estimated object height. The final pick XY blends stereo and overhead information instead of trusting either camera alone."

Good numbers to say:

- "The active survey uses YOLO confidence 0.35 and requires 5 out of 5 burst hits."
- "The calibration report shows overhead homography RMS between about 4 and 12 mm, and stereo XYZ RMSE about 8 mm."
- "The final run used `yolo_weights/full_data.pt` and RAFT-Stereo on CUDA."

## Slide 2: Bag-Local AABB Planner

"Once we have candidate objects, the planner does two jobs together: it chooses the next pick and computes a placement target in the bag. The active sequence is `bag_local_aabb_joint`."

"Each object is converted into a 3D axis-aligned bounding box from the live vision estimate. The bag is represented in a local frame using the calibrated `New Bag Test` scene, which is a 260 by 150 mm bag footprint with a 250 mm planning height. The planner evaluates deterministic placements on the bag floor and on top of already placed boxes."

"The scoring is intentionally conservative. It rejects placements that are outside the bag, intersect padded placed boxes, have too much overhang, lack enough support, clip the bag height, or block a vertical descent. It also does a limited future feasibility check so the next placement is not chosen completely greedily."

"One important implementation detail: the final joint selector prefers floor placements before stacking, and among floor placements it tends to fill with smaller footprint items first. That is why the live order may not look like simple largest-object-first sorting."

Safe phrasing:

- "deterministic 2.5D AABB search"
- "future-feasibility aware"
- "geometry and support based"

Avoid saying:

- "globally optimal"
- "full physics simulation"
- "class fragility was used in the wet run"

## Slide 3: Robot Execution and Recovery

"The robot layer takes the planned XY, Z, and phi and turns them into synchronized stepper commands. Python handles planar IK, wrist yaw compensation, and soft-limit checks. The Teensy firmware executes synchronized four-axis moves with `MOVESYNC_T`."

"The gripper is not just open-loop. The final pick path computes an initial servo opening from the object geometry, then uses the Teensy's `DG` dynamic grip command. The INA219 current monitor watches for a contact signature as the servo closes, so the gripper can stop based on contact rather than a fixed angle alone."

"For safety and reliability, release is gated by a missed-pick watchdog. The robot first moves to place hover, then a YOLO-only burst checks whether the same object is still at the pick site. If it is, the script cancels release and runs recovery; if it is clear, the robot descends and opens the claw."

"The latest saved wet-run evidence shows the final script using this autonomous missed-pick recovery path, writing run snapshots and a manifest, and reaching a final bag-state count of 11 placed boxes before the next object had no valid candidate and the run was quit."

## Short Closing

"The main engineering point is that the demo is not a single perception model glued to robot motion. It is a closed loop: perception estimates geometry, planning checks whether that geometry can fit, motion verifies safety before moving, and the release step has a watchdog so the system can notice a failed pick before dropping nothing into the bag."

## One-Sentence Backup Answers

"Why both stereo and overhead?"  
"Stereo gives dense depth and object height; overhead gives a cleaner top-down XY correction through the height-indexed homography."

"Why RAFT instead of only triangulating detections?"  
"RAFT gives dense disparity over the whole segmented object, so we can estimate a point cloud, height, footprint, and robust top surface instead of only a sparse center point."

"Why not just pick largest object first?"  
"The final selector is placement-aware. It filters by pickability, then asks whether each candidate can actually fit in the current bag state."

"Is the planner optimal?"  
"No. It is a deterministic constrained search with scoring and limited future feasibility, designed to be predictable and fast for the demo."

"Does it use dynamic release?"  
"The firmware supports dynamic release, but the final wet-run config uses fixed servo release. Dynamic grip is the active current-aware behavior."
