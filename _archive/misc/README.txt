Grocery_Buildup_fixed
=====================

Drop these files into your VS Code Grocery_Buildup folder, replacing the same-named files.

Files included:
- Teensy_Code.ino
- robot.py
- manual_cartesian_jog.py
- robot_camera_motion_test.py
- stereo_apriltag_viewer.py
- test_robot.py

Main fixes in this bundle:
1) robot.py now waits for the final standalone "HOMED" line, not "HOMED_AXIS ...".
   This prevents Python from sending moves while Teensy is still homing J2.

2) robot_camera_motion_test.py now checks every move result.
   If the move to (200, 200) fails, it stops instead of doing jogs from the wrong state.

3) robot_camera_motion_test.py uses live OpenCV preview / SPACE to continue, so the camera
   window does not freeze while Python waits for terminal input.

4) robot.py keeps one shared state estimate q_est.
   move_cartesian() and jog() both update/read from the same state pot.

Run order:
1) Upload Teensy_Code.ino to Teensy.
2) python manual_cartesian_jog.py      # manual validation
3) python stereo_apriltag_viewer.py    # camera/tag validation
4) python robot_camera_motion_test.py  # robot-camera motion test
