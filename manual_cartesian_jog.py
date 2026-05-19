# manual_cartesian_jog.py
# Simple keyboard Cartesian jog controller for the grocery bagger robot.
#
# Controls:
#   Arrow keys: jog X/Y
#   [ / ]     : jog Z down/up, depending on your z sign convention
#   , / .     : rotate wrist phi -/+ deg
#   o / c     : open / close claw servo
#   p         : print estimated pose
#   h         : run HOME and reset Python estimate to home_pose
#   e         : enable motors and re-send driver current settings on Teensy
#   d         : disable motors
#   q / ESC   : quit
#
# This uses only the Python standard library + your robot.py.
# On Windows it uses msvcrt for immediate key reads.

import time
import msvcrt

from robot import Robot
from robot_config import (
    DEFAULT_TRAVEL_Z_MM,
    HOME_Z_MM,
    LOW_Z_MM,
    ROBOT_CONFIG,
    print_startup_config,
)
from camera_config import OVERHEAD_INDEX, STEREO_INDEX


# ============================================================
# EDIT THIS SECTION FIRST
# ============================================================

# Jog sizes. Start small.
XY_JOG_MM = 1.0
Z_JOG_MM = 5.0
PHI_JOG_DEG = 5.0

# Servo angles. Change these to match your claw.
CLAW_OPEN_DEG = 20
CLAW_CLOSED_DEG = 90

# Optional startup behavior.
DO_HOME_ON_START = False


# ============================================================
# KEYBOARD HELPERS
# ============================================================

def get_key():
    """Read one key from Windows terminal. Handles arrow keys."""
    ch = msvcrt.getch()

    # Arrow/function keys come as two-byte sequences.
    if ch in (b"\x00", b"\xe0"):
        ch2 = msvcrt.getch()
        if ch2 == b"H":
            return "UP"
        if ch2 == b"P":
            return "DOWN"
        if ch2 == b"K":
            return "LEFT"
        if ch2 == b"M":
            return "RIGHT"
        return None

    try:
        return ch.decode("utf-8").lower()
    except UnicodeDecodeError:
        return None


def print_controls():
    print("\nManual Cartesian Jog Controller")
    print("================================")
    print("Arrow keys : jog XY")
    print("[ / ]      : jog Z - / +")
    print(", / .      : rotate wrist phi - / +")
    print("o / c      : open / close claw")
    print("p          : print estimated pose")
    print("h          : home robot")
    print("e / d      : enable / disable motors")
    print("q or ESC   : quit")
    print(f"\nJog sizes: XY={XY_JOG_MM} mm, Z={Z_JOG_MM} mm, phi={PHI_JOG_DEG} deg")
    print("Press keys directly. No Enter needed.\n")


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print_startup_config("manual_cartesian_jog.py", OVERHEAD_INDEX, STEREO_INDEX)
    robot = Robot(ROBOT_CONFIG)

    try:
        print_controls()

        robot.enable(True)
        time.sleep(0.2)

        if DO_HOME_ON_START:
            robot.home()

        robot.print_estimate()

        while True:
            key = get_key()
            if key is None:
                continue

            # Quit
            if key in ("q", "\x1b"):
                print("Quitting...")
                break

            # Enable / disable / home
            elif key == "e":
                robot.enable(True)
            elif key == "d":
                robot.enable(False)
            elif key == "h":
                robot.home()
                robot.print_estimate()

            # Print pose
            elif key == "p":
                robot.print_estimate()

            # Claw
            elif key == "o":
                robot.servo(CLAW_OPEN_DEG)
            elif key == "c":
                robot.servo(CLAW_CLOSED_DEG)

            # XY jogs
            elif key == "UP":
                robot.jog(dy=+XY_JOG_MM)
                robot.print_estimate()
            elif key == "DOWN":
                robot.jog(dy=-XY_JOG_MM)
                robot.print_estimate()
            elif key == "RIGHT":
                robot.jog(dx=+XY_JOG_MM)
                robot.print_estimate()
            elif key == "LEFT":
                robot.jog(dx=-XY_JOG_MM)
                robot.print_estimate()

            # Z jogs
            elif key == "[":
                robot.jog(dz=-Z_JOG_MM)
                robot.print_estimate()
            elif key == "]":
                robot.jog(dz=+Z_JOG_MM)
                robot.print_estimate()

            # Wrist rotation
            elif key == ",":
                robot.jog(dphi=-PHI_JOG_DEG)
                robot.print_estimate()
            elif key == ".":
                robot.jog(dphi=+PHI_JOG_DEG)
                robot.print_estimate()

            else:
                # Ignore unknown keys quietly.
                pass

    finally:
        # Do not forcibly disable motors here unless you want the arm to relax on exit.
        # robot.enable(False)
        robot.close()


if __name__ == "__main__":
    main()
