# robot.py
# Simple Python-side controller for the MAE 162 grocery bagger RRPR arm.
# Talks to Teensy firmware using:
#   MOVE j1 j2 j3 j4
#   MOVESYNC j1 j2 j3 j4
#   MOVESYNC_T j1 j2 j3 j4 time_ms
#   HOME, POS, EN, SERVO

from dataclasses import dataclass, field
import math
import time

try:
    import serial
except ImportError:
    serial = None


# ============================================================
# CONFIGURATION
# Edit this section first.
# ============================================================

@dataclass
class JointPose:
    # Geometric robot pose, not motor steps.
    q1_deg: float = 0.0      # shoulder angle
    q2_deg: float = 0.0      # elbow angle relative to link 1
    z_mm: float = 0.0        # vertical position convention: edit to match your robot
    phi_deg: float = 0.0     # claw/wrist heading in world frame


@dataclass
class RobotConfig:
    port: str = "COM4"
    baud: int = 115200

    # Link lengths in mm.
    L1_mm: float = 450.0
    L2_mm: float = 450.0

    # Steps per unit from your old code / current hardware.
    steps_per_deg_j1: float = 142.857
    steps_per_deg_j2: float = 142.857
    steps_per_mm_j3: float = 50.93
    steps_per_deg_j4: float = 4.444

    # Sign conventions from your old controller.
    # Old code used: J1 = -q1, J2 = +(q1 + q2), J3 = -z.
    j1_sign: int = -1
    j2_sign: int = 1
    j3_sign: int = -1
    j4_sign: int = 1

    # IMPORTANT:
    # After Teensy HOME finishes, the firmware sets step counters to HOME_J*.
    # This Python value tells the IK what geometric pose that switch position means.
    home_pose: JointPose = field(default_factory=lambda: JointPose(
        q1_deg=-95.0,
        q2_deg=175.0,
        z_mm=100.0,
        phi_deg=0.0,
    ))

    # Teensy step values assigned at the physical home pose.
    home_steps_j1: int = 0
    home_steps_j2: int = 0
    home_steps_j3: int = 0
    home_steps_j4: int = 0

    # IK branch. For your old script, elbow_down=False matches theta2 = +acos(...).
    elbow_down: bool = False

    # Use synchronized motion if firmware supports it.
    use_sync_motion: bool = True

    # If not None, Python sends MOVESYNC_T with this duration for synchronized moves.
    # This is useful for small 5-10 mm jogs that otherwise twitch/jitter.
    default_move_time_s: float | None = 0.45


# ============================================================
# ROBOT CLASS
# ============================================================

class Robot:
    def __init__(self, config: RobotConfig = RobotConfig(), connect: bool = True):
        self.cfg = config
        self.ser = None

        # Python's estimate of where the robot is geometrically.
        self.q_est = JointPose(
            config.home_pose.q1_deg,
            config.home_pose.q2_deg,
            config.home_pose.z_mm,
            config.home_pose.phi_deg,
        )

        if connect:
            self.connect()

    # -------------------------
    # Serial helpers
    # -------------------------

    def connect(self):
        if serial is None:
            print("pyserial is not installed. Run: pip install pyserial")
            return False

        print(f"Connecting to {self.cfg.port} at {self.cfg.baud}...")
        self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=0.5)
        time.sleep(2.0)
        self.flush()
        print("Connected.")
        return True

    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    def flush(self):
        if not self.ser:
            return
        while self.ser.in_waiting:
            line = self.ser.readline().decode(errors="replace").strip()
            if line:
                print(f"[Teensy] {line}")

    def send(self, cmd: str):
        print(f"> {cmd}")
        if self.ser:
            self.ser.write((cmd + "\n").encode())

    def read_until(self, wanted: str | tuple[str, ...], timeout_s: float = 30.0, match: str = "contains"):
        """
        Read Teensy lines until a desired response appears.

        match="contains"  -> success if wanted text is anywhere in the line.
        match="exact"     -> success only if the full line exactly equals wanted.
        match="startswith"-> success if the line starts with wanted.

        The exact mode matters for HOME because the Teensy prints both:
            HOMED_AXIS J1_COUPLED
            HOMED_AXIS J2
            HOMED
        and Python must wait for the final standalone HOMED.
        """
        if not self.ser:
            return True  # dry-run mode

        if isinstance(wanted, str):
            wanted_tuple = (wanted,)
        else:
            wanted_tuple = wanted

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue

            print(f"[Teensy] {line}")

            if match == "exact":
                if any(line == w for w in wanted_tuple):
                    return True
            elif match == "startswith":
                if any(line.startswith(w) for w in wanted_tuple):
                    return True
            else:
                if any(w in line for w in wanted_tuple):
                    return True

            if line.startswith("ERR"):
                return False

        print(f"Timed out waiting for: {wanted}")
        return False

    # -------------------------
    # Teensy commands
    # -------------------------

    def enable(self, state: bool = True):
        self.send(f"EN {1 if state else 0}")
        return self.read_until("ENABLED" if state else "DISABLED", 3.0)

    def init_drivers(self):
        self.send("INITDRIVERS")
        return self.read_until("DRIVERS_OK", 3.0)

    def home(self):
        """
        Runs the real Teensy HOME routine, then tells Python:
        'Assume the robot is now at cfg.home_pose.'
        """
        self.send("HOME")
        ok = self.read_until("HOMED", 120.0, match="exact")
        if ok:
            self.q_est = JointPose(
                self.cfg.home_pose.q1_deg,
                self.cfg.home_pose.q2_deg,
                self.cfg.home_pose.z_mm,
                self.cfg.home_pose.phi_deg,
            )
        return ok

    def assume_homed(self):
        """
        Do NOT move the robot.
        Use this only when the physical robot is already sitting at the known home pose.
        """
        self.q_est = JointPose(
            self.cfg.home_pose.q1_deg,
            self.cfg.home_pose.q2_deg,
            self.cfg.home_pose.z_mm,
            self.cfg.home_pose.phi_deg,
        )

    def zero(self):
        self.send("ZERO")
        ok = self.read_until("ZEROED", 3.0)
        if ok:
            self.q_est = JointPose(0.0, 0.0, 0.0, 0.0)
        return ok

    def servo(self, angle_deg: int):
        self.send(f"SERVO {angle_deg}")
        return self.read_until("SERVO OK", 3.0)

    def pos_steps(self):
        self.send("POS")
        if not self.ser:
            return None
        deadline = time.time() + 3.0
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            print(f"[Teensy] {line}")
            if line.startswith("POS"):
                parts = line.split()
                if len(parts) == 5:
                    return tuple(int(v) for v in parts[1:])
            if line.startswith("ERR"):
                return None
        return None

    def sync_estimate_from_teensy_steps(self):
        """
        Do NOT move the robot.
        Reads Teensy POS and reconstructs Python's estimated geometric pose.

        This lets you restart the Python script without rehoming, as long as:
          1) the Teensy has NOT been reset since the last real HOME/ZERO,
          2) the motors have not skipped/slipped,
          3) cfg.home_pose is still the same calibration.
        """
        steps = self.pos_steps()
        if steps is None:
            print("Could not read POS. Estimate unchanged.")
            return False

        self.q_est = self.steps_to_joints(*steps)
        self.print_estimate()
        return True

    # -------------------------
    # Kinematics
    # -------------------------

    def fk(self, q: JointPose | None = None):
        """Forward kinematics for the planar RR part."""
        if q is None:
            q = self.q_est

        q1 = math.radians(q.q1_deg)
        q2 = math.radians(q.q2_deg)

        x = self.cfg.L1_mm * math.cos(q1) + self.cfg.L2_mm * math.cos(q1 + q2)
        y = self.cfg.L1_mm * math.sin(q1) + self.cfg.L2_mm * math.sin(q1 + q2)
        return x, y, q.z_mm, q.phi_deg

    def ik(self, x_mm: float, y_mm: float, z_mm: float | None = None, phi_deg: float | None = None):
        """Inverse kinematics for target Cartesian pose. Returns JointPose or None."""
        if z_mm is None:
            z_mm = self.q_est.z_mm
        if phi_deg is None:
            phi_deg = self.q_est.phi_deg

        L1 = self.cfg.L1_mm
        L2 = self.cfg.L2_mm
        r2 = x_mm*x_mm + y_mm*y_mm

        cos_q2 = (r2 - L1*L1 - L2*L2) / (2.0 * L1 * L2)
        if cos_q2 < -1.0 or cos_q2 > 1.0:
            print("Target is outside the 2-link reachable workspace. Not moving.")
            return None

        q2 = math.acos(cos_q2)
        if self.cfg.elbow_down:
            q2 = -q2

        q1 = math.atan2(y_mm, x_mm) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))

        return JointPose(
            q1_deg=math.degrees(q1),
            q2_deg=math.degrees(q2),
            z_mm=z_mm,
            phi_deg=phi_deg,
        )

    # -------------------------
    # Joint pose <-> motor steps
    # -------------------------

    def motor_coords_from_joints(self, q: JointPose):
        """
        Converts geometric joints to motor coordinates.

        Critical coupling convention:
            J1 motor coordinate = q1
            J2 motor coordinate = q1 + q2
        """
        m1_deg = q.q1_deg
        m2_deg = q.q1_deg + q.q2_deg
        m3_mm = q.z_mm
        m4_deg = q.phi_deg - (q.q1_deg + q.q2_deg)
        return m1_deg, m2_deg, m3_mm, m4_deg

    def joints_from_motor_coords(self, m1_deg: float, m2_deg: float, m3_mm: float, m4_deg: float):
        """
        Inverse of motor_coords_from_joints().
        """
        q1_deg = m1_deg
        q2_deg = m2_deg - m1_deg
        z_mm = m3_mm
        phi_deg = m4_deg + m2_deg
        return JointPose(q1_deg, q2_deg, z_mm, phi_deg)

    def joints_to_steps(self, q: JointPose):
        """
        Converts desired geometric joints to absolute Teensy step targets.
        Uses cfg.home_pose as the geometric offset for the firmware's HOME step counters.
        """
        m = self.motor_coords_from_joints(q)
        mh = self.motor_coords_from_joints(self.cfg.home_pose)

        j1 = self.cfg.home_steps_j1 + round(self.cfg.j1_sign * (m[0] - mh[0]) * self.cfg.steps_per_deg_j1)
        j2 = self.cfg.home_steps_j2 + round(self.cfg.j2_sign * (m[1] - mh[1]) * self.cfg.steps_per_deg_j2)
        j3 = self.cfg.home_steps_j3 + round(self.cfg.j3_sign * (m[2] - mh[2]) * self.cfg.steps_per_mm_j3)
        j4 = self.cfg.home_steps_j4 + round(self.cfg.j4_sign * (m[3] - mh[3]) * self.cfg.steps_per_deg_j4)

        return int(j1), int(j2), int(j3), int(j4)

    def steps_to_joints(self, j1: int, j2: int, j3: int, j4: int):
        """
        Converts Teensy absolute step counters back to Python's estimated geometric joints.
        This is what allows re-attaching to the current pose without rehoming.
        """
        mh = self.motor_coords_from_joints(self.cfg.home_pose)

        m1 = mh[0] + (j1 - self.cfg.home_steps_j1) / (self.cfg.j1_sign * self.cfg.steps_per_deg_j1)
        m2 = mh[1] + (j2 - self.cfg.home_steps_j2) / (self.cfg.j2_sign * self.cfg.steps_per_deg_j2)
        m3 = mh[2] + (j3 - self.cfg.home_steps_j3) / (self.cfg.j3_sign * self.cfg.steps_per_mm_j3)
        m4 = mh[3] + (j4 - self.cfg.home_steps_j4) / (self.cfg.j4_sign * self.cfg.steps_per_deg_j4)

        return self.joints_from_motor_coords(m1, m2, m3, m4)

    # -------------------------
    # Motion API
    # -------------------------

    def move_joints(self, q1_deg=None, q2_deg=None, z_mm=None, phi_deg=None, move_time_s: float | None = None):
        """Absolute joint-space move. Unspecified values stay where Python thinks they are."""
        q = JointPose(
            self.q_est.q1_deg if q1_deg is None else q1_deg,
            self.q_est.q2_deg if q2_deg is None else q2_deg,
            self.q_est.z_mm if z_mm is None else z_mm,
            self.q_est.phi_deg if phi_deg is None else phi_deg,
        )
        steps = self.joints_to_steps(q)
        ok = self.move_steps(*steps, move_time_s=move_time_s)
        if ok:
            self.q_est = q
        return ok

    def move_cartesian(self, x_mm=None, y_mm=None, z_mm=None, phi_deg=None, move_time_s: float | None = None):
        """Absolute Cartesian move using IK."""
        x0, y0, z0, phi0 = self.fk()
        x = x0 if x_mm is None else x_mm
        y = y0 if y_mm is None else y_mm
        z = z0 if z_mm is None else z_mm
        phi = phi0 if phi_deg is None else phi_deg

        q = self.ik(x, y, z, phi)
        if q is None:
            return False

        steps = self.joints_to_steps(q)
        ok = self.move_steps(*steps, move_time_s=move_time_s)
        if ok:
            self.q_est = q
        return ok

    def move_steps(self, j1: int, j2: int, j3: int, j4: int, move_time_s: float | None = None):
        """
        Move to absolute Teensy step targets.

        move_time_s:
          None + use_sync_motion=True: uses default_move_time_s if set, else MOVESYNC.
          number: sends MOVESYNC_T with the requested target time.
        """
        if move_time_s is None:
            move_time_s = self.cfg.default_move_time_s

        if self.cfg.use_sync_motion:
            if move_time_s is not None:
                time_ms = max(1, int(round(move_time_s * 1000.0)))
                self.send(f"MOVESYNC_T {j1} {j2} {j3} {j4} {time_ms}")
            else:
                self.send(f"MOVESYNC {j1} {j2} {j3} {j4}")
        else:
            self.send(f"MOVE {j1} {j2} {j3} {j4}")

        if not self.read_until(("MOVING", "MOVING_SYNC"), 5.0, match="startswith"):
            return False
        return self.read_until("DONE", 90.0, match="exact")

    def jog(self, dx=0.0, dy=0.0, dz=0.0, dphi=0.0, mode="jog", move_time_s: float | None = None):
        """
        Relative Cartesian jog.
        """
        x, y, z, phi = self.fk()
        return self.move_cartesian(
            x_mm=x + dx,
            y_mm=y + dy,
            z_mm=z + dz,
            phi_deg=phi + dphi,
            move_time_s=move_time_s,
        )

    # Friendly aliases.
    relative_move = jog
    relativeMove = jog
    relativeMoveJog = jog

    def print_estimate(self):
        x, y, z, phi = self.fk()
        print("Estimated joint pose:")
        print(f"  q1={self.q_est.q1_deg:.2f} deg, q2={self.q_est.q2_deg:.2f} deg, z={self.q_est.z_mm:.2f} mm, phi={self.q_est.phi_deg:.2f} deg")
        print("Estimated Cartesian pose:")
        print(f"  x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm, phi={phi:.2f} deg")
        print("Commanded steps for this estimate:")
        print(f"  {self.joints_to_steps(self.q_est)}")
