# robot.py
# Simple Python-side controller for the MAE 162 grocery bagger RRPR arm.
# Talks to Teensy firmware using:
#   MOVE j1 j2 j3 j4
#   MOVESYNC j1 j2 j3 j4
#   MOVESYNC_T j1 j2 j3 j4 time_ms
#   HOME, POS, EN, SERVO

from dataclasses import dataclass, field
import math
import re
import time
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None

try:
    from config.soft_limits import SoftLimitConfig, default_config, plan_safe_path
except ImportError:
    SoftLimitConfig = None
    default_config = None
    plan_safe_path = None


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

    x_survey_mm: float = 200.0
    y_survey_mm: float = 200.0
    z_survey_mm: float = 50.0
    phi_survey_deg: float = 0.0

    # ------------------------------------------------------------
    # Ground-plane / vertical reference convention
    # ------------------------------------------------------------
    # Measured geometry, in mm:
    #   robot origin height above ground plane = 575 mm
    #   L1 plane to L2 / J3 reference plane vertical offset = 55 mm
    #   when robot z_mm/J3 = 0, EE plane is 400 mm below the L2/J3 reference
    #
    # Therefore:
    #   EE_ground_height_mm = 575 - 55 - (400 - z_robot_mm)
    #                       = 120 + z_robot_mm
    #
    # Inverse:
    #   z_robot_mm = EE_ground_height_mm - 120
    ground_origin_height_mm: float = 575.0
    l1_to_l2_height_drop_mm: float = 55.0
    j3_zero_to_ee_drop_mm: float = 400.0

    # Teensy step values assigned at the physical home pose.
    # J3 is special in this repo's firmware: after HOME, the controller leaves
    # the prismatic axis at the lifted pre-home height rather than resetting it
    # to mechanical-bottom step zero. If home_steps_j3 is omitted, derive it
    # from home_pose.z_mm so Python FK and direct Teensy J3 mm stay aligned.
    home_steps_j1: int = 0
    home_steps_j2: int = 0
    home_steps_j3: int | None = None
    home_steps_j4: int = 0

    # IK branch. For your old script, elbow_down=False matches theta2 = +acos(...).
    elbow_down: bool = False

    # Use synchronized motion if firmware supports it.
    use_sync_motion: bool = True

    # If not None, Python sends MOVESYNC_T with this duration for synchronized moves.
    # This is useful for small 5-10 mm jogs that otherwise twitch/jitter.
    default_move_time_s: float | None = 0.45

    # Optional software keep-out zones. Disabled by default so old scripts behave the same.
    # To enable in a script:
    #   cfg = RobotConfig(..., soft_limits_enabled=True, soft_limits_path="config/soft_limits_config.json")
    soft_limits_enabled: bool = True
    soft_limits_path: str | None = "config/soft_limits_config.json"
    soft_limits_verbose: bool = True
    soft_limits_strict: bool = True

    def __post_init__(self):
        if self.home_steps_j3 is None:
            self.home_steps_j3 = int(round(
                float(self.j3_sign) * float(self.home_pose.z_mm) * float(self.steps_per_mm_j3)
            ))


@dataclass
class DynamicFirmwareResult:
    ok: bool
    z_empirical_mm: float | None = None
    servo_empirical_deg: float | None = None
    raw_lines: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ZCommandTrace:
    target_robot_z_mm: float
    current_fk_z_mm: float
    current_q_est_z_mm: float
    current_est_j3_steps: int
    current_est_j3_mm: float
    current_est_direct_j3_mm: float
    target_est_j3_steps: int
    target_est_j3_mm: float
    target_direct_j3_mm: float
    actual_steps: tuple[int, int, int, int] | None = None
    actual_j3_mm: float | None = None
    actual_direct_j3_mm: float | None = None
    inferred_robot_to_direct_offset_mm: float | None = None
    warnings: list[str] = field(default_factory=list)


def parse_dynamic_state_lines(lines: list[str]) -> DynamicFirmwareResult:
    z_empirical_mm: float | None = None
    servo_empirical_deg: float | None = None

    for raw in lines:
        line = str(raw).strip()
        z_match = re.search(r"z_empirical\s*=\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if z_match is not None:
            try:
                z_empirical_mm = float(z_match.group(1))
            except ValueError:
                pass

        servo_match = re.search(r"servo_empirical\s*=\s*([-+]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
        if servo_match is not None:
            try:
                servo_empirical_deg = float(servo_match.group(1))
            except ValueError:
                pass

    ok = (z_empirical_mm is not None) or (servo_empirical_deg is not None)
    return DynamicFirmwareResult(
        ok=ok,
        z_empirical_mm=z_empirical_mm,
        servo_empirical_deg=servo_empirical_deg,
        raw_lines=[str(line) for line in lines],
        error=None if ok else "dynamic_state_not_found",
    )


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

        self.soft_cfg = None
        if self.cfg.soft_limits_enabled:
            self.load_soft_limits(self.cfg.soft_limits_path)

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

    def dynset(self, key: str, value, timeout_s: float = 3.0) -> bool:
        if isinstance(value, bool):
            value_str = "1" if value else "0"
        elif isinstance(value, int):
            value_str = str(value)
        elif isinstance(value, float):
            value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        self.send(f"dynset {key} {value_str}")
        ok = self.read_until("dynset OK", timeout_s=float(timeout_s))

        # The firmware prints the full dynamic-settings block after acknowledging
        # the dynset command. Drain that chatter so the next command starts clean.
        if self.ser:
            time.sleep(0.05)
            self.flush()
        return ok

    def _read_lines_until_done(self, timeout_s: float = 30.0):
        if not self.ser:
            return True, [], None

        raw_lines: list[str] = []
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            raw_lines.append(line)
            print(f"[Teensy] {line}")
            if line.startswith("ERR"):
                return False, raw_lines, line
            if line == "DONE":
                return True, raw_lines, None
        return False, raw_lines, "timeout"

    def set_servo_fractional(self, angle_deg: float) -> bool:
        self.send(f"servo = {float(angle_deg):.3f}")
        if not self.ser:
            return True

        deadline = time.time() + 3.0
        saw_ack = False
        saw_err = False
        while time.time() < deadline:
            line = self.ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            print(f"[Teensy] {line}")
            if line.startswith("ERR"):
                saw_err = True
                break
            if ("DONE" in line) or ("# servo ->" in line):
                saw_ack = True
                break
        return saw_ack or not saw_err

    def show_dynamic_state(self) -> DynamicFirmwareResult:
        self.send("SHOW")
        if not self.ser:
            return DynamicFirmwareResult(ok=True, raw_lines=[])

        # The firmware prints a short burst of diagnostic lines after SHOW.
        # Stop once the stream goes idle instead of waiting a fixed multi-second timeout.
        idle_deadline = time.time() + 0.2
        deadline = time.time() + 0.75
        raw_lines: list[str] = []
        while time.time() < deadline:
            if self.ser.in_waiting:
                line = self.ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                raw_lines.append(line)
                print(f"[Teensy] {line}")
                idle_deadline = time.time() + 0.2
                if line.startswith("ERR"):
                    return DynamicFirmwareResult(ok=False, raw_lines=raw_lines, error=line)
                continue

            if raw_lines and time.time() >= idle_deadline:
                break
            time.sleep(0.01)

        result = parse_dynamic_state_lines(raw_lines)
        result.raw_lines = raw_lines
        return result

    def dynamic_lower(
        self,
        z_start_mm,
        deriv_thresh_ma,
        n_steps,
        servo_deg=None,
        signed_only=False,
        timeout_s=90,
    ) -> DynamicFirmwareResult:
        cmd = (
            f"DL {float(z_start_mm):.3f} {float(deriv_thresh_ma):.3f} "
            f"{int(n_steps)}"
        )
        if servo_deg is not None:
            cmd += f" {float(servo_deg):.3f} {1 if signed_only else 0}"
        self.send(cmd)
        ok, raw_lines, error = self._read_lines_until_done(timeout_s=float(timeout_s))
        if not ok:
            return DynamicFirmwareResult(ok=False, raw_lines=raw_lines, error=error)

        state = self.show_dynamic_state()
        state.ok = state.ok and ok
        state.raw_lines = raw_lines + state.raw_lines
        state.error = state.error if state.error is not None else error
        return state

    def dynamic_lower_robot_z(
        self,
        robot_z_mm,
        deriv_thresh_ma,
        n_steps,
        servo_deg=None,
        signed_only=False,
        timeout_s=90,
    ) -> DynamicFirmwareResult:
        cmd = (
            f"DLR {float(robot_z_mm):.3f} {float(deriv_thresh_ma):.3f} "
            f"{int(n_steps)}"
        )
        if servo_deg is not None:
            cmd += f" {float(servo_deg):.3f} {1 if signed_only else 0}"
        self.send(cmd)
        ok, raw_lines, error = self._read_lines_until_done(timeout_s=float(timeout_s))
        if not ok:
            return DynamicFirmwareResult(ok=False, raw_lines=raw_lines, error=error)

        state = self.show_dynamic_state()
        state.ok = state.ok and ok
        state.raw_lines = raw_lines + state.raw_lines
        state.error = state.error if state.error is not None else error
        return state

    def dynamic_grip(
        self,
        angle_start_deg,
        deriv_thresh_ma,
        n_steps,
        signed_only=False,
        timeout_s=45,
    ) -> DynamicFirmwareResult:
        cmd = (
            f"DG {float(angle_start_deg):.3f} {float(deriv_thresh_ma):.3f} "
            f"{int(n_steps)} {1 if signed_only else 0}"
        )
        self.send(cmd)
        ok, raw_lines, error = self._read_lines_until_done(timeout_s=float(timeout_s))
        if not ok:
            return DynamicFirmwareResult(ok=False, raw_lines=raw_lines, error=error)

        state = self.show_dynamic_state()
        state.ok = state.ok and ok
        state.raw_lines = raw_lines + state.raw_lines
        state.error = state.error if state.error is not None else error
        return state

    def teensy_direct_j3_mm_from_steps(self, j3_steps: int) -> float:
        return float(j3_steps) / (float(self.cfg.j3_sign) * float(self.cfg.steps_per_mm_j3))

    def robot_z_to_teensy_direct_j3_mm(self, robot_z_mm: float) -> float:
        home_direct_j3_mm = self.teensy_direct_j3_mm_from_steps(int(self.cfg.home_steps_j3))
        return float(robot_z_mm) - float(self.cfg.home_pose.z_mm) + float(home_direct_j3_mm)

    def teensy_direct_j3_mm_to_robot_z(self, direct_j3_mm: float) -> float:
        home_direct_j3_mm = self.teensy_direct_j3_mm_from_steps(int(self.cfg.home_steps_j3))
        return float(direct_j3_mm) - float(home_direct_j3_mm) + float(self.cfg.home_pose.z_mm)

    def infer_robot_to_direct_z_offset_mm(self, include_actual_pos: bool = True) -> float:
        direct_j3_mm: float | None = None
        if include_actual_pos:
            steps = self.pos_steps()
            if steps is not None:
                direct_j3_mm = self.teensy_direct_j3_mm_from_steps(int(steps[2]))

        if direct_j3_mm is None:
            est_steps = self.joints_to_steps(self.q_est)
            direct_j3_mm = self.teensy_direct_j3_mm_from_steps(int(est_steps[2]))

        _, _, z_fk, _ = self.fk()
        return float(z_fk) - float(direct_j3_mm)

    def z_command_trace(self, target_z_mm: float, include_actual_pos: bool = False) -> ZCommandTrace:
        x_fk, y_fk, z_fk, phi_fk = self.fk()
        current_steps = self.joints_to_steps(self.q_est)
        target_q = JointPose(
            q1_deg=self.q_est.q1_deg,
            q2_deg=self.q_est.q2_deg,
            z_mm=float(target_z_mm),
            phi_deg=self.q_est.phi_deg,
        )
        target_steps = self.joints_to_steps(target_q)

        actual_steps = None
        actual_j3_mm = None
        actual_direct_j3_mm = None
        warnings: list[str] = []
        if include_actual_pos:
            actual_steps = self.pos_steps()
            if actual_steps is not None:
                actual_j3_mm = float(self.steps_to_joints(*actual_steps).z_mm)
                actual_direct_j3_mm = self.teensy_direct_j3_mm_from_steps(int(actual_steps[2]))
                if abs(actual_j3_mm - float(z_fk)) > 2.0:
                    warnings.append("actual_pos_vs_fk_z_mismatch")
            else:
                warnings.append("actual_pos_unavailable")

        inferred_offset = None
        if actual_direct_j3_mm is not None:
            inferred_offset = float(z_fk) - float(actual_direct_j3_mm)

        return ZCommandTrace(
            target_robot_z_mm=float(target_z_mm),
            current_fk_z_mm=float(z_fk),
            current_q_est_z_mm=float(self.q_est.z_mm),
            current_est_j3_steps=int(current_steps[2]),
            current_est_j3_mm=float(self.steps_to_joints(0, 0, current_steps[2], 0).z_mm),
            current_est_direct_j3_mm=self.teensy_direct_j3_mm_from_steps(int(current_steps[2])),
            target_est_j3_steps=int(target_steps[2]),
            target_est_j3_mm=float(self.steps_to_joints(0, 0, target_steps[2], 0).z_mm),
            target_direct_j3_mm=self.robot_z_to_teensy_direct_j3_mm(float(target_z_mm)),
            actual_steps=actual_steps,
            actual_j3_mm=actual_j3_mm,
            actual_direct_j3_mm=actual_direct_j3_mm,
            inferred_robot_to_direct_offset_mm=inferred_offset,
            warnings=warnings,
        )

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
    # Ground-plane / Z conversion helpers
    # -------------------------

    def robot_z_to_ground_height_mm(self, z_mm: float | None = None) -> float:
        """
        Convert robot z_mm/J3 coordinate to physical end-effector height above the ground plane.

        Using the measured convention:
            h_ground = origin_height - L1_to_L2_drop - (J3_zero_drop - z_robot)

        With your current measured values:
            h_ground = 575 - 55 - (400 - z_robot) = 120 + z_robot
        """
        if z_mm is None:
            z_mm = self.q_est.z_mm
        return (
            float(self.cfg.ground_origin_height_mm)
            - float(self.cfg.l1_to_l2_height_drop_mm)
            - (float(self.cfg.j3_zero_to_ee_drop_mm) - float(z_mm))
        )

    def ground_height_to_robot_z_mm(self, ground_height_mm: float) -> float:
        """
        Convert physical height above the ground plane to robot z_mm/J3 coordinate.

        Inverse of robot_z_to_ground_height_mm().
        With your current measured values:
            z_robot = h_ground - 120
        """
        return (
            float(ground_height_mm)
            - float(self.cfg.ground_origin_height_mm)
            + float(self.cfg.l1_to_l2_height_drop_mm)
            + float(self.cfg.j3_zero_to_ee_drop_mm)
        )

    def object_ground_height_to_lookup_z_mm(self, object_ground_height_mm: float) -> float:
        """
        Convert an object/tag height above ground into the z value used for H(z).

        The overhead homography lookup was calibrated with the EE tag at robot z_mm.
        So if a target tag is on a surface/object at a known ground height, convert
        that physical ground height into the equivalent robot z_mm layer.
        """
        return self.ground_height_to_robot_z_mm(object_ground_height_mm)

    def hover_z_for_ground_object_mm(self, object_ground_height_mm: float, clearance_mm: float = 35.0) -> float:
        """
        Robot z_mm needed for the EE to hover clearance_mm above an object whose top/tag
        height is object_ground_height_mm above the ground plane.
        """
        return self.ground_height_to_robot_z_mm(float(object_ground_height_mm) + float(clearance_mm))

    def print_ground_reference(self):
        """Print current Z in both robot and ground-plane coordinates."""
        _, _, z, _ = self.fk()
        print("Ground-plane Z reference:")
        print(f"  robot z_mm/J3 = {z:.2f} mm")
        print(f"  EE ground height = {self.robot_z_to_ground_height_mm(z):.2f} mm")
        print("  formula: h_ground = origin - L1_to_L2_drop - (J3_zero_drop - z_robot)")
        print(
            f"           h_ground = {self.cfg.ground_origin_height_mm:.1f} "
            f"- {self.cfg.l1_to_l2_height_drop_mm:.1f} "
            f"- ({self.cfg.j3_zero_to_ee_drop_mm:.1f} - z_robot)"
        )

    # -------------------------
    # Soft-limit helpers
    # -------------------------

    def load_soft_limits(self, path: str | None = None):
        """Load software keep-out boxes from JSON.

        This does not move the robot. It only enables a Cartesian safety filter
        for move_cartesian() and jog(). Existing low-level move_steps() remains
        unfiltered because it has no reliable Cartesian path information.
        """
        if SoftLimitConfig is None:
            print("soft_limits.py could not be imported. Soft limits disabled.")
            self.soft_cfg = None
            self.cfg.soft_limits_enabled = False
            return False

        if path is None:
            path = self.cfg.soft_limits_path

        if path is None:
            self.soft_cfg = default_config() if default_config is not None else None
            self.cfg.soft_limits_enabled = self.soft_cfg is not None
            return self.cfg.soft_limits_enabled

        p = Path(path)
        if p.exists():
            self.soft_cfg = SoftLimitConfig.load_json(p)
            self.cfg.soft_limits_path = str(p)
            self.cfg.soft_limits_enabled = True
            if self.cfg.soft_limits_verbose:
                print(f"[SOFT LIMITS] Loaded {len(self.soft_cfg.boxes)} forbidden box(es) from {p.resolve()}")
            return True

        print(f"[SOFT LIMITS] Config not found: {p.resolve()}. Soft limits disabled.")
        self.soft_cfg = None
        self.cfg.soft_limits_enabled = False
        return False

    def set_soft_limits_enabled(self, enabled: bool = True):
        """Enable/disable already-loaded soft limits."""
        if enabled and self.soft_cfg is None:
            return self.load_soft_limits(self.cfg.soft_limits_path)
        self.cfg.soft_limits_enabled = bool(enabled)
        print(f"[SOFT LIMITS] {'ENABLED' if enabled else 'DISABLED'}")
        return True

    def current_xyz(self):
        x, y, z, _ = self.fk()
        return (float(x), float(y), float(z))

    def check_cartesian_pose_safe(self, x_mm=None, y_mm=None, z_mm=None):
        """Return (ok, reason) for one Cartesian pose using the loaded soft-limit config."""
        if not self.cfg.soft_limits_enabled or self.soft_cfg is None:
            return True, "soft limits disabled"
        x0, y0, z0, _ = self.fk()
        p = (
            float(x0 if x_mm is None else x_mm),
            float(y0 if y_mm is None else y_mm),
            float(z0 if z_mm is None else z_mm),
        )
        from config.soft_limits import is_pose_safe
        return is_pose_safe(p, self.soft_cfg)

    def max_safe_z_at_xy(self, x_mm: float, y_mm: float) -> float | None:
        """Return the highest allowed Z at this XY, or None if no box covers this XY."""
        if not self.cfg.soft_limits_enabled or self.soft_cfg is None:
            return None
        limits = []
        for box in self.soft_cfg.boxes:
            be = box.expanded()
            if be.xmin <= x_mm <= be.xmax and be.ymin <= y_mm <= be.ymax:
                limits.append(be.z_min)
        if not limits:
            return None
        # Safe means z must be below the first active z threshold.
        return min(limits) - 1.0

    def _soft_pose_reason(self, x: float, y: float, z: float) -> str:
        zmax = self.max_safe_z_at_xy(x, y)
        if zmax is None:
            return f"({x:.1f}, {y:.1f}, {z:.1f})"
        return f"({x:.1f}, {y:.1f}, {z:.1f}); at this XY, use z <= {zmax:.1f} mm"

    def _step_segment_is_safe(self, target_steps: tuple[int, int, int, int], samples: int | None = None):
        """Check the ACTUAL firmware-like move path by interpolating motor steps.

        This is stricter than checking a straight Cartesian line. MOVESYNC_T moves
        motor coordinates toward target step counts, so the end-effector XY can bow
        through a keep-out region even if the Cartesian chord looked safe.
        """
        if not self.cfg.soft_limits_enabled or self.soft_cfg is None:
            return True, "soft limits disabled"

        start_steps = self.joints_to_steps(self.q_est)
        end_steps = tuple(int(v) for v in target_steps)
        max_delta = max(abs(e - a) for a, e in zip(start_steps, end_steps))
        if samples is None:
            # A coarse but safe-ish sample density. Step counts are large, so cap it.
            samples = int(max(25, min(250, max_delta // 75 + 2)))

        from config.soft_limits import is_pose_safe
        for i in range(samples + 1):
            t = i / samples
            js = tuple(round(a + (e - a) * t) for a, e in zip(start_steps, end_steps))
            q = self.steps_to_joints(*js)
            x, y, z, _ = self.fk(q)
            ok, reason = is_pose_safe((float(x), float(y), float(z)), self.soft_cfg)
            if not ok:
                return False, f"motor-step path unsafe at sample {i}/{samples}: FK=({x:.1f}, {y:.1f}, {z:.1f}); {reason}"
        return True, "motor-step path safe"

    def plan_cartesian_path(self, x_mm=None, y_mm=None, z_mm=None):
        """Plan a soft-limit-safe Cartesian path from current XYZ to target XYZ."""
        x0, y0, z0, _ = self.fk()
        start = (float(x0), float(y0), float(z0))
        target = (
            float(x0 if x_mm is None else x_mm),
            float(y0 if y_mm is None else y_mm),
            float(z0 if z_mm is None else z_mm),
        )
        if not self.cfg.soft_limits_enabled or self.soft_cfg is None:
            return True, [start, target], "soft limits disabled"
        if plan_safe_path is None:
            return False, [start], "soft_limits.plan_safe_path unavailable"

        # Fail closed if the requested final pose is inside a forbidden raised region.
        ok_pose, pose_reason = self.check_cartesian_pose_safe(target[0], target[1], target[2])
        if not ok_pose:
            return False, [start], "target unsafe: " + self._soft_pose_reason(target[0], target[1], target[2]) + " | " + pose_reason

        return plan_safe_path(start, target, self.soft_cfg)

    def _move_cartesian_raw(self, x_mm=None, y_mm=None, z_mm=None, phi_deg=None, move_time_s: float | None = None):
        """Unfiltered Cartesian move. Use internally after soft-limit planning."""
        x0, y0, z0, phi0 = self.fk()
        x = x0 if x_mm is None else x_mm
        y = y0 if y_mm is None else y_mm
        z = z0 if z_mm is None else z_mm
        phi = phi0 if phi_deg is None else phi_deg

        q = self.ik(x, y, z, phi)
        if q is None:
            return False

        steps = self.joints_to_steps(q)
        ok_path, why = self._step_segment_is_safe(steps)
        if not ok_path:
            print(f"[SOFT LIMITS] BLOCKED actual motor path to ({x:.1f}, {y:.1f}, {z:.1f}): {why}")
            return False
        ok = self.move_steps(*steps, move_time_s=move_time_s)
        if ok:
            self.q_est = q
        return ok

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
        """Absolute Cartesian move using IK, optionally filtered by soft limits.

        If soft limits are enabled, this samples the straight-line Cartesian path.
        If needed, it inserts simple detour waypoints around forbidden XY boxes.
        """
        x0, y0, z0, phi0 = self.fk()
        x = x0 if x_mm is None else x_mm
        y = y0 if y_mm is None else y_mm
        z = z0 if z_mm is None else z_mm
        phi = phi0 if phi_deg is None else phi_deg

        ok, path, reason = self.plan_cartesian_path(x, y, z)
        if not ok:
            print(f"[SOFT LIMITS] BLOCKED move_cartesian to ({x:.1f}, {y:.1f}, {z:.1f}): {reason}")
            return False

        if self.cfg.soft_limits_enabled and self.soft_cfg is not None and self.cfg.soft_limits_verbose:
            print(f"[SOFT LIMITS] PLAN: {reason}")
            for i, p in enumerate(path[1:], start=1):
                print(f"  waypoint {i}/{len(path)-1}: x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f}")

        # Follow all waypoints. Keep the requested phi for every waypoint.
        for wx, wy, wz in path[1:]:
            if not self._move_cartesian_raw(wx, wy, wz, phi, move_time_s=move_time_s):
                return False
        return True

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
        try:
            print(f"  EE ground height={self.robot_z_to_ground_height_mm(z):.2f} mm")
        except Exception:
            pass
        print("Commanded steps for this estimate:")
        print(f"  {self.joints_to_steps(self.q_est)}")
