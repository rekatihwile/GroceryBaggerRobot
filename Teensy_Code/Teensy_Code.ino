// ============================================================
//  arm_firmware_homing_coupled_sync.ino
//  Teensy 4.1 firmware for RRPR robotic manipulator
//
//  Build-up homing version:
//    - LIMITS reads J1/J2 switches
//    - HOMEJ1 homes J1 while synchronously compensating J2
//    - HOMEJ2 homes J2 normally
//    - HOMEJ3 drops J3 to its mechanical bottom and lifts to clearance
//    - HOME homes J1-with-J2-compensation, then J2
//    - Every J1/J2 homing command first lifts J3 by a fixed distance
//
//  J3 drop behavior:
//    - J3 stays enabled during the drop
//    - Its TMC2209 run current is temporarily lowered (J3_SOFT_DROP_CURRENT_MA)
//      so gravity overcomes the holding torque and the carriage slips
//      smoothly to the mechanical bottom instead of free-falling
//
//  Serial protocol (115200 baud, newline-terminated):
//
//    PC -> Teensy:
//      MOVE <j1> <j2> <j3> <j4>       Move all joints to absolute step targets
//      MOVESYNC <j1> <j2> <j3> <j4>   Move with synchronized target time
//      MOVESYNC_T <j1> <j2> <j3> <j4> <time_ms>  Same, but force a minimum target time
//      INITDRIVERS                  Re-send TMC2209 current/microstep settings
//      HOME                         Real homing: lift J3, then J1 with J2 compensation, then J2
//      HOMEJ1                       Lift J3, then home J1 while compensating J2
//      HOMEJ2                       Lift J3, then home J2
//      HOMEJ3                       Soft-drop J3 to bottom, then lift to clearance
//      ZERO                         Set current position as step zero for all joints
//      POS                          Request current step positions
//      LIMITS                       Request current limit switch states
//      EN <1|0>                     Enable / disable stepper drivers
//      SERVO <angle>                Move claw servo
//
//    Teensy -> PC:
//      READY
//      MOVING
//      MOVING_SYNC <seconds>
//      DONE
//      HOMING
//      HOMED
//      HOMED_AXIS <J1|J2|J3>
//      LIFTING_J3 / LIFTED_J3
//      DROPPING_J3 / DROPPED_J3
//      POS <j1> <j2> <j3> <j4>
//      LIMITS J1 <0|1> J2 <0|1>
//      ENABLED / DISABLED
//      ERR <message>
// ============================================================

#include <AccelStepper.h>
#include <TMCStepper.h>
#include <Servo.h>
#include <math.h>

// ----------------------------------------
// HARDWARE CONSTANTS
// ----------------------------------------

#define R_SENSE 0.11f
#define UART_PORT Serial1

// ----------------------------------------
// PIN ASSIGNMENTS
// ----------------------------------------

#define J1_STEP 2
#define J1_DIR 3
#define J1_EN 4

#define J2_STEP 5
#define J2_DIR 6
#define J2_EN 7

#define J3_STEP 8
#define J3_DIR 9
#define J3_EN 10

#define J4_STEP 11
#define J4_DIR 12
#define J4_EN 13

#define SERVO_PIN 14

// ----------------------------------------
// LIMIT SWITCH PINS
// ----------------------------------------
// Assumed wiring:
//   Teensy pin ---- switch ---- GND
//   pinMode = INPUT_PULLUP
//   unpressed = HIGH
//   pressed   = LOW

#define J1_LIMIT_PIN 15
#define J2_LIMIT_PIN 16

#define LIMIT_PRESSED LOW

// ----------------------------------------
// HOME POSITIONS  (step counts)
// ----------------------------------------
// These are the motor-coordinate values assigned to the physical switch pose.
// They are NOT automatically geometric q1/q2 zero.

const long HOME_J1 = 0;
const long HOME_J2 = 0;
const long HOME_J3 = 0;
const long HOME_J4 = 0;

// ----------------------------------------
// HOMING SETTINGS
// ----------------------------------------
// Direction convention:
//   +1 = positive step direction moves toward switch
//   -1 = negative step direction moves toward switch
//
// If an axis moves away from the switch during homing, flip that sign.

const int J1_HOME_DIR = 1;
const int J2_HOME_DIR = 1;

// Very slow build-up speeds.
// If your old code used about 142.857 steps/deg,
// then 600 steps/s ≈ 4.2 deg/s, and 120 steps/s ≈ 0.84 deg/s.

const float HOME_FAST_SPEED = 600.0;
const float HOME_SLOW_SPEED = 120.0;
const float HOME_ACCEL = 800.0;

const long HOME_BACKOFF_STEPS = 400;
const unsigned long HOME_TIMEOUT_MS = 90000;

// ----------------------------------------
// J3 PRE-HOMING LIFT / SOFT DROP
// ----------------------------------------
// Before every J1/J2 homing sequence, J3 is dropped to its mechanical
// bottom and then lifted to a fixed clearance height to clear the end
// effector from the work surface.
//
//   J3_STEPS_PER_MM        = TUNE to your actual J3 calibration.
//   J3_PRE_HOME_LIFT_MM    = lift distance in mm above mechanical bottom.
//   J3_LIFT_DIR            = +1 or -1. Flip if it lifts the wrong way.

const float J3_PULLEY_DIAMETER_MM = 10.0f;
const float J3_STEPS_PER_REV = 1600;         // ~51428.57
const float J3_MM_PER_REV = (float)M_PI * J3_PULLEY_DIAMETER_MM; // ~31.416

const float J3_STEPS_PER_MM = J3_STEPS_PER_REV/J3_MM_PER_REV;
const float J3_PRE_HOME_LIFT_MM = 100.0;
const int J3_LIFT_DIR = -1;

// J3 TMC2209 currents (mA):
//   J3_RUN_CURRENT_MA       = normal holding / motion current.
//   J3_SOFT_DROP_CURRENT_MA = lowered current used during the drop.
//
// Tuning J3_SOFT_DROP_CURRENT_MA:
//   too HIGH -> holding torque > gravity, carriage will not move.
//   too LOW  -> motor near-disabled, carriage free-falls (the old behavior).
// Start around 200-300 mA and adjust until the descent looks controlled.

const int J3_RUN_CURRENT_MA = 1400;
const int J3_SOFT_DROP_CURRENT_MA = 1600;

// Robust pre-home behavior for J3:
//   1) Lower J3 holding current so it slides smoothly to the mechanical bottom.
//   2) Restore normal current and assign that bottom to J3_DROPPED_STEPS.
//   3) Move J3 to one absolute lifted clearance height.
// This avoids stacking another +100 mm lift every time HOME is called.
// WARNING: use this only if the drop is mechanically safe and has a hard stop.

const bool J3_DROP_BEFORE_LIFT = true;
const long J3_DROPPED_STEPS = 0;
const unsigned long J3_KICK_DISABLE_MS = 100; // brief full-disable to break mid-travel stiction
const unsigned long J3_DROP_MS = 2000;        // remaining soft-current descent time
const unsigned long J3_LIFT_TIMEOUT_MS = 15000;

// ----------------------------------------
// J1/J2 COUPLING COMPENSATION DURING J1 HOMING
// ----------------------------------------
// While J1 homes, J2 must move simultaneously or the physical elbow angle q2
// changes because the J2 belt is routed through the moving J1 structure.
//
// This constant means:
//   J2_speed = J1_speed * J2_COMP_PER_J1_STEP_DURING_J1_HOME
//
// If the elbow motion gets worse during HOMEJ1, flip this sign.
// If the ratio is not exactly 1:1, tune the magnitude.

const float J2_COMP_PER_J1_STEP_DURING_J1_HOME = -1.0;

// ----------------------------------------
// NORMAL MOTION SETTINGS
// ----------------------------------------
// These are the per-axis limits used by normal MOVE and as the maximum
// allowed limits for MOVESYNC. Tune these in one place.

const float J1_NORMAL_MAX_SPEED = 25000.0;
const float J1_NORMAL_ACCEL = 4000.0;

const float J2_NORMAL_MAX_SPEED = 25000.0;
const float J2_NORMAL_ACCEL = 4000.0;

const float J3_NORMAL_MAX_SPEED = 12000.0;
const float J3_NORMAL_ACCEL = 6000.0;

const float J4_NORMAL_MAX_SPEED = 40000.0;
const float J4_NORMAL_ACCEL = 20000.0;

// Minimum nonzero synced move time. Prevents goofy math on tiny moves.
const float SYNC_MIN_TIME_SEC = 0.20;

// ----------------------------------------
// STEPPER OBJECTS
// ----------------------------------------

AccelStepper J1(AccelStepper::DRIVER, J1_STEP, J1_DIR);
AccelStepper J2(AccelStepper::DRIVER, J2_STEP, J2_DIR);
AccelStepper J3(AccelStepper::DRIVER, J3_STEP, J3_DIR);
AccelStepper J4(AccelStepper::DRIVER, J4_STEP, J4_DIR);

// ----------------------------------------
// DRIVER OBJECTS
// ----------------------------------------

TMC2209Stepper driver1(&UART_PORT, R_SENSE, 0);
TMC2209Stepper driver2(&UART_PORT, R_SENSE, 1);
TMC2209Stepper driver3(&UART_PORT, R_SENSE, 2);
TMC2209Stepper driver4(&UART_PORT, R_SENSE, 3);

// ----------------------------------------
// SERVO
// ----------------------------------------

Servo clawServo;
int servoAngle = 40;

// ----------------------------------------
// STATE
// ----------------------------------------

bool enabled = true;
bool moving = false;

// ========================================
// SETUP
// ========================================

void setup()
{

    Serial.begin(115200);
    UART_PORT.begin(115200);

    pinMode(J1_EN, OUTPUT);
    pinMode(J2_EN, OUTPUT);
    pinMode(J3_EN, OUTPUT);
    pinMode(J4_EN, OUTPUT);

    pinMode(J1_LIMIT_PIN, INPUT_PULLUP);
    pinMode(J2_LIMIT_PIN, INPUT_PULLUP);

    enableMotors(true);

    setupAllDrivers();

    restoreNormalMotionSettings();

    clawServo.attach(SERVO_PIN);
    clawServo.write(servoAngle);

    // Unknown until homed, but initialize counters safely.
    J1.setCurrentPosition(0);
    J2.setCurrentPosition(0);
    J3.setCurrentPosition(0);
    J4.setCurrentPosition(0);

    Serial.println("READY");
}

// ========================================
// MAIN LOOP
// ========================================

void loop()
{

    J1.run();
    J2.run();
    J3.run();
    J4.run();

    if (moving)
    {
        if (!J1.isRunning() && !J2.isRunning() &&
            !J3.isRunning() && !J4.isRunning())
        {
            moving = false;
            Serial.println("DONE");
        }
    }

    readCommands();
}

// ========================================
// DRIVER SETUP
// ========================================

void setupDriver(TMC2209Stepper &driver, int current_mA)
{
    driver.begin();
    driver.rms_current(current_mA);
    driver.microsteps(8);
    driver.en_spreadCycle(true);
}

void setupAllDrivers()
{
    // Re-send these whenever motor power may have been cycled.
    // This fixes the common case where the Teensy boots before the 24V supply.
    setupDriver(driver1, 1400);
    setupDriver(driver2, 1700);
    setupDriver(driver3, J3_RUN_CURRENT_MA);
    setupDriver(driver4, 2000);
}

// ========================================
// ENABLE / DISABLE
// ========================================

void enableMotors(bool state)
{
    enabled = state;

    // EN pins are active-LOW on TMC2209
    digitalWrite(J1_EN, !state);
    digitalWrite(J2_EN, !state);
    digitalWrite(J3_EN, !state);
    digitalWrite(J4_EN, !state);

    if (state)
    {
        delay(50);
        setupAllDrivers();
    }
}

void enableJ3Motor(bool state)
{
    // EN pins are active-LOW on TMC2209.
    // This intentionally affects ONLY the vertical J3 axis.
    // Note: soft-drop homing no longer toggles this. Kept available as an
    // escape hatch for diagnostics or manual full-disable.
    digitalWrite(J3_EN, !state);

    if (state)
    {
        delay(50);
        setupDriver(driver3, J3_RUN_CURRENT_MA);
        J3.setMaxSpeed(J3_NORMAL_MAX_SPEED);
        J3.setAcceleration(J3_NORMAL_ACCEL);
    }
}

// ========================================
// MOTION SETTINGS
// ========================================

void restoreNormalMotionSettings()
{
    J1.setMaxSpeed(J1_NORMAL_MAX_SPEED);
    J1.setAcceleration(J1_NORMAL_ACCEL);

    J2.setMaxSpeed(J2_NORMAL_MAX_SPEED);
    J2.setAcceleration(J2_NORMAL_ACCEL);

    J3.setMaxSpeed(J3_NORMAL_MAX_SPEED);
    J3.setAcceleration(J3_NORMAL_ACCEL);

    J4.setMaxSpeed(J4_NORMAL_MAX_SPEED);
    J4.setAcceleration(J4_NORMAL_ACCEL);
}

void setHomingMotionSettings()
{
    J1.setMaxSpeed(HOME_FAST_SPEED);
    J1.setAcceleration(HOME_ACCEL);

    J2.setMaxSpeed(HOME_FAST_SPEED);
    J2.setAcceleration(HOME_ACCEL);
}

// This is important after runSpeed() homing.
// It clears stale moveTo() targets so an axis does not jump afterward.
void holdAxisHere(AccelStepper &axis)
{
    long p = axis.currentPosition();
    axis.setCurrentPosition(p); // AccelStepper also clears target to current position
    axis.setSpeed(0);
}

void holdAllAxesHere()
{
    holdAxisHere(J1);
    holdAxisHere(J2);
    holdAxisHere(J3);
    holdAxisHere(J4);
}

// ========================================
// LIMIT SWITCH HELPERS
// ========================================

bool limitPressed(uint8_t pin)
{
    if (digitalRead(pin) != LIMIT_PRESSED)
    {
        return false;
    }

    delay(5); // crude debounce, okay for slow homing

    return digitalRead(pin) == LIMIT_PRESSED;
}

void printLimits()
{
    Serial.print("LIMITS J1 ");
    Serial.print(limitPressed(J1_LIMIT_PIN) ? 1 : 0);
    Serial.print(" J2 ");
    Serial.println(limitPressed(J2_LIMIT_PIN) ? 1 : 0);
}

// ========================================
// HOMING HELPERS
// ========================================

bool timedOut(unsigned long startTime)
{
    return (millis() - startTime) > HOME_TIMEOUT_MS;
}

// ------------------------------------------------------------
// Pre-homing J3 lift.
// Raises the end effector clear of the work surface before any J1/J2 motion.
// Relative move using normal J3 motion settings.
// ------------------------------------------------------------

bool dropJ3ToMechanicalBottom()
{

    Serial.println("DROPPING_J3");

    moving = false;
    holdAxisHere(J3);

    // Phase 1: brief FULL disable to break stiction in the middle of travel.
    // Cogging + screw/belt friction mid-travel can be enough that lowered
    // current alone will not start motion -- the rotor stays magnetically
    // "stuck" to its current pole. Cutting the field entirely lets gravity
    // nudge the carriage into motion and build a bit of momentum.
    //
    // If the kick drops the carriage too far / too fast: shorten J3_KICK_DISABLE_MS.
    // If the carriage still won't start moving mid-travel: lengthen it.
    digitalWrite(J3_EN, HIGH); // active-LOW: HIGH = disabled
    delay(J3_KICK_DISABLE_MS);

    // Phase 2: re-enable with lowered current to brake the now-moving carriage
    // smoothly to the mechanical bottom. Microstep / spreadCycle register
    // settings persist across the EN toggle, so we only need to re-send
    // the run-current register.
    digitalWrite(J3_EN, LOW); // active-LOW: LOW = enabled
    delay(20);                // brief driver power-up settle
    driver3.rms_current(J3_SOFT_DROP_CURRENT_MA);

    delay(J3_DROP_MS);

    // Restore normal holding current and accept the bottomed-out position
    // as our J3 reference. AccelStepper's step counter is stale (the rotor
    // physically slipped) so we re-zero it here.
    driver3.rms_current(J3_RUN_CURRENT_MA);
    J3.setCurrentPosition(J3_DROPPED_STEPS);
    holdAxisHere(J3);

    Serial.println("DROPPED_J3");
    return true;
}

bool liftJ3ToPreHomeHeight()
{

    Serial.println("LIFTING_J3");

    restoreNormalMotionSettings();
    holdAxisHere(J3);

    // Absolute lifted target measured from the dropped/bottom position.
    // Because this is absolute, repeated HOME commands do not keep lifting higher.
    long liftSteps = (long)lround(
        (float)J3_LIFT_DIR * J3_PRE_HOME_LIFT_MM * J3_STEPS_PER_MM);

    long target = J3_DROPPED_STEPS + liftSteps;

    if (J3.currentPosition() == target)
    {
        Serial.println("LIFTED_J3");
        return true;
    }

    J3.moveTo(target);

    unsigned long startTime = millis();

    while (J3.distanceToGo() != 0)
    {
        J3.run();

        if ((millis() - startTime) > J3_LIFT_TIMEOUT_MS)
        {
            Serial.println("ERR J3 lift timeout");
            holdAxisHere(J3);
            return false;
        }
    }

    holdAxisHere(J3);
    Serial.println("LIFTED_J3");
    return true;
}

bool liftJ3ForHoming()
{

    // Robust mode: always re-reference J3 by letting it settle to the bottom
    // before lifting to one fixed clearance height.
    if (J3_DROP_BEFORE_LIFT)
    {
        if (!dropJ3ToMechanicalBottom())
        {
            return false;
        }
    }

    return liftJ3ToPreHomeHeight();
}

// ------------------------------------------------------------
// Plain single-axis homing helpers, used for J2.
// ------------------------------------------------------------

bool moveAxisAwayUntilReleased(AccelStepper &axis,
                               uint8_t limitPin,
                               int homeDir,
                               const char *axisName)
{

    unsigned long startTime = millis();

    while (limitPressed(limitPin))
    {
        axis.setSpeed(-homeDir * HOME_SLOW_SPEED);
        axis.runSpeed();

        if (timedOut(startTime))
        {
            Serial.print("ERR ");
            Serial.print(axisName);
            Serial.println(" switch stayed pressed during release");
            holdAxisHere(axis);
            return false;
        }
    }

    holdAxisHere(axis);
    return true;
}

bool moveAxisUntilPressed(AccelStepper &axis,
                          uint8_t limitPin,
                          int homeDir,
                          float speed,
                          const char *axisName)
{

    unsigned long startTime = millis();

    while (!limitPressed(limitPin))
    {
        axis.setSpeed(homeDir * speed);
        axis.runSpeed();

        if (timedOut(startTime))
        {
            Serial.print("ERR ");
            Serial.print(axisName);
            Serial.println(" homing timeout before switch press");
            holdAxisHere(axis);
            return false;
        }
    }

    holdAxisHere(axis);
    return true;
}

bool backOffAxisFixedDistance(AccelStepper &axis,
                              int homeDir,
                              const char *axisName)
{

    holdAxisHere(axis);

    long target = axis.currentPosition() - homeDir * HOME_BACKOFF_STEPS;
    axis.moveTo(target);

    unsigned long startTime = millis();

    while (axis.distanceToGo() != 0)
    {
        axis.run();

        if (timedOut(startTime))
        {
            Serial.print("ERR ");
            Serial.print(axisName);
            Serial.println(" backoff timeout");
            holdAxisHere(axis);
            return false;
        }
    }

    holdAxisHere(axis);
    return true;
}

bool homeAxis(AccelStepper &axis,
              uint8_t limitPin,
              int homeDir,
              long assignedHomePosition,
              const char *axisName)
{

    Serial.print("HOMING_AXIS ");
    Serial.println(axisName);

    moving = false;
    setHomingMotionSettings();

    if (!moveAxisAwayUntilReleased(axis, limitPin, homeDir, axisName))
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    if (!moveAxisUntilPressed(axis, limitPin, homeDir, HOME_FAST_SPEED, axisName))
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    if (!backOffAxisFixedDistance(axis, homeDir, axisName))
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    if (!moveAxisUntilPressed(axis, limitPin, homeDir, HOME_SLOW_SPEED, axisName))
    {
        restoreNormalMotionSettings();
        return false;
    }

    axis.setCurrentPosition(assignedHomePosition);
    axis.setSpeed(0);

    Serial.print("HOMED_AXIS ");
    Serial.println(axisName);

    restoreNormalMotionSettings();
    return true;
}

// ------------------------------------------------------------
// Coupled J1 homing helpers.
// These physically run J1 and J2 at the same time.
// ------------------------------------------------------------

float j2CompSpeedFromJ1Speed(float j1Speed)
{
    return j1Speed * J2_COMP_PER_J1_STEP_DURING_J1_HOME;
}

long j2CompDeltaFromJ1Delta(long j1Delta)
{
    return lround((float)j1Delta * J2_COMP_PER_J1_STEP_DURING_J1_HOME);
}

bool moveJ1CoupledUntilLimitState(bool wantPressed,
                                  float j1Speed,
                                  const char *phaseName)
{

    unsigned long startTime = millis();

    while (limitPressed(J1_LIMIT_PIN) != wantPressed)
    {
        J1.setSpeed(j1Speed);
        J2.setSpeed(j2CompSpeedFromJ1Speed(j1Speed));

        // These calls are both made on every loop iteration.
        // That is the key difference from sequential homing.
        J1.runSpeed();
        J2.runSpeed();

        if (timedOut(startTime))
        {
            Serial.print("ERR J1_COUPLED timeout during ");
            Serial.println(phaseName);
            holdAxisHere(J1);
            holdAxisHere(J2);
            return false;
        }
    }

    holdAxisHere(J1);
    holdAxisHere(J2);
    return true;
}

bool backOffJ1CoupledFixedDistance()
{

    holdAxisHere(J1);
    holdAxisHere(J2);

    long j1Start = J1.currentPosition();
    long j2Start = J2.currentPosition();

    long j1Delta = -J1_HOME_DIR * HOME_BACKOFF_STEPS;
    long j2Delta = j2CompDeltaFromJ1Delta(j1Delta);

    J1.moveTo(j1Start + j1Delta);
    J2.moveTo(j2Start + j2Delta);

    unsigned long startTime = millis();

    while (J1.distanceToGo() != 0 || J2.distanceToGo() != 0)
    {
        J1.run();
        J2.run();

        if (timedOut(startTime))
        {
            Serial.println("ERR J1_COUPLED backoff timeout");
            holdAxisHere(J1);
            holdAxisHere(J2);
            return false;
        }
    }

    holdAxisHere(J1);
    holdAxisHere(J2);
    return true;
}

bool homeJ1Coupled()
{

    Serial.println("HOMING_AXIS J1_COUPLED");

    moving = false;
    setHomingMotionSettings();
    holdAxisHere(J1);
    holdAxisHere(J2);

    // Step 0:
    // If J1 switch is already pressed, back away from the J1 switch.
    // J2 compensates during that release motion too.
    if (!moveJ1CoupledUntilLimitState(false,
                                      -J1_HOME_DIR * HOME_SLOW_SPEED,
                                      "release"))
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    // Step 1:
    // Approach J1 switch with J2 compensation.
    if (!moveJ1CoupledUntilLimitState(true,
                                      J1_HOME_DIR * HOME_FAST_SPEED,
                                      "fast approach"))
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    // Step 2:
    // Back off a fixed distance with J2 compensation.
    if (!backOffJ1CoupledFixedDistance())
    {
        restoreNormalMotionSettings();
        return false;
    }

    delay(100);

    // Step 3:
    // Final slow approach with J2 compensation.
    if (!moveJ1CoupledUntilLimitState(true,
                                      J1_HOME_DIR * HOME_SLOW_SPEED,
                                      "slow approach"))
    {
        restoreNormalMotionSettings();
        return false;
    }

    // Step 4:
    // Assign J1 home. Do NOT assign final J2 home here; J2 was only compensating.
    // But DO clear J2's stale target so it cannot jump after this function returns.
    J1.setCurrentPosition(HOME_J1);
    J1.setSpeed(0);

    holdAxisHere(J2);

    Serial.println("HOMED_AXIS J1_COUPLED");

    restoreNormalMotionSettings();
    return true;
}

bool homeJ2Only()
{
    return homeAxis(J2, J2_LIMIT_PIN, J2_HOME_DIR, HOME_J2, "J2");
}

bool homeJ3Only()
{
    // Drop J3 to its mechanical bottom with lowered current, then lift to
    // the standard clearance height. After this, J3's step counter is
    // referenced and J3 is parked at +J3_PRE_HOME_LIFT_MM above bottom.
    if (!liftJ3ForHoming())
    {
        return false;
    }
    Serial.println("HOMED_AXIS J3");
    J3.setCurrentPosition(HOME_J3);
    return true;
}

bool homeAll()
{
    Serial.println("HOMING");

    enableMotors(true);
    moving = false;
    holdAllAxesHere();

    // Lift J3 before any J1/J2 motion.
    if (!liftJ3ForHoming())
    {
        Serial.println("ERR HOME failed on J3 lift");
        return false;
    }

    delay(100);

    // Home J1 while actively compensating J2.
    if (!homeJ1Coupled())
    {
        Serial.println("ERR HOME failed on J1_COUPLED");
        return false;
    }

    delay(300);

    // Now J1 is referenced. Home J2 normally.
    if (!homeJ2Only())
    {
        Serial.println("ERR HOME failed on J2");
        return false;
    }

    J3.setCurrentPosition(HOME_J3);
    J4.setCurrentPosition(HOME_J4);
    holdAllAxesHere();

    Serial.println("HOMED");
    return true;
}

// ========================================
// SYNCHRONIZED MOTION HELPERS
// ========================================

float estimateMoveTimeSec(long distanceSteps, float maxSpeed, float accel)
{
    float d = fabs((float)distanceSteps);

    if (d < 1.0)
    {
        return 0.0;
    }

    // Distance needed to accelerate from 0 to maxSpeed and then decelerate to 0.
    float accelDistanceTotal = (maxSpeed * maxSpeed) / accel;

    if (d <= accelDistanceTotal)
    {
        // Triangular profile: accelerate then immediately decelerate.
        return 2.0 * sqrt(d / accel);
    }

    // Trapezoidal profile: accel, cruise, decel.
    float accelTimeTotal = 2.0 * maxSpeed / accel;
    float cruiseDistance = d - accelDistanceTotal;
    float cruiseTime = cruiseDistance / maxSpeed;

    return accelTimeTotal + cruiseTime;
}

void configureAxisForTargetTime(AccelStepper &axis,
                                long target,
                                float normalMaxSpeed,
                                float normalAccel,
                                float targetTimeSec)
{

    long dSteps = labs(target - axis.currentPosition());

    if (dSteps == 0)
    {
        axis.moveTo(target);
        return;
    }

    float d = (float)dSteps;
    float T = max(targetTimeSec, SYNC_MIN_TIME_SEC);

    // First try a gentle triangular profile that exactly fits the target time.
    float triangularAccel = 4.0 * d / (T * T);
    float triangularSpeed = 2.0 * d / T;

    if (triangularAccel <= normalAccel && triangularSpeed <= normalMaxSpeed)
    {
        axis.setAcceleration(max(triangularAccel, 1.0f));
        axis.setMaxSpeed(max(triangularSpeed, 1.0f));
        axis.moveTo(target);
        return;
    }

    // Otherwise use normal acceleration and solve for the cruise speed needed
    // to finish in time T: d = v*T - v^2/a.
    float a = normalAccel;
    float discriminant = a * a * T * T - 4.0 * a * d;

    float v;
    if (discriminant > 0.0)
    {
        v = (a * T - sqrt(discriminant)) / 2.0;
    }
    else
    {
        // Should only happen because of rounding/tiny timing edge cases.
        v = normalMaxSpeed;
    }

    v = constrain(v, 1.0f, normalMaxSpeed);

    axis.setAcceleration(a);
    axis.setMaxSpeed(v);
    axis.moveTo(target);
}

void moveSyncToTimed(long s1, long s2, long s3, long s4, float requestedTimeSec)
{

    enableMotors(true);
    moving = false;

    // Each axis uses an AccelStepper triangular/trapezoidal velocity curve.
    // requestedTimeSec stretches short jogs so they do not twitch violently.
    float t1 = estimateMoveTimeSec(s1 - J1.currentPosition(), J1_NORMAL_MAX_SPEED, J1_NORMAL_ACCEL);
    float t2 = estimateMoveTimeSec(s2 - J2.currentPosition(), J2_NORMAL_MAX_SPEED, J2_NORMAL_ACCEL);
    float t3 = estimateMoveTimeSec(s3 - J3.currentPosition(), J3_NORMAL_MAX_SPEED, J3_NORMAL_ACCEL);
    float t4 = estimateMoveTimeSec(s4 - J4.currentPosition(), J4_NORMAL_MAX_SPEED, J4_NORMAL_ACCEL);

    float targetTime = max(max(t1, t2), max(t3, t4));
    targetTime = max(targetTime, SYNC_MIN_TIME_SEC);
    targetTime = max(targetTime, requestedTimeSec);

    configureAxisForTargetTime(J1, s1, J1_NORMAL_MAX_SPEED, J1_NORMAL_ACCEL, targetTime);
    configureAxisForTargetTime(J2, s2, J2_NORMAL_MAX_SPEED, J2_NORMAL_ACCEL, targetTime);
    configureAxisForTargetTime(J3, s3, J3_NORMAL_MAX_SPEED, J3_NORMAL_ACCEL, targetTime);
    configureAxisForTargetTime(J4, s4, J4_NORMAL_MAX_SPEED, J4_NORMAL_ACCEL, targetTime);

    moving = true;

    Serial.print("MOVING_SYNC ");
    Serial.println(targetTime, 3);
}

void moveSyncTo(long s1, long s2, long s3, long s4)
{
    moveSyncToTimed(s1, s2, s3, s4, 0.0);
}

void moveNormalTo(long s1, long s2, long s3, long s4)
{
    enableMotors(true);
    restoreNormalMotionSettings();

    J1.moveTo(s1);
    J2.moveTo(s2);
    J3.moveTo(s3);
    J4.moveTo(s4);

    moving = true;
    Serial.println("MOVING");
}

// ========================================
// COMMAND PARSER
// ========================================

void readCommands()
{

    if (!Serial.available())
        return;

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // ------------------------------------------
    // MOVESYNC <j1> <j2> <j3> <j4>
    //   Absolute step move with synchronized target time.
    //
    // IMPORTANT: check MOVESYNC before MOVE.
    // "MOVESYNC" starts with "MOVE", so MOVE must not be first.
    // ------------------------------------------

    if (cmd.startsWith("MOVESYNC_T"))
    {

        long s1, s2, s3, s4;
        long timeMs;
        int matched = sscanf(cmd.c_str(), "MOVESYNC_T %ld %ld %ld %ld %ld", &s1, &s2, &s3, &s4, &timeMs);

        if (matched == 5)
        {
            float requestedTimeSec = max(0.001f, ((float)timeMs) / 1000.0f);
            moveSyncToTimed(s1, s2, s3, s4, requestedTimeSec);
        }
        else
        {
            Serial.println("ERR MOVESYNC_T requires 4 integer arguments plus time_ms");
        }
    }

    else if (cmd.startsWith("MOVESYNC"))
    {

        long s1, s2, s3, s4;
        int matched = sscanf(cmd.c_str(), "MOVESYNC %ld %ld %ld %ld", &s1, &s2, &s3, &s4);

        if (matched == 4)
        {
            moveSyncTo(s1, s2, s3, s4);
        }
        else
        {
            Serial.println("ERR MOVESYNC requires 4 integer arguments");
        }
    }

    // ------------------------------------------
    // MOVE <j1> <j2> <j3> <j4>
    // ------------------------------------------

    else if (cmd.startsWith("MOVE "))
    {

        long s1, s2, s3, s4;
        int matched = sscanf(cmd.c_str(), "MOVE %ld %ld %ld %ld", &s1, &s2, &s3, &s4);

        if (matched == 4)
        {
            moveNormalTo(s1, s2, s3, s4);
        }
        else
        {
            Serial.println("ERR MOVE requires 4 integer arguments");
        }
    }

    // ------------------------------------------
    // HOME
    // ------------------------------------------

    else if (cmd == "HOME")
    {
        homeAll();
    }

    // ------------------------------------------
    // HOMEJ1
    //   Lift J3, then home J1 with synchronous J2 compensation.
    // ------------------------------------------

    else if (cmd == "HOMEJ1")
    {
        Serial.println("HOMING");
        if (liftJ3ForHoming() && homeJ1Coupled())
        {
            Serial.println("HOMED");
        }
    }

    // ------------------------------------------
    // HOMEJ2
    //   Lift J3, then home J2.
    // ------------------------------------------

    else if (cmd == "HOMEJ2")
    {
        Serial.println("HOMING");
        if (liftJ3ForHoming() && homeJ2Only())
        {
            Serial.println("HOMED");
        }
    }

    // ------------------------------------------
    // HOMEJ3
    //   Soft-drop J3 to its mechanical bottom, then lift to clearance.
    //   J3 is referenced to J3_DROPPED_STEPS at the bottom.
    // ------------------------------------------

    else if (cmd == "HOMEJ3")
    {
        Serial.println("HOMING");
        enableMotors(true);
        if (homeJ3Only())
        {
            Serial.println("HOMED");
        }
    }

    // ------------------------------------------
    // ZERO
    // ------------------------------------------

    else if (cmd.startsWith("ZERO"))
    {
        J1.setCurrentPosition(0);
        J2.setCurrentPosition(0);
        J3.setCurrentPosition(0);
        J4.setCurrentPosition(0);
        holdAllAxesHere();
        Serial.println("ZEROED");
    }

    // ------------------------------------------
    // POS
    // ------------------------------------------

    else if (cmd.startsWith("POS"))
    {
        Serial.print("POS ");
        Serial.print(J1.currentPosition());
        Serial.print(" ");
        Serial.print(J2.currentPosition());
        Serial.print(" ");
        Serial.print(J3.currentPosition());
        Serial.print(" ");
        Serial.println(J4.currentPosition());
    }

    // ------------------------------------------
    // LIMITS
    // ------------------------------------------

    else if (cmd.startsWith("LIMITS"))
    {
        printLimits();
    }

    // ------------------------------------------
    // ENABLE
    // ------------------------------------------

    else if (cmd == "ENABLE")
    {
        enableMotors(true);
        Serial.println("ENABLED");
    }

    // ------------------------------------------
    // DISABLE
    // ------------------------------------------

    else if (cmd == "DISABLE")
    {
        enableMotors(false);
        Serial.println("DISABLED");
    }

    // ------------------------------------------
    // MOTORS?
    // ------------------------------------------

    else if (cmd == "MOTORS?")
    {
        Serial.println(enabled ? "ENABLED" : "DISABLED");
    }

    // ------------------------------------------
    // EN <1|0>
    // ------------------------------------------

    else if (cmd.startsWith("EN"))
    {
        int e = cmd.substring(3).toInt();
        enableMotors(e);
        Serial.println(enabled ? "ENABLED" : "DISABLED");
    }

    // ------------------------------------------
    // INITDRIVERS
    //   Re-send current limits/microstep settings after motor power cycle.
    // ------------------------------------------

    else if (cmd == "INITDRIVERS")
    {
        setupAllDrivers();
        Serial.println("DRIVERS_OK");
    }

    // ------------------------------------------
    // SERVO <angle>
    // ------------------------------------------

    else if (cmd.startsWith("SERVO"))
    {
        int angle = constrain(cmd.substring(6).toInt(), 10, 70);
        servoAngle = angle;
        clawServo.write(servoAngle);
        Serial.println("SERVO OK");
    }

    else if (cmd.length() > 0)
    {
        Serial.print("ERR unknown command: ");
        Serial.println(cmd);
    }
}