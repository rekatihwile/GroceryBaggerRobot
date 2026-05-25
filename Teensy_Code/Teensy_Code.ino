    // ============================================================
    //  arm_firmware_dynamic.ino
    //  Teensy 4.1 firmware for RRPR robotic manipulator
    //  + INA219 servo-line current monitoring with dynamic grip
    //    and dynamic lower routines.
    //
    //  All original protocol commands from arm_firmware_homing_coupled_sync.ino
    //  are preserved. New commands:
    //
    //    HELP                                 Print this help
    //    SHOW                                 Print stored empirical values + POS
    //    STREAM ON | STREAM OFF               Idle current streaming on/off (plotter)
    //    J3 = <expr>                          Move J3 to absolute mm (e.g. J3 = 150)
    //    SERVO = <expr>                       Set servo to fractional degrees
    //    dynamiclower(z_start=<mm>, percent_rise=<pct>)
    //                                         Slow descent until current avg > trigger.
    //                                         Stores final height in z_empirical.
    //    dynamicgrip(angle_start=<deg>, percent_rise=<pct>)
    //                                         Slow close until current avg > trigger.
    //                                         Stores final angle in servo_empirical.
    //    DLR <robot_z_mm> <deriv_thresh> <N_steps> [<servo_deg> [<signed_0_1>]]
    //                                         Dynamic lower using the same robot/J3
    //                                         height frame used by Python FK/move commands.
    //    DL / DG                              Aliases with positional args, e.g.
    //                                           DL 150 5
    //                                           DG 50 10
    //
    //  Value expressions in J3=, servo=, and the dynamic args support:
    //    - bare numbers (units 'mm' / 'deg' allowed and ignored)
    //    - the symbols z_empirical, servo_empirical
    //    - a symbol +/- a number, e.g. (z_empirical + 30)
    //
    //  Cancel any dynamic op: send any line (just press Enter).
    //
    //  INA219 wiring (in series with servo +V line):
    //    SCL -> pin 19, SDA -> pin 18 (default Wire on Teensy 4.1)
    //    Vin+ from PSU, Vin- to servo +V
    //
    //  All telemetry from dynamic operations is formatted for the Arduino IDE
    //  Serial Plotter. Human-readable messages are prefixed with '# '.
    // ============================================================

    #include <AccelStepper.h>
    #include <TMCStepper.h>
    #include <Servo.h>
    #include <Wire.h>
    #include <Adafruit_INA219.h>
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
    #define J1_DIR  3
    #define J1_EN   4

    #define J2_STEP 5
    #define J2_DIR  6
    #define J2_EN   7

    #define J3_STEP 8
    #define J3_DIR  9
    #define J3_EN   10

    #define J4_STEP 11
    #define J4_DIR  12
    #define J4_EN   13

    #define SERVO_PIN 14

    #define J1_LIMIT_PIN 15
    #define J2_LIMIT_PIN 16
    #define LIMIT_PRESSED LOW



    // ----------------------------------------
    // J3 GEOMETRY / PRE-HOMING SOFT DROP
    // ----------------------------------------

    const float J3_PULLEY_DIAMETER_MM = 10.0f;
    const float J3_STEPS_PER_REV      = 200*8;
    const float J3_MM_PER_REV         = (float)M_PI * J3_PULLEY_DIAMETER_MM;
    const float J3_STEPS_PER_MM       = J3_STEPS_PER_REV / J3_MM_PER_REV;   // ~50.93
    const float J3_PRE_HOME_LIFT_MM   = 100.0;
    const int   J3_LIFT_DIR           = -1;
    const int   J3_DESCEND_DIR        = -J3_LIFT_DIR;                       // +1 when lift dir = -1

    const int J3_RUN_CURRENT_MA        = 1400;
    const int J3_SOFT_DROP_CURRENT_MA  = 1600;

    const bool J3_DROP_BEFORE_LIFT = true;
    const long J3_DROPPED_STEPS    = 0;
    const unsigned long J3_KICK_DISABLE_MS = 100;
    const unsigned long J3_DROP_MS         = 2000;
    const unsigned long J3_LIFT_TIMEOUT_MS = 15000;
    // ----------------------------------------
    // HOME POSITIONS  (step counts)
    // ----------------------------------------

    const long HOME_J1 = 0;
    const long HOME_J2 = 0;
    const long HOME_J3 = 100/J3_STEPS_PER_MM;
    const long HOME_J4 = 0;

    // ----------------------------------------
    // HOMING SETTINGS
    // ----------------------------------------

    const int J1_HOME_DIR = 1;
    const int J2_HOME_DIR = 1;

    const float HOME_FAST_SPEED = 600.0;
    const float HOME_SLOW_SPEED = 120.0;
    const float HOME_ACCEL      = 800.0;

    const long HOME_BACKOFF_STEPS = 400;
    const unsigned long HOME_TIMEOUT_MS = 90000;
    // ----------------------------------------
    // J1/J2 COUPLING COMPENSATION
    // ----------------------------------------

    const float J2_COMP_PER_J1_STEP_DURING_J1_HOME = -1.0;

    // ----------------------------------------
    // NORMAL MOTION SETTINGS
    // ----------------------------------------

    const float J1_NORMAL_MAX_SPEED = 40000.0;
    const float J1_NORMAL_ACCEL     = 9000.0;

    const float J2_NORMAL_MAX_SPEED = 40000.0;
    const float J2_NORMAL_ACCEL     = 9000.0;

    const float J3_NORMAL_MAX_SPEED = 12000.0;
    const float J3_NORMAL_ACCEL     = 9000.0;

    const float J4_NORMAL_MAX_SPEED = 40000.0;
    const float J4_NORMAL_ACCEL     = 20000.0;

    const float SYNC_MIN_TIME_SEC = 0.20;

    // ----------------------------------------
    // SERVO CALIBRATION (degrees <-> microseconds)
    // ----------------------------------------
    // Matches the default Servo.h mapping so write(angle) and writeMicroseconds()
    // stay consistent. If your servo uses a non-standard range, tune these.

    const int   SERVO_MIN_US    = 544;
    const int   SERVO_MAX_US    = 2400;
    const float SERVO_RANGE_DEG = 180.0f;

    const float SERVO_ANGLE_MIN_DEG = 10.0f;
    const float SERVO_ANGLE_MAX_DEG = 70.0f;

    // ----------------------------------------
    // DYNAMIC OPERATION SETTINGS  (tune to taste)
    // ----------------------------------------

    const unsigned long DYN_SAMPLE_PERIOD_MS = 10;   // 100 Hz current sampling
    const int           DYN_BASELINE_SAMPLES = 40;   // ~400 ms baseline window
    const int           DYN_FIXED_AVG_SAMPLES = 100; // first N phase-2 samples define fixed trigger reference
    const unsigned long DYN_SETTLE_MS        = 400;  // post-positioning settle before baseline
    const int           DYN_DERIV_PATTERN_NONZERO_N_MAX = 8;

    // Dynamic grip: how much the servo closes per current sample (deg).
    // At 100 Hz sampling, 0.05 deg/sample = 5 deg/sec.
    const float DYN_GRIP_DEG_PER_SAMPLE = 0.05f;
    const float DYN_GRIP_MIN_ANGLE_DEG  = 5.0f;     // safety floor before full close
    const unsigned long DYN_GRIP_TIMEOUT_MS = 20000;

    // Dynamic lower: J3 descent speed in mm/s, plus safety bounds.
    const float DYN_LOWER_SPEED_MM_S    = 50.0f;
    const float DYN_LOWER_MIN_HEIGHT_MM = 0.0f;      // mechanical bottom
    const unsigned long DYN_LOWER_TIMEOUT_MS = 60000;
    const float DEFAULT_DYN_Z_SERVO_L_PRESET_MM = 50.0f; // default geometric offset for z compensation from servo angle

    // ----------------------------------------
    // USER-TUNABLE DYNAMIC SETTINGS (runtime)
    // Can be changed with terminal command: dynset <key> <value>
    // ----------------------------------------

    // Default command parameters (used when args are omitted).
    float dyn_default_dg_start_deg         = 50.0f;
    float dyn_default_dl_start_mm          = 250.0f;
    float dyn_default_deriv_thresh_ma      = 30.0f;
    int   dyn_default_deriv_n_steps_grip   = 1;
    int   dyn_default_deriv_n_steps_lower  = 1;
    bool  dyn_default_signed_only          = false;
    float dyn_default_dl_servo_deg         = NAN;     // NAN => use current servo angle

    // Dynamic behavior tuning.
    float dyn_grip_open_offset_deg         = 5.0f;
    float dyn_grip_deg_per_sample_runtime  = 1.0f;
    float dyn_lower_speed_mm_s_runtime     = 50.0f;
    bool  dyn_grip_object_squishable       = false;
    float dyn_grip_post_contact_extra_close_deg = 0.0f;

    // Secondary pattern gate tuning.
    float dyn_deriv_nonzero_eps_ma         = 1.0f;    // |dI| <= eps treated as zero/noise
    int   dyn_deriv_pattern_nonzero_n      = 2;       // lookback over last N non-zero derivs
    float dyn_deriv_pattern_sum_grip_ma    = 1500.0f;  // grip trigger if all-positive sum over lookback exceeds this
    float dyn_deriv_pattern_sum_lower_ma   = 50.0f;  // lower trigger if all-positive sum over lookback exceeds this

    // Idle streaming sample period (slower than dynamic ops to keep monitor readable).
    const unsigned long IDLE_STREAM_PERIOD_MS = 50;  // 20 Hz

    // ----------------------------------------
    // STEPPER, DRIVER, SERVO, INA219 OBJECTS
    // ----------------------------------------

    AccelStepper J1(AccelStepper::DRIVER, J1_STEP, J1_DIR);
    AccelStepper J2(AccelStepper::DRIVER, J2_STEP, J2_DIR);
    AccelStepper J3(AccelStepper::DRIVER, J3_STEP, J3_DIR);
    AccelStepper J4(AccelStepper::DRIVER, J4_STEP, J4_DIR);

    TMC2209Stepper driver1(&UART_PORT, R_SENSE, 0);
    TMC2209Stepper driver2(&UART_PORT, R_SENSE, 1);
    TMC2209Stepper driver3(&UART_PORT, R_SENSE, 2);
    TMC2209Stepper driver4(&UART_PORT, R_SENSE, 3);

    Servo clawServo;
    Adafruit_INA219 ina219;

    // ----------------------------------------
    // STATE
    // ----------------------------------------

    bool enabled = true;
    bool moving  = false;

    float currentServoAngleDeg = 65.0f;

    float z_empirical_mm     = NAN;     // set by dynamiclower
    float servo_empirical_deg = NAN;    // set by dynamicgrip
    float dyn_z_servo_l_preset_mm = DEFAULT_DYN_Z_SERVO_L_PRESET_MM;

    bool ina219_ok   = false;
    bool stream_idle = false;            // idle continuous current stream on by default

    unsigned long lastIdleStreamMs = 0;

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
        writeServoFractional(currentServoAngleDeg);

        J1.setCurrentPosition(0);
        J2.setCurrentPosition(0);
        J3.setCurrentPosition(0);
        J4.setCurrentPosition(0);

        // INA219 on default Wire (Teensy 4.1: SDA=18, SCL=19)
        Wire.begin();
        ina219_ok = ina219.begin();
        if (!ina219_ok)
        {
            Serial.println("# WARN: INA219 not detected on I2C. Dynamic ops will refuse to run.");
        }
        // Default calibration (32V / 2A) works for typical servos.
        // For higher resolution on low current, uncomment:
        //   ina219.setCalibration_16V_400mA();

        Serial.println("READY");
        Serial.println("# Type HELP for the full command list.");
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

        if (stream_idle && ina219_ok && !moving)
        {
            unsigned long now = millis();
            if (now - lastIdleStreamMs >= IDLE_STREAM_PERIOD_MS)
            {
                lastIdleStreamMs = now;
                float c = ina219.getCurrent_mA();
                Serial.print("Current_mA:");
                Serial.println(c, 2);
            }
        }

        readCommands();
    }

    // ========================================
    // DRIVER SETUP / ENABLE
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
        setupDriver(driver1, 1400);
        setupDriver(driver2, 1700);
        setupDriver(driver3, J3_RUN_CURRENT_MA);
        setupDriver(driver4, 2000);
    }

    void enableMotors(bool state)
    {
        enabled = state;

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

    // ========================================
    // MOTION SETTINGS / HOLD
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

    void holdAxisHere(AccelStepper &axis)
    {
        long p = axis.currentPosition();
        axis.setCurrentPosition(p);
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
        if (digitalRead(pin) != LIMIT_PRESSED) return false;
        delay(5);
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
    // HOMING (original logic, unchanged)
    // ========================================

    bool timedOut(unsigned long startTime)
    {
        return (millis() - startTime) > HOME_TIMEOUT_MS;
    }

    bool dropJ3ToMechanicalBottom()
    {
        Serial.println("DROPPING_J3");

        moving = false;
        holdAxisHere(J3);

        digitalWrite(J3_EN, HIGH);
        delay(J3_KICK_DISABLE_MS);

        digitalWrite(J3_EN, LOW);
        delay(20);
        driver3.rms_current(J3_SOFT_DROP_CURRENT_MA);

        delay(J3_DROP_MS);

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
        if (J3_DROP_BEFORE_LIFT)
        {
            if (!dropJ3ToMechanicalBottom()) return false;
        }
        return liftJ3ToPreHomeHeight();
    }

    bool moveAxisAwayUntilReleased(AccelStepper &axis, uint8_t limitPin, int homeDir, const char *axisName)
    {
        unsigned long startTime = millis();
        while (limitPressed(limitPin))
        {
            axis.setSpeed(-homeDir * HOME_SLOW_SPEED);
            axis.runSpeed();
            if (timedOut(startTime))
            {
                Serial.print("ERR "); Serial.print(axisName);
                Serial.println(" switch stayed pressed during release");
                holdAxisHere(axis);
                return false;
            }
        }
        holdAxisHere(axis);
        return true;
    }

    bool moveAxisUntilPressed(AccelStepper &axis, uint8_t limitPin, int homeDir, float speed, const char *axisName)
    {
        unsigned long startTime = millis();
        while (!limitPressed(limitPin))
        {
            axis.setSpeed(homeDir * speed);
            axis.runSpeed();
            if (timedOut(startTime))
            {
                Serial.print("ERR "); Serial.print(axisName);
                Serial.println(" homing timeout before switch press");
                holdAxisHere(axis);
                return false;
            }
        }
        holdAxisHere(axis);
        return true;
    }

    bool backOffAxisFixedDistance(AccelStepper &axis, int homeDir, const char *axisName)
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
                Serial.print("ERR "); Serial.print(axisName);
                Serial.println(" backoff timeout");
                holdAxisHere(axis);
                return false;
            }
        }
        holdAxisHere(axis);
        return true;
    }

    bool homeAxis(AccelStepper &axis, uint8_t limitPin, int homeDir, long assignedHomePosition, const char *axisName)
    {
        Serial.print("HOMING_AXIS "); Serial.println(axisName);

        moving = false;
        setHomingMotionSettings();

        if (!moveAxisAwayUntilReleased(axis, limitPin, homeDir, axisName)) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!moveAxisUntilPressed(axis, limitPin, homeDir, HOME_FAST_SPEED, axisName)) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!backOffAxisFixedDistance(axis, homeDir, axisName)) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!moveAxisUntilPressed(axis, limitPin, homeDir, HOME_SLOW_SPEED, axisName)) { restoreNormalMotionSettings(); return false; }

        axis.setCurrentPosition(assignedHomePosition);
        axis.setSpeed(0);

        Serial.print("HOMED_AXIS "); Serial.println(axisName);
        restoreNormalMotionSettings();
        return true;
    }

    float j2CompSpeedFromJ1Speed(float j1Speed)  { return j1Speed * J2_COMP_PER_J1_STEP_DURING_J1_HOME; }
    long  j2CompDeltaFromJ1Delta(long j1Delta)   { return lround((float)j1Delta * J2_COMP_PER_J1_STEP_DURING_J1_HOME); }

    bool moveJ1CoupledUntilLimitState(bool wantPressed, float j1Speed, const char *phaseName)
    {
        unsigned long startTime = millis();
        while (limitPressed(J1_LIMIT_PIN) != wantPressed)
        {
            J1.setSpeed(j1Speed);
            J2.setSpeed(j2CompSpeedFromJ1Speed(j1Speed));
            J1.runSpeed();
            J2.runSpeed();
            if (timedOut(startTime))
            {
                Serial.print("ERR J1_COUPLED timeout during "); Serial.println(phaseName);
                holdAxisHere(J1); holdAxisHere(J2);
                return false;
            }
        }
        holdAxisHere(J1); holdAxisHere(J2);
        return true;
    }

    bool backOffJ1CoupledFixedDistance()
    {
        holdAxisHere(J1); holdAxisHere(J2);
        long j1Start = J1.currentPosition();
        long j2Start = J2.currentPosition();
        long j1Delta = -J1_HOME_DIR * HOME_BACKOFF_STEPS;
        long j2Delta = j2CompDeltaFromJ1Delta(j1Delta);
        J1.moveTo(j1Start + j1Delta);
        J2.moveTo(j2Start + j2Delta);
        unsigned long startTime = millis();
        while (J1.distanceToGo() != 0 || J2.distanceToGo() != 0)
        {
            J1.run(); J2.run();
            if (timedOut(startTime))
            {
                Serial.println("ERR J1_COUPLED backoff timeout");
                holdAxisHere(J1); holdAxisHere(J2);
                return false;
            }
        }
        holdAxisHere(J1); holdAxisHere(J2);
        return true;
    }

    bool homeJ1Coupled()
    {
        Serial.println("HOMING_AXIS J1_COUPLED");
        moving = false;
        setHomingMotionSettings();
        holdAxisHere(J1); holdAxisHere(J2);

        if (!moveJ1CoupledUntilLimitState(false, -J1_HOME_DIR * HOME_SLOW_SPEED, "release")) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!moveJ1CoupledUntilLimitState(true,  J1_HOME_DIR * HOME_FAST_SPEED, "fast approach")) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!backOffJ1CoupledFixedDistance()) { restoreNormalMotionSettings(); return false; }
        delay(100);
        if (!moveJ1CoupledUntilLimitState(true,  J1_HOME_DIR * HOME_SLOW_SPEED, "slow approach")) { restoreNormalMotionSettings(); return false; }

        J1.setCurrentPosition(HOME_J1);
        J1.setSpeed(0);
        holdAxisHere(J2);

        Serial.println("HOMED_AXIS J1_COUPLED");
        restoreNormalMotionSettings();
        return true;
    }

    bool homeJ2Only() { return homeAxis(J2, J2_LIMIT_PIN, J2_HOME_DIR, HOME_J2, "J2"); }

    bool homeJ3Only()
    {
        if (!liftJ3ForHoming()) return false;
        Serial.println("HOMED_AXIS J3");
        // NOTE: do NOT reset J3 position here.
        // dropJ3ToMechanicalBottom() sets position = 0 (mechanical bottom = 0 mm),
        // then liftJ3ToPreHomeHeight() moves to -5093 steps (= J3_PRE_HOME_LIFT_MM = 100 mm).
        // Resetting to HOME_J3=0 would cause J3=250mm to overshoot by the pre-home lift height.
        return true;
    }

    bool homeAll()
    {
        Serial.println("HOMING");
        enableMotors(true);
        moving = false;
        holdAllAxesHere();

        if (!liftJ3ForHoming())   { Serial.println("ERR HOME failed on J3 lift");   return false; }
        delay(100);
        if (!homeJ1Coupled())     { Serial.println("ERR HOME failed on J1_COUPLED"); return false; }
        delay(300);
        if (!homeJ2Only())        { Serial.println("ERR HOME failed on J2");        return false; }

        // J3 position is already correct after the drop+lift sequence; do not reset.
        J4.setCurrentPosition(HOME_J4);
        holdAllAxesHere();
        Serial.println("HOMED");
        return true;
    }

    // ========================================
    // SYNCHRONIZED MOTION (original logic, unchanged)
    // ========================================

    float estimateMoveTimeSec(long distanceSteps, float maxSpeed, float accel)
    {
        float d = fabs((float)distanceSteps);
        if (d < 1.0) return 0.0;
        float accelDistanceTotal = (maxSpeed * maxSpeed) / accel;
        if (d <= accelDistanceTotal) return 2.0 * sqrt(d / accel);
        float accelTimeTotal = 2.0 * maxSpeed / accel;
        float cruiseDistance = d - accelDistanceTotal;
        float cruiseTime = cruiseDistance / maxSpeed;
        return accelTimeTotal + cruiseTime;
    }

    void configureAxisForTargetTime(AccelStepper &axis, long target,
                                    float normalMaxSpeed, float normalAccel, float targetTimeSec)
    {
        long dSteps = labs(target - axis.currentPosition());
        if (dSteps == 0) { axis.moveTo(target); return; }
        float d = (float)dSteps;
        float T = max(targetTimeSec, SYNC_MIN_TIME_SEC);

        float triangularAccel = 4.0 * d / (T * T);
        float triangularSpeed = 2.0 * d / T;

        if (triangularAccel <= normalAccel && triangularSpeed <= normalMaxSpeed)
        {
            axis.setAcceleration(max(triangularAccel, 1.0f));
            axis.setMaxSpeed(max(triangularSpeed, 1.0f));
            axis.moveTo(target);
            return;
        }

        float a = normalAccel;
        float discriminant = a * a * T * T - 4.0 * a * d;
        float v = (discriminant > 0.0) ? (a * T - sqrt(discriminant)) / 2.0 : normalMaxSpeed;
        v = constrain(v, 1.0f, normalMaxSpeed);

        axis.setAcceleration(a);
        axis.setMaxSpeed(v);
        axis.moveTo(target);
    }

    void moveSyncToTimed(long s1, long s2, long s3, long s4, float requestedTimeSec)
    {
        enableMotors(true);
        moving = false;

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

    void moveSyncTo(long s1, long s2, long s3, long s4) { moveSyncToTimed(s1, s2, s3, s4, 0.0); }

    void moveNormalTo(long s1, long s2, long s3, long s4)
    {
        enableMotors(true);
        restoreNormalMotionSettings();
        J1.moveTo(s1); J2.moveTo(s2); J3.moveTo(s3); J4.moveTo(s4);
        moving = true;
        Serial.println("MOVING");
    }

    // ============================================================
    //  NEW: SERVO FRACTIONAL CONTROL
    // ============================================================

    void writeServoFractional(float angleDeg)
    {
        angleDeg = constrain(angleDeg, SERVO_ANGLE_MIN_DEG, SERVO_ANGLE_MAX_DEG);
        int us = (int)lround((float)SERVO_MIN_US +
                            (angleDeg / SERVO_RANGE_DEG) * (float)(SERVO_MAX_US - SERVO_MIN_US));
        clawServo.writeMicroseconds(us);
        currentServoAngleDeg = angleDeg;
    }

    // ============================================================
    //  NEW: J3 mm <-> steps CONVERSION + BLOCKING MOVE
    // ============================================================
    //
    //  J3 height convention: increasing mm = more lifted (higher off the floor).
    //  At the mechanical bottom (after HOMEJ3), J3 = 0 mm.
    //  With J3_LIFT_DIR = -1, lifting up = negative step count.

    long  j3MmToSteps(float mm)    { return (long)lround((float)J3_LIFT_DIR * mm * J3_STEPS_PER_MM); }
    float j3StepsToMm(long steps)  { return (float)steps / ((float)J3_LIFT_DIR * J3_STEPS_PER_MM); }

    float zEmpiricalFromContactAndServo(float contact_mm, float servo_angle_deg)
    {
        float angle_rad = servo_angle_deg * ((float)M_PI / 180.0f);
        return contact_mm + dyn_z_servo_l_preset_mm * cos(angle_rad);
    }

    float robotZToDynamicLowerMm(float robot_z_mm)
    {
        // Python and the firmware should use the same J3-mm height convention:
        // larger mm = higher, 0 mm = mechanical bottom. Python handles the
        // firmware's non-zero post-HOME step count via its home_steps_j3 config.
        return robot_z_mm;
    }

    bool moveJ3ToMmBlocking(float mm)
    {
        enableMotors(true);
        restoreNormalMotionSettings();
        holdAxisHere(J3);

        long target = j3MmToSteps(mm);
        if (J3.currentPosition() == target) return true;

        J3.moveTo(target);

        unsigned long t0 = millis();
        while (J3.distanceToGo() != 0)
        {
            J3.run();
            if ((millis() - t0) > J3_LIFT_TIMEOUT_MS)
            {
                Serial.println("# ERR J3 move timeout");
                holdAxisHere(J3);
                return false;
            }
        }
        holdAxisHere(J3);
        return true;
    }

    // ============================================================
    //  NEW: CANCEL HANDLING + TELEMETRY
    // ============================================================

    bool drainSerialAsCancel()
    {
        if (!Serial.available()) return false;
        // Eat the whole pending line so it doesn't get re-parsed as a command.
        while (Serial.available())
        {
            char c = Serial.read();
            if (c == '\n') break;
            delayMicroseconds(200);
        }
        return true;
    }

    void plotterLine4(float c, float avg, float base, float trig)
    {
        float pctRise = 0.0f;
        if (fabs(base) > 1e-6f)
        {
            pctRise = ((c - base) / base) * 100.0f;
        }

        Serial.print("Current_mA:"); Serial.print(c,    2);
        Serial.print(",Avg:");        Serial.print(avg,  2);
        Serial.print(",Base:");       Serial.print(base, 2);
        Serial.print(",Trig:");       Serial.print(trig, 2);
        Serial.print(",PctRise:");    Serial.println(pctRise, 2);
    }

    void plotterDeriv(float c, float deriv, float threshold, int consec)
    {
        Serial.print("Current_mA:"); Serial.print(c,       2);
        Serial.print(",Deriv:");     Serial.print(deriv,    2);
        Serial.print(",Thresh:");    Serial.print(threshold, 2);
        Serial.print(",Consec:");    Serial.println(consec);
    }

    // ============================================================
    //  NEW: EXPRESSION PARSER
    //  Supports: <number>[mm|deg], z_empirical, servo_empirical,
    //            <symbol> +|- <number>
    // ============================================================

    bool resolveTerm(String s, float& out)
    {
        s.trim();
        if (s.length() == 0) return false;

        if (s == "z_empirical")
        {
            if (isnan(z_empirical_mm)) { Serial.println("# ERR z_empirical not set yet"); return false; }
            out = z_empirical_mm; return true;
        }
        if (s == "servo_empirical")
        {
            if (isnan(servo_empirical_deg)) { Serial.println("# ERR servo_empirical not set yet"); return false; }
            out = servo_empirical_deg; return true;
        }

        char c = s.charAt(0);
        if (isDigit(c) || c == '-' || c == '+' || c == '.')
        {
            out = s.toFloat();
            return true;
        }

        Serial.print("# ERR could not resolve term '"); Serial.print(s); Serial.println("'");
        return false;
    }

    bool parseValueExpression(String s, float& out)
    {
        s.toLowerCase();
        // Strip units and whitespace
        s.replace(" ", "");
        s.replace("\t", "");
        s.replace("mm", "");
        s.replace("deg", "");

        if (s.length() == 0) return false;

        // Look for + or - operator NOT in position 0 (would be sign of leading number).
        int op_idx = -1;
        char op_char = 0;
        for (unsigned int i = 1; i < s.length(); i++)
        {
            char ch = s.charAt(i);
            if (ch == '+' || ch == '-') { op_idx = i; op_char = ch; break; }
        }

        if (op_idx > 0)
        {
            String lhs = s.substring(0, op_idx);
            String rhs = s.substring(op_idx + 1);
            float l, r;
            if (!resolveTerm(lhs, l)) return false;
            if (!resolveTerm(rhs, r)) return false;
            out = (op_char == '+') ? (l + r) : (l - r);
            return true;
        }

        return resolveTerm(s, out);
    }

    // Parse "(a, b)" or "a b" or "a,b" or "z_start=150mm, percent_rise=5" -> two floats.
    bool parseTwoArgs(String args, float& a, float& b, float defA, float defB)
    {
        a = defA; b = defB;
        args.trim();
        if (args.startsWith("(") && args.endsWith(")"))
            args = args.substring(1, args.length() - 1);
        args.trim();
        if (args.length() == 0) return true;

        // Split on first comma; if no comma, split on first whitespace run.
        int splitIdx = args.indexOf(',');
        String first, second;
        if (splitIdx >= 0)
        {
            first  = args.substring(0, splitIdx);
            second = args.substring(splitIdx + 1);
        }
        else
        {
            int sp = args.indexOf(' ');
            if (sp < 0) { first = args; second = ""; }
            else        { first = args.substring(0, sp); second = args.substring(sp + 1); }
        }

        // Strip "keyword=" prefix.
        int eqA = first.indexOf('=');
        if (eqA >= 0) first = first.substring(eqA + 1);
        int eqB = second.indexOf('=');
        if (eqB >= 0) second = second.substring(eqB + 1);

        first.trim(); second.trim();

        if (first.length()  > 0 && !parseValueExpression(first,  a)) return false;
        if (second.length() > 0 && !parseValueExpression(second, b)) return false;
        return true;
    }

    bool parseThreeArgs(String args, float& a, float& b, float& c, float defA, float defB, float defC)
    {
        a = defA; b = defB; c = defC;
        args.trim();
        if (args.startsWith("(") && args.endsWith(")"))
            args = args.substring(1, args.length() - 1);
        args.trim();
        if (args.length() == 0) return true;

        // Split into up to 3 tokens (comma-separated; fall back to whitespace).
        String tokens[3] = {"", "", ""};
        int count = 0;
        String rem = args;
        for (int i = 0; i < 3 && rem.length() > 0; i++)
        {
            int sep = rem.indexOf(',');
            if (sep < 0) sep = rem.indexOf(' ');
            if (sep < 0) { tokens[count++] = rem; rem = ""; }
            else         { tokens[count++] = rem.substring(0, sep); rem = rem.substring(sep + 1); rem.trim(); }
        }

        float* out[3] = { &a, &b, &c };
        for (int i = 0; i < count; i++)
        {
            int eq = tokens[i].indexOf('=');
            String val = (eq >= 0) ? tokens[i].substring(eq + 1) : tokens[i];
            val.trim();
            if (val.length() > 0 && !parseValueExpression(val, *out[i])) return false;
        }
        return true;
    }

    // Generic float-arg parser: fills vals[0..n-1] from space/comma-delimited args,
    // strips keyword= prefixes, falls back to defs[] for missing tokens.
    bool parseFloatArgs(String args, float* vals, int n, const float* defs)
    {
        for (int i = 0; i < n; i++) vals[i] = defs[i];
        args.trim();
        if (args.startsWith("(") && args.endsWith(")"))
            args = args.substring(1, args.length() - 1);
        args.trim();
        if (args.length() == 0) return true;

        String rem = args;
        for (int i = 0; i < n && rem.length() > 0; i++)
        {
            int sep = rem.indexOf(',');
            if (sep < 0) sep = rem.indexOf(' ');
            String tok;
            if (sep < 0) { tok = rem; rem = ""; }
            else         { tok = rem.substring(0, sep); rem = rem.substring(sep + 1); rem.trim(); }
            int eq = tok.indexOf('=');
            if (eq >= 0) tok = tok.substring(eq + 1);
            tok.trim();
            if (tok.length() > 0 && !parseValueExpression(tok, vals[i])) return false;
        }
        return true;
    }

    // ============================================================
    //  NEW: DYNAMIC GRIP
    // ============================================================

    float dynamicGrip(float angle_start_deg, float deriv_threshold, int N_steps, bool signed_only)
    {
        if (!ina219_ok)
        {
            Serial.println("# ERR INA219 not available; cannot run dynamicgrip");
            return NAN;
        }

        float angle_open = min(angle_start_deg + dyn_grip_open_offset_deg, (float)SERVO_ANGLE_MAX_DEG);
        if (N_steps < 1) N_steps = 1;
        int pattern_n = constrain(dyn_deriv_pattern_nonzero_n, 1, DYN_DERIV_PATTERN_NONZERO_N_MAX);

        Serial.print("# DYNAMIC GRIP: angle_start="); Serial.print(angle_start_deg, 2);
        Serial.print(" (open_from="); Serial.print(angle_open, 2); Serial.print(")");
        Serial.print(" deriv_thresh="); Serial.print(deriv_threshold, 2);
        Serial.print(" N="); Serial.print(N_steps);
        Serial.print(" mode="); Serial.println(signed_only ? "signed" : "magnitude");
        Serial.print("# Secondary gate: last "); Serial.print(pattern_n);
        Serial.print(" non-zero dI positive and sum > "); Serial.println(dyn_deriv_pattern_sum_grip_ma, 2);

        writeServoFractional(angle_open);
        delay(DYN_SETTLE_MS);
        Serial.println("# Closing gripper - derivative trigger active. Send any line to cancel.");

        float angle = angle_open;
        float c_prev = NAN;
        int   consec = 0;
        float recent_nz_derivs[DYN_DERIV_PATTERN_NONZERO_N_MAX] = {0};
        int   recent_nz_count = 0;
        unsigned long t_start = millis();
        unsigned long next_sample = millis();

        while (angle > DYN_GRIP_MIN_ANGLE_DEG)
        {
            if (drainSerialAsCancel())
            {
                Serial.println("# CANCELLED.");
                return NAN;
            }
            if ((millis() - t_start) > DYN_GRIP_TIMEOUT_MS)
            {
                Serial.println("# ERR dynamicgrip: timeout reached without contact");
                return NAN;
            }

            unsigned long now = millis();
            if ((long)(now - next_sample) < 0) continue;
            next_sample = now + DYN_SAMPLE_PERIOD_MS;

            float c = ina219.getCurrent_mA();

            if (isnan(c_prev))
            {
                c_prev = c;
                plotterDeriv(c, 0.0f, deriv_threshold, 0);
                angle -= dyn_grip_deg_per_sample_runtime;
                writeServoFractional(angle);
                continue;
            }

            float deriv = c - c_prev;
            c_prev = c;

            bool above = signed_only ? (deriv > deriv_threshold)
                                     : (fabs(deriv) > deriv_threshold);
            consec = above ? consec + 1 : 0;

            if (fabs(deriv) > dyn_deriv_nonzero_eps_ma)
            {
                if (recent_nz_count < DYN_DERIV_PATTERN_NONZERO_N_MAX)
                {
                    recent_nz_derivs[recent_nz_count++] = deriv;
                }
                else
                {
                    for (int i = 0; i < DYN_DERIV_PATTERN_NONZERO_N_MAX - 1; i++)
                        recent_nz_derivs[i] = recent_nz_derivs[i + 1];
                    recent_nz_derivs[DYN_DERIV_PATTERN_NONZERO_N_MAX - 1] = deriv;
                }
            }

            bool pattern_hit = false;
            if (recent_nz_count >= pattern_n)
            {
                bool all_pos = true;
                float sum = 0.0f;
                int start_idx = recent_nz_count - pattern_n;
                for (int i = start_idx; i < recent_nz_count; i++)
                {
                    float d = recent_nz_derivs[i];
                    if (d <= 0.0f) { all_pos = false; break; }
                    sum += d;
                }
                pattern_hit = all_pos && (sum > dyn_deriv_pattern_sum_grip_ma);
            }

            plotterDeriv(c, deriv, deriv_threshold, consec);

            if (consec >= N_steps || pattern_hit)
            {
                Serial.print("# Trigger source: ");
                if (consec >= N_steps) Serial.println("derivative consecutive gate");
                else                   Serial.println("non-zero derivative pattern gate");
                Serial.print("# CONTACT detected at angle = ");
                Serial.print(angle, 2); Serial.println(" deg");
                float final_angle = angle;
                if (dyn_grip_object_squishable && dyn_grip_post_contact_extra_close_deg > 0.0f)
                {
                    final_angle = max((float)SERVO_ANGLE_MIN_DEG, angle - dyn_grip_post_contact_extra_close_deg);
                    if (final_angle < angle - 1e-6f)
                    {
                        Serial.print("# Squishable extra close: ");
                        Serial.print(angle, 2);
                        Serial.print(" -> ");
                        Serial.print(final_angle, 2);
                        Serial.println(" deg");
                        writeServoFractional(final_angle);
                    }
                }
                servo_empirical_deg = final_angle;
                Serial.print("# Stored servo_empirical = ");
                Serial.print(servo_empirical_deg, 2); Serial.println(" deg");
                return final_angle;
            }

            angle -= dyn_grip_deg_per_sample_runtime;
            writeServoFractional(angle);
        }

        Serial.print("# WARN dynamicgrip: reached min angle (");
        Serial.print(DYN_GRIP_MIN_ANGLE_DEG, 2); Serial.println(") without contact");
        servo_empirical_deg = angle;
        Serial.print("# Stored servo_empirical = ");
        Serial.print(servo_empirical_deg, 2); Serial.println(" deg");
        return angle;
    }

    // ============================================================
    //  NEW: DYNAMIC LOWER (J3)
    // ============================================================

    float dynamicLower(float z_start_mm, float deriv_threshold, int N_steps, float servo_angle_deg, bool signed_only)
    {
        if (!ina219_ok)
        {
            Serial.println("# ERR INA219 not available; cannot run dynamiclower");
            return NAN;
        }

        if (!isnan(servo_angle_deg))
        {
            servo_angle_deg = constrain(servo_angle_deg, SERVO_ANGLE_MIN_DEG, SERVO_ANGLE_MAX_DEG);
            writeServoFractional(servo_angle_deg);
            Serial.print("# servo -> "); Serial.print(servo_angle_deg, 2); Serial.println(" deg");
        }

        if (N_steps < 1) N_steps = 1;
        int pattern_n = constrain(dyn_deriv_pattern_nonzero_n, 1, DYN_DERIV_PATTERN_NONZERO_N_MAX);

        Serial.print("# DYNAMIC LOWER: z_start=");
        Serial.print(z_start_mm, 2);
        Serial.print(" mm   deriv_thresh="); Serial.print(deriv_threshold, 2);
        Serial.print(" N="); Serial.print(N_steps);
        Serial.print(" servo="); Serial.print(currentServoAngleDeg, 2);
        Serial.print(" mode="); Serial.println(signed_only ? "signed" : "magnitude");
        Serial.print("# Secondary gate: last "); Serial.print(pattern_n);
        Serial.print(" non-zero dI positive and sum > "); Serial.println(dyn_deriv_pattern_sum_lower_ma, 2);

        if (!moveJ3ToMmBlocking(z_start_mm)) return NAN;
        delay(DYN_SETTLE_MS);
        Serial.println("# Descending - derivative trigger active. Send any line to cancel.");

        float descent_steps_per_sec = (float)J3_DESCEND_DIR * dyn_lower_speed_mm_s_runtime * J3_STEPS_PER_MM;
        J3.setMaxSpeed(max((float)fabs(descent_steps_per_sec) * 4.0f, 500.0f));
        J3.setAcceleration(J3_NORMAL_ACCEL);
        J3.setSpeed(descent_steps_per_sec);

        float c_prev = NAN;
        int   consec = 0;
        float recent_nz_derivs[DYN_DERIV_PATTERN_NONZERO_N_MAX] = {0};
        int   recent_nz_count = 0;
        unsigned long t_start = millis();
        unsigned long next_sample = millis();

        while (true)
        {
            J3.runSpeed();

            if (drainSerialAsCancel())
            {
                J3.setSpeed(0); holdAxisHere(J3);
                Serial.println("# CANCELLED during descent.");
                return NAN;
            }
            if ((millis() - t_start) > DYN_LOWER_TIMEOUT_MS)
            {
                J3.setSpeed(0); holdAxisHere(J3);
                Serial.println("# ERR dynamiclower: timeout reached without contact");
                return NAN;
            }

            float current_mm = j3StepsToMm(J3.currentPosition());
            if (current_mm <= DYN_LOWER_MIN_HEIGHT_MM)
            {
                J3.setSpeed(0); holdAxisHere(J3);
                Serial.println("# WARN dynamiclower: reached min height without contact");
                z_empirical_mm = zEmpiricalFromContactAndServo(current_mm, currentServoAngleDeg);
                Serial.print("# Raw contact z = ");
                Serial.print(current_mm, 2); Serial.println(" mm");
                Serial.print("# Comp using servo angle ");
                Serial.print(currentServoAngleDeg, 2); Serial.println(" deg");
                Serial.print("# Stored z_empirical = ");
                Serial.print(z_empirical_mm, 2); Serial.println(" mm");
                return current_mm;
            }

            unsigned long now = millis();
            if ((long)(now - next_sample) < 0) continue;
            next_sample = now + DYN_SAMPLE_PERIOD_MS;

            float c = ina219.getCurrent_mA();

            if (isnan(c_prev))
            {
                c_prev = c;
                plotterDeriv(c, 0.0f, deriv_threshold, 0);
                continue;
            }

            float deriv = c - c_prev;
            c_prev = c;

            bool above = signed_only ? (deriv > deriv_threshold)
                                     : (fabs(deriv) > deriv_threshold);
            consec = above ? consec + 1 : 0;

            if (fabs(deriv) > dyn_deriv_nonzero_eps_ma)
            {
                if (recent_nz_count < DYN_DERIV_PATTERN_NONZERO_N_MAX)
                {
                    recent_nz_derivs[recent_nz_count++] = deriv;
                }
                else
                {
                    for (int i = 0; i < DYN_DERIV_PATTERN_NONZERO_N_MAX - 1; i++)
                        recent_nz_derivs[i] = recent_nz_derivs[i + 1];
                    recent_nz_derivs[DYN_DERIV_PATTERN_NONZERO_N_MAX - 1] = deriv;
                }
            }

            bool pattern_hit = false;
            if (recent_nz_count >= pattern_n)
            {
                bool all_pos = true;
                float sum = 0.0f;
                int start_idx = recent_nz_count - pattern_n;
                for (int i = start_idx; i < recent_nz_count; i++)
                {
                    float d = recent_nz_derivs[i];
                    if (d <= 0.0f) { all_pos = false; break; }
                    sum += d;
                }
                pattern_hit = all_pos && (sum > dyn_deriv_pattern_sum_lower_ma);
            }

            plotterDeriv(c, deriv, deriv_threshold, consec);

            if (consec >= N_steps || pattern_hit)
            {
                J3.setSpeed(0); holdAxisHere(J3);
                Serial.print("# Trigger source: ");
                if (consec >= N_steps) Serial.println("derivative consecutive gate");
                else                   Serial.println("non-zero derivative pattern gate");
                float final_mm = j3StepsToMm(J3.currentPosition());
                Serial.print("# CONTACT detected at J3 = ");
                Serial.print(final_mm, 2); Serial.println(" mm");
                z_empirical_mm = zEmpiricalFromContactAndServo(final_mm, currentServoAngleDeg);
                Serial.print("# Comp using servo angle ");
                Serial.print(currentServoAngleDeg, 2); Serial.println(" deg");
                Serial.print("# Stored z_empirical = ");
                Serial.print(z_empirical_mm, 2); Serial.println(" mm");
                return final_mm;
            }
        }
    }

    // ============================================================
    //  NEW: HELP / SHOW
    // ============================================================

    void printStored()
    {
        Serial.print("# z_empirical    = ");
        if (isnan(z_empirical_mm))    Serial.println("<unset>");
        else { Serial.print(z_empirical_mm, 2);    Serial.println(" mm"); }

        Serial.print("# servo_empirical= ");
        if (isnan(servo_empirical_deg)) Serial.println("<unset>");
        else { Serial.print(servo_empirical_deg, 2); Serial.println(" deg"); }

        Serial.print("# servo (current)= "); Serial.print(currentServoAngleDeg, 2); Serial.println(" deg");
        Serial.print("# l_preset      = "); Serial.print(dyn_z_servo_l_preset_mm, 2); Serial.println(" mm");
        Serial.print("# J3    (current)= "); Serial.print(j3StepsToMm(J3.currentPosition()), 2); Serial.println(" mm");
        Serial.print("# POS  J1="); Serial.print(J1.currentPosition());
        Serial.print(" J2=");       Serial.print(J2.currentPosition());
        Serial.print(" J3=");       Serial.print(J3.currentPosition());
        Serial.print(" J4=");       Serial.println(J4.currentPosition());
    }

    void printHelp()
    {
        Serial.println("# === Manipulator + INA219 firmware ===");
        Serial.println("# Stepper commands (case-insensitive):");
        Serial.println("#   HOME / HOMEJ1 / HOMEJ2 / HOMEJ3");
        Serial.println("#   MOVE <j1> <j2> <j3> <j4>          (absolute step counts)");
        Serial.println("#   MOVESYNC <j1> <j2> <j3> <j4>");
        Serial.println("#   MOVESYNC_T <j1> <j2> <j3> <j4> <ms>");
        Serial.println("#   ZERO / POS / LIMITS");
        Serial.println("#   ENABLE / DISABLE / EN <1|0> / MOTORS?");
        Serial.println("#   INITDRIVERS");
        Serial.println("#");
        Serial.println("# Servo:");
        Serial.println("#   SERVO <angle>           integer 10..70 (original)");
        Serial.println("#   servo = <expr>          fractional deg, accepts servo_empirical");
        Serial.println("#");
        Serial.println("# Prismatic J3 (mm above mechanical bottom):");
        Serial.println("#   J3 = <expr>             e.g. J3 = 200, J3 = z_empirical, J3 = z_empirical + 30");
        Serial.println("#");
        Serial.println("# Dynamic operations (block until trigger / cancel):");
        Serial.println("#   dynamiclower <z_start> <deriv_thresh> <N> [<servo_deg> [<signed>]]  -> sets z_empirical");
        Serial.println("#   DLR <robot_z_mm> <deriv_thresh> <N> [<servo_deg> [<signed>]]         same frame as Python FK");
        Serial.println("#     deriv_thresh: mA/sample spike to detect; N: consecutive samples required");
        Serial.println("#     servo_deg optional; signed=1 for positive-only derivative (default 0=magnitude)");
        Serial.println("#   dynamicgrip <angle_start> <deriv_thresh> <N> [<signed>]  -> sets servo_empirical");
        Serial.println("#   dynset? | dynshow        print dynamic tuning settings");
        Serial.println("#   dynset <key> <value>     set dynamic tuning value");
        Serial.println("#     keys: dg_start_deg dl_start_mm deriv_thresh_ma deriv_n_steps_grip deriv_n_steps_lower signed_only");
        Serial.println("#           dl_servo_deg(current|deg) grip_open_offset_deg grip_deg_per_sample");
        Serial.println("#           grip_object_squishable grip_post_contact_extra_close_deg");
        Serial.println("#           lower_speed_mm_s deriv_nonzero_eps_ma pattern_nonzero_n");
        Serial.println("#           pattern_sum_grip_ma pattern_sum_lower_ma");
        Serial.println("#   set_l_preset <mm>       runtime set for z compensation (z += l_preset*cos(servo))");
        Serial.println("#   Aliases: DL / DG");
        Serial.println("#   Defaults are from dynset (DG uses dg_start_deg; DL uses dl_start_mm).");
        Serial.println("#   Cancel: send any line (just press Enter).");
        Serial.println("#");
        Serial.println("# Telemetry / state:");
        Serial.println("#   SHOW                    print stored empirical values and current pose");
        Serial.println("#   STREAM ON | STREAM OFF  idle current stream toggle (plotter)");
        Serial.println("#   HELP                    this message");
        Serial.println("# ======================================");
    }

    // ============================================================
    //  NEW: J3 / SERVO setters used by the parser
    // ============================================================

    void handleJ3Assign(String rest)
    {
        rest.trim();
        if (rest.startsWith("=")) { rest = rest.substring(1); rest.trim(); }
        float mm;
        if (!parseValueExpression(rest, mm))
        {
            Serial.println("# ERR J3: invalid expression");
            return;
        }
        Serial.print("# J3 -> "); Serial.print(mm, 2); Serial.println(" mm");
        moveJ3ToMmBlocking(mm);
        Serial.println("DONE");
    }

    void handleServoAssign(String rest)
    {
        rest.trim();
        if (rest.startsWith("=")) { rest = rest.substring(1); rest.trim(); }
        float deg;
        if (!parseValueExpression(rest, deg))
        {
            Serial.println("# ERR servo: invalid expression");
            return;
        }
        deg = constrain(deg, SERVO_ANGLE_MIN_DEG, SERVO_ANGLE_MAX_DEG);
        writeServoFractional(deg);
        Serial.print("# servo -> "); Serial.print(deg, 2); Serial.println(" deg");
    }

    void handleSetLPreset(String rest)
    {
        rest.trim();
        float mm;
        if (!parseValueExpression(rest, mm))
        {
            Serial.println("# ERR set_l_preset: invalid value");
            return;
        }
        if (mm < 0.0f)
        {
            Serial.println("# ERR set_l_preset: value must be >= 0 mm");
            return;
        }

        dyn_z_servo_l_preset_mm = mm;
        Serial.print("# l_preset -> "); Serial.print(dyn_z_servo_l_preset_mm, 2); Serial.println(" mm");
    }

    void printDynamicSettings()
    {
        Serial.println("# Dynamic settings:");
        Serial.print("#   dg_start_deg      = "); Serial.println(dyn_default_dg_start_deg, 3);
        Serial.print("#   dl_start_mm       = "); Serial.println(dyn_default_dl_start_mm, 3);
        Serial.print("#   deriv_thresh_ma   = "); Serial.println(dyn_default_deriv_thresh_ma, 3);
        Serial.print("#   deriv_n_steps_grip  = "); Serial.println(dyn_default_deriv_n_steps_grip);
        Serial.print("#   deriv_n_steps_lower = "); Serial.println(dyn_default_deriv_n_steps_lower);
        Serial.print("#   signed_only       = "); Serial.println(dyn_default_signed_only ? 1 : 0);
        Serial.print("#   dl_servo_deg      = ");
        if (isnan(dyn_default_dl_servo_deg)) Serial.println("current");
        else Serial.println(dyn_default_dl_servo_deg, 3);

        Serial.print("#   grip_open_offset_deg = "); Serial.println(dyn_grip_open_offset_deg, 3);
        Serial.print("#   grip_deg_per_sample  = "); Serial.println(dyn_grip_deg_per_sample_runtime, 4);
        Serial.print("#   lower_speed_mm_s     = "); Serial.println(dyn_lower_speed_mm_s_runtime, 3);
        Serial.print("#   grip_object_squishable = "); Serial.println(dyn_grip_object_squishable ? 1 : 0);
        Serial.print("#   grip_post_contact_extra_close_deg = "); Serial.println(dyn_grip_post_contact_extra_close_deg, 3);

        Serial.print("#   deriv_nonzero_eps_ma   = "); Serial.println(dyn_deriv_nonzero_eps_ma, 3);
        Serial.print("#   pattern_nonzero_n      = "); Serial.println(dyn_deriv_pattern_nonzero_n);
        Serial.print("#   pattern_sum_grip_ma    = "); Serial.println(dyn_deriv_pattern_sum_grip_ma, 3);
        Serial.print("#   pattern_sum_lower_ma   = "); Serial.println(dyn_deriv_pattern_sum_lower_ma, 3);
    }

    void handleDynSet(String rest)
    {
        rest.trim();
        if (rest.length() == 0 || rest == "?")
        {
            printDynamicSettings();
            return;
        }

        int sp = rest.indexOf(' ');
        if (sp < 0)
        {
            Serial.println("# ERR dynset: usage dynset <key> <value>");
            return;
        }

        String key = rest.substring(0, sp);
        String val = rest.substring(sp + 1);
        key.trim();
        val.trim();

        float f;
        if (key == "dg_start_deg")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset dg_start_deg"); return; }
            dyn_default_dg_start_deg = constrain(f, SERVO_ANGLE_MIN_DEG, SERVO_ANGLE_MAX_DEG);
        }
        else if (key == "dl_start_mm")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset dl_start_mm"); return; }
            dyn_default_dl_start_mm = max(0.0f, f);
        }
        else if (key == "deriv_thresh_ma")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset deriv_thresh_ma"); return; }
            dyn_default_deriv_thresh_ma = max(0.0f, f);
        }
        else if (key == "deriv_n_steps_grip")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset deriv_n_steps_grip"); return; }
            dyn_default_deriv_n_steps_grip = max(1, (int)f);
        }
        else if (key == "deriv_n_steps_lower")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset deriv_n_steps_lower"); return; }
            dyn_default_deriv_n_steps_lower = max(1, (int)f);
        }
        else if (key == "deriv_n_steps")
        {
            // Backward-compatible alias: set both at once.
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset deriv_n_steps"); return; }
            int n = max(1, (int)f);
            dyn_default_deriv_n_steps_grip = n;
            dyn_default_deriv_n_steps_lower = n;
        }
        else if (key == "signed_only")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset signed_only"); return; }
            dyn_default_signed_only = (f != 0.0f);
        }
        else if (key == "dl_servo_deg")
        {
            if (val == "current") dyn_default_dl_servo_deg = NAN;
            else
            {
                if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset dl_servo_deg"); return; }
                dyn_default_dl_servo_deg = constrain(f, SERVO_ANGLE_MIN_DEG, SERVO_ANGLE_MAX_DEG);
            }
        }
        else if (key == "grip_open_offset_deg")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset grip_open_offset_deg"); return; }
            dyn_grip_open_offset_deg = max(0.0f, f);
        }
        else if (key == "grip_deg_per_sample")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset grip_deg_per_sample"); return; }
            dyn_grip_deg_per_sample_runtime = max(0.001f, f);
        }
        else if (key == "lower_speed_mm_s")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset lower_speed_mm_s"); return; }
            dyn_lower_speed_mm_s_runtime = max(0.1f, f);
        }
        else if (key == "grip_object_squishable")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset grip_object_squishable"); return; }
            dyn_grip_object_squishable = (f != 0.0f);
        }
        else if (key == "grip_post_contact_extra_close_deg")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset grip_post_contact_extra_close_deg"); return; }
            dyn_grip_post_contact_extra_close_deg = max(0.0f, f);
        }
        else if (key == "deriv_nonzero_eps_ma")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset deriv_nonzero_eps_ma"); return; }
            dyn_deriv_nonzero_eps_ma = max(0.0f, f);
        }
        else if (key == "pattern_nonzero_n")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset pattern_nonzero_n"); return; }
            dyn_deriv_pattern_nonzero_n = constrain((int)f, 1, DYN_DERIV_PATTERN_NONZERO_N_MAX);
        }
        else if (key == "pattern_sum_grip_ma")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset pattern_sum_grip_ma"); return; }
            dyn_deriv_pattern_sum_grip_ma = max(0.0f, f);
        }
        else if (key == "pattern_sum_lower_ma")
        {
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset pattern_sum_lower_ma"); return; }
            dyn_deriv_pattern_sum_lower_ma = max(0.0f, f);
        }
        else if (key == "pattern_sum_ma")
        {
            // Backward-compatible alias: set both thresholds at once.
            if (!parseValueExpression(val, f)) { Serial.println("# ERR dynset pattern_sum_ma"); return; }
            f = max(0.0f, f);
            dyn_deriv_pattern_sum_grip_ma = f;
            dyn_deriv_pattern_sum_lower_ma = f;
        }
        else
        {
            Serial.print("# ERR dynset: unknown key '"); Serial.print(key); Serial.println("'");
            return;
        }

        Serial.println("# dynset OK");
        printDynamicSettings();
    }

    // ============================================================
    //  COMMAND PARSER
    // ============================================================

    void readCommands()
    {
        if (!Serial.available()) return;

        String cmd = Serial.readStringUntil('\n');
        cmd.replace("\r", "");
        cmd.trim();
        if (cmd.length() == 0) return;

        // Case-insensitive: lowercase everything. Original sscanf format strings
        // updated to lowercase below.
        cmd.toLowerCase();

        // ----- MOVESYNC_T first (longest prefix) -----
        if (cmd.startsWith("movesync_t"))
        {
            long s1, s2, s3, s4, timeMs;
            int matched = sscanf(cmd.c_str(), "movesync_t %ld %ld %ld %ld %ld",
                                &s1, &s2, &s3, &s4, &timeMs);
            if (matched == 5)
            {
                float requestedTimeSec = max(0.001f, ((float)timeMs) / 1000.0f);
                moveSyncToTimed(s1, s2, s3, s4, requestedTimeSec);
            }
            else Serial.println("ERR MOVESYNC_T requires 4 integer args plus time_ms");
            return;
        }

        if (cmd.startsWith("movesync"))
        {
            long s1, s2, s3, s4;
            int matched = sscanf(cmd.c_str(), "movesync %ld %ld %ld %ld", &s1, &s2, &s3, &s4);
            if (matched == 4) moveSyncTo(s1, s2, s3, s4);
            else Serial.println("ERR MOVESYNC requires 4 integer args");
            return;
        }

        if (cmd.startsWith("move "))
        {
            long s1, s2, s3, s4;
            int matched = sscanf(cmd.c_str(), "move %ld %ld %ld %ld", &s1, &s2, &s3, &s4);
            if (matched == 4) moveNormalTo(s1, s2, s3, s4);
            else Serial.println("ERR MOVE requires 4 integer args");
            return;
        }

        if (cmd == "home")    { homeAll(); return; }
        if (cmd == "homej1")  { Serial.println("HOMING"); if (liftJ3ForHoming() && homeJ1Coupled()) Serial.println("HOMED"); return; }
        if (cmd == "homej2")  { Serial.println("HOMING"); if (liftJ3ForHoming() && homeJ2Only())    Serial.println("HOMED"); return; }
        if (cmd == "homej3")
        {
            Serial.println("HOMING");
            enableMotors(true);
            if (homeJ3Only()) Serial.println("HOMED");
            return;
        }

        if (cmd.startsWith("zero"))
        {
            J1.setCurrentPosition(0); J2.setCurrentPosition(0);
            J3.setCurrentPosition(0); J4.setCurrentPosition(0);
            holdAllAxesHere();
            Serial.println("ZEROED");
            return;
        }

        if (cmd.startsWith("pos"))
        {
            Serial.print("POS ");
            Serial.print(J1.currentPosition()); Serial.print(" ");
            Serial.print(J2.currentPosition()); Serial.print(" ");
            Serial.print(J3.currentPosition()); Serial.print(" ");
            Serial.println(J4.currentPosition());
            return;
        }

        if (cmd.startsWith("limits"))     { printLimits(); return; }
        if (cmd == "enable")              { enableMotors(true);  Serial.println("ENABLED");  return; }
        if (cmd == "disable")             { enableMotors(false); Serial.println("DISABLED"); return; }
        if (cmd == "motors?")             { Serial.println(enabled ? "ENABLED" : "DISABLED"); return; }
        if (cmd == "initdrivers")         { setupAllDrivers(); Serial.println("DRIVERS_OK"); return; }
        if (cmd.startsWith("en "))
        {
            int e = cmd.substring(3).toInt();
            enableMotors(e);
            Serial.println(enabled ? "ENABLED" : "DISABLED");
            return;
        }

        if (cmd == "help")                { printHelp(); return; }
        if (cmd == "show")                { printStored(); return; }
        if (cmd == "dynset?" || cmd == "dynshow") { printDynamicSettings(); return; }
        if (cmd.startsWith("dynset"))
        {
            String rest = cmd.substring(6);
            handleDynSet(rest);
            return;
        }
        if (cmd == "stream on")           { stream_idle = true;  Serial.println("# STREAM ON");  return; }
        if (cmd == "stream off")          { stream_idle = false; Serial.println("# STREAM OFF"); return; }
        if (cmd.startsWith("set_l_preset"))
        {
            String rest = cmd.substring(12);
            rest.trim();
            if (rest.length() == 0)
            {
                Serial.print("# l_preset = "); Serial.print(dyn_z_servo_l_preset_mm, 2); Serial.println(" mm");
                return;
            }
            handleSetLPreset(rest);
            return;
        }

        // ---- dynamicgrip / dg ----
        // DG <angle_start> <deriv_thresh> <N_steps> [<signed_0_1>]
        if (cmd.startsWith("dynamicgrip") || cmd.startsWith("dg"))
        {
            int sp = cmd.startsWith("dynamicgrip") ? 11 : 2;
            String args = cmd.substring(sp); args.trim();
            const float defs[4] = {
                dyn_default_dg_start_deg,
                dyn_default_deriv_thresh_ma,
                (float)dyn_default_deriv_n_steps_grip,
                dyn_default_signed_only ? 1.0f : 0.0f
            };
            float vals[4];
            if (!parseFloatArgs(args, vals, 4, defs))
            {
                Serial.println("# ERR dynamicgrip: bad args");
                return;
            }
            float result = dynamicGrip(vals[0], vals[1], (int)vals[2], vals[3] != 0.0f);
            if (!isnan(result)) Serial.println("DONE");
            return;
        }

        // ---- dynamiclower_robot / dlr ----
        // DLR <robot_z_mm> <deriv_thresh> <N_steps> [<servo_deg> [<signed_0_1>]]
        if (cmd.startsWith("dynamiclower_robot") || cmd.startsWith("dlr"))
        {
            int sp = cmd.startsWith("dynamiclower_robot") ? 18 : 3;
            String args = cmd.substring(sp); args.trim();
            const float defs[5] = {
                dyn_default_dl_start_mm,
                dyn_default_deriv_thresh_ma,
                (float)dyn_default_deriv_n_steps_lower,
                dyn_default_dl_servo_deg,
                dyn_default_signed_only ? 1.0f : 0.0f
            };
            float vals[5];
            if (!parseFloatArgs(args, vals, 5, defs))
            {
                Serial.println("# ERR dynamiclower_robot: bad args");
                return;
            }
            float z_start_mm = robotZToDynamicLowerMm(vals[0]);
            Serial.print("# DLR robot_z_mm = "); Serial.print(vals[0], 2);
            Serial.print(" -> direct_J3_mm = "); Serial.println(z_start_mm, 2);
            float result = dynamicLower(z_start_mm, vals[1], (int)vals[2], vals[3], vals[4] != 0.0f);
            if (!isnan(result)) Serial.println("DONE");
            return;
        }

        // ---- dynamiclower / dl ----
        // DL <z_start> <deriv_thresh> <N_steps> [<servo_deg> [<signed_0_1>]]
        if (cmd.startsWith("dynamiclower") || cmd.startsWith("dl"))
        {
            int sp = cmd.startsWith("dynamiclower") ? 12 : 2;
            String args = cmd.substring(sp); args.trim();
            const float defs[5] = {
                dyn_default_dl_start_mm,
                dyn_default_deriv_thresh_ma,
                (float)dyn_default_deriv_n_steps_lower,
                dyn_default_dl_servo_deg,
                dyn_default_signed_only ? 1.0f : 0.0f
            };
            float vals[5];
            if (!parseFloatArgs(args, vals, 5, defs))
            {
                Serial.println("# ERR dynamiclower: bad args");
                return;
            }
            float result = dynamicLower(vals[0], vals[1], (int)vals[2], vals[3], vals[4] != 0.0f);
            if (!isnan(result)) Serial.println("DONE");
            return;
        }

        // ---- J3 = <expr> ----
        if (cmd.startsWith("j3"))
        {
            handleJ3Assign(cmd.substring(2));
            return;
        }

        // ---- servo = <expr>  OR  servo <int> (original) ----
        if (cmd.startsWith("servo"))
        {
            String rest = cmd.substring(5);
            rest.trim();
            // Original protocol: "SERVO <int>" -- still supported because the integer
            // parser flow goes through handleServoAssign which accepts a bare number.
            handleServoAssign(rest);
            return;
        }

        if (cmd.length() > 0)
        {
            Serial.print("ERR unknown command: "); Serial.println(cmd);
            Serial.println("# Type HELP for the command list.");
        }
    }
