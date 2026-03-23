/*
 * main.cpp
 * ========
 * ESP32 Robot Arm Motor Controller
 * 4-DOF Desktop Arm — ARM_IK_CONVENTIONS_v4
 *
 * Architecture:
 *   IK runs on LAPTOP (Python IKSolver.py + ik_gui.py)
 *   This ESP32 is a PURE MOTOR CONTROLLER + SAFETY BACKSTOP
 *
 * Serial protocol — IK mode (from laptop GUI):
 *   Receive:  "B:<steps> S:<steps> E:<steps> W:<steps> G:<us> V:<pct>\n"
 *              G: gripper pulse width µs (900–2500), optional
 *              V: speed percentage 5–100, optional (default 100 = full)
 *   Reply OK: "OK B:<steps> S:<steps> E:<steps> W:<steps>\n"
 *   Reply ERR:"ERR: <reason>\n"
 *
 * Manual commands (single char, no newline needed):
 *   1/2/3/4   select BASE / SHOULDER / ELBOW / WRIST
 *   w / s     jog selected motor forward / backward
 *   h         home selected motor
 *   a         home ALL motors  (ELBOW → SHOULDER → BASE → WRIST)
 *   i         print joint status
 *   ?         print help
 *   y / n     respond to startup safety prompt
 *
 * Homing order: ELBOW → SHOULDER → BASE → WRIST
 * Stepping:     hardware timer interrupt, all 4 motors simultaneously
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ═══════════════════════════════════════════════════════════════
// PIN DEFINITIONS
// ═══════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════
// GRIPPER (PCA9685 via I2C)
// ═══════════════════════════════════════════════════════════════

#define I2C_SDA              21
#define I2C_SCL              22
#define PCA9685_ADDR       0x40
#define GRIPPER_CHANNEL       3    // confirmed channel 3
#define GRIPPER_PWM_MIN     900    // µs — fully open   (spec 9.11: startup position)
#define GRIPPER_PWM_MAX    2500    // µs — fully closed
#define PCA9685_FREQ         50    // Hz — standard servo frequency

#define SHOULDER_STEP_PIN   12
#define SHOULDER_DIR_PIN    13
#define SHOULDER_LIMIT_PIN  33

#define ELBOW_STEP_PIN       5
#define ELBOW_DIR_PIN        4
#define ELBOW_LIMIT_PIN     32

#define BASE_STEP_PIN       19
#define BASE_DIR_PIN        18
#define BASE_LIMIT_PIN      26

#define WRIST_STEP_PIN       2
#define WRIST_DIR_PIN       15
#define WRIST_LIMIT_PIN     27

#define ESTOP_PIN           14   // INPUT_PULLUP — LOW = triggered

// ═══════════════════════════════════════════════════════════════
// MOTOR INDICES
// ═══════════════════════════════════════════════════════════════

#define MOTOR_BASE      0
#define MOTOR_SHOULDER  1
#define MOTOR_ELBOW     2
#define MOTOR_WRIST     3
#define NUM_MOTORS      4

// ═══════════════════════════════════════════════════════════════
// GEAR RATIOS & STEPS  —  must match IKSolver.py exactly
// ═══════════════════════════════════════════════════════════════

#define STEPS_PER_REV           200

#define BASE_GEAR_RATIO         5.0f
#define SHOULDER_GEAR_RATIO    20.0f
#define ELBOW_GEAR_RATIO        5.0f
#define WRIST_GEAR_RATIO        1.6f      // 32/20 teeth EXACT — not 1.61

#define BASE_STEPS_PER_REV      ((long)(STEPS_PER_REV * BASE_GEAR_RATIO))      // 1000
#define SHOULDER_STEPS_PER_REV  ((long)(STEPS_PER_REV * SHOULDER_GEAR_RATIO))  // 4000
#define ELBOW_STEPS_PER_REV     ((long)(STEPS_PER_REV * ELBOW_GEAR_RATIO))     // 1000
#define WRIST_STEPS_PER_REV     ((long)(STEPS_PER_REV * WRIST_GEAR_RATIO))     //  320

// Steps per degree — must match IKSolver.py
#define BASE_STEPS_PER_DEG      (BASE_STEPS_PER_REV     / 360.0f)   //  2.778
#define SHOULDER_STEPS_PER_DEG  (SHOULDER_STEPS_PER_REV / 360.0f)   // 11.111
#define ELBOW_STEPS_PER_DEG     (ELBOW_STEPS_PER_REV    / 360.0f)   //  2.778
#define WRIST_STEPS_PER_DEG     (WRIST_STEPS_PER_REV    / 360.0f)   //  0.889

// ═══════════════════════════════════════════════════════════════
// DIR INVERT FLAGS
// Set to true if a motor physically runs in the wrong direction
// ═══════════════════════════════════════════════════════════════

#define BASE_DIR_INVERT      false
#define SHOULDER_DIR_INVERT  true
#define ELBOW_DIR_INVERT     false
#define WRIST_DIR_INVERT     false

// ═══════════════════════════════════════════════════════════════
// HOMING CONFIGURATION
// ═══════════════════════════════════════════════════════════════

// Which level on DIR pin drives toward the limit switch
#define BASE_HOME_DIR      false
#define SHOULDER_HOME_DIR  false
#define ELBOW_HOME_DIR     false
#define WRIST_HOME_DIR     true

// Degrees to back off after hitting limit (until switch releases)
#define BASE_SWITCH_RELEASE_DEG       5.0f
#define SHOULDER_SWITCH_RELEASE_DEG  10.0f
#define ELBOW_SWITCH_RELEASE_DEG      5.0f
#define WRIST_SWITCH_RELEASE_DEG      5.0f

// Park delta: degrees from mechanical zero to IK zero
#define BASE_PARK_DELTA_DEG      110.0f   // limit at -110°, IK zero at 0°
#define SHOULDER_PARK_DELTA_DEG    7.0f   // limit IS IK zero (vertical)
#define ELBOW_PARK_DELTA_DEG     130.0f   // limit at -115°, IK zero at 0°
#define WRIST_PARK_DELTA_DEG       31.0f   // calibrate empirically after assembly

// Homing speed: µs per step half-period (larger = slower)
#define HOMING_SPEED_US   4000

// Stall detection: max steps before declaring obstruction (full travel + 20%)
#define BASE_MAX_HOMING_STEPS      1400
#define SHOULDER_MAX_HOMING_STEPS  5000
#define ELBOW_MAX_HOMING_STEPS     1400
#define WRIST_MAX_HOMING_STEPS      500

// Startup safety prompt timeout
#define SAFETY_PROMPT_TIMEOUT_MS  30000

// ═══════════════════════════════════════════════════════════════
// SOFT LIMITS  —  must match IKSolver.py exactly
// ═══════════════════════════════════════════════════════════════

#define BASE_MIN_DEG       -110.0f
#define BASE_MAX_DEG        110.0f
#define SHOULDER_MIN_DEG      0.0f
#define SHOULDER_MAX_DEG     98.0f
#define ELBOW_MIN_DEG      -115.0f
#define ELBOW_MAX_DEG         0.0f
#define WRIST_MIN_DEG      -149.0f
#define WRIST_MAX_DEG       26.0f

// Step limits derived from degree limits
#define BASE_STEPS_MIN      ((long)(BASE_MIN_DEG     * BASE_STEPS_PER_DEG))
#define BASE_STEPS_MAX      ((long)(BASE_MAX_DEG     * BASE_STEPS_PER_DEG))
#define SHOULDER_STEPS_MIN  ((long)(SHOULDER_MIN_DEG * SHOULDER_STEPS_PER_DEG))
#define SHOULDER_STEPS_MAX  ((long)(SHOULDER_MAX_DEG * SHOULDER_STEPS_PER_DEG))
#define ELBOW_STEPS_MIN     ((long)(ELBOW_MIN_DEG    * ELBOW_STEPS_PER_DEG))
#define ELBOW_STEPS_MAX     ((long)(ELBOW_MAX_DEG    * ELBOW_STEPS_PER_DEG))
#define WRIST_STEPS_MIN     ((long)(WRIST_MIN_DEG    * WRIST_STEPS_PER_DEG))
#define WRIST_STEPS_MAX     ((long)(WRIST_MAX_DEG    * WRIST_STEPS_PER_DEG))

// ═══════════════════════════════════════════════════════════════
// SPEED CONFIGURATION
// ═══════════════════════════════════════════════════════════════

#define TIMER_INTERVAL_US         100   // timer fires every 100 µs

// Per-motor speed caps (ticks per step, larger = slower)
// steps/sec = 1,000,000 / (TIMER_INTERVAL_US * ticks)
// 25 ticks = 400 steps/sec  |  50 = 200  |  100 = 100
#define BASE_MAX_SPEED_TICKS       50   // 200 steps/sec  (5:1 gear)
#define SHOULDER_MAX_SPEED_TICKS   80   // 200 steps/sec  (20:1 gear, heavy load)
#define ELBOW_MAX_SPEED_TICKS      50   // 200 steps/sec  (5:1 gear)
#define WRIST_MAX_SPEED_TICKS      50   // 200 steps/sec  (1.6:1 gear, light load)

#define MIN_SPEED_TICKS           150   // slowest tick interval (ramp start/end, all motors)
#define ACCEL_STEPS               300   // ramp length in steps (scaled per motor)

// ═══════════════════════════════════════════════════════════════
// WATCHDOG
// ═══════════════════════════════════════════════════════════════

#define WATCHDOG_MS   500   // hold position if no IK command for 500 ms

// ═══════════════════════════════════════════════════════════════
// MOTOR STATE  —  shared between main loop and timer ISR
// ═══════════════════════════════════════════════════════════════

// ── Gripper globals ─────────────────────────────────────────────
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);

volatile int  gripperTargetUs  = GRIPPER_PWM_MIN;  // target pulse width µs
volatile bool gripperPending   = false;             // deferred write flag

// Convert µs pulse width to PCA9685 12-bit count
// period = 20000µs at 50Hz; counts = us * 4096 / 20000
uint16_t pulseUsToCounts(int us) {
  return (uint16_t)((long)us * 4096L / 20000L);
}

// Apply gripper PWM — call from main loop only (not from ISR)
void applyGripper() {
  if (!gripperPending) return;
  gripperPending = false;
  int us = gripperTargetUs;
  if (us < GRIPPER_PWM_MIN) us = GRIPPER_PWM_MIN;
  if (us > GRIPPER_PWM_MAX) us = GRIPPER_PWM_MAX;
  pwm.setPWM(GRIPPER_CHANNEL, 0, pulseUsToCounts(us));
}

struct MotorState {
  // Position tracking
  volatile long currentSteps;   // current absolute step count from IK zero
  volatile long targetSteps;    // target  absolute step count

  // Ramping state (used by timer ISR)
  volatile int  tickCount;      // ticks elapsed since last step pulse
  volatile long stepsMoving;    // total steps in the current move
  volatile long stepsDone;      // steps completed in the current move
  volatile int  maxSpeedTicks;  // fastest tick interval for this move (scaled per-motor)
  volatile int  accelSteps;     // ramp length for this move (scaled per-motor)
  volatile long startDelay;     // ticks to hold before first step (pacer sync for large moves)

  // Status flags
  volatile bool stepPending;  // two-tick pulse: HIGH was raised last tick, LOW due next
  volatile int  stepDir;      // +1 or -1, latched when step is raised
  bool homed;

  // Hardware
  int  stepPin;
  int  dirPin;
  int  limitPin;
  bool homeDir;     // HIGH or LOW drives toward limit switch
  bool dirInvert;   // flip if wired backwards
};

MotorState motors[NUM_MOTORS];

// ═══════════════════════════════════════════════════════════════
// TIMER INTERRUPT
// ═══════════════════════════════════════════════════════════════

hw_timer_t*  stepTimer = NULL;
portMUX_TYPE timerMux  = portMUX_INITIALIZER_UNLOCKED;

// Returns tick interval for a given step position in a move (trapezoid ramp).
// ISR-SAFE: pure integer arithmetic only — no floats, no min()/max() templates.
// Linear interpolation: ticks = MIN_SPEED_TICKS - range * progress / ramp
int IRAM_ATTR calcTicks(long done, long total, int maxTicks, int accel) {
  if (total <= 0) return MIN_SPEED_TICKS;
  if (accel  <= 0) return maxTicks;          // streaming: no ramp, run at full speed always

  // Ramp = shorter of accel steps or half the move — no min() template
  long ramp = (long)accel;
  if (ramp > total / 2) ramp = total / 2;
  if (ramp < 1)         ramp = 1;          // guard divide-by-zero

  long range = (long)(MIN_SPEED_TICKS - maxTicks);  // tick-interval span

  if (done < ramp) {
    // Accelerating — integer linear interpolation (no floats)
    return (int)(MIN_SPEED_TICKS - range * done / ramp);
  }
  if (done > total - ramp) {
    // Decelerating
    return (int)(MIN_SPEED_TICKS - range * (total - done) / ramp);
  }
  return maxTicks;
}

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timerMux);

  for (int i = 0; i < NUM_MOTORS; i++) {
    MotorState& m = motors[i];

    // ── Start delay: hold motor before first step (pacer sync for large moves) ──
    if (m.startDelay > 0) {
      m.startDelay--;
      continue;
    }

    // ── Two-tick pulse: complete the LOW half of the previous step ──
    // Step pin was raised last tick; lower it now and commit position.
    // No delayMicroseconds — pulse width = timer interval = 100 µs (>> A4988 min 1 µs).
    if (m.stepPending) {
      GPIO.out_w1tc  = (1UL << m.stepPin);  // step LOW
      m.currentSteps += m.stepDir;
      m.stepsDone++;
      m.stepPending  = false;
      continue;  // this motor is done for this tick
    }

    if (m.currentSteps == m.targetSteps) {
      continue;
    }

    m.tickCount++;

    int ticks = calcTicks(m.stepsDone, m.stepsMoving, m.maxSpeedTicks, m.accelSteps);
    if (m.tickCount < ticks) continue;
    m.tickCount = 0;

    bool goFwd = (m.targetSteps > m.currentSteps);
    bool level = goFwd ? HIGH : LOW;
    if (m.dirInvert) level = !level;

    // ISR-safe GPIO writes — no digitalWrite() in ISR (causes LoadProhibited)
    // All step/dir/step pins are < 32 so GPIO bank 0 (out_w1ts/w1tc) covers all
    if (level) GPIO.out_w1ts = (1UL << m.dirPin);
    else       GPIO.out_w1tc = (1UL << m.dirPin);

    // ── Raise step pin; LOW half completes next tick (two-tick pulse) ──
    GPIO.out_w1ts = (1UL << m.stepPin);
    m.stepDir     = goFwd ? 1 : -1;
    m.stepPending = true;
    // currentSteps and stepsDone updated on the following tick when step goes LOW
  }

  portEXIT_CRITICAL_ISR(&timerMux);
}

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

bool allHomed() {
  for (int i = 0; i < NUM_MOTORS; i++)
    if (!motors[i].homed) return false;
  return true;
}

bool allStopped() {
  for (int i = 0; i < NUM_MOTORS; i++) {
    portENTER_CRITICAL(&timerMux);
    bool stopped = (motors[i].currentSteps == motors[i].targetSteps);
    portEXIT_CRITICAL(&timerMux);
    if (!stopped) return false;
  }
  return true;
}

bool estopTriggered() {
  return digitalRead(ESTOP_PIN) == LOW;
}

// Immediately halt all motors and disable timer
void triggerEstop() {
  timerAlarmDisable(stepTimer);
  for (int i = 0; i < NUM_MOTORS; i++) {
    portENTER_CRITICAL(&timerMux);
    motors[i].targetSteps = motors[i].currentSteps;  // cancel move
    motors[i].stepPending = false;
    motors[i].startDelay  = 0;
    portEXIT_CRITICAL(&timerMux);
  }
  timerAlarmEnable(stepTimer);
  Serial.println("!!! ESTOP TRIGGERED — all motors halted. Re-home before resuming.");
}

void waitForStop() {
  while (!allStopped()) delay(10);
}

void setTarget(int motor, long steps, int maxTicks = BASE_MAX_SPEED_TICKS, int accel = ACCEL_STEPS, long sDelay = 0) {
  portENTER_CRITICAL(&timerMux);
  long delta                   = steps - motors[motor].currentSteps;
  motors[motor].targetSteps    = steps;
  motors[motor].stepsMoving    = abs(delta);
  motors[motor].stepsDone      = 0;
  motors[motor].tickCount      = 0;
  motors[motor].maxSpeedTicks  = maxTicks;
  motors[motor].accelSteps     = accel;
  motors[motor].startDelay     = sDelay;
  motors[motor].stepPending    = false;  // cancel any in-flight pulse
  portEXIT_CRITICAL(&timerMux);
}

long degreesToSteps(float deg, int motor) {
  switch (motor) {
    case MOTOR_BASE:     return (long)(deg * BASE_STEPS_PER_DEG);
    case MOTOR_SHOULDER: return (long)(deg * SHOULDER_STEPS_PER_DEG);
    case MOTOR_ELBOW:    return (long)(deg * ELBOW_STEPS_PER_DEG);
    case MOTOR_WRIST:    return (long)(deg * WRIST_STEPS_PER_DEG);
    default: return 0;
  }
}

// Not called from main path (float unsafe while timer runs) — available for offline debug use
float stepsToDegrees(long steps, int motor) {
  switch (motor) {
    case MOTOR_BASE:     return steps / BASE_STEPS_PER_DEG;
    case MOTOR_SHOULDER: return steps / SHOULDER_STEPS_PER_DEG;
    case MOTOR_ELBOW:    return steps / ELBOW_STEPS_PER_DEG;
    case MOTOR_WRIST:    return steps / WRIST_STEPS_PER_DEG;
    default: return 0.0f;
  }
}

// ═══════════════════════════════════════════════════════════════
// SAFETY BACKSTOP
// Called before every IK move and every jog
// ═══════════════════════════════════════════════════════════════

bool checkStepLimits(long b, long s, long e, long w, char* errMsg) {
  if (b < BASE_STEPS_MIN || b > BASE_STEPS_MAX) {
    sprintf(errMsg, "LIMIT B (%ld)  range [%ld, %ld]", b, BASE_STEPS_MIN, BASE_STEPS_MAX);
    return false;
  }
  if (s < SHOULDER_STEPS_MIN || s > SHOULDER_STEPS_MAX) {
    sprintf(errMsg, "LIMIT S (%ld)  range [%ld, %ld]", s, SHOULDER_STEPS_MIN, SHOULDER_STEPS_MAX);
    return false;
  }
  if (e < ELBOW_STEPS_MIN || e > ELBOW_STEPS_MAX) {
    sprintf(errMsg, "LIMIT E (%ld)  range [%ld, %ld]", e, ELBOW_STEPS_MIN, ELBOW_STEPS_MAX);
    return false;
  }
  if (w < WRIST_STEPS_MIN || w > WRIST_STEPS_MAX) {
    sprintf(errMsg, "LIMIT W (%ld)  range [%ld, %ld]", w, WRIST_STEPS_MIN, WRIST_STEPS_MAX);
    return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════
// HOMING
// ═══════════════════════════════════════════════════════════════

// Declared here (before homeAll) so homeAll() can reset them after homing completes
String        serialBuffer     = "";
unsigned long lastCommandMs    = 0;
unsigned long stateBroadcastMs = 0;
bool          watchdogFired    = false;   // true after first timeout — suppresses repeat broadcasts
bool          watchdogArmed    = false;   // armed only after first valid IK command (spec 8.5)
bool          presetInProgress = false;   // true after P:1 command — waiting for allStopped() to broadcast DONE:
bool          wasMoving        = false;   // tracks allStopped() transition for DONE: detection
#define STATE_BROADCAST_MS  500

void homeMotor(int motor) {
  MotorState& m = motors[motor];
  const char* names[] = {"BASE", "SHOULDER", "ELBOW", "WRIST"};

  float releaseDeg, parkDeg;
  long  maxSteps;

  switch (motor) {
    case MOTOR_BASE:
      releaseDeg = BASE_SWITCH_RELEASE_DEG;
      parkDeg    = BASE_PARK_DELTA_DEG;
      maxSteps   = BASE_MAX_HOMING_STEPS;
      break;
    case MOTOR_SHOULDER:
      releaseDeg = SHOULDER_SWITCH_RELEASE_DEG;
      parkDeg    = SHOULDER_PARK_DELTA_DEG;
      maxSteps   = SHOULDER_MAX_HOMING_STEPS;
      break;
    case MOTOR_ELBOW:
      releaseDeg = ELBOW_SWITCH_RELEASE_DEG;
      parkDeg    = ELBOW_PARK_DELTA_DEG;
      maxSteps   = ELBOW_MAX_HOMING_STEPS;
      break;
    case MOTOR_WRIST:
      releaseDeg = WRIST_SWITCH_RELEASE_DEG;
      parkDeg    = WRIST_PARK_DELTA_DEG;
      maxSteps   = WRIST_MAX_HOMING_STEPS;
      break;
    default:
      return;   // invalid motor index — timer never reached disable, nothing to re-enable
  }

  Serial.printf("\n>>> HOMING %s...\n", names[motor]);

  // Stop motor cleanly before disabling the timer.
  // Disabling mid-move leaves currentSteps at a mid-transit value,
  // corrupting position tracking for the rest of the session.
  portENTER_CRITICAL(&timerMux);
  motors[motor].targetSteps  = motors[motor].currentSteps;
  motors[motor].startDelay   = 0;
  motors[motor].stepPending  = false;
  portEXIT_CRITICAL(&timerMux);
  while (!allStopped()) delay(5);   // brief spin — motor was already near-stopped

  // Disable timer interrupt — we step directly during homing
  timerAlarmDisable(stepTimer);

  // ── Phase 1: Drive toward limit switch ───────────────────────
  bool dirToLimit = m.homeDir;
  bool sig = dirToLimit ? HIGH : LOW;
  if (m.dirInvert) sig = !sig;
  digitalWrite(m.dirPin, sig);

  long taken    = 0;
  bool limitHit = false;

  while (taken < maxSteps) {
    if (digitalRead(m.limitPin) == LOW) { limitHit = true; break; }
    digitalWrite(m.stepPin, HIGH); delayMicroseconds(HOMING_SPEED_US);
    digitalWrite(m.stepPin, LOW);  delayMicroseconds(HOMING_SPEED_US);
    taken++;
  }

  if (!limitHit) {
    // Stall detected
    timerAlarmEnable(stepTimer);
    Serial.printf("!!! STALL: %s limit not reached after %ld steps\n", names[motor], maxSteps);
    Serial.println("!!! Check for obstruction. Send 'a' to retry.");
    m.homed = false;
    return;
  }

  Serial.printf("  Limit hit after %ld steps. Backing off %.1f deg...\n", taken, releaseDeg);

  // ── Phase 2: Back off until switch releases ───────────────────
  sig = !dirToLimit ? HIGH : LOW;
  if (m.dirInvert) sig = !sig;
  digitalWrite(m.dirPin, sig);

  long releaseSteps = degreesToSteps(releaseDeg, motor);
  for (long i = 0; i < releaseSteps; i++) {
    if (digitalRead(m.limitPin) == HIGH) break;   // released
    digitalWrite(m.stepPin, HIGH); delayMicroseconds(HOMING_SPEED_US);
    digitalWrite(m.stepPin, LOW);  delayMicroseconds(HOMING_SPEED_US);
  }

  // ── Phase 3: Set mechanical zero ─────────────────────────────
  portENTER_CRITICAL(&timerMux);
  m.currentSteps = 0;
  m.targetSteps  = 0;
  portEXIT_CRITICAL(&timerMux);
  Serial.println("  Mechanical zero set.");

  // ── Phase 4: Drive from mechanical zero to IK zero ───────────
  if (parkDeg > 0.0f) {
    long parkSteps = degreesToSteps(parkDeg, motor);
    Serial.printf("  Parking %.1f deg (%ld steps) to IK zero...\n", parkDeg, parkSteps);

    // Drive away from limit switch
    sig = !dirToLimit ? HIGH : LOW;
    if (m.dirInvert) sig = !sig;
    digitalWrite(m.dirPin, sig);

    for (long i = 0; i < parkSteps; i++) {
      digitalWrite(m.stepPin, HIGH); delayMicroseconds(HOMING_SPEED_US);
      digitalWrite(m.stepPin, LOW);  delayMicroseconds(HOMING_SPEED_US);
    }
  }

  // ── Phase 5: Set IK zero ──────────────────────────────────────
  portENTER_CRITICAL(&timerMux);
  m.currentSteps = 0;
  m.targetSteps  = 0;
  m.stepsDone    = 0;
  m.stepsMoving  = 0;
  m.tickCount    = 0;
  m.startDelay   = 0;
  portEXIT_CRITICAL(&timerMux);

  m.homed = true;
  timerAlarmEnable(stepTimer);

  Serial.printf("  %s HOMED — IK zero set. OK\n", names[motor]);
}

void homeAll() {
  Serial.println("\n╔════════════════════════════════════════════╗");
  Serial.println("║          HOMING ALL MOTORS                 ║");
  Serial.println("║  Order: ELBOW > SHOULDER > BASE > WRIST   ║");
  Serial.println("╚════════════════════════════════════════════╝");

  homeMotor(MOTOR_ELBOW);
  homeMotor(MOTOR_SHOULDER);
  homeMotor(MOTOR_BASE);
  homeMotor(MOTOR_WRIST);

  // Reset gripper to fully open on rehome (spec 9.11)
  gripperTargetUs = GRIPPER_PWM_MIN;
  gripperPending  = true;

  // Timer starts here — after all motors homed, per spec 9.2
  timerAlarmEnable(stepTimer);

  // Reset watchdog — homing takes ~35s, without this the watchdog fires
  // immediately on the first loop() iteration after homeAll() returns.
  // watchdogArmed=false: watchdog only starts counting after the first IK command
  // is received (spec 8.5: "arms after first valid IK command").
  lastCommandMs = millis();
  watchdogFired = false;
  watchdogArmed = false;

  Serial.println("\nOK: ALL MOTORS HOMED — IK zero ready.\n");
}

// ═══════════════════════════════════════════════════════════════
// STARTUP SAFETY PROMPT
// ═══════════════════════════════════════════════════════════════

bool startupSafetyPrompt() {
  // Flush any garbage on RX line before showing prompt
  // (noise from floating RX before Python connects could trigger false 'y')
  delay(100);
  while (Serial.available()) Serial.read();

  Serial.println("\n╔══════════════════════════════════════════════════════╗");
  Serial.println("║              STARTUP SAFETY CHECK                   ║");
  Serial.println("╠══════════════════════════════════════════════════════╣");
  Serial.println("║  Homing will move joints in this order:              ║");
  Serial.println("║    1. ELBOW    drives to tuck limit                  ║");
  Serial.println("║    2. SHOULDER drives to vertical limit              ║");
  Serial.println("║    3. BASE     drives to full-right limit            ║");
  Serial.println("║    4. WRIST    drives to limit                       ║");
  Serial.println("╠══════════════════════════════════════════════════════╣");
  Serial.println("║  Manually position arm to a safe starting pose:      ║");
  Serial.println("║    ELBOW:    roughly tucked (interior ~20-40 deg)    ║");
  Serial.println("║    SHOULDER: roughly vertical (pointing up)          ║");
  Serial.println("║    BASE:     roughly centered (pointing forward)     ║");
  Serial.println("║    WRIST:    roughly horizontal                      ║");
  Serial.println("║  Steppers can be back-driven by hand when idle.      ║");
  Serial.println("╠══════════════════════════════════════════════════════╣");
  Serial.println("║  Is it safe to home?                                 ║");
  Serial.println("║    y = YES begin homing   n = NO manual mode only    ║");
  Serial.println("╚══════════════════════════════════════════════════════╝");
  Serial.printf("\nWaiting %d sec for response...\n", SAFETY_PROMPT_TIMEOUT_MS / 1000);

  unsigned long startMs = millis();
  while (millis() - startMs < SAFETY_PROMPT_TIMEOUT_MS) {
    if (Serial.available()) {
      char c = Serial.read();
      while (Serial.available()) Serial.read();   // flush rest

      if (c == 'y' || c == 'Y') {
        Serial.println("OK: Confirmed — beginning homing sequence...");
        return true;
      }
      if (c == 'n' || c == 'N') {
        Serial.println("OK: Homing skipped — manual mode. Send 'a' when ready.\n");
        return false;
      }
    }
    delay(50);
  }

  Serial.println("OK: Timeout — homing skipped. Send 'a' when ready.\n");
  return false;
}

// ═══════════════════════════════════════════════════════════════
// IK COMMAND HANDLER  (from laptop GUI)
// ═══════════════════════════════════════════════════════════════

void handleIKCommand(const String& line) {
  int bIdx = line.indexOf("B:");
  int sIdx = line.indexOf("S:");
  int eIdx = line.indexOf("E:");
  int wIdx = line.indexOf("W:");

  if (bIdx < 0 || sIdx < 0 || eIdx < 0 || wIdx < 0) {
    Serial.println("ERR: Invalid command");
    return;
  }

  long b = line.substring(bIdx + 2).toInt();
  long s = line.substring(sIdx + 2).toInt();
  long e = line.substring(eIdx + 2).toInt();
  long w = line.substring(wIdx + 2).toInt();

  // V: speed percentage (optional, default 100 = full speed)
  // NOTE: integer arithmetic only — no floats.
  // The ESP32 Arduino framework does not save FPU registers across the timer ISR,
  // so any float operation in the main loop that gets preempted corrupts FPU state.
  int vIdx = line.indexOf("V:");
  int vPct = 100;
  if (vIdx >= 0) {
    vPct = (int)line.substring(vIdx + 2).toInt();
    if (vPct < 5)   vPct = 5;    // no templates, no min()/max() — plain comparisons
    if (vPct > 100) vPct = 100;
  }

  // P: preset flag (optional, default 0)
  // P:1 = one-shot preset move — use trapezoidal ramp, broadcast DONE: on completion
  // P:0 or absent = streaming — accel=0, no ramp (ramp resets every 100ms at 10Hz)
  int pIdx   = line.indexOf("P:");
  bool isPreset = (pIdx >= 0 && line.substring(pIdx + 2).toInt() == 1);

  if (!allHomed()) {
    Serial.println("ERR: NOT_HOMED");
    return;
  }

  char errMsg[100];
  if (!checkStepLimits(b, s, e, w, errMsg)) {
    Serial.printf("ERR: Steps out of range %s\n", errMsg);
    return;
  }

  // G: gripper pulse width in µs (optional) — deferred to main loop (I2C not ISR-safe)
  // Parsed after validation so a rejected command doesn't move the gripper
  int gIdx = line.indexOf("G:");
  if (gIdx >= 0) {
    int gUs = (int)line.substring(gIdx + 2).toInt();
    if (gUs < GRIPPER_PWM_MIN) gUs = GRIPPER_PWM_MIN;
    if (gUs > GRIPPER_PWM_MAX) gUs = GRIPPER_PWM_MAX;
    gripperTargetUs = gUs;
    gripperPending  = true;
  }

  // — Coordinated speed scaling: all motors finish at the same time —
  //
  // IK commands are either:
  //   Streaming (all deltas < 151) — 10Hz tracking updates. No ramp: ramp resets
  //     every 100ms anyway so motors never leave ramp-start speed. accel=0.
  //   Large/preset (any delta >= 151) — one-shot move, runs to completion before
  //     next command. Full proportional ramp for smooth acceleration. accel>0.
  //
  // Start delay is never used for IK commands — it caused freeze on catch-up moves.
  // startDelay remains available for future use via setTarget directly.
  //   STREAM_THRESH = 151 (one above the GUI jerk limit of 150 steps/axis)

  // — Coordinated pacer: mTicks per motor derived from slowest motor duration —
  // Streaming (P:0): accel=0 always — ramp resets every 100ms at 10Hz so motor
  //   never leaves ramp-start speed. Pacer (mTicks) provides speed control.
  // Preset (P:1): full proportional ramp — one-shot move, no follow-up commands
  //   to reset stepsDone. DONE: broadcast when allStopped() after preset.

  const int motorCaps[NUM_MOTORS] = {
    BASE_MAX_SPEED_TICKS,
    SHOULDER_MAX_SPEED_TICKS,
    ELBOW_MAX_SPEED_TICKS,
    WRIST_MAX_SPEED_TICKS
  };

  long targets[NUM_MOTORS] = {b, s, e, w};
  long deltas[NUM_MOTORS];
  long pacerDuration = 1;
  long pacerSteps    = 1;   // only used for preset ramp scaling

  // Snapshot current positions atomically
  long curSteps[NUM_MOTORS];
  portENTER_CRITICAL(&timerMux);
  for (int i = 0; i < NUM_MOTORS; i++) curSteps[i] = motors[i].currentSteps;
  portEXIT_CRITICAL(&timerMux);

  for (int i = 0; i < NUM_MOTORS; i++) {
    deltas[i] = abs(targets[i] - curSteps[i]);
    long duration = deltas[i] * (long)motorCaps[i];
    if (duration > pacerDuration) {
      pacerDuration = duration;
      pacerSteps    = deltas[i];
    }
  }

  pacerDuration = (pacerDuration * 100L) / (long)vPct;

  for (int i = 0; i < NUM_MOTORS; i++) {
    int mTicks, mAccel;
    if (deltas[i] == 0) {
      mTicks = motorCaps[i];
      mAccel = 0;
    } else {
      mTicks = (int)((pacerDuration + deltas[i] - 1) / deltas[i]);
      // Preset (P:1): no cap — pacer unconstrained so all motors finish together.
      // Streaming (P:0): cap at MIN_SPEED_TICKS — prevents stall at very slow speeds.
      if (!isPreset && mTicks > MIN_SPEED_TICKS) mTicks = MIN_SPEED_TICKS;
      if (isPreset) {
        mAccel = (int)((long)ACCEL_STEPS * deltas[i] / pacerSteps);
        if (mAccel < 1) mAccel = 1;
      } else {
        mAccel = 0;   // streaming — no ramp
      }
    }
    setTarget(i, targets[i], mTicks, mAccel, 0);
  }

  // Mark preset in progress so loop() broadcasts DONE: when arm stops
  if (isPreset) {
    presetInProgress = true;
    wasMoving        = false;   // reset so transition is detected cleanly
  }

  lastCommandMs  = millis();
  watchdogArmed  = true;    // arm watchdog — spec 8.5: fires after first valid IK command
  watchdogFired  = false;   // re-arm — next disconnect will broadcast TIMEOUT again
  Serial.printf("OK B:%ld S:%ld E:%ld W:%ld\n", b, s, e, w);
}

// ═══════════════════════════════════════════════════════════════
// MANUAL JOG & STATUS
// ═══════════════════════════════════════════════════════════════

int activeMotor = MOTOR_SHOULDER;

void printStatus() {
  const char* names[] = {"BASE", "SHOULDER", "ELBOW", "WRIST"};
  long steps[NUM_MOTORS];
  portENTER_CRITICAL(&timerMux);
  for (int i = 0; i < NUM_MOTORS; i++) steps[i] = motors[i].currentSteps;
  portEXIT_CRITICAL(&timerMux);
  Serial.println("\n---- JOINT STATUS --------------------------------");
  for (int i = 0; i < NUM_MOTORS; i++) {
    Serial.printf("  %s: %ld steps  [%s]\n",
        names[i], steps[i],
        motors[i].homed ? "HOMED" : "NOT HOMED");
  }
  Serial.println("--------------------------------------------------\n");
}

void printHelp() {
  Serial.println("\n╔═══════════════════════════════════════════════╗");
  Serial.println("║     ESP32 ROBOT ARM MOTOR CONTROLLER         ║");
  Serial.println("╠═══════════════════════════════════════════════╣");
  Serial.println("║  SELECT:  1=BASE 2=SHOULDER 3=ELBOW 4=WRIST  ║");
  Serial.println("║  JOG:     w=forward  s=backward (50 steps)   ║");
  Serial.println("║  HOME:    h=selected motor  a=ALL motors      ║");
  Serial.println("║  INFO:    i=joint status    ?=this help       ║");
  Serial.println("║  PROMPT:  y=yes             n=no              ║");
  Serial.println("╠═══════════════════════════════════════════════╣");
  Serial.println("║  IK MODE (from laptop GUI):                   ║");
  Serial.println("║    B:<n> S:<n> E:<n> W:<n> G:<us> V:<pct>   ║");
  Serial.println("║    G=gripper µs (900-2500)  V=speed% (5-100) ║");
  Serial.println("║    Reply: OK B:<n> S:<n> E:<n> W:<n>         ║");
  Serial.println("║           ERR: <reason>                       ║");
  Serial.println("╚═══════════════════════════════════════════════╝\n");
}

void jogMotor(int motor, int direction) {
  if (!motors[motor].homed) {
    Serial.println("ERR: Motor not homed");
    return;
  }

  long jogSteps = 50;

  // Snapshot all positions atomically — used for both target computation and limit check
  long b, s, e, w;
  portENTER_CRITICAL(&timerMux);
  b = motors[MOTOR_BASE].currentSteps;
  s = motors[MOTOR_SHOULDER].currentSteps;
  e = motors[MOTOR_ELBOW].currentSteps;
  w = motors[MOTOR_WRIST].currentSteps;
  portEXIT_CRITICAL(&timerMux);

  long curSteps = (motor == MOTOR_BASE) ? b : (motor == MOTOR_SHOULDER) ? s :
                  (motor == MOTOR_ELBOW) ? e : w;
  long target = curSteps + direction * jogSteps;

  switch (motor) {
    case MOTOR_BASE:     b = target; break;
    case MOTOR_SHOULDER: s = target; break;
    case MOTOR_ELBOW:    e = target; break;
    case MOTOR_WRIST:    w = target; break;
  }

  char errMsg[100];
  if (!checkStepLimits(b, s, e, w, errMsg)) {
    Serial.printf("ERR: SOFT LIMIT %s\n", errMsg);
    return;
  }

  const int motorCaps[NUM_MOTORS] = {
    BASE_MAX_SPEED_TICKS, SHOULDER_MAX_SPEED_TICKS,
    ELBOW_MAX_SPEED_TICKS, WRIST_MAX_SPEED_TICKS
  };
  setTarget(motor, target, motorCaps[motor]);
  waitForStop();
  lastCommandMs = millis();  // reset watchdog — jog is a valid deliberate command
  printStatus();
}

void handleManualCommand(char cmd) {
  switch (cmd) {
    case '1': activeMotor = MOTOR_BASE;     Serial.println("Active: BASE");     break;
    case '2': activeMotor = MOTOR_SHOULDER; Serial.println("Active: SHOULDER"); break;
    case '3': activeMotor = MOTOR_ELBOW;    Serial.println("Active: ELBOW");    break;
    case '4': activeMotor = MOTOR_WRIST;    Serial.println("Active: WRIST");    break;
    case 'w': case 'W': jogMotor(activeMotor, +1); break;
    case 's': case 'S': jogMotor(activeMotor, -1); break;
    case 'h': case 'H': homeMotor(activeMotor);    break;
    case 'a': case 'A': homeAll();                 break;
    case 'i': case 'I': printStatus();             break;
    case '?':           printHelp();               break;
    case '\n': case '\r': break;
    default: break;
  }
}

// ═══════════════════════════════════════════════════════════════
// SETUP
// ═══════════════════════════════════════════════════════════════

void setup() {
  delay(2000);
  Serial.begin(115200);
  Serial.println("BOOT: ESP32 RESET");   // synchronisation signal — Python pauses on this
  Serial.println("\nESP32 ROBOT ARM BOOTING...");

  // Initialise PCA9685 servo driver
  Wire.begin(I2C_SDA, I2C_SCL);
  pwm.begin();
  pwm.setPWMFreq(PCA9685_FREQ);
  delay(500);                          // allow oscillator to stabilise after freq set — first write ignored without this
  // Startup: gripper fully open (900µs) per spec 9.11
  pwm.setPWM(GRIPPER_CHANNEL, 0, pulseUsToCounts(GRIPPER_PWM_MIN));
  Serial.println("PCA9685 gripper initialised.");

  // Initialise motor state structs
  motors[MOTOR_BASE]     = {0,0,0,0,0, BASE_MAX_SPEED_TICKS,     ACCEL_STEPS, 0, false, 0, false,
                             BASE_STEP_PIN,     BASE_DIR_PIN,     BASE_LIMIT_PIN,
                             BASE_HOME_DIR,     BASE_DIR_INVERT};
  motors[MOTOR_SHOULDER] = {0,0,0,0,0, SHOULDER_MAX_SPEED_TICKS, ACCEL_STEPS, 0, false, 0, false,
                             SHOULDER_STEP_PIN, SHOULDER_DIR_PIN, SHOULDER_LIMIT_PIN,
                             SHOULDER_HOME_DIR, SHOULDER_DIR_INVERT};
  motors[MOTOR_ELBOW]    = {0,0,0,0,0, ELBOW_MAX_SPEED_TICKS,    ACCEL_STEPS, 0, false, 0, false,
                             ELBOW_STEP_PIN,    ELBOW_DIR_PIN,    ELBOW_LIMIT_PIN,
                             ELBOW_HOME_DIR,    ELBOW_DIR_INVERT};
  motors[MOTOR_WRIST]    = {0,0,0,0,0, WRIST_MAX_SPEED_TICKS,    ACCEL_STEPS, 0, false, 0, false,
                             WRIST_STEP_PIN,    WRIST_DIR_PIN,    WRIST_LIMIT_PIN,
                             WRIST_HOME_DIR,    WRIST_DIR_INVERT};

  // Configure pins
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(motors[i].stepPin,  OUTPUT);
    pinMode(motors[i].dirPin,   OUTPUT);
    pinMode(motors[i].limitPin, INPUT_PULLUP);
    digitalWrite(motors[i].stepPin, LOW);
    digitalWrite(motors[i].dirPin,  LOW);
  }

  // Configure ESTOP pin
  pinMode(ESTOP_PIN, INPUT_PULLUP);

  // Start hardware timer (Timer 0, prescaler 80 → 1 µs ticks)
  // NOTE: timer is INITIALISED here but NOT enabled — it must only start
  // after homing completes (spec 9.2). timerAlarmEnable() called in homeAll().
  stepTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(stepTimer, &onTimer, true);
  timerAlarmWrite(stepTimer, TIMER_INTERVAL_US, true);
  // timerAlarmEnable — deferred to homeAll()
  Serial.println("Timer interrupt initialised (not yet started).");

  lastCommandMs = millis();
  printHelp();

  // Show startup safety prompt; home if user confirms
  if (startupSafetyPrompt()) {
    homeAll();
  }
}

// ═══════════════════════════════════════════════════════════════
// MAIN LOOP
// ═══════════════════════════════════════════════════════════════

void loop() {
  // ESTOP check — highest priority
  if (estopTriggered()) {
    triggerEstop();
    // Hold here until ESTOP releases, then require re-home
    while (estopTriggered()) delay(10);
    Serial.println("ESTOP released. Send 'a' to re-home.");
    // Mark all motors un-homed so IK commands are rejected until re-home
    for (int i = 0; i < NUM_MOTORS; i++) motors[i].homed = false;
    return;
  }

  // Apply deferred gripper write (I2C must not be called from ISR)
  applyGripper();

  // STATE broadcast every 500ms — Option A synchronisation (spec 8.4)
  if (allHomed() && (millis() - stateBroadcastMs > STATE_BROADCAST_MS)) {
    stateBroadcastMs = millis();
    long b, s, e, w;
    portENTER_CRITICAL(&timerMux);
    b = motors[MOTOR_BASE].currentSteps;
    s = motors[MOTOR_SHOULDER].currentSteps;
    e = motors[MOTOR_ELBOW].currentSteps;
    w = motors[MOTOR_WRIST].currentSteps;
    portEXIT_CRITICAL(&timerMux);
    Serial.printf("STATE: B:%ld S:%ld E:%ld W:%ld\n", b, s, e, w);
  }

  // DONE: broadcast — fires once when a P:1 preset move completes.
  // Detects allStopped() transition (wasMoving → stopped) so Python knows
  // to fire keepalive immediately and prevent watchdog timeout.
  if (presetInProgress && allHomed()) {
    bool stopped = allStopped();
    if (!stopped) {
      wasMoving = true;   // arm is moving — latch so we detect the transition
    } else if (wasMoving && stopped) {
      // Transition: was moving, now stopped — preset complete
      long b, s, e, w;
      portENTER_CRITICAL(&timerMux);
      b = motors[MOTOR_BASE].currentSteps;
      s = motors[MOTOR_SHOULDER].currentSteps;
      e = motors[MOTOR_ELBOW].currentSteps;
      w = motors[MOTOR_WRIST].currentSteps;
      portEXIT_CRITICAL(&timerMux);
      Serial.printf("DONE: B:%ld S:%ld E:%ld W:%ld\n", b, s, e, w);
      lastCommandMs    = millis();   // reset watchdog clock
      watchdogArmed    = false;      // disarm — arm is idle deliberately after preset
                                     // re-arms on next IK command (keepalive or vision)
      presetInProgress = false;
      wasMoving        = false;
    }
  }

  // Watchdog: arms after first valid IK command (spec 8.5).
  // Only fires if armed — prevents spurious timeout between homing and first command.
  if (watchdogArmed && allHomed() && allStopped() && (millis() - lastCommandMs > WATCHDOG_MS)) {
    lastCommandMs = millis();
    if (!watchdogFired) {
      watchdogFired = true;
      Serial.println("WATCHDOG: TIMEOUT");  // spec 8.3 — fires once per disconnect
    }
    for (int i = 0; i < NUM_MOTORS; i++) {
      portENTER_CRITICAL(&timerMux);
      motors[i].targetSteps = motors[i].currentSteps;
      portEXIT_CRITICAL(&timerMux);
    }
  }

  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        String line = serialBuffer;
        serialBuffer = "";
        line.trim();

        if (line.indexOf("B:") >= 0) {
          // IK command from laptop GUI
          handleIKCommand(line);
        }
        else if (line.length() == 1) {
          handleManualCommand(line[0]);
        }
        else {
          Serial.println("ERR: Invalid command");
        }
      }
    }
    else {
      // Single-char manual commands fire immediately (no newline needed)
      if (serialBuffer.length() == 0 &&
          (c=='w'||c=='W'||c=='s'||c=='S'||
           c=='1'||c=='2'||c=='3'||c=='4'||
           c=='h'||c=='H'||c=='a'||c=='A'||
           c=='i'||c=='I'||c=='?'||
           c=='y'||c=='Y'||c=='n'||c=='N')) {
        handleManualCommand(c);
      }
      else {
        if (serialBuffer.length() >= 128) {
          serialBuffer = "";
          Serial.println("ERR: Buffer overflow");  // spec 8.3
        } else {
          serialBuffer += c;
        }
      }
    }
  }
}
