/*
  arm_controller.ino — Robot Arm + Conveyor Belt + Ultrasonic Sensor
  ===================================================================

  SERVO PINS:
    Base=3, Shoulder=5, Elbow=6, WristRoll=9, WristPitch=10, Gripper=11

  CONVEYOR (BO Motor via L298N / L293D):
    ENA = 7   (digital ON/OFF — pin 7 has no PWM on Uno)
    IN1 = 8   (direction pin 1)
    IN2 = 12  (direction pin 2)

  ULTRASONIC (HC-SR04):
    TRIG = 4
    ECHO = 2

  PROTOCOL (new commands):
    CONVEYOR:ON           → start conveyor at current speed
    CONVEYOR:OFF          → stop conveyor
    CONVEYOR:SPEED:N      → set PWM speed 0-255
    SENSOR:ON             → start continuous object detection
    SENSOR:OFF            → stop detection
    SENSOR:THRESHOLD:N    → set detection distance in cm (default 5)
    SENSOR:DELAY:N        → set delay (ms) between detection and stop (default 0)
    SENSOR:POLL           → single distance reading, returns DISTANCE:xx.x

  Arduino sends:
    OBJECT_DETECTED       → when sensor detects object within threshold
    CONVEYOR_STOPPED      → after conveyor stops due to detection
    DISTANCE:xx.x         → response to SENSOR:POLL

  Existing protocol unchanged:
    b,s,e,w,p,g           → move servos (sends DONE:X after each, ALL_DONE at end)
    HOME                  → all servos to home
    SPEED:N               → servo step delay in ms
    READY                 → sent on boot
*/

#include <Servo.h>

// ---- SERVO PINS ----
Servo servos[6];
const int SERVO_PINS[6] = {3, 5, 6, 9, 10, 11};
int currentAngles[6]    = {90, 90, 90, 90, 90, 150};
int targetAngles[6]     = {90, 90, 90, 90, 90, 150};

// Move order: Elbow(2) → Shoulder(1) → Base(0) → WristRoll(3) → WristPitch(4)
const int MOVE_ORDER[5] = {2, 1, 0, 3, 4};
const char JOINT_LETTERS[6] = {'b', 's', 'e', 'w', 'p', 'g'};

int stepDelay = 15;  // ms per degree

// ---- CONVEYOR MOTOR (L298N / L293D) ----
const int CONV_ENA = 7;   // digital ON/OFF (no PWM on pin 7)
const int CONV_IN1 = 8;   // direction
const int CONV_IN2 = 12;  // direction

bool conveyorRunning = false;
int  conveyorSpeed   = 200;  // PWM 0-255

// ---- ULTRASONIC SENSOR (HC-SR04) ----
const int TRIG_PIN = 4;
const int ECHO_PIN = 2;

bool sensorActive       = false;
float detectionThreshold = 5.0;   // cm
unsigned long sensorDelay = 0;    // ms delay between detection and conveyor stop
unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 50;  // ms between readings when active

// ---- INPUT BUFFER ----
String inputBuffer = "";


// ===================================================================
// SETUP
// ===================================================================
void setup() {
  Serial.begin(115200);

  // Attach servos
  for (int i = 0; i < 6; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(currentAngles[i]);
  }

  // Conveyor motor pins
  pinMode(CONV_ENA, OUTPUT);
  pinMode(CONV_IN1, OUTPUT);
  pinMode(CONV_IN2, OUTPUT);
  digitalWrite(CONV_IN1, LOW);
  digitalWrite(CONV_IN2, LOW);
  digitalWrite(CONV_ENA, LOW);

  // Ultrasonic pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  delay(500);
  Serial.println("READY");
}


// ===================================================================
// MAIN LOOP
// ===================================================================
void loop() {
  // ---- Read serial commands ----
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  // ---- Continuous sensor monitoring ----
  if (sensorActive && conveyorRunning) {
    unsigned long now = millis();
    if (now - lastSensorRead >= SENSOR_INTERVAL) {
      lastSensorRead = now;
      float dist = readDistanceCm();

      if (dist > 0 && dist <= detectionThreshold) {
        // Object detected!
        Serial.println("OBJECT_DETECTED");

        // Apply configured delay before stopping
        if (sensorDelay > 0) {
          delay(sensorDelay);
        }

        // Stop conveyor
        stopConveyor();
        Serial.println("CONVEYOR_STOPPED");

        // Stop sensing until Python tells us to resume
        sensorActive = false;
      }
    }
  }
}


// ===================================================================
// COMMAND PROCESSING
// ===================================================================
void processCommand(String cmd) {

  // ---- HOME ----
  if (cmd == "HOME") {
    int homeAngles[6] = {90, 90, 90, 90, 90, 150};
    for (int i = 0; i < 6; i++) targetAngles[i] = homeAngles[i];

    // Move arm joints in safe order
    for (int i = 0; i < 5; i++) {
      int idx = MOVE_ORDER[i];
      moveServoSmooth(idx);
      Serial.print("DONE:");
      Serial.println(JOINT_LETTERS[idx]);
    }
    // Gripper
    moveServoSmooth(5);
    Serial.print("DONE:");
    Serial.println(JOINT_LETTERS[5]);
    Serial.println("ALL_DONE");
    return;
  }

  // ---- SPEED ----
  if (cmd.startsWith("SPEED:")) {
    stepDelay = cmd.substring(6).toInt();
    if (stepDelay < 1) stepDelay = 1;
    Serial.print("SPEED_SET:");
    Serial.println(stepDelay);
    return;
  }

  // ---- CONVEYOR:ON ----
  if (cmd == "CONVEYOR:ON") {
    startConveyor();
    Serial.println("CONVEYOR_ON");
    return;
  }

  // ---- CONVEYOR:OFF ----
  if (cmd == "CONVEYOR:OFF") {
    stopConveyor();
    Serial.println("CONVEYOR_OFF");
    return;
  }

  // ---- CONVEYOR:SPEED:N ----
  if (cmd.startsWith("CONVEYOR:SPEED:")) {
    conveyorSpeed = cmd.substring(15).toInt();
    conveyorSpeed = constrain(conveyorSpeed, 0, 255);
    // Pin 7 is digital-only, so speed is stored but not PWM-controllable.
    // Motor runs at full voltage when ON.
    Serial.print("CONVEYOR_SPEED_SET:");
    Serial.println(conveyorSpeed);
    return;
  }

  // ---- SENSOR:ON ----
  if (cmd == "SENSOR:ON") {
    sensorActive = true;
    Serial.println("SENSOR_ON");
    return;
  }

  // ---- SENSOR:OFF ----
  if (cmd == "SENSOR:OFF") {
    sensorActive = false;
    Serial.println("SENSOR_OFF");
    return;
  }

  // ---- SENSOR:THRESHOLD:N ----
  if (cmd.startsWith("SENSOR:THRESHOLD:")) {
    detectionThreshold = cmd.substring(17).toFloat();
    if (detectionThreshold < 0.5) detectionThreshold = 0.5;
    Serial.print("THRESHOLD_SET:");
    Serial.println(detectionThreshold);
    return;
  }

  // ---- SENSOR:DELAY:N ----
  if (cmd.startsWith("SENSOR:DELAY:")) {
    sensorDelay = cmd.substring(13).toInt();
    Serial.print("SENSOR_DELAY_SET:");
    Serial.println(sensorDelay);
    return;
  }

  // ---- SENSOR:POLL ----
  if (cmd == "SENSOR:POLL") {
    float dist = readDistanceCm();
    Serial.print("DISTANCE:");
    Serial.println(dist, 1);
    return;
  }

  // ---- SERVO ANGLES: b,s,e,w,p,g ----
  int angles[6];
  int count = 0;
  int startIdx = 0;

  for (int i = 0; i <= (int)cmd.length(); i++) {
    if (i == (int)cmd.length() || cmd.charAt(i) == ',') {
      if (count < 6) {
        angles[count] = cmd.substring(startIdx, i).toInt();
        count++;
      }
      startIdx = i + 1;
    }
  }

  if (count == 6) {
    for (int i = 0; i < 6; i++) {
      targetAngles[i] = constrain(angles[i], 0, 180);
    }

    // Move arm joints in safe order
    for (int i = 0; i < 5; i++) {
      int idx = MOVE_ORDER[i];
      moveServoSmooth(idx);
      Serial.print("DONE:");
      Serial.println(JOINT_LETTERS[idx]);
    }

    // Move gripper last
    moveServoSmooth(5);
    Serial.print("DONE:");
    Serial.println(JOINT_LETTERS[5]);

    Serial.println("ALL_DONE");
  }
}


// ===================================================================
// SERVO MOVEMENT
// ===================================================================
void moveServoSmooth(int idx) {
  int current = currentAngles[idx];
  int target  = targetAngles[idx];

  if (current == target) return;

  int step = (target > current) ? 1 : -1;
  while (current != target) {
    current += step;
    servos[idx].write(current);
    delay(stepDelay);
  }
  currentAngles[idx] = target;
}


// ===================================================================
// CONVEYOR MOTOR
// ===================================================================
void startConveyor() {
  digitalWrite(CONV_IN2, HIGH);
  digitalWrite(CONV_IN1, LOW);
  digitalWrite(CONV_ENA, HIGH);  // Pin 7 = digital only, full speed
  conveyorRunning = true;
}

void stopConveyor() {
  digitalWrite(CONV_IN1, LOW);
  digitalWrite(CONV_IN2, LOW);
  digitalWrite(CONV_ENA, LOW);  // Pin 7 = digital only
  conveyorRunning = false;
}


// ===================================================================
// ULTRASONIC SENSOR
// ===================================================================
float readDistanceCm() {
  // Send 10us trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo — timeout at 30ms (~5m max)
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return -1.0;  // No echo (out of range)
  }

  // Speed of sound: 343 m/s → 0.0343 cm/us → distance = duration * 0.0343 / 2
  float distance = duration * 0.0343 / 2.0;
  return distance;
}
