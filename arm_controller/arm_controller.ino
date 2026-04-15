/*
 * arm_controller.ino — Arduino Servo Controller for 6-DOF Robot Arm
 * =================================================================
 * 
 * Protocol:
 *   - On boot: sends "READY\n"
 *   - Receives: "b,s,e,w,p,g\n" (6 servo angles, 0-180)
 *   - Moves joints in safe order: Elbow → Shoulder → Base → WristRoll → WristPitch → Gripper
 *   - Sends "DONE:X\n" after each joint (X = joint letter)
 *   - Sends "ALL_DONE\n" when all joints moved
 *   - Receives: "HOME\n" → moves all to 90° (gripper to 150°)
 *   - Receives: "SPEED:N\n" → sets delay per degree step to N ms
 *
 * Wiring:
 *   Pin 3  = Base (MG996R)       — powered externally at 5.5V
 *   Pin 5  = Shoulder (MG996R)
 *   Pin 6  = Elbow (MG996R)
 *   Pin 9  = Wrist Roll (SG90)
 *   Pin 10 = Wrist Pitch (SG90)
 *   Pin 11 = Gripper (SG90)
 *
 * IMPORTANT: MG996R servos MUST be powered externally (5-6V, 2A+).
 *            Do NOT power them from Arduino 5V pin.
 *            All servo GNDs must be connected to Arduino GND.
 */

#include <Servo.h>

// --- Pin assignments ---
const int SERVO_PINS[] = {3, 5, 6, 9, 10, 11};  // b, s, e, w, p, g
const int NUM_SERVOS = 6;

// --- Joint identifiers for protocol ---
const char JOINT_IDS[] = {'b', 's', 'e', 'w', 'p', 'g'};

// --- Safe movement order (indices into servo arrays) ---
// Elbow(2) → Shoulder(1) → Base(0) → WristRoll(3) → WristPitch(4) → Gripper(5)
const int MOVE_ORDER[] = {2, 1, 0, 3, 4, 5};

// --- Home positions ---
const int HOME_ANGLES[] = {90, 90, 90, 90, 90, 150};

// --- Movement speed ---
int stepDelay = 15;  // ms per degree step

// --- Servo objects ---
Servo servos[NUM_SERVOS];

// --- Current positions ---
int currentAngles[NUM_SERVOS];

// --- Input buffer ---
char inputBuffer[64];
int bufferPos = 0;


void setup() {
    Serial.begin(115200);

    // Attach servos and move to home
    for (int i = 0; i < NUM_SERVOS; i++) {
        servos[i].attach(SERVO_PINS[i]);
        servos[i].write(HOME_ANGLES[i]);
        currentAngles[i] = HOME_ANGLES[i];
    }

    delay(1000);  // Let servos settle
    Serial.println("READY");
}


void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (bufferPos > 0) {
                inputBuffer[bufferPos] = '\0';
                processCommand(inputBuffer);
                bufferPos = 0;
            }
        } else if (bufferPos < 63) {
            inputBuffer[bufferPos++] = c;
        }
    }
}


void processCommand(const char* cmd) {
    // --- HOME command ---
    if (strcmp(cmd, "HOME") == 0) {
        moveToHome();
        return;
    }

    // --- SPEED command ---
    if (strncmp(cmd, "SPEED:", 6) == 0) {
        int newSpeed = atoi(cmd + 6);
        if (newSpeed > 0 && newSpeed <= 100) {
            stepDelay = newSpeed;
            Serial.print("SPEED_SET:");
            Serial.println(stepDelay);
        }
        return;
    }

    // --- Angle command: "b,s,e,w,p,g" ---
    int targetAngles[NUM_SERVOS];
    if (parseAngles(cmd, targetAngles)) {
        moveToAngles(targetAngles);
    }
}


bool parseAngles(const char* cmd, int* angles) {
    // Parse 6 comma-separated integers
    int count = 0;
    const char* ptr = cmd;

    while (count < NUM_SERVOS && *ptr) {
        angles[count] = atoi(ptr);

        // Clamp to safe range
        if (angles[count] < 0) angles[count] = 0;
        if (angles[count] > 180) angles[count] = 180;

        count++;

        // Skip to next comma or end
        while (*ptr && *ptr != ',') ptr++;
        if (*ptr == ',') ptr++;
    }

    return (count == NUM_SERVOS);
}


void moveToAngles(int* targetAngles) {
    // Move each joint in safe order, one at a time
    for (int i = 0; i < NUM_SERVOS; i++) {
        int idx = MOVE_ORDER[i];
        smoothMove(idx, targetAngles[idx]);

        // Report completion of this joint
        Serial.print("DONE:");
        Serial.println(JOINT_IDS[idx]);
    }
    Serial.println("ALL_DONE");
}


void moveToHome() {
    int homeTarget[NUM_SERVOS];
    for (int i = 0; i < NUM_SERVOS; i++) {
        homeTarget[i] = HOME_ANGLES[i];
    }
    moveToAngles(homeTarget);
}


void smoothMove(int servoIdx, int targetAngle) {
    int current = currentAngles[servoIdx];

    if (current == targetAngle) return;

    int step = (targetAngle > current) ? 1 : -1;

    while (current != targetAngle) {
        current += step;
        servos[servoIdx].write(current);
        currentAngles[servoIdx] = current;
        delay(stepDelay);
    }
}
