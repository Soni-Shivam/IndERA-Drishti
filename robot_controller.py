"""
Robot Arm Controller — IK + Arduino + Simulator + Conveyor Automation
======================================================================
Modes:
  1. MANUAL  — same as before: enter X,Y,Z, pick, place, home, grip
  2. CONVEYOR — fully automatic pick-and-place loop:
       conveyor runs → sensor detects object → belt stops →
       arm picks from CONVEYOR_PICK_POINT → places at CONVEYOR_PLACE_POINT →
       arm goes home → belt restarts → repeat forever

All conveyor settings live in arm_common.py:
  CONVEYOR_SPEED, SENSOR_THRESHOLD_CM, SENSOR_STOP_DELAY_MS,
  CONVEYOR_PICK_POINT, CONVEYOR_PLACE_POINT

Usage:
  1. Upload arm_controller.ino to Arduino
  2. Run: python robot_controller.py
  3. Type 'conveyor' to start automation, or use manual commands

Requirements:
  pip install roboticstoolbox-python spatialmath-python swift-sim pyserial
"""

import sys
import time
import math
import numpy as np

try:
    import serial
except ImportError:
    print("ERROR: pyserial not found. Install with: pip install pyserial")
    sys.exit(1)

try:
    import swift
except ImportError:
    print("ERROR: swift-sim not found. Install with: pip install swift-sim")
    sys.exit(1)

from arm_common import (
    load_robot, robust_ik, ik_to_servo_angles, servo_to_ik_angles,
    user_to_robot_coords,
    SERIAL_PORT, SERIAL_BAUD, SERVO_SPEED,
    HOME_ANGLES, JOINT_NAMES, JOINT_IDS,
    MOVE_ORDER, NUM_ARM_JOINTS, END_LINK,
    # Conveyor settings
    CONVEYOR_SPEED, SENSOR_THRESHOLD_CM, SENSOR_STOP_DELAY_MS,
    CONVEYOR_PICK_POINT, CONVEYOR_PLACE_POINT,
)


# =========================================================================
# ARDUINO COMMUNICATION
# =========================================================================

class ArduinoController:
    def __init__(self, port, baud):
        self.ser = None
        self.connected = False
        try:
            self.ser = serial.Serial(port, baud, timeout=2)
            time.sleep(2)

            start = time.time()
            while time.time() - start < 5:
                if self.ser.in_waiting:
                    line = self._read_line()
                    if line == "READY":
                        self.connected = True
                        print("✅ Arduino connected and ready")
                        return

            print("⚠ Arduino connected but no READY signal")
            self.connected = True
        except Exception as e:
            print(f"⚠ Arduino not connected: {e}")
            self.connected = False

    def _read_line(self):
        try:
            raw = self.ser.readline()
            return raw.decode('utf-8', errors='ignore').strip()
        except Exception:
            return ""

    def send_angles(self, servo_angles, gripper_angle=150):
        if not self.connected:
            print("  [Arduino not connected — skipping]")
            return []
        all_angles = list(servo_angles) + [gripper_angle]
        cmd = ",".join(str(a) for a in all_angles)
        self.ser.write(f"{cmd}\n".encode())
        return self._read_until_done(timeout=30)

    def send_raw_command(self, cmd_str):
        if not self.connected:
            return []
        self.ser.write(f"{cmd_str}\n".encode())
        return self._read_until_done(timeout=30)

    def send_home(self):
        if not self.connected:
            return []
        self.ser.write(b"HOME\n")
        return self._read_until_done(timeout=30)

    def set_speed(self, speed_ms):
        if not self.connected:
            return
        self.ser.write(f"SPEED:{speed_ms}\n".encode())
        time.sleep(0.1)
        if self.ser.in_waiting:
            self._read_line()

    def wait_for_joint_done(self, timeout=10):
        if not self.connected:
            return None
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self._read_line()
                if line.startswith("DONE:") or line == "ALL_DONE":
                    return line
                if line:
                    print(f"  [Arduino] {line}")
            time.sleep(0.01)
        return None

    def _read_until_done(self, timeout=30):
        messages = []
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self._read_line()
                if line:
                    messages.append(line)
                if line == "ALL_DONE":
                    break
            time.sleep(0.01)
        return messages

    # ---- CONVEYOR COMMANDS ----

    def conveyor_on(self):
        """Start conveyor belt."""
        if not self.connected:
            return
        self.ser.write(b"CONVEYOR:ON\n")
        self._wait_for_ack("CONVEYOR_ON", timeout=3)

    def conveyor_off(self):
        """Stop conveyor belt."""
        if not self.connected:
            return
        self.ser.write(b"CONVEYOR:OFF\n")
        self._wait_for_ack("CONVEYOR_OFF", timeout=3)

    def set_conveyor_speed(self, speed):
        """Set conveyor PWM speed (0-255)."""
        if not self.connected:
            return
        self.ser.write(f"CONVEYOR:SPEED:{speed}\n".encode())
        self._wait_for_ack("CONVEYOR_SPEED_SET", timeout=3)

    def sensor_on(self):
        """Enable continuous object detection on Arduino."""
        if not self.connected:
            return
        self.ser.write(b"SENSOR:ON\n")
        self._wait_for_ack("SENSOR_ON", timeout=3)

    def sensor_off(self):
        """Disable object detection."""
        if not self.connected:
            return
        self.ser.write(b"SENSOR:OFF\n")
        self._wait_for_ack("SENSOR_OFF", timeout=3)

    def set_sensor_threshold(self, cm):
        """Set detection distance threshold (cm)."""
        if not self.connected:
            return
        self.ser.write(f"SENSOR:THRESHOLD:{cm}\n".encode())
        self._wait_for_ack("THRESHOLD_SET", timeout=3)

    def set_sensor_delay(self, ms):
        """Set delay (ms) between detection and conveyor stop."""
        if not self.connected:
            return
        self.ser.write(f"SENSOR:DELAY:{ms}\n".encode())
        self._wait_for_ack("SENSOR_DELAY_SET", timeout=3)

    def poll_distance(self):
        """Read a single distance from ultrasonic sensor. Returns cm or -1."""
        if not self.connected:
            return -1
        self.ser.write(b"SENSOR:POLL\n")
        start = time.time()
        while time.time() - start < 3:
            if self.ser.in_waiting:
                line = self._read_line()
                if line.startswith("DISTANCE:"):
                    try:
                        return float(line.split(":")[1])
                    except ValueError:
                        return -1
            time.sleep(0.01)
        return -1

    def wait_for_object_detected(self, timeout=300):
        """
        Block until Arduino sends OBJECT_DETECTED + CONVEYOR_STOPPED.
        Returns True if detected, False on timeout.
        Timeout default is 5 minutes (conveyor may run a while).
        """
        if not self.connected:
            return False
        detected = False
        stopped = False
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self._read_line()
                if line == "OBJECT_DETECTED":
                    detected = True
                    print("  📦 Object detected by sensor!")
                elif line == "CONVEYOR_STOPPED":
                    stopped = True
                    print("  🛑 Conveyor stopped.")
                elif line:
                    print(f"  [Arduino] {line}")

                if detected and stopped:
                    return True
            time.sleep(0.01)
        return False

    def _wait_for_ack(self, expected_prefix, timeout=3):
        """Wait for an acknowledgement message from Arduino."""
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self._read_line()
                if line.startswith(expected_prefix):
                    return line
                if line:
                    print(f"  [Arduino] {line}")
            time.sleep(0.01)
        return None

    def close(self):
        if self.ser:
            self.ser.close()


# =========================================================================
# SYNCHRONIZED SIMULATOR
# =========================================================================

class SyncSimulator:
    def __init__(self, robot):
        self.robot = robot
        self.env = None
        self.current_q = np.zeros(robot.n)
        self.robot.q = self.current_q

    def launch(self):
        print("Launching Swift simulator...")
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.env.add(self.robot)
        self.env.step(0.05)

    def move_joint_smooth(self, joint_idx, target_rad, steps=50):
        if joint_idx >= self.robot.n:
            return
        start_val = self.current_q[joint_idx]
        for i in range(1, steps + 1):
            frac = i / steps
            self.current_q[joint_idx] = start_val + (target_rad - start_val) * frac
            self.robot.q = self.current_q.copy()
            self.env.step(0.02)

    def move_to_home(self):
        home_angles = np.zeros(NUM_ARM_JOINTS)
        for idx in MOVE_ORDER:
            self.move_joint_smooth(idx, home_angles[idx], steps=50)

    def move_to_angles(self, ik_arm):
        for idx in MOVE_ORDER:
            if idx < len(ik_arm):
                self.move_joint_smooth(idx, ik_arm[idx], steps=50)

    def get_current_arm_angles(self):
        return self.current_q[:NUM_ARM_JOINTS].copy()

    def hold(self):
        if self.env:
            self.env.hold()


# =========================================================================
# MAIN CONTROLLER
# =========================================================================

def main():
    print("=" * 60)
    print("  ROBOT ARM CONTROLLER")
    print("  IK + Arduino + Simulator + Conveyor Automation")
    print("=" * 60)
    print()

    # Load robot
    robot = load_robot()
    print(f"Robot loaded: {robot.name}, {robot.n} joints")

    # Verify joint order
    print("\nVerifying joint mapping:")
    joint_names_urdf = [link.name for link in robot if hasattr(link, 'jtype') and link.jtype in ('R', 'P')]
    expected = ["base_rotate", "shoulder_joint", "elbow_joint", "wrist_roll_joint", "wrist_pitch_joint"]
    for i, (got, want) in enumerate(zip(joint_names_urdf, expected)):
        status = "✅" if got == want else "❌ MISMATCH"
        print(f"  Joint {i}: {got} {status}")
    print()

    # Connect Arduino
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    arduino = ArduinoController(SERIAL_PORT, SERIAL_BAUD)
    arduino.set_speed(SERVO_SPEED)

    # Launch simulator
    sim = SyncSimulator(robot)
    sim.launch()

    print("\n=== WORKSPACE INFO ===")
    print("  Max reach: ~30cm from shoulder")
    print("  Base height: ~8cm")
    print("  Coordinates in CENTIMETERS: X=forward, Y=left/right, Z=up")
    print()

    # ---- CONFIG ----
    LIFT_HEIGHT_CM = 10.0
    GRIP_SETTLE_SEC = 2.0

    # ---- GRIPPER STATE ----
    gripper_state = 150

    # ==================================================================
    # HELPER FUNCTIONS
    # ==================================================================

    def move_to_position(x_cm, y_cm, z_cm, gripper_angle=None):
        nonlocal gripper_state
        if gripper_angle is None:
            gripper_angle = gripper_state
        x_r, y_r, z_r = user_to_robot_coords(x_cm, y_cm, z_cm)
        target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)

        print(f"\n  Target (you):   ({x_cm:.1f}, {y_cm:.1f}, {z_cm:.1f}) cm")
        print(f"  Target (robot): ({x_r:.1f}, {y_r:.1f}, {z_r:.1f}) cm")

        print("  Solving IK...")
        ik_arm, err = robust_ik(robot, target_m, max_attempts=50)

        if ik_arm is None:
            print("\n  ❌ Point is NOT reachable!")
            return None

        print(f"  ✅ Reachable! (error: {err * 100:.3f} cm)")

        servo_angles = ik_to_servo_angles(ik_arm)
        ik_deg = np.degrees(ik_arm)

        print(f"\n  {'Joint':<14} {'IK (deg)':>10} {'Servo':>8}")
        print(f"  {'-' * 34}")
        for i in range(NUM_ARM_JOINTS):
            print(f"  {JOINT_NAMES[i]:<14} {ik_deg[i]:>+10.1f}° {servo_angles[i]:>7}°")

        print(f"\n  Moving (Elbow→Shoulder→Base→WristRoll→WristPitch)...")

        if arduino.connected:
            all_servo = servo_angles + [gripper_angle]
            cmd = ",".join(str(a) for a in all_servo)
            arduino.ser.write(f"{cmd}\n".encode())

        for order_idx in MOVE_ORDER:
            if arduino.connected:
                msg = arduino.wait_for_joint_done(timeout=10)
                if msg:
                    print(f"    [Arduino] {msg}")
                else:
                    print(f"    [Arduino] Timeout: {JOINT_NAMES[order_idx]}")

            sim.move_joint_smooth(order_idx, ik_arm[order_idx], steps=50)
            print(f"    [Sim] {JOINT_NAMES[order_idx]} → {ik_deg[order_idx]:+.1f}°")

        if arduino.connected:
            msg = arduino.wait_for_joint_done(timeout=10)
            if msg:
                print(f"    [Arduino] {msg}")

        print("  ✅ Move complete!")
        return ik_arm

    def find_lift_point(x_cm, y_cm, z_cm, gripper_angle=50):
        max_lift_z = z_cm + LIFT_HEIGHT_CM

        for try_z in [max_lift_z, z_cm + 7, z_cm + 5, z_cm + 3]:
            x_r, y_r, z_r = user_to_robot_coords(x_cm, y_cm, try_z)
            target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)
            ik_arm, err = robust_ik(robot, target_m, max_attempts=20)
            if ik_arm is not None:
                return x_cm, y_cm, try_z

        dist = math.sqrt(x_cm ** 2 + y_cm ** 2)
        if dist < 0.1:
            return None

        for shrink in [0.8, 0.6, 0.4, 0.2]:
            pull_x = x_cm * shrink
            pull_y = y_cm * shrink
            for try_z in [max_lift_z, z_cm + 7, z_cm + 5]:
                x_r, y_r, z_r = user_to_robot_coords(pull_x, pull_y, try_z)
                target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)
                ik_arm, err = robust_ik(robot, target_m, max_attempts=20)
                if ik_arm is not None:
                    return pull_x, pull_y, try_z

        return None

    def send_gripper(angle, label=""):
        nonlocal gripper_state
        current_arm = sim.get_current_arm_angles()
        servo_angles = ik_to_servo_angles(current_arm)
        msgs = arduino.send_angles(servo_angles, gripper_angle=angle)
        for m in msgs:
            print(f"    [Arduino] {m}")
        gripper_state = angle
        if label:
            print(f"  {label}")

    def do_pick(px, py, pz):
        """Execute full pick sequence. Returns True on success."""
        nonlocal gripper_state
        print("=" * 40)
        print(f"  PICK SEQUENCE: ({px}, {py}, {pz}) cm")
        print("=" * 40)

        # Step 1: Open gripper, approach above
        approach_z = pz + 3.0
        print(f"\n[Step 1/5] Approaching above object at Z={approach_z:.1f}cm...")
        send_gripper(150, "Gripper opened.")
        result = move_to_position(px, py, approach_z, gripper_angle=150)
        if result is None:
            print("❌ Pick aborted — approach point not reachable.")
            return False

        # Step 2: Lower to pick height
        print(f"\n[Step 2/5] Lowering to Z={pz:.1f}cm...")
        result = move_to_position(px, py, pz, gripper_angle=150)
        if result is None:
            print("❌ Pick aborted — pick point not reachable.")
            return False

        # Step 3: Close gripper
        print(f"\n[Step 3/5] Closing gripper...")
        send_gripper(50, "Gripper closed.")
        print(f"  Waiting {GRIP_SETTLE_SEC:.1f}s for firm grip...")
        time.sleep(GRIP_SETTLE_SEC)

        # Step 4: Lift
        print(f"\n[Step 4/5] Finding best lift point...")
        lift_point = find_lift_point(px, py, pz, gripper_angle=50)
        if lift_point is None:
            print("❌ Could not find a reachable lift point!")
            return False

        lx, ly, lz = lift_point
        if lx != px or ly != py:
            print(f"  Pulling inward to ({lx:.1f}, {ly:.1f}, {lz:.1f}) cm")
        else:
            print(f"  Lifting straight up to Z={lz:.1f}cm")

        result = move_to_position(lx, ly, lz, gripper_angle=50)
        if result is None:
            print("❌ Lift failed.")
            return False

        print(f"\n[Step 5/5] ✅ PICK COMPLETE!")
        return True

    def do_place(px, py, pz):
        """Execute full place sequence. Returns True on success."""
        nonlocal gripper_state
        print("=" * 40)
        print(f"  PLACE SEQUENCE: ({px}, {py}, {pz}) cm")
        print("=" * 40)

        # Step 1: Move above place point
        approach_z = pz + 3.0
        print(f"\n[Step 1/4] Moving above place point at Z={approach_z:.1f}cm...")
        result = move_to_position(px, py, approach_z, gripper_angle=50)
        if result is None:
            print("❌ Place aborted — approach not reachable.")
            return False

        # Step 2: Lower
        print(f"\n[Step 2/4] Lowering to Z={pz:.1f}cm...")
        result = move_to_position(px, py, pz, gripper_angle=50)
        if result is None:
            print("❌ Place aborted — place point not reachable.")
            return False

        # Step 3: Release
        print(f"\n[Step 3/4] Releasing object...")
        send_gripper(150, "Gripper opened — object released.")
        time.sleep(0.5)

        # Step 4: Lift away
        print(f"\n[Step 4/4] Lifting away...")
        lift_point = find_lift_point(px, py, pz, gripper_angle=150)
        if lift_point is not None:
            lx, ly, lz = lift_point
            move_to_position(lx, ly, lz, gripper_angle=150)
        else:
            print("⚠ Could not lift away, but object is placed.")

        print(f"\n✅ PLACE COMPLETE!")
        return True

    def do_home():
        """Go home and reset gripper."""
        nonlocal gripper_state
        print("Going home...")
        msgs = arduino.send_home()
        sim.move_to_home()
        gripper_state = 150
        for m in msgs:
            print(f"  [Arduino] {m}")
        print("Home reached.")

    # ==================================================================
    # CONVEYOR AUTOMATION LOOP
    # ==================================================================

    def run_conveyor_automation():
        """
        Fully automatic conveyor pick-and-place loop.

        Flow:
          1. Configure sensor threshold + delay on Arduino
          2. Start conveyor belt + enable sensor
          3. Wait for OBJECT_DETECTED + CONVEYOR_STOPPED from Arduino
          4. Pick from CONVEYOR_PICK_POINT
          5. Place at CONVEYOR_PLACE_POINT
          6. Go home
          7. Repeat from step 2

        Press Ctrl+C to stop and return to manual mode.
        """
        px, py, pz = CONVEYOR_PICK_POINT
        plx, ply, plz = CONVEYOR_PLACE_POINT

        print("\n" + "=" * 60)
        print("  CONVEYOR AUTOMATION MODE")
        print("=" * 60)
        print(f"  Conveyor speed:     {CONVEYOR_SPEED} (PWM 0-255)")
        print(f"  Sensor threshold:   {SENSOR_THRESHOLD_CM} cm")
        print(f"  Stop delay:         {SENSOR_STOP_DELAY_MS} ms")
        print(f"  Pick point:         ({px}, {py}, {pz}) cm")
        print(f"  Place point:        ({plx}, {ply}, {plz}) cm")
        print(f"\n  Press Ctrl+C at any time to stop and return to manual.\n")

        # Pre-check: verify both points are reachable before starting
        print("  Pre-checking pick point reachability...")
        x_r, y_r, z_r = user_to_robot_coords(px, py, pz)
        target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)
        test_ik, _ = robust_ik(robot, target_m, max_attempts=30)
        if test_ik is None:
            print(f"  ❌ Pick point ({px}, {py}, {pz}) is NOT reachable!")
            print("  Update CONVEYOR_PICK_POINT in arm_common.py and retry.")
            return

        print("  Pre-checking place point reachability...")
        x_r, y_r, z_r = user_to_robot_coords(plx, ply, plz)
        target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)
        test_ik, _ = robust_ik(robot, target_m, max_attempts=30)
        if test_ik is None:
            print(f"  ❌ Place point ({plx}, {ply}, {plz}) is NOT reachable!")
            print("  Update CONVEYOR_PLACE_POINT in arm_common.py and retry.")
            return

        print("  ✅ Both points reachable. Starting automation...\n")

        # Configure Arduino sensor
        arduino.set_conveyor_speed(CONVEYOR_SPEED)
        arduino.set_sensor_threshold(SENSOR_THRESHOLD_CM)
        arduino.set_sensor_delay(SENSOR_STOP_DELAY_MS)

        # Make sure arm is home before starting
        do_home()

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n{'─' * 50}")
                print(f"  CYCLE #{cycle_count}")
                print(f"{'─' * 50}")

                # Start conveyor + sensor
                print("\n  ▶ Starting conveyor belt...")
                arduino.conveyor_on()
                arduino.sensor_on()
                print("  ⏳ Waiting for object on conveyor...\n")

                # Block until object detected (5 min timeout)
                detected = arduino.wait_for_object_detected(timeout=300)

                if not detected:
                    print("  ⚠ Timeout — no object detected in 5 minutes.")
                    print("  Restarting cycle...")
                    continue

                # Small settle time for object to stop moving
                time.sleep(0.3)

                # PICK
                pick_ok = do_pick(px, py, pz)
                if not pick_ok:
                    print("  ⚠ Pick failed — going home, restarting cycle...")
                    do_home()
                    continue

                # PLACE
                place_ok = do_place(plx, ply, plz)
                if not place_ok:
                    print("  ⚠ Place failed — going home, restarting cycle...")
                    do_home()
                    continue

                # HOME
                do_home()

                print(f"\n  ✅ Cycle #{cycle_count} complete! Restarting conveyor...\n")

        except KeyboardInterrupt:
            print("\n\n  🛑 Conveyor automation stopped by user.")
            arduino.conveyor_off()
            arduino.sensor_off()
            do_home()
            print("  Returned to manual mode.\n")

    # ==================================================================
    # MAIN LOOP
    # ==================================================================

    while True:
        grip_label = "OPEN" if gripper_state >= 100 else "CLOSED"
        print("-" * 40)
        print(f"Gripper: {grip_label} ({gripper_state}°)")
        print("Commands:")
        print("  X,Y,Z        — move to position")
        print("  pick X,Y,Z   — pick object at position & lift")
        print("  place X,Y,Z  — place held object & lift")
        print("  home         — go to home position")
        print("  grip:open    — open gripper")
        print("  grip:close   — close gripper")
        print("  conveyor     — start conveyor automation loop")
        print("  sensor:poll  — read ultrasonic sensor distance")
        print("  belt:on      — manually start conveyor belt")
        print("  belt:off     — manually stop conveyor belt")
        print("  quit         — exit")
        print()

        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ('quit', 'q', 'exit'):
            break

        # ---- HOME ----
        if user_input.lower() == 'home':
            do_home()
            continue

        # ---- GRIPPER ----
        if user_input.lower() == 'grip:open':
            send_gripper(150, "Gripper opened.")
            continue

        if user_input.lower() == 'grip:close':
            send_gripper(50, "Gripper closed.")
            continue

        # ---- CONVEYOR AUTOMATION ----
        if user_input.lower() == 'conveyor':
            run_conveyor_automation()
            continue

        # ---- MANUAL BELT CONTROL ----
        if user_input.lower() == 'belt:on':
            arduino.set_conveyor_speed(CONVEYOR_SPEED)
            arduino.conveyor_on()
            print(f"  Conveyor ON (speed={CONVEYOR_SPEED})")
            continue

        if user_input.lower() == 'belt:off':
            arduino.conveyor_off()
            print("  Conveyor OFF")
            continue

        # ---- SENSOR POLL ----
        if user_input.lower() == 'sensor:poll':
            dist = arduino.poll_distance()
            if dist < 0:
                print("  Sensor: no echo (out of range or error)")
            else:
                print(f"  Sensor: {dist:.1f} cm")
            continue

        # ---- PICK COMMAND ----
        if user_input.lower().startswith('pick ') or user_input.lower().startswith('pick,'):
            try:
                coords_str = user_input[4:].strip()
                parts = coords_str.replace(',', ' ').split()
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                print("Usage: pick X,Y,Z  (e.g., pick 15 0 10)\n")
                continue
            do_pick(px, py, pz)
            continue

        # ---- PLACE COMMAND ----
        if user_input.lower().startswith('place ') or user_input.lower().startswith('place,'):
            try:
                coords_str = user_input[5:].strip()
                parts = coords_str.replace(',', ' ').split()
                px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                print("Usage: place X,Y,Z  (e.g., place 10 5 10)\n")
                continue
            do_place(px, py, pz)
            continue

        # ---- MANUAL MOVE (X,Y,Z) ----
        try:
            parts = user_input.replace(',', ' ').split()
            if len(parts) == 3:
                x_cm, y_cm, z_cm = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                x_cm = float(input("X (cm): ").strip())
                y_cm = float(input("Y (cm): ").strip())
                z_cm = float(input("Z (cm): ").strip())
        except ValueError:
            print("Invalid input. Enter numbers or a command.\n")
            continue

        move_to_position(x_cm, y_cm, z_cm)

    # Cleanup
    print("\nShutting down...")
    arduino.conveyor_off()
    arduino.sensor_off()
    arduino.close()
    print("Goodbye!")


if __name__ == "__main__":
    main()