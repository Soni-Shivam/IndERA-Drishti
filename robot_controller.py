"""
Robot Arm Controller — IK + Arduino + Synchronized Simulator
=============================================================
1. Takes X, Y, Z coordinates (in cm)
2. Runs IK solver to check if reachable
3. Converts IK angles (radians) to servo angles (0-180°)
4. Sends servo angles to Arduino
5. Arduino moves each motor one-by-one (safe order)
6. Simulator mirrors each motor movement in sync

Usage:
  1. Upload arm_controller.ino to Arduino
  2. Run: python robot_controller.py
  3. Enter X, Y, Z coordinates when prompted

Requirements:
  pip install roboticstoolbox-python spatialmath-python swift-sim pyserial
"""

import sys
import time
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

# Import everything from the shared module — single source of truth
from arm_common import (
    load_robot, robust_ik, ik_to_servo_angles, servo_to_ik_angles,
    SERIAL_PORT, SERIAL_BAUD, SERVO_SPEED,
    HOME_ANGLES, JOINT_NAMES, JOINT_IDS,
    MOVE_ORDER, NUM_ARM_JOINTS, END_LINK,
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
            time.sleep(2)  # Wait for Arduino reset

            # Wait for READY signal
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
        """Read a line from serial with safe decoding."""
        try:
            raw = self.ser.readline()
            return raw.decode('utf-8', errors='ignore').strip()
        except Exception:
            return ""

    def send_angles(self, servo_angles, gripper_angle=150):
        """
        Send 6 servo angles to Arduino.
        servo_angles: list of 5 arm servo angles (0-180)
        gripper_angle: gripper servo angle (50-150)

        Returns list of progress messages.
        """
        if not self.connected:
            print("  [Arduino not connected — skipping]")
            return []

        all_angles = list(servo_angles) + [gripper_angle]
        cmd = ",".join(str(a) for a in all_angles)
        self.ser.write(f"{cmd}\n".encode())

        return self._read_until_done(timeout=30)

    def send_raw_command(self, cmd_str):
        """Send a raw command string and return progress messages."""
        if not self.connected:
            return []
        self.ser.write(f"{cmd_str}\n".encode())
        return self._read_until_done(timeout=30)

    def send_home(self):
        """Send HOME command."""
        if not self.connected:
            return []
        self.ser.write(b"HOME\n")
        return self._read_until_done(timeout=30)

    def set_speed(self, speed_ms):
        """Set movement speed (ms per degree)."""
        if not self.connected:
            return
        self.ser.write(f"SPEED:{speed_ms}\n".encode())
        time.sleep(0.1)
        if self.ser.in_waiting:
            self._read_line()

    def wait_for_joint_done(self, timeout=10):
        """
        Wait for the next DONE: or ALL_DONE message.
        Returns the message, or None on timeout.
        """
        if not self.connected:
            return None
        start = time.time()
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                line = self._read_line()
                if line.startswith("DONE:") or line == "ALL_DONE":
                    return line
                # Print other messages (debugging, etc.)
                if line:
                    print(f"  [Arduino] {line}")
            time.sleep(0.01)  # Don't busy-wait
        return None  # Timeout

    def _read_until_done(self, timeout=30):
        """Read all messages until ALL_DONE or timeout."""
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
        """Launch Swift simulator."""
        print("Launching Swift simulator...")
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        self.env.add(self.robot)
        self.env.step(0.05)

    def move_joint_smooth(self, joint_idx, target_rad, steps=50):
        """
        Animate a single joint moving to target angle.
        joint_idx: index into the robot's joint array
        target_rad: target angle in radians
        """
        if joint_idx >= self.robot.n:
            return

        start_val = self.current_q[joint_idx]

        for i in range(1, steps + 1):
            frac = i / steps
            self.current_q[joint_idx] = start_val + (target_rad - start_val) * frac
            self.robot.q = self.current_q.copy()
            self.env.step(0.02)

    def move_to_home(self):
        """Animate all arm joints back to zero (home)."""
        home_angles = np.zeros(NUM_ARM_JOINTS)
        for idx in MOVE_ORDER:
            self.move_joint_smooth(idx, home_angles[idx], steps=50)
            print(f"  [Sim] {JOINT_NAMES[idx]} → home")

    def move_to_angles(self, ik_arm):
        """
        Move arm joints one by one in safe order, animating each.
        ik_arm: array of 5 IK angles in radians
        """
        for idx in MOVE_ORDER:
            if idx < len(ik_arm):
                self.move_joint_smooth(idx, ik_arm[idx], steps=50)
                print(f"  [Sim] {JOINT_NAMES[idx]} → {np.degrees(ik_arm[idx]):+.1f}°")

    def get_current_arm_angles(self):
        """Return current arm joint angles (radians) for servo conversion."""
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
    print("  IK + Arduino + Synchronized Simulator")
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
    print("  Enter coordinates in CENTIMETERS")
    print("  X = forward, Y = left/right, Z = up/down")
    print()

    while True:
        print("-" * 40)
        print("Commands:")
        print("  Enter X,Y,Z  — move to position")
        print("  home         — go to home position")
        print("  grip:open    — open gripper")
        print("  grip:close   — close gripper")
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
            print("Going home...")
            msgs = arduino.send_home()
            sim.move_to_home()
            for m in msgs:
                print(f"  [Arduino] {m}")
            print("Home reached.")
            continue

        # ---- GRIPPER ----
        if user_input.lower() == 'grip:open':
            # Use the CORRECT conversion function for current arm angles
            current_arm = sim.get_current_arm_angles()
            servo_angles = ik_to_servo_angles(current_arm)
            msgs = arduino.send_angles(servo_angles, gripper_angle=150)
            for m in msgs:
                print(f"  [Arduino] {m}")
            print("Gripper opened.")
            continue

        if user_input.lower() == 'grip:close':
            current_arm = sim.get_current_arm_angles()
            servo_angles = ik_to_servo_angles(current_arm)
            msgs = arduino.send_angles(servo_angles, gripper_angle=50)
            for m in msgs:
                print(f"  [Arduino] {m}")
            print("Gripper closed.")
            continue

        # ---- PARSE X, Y, Z ----
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

        # Convert to meters
        target_m = (x_cm / 100.0, y_cm / 100.0, z_cm / 100.0)

        print(f"\nTarget: ({x_cm}, {y_cm}, {z_cm}) cm")

        # ---- SOLVE IK ----
        print("Solving IK...")
        ik_arm, err = robust_ik(robot, target_m, max_attempts=50)

        if ik_arm is None:
            print("\n❌ Point is NOT reachable!")
            print("Try a position within ~30cm reach and Z > 8cm.")
            continue

        print(f"✅ Point is reachable! (error: {err * 100:.3f} cm)")

        # Convert to servo angles
        servo_angles = ik_to_servo_angles(ik_arm)

        print("\nJoint angles:")
        print(f"  {'Joint':<14} {'IK (rad)':>10} {'IK (deg)':>10} {'Servo':>8}")
        print(f"  {'-' * 44}")
        ik_deg = np.degrees(ik_arm)
        for i in range(NUM_ARM_JOINTS):
            print(f"  {JOINT_NAMES[i]:<14} {ik_arm[i]:>+10.3f} {ik_deg[i]:>+10.1f}° {servo_angles[i]:>7}°")

        # ---- MOVE: Arduino + Simulator in sync ----
        print(f"\nMoving robot (safe order: Elbow→Shoulder→Base→WristRoll→WristPitch)...")

        # Send full command to Arduino (it will move joints in its own safe order)
        if arduino.connected:
            all_servo = servo_angles + [150]  # keep gripper open
            cmd = ",".join(str(a) for a in all_servo)
            arduino.ser.write(f"{cmd}\n".encode())

        # Move each joint in safe order, syncing simulator with Arduino progress
        for order_idx in MOVE_ORDER:
            # Wait for Arduino to finish this joint
            if arduino.connected:
                msg = arduino.wait_for_joint_done(timeout=10)
                if msg:
                    print(f"  [Arduino] {msg}")
                else:
                    print(f"  [Arduino] Timeout waiting for joint {JOINT_NAMES[order_idx]}")

            # Animate that joint in simulator
            sim.move_joint_smooth(order_idx, ik_arm[order_idx], steps=50)
            print(f"  [Sim] {JOINT_NAMES[order_idx]} → {ik_deg[order_idx]:+.1f}°")

        # Wait for Arduino ALL_DONE if not already received
        if arduino.connected:
            msg = arduino.wait_for_joint_done(timeout=10)
            if msg:
                print(f"  [Arduino] {msg}")

        # FK verification using the ACTUAL angles we sent
        q_verify = np.zeros(robot.n)
        q_verify[:NUM_ARM_JOINTS] = ik_arm
        robot.q = q_verify
        fk = robot.fkine(q_verify, end=END_LINK)
        fk_pos = fk.t * 100  # meters to cm
        pos_err = np.sqrt(
            (fk_pos[0] - x_cm) ** 2 +
            (fk_pos[1] - y_cm) ** 2 +
            (fk_pos[2] - z_cm) ** 2
        )
        print(f"\n  FK verification: ({fk_pos[0]:.1f}, {fk_pos[1]:.1f}, {fk_pos[2]:.1f}) cm")
        print(f"  Position error: {pos_err:.3f} cm")
        print("  ✅ Movement complete!")

    # Cleanup
    print("\nShutting down...")
    arduino.close()
    print("Goodbye!")


if __name__ == "__main__":
    main()