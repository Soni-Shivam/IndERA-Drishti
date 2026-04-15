"""
Standalone IK Solver — Command Line
====================================
Solves IK for X,Y,Z positions and prints joint + servo angles.
No Arduino or simulator needed.

Usage: python ik_solver.py

Requirements:
  pip install roboticstoolbox-python spatialmath-python
"""

import numpy as np
from arm_common import (
    load_robot, robust_ik, ik_to_servo_angles,
    JOINT_NAMES, NUM_ARM_JOINTS,
)


def main():
    print("=== IK SOLVER ===\n")
    robot = load_robot()
    print(f"Robot: {robot.name}, {robot.n} joints\n")

    while True:
        try:
            inp = input("X,Y,Z (cm) or quit: ").strip()
            if inp.lower() in ('q', 'quit', 'exit'):
                break
            parts = inp.replace(',', ' ').split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            print("Invalid input. Enter three numbers like: 15 0 20\n")
            continue

        target_m = (x / 100.0, y / 100.0, z / 100.0)

        print(f"Solving IK for ({x}, {y}, {z}) cm...")
        ik_arm, err = robust_ik(robot, target_m, max_attempts=50)

        if ik_arm is None:
            print("❌ Not reachable!\n")
            continue

        servo = ik_to_servo_angles(ik_arm)
        ik_deg = np.degrees(ik_arm)

        print(f"✅ Reachable! (error: {err * 100:.3f} cm)")
        print(f"  {'Joint':<14} {'IK (deg)':>10} {'Servo':>8}")
        print(f"  {'-' * 34}")
        for i in range(NUM_ARM_JOINTS):
            print(f"  {JOINT_NAMES[i]:<14} {ik_deg[i]:>+10.1f}° {servo[i]:>7}°")

        # Arduino-ready command string (with gripper open at 150)
        arduino_cmd = ",".join(str(s) for s in servo + [150])
        print(f"\n  Arduino command: {arduino_cmd}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
