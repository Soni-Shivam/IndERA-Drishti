"""
Standalone IK Visualizer — No Arduino needed
==============================================
Solves IK and animates the robot in the Swift 3D simulator.

Usage: python ik_visualize.py

Requirements:
  pip install roboticstoolbox-python spatialmath-python swift-sim
"""

import numpy as np

try:
    import swift
except ImportError:
    print("ERROR: swift-sim not found. Install with: pip install swift-sim")
    import sys
    sys.exit(1)

from arm_common import (
    load_robot, robust_ik, ik_to_servo_angles, user_to_robot_coords,
    JOINT_NAMES, MOVE_ORDER, NUM_ARM_JOINTS, END_LINK,
)


def main():
    print("=== IK VISUALIZER ===\n")
    robot = load_robot()
    print(f"Robot: {robot.name}, {robot.n} joints")

    print("Launching simulator...")
    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)
    env.step(0.05)

    # Track current joint angles
    current_q = np.zeros(robot.n)

    print("  X=forward  Y=left/right  Z=up\n")

    while True:
        try:
            inp = input("X,Y,Z (cm) or home/quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if inp.lower() in ('q', 'quit', 'exit'):
            break

        # ---- HOME ----
        if inp.lower() == 'home':
            for idx in MOVE_ORDER:
                start_val = current_q[idx]
                for i in range(1, 51):
                    current_q[idx] = start_val * (1 - i / 50)
                    robot.q = current_q.copy()
                    env.step(0.02)
                print(f"  [Sim] {JOINT_NAMES[idx]} → home")
            print("Home.\n")
            continue

        # ---- PARSE COORDINATES ----
        try:
            parts = inp.replace(',', ' ').split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            print("Invalid. Enter three numbers like: 15 0 20\n")
            continue

        x_r, y_r, z_r = user_to_robot_coords(x, y, z)
        target_m = (x_r / 100.0, y_r / 100.0, z_r / 100.0)

        print(f"Solving IK for ({x}, {y}, {z}) cm → robot: ({x_r:.1f}, {y_r:.1f}, {z_r:.1f}) cm...")
        ik_arm, err = robust_ik(robot, target_m, max_attempts=50)

        if ik_arm is None:
            print("❌ Not reachable!\n")
            continue

        servo = ik_to_servo_angles(ik_arm)
        ik_deg = np.degrees(ik_arm)

        print(f"✅ Reachable! (error: {err * 100:.3f} cm)")
        for i in range(NUM_ARM_JOINTS):
            print(f"  {JOINT_NAMES[i]:<14} {ik_deg[i]:>+8.1f}°  servo: {servo[i]:>4}°")

        # ---- ANIMATE ----
        # Move joints one by one in safe order (same as Arduino)
        for idx in MOVE_ORDER:
            target_rad = ik_arm[idx]
            start_val = current_q[idx]
            for i in range(1, 51):
                frac = i / 50
                current_q[idx] = start_val + (target_rad - start_val) * frac
                robot.q = current_q.copy()
                env.step(0.02)
            print(f"  [Sim] {JOINT_NAMES[idx]} done")

        # FK verification
        q_verify = current_q.copy()
        fk = robot.fkine(q_verify, end=END_LINK)
        fk_pos = fk.t * 100
        pos_err = np.sqrt(
            (fk_pos[0] - x) ** 2 +
            (fk_pos[1] - y) ** 2 +
            (fk_pos[2] - z) ** 2
        )
        print(f"  FK check: ({fk_pos[0]:.1f}, {fk_pos[1]:.1f}, {fk_pos[2]:.1f}) cm, err: {pos_err:.3f} cm")
        print("✅ Done!\n")

    print("Bye!")


if __name__ == "__main__":
    main()