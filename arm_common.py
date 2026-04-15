"""
arm_common.py — Shared module for robot arm IK pipeline
========================================================
Single source of truth for:
  - Robot loading
  - IK solving (robust multi-attempt)
  - Servo angle conversion
  - Constants and configuration (arm + conveyor)

All other files import from here. No more copy-paste divergence.
"""

import os
import sys
import numpy as np

try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
except ImportError:
    print("ERROR: robotics libraries not found.")
    print("Install with: pip install roboticstoolbox-python spatialmath-python")
    sys.exit(1)


# =========================================================================
# CONFIGURATION
# =========================================================================

SERIAL_PORT = "COM6"
SERIAL_BAUD = 115200
SERVO_SPEED = 15        # ms per degree step on Arduino

# Servo home positions (all at 90° = robot straight up)
HOME_ANGLES = [90, 90, 90, 90, 90, 150]  # b, s, e, w, p, g (gripper open=150)

# Joint names matching Arduino order: b, s, e, w, p, g
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist Roll", "Wrist Pitch", "Gripper"]
JOINT_IDS = ["b", "s", "e", "w", "p", "g"]

# Safe movement order (indices into the 5 arm joints)
# Elbow(2) → Shoulder(1) → Base(0) → Wrist Roll(3) → Wrist Pitch(4)
MOVE_ORDER = [2, 1, 0, 3, 4]

# Number of arm joints (excluding gripper fingers)
NUM_ARM_JOINTS = 5

# IK end effector link name
END_LINK = "gripper_base_link"

# Per-joint servo limits [min, max] — SHOULDER capped at 140 to prevent inversion
SERVO_LIMITS = [
    (0, 180),    # Base
    (0, 140),    # Shoulder — robot inverts beyond ~140°
    (0, 180),    # Elbow
    (0, 180),    # Wrist Roll
    (0, 180),    # Wrist Pitch
]

# =========================================================================
# CONVEYOR BELT CONFIGURATION
# =========================================================================
# All conveyor/sensor settings are here so you only edit this file.

CONVEYOR_SPEED       = 200    # PWM 0-255 (motor speed)
SENSOR_THRESHOLD_CM  = 5.0    # Object detected if distance <= this (cm)
SENSOR_STOP_DELAY_MS = 0      # ms to wait AFTER detection before stopping belt
                               # (lets object travel a bit further onto the belt)

# ---- PICK & PLACE POSITIONS (cm, in YOUR table coordinates) ----
CONVEYOR_PICK_POINT  = (17, 0, 16)      # Where the arm grabs the object
CONVEYOR_PLACE_POINT = (-6, -20, 18)    # Where the arm puts the object

# =========================================================================
# COORDINATE ALIGNMENT
# =========================================================================
TABLE_ROTATION_DEG = 17.0


def user_to_robot_coords(x_user, y_user, z_user):
    """
    Rotate user's table coordinates into the robot's coordinate system.
    Only rotates X and Y around Z-axis. Z (height) stays the same.
    """
    angle_rad = np.radians(TABLE_ROTATION_DEG)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    x_robot = x_user * cos_a - y_user * sin_a
    y_robot = x_user * sin_a + y_user * cos_a
    z_robot = z_user

    return x_robot, y_robot, z_robot


# =========================================================================
# ROBOT LOADING
# =========================================================================

def load_robot(urdf_filename="custom_arm.urdf"):
    """
    Load the robot URDF.
    Looks for the URDF in the same directory as the calling script,
    falling back to the directory of this module.
    """
    import inspect
    caller_frame = inspect.stack()[1]
    caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
    urdf_path = os.path.join(caller_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(module_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        sys.exit(1)

    robot = rtb.ERobot.URDF(urdf_path)
    robot.q = np.zeros(robot.n)

    print(f"  URDF joints ({robot.n} total):")
    for i, link in enumerate(robot):
        if hasattr(link, 'jtype') and link.jtype in ('R', 'P'):
            print(f"    [{link.jindex}] {link.name} (axis: {link.jtype})")

    return robot


# =========================================================================
# IK SOLVER
# =========================================================================

def robust_ik(robot, target_xyz_m, max_attempts=50):
    """
    Solve IK for a target XYZ position (in meters).

    Uses position-only mask [1,1,1,0,0,0].
    Tries multiple random initial guesses.
    Validates servo range and prefers minimal-movement solutions.

    Returns:
        (ik_arm_angles, position_error_m) on success
        (None, None) on failure
    """
    x_m, y_m, z_m = target_xyz_m
    Tep = SE3.Trans(x_m, y_m, z_m)

    best_arm = None
    best_err = float('inf')
    best_cost = float('inf')

    for attempt in range(max_attempts):
        if attempt == 0:
            q0 = np.zeros(robot.n)
        elif attempt < 10:
            q0 = np.random.uniform(-0.5, 0.5, robot.n)
            q0[3] = 0.0
            for fi in range(NUM_ARM_JOINTS, robot.n):
                q0[fi] = 0.0
        else:
            q0 = np.random.uniform(-1.5, 1.5, robot.n)
            q0[3] = np.random.uniform(-0.3, 0.3)
            for fi in range(NUM_ARM_JOINTS, robot.n):
                q0[fi] = 0.0

        sol = robot.ik_LM(
            Tep,
            end=END_LINK,
            mask=[1, 1, 1, 0, 0, 0],
            q0=q0,
            ilimit=500,
            slimit=100,
            tol=1e-6
        )

        if not sol[1]:
            continue

        q_solution = np.array(sol[0])

        fk = robot.fkine(q_solution, end=END_LINK)
        err = np.linalg.norm(Tep.t - fk.t)

        if err >= 0.01:
            continue

        arm = q_solution[:NUM_ARM_JOINTS].copy()
        arm[3] = 0.0
        arm[2] = abs(arm[2])

        servo = ik_to_servo_angles(arm)
        in_limits = all(
            SERVO_LIMITS[i][0] <= servo[i] <= SERVO_LIMITS[i][1]
            for i in range(NUM_ARM_JOINTS)
        )
        if not in_limits:
            continue

        q_verify = q_solution.copy()
        q_verify[:NUM_ARM_JOINTS] = arm
        fk_verify = robot.fkine(q_verify, end=END_LINK)
        err_verify = np.linalg.norm(Tep.t - fk_verify.t)

        if err_verify >= 0.02:
            continue

        roll_penalty = abs(arm[3]) * 0.01
        joint_cost = np.sum(np.abs(arm)) * 0.0001
        total_cost = err_verify + joint_cost + roll_penalty

        if total_cost < best_cost:
            best_cost = total_cost
            best_err = err_verify
            best_arm = arm.copy()

    if best_arm is not None:
        return best_arm, best_err

    return None, None


# =========================================================================
# ANGLE CONVERSION: IK radians → Servo degrees
# =========================================================================

def ik_to_servo_angles(ik_arm):
    """
    Convert 5 IK joint angles (radians) to servo angles (0-180°).
    """
    ik_deg = np.degrees(ik_arm)

    servo_angles = [
        90 - ik_deg[0],   # Base
        90 - ik_deg[1],   # Shoulder
        90 + ik_deg[2],   # Elbow
        90 + ik_deg[3],   # Wrist Roll
        90 - ik_deg[4],   # Wrist Pitch
    ]

    return [int(max(SERVO_LIMITS[i][0], min(SERVO_LIMITS[i][1], a))) for i, a in enumerate(servo_angles)]


def servo_to_ik_angles(servo_angles):
    """
    Convert 5 servo angles (degrees, 0-180) back to IK radians.
    """
    ik_deg = [
        90 - servo_angles[0],
        90 - servo_angles[1],
        servo_angles[2] - 90,
        servo_angles[3] - 90,
        90 - servo_angles[4],
    ]
    return np.radians(ik_deg)