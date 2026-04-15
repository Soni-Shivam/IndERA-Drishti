"""
arm_common.py — Shared module for robot arm IK pipeline
========================================================
Single source of truth for:
  - Robot loading
  - IK solving (robust multi-attempt)
  - Servo angle conversion
  - Constants and configuration

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


# =========================================================================
# ROBOT LOADING
# =========================================================================

def load_robot(urdf_filename="custom_arm.urdf"):
    """
    Load the robot URDF.
    Looks for the URDF in the same directory as the calling script,
    falling back to the directory of this module.
    """
    # Try caller's directory first
    import inspect
    caller_frame = inspect.stack()[1]
    caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
    urdf_path = os.path.join(caller_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        # Fall back to this module's directory
        module_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(module_dir, urdf_filename)

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found at {urdf_path}")
        sys.exit(1)

    robot = rtb.ERobot.URDF(urdf_path)
    robot.q = np.zeros(robot.n)

    # Print joint order for verification
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

    Args:
        robot: ERobot instance
        target_xyz_m: tuple/list of (x, y, z) in meters
        max_attempts: number of random restarts

    Returns:
        (ik_arm_angles, position_error_m) on success
            ik_arm_angles: np.array of 5 joint angles in radians,
                           with wrist_roll forced to 0 and elbow forced positive
        (None, None) on failure
    """
    x_m, y_m, z_m = target_xyz_m

    # Position-only target — no orientation constraint.
    # Just use a pure translation, no SE3.OA nonsense.
    Tep = SE3.Trans(x_m, y_m, z_m)

    best_arm = None
    best_err = float('inf')
    best_cost = float('inf')

    for attempt in range(max_attempts):
        # Generate initial guess
        if attempt == 0:
            q0 = np.zeros(robot.n)
        elif attempt < 10:
            # Small perturbations near zero (prefer natural poses)
            q0 = np.random.uniform(-0.5, 0.5, robot.n)
            q0[3] = 0.0    # Keep wrist roll near zero
            # Fix finger joints to zero
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

        # sol is (q, success, ...) — sol[1] is bool-like
        if not sol[1]:
            continue

        q_solution = np.array(sol[0])

        # FK verification
        fk = robot.fkine(q_solution, end=END_LINK)
        err = np.linalg.norm(Tep.t - fk.t)

        if err >= 0.01:  # > 1cm error — reject
            continue

        # Extract arm joints and apply post-processing
        # IMPORTANT: apply abs(elbow) and zero wrist_roll BEFORE checking servo range
        arm = q_solution[:NUM_ARM_JOINTS].copy()
        arm[3] = 0.0           # Force wrist roll to zero
        arm[2] = abs(arm[2])   # Force elbow positive

        # Check servo angles are in 0-180 range AFTER post-processing
        servo = ik_to_servo_angles(arm)
        if not all(0 <= s <= 180 for s in servo):
            continue

        # Verify FK with the POST-PROCESSED angles
        q_verify = q_solution.copy()
        q_verify[:NUM_ARM_JOINTS] = arm
        fk_verify = robot.fkine(q_verify, end=END_LINK)
        err_verify = np.linalg.norm(Tep.t - fk_verify.t)

        if err_verify >= 0.02:  # Allow slightly more tolerance after post-processing
            continue

        # Cost: prefer minimal movement
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

    IK convention: 0 rad = home position (straight up)
    Servo convention: 90° = home position (straight up)

    Mapping:
      Base:        servo = 90 - degrees(ik)
      Shoulder:    servo = 90 - degrees(ik)
      Elbow:       servo = 90 + degrees(ik)   (ik already forced positive via abs())
      Wrist Roll:  servo = 90 + degrees(ik)   (ik forced to 0, so always 90°)
      Wrist Pitch: servo = 90 - degrees(ik)

    Args:
        ik_arm: array-like of 5 joint angles in radians

    Returns:
        list of 5 servo angles (int), clamped to 0-180
    """
    ik_deg = np.degrees(ik_arm)

    servo_angles = [
        90 - ik_deg[0],   # Base
        90 - ik_deg[1],   # Shoulder
        90 + ik_deg[2],   # Elbow
        90 + ik_deg[3],   # Wrist Roll
        90 - ik_deg[4],   # Wrist Pitch
    ]

    return [int(max(0, min(180, a))) for a in servo_angles]


def servo_to_ik_angles(servo_angles):
    """
    Convert 5 servo angles (degrees, 0-180) back to IK radians.

    Inverse of ik_to_servo_angles.

    Args:
        servo_angles: list of 5 servo angles in degrees

    Returns:
        np.array of 5 IK angles in radians
    """
    ik_deg = [
        90 - servo_angles[0],   # Base
        90 - servo_angles[1],   # Shoulder
        servo_angles[2] - 90,   # Elbow
        servo_angles[3] - 90,   # Wrist Roll
        90 - servo_angles[4],   # Wrist Pitch
    ]
    return np.radians(ik_deg)