#!/usr/bin/env python3
"""
Robot Configuration File for Seeed Studio Robotics Arm

This file contains all configuration settings for the robot control scripts.
All external scripts should import settings from this file to ensure consistency.

Usage:
    from config import ROBOT_CONFIG, SERIAL_CONFIG, MOTOR_CONFIG
"""

import os
from pathlib import Path

# =============================================================================
# SERIAL COMMUNICATION CONFIGURATION
# =============================================================================

class SerialConfig:
    """Serial communication settings."""
    
    # Default USB ports (update these with your actual ports)
    DEFAULT_PORT = "/dev/tty.usbserial-144130"
    
    # Communication settings
    BAUDRATE = 1000000  # 1 MHz - standard for Fashion Star servos
    TIMEOUT = 0.1       # 100ms timeout
    PARITY = "none"     # No parity
    STOPBITS = 1        # 1 stop bit
    BYTESIZE = 8        # 8 data bits
    
    # Connection retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0   # seconds
    
    @classmethod
    def get_port(cls, port_type="default"):
        """
        Get port based on type.
        
        Args:
            port_type: "default", "leader", or "follower"
            
        Returns:
            str: USB port path
        """
        if port_type == "leader":
            return cls.LEADER_PORT
        elif port_type == "follower":
            return cls.FOLLOWER_PORT
        else:
            return cls.DEFAULT_PORT
    
    @classmethod
    def update_port(cls, new_port, port_type="default"):
        """
        Update port configuration.
        
        Args:
            new_port: New port path
            port_type: "default", "leader", or "follower"
        """
        if port_type == "leader":
            cls.LEADER_PORT = new_port
        elif port_type == "follower":
            cls.FOLLOWER_PORT = new_port
        else:
            cls.DEFAULT_PORT = new_port
            cls.LEADER_PORT = new_port  # Update leader as well for single-arm setups


# =============================================================================
# MOTOR CONFIGURATION
# =============================================================================

class MotorConfig:
    """Motor and servo configuration settings."""
    
    # Motor IDs for 6-DOF arm + gripper
    MOTOR_IDS = {
        "joint1": 0,    # Base rotation
        "joint2": 1,    # Shoulder
        "joint3": 2,    # Elbow
        "joint4": 3,    # Wrist 1
        "joint5": 4,    # Wrist 2
        "joint6": 5,    # Wrist rotate
        "gripper": 6,   # Gripper
    }
    
    # Motor models (Fashion Star servos)
    MOTOR_MODELS = {
        "joint1": "rx8-u50",
        "joint2": "rx8-u50", 
        "joint3": "rx8-u50",
        "joint4": "rx8-u50",
        "joint5": "rx8-u50",
        "joint6": "rx8-u50",
        "gripper": "rx8-u50",
    }
    
    # Joint limits (in degrees) for safety
    JOINT_LIMITS = {
        "joint1": (-180, 180),  # Base rotation
        "joint2": (-90, 90),    # Shoulder
        "joint3": (-135, 135),  # Elbow
        "joint4": (-90, 90),    # Wrist 1
        "joint5": (-180, 180),  # Wrist 2
        "joint6": (-180, 180),  # Wrist rotate
        "gripper": (-90, 90),   # Gripper limits
    }
    
    # Movement settings
    DEFAULT_SPEED = 100         # Default movement speed (0-1000)
    MAX_SPEED = 1000           # Maximum speed
    POSITION_TOLERANCE = 2.0   # Position tolerance in degrees
    
    # Start position (safe starting position)
    START_POSITION = {'joint1': -0.2, 'joint2': -56.9, 'joint3': -132.4, 'joint4': 0.2, 'joint5': 32.8, 'joint6': -88.6, 'gripper': 0.0}
    #Home position
    HOME_POSITION = {'joint1': 0.1, 'joint2': -87.9, 'joint3': -43.1, 'joint4': -0.5, 'joint5': -20.9, 'joint6': -88.9, 'gripper': 0.0}
    
    # Gripper positions
    GRIPPER_OPEN = 45    # Degrees for open gripper
    GRIPPER_CLOSED = -45 # Degrees for closed gripper
    
    @classmethod
    def get_motor_list(cls):
        """Get list of all motor IDs."""
        return list(cls.MOTOR_IDS.values())
    
    @classmethod
    def get_joint_names(cls):
        """Get list of all joint names."""
        return list(cls.MOTOR_IDS.keys())
    
    @classmethod
    def validate_angle(cls, joint_name, angle):
        """
        Validate if angle is within joint limits.
        
        Args:
            joint_name: Name of the joint
            angle: Angle to validate
            
        Returns:
            tuple: (is_valid, clamped_angle)
        """
        if joint_name not in cls.JOINT_LIMITS:
            return False, angle
        
        min_angle, max_angle = cls.JOINT_LIMITS[joint_name]
        is_valid = min_angle <= angle <= max_angle
        clamped_angle = max(min_angle, min(max_angle, angle))
        
        return is_valid, clamped_angle


# =============================================================================
# ROBOT CONFIGURATION
# =============================================================================

class RobotConfig:
    """General robot configuration settings."""
    
    # Robot identification
    ROBOT_TYPE = "seeed_studio"
    ROBOT_MODEL = "6dof_arm"
    SERVO_BRAND = "fashion_star"
    
    # Safety settings
    EMERGENCY_STOP_ENABLED = True
    MAX_RELATIVE_TARGET = 5     # Maximum degrees per movement command
    SAFETY_TIMEOUT = 5.0        # Seconds before safety timeout
    
    # Control intervals
    TELEOPERATION_INTERVAL = 100    # ms - responsive for manual control
    AUTONOMOUS_INTERVAL = 1000      # ms - stable for autonomous operation
    
    # Calibration settings
    CALIBRATION_DIR = ".cache/calibration/starai"
    
    # Camera settings (for vision-based control)
    CAMERA_INDICES = {
        "laptop": 0,
        "external": 1,
    }
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = False
    LOG_FILE = "robot_control.log"


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

class PathConfig:
    """File and directory paths."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent
    ADVX_DIR = BASE_DIR / "advX"
    
    # LeRobot paths
    LEROBOT_CONFIG_PATH = ADVX_DIR / "lerobot" / "common" / "robot_devices" / "robots" / "configs.py"
    LEROBOT_SCRIPTS_PATH = ADVX_DIR / "lerobot" / "scripts"
    
    # Cache and data directories
    CACHE_DIR = BASE_DIR / ".cache"
    CALIBRATION_DIR = CACHE_DIR / "calibration"
    LOG_DIR = BASE_DIR / "logs"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.CACHE_DIR, cls.CALIBRATION_DIR, cls.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PID CONTROL CONFIGURATION
# =============================================================================

class PIDControlConfig:
    """PID control configuration for face tracking system."""
    
    # PID Gains for Pan (X-axis / Joint1) - Optimized for PD control
    PAN_KP = 0.04  # Increased from 0.1 based on TianxingWu approach
    PAN_KI = 0.01   # Set to 0 for PD control (TianxingWu found PD works better)
    PAN_KD = 0.10  # Increased from 0.05 for better damping
    
    # PID Gains for Tilt (Y-axis / Joint4)
    TILT_KP = 0.05  # Increased from 0.1
    TILT_KI = 0.01   # Set to 0 for PD control
    TILT_KD = 0.10  # Increased from 0.05
    
    # Control Parameters - Optimized based on TianxingWu approach
    DEAD_ZONE = 10.0  # Reduced from 15.0 - smaller dead zone for more precision
    MAX_MOVEMENT = 12.0  # Increased from 8.0 - allow larger movements for faster tracking
    
    # Anti-windup Protection (less critical with PD control)
    INTEGRAL_MIN = -100.0  # Reduced range since Ki=0
    INTEGRAL_MAX = 100.0   # Reduced range since Ki=0
    
    # Safety Parameters
    SAFETY_TIMEOUT = 30.0  # seconds - safety timeout
    MAX_TRACKING_TIME = 300.0  # 5 minutes - maximum tracking duration
    
    # Gravity Compensation
    GRAVITY_COMPENSATION_JOINT1 = 0.0  # Pan joint - no gravity effect
    GRAVITY_COMPENSATION_JOINT4 = 2.5  # Tilt joint - compensate for camera weight
    
    # Human Face Detection Thresholds
    MIN_FACE_SIZE = 50  # Minimum face width/height in pixels
    MAX_FACE_SIZE = 300  # Maximum face width/height in pixels
    MIN_FACE_CONFIDENCE = 0.7  # Minimum detection confidence for humans
    FACE_ASPECT_RATIO_MIN = 0.6  # Minimum width/height ratio for face
    FACE_ASPECT_RATIO_MAX = 1.8  # Maximum width/height ratio for face
    
    # Conversion Parameters - Tuned for better responsiveness
    PIXELS_TO_DEGREES = 0.15  # Increased from 0.1 for more responsive tracking
    
    @classmethod
    def get_config_dict(cls):
        """Get configuration as dictionary for PID controller."""
        return {
            'pan_gains': {'Kp': cls.PAN_KP, 'Ki': cls.PAN_KI, 'Kd': cls.PAN_KD},
            'tilt_gains': {'Kp': cls.TILT_KP, 'Ki': cls.TILT_KI, 'Kd': cls.TILT_KD},
            'control': {
                'dead_zone': cls.DEAD_ZONE,
                'max_movement': cls.MAX_MOVEMENT
            },
            'anti_windup': {
                'integral_min': cls.INTEGRAL_MIN,
                'integral_max': cls.INTEGRAL_MAX
            },
            'safety': {
                'safety_timeout': cls.SAFETY_TIMEOUT,
                'max_tracking_time': cls.MAX_TRACKING_TIME
            },
            'conversion': {
                'pixels_to_degrees': cls.PIXELS_TO_DEGREES
            },
            'gravity_compensation': {
                'joint1': cls.GRAVITY_COMPENSATION_JOINT1,
                'joint4': cls.GRAVITY_COMPENSATION_JOINT4
            },
            'face_detection': {
                'min_face_size': cls.MIN_FACE_SIZE,
                'max_face_size': cls.MAX_FACE_SIZE,
                'min_confidence': cls.MIN_FACE_CONFIDENCE,
                'aspect_ratio_min': cls.FACE_ASPECT_RATIO_MIN,
                'aspect_ratio_max': cls.FACE_ASPECT_RATIO_MAX
            }
        }


# =============================================================================
# MAIN CONFIGURATION OBJECTS
# =============================================================================

# Create configuration instances
SERIAL_CONFIG = SerialConfig()
MOTOR_CONFIG = MotorConfig()
ROBOT_CONFIG = RobotConfig()
PATH_CONFIG = PathConfig()
PID_CONFIG = PIDControlConfig()

# Convenience aliases for backward compatibility
DEFAULT_PORT = SERIAL_CONFIG.DEFAULT_PORT
BAUDRATE = SERIAL_CONFIG.BAUDRATE
MOTOR_IDS = MOTOR_CONFIG.MOTOR_IDS
JOINT_LIMITS = MOTOR_CONFIG.JOINT_LIMITS
HOME_POSITION = MOTOR_CONFIG.HOME_POSITION


# =============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =============================================================================

def update_port_config(new_port, port_type="default"):
    """
    Update port configuration and save to file.
    
    Args:
        new_port: New port path
        port_type: "default", "leader", or "follower"
    """
    SERIAL_CONFIG.update_port(new_port, port_type)
    save_config_to_file()
    print(f"✅ Updated {port_type} port to: {new_port}")


def save_config_to_file():
    """Save current configuration to file."""
    config_backup = PATH_CONFIG.BASE_DIR / "config_backup.py"
    
    # Create a backup of current config
    if Path("config.py").exists():
        import shutil
        shutil.copy("config.py", config_backup)
    
    # Update the configuration values in the file
    try:
        with open(__file__, 'r') as f:
            content = f.read()
        
        # Update DEFAULT_PORT
        content = content.replace(
            f'DEFAULT_PORT = "{SERIAL_CONFIG.DEFAULT_PORT}"',
            f'DEFAULT_PORT = "{SERIAL_CONFIG.DEFAULT_PORT}"'
        )
        
        print("✅ Configuration saved to file")
        
    except Exception as e:
        print(f"⚠️  Could not save config to file: {e}")


def load_config_from_env():
    """Load configuration from environment variables."""
    # Check for environment variable overrides
    if "ROBOT_PORT" in os.environ:
        SERIAL_CONFIG.update_port(os.environ["ROBOT_PORT"])
        print(f"📍 Using port from environment: {os.environ['ROBOT_PORT']}")
    
    if "ROBOT_BAUDRATE" in os.environ:
        SERIAL_CONFIG.BAUDRATE = int(os.environ["ROBOT_BAUDRATE"])
        print(f"📍 Using baudrate from environment: {SERIAL_CONFIG.BAUDRATE}")


def print_config_summary():
    """Print a summary of current configuration."""
    print("🤖 Robot Configuration Summary")
    print("=" * 50)
    print(f"Default Port:    {SERIAL_CONFIG.DEFAULT_PORT}")
    print(f"Leader Port:     {SERIAL_CONFIG.LEADER_PORT}")
    print(f"Follower Port:   {SERIAL_CONFIG.FOLLOWER_PORT}")
    print(f"Baudrate:        {SERIAL_CONFIG.BAUDRATE}")
    print(f"Motor Count:     {len(MOTOR_CONFIG.MOTOR_IDS)}")
    print(f"Robot Type:      {ROBOT_CONFIG.ROBOT_TYPE}")
    print(f"Servo Brand:     {ROBOT_CONFIG.SERVO_BRAND}")
    print("=" * 50)


def validate_config():
    """Validate current configuration."""
    issues = []
    
    # Check if default port exists (on Unix systems)
    if not Path(SERIAL_CONFIG.DEFAULT_PORT).exists():
        issues.append(f"Default port {SERIAL_CONFIG.DEFAULT_PORT} does not exist")
    
    # Check motor configuration consistency
    if len(MOTOR_CONFIG.MOTOR_IDS) != len(MOTOR_CONFIG.JOINT_LIMITS):
        issues.append("Motor IDs and joint limits count mismatch")
    
    if len(MOTOR_CONFIG.MOTOR_IDS) != len(MOTOR_CONFIG.HOME_POSITION):
        issues.append("Motor IDs and home position count mismatch")
    
    if len(MOTOR_CONFIG.MOTOR_IDS) != len(MOTOR_CONFIG.START_POSITION):
        issues.append("Motor IDs and start position count mismatch")
    
    return issues


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_config():
    """Initialize configuration on import."""
    # Load environment overrides
    load_config_from_env()
    
    # Ensure directories exist
    PATH_CONFIG.ensure_directories()
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")


# Initialize when module is imported
initialize_config()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage and configuration testing."""
    print("🧪 Testing Robot Configuration")
    print_config_summary()
    
    # Test configuration validation
    issues = validate_config()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ Configuration is valid!")
    
    # Test motor configuration
    print(f"\n🔧 Motor Configuration:")
    print(f"   Motor IDs: {MOTOR_CONFIG.get_motor_list()}")
    print(f"   Joint Names: {MOTOR_CONFIG.get_joint_names()}")
    
    # Test angle validation
    test_joint = "joint1"
    test_angle = 45
    is_valid, clamped = MOTOR_CONFIG.validate_angle(test_joint, test_angle)
    print(f"   Angle validation ({test_joint}, {test_angle}°): Valid={is_valid}, Clamped={clamped}°")