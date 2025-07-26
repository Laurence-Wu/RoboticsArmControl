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
    DEFAULT_PORT = "/dev/tty.usbserial-144140"
    LEADER_PORT = "/dev/tty.usbmodem575E0031751"    # For teleoperation leader arm
    FOLLOWER_PORT = "/dev/tty.usbmodem575E0032081"  # For teleoperation follower arm
    
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
    
    # Home position (safe starting position)
    HOME_POSITION = {'joint1': -0.2, 'joint2': -56.9, 'joint3': -132.4, 'joint4': 0.2, 'joint5': 32.8, 'joint6': -88.6, 'gripper': 0.0}
    
    
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
# MAIN CONFIGURATION OBJECTS
# =============================================================================

# Create configuration instances
SERIAL_CONFIG = SerialConfig()
MOTOR_CONFIG = MotorConfig()
ROBOT_CONFIG = RobotConfig()
PATH_CONFIG = PathConfig()

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
        
        # Update LEADER_PORT
        content = content.replace(
            f'LEADER_PORT = "{SERIAL_CONFIG.LEADER_PORT}"',
            f'LEADER_PORT = "{SERIAL_CONFIG.LEADER_PORT}"'
        )
        
        # Update FOLLOWER_PORT
        content = content.replace(
            f'FOLLOWER_PORT = "{SERIAL_CONFIG.FOLLOWER_PORT}"',
            f'FOLLOWER_PORT = "{SERIAL_CONFIG.FOLLOWER_PORT}"'
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
    
    # Check if ports exist (on Unix systems)
    for port_name, port_path in [
        ("Default", SERIAL_CONFIG.DEFAULT_PORT),
        ("Leader", SERIAL_CONFIG.LEADER_PORT),
        ("Follower", SERIAL_CONFIG.FOLLOWER_PORT)
    ]:
        if not Path(port_path).exists():
            issues.append(f"{port_name} port {port_path} does not exist")
    
    # Check motor configuration consistency
    if len(MOTOR_CONFIG.MOTOR_IDS) != len(MOTOR_CONFIG.JOINT_LIMITS):
        issues.append("Motor IDs and joint limits count mismatch")
    
    if len(MOTOR_CONFIG.MOTOR_IDS) != len(MOTOR_CONFIG.HOME_POSITION):
        issues.append("Motor IDs and home position count mismatch")
    
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