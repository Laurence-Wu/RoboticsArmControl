"""
Utils Package for Robot Control

This package contains utility modules for controlling the Seeed Studio Robotics Arm:
- config.py: Configuration settings and constants
- simple_robotic_test.py: Simple robot control and testing functions
- disable_all_motors.py: Emergency stop and motor shutdown utilities
"""

# Import main configuration objects for easy access
try:
    from .config import (
        SERIAL_CONFIG,
        MOTOR_CONFIG,
        ROBOT_CONFIG,
        PATH_CONFIG,
        DEFAULT_PORT,
        BAUDRATE,
        MOTOR_IDS,
        JOINT_LIMITS,
        HOME_POSITION
    )
except ImportError:
    # Fallback for when running scripts directly
    pass

__version__ = "1.0.0"
__author__ = "Robot Control Team"