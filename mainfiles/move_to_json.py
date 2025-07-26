#!/usr/bin/env python3
"""
Script to move robot to positions specified in a JSON file

This script reads joint positions from a JSON file and moves the robot
to those positions. The JSON should contain joint names and angles.

Example JSON format:
{
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": -90.0,
    "joint4": -45.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "gripper": 0.0
}
"""

import json
import time
import sys
import argparse

# Import robot configuration
try:
    from mainfiles.config import HOME_POSITION
    from simple_robot_control import SimpleRobotController
    print("✅ Configuration loaded successfully")
except ImportError as e:
    print(f"❌ Could not import configuration: {e}")
    print("Make sure config.py and simple_robot_control.py are in the same directory.")
    sys.exit(1)

def load_positions_from_json(json_file):
    """Load joint positions from JSON file"""
    try:
        with open(json_file, 'r') as f:
            positions = json.load(f)
        print(f"✅ Loaded positions from {json_file}")
        return positions
    except FileNotFoundError:
        print(f"❌ File not found: {json_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def validate_positions(positions, controller):
    """Validate that all positions are within joint limits"""
    if not positions:
        return False
    
    valid_positions = {}
    for joint, angle in positions.items():
        if joint not in controller.motor_ids:
            print(f"⚠️  Unknown joint: {joint}")
            continue
        
        # Check limits
        min_angle, max_angle = controller.joint_limits[joint]
        if angle < min_angle or angle > max_angle:
            print(f"❌ Angle {angle}° out of range for {joint} ({min_angle}° to {max_angle}°)")
            return False
        
        valid_positions[joint] = angle
    
    return valid_positions

def move_to_json_positions(json_file_path, speed=50, velocity=None, verbose=True):
    """
    Move robot to positions specified in a JSON file
    
    Args:
        json_file_path (str): Path to JSON file containing joint positions
        speed (int): Movement speed (0-1000, default: 50)
        velocity (float or None): Movement velocity in dps (default: None)
        verbose (bool): Whether to print status messages (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print("🤖 Moving robot to JSON-specified positions")
        print("=" * 50)
    
    # Create robot controller
    controller = SimpleRobotController()
    
    try:
        # Connect to robot
        if verbose:
            print("🔌 Connecting to robot...")
        if not controller.connect():
            if verbose:
                print("❌ Failed to connect to robot.")
            return False
        
        if verbose:
            print("✅ Robot connected successfully!")
        
        # Load positions from JSON file
        target_positions = load_positions_from_json(json_file_path)
        if not target_positions:
            if verbose:
                print("❌ Failed to load positions.")
            return False
        
        # Validate positions
        valid_positions = validate_positions(target_positions, controller)
        if not valid_positions:
            if verbose:
                print("❌ Invalid positions found.")
            return False
        
        target_positions = valid_positions
        if verbose:
            print(f"🎯 Target positions: {target_positions}")
        
        # Show current positions
        if verbose:
            print("\n📍 Current positions:")
            controller.read_positions()
        
        # Move to target positions
        if verbose:
            print(f"\n🤖 Moving to target positions (speed: {speed}, velocity: {velocity})...")
        success = controller.move_multiple_joints(target_positions, speed=speed, velocity=velocity)
        
        if success:
            if verbose:
                print("✅ Successfully moved to target positions!")
                
                # Wait a moment and show final positions
                time.sleep(1)
                print("\n📍 Final positions:")
                controller.read_positions()
        else:
            if verbose:
                print("❌ Failed to move to target positions")
        
        return success
        
    except KeyboardInterrupt:
        if verbose:
            print("\n🛑 Interrupted by user")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Disconnect from robot
        controller.disconnect()
        if verbose:
            print("👋 Script completed!")

def move_to_home_position(speed=50, velocity=None, verbose=True):
    """
    Move robot to home position
    
    Args:
        speed (int): Movement speed (0-1000, default: 50)
        velocity (float or None): Movement velocity in dps (default: None)
        verbose (bool): Whether to print status messages (default: True)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if verbose:
        print("🏠 Moving robot to home position")
        print("=" * 50)
    
    # Create robot controller
    controller = SimpleRobotController()
    
    try:
        # Connect to robot
        if verbose:
            print("🔌 Connecting to robot...")
        if not controller.connect():
            if verbose:
                print("❌ Failed to connect to robot.")
            return False
        
        if verbose:
            print("✅ Robot connected successfully!")
            print(f"🏠 Home position: {HOME_POSITION}")
        
        # Show current positions
        if verbose:
            print("\n📍 Current positions:")
            controller.read_positions()
        
        # Move to home position
        if verbose:
            print(f"\n🤖 Moving to home position (speed: {speed}, velocity: {velocity})...")
        success = controller.move_multiple_joints(HOME_POSITION, speed=speed, velocity=velocity)
        
        if success:
            if verbose:
                print("✅ Successfully moved to home position!")
                
                # Wait a moment and show final positions
                time.sleep(1)
                print("\n📍 Final positions:")
                controller.read_positions()
        else:
            if verbose:
                print("❌ Failed to move to home position")
        
        return success
        
    except KeyboardInterrupt:
        if verbose:
            print("\n🛑 Interrupted by user")
        return False
    except Exception as e:
        if verbose:
            print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Disconnect from robot
        controller.disconnect()
        if verbose:
            print("👋 Script completed!")

def main():
    """Main function to move robot to JSON-specified positions"""
    parser = argparse.ArgumentParser(description='Move robot to positions from JSON file')
    parser.add_argument('json_file', nargs='?', help='JSON file containing joint positions')
    parser.add_argument('--speed', type=int, default=50, help='Movement speed (0-1000, default: 50)')
    parser.add_argument('--velocity', type=float, default=None, help='Movement velocity in dps (default: None)')
    parser.add_argument('--home', action='store_true', help='Move to home position instead')
    parser.add_argument('--show-home', action='store_true', help='Show home position and exit')
    
    args = parser.parse_args()
    
    print("🤖 Robot Position Control Script")
    print("=" * 50)
    
    # Show home position if requested
    if args.show_home:
        print(f"\n🏠 Home position from config: {HOME_POSITION}")
        return
    
    # Determine target positions
    if args.home:
        # Use max velocity for home movement
        success = move_to_home_position(speed=args.speed, velocity=70)
    else:
        success = move_to_json_positions(args.json_file, speed=args.speed, velocity=args.velocity)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 