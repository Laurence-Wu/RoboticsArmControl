#!/usr/bin/env python3
"""
Standalone script to check and adjust joint2 and joint3 to their target positions

This script can be run independently to ensure both joint2 and joint3 are at the correct
positions for face tracking operations.

Usage:
    python3 adjust_joint3.py
    python3 adjust_joint3.py --joint2 42.4 --joint3 -132.0 --tolerance 2.0 --speed 100
    python3 adjust_joint3.py --joint2-only 42.4
    python3 adjust_joint3.py --joint3-only -132.0
"""

import argparse
import sys
import time

# Import robot configuration
try:
    from mainfiles.config import SERIAL_CONFIG, MOTOR_CONFIG
    from simple_robot_control import SimpleRobotController
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Make sure config.py and simple_robot_control.py are available")
    ROBOT_AVAILABLE = False

def check_and_adjust_arm_joints(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """
    Check and adjust both joint2 and joint3 to their target positions.
    This function ensures both joints are properly positioned together.
    
    Args:
        robot_controller: SimpleRobotController instance
        joint2_target: Target angle for joint2 (default: -42.4)
        joint3_target: Target angle for joint3 (default: -132.0)
        tolerance: Acceptable deviation from target (default: 2.0 degrees)
        speed: Movement speed (default: 100)
        
    Returns:
        bool: True if both joints are at target angles or were successfully adjusted
    """
    if not robot_controller or not robot_controller.is_connected:
        print("⚠️  Robot controller not available for arm joint adjustment")
        return False
    
    try:
        print("🔧 Checking and adjusting arm joints (joint2 & joint3)...")
        
        # Read current positions for both joints
        current_joint2 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint2"])
        current_joint3 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint3"])
        
        if current_joint2 is None:
            print("⚠️  Could not read joint2 position - servo may not be responding")
            return False
        
        if current_joint3 is None:
            print("⚠️  Could not read joint3 position - servo may not be responding")
            return False
        
        # Check if both joints are within tolerance
        joint2_diff = abs(current_joint2 - joint2_target)
        joint3_diff = abs(current_joint3 - joint3_target)
        
        joint2_ok = joint2_diff <= tolerance
        joint3_ok = joint3_diff <= tolerance
        
        # Report current status
        print(f"📊 Current positions: Joint2={current_joint2:.1f}°, Joint3={current_joint3:.1f}°")
        print(f"📊 Target positions: Joint2={joint2_target}°, Joint3={joint3_target}°")
        
        if joint2_ok and joint3_ok:
            print(f"✅ Both joints are at correct positions (tolerance: ±{tolerance}°)")
            return True
        
        # Adjust joints that need adjustment
        adjustments_made = False
        
        if not joint2_ok:
            print(f"🔄 Adjusting joint2 from {current_joint2:.1f}° to {joint2_target}° (difference: {joint2_diff:.1f}°)")
            success = robot_controller.move_joint("joint2", joint2_target, speed=speed)
            if success:
                print(f"✅ Joint2 successfully adjusted to {joint2_target}°")
                adjustments_made = True
            else:
                print(f"❌ Failed to adjust joint2 to {joint2_target}°")
                return False
        
        if not joint3_ok:
            print(f"🔄 Adjusting joint3 from {current_joint3:.1f}° to {joint3_target}° (difference: {joint3_diff:.1f}°)")
            success = robot_controller.move_joint("joint3", joint3_target, speed=speed)
            if success:
                print(f"✅ Joint3 successfully adjusted to {joint3_target}°")
                adjustments_made = True
            else:
                print(f"❌ Failed to adjust joint3 to {joint3_target}°")
                return False
        
        if adjustments_made:
            # Wait a moment and verify final positions
            time.sleep(1.0)
            
            final_joint2 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint2"])
            final_joint3 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint3"])
            
            if final_joint2 is not None and final_joint3 is not None:
                final_joint2_diff = abs(final_joint2 - joint2_target)
                final_joint3_diff = abs(final_joint3 - joint3_target)
                
                if final_joint2_diff <= tolerance and final_joint3_diff <= tolerance:
                    print(f"✅ Verification: Both joints confirmed at target positions")
                    print(f"   Joint2: {final_joint2:.1f}°, Joint3: {final_joint3:.1f}°")
                else:
                    print(f"⚠️  Verification: Some joints may not have reached target positions")
                    print(f"   Joint2: {final_joint2:.1f}° (target: {joint2_target}°), Joint3: {final_joint3:.1f}° (target: {joint3_target}°)")
        
        return True
                
    except Exception as e:
        print(f"❌ Error checking/adjusting arm joints: {e}")
        return False

def ensure_arm_configuration(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """
    Ensure both joint2 and joint3 are at their target positions for proper arm configuration.
    
    Args:
        robot_controller: SimpleRobotController instance
        joint2_target: Target angle for joint2 (default: -42.4)
        joint3_target: Target angle for joint3 (default: -132.0)
        tolerance: Acceptable deviation from target (default: 2.0 degrees)
        speed: Movement speed (default: 100)
        
    Returns:
        bool: True if both joints are properly positioned
    """
    return check_and_adjust_arm_joints(robot_controller, joint2_target, joint3_target, tolerance, speed)

def main():
    """Main function for arm configuration adjustment script."""
    parser = argparse.ArgumentParser(description="Check and adjust joint2 and joint3 to target angles")
    parser.add_argument("--joint2", type=float, default=-42.4, 
                       help="Target angle for joint2 (default: -42.4)")
    parser.add_argument("--joint3", type=float, default=-132.0, 
                       help="Target angle for joint3 (default: -132.0)")
    parser.add_argument("--joint2-only", type=float, metavar="ANGLE",
                       help="Adjust only joint2 to specified angle")
    parser.add_argument("--joint3-only", type=float, metavar="ANGLE",
                       help="Adjust only joint3 to specified angle")
    parser.add_argument("--tolerance", type=float, default=2.0,
                       help="Acceptable deviation from target in degrees (default: 2.0)")
    parser.add_argument("--speed", type=int, default=100,
                       help="Movement speed (0-1000, default: 100)")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check positions, don't adjust")
    
    args = parser.parse_args()
    
    print("🔧 Arm Configuration Check and Adjustment Tool")
    print("=" * 60)
    
    # Determine operation mode
    if args.joint2_only is not None:
        print(f"Mode: Joint2 only adjustment")
        print(f"Joint2 target: {args.joint2_only}°")
    elif args.joint3_only is not None:
        print(f"Mode: Joint3 only adjustment")
        print(f"Joint3 target: {args.joint3_only}°")
    else:
        print(f"Mode: Full arm configuration")
        print(f"Joint2 target: {args.joint2}°")
        print(f"Joint3 target: {args.joint3}°")
    
    print(f"Tolerance: ±{args.tolerance}°")
    print(f"Speed: {args.speed}")
    print(f"Operation: {'Check only' if args.check_only else 'Check and adjust'}")
    print("=" * 60)
    
    if not ROBOT_AVAILABLE:
        print("❌ Robot modules not available")
        sys.exit(1)
    
    # Create robot controller
    robot = SimpleRobotController()
    
    try:
        # Connect to robot
        print("🔌 Connecting to robot...")
        if not robot.connect():
            print("❌ Failed to connect to robot")
            sys.exit(1)
        
        print("✅ Robot connected successfully")
        
        # Determine what to adjust based on arguments
        if args.joint2_only is not None:
            # Adjust only joint2
            print(f"📊 Checking joint2 position...")
            current_angle = robot.servo_manager.query_servo_angle(robot.motor_ids["joint2"])
            
            if current_angle is None:
                print("❌ Could not read joint2 position - servo may not be responding")
                sys.exit(1)
            
            print(f"📊 Current joint2 position: {current_angle:.1f}°")
            
            if args.check_only:
                angle_diff = abs(current_angle - args.joint2_only)
                if angle_diff <= args.tolerance:
                    print(f"✅ Joint2 is at correct position: {current_angle:.1f}°")
                else:
                    print(f"⚠️  Joint2 needs adjustment: {angle_diff:.1f}° difference from target")
            else:
                # For single joint adjustment, we still use the combined function but set the other joint to current position
                current_joint3 = robot.servo_manager.query_servo_angle(robot.motor_ids["joint3"])
                if current_joint3 is None:
                    current_joint3 = -132.0  # Default fallback
                success = check_and_adjust_arm_joints(robot, args.joint2_only, current_joint3, args.tolerance, args.speed)
                if not success:
                    print("❌ Joint2 adjustment failed")
                    sys.exit(1)
        
        elif args.joint3_only is not None:
            # Adjust only joint3
            print(f"📊 Checking joint3 position...")
            current_angle = robot.servo_manager.query_servo_angle(robot.motor_ids["joint3"])
            
            if current_angle is None:
                print("❌ Could not read joint3 position - servo may not be responding")
                sys.exit(1)
            
            print(f"📊 Current joint3 position: {current_angle:.1f}°")
            
            if args.check_only:
                angle_diff = abs(current_angle - args.joint3_only)
                if angle_diff <= args.tolerance:
                    print(f"✅ Joint3 is at correct position: {current_angle:.1f}°")
                else:
                    print(f"⚠️  Joint3 needs adjustment: {angle_diff:.1f}° difference from target")
            else:
                # For single joint adjustment, we still use the combined function but set the other joint to current position
                current_joint2 = robot.servo_manager.query_servo_angle(robot.motor_ids["joint2"])
                if current_joint2 is None:
                    current_joint2 = -42.4  # Default fallback
                success = check_and_adjust_arm_joints(robot, current_joint2, args.joint3_only, args.tolerance, args.speed)
                if not success:
                    print("❌ Joint3 adjustment failed")
                    sys.exit(1)
        
        else:
            # Adjust both joints (full arm configuration)
            if args.check_only:
                print("🔍 Check-only mode: Verifying current positions...")
                joint2_angle = robot.servo_manager.query_servo_angle(robot.motor_ids["joint2"])
                joint3_angle = robot.servo_manager.query_servo_angle(robot.motor_ids["joint3"])
                
                if joint2_angle is not None and joint3_angle is not None:
                    joint2_diff = abs(joint2_angle - args.joint2)
                    joint3_diff = abs(joint3_angle - args.joint3)
                    
                    print(f"📊 Current positions: Joint2={joint2_angle:.1f}°, Joint3={joint3_angle:.1f}°")
                    
                    if joint2_diff <= args.tolerance and joint3_diff <= args.tolerance:
                        print("✅ Both joints are at correct positions")
                    else:
                        print(f"⚠️  Adjustments needed: Joint2 diff={joint2_diff:.1f}°, Joint3 diff={joint3_diff:.1f}°")
                else:
                    print("❌ Could not read joint positions")
            else:
                success = check_and_adjust_arm_joints(robot, args.joint2, args.joint3, args.tolerance, args.speed)
                if not success:
                    print("❌ Arm configuration adjustment failed")
                    sys.exit(1)
        
        print("🎉 Operation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        # Disconnect robot
        if robot.is_connected:
            robot.disconnect()
            print("🔌 Robot disconnected")

if __name__ == "__main__":
    main() 