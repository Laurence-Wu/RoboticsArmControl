#!/usr/bin/env python3
"""
Disable All Motors Script for Utils Directory

This script safely disables and powers off all motors in the robotic arm.
Use this for emergency stops or safe shutdown procedures.

Usage:
    python3 utils/disable_all_motors.py
"""

import time
import sys
import os

# Import robot configuration
try:
    # Try local import first (when running from utils directory)
    from config import SERIAL_CONFIG, MOTOR_CONFIG
except ImportError:
    try:
        # Try utils.config import (when running from parent directory)
        from utils.config import SERIAL_CONFIG, MOTOR_CONFIG
    except ImportError:
        try:
            # Try mainfiles.config as fallback
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mainfiles.config import SERIAL_CONFIG, MOTOR_CONFIG
        except ImportError:
            print("❌ Could not import config. Make sure config.py is available.")
            print("   Try running from the project root, utils, or mainfiles directory.")
            sys.exit(1)

# Try to import required modules
try:
    import fashionstar_uart_sdk as uservo
    import serial
except ImportError:
    print("❌ Missing required modules. Install with:")
    print("pip install fashionstar-uart-sdk pyserial")
    sys.exit(1)


class MotorShutdownController:
    """Controller for safely shutting down all motors."""
    
    def __init__(self, port=None, baudrate=None):
        """
        Initialize the shutdown controller.
        
        Args:
            port: USB port for the robot arm (uses config default if None)
            baudrate: Communication baudrate (uses config default if None)
        """
        self.port = port or SERIAL_CONFIG.DEFAULT_PORT
        self.baudrate = baudrate or SERIAL_CONFIG.BAUDRATE
        self.uart = None
        self.servo_manager = None
        self.is_connected = False
        
        # Import motor configuration from config.py
        self.motor_ids = MOTOR_CONFIG.MOTOR_IDS.copy()
    
    def connect(self):
        """Connect to the robot arm."""
        try:
            print(f"🔌 Connecting to robot at {self.port}...")
            
            # Initialize serial connection
            self.uart = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=1,
                bytesize=8,
                timeout=0.1
            )
            
            # Initialize servo manager
            self.servo_manager = uservo.UartServoManager(self.uart)
            
            self.is_connected = True
            print("✅ Successfully connected to robot!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            print("Check that:")
            print(f"  - Robot is connected to {self.port}")
            print("  - Robot is powered on")
            print("  - USB port has correct permissions")
            return False
    
    def disconnect(self):
        """Disconnect from the robot arm."""
        if self.uart and self.uart.is_open:
            self.uart.close()
            self.is_connected = False
            print("✅ Disconnected from robot")
    
    def disable_single_motor(self, motor_id, joint_name=None):
        """
        Disable a single motor.
        
        Args:
            motor_id: Motor ID (0-6)
            joint_name: Optional joint name for logging
        """
        try:
            # Use the correct Fashion Star servo SDK method
            # Based on inspection, the correct method is disable_torque()
            self.servo_manager.disable_torque(motor_id)
            print(f"✅ Disabled torque for motor {motor_id}" + (f" ({joint_name})" if joint_name else ""))
            return True
            
        except Exception as e:
            print(f"❌ Failed to disable motor {motor_id}" + (f" ({joint_name})" if joint_name else "") + f": {e}")
            return False
    
    def disable_all_motors(self):
        """Disable all motors at once."""
        if not self.is_connected:
            print("❌ Robot not connected")
            return False
        
        print("🛑 DISABLING ALL MOTORS...")
        print("=" * 40)
        
        success_count = 0
        total_motors = len(self.motor_ids)
        
        # Disable each motor individually (broadcast doesn't seem to work reliably)
        print("🔄 Disabling motors individually...")
        for joint_name, motor_id in self.motor_ids.items():
            if self.disable_single_motor(motor_id, joint_name):
                success_count += 1
            time.sleep(0.1)  # Small delay between commands
        
        print("=" * 40)
        print(f"📊 Disabled {success_count}/{total_motors} motors")
        
        if success_count == total_motors:
            print("✅ ALL MOTORS SUCCESSFULLY DISABLED")
            print("⚠️  Robot arm is now free to move - support manually if needed!")
            return True
        elif success_count > 0:
            print("⚠️  PARTIAL SUCCESS - Some motors may still be active")
            return False
        else:
            print("❌ FAILED TO DISABLE ANY MOTORS")
            return False
    
    def emergency_stop(self):
        """Emergency stop - try multiple methods to disable all motors."""
        print("🚨 EMERGENCY STOP ACTIVATED 🚨")
        print("=" * 50)
        
        if not self.is_connected:
            print("❌ Robot not connected - cannot send stop commands")
            return False
        
        # Direct motor disable using the proven method
        success = self.disable_all_motors()
        
        if success:
            print("✅ EMERGENCY STOP SUCCESSFUL")
        else:
            print("❌ EMERGENCY STOP FAILED - MANUALLY POWER OFF THE ROBOT")
        
        return success
    
    def status_check(self):
        """Check the status of all motors."""
        if not self.is_connected:
            print("❌ Robot not connected")
            return
        
        print("📊 Motor Status Check")
        print("=" * 30)
        
        for joint_name, motor_id in self.motor_ids.items():
            try:
                # Try to query the motor
                angle = self.servo_manager.query_servo_angle(motor_id)
                if angle is not None:
                    print(f"✅ {joint_name} (ID: {motor_id}) - Active (angle: {angle}°)")
                else:
                    print(f"❌ {joint_name} (ID: {motor_id}) - Not responding")
            except Exception as e:
                print(f"❌ {joint_name} (ID: {motor_id}) - Error: {e}")


def parse_command_line_args():
    """Parse command line arguments and return mode flags."""
    emergency_mode = False
    status_mode = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--emergency', '-e']:
            emergency_mode = True
            print("🚨 EMERGENCY MODE ACTIVATED")
        elif sys.argv[1] in ['--status', '-s']:
            status_mode = True
            print("📊 STATUS CHECK MODE")
        else:
            print("Usage: python3 utils/disable_all_motors.py [--emergency|-e] [--status|-s]")
            return None, None
    
    return emergency_mode, status_mode


def run_interactive_mode(controller):
    """Run interactive mode for motor control."""
    print("\nWhat would you like to do?")
    print("1. Disable all motors (safe shutdown)")
    print("2. Emergency stop (try all methods)")
    print("3. Check motor status")
    print("4. Exit without changes")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        controller.disable_all_motors()
    elif choice == "2":
        controller.emergency_stop()
    elif choice == "3":
        controller.status_check()
    elif choice == "4":
        print("👋 Exiting without changes")
    else:
        print("❌ Invalid choice")


def main():
    """Main function."""
    print("🛑 Motor Disable Script for Seeed Studio Robotics Arm (Utils)")
    print("=" * 70)
    
    # Parse command line arguments
    result = parse_command_line_args()
    if result == (None, None):
        return
    emergency_mode, status_mode = result
    
    # Configuration from config.py
    print(f"📍 Using port: {SERIAL_CONFIG.DEFAULT_PORT}")
    print(f"📡 Baudrate: {SERIAL_CONFIG.BAUDRATE}")
    
    # Create controller
    controller = MotorShutdownController()
    
    try:
        # Connect to robot
        if not controller.connect():
            print("❌ Failed to connect to robot.")
            if emergency_mode:
                print("🚨 EMERGENCY: MANUALLY POWER OFF THE ROBOT IMMEDIATELY!")
            return
        
        # Execute based on mode
        if emergency_mode:
            controller.emergency_stop()
        elif status_mode:
            controller.status_check()
        else:
            run_interactive_mode(controller)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user - attempting emergency stop...")
        try:
            controller.emergency_stop()
        except Exception as emergency_error:
            print(f"🚨 EMERGENCY STOP FAILED: {emergency_error}")
            print("🚨 EMERGENCY: MANUALLY POWER OFF THE ROBOT!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🚨 If robot is still active, manually power it off!")
    finally:
        controller.disconnect()
        print("👋 Script completed")


if __name__ == "__main__":
    main()
