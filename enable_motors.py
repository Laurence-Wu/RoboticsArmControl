#!/usr/bin/env python3
"""
Motor Enable Script for Seeed Studio Robotics Arm

This script re-enables all motors after they have been disabled.
Use this after running disable_motors.py or emergency_stop.py.

Usage:
    python3 enable_motors.py [PORT]

Example:
    python3 enable_motors.py /dev/tty.usbmodem575E0031751
"""

import sys
import time

# Import robot configuration
try:
    from config import SERIAL_CONFIG, MOTOR_CONFIG
except ImportError:
    print("❌ Could not import config.py. Make sure it's in the same directory.")
    sys.exit(1)

# Try to import required modules
try:
    import fashionstar_uart_sdk as uservo
    import serial
except ImportError:
    print("❌ Missing required modules. Install with:")
    print("pip install fashionstar-uart-sdk pyserial")
    sys.exit(1)


def enable_motors(port=None):
    """
    Enable all motors and restore servo control.
    
    Args:
        port: USB port for the robot arm (uses config default if None)
    """
    if port is None:
        port = SERIAL_CONFIG.DEFAULT_PORT
    
    print("🔋 ENABLING MOTORS")
    print(f"🔌 Connecting to {port}...")
    
    try:
        # Connection setup
        uart = serial.Serial(
            port=port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=1,
            bytesize=8,
            timeout=0.1
        )
        
        servo_manager = uservo.UartServoManager(uart)
        time.sleep(0.01)  # Brief pause for initialization
        
        print("🔋 Enabling all motors...")
        
        # Motor IDs from configuration
        motor_ids = MOTOR_CONFIG.get_motor_list()
        
        enabled_count = 0
        
        # Enable each motor
        for motor_id in motor_ids:
            try:
                # Enable torque (power) - this puts the motor under servo control
                servo_manager.set_servo_torque_switch(motor_id, True)
                enabled_count += 1
                print(f"  ✅ Motor {motor_id} enabled")
                time.sleep(0.1)  # Small delay between motors
            except Exception as e:
                print(f"  ⚠️  Motor {motor_id}: {e}")
        
        print(f"\n📊 Enabled {enabled_count}/{len(motor_ids)} motors")
        
        if enabled_count > 0:
            print("✅ MOTORS ENABLED!")
            print("⚠️  Robot is now under servo control!")
            print("⚠️  Motors may move to hold position - ensure safe workspace!")
            print("✅ You can now use the robot control scripts!")
        else:
            print("❌ No motors were enabled - check connections!")
        
        uart.close()
        
    except Exception as e:
        print(f"❌ Motor enable failed: {e}")
        print("Troubleshooting:")
        print("  1. Check robot power")
        print("  2. Check USB connection")
        print("  3. Verify correct port")
        print("  4. Try running simple_robot_control.py for diagnostics")


def main():
    """Main function."""
    # Get port from command line argument or use config default
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = SERIAL_CONFIG.DEFAULT_PORT
    
    print("🔋 Motor Enable for Seeed Studio Robotics Arm")
    print("=" * 50)
    print("This will enable all motors and restore servo control.")
    print("The robot may move to hold its current position.")
    print("")
    
    # Safety warning
    print("⚠️  SAFETY WARNING:")
    print("  - Ensure clear workspace around robot")
    print("  - Be ready to power off if needed")
    print("  - Robot may move when motors are enabled")
    print("")
    
    # Confirmation
    response = input("Continue with motor enable? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        enable_motors(port)
    else:
        print("❌ Motor enable cancelled")


if __name__ == "__main__":
    main()