#!/usr/bin/env python3
"""
Emergency Stop Script for Seeed Studio Robotics Arm

This script immediately disables all motors for emergency situations.
It's designed to be fast and simple.

Usage:
    python3 emergency_stop.py [PORT]

Example:
    python3 emergency_stop.py /dev/tty.usbmodem575E0031751
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


def emergency_stop(port=None):
    """
    Emergency stop - immediately disable all motors.
    
    Args:
        port: USB port for the robot arm (uses config default if None)
    """
    if port is None:
        port = SERIAL_CONFIG.DEFAULT_PORT
    
    print("🚨 EMERGENCY STOP ACTIVATED!")
    print(f"🔌 Connecting to {port}...")
    
    try:
        # Quick connection setup
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
        
        print("🔌 Disabling all motors...")
        
        # Motor IDs from configuration
        motor_ids = MOTOR_CONFIG.get_motor_list()
        
        disabled_count = 0
        
        # Disable each motor
        for motor_id in motor_ids:
            try:
                # Disable torque (power) - this makes the motor free to move
                servo_manager.disable_torque(motor_id)
                disabled_count += 1
                print(f"  ✅ Motor {motor_id} disabled")
            except Exception as e:
                print(f"  ⚠️  Motor {motor_id}: {e}")
        
        print(f"\n📊 Disabled {disabled_count}/{len(motor_ids)} motors")
        
        if disabled_count > 0:
            print("✅ EMERGENCY STOP COMPLETE!")
            print("⚠️  Robot arm is now free to move - support it manually if needed!")
            print("⚠️  Motors are disabled - use enable script to restore control!")
        else:
            print("❌ No motors were disabled - check connections!")
        
        uart.close()
        
    except Exception as e:
        print(f"❌ Emergency stop failed: {e}")
        print("Manual emergency stop:")
        print("  1. Turn off robot power")
        print("  2. Disconnect USB cable")
        print("  3. Support robot arm manually")


def main():
    """Main function."""
    # Get port from command line argument or use config default
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = SERIAL_CONFIG.DEFAULT_PORT
    
    print("🚨 Emergency Stop for Seeed Studio Robotics Arm")
    print("=" * 50)
    print("This will immediately disable all motors!")
    print("The robot arm will become free to move.")
    print("")
    
    # Quick confirmation
    response = input("Continue with emergency stop? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        emergency_stop(port)
    else:
        print("❌ Emergency stop cancelled")


if __name__ == "__main__":
    main()