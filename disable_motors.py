#!/usr/bin/env python3
"""
Disable Motors Script

Simple script to safely disable and power off all motors.
This is a gentler version compared to emergency_stop.py

Usage:
    python3 disable_motors.py
"""

import time
import sys

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


def disable_all_motors(port=None):
    """
    Safely disable all motors.
    
    Args:
        port: USB port for the robot arm (uses config default if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if port is None:
        port = SERIAL_CONFIG.DEFAULT_PORT
    
    print("🛑 Disabling All Motors")
    print("=" * 30)
    print(f"🔌 Connecting to {port}...")
    
    uart = None
    try:
        # Connect to robot
        uart = serial.Serial(
            port=port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=1,
            bytesize=8,
            timeout=0.1
        )
        
        servo_manager = uservo.UartServoManager(uart)
        time.sleep(0.01)  # Brief initialization pause
        
        print("✅ Connected successfully")
        print("🔄 Disabling motors...")
        
        # Get all motor IDs
        motor_ids = MOTOR_CONFIG.get_motor_list()
        joint_names = MOTOR_CONFIG.get_joint_names()
        
        disabled_count = 0
        total_motors = len(motor_ids)
        
        # Disable each motor individually
        for i, motor_id in enumerate(motor_ids):
            joint_name = joint_names[i]
            try:
                # Disable torque - this makes the motor free to move
                servo_manager.disable_torque(motor_id)
                disabled_count += 1
                print(f"  ✅ {joint_name} (ID: {motor_id}) - Disabled")
            except Exception as e:
                print(f"  ❌ {joint_name} (ID: {motor_id}) - Failed: {e}")
            
            time.sleep(0.1)  # Small delay between commands
        
        print(f"\n📊 Results: {disabled_count}/{total_motors} motors disabled")
        
        if disabled_count == total_motors:
            print("✅ ALL MOTORS SUCCESSFULLY DISABLED")
            print("⚠️  Robot arm is now free to move - support manually if needed!")
            return True
        elif disabled_count > 0:
            print("⚠️  PARTIAL SUCCESS - Some motors may still be active")
            return False
        else:
            print("❌ FAILED TO DISABLE ANY MOTORS")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print(f"  - Check that robot is connected to {port}")
        print("  - Verify robot power is on")
        print("  - Check USB cable connection")
        return False
    
    finally:
        # Always close the connection
        if uart and uart.is_open:
            uart.close()
            print("🔌 Disconnected")


def main():
    """Main function."""
    print("🤖 Motor Disable Script for Seeed Studio Robotics Arm")
    print("=" * 60)
    
    # Get port from command line or use default
    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"📍 Using port from command line: {port}")
    else:
        port = SERIAL_CONFIG.DEFAULT_PORT
        print(f"📍 Using default port: {port}")
    
    print("\nThis script will safely disable all motors.")
    print("The robot will become free to move after disabling.")
    print("Use enable_motors.py to restore motor control.")
    print()
    
    # Confirmation
    response = input("Continue with motor disable? (Y/n): ").strip().lower()
    if response in ['', 'y', 'yes']:
        success = disable_all_motors(port)
        
        if success:
            print("\n🎉 Motor disable completed successfully!")
            print("📝 Next steps:")
            print("   - Use 'python3 enable_motors.py' to re-enable motors")
            print("   - Support robot arm manually if needed")
        else:
            print("\n⚠️  Motor disable had issues!")
            print("📝 Recommendations:")
            print("   - Try 'python3 emergency_stop.py' for force disable")
            print("   - Check robot connections and power")
            print("   - Manually power off robot if necessary")
    else:
        print("❌ Motor disable cancelled")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        print("⚠️  Motors may still be active - run script again or use emergency_stop.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("🚨 If motors are still active, use emergency_stop.py or manually power off")
