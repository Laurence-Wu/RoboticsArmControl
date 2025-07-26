#!/usr/bin/env python3
"""
Quick Robot Setup Script for Seeed Studio Robotics Arm

This script helps you quickly set up and test your robot arm.

Usage: python3 quick_robot_setup.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Import robot configuration
try:
    from config import SERIAL_CONFIG, update_port_config
except ImportError:
    print("❌ Could not import config.py. Make sure it's in the same directory.")
    sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fashionstar-uart-sdk",
        "pyserial",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "fashionstar-uart-sdk":
                import fashionstar_uart_sdk
            elif package == "pyserial":
                import serial
            elif package == "numpy":
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies are installed!")
        return True

def find_robot_ports():
    """Find potential robot ports."""
    print("\n🔍 Looking for robot ports...")
    
    potential_ports = []
    dev_path = Path("/dev")
    
    # Common patterns for robot arms on macOS/Linux
    patterns = ["tty.*", "ttyUSB*", "ttyACM*", "cu.usbmodem*"]
    
    for pattern in patterns:
        potential_ports.extend(dev_path.glob(pattern))
    
    if potential_ports:
        print("📍 Found potential robot ports:")
        for i, port in enumerate(potential_ports):
            print(f"  {i+1}. {port}")
        return [str(port) for port in potential_ports]
    else:
        print("❌ No robot ports found. Make sure your robot is connected and powered on.")
        return []

def update_robot_config(port):
    """Update robot configuration with the selected port."""
    print(f"\n⚙️ Updating robot configuration to use port: {port}")
    
    try:
        # Update the centralized configuration
        update_port_config(port, "default")
        print("✅ Updated config.py with new port")
        print("✅ All scripts will now use the new port automatically")
        
        # Also note about LeRobot configuration
        config_file = "advX/lerobot/common/robot_devices/robots/configs.py"
        if os.path.exists(config_file):
            print(f"📝 Note: You may also want to update the ports in {config_file}")
            print(f"   Look for StaraiRobotConfig and update the port values to: {port}")
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        print("You may need to manually update config.py")

def test_robot_connection(port):
    """Test basic robot connection."""
    print(f"\n🧪 Testing robot connection on {port}...")
    
    try:
        import serial
        import time
        
        # Try to open the port
        uart = serial.Serial(
            port=port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=1,
            bytesize=8,
            timeout=0.1
        )
        
        if uart.is_open:
            print("✅ Successfully opened serial connection!")
            uart.close()
            return True
        else:
            print("❌ Failed to open serial connection")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🤖 Quick Robot Setup for Seeed Studio Robotics Arm")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Step 2: Find robot ports
    ports = find_robot_ports()
    
    if not ports:
        print("\n❌ No robot ports found. Please:")
        print("  1. Connect your robot via USB")
        print("  2. Power on your robot")
        print("  3. Run this script again")
        return
    
    # Step 3: Select port
    if len(ports) == 1:
        selected_port = ports[0]
        print(f"\n✅ Using port: {selected_port}")
    else:
        print(f"\n📍 Multiple ports found. Please select one:")
        for i, port in enumerate(ports):
            print(f"  {i+1}. {port}")
        
        try:
            choice = int(input("\nEnter choice (1-{}): ".format(len(ports)))) - 1
            if 0 <= choice < len(ports):
                selected_port = ports[choice]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid input")
            return
    
    # Step 4: Test connection
    if not test_robot_connection(selected_port):
        print(f"\n❌ Failed to connect to {selected_port}")
        print("Please check:")
        print("  - Robot is powered on")
        print("  - USB cable is connected properly")
        print("  - Port permissions (try: sudo chmod 666 {})".format(selected_port))
        return
    
    # Step 5: Update configuration
    update_robot_config(selected_port)
    
    # Step 6: Show next steps
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test basic control:")
    print("   python3 simple_robot_control.py")
    print("")
    print("2. Use advanced control:")
    print("   python3 starai_robot_controller.py --mode manual")
    print("")
    print("3. Use LeRobot teleoperation:")
    print("   cd advX && python3 lerobot/scripts/control_robot.py --robot.type=starai --control.type=teleoperate")
    print("")
    print("⚠️  Safety reminders:")
    print("  - Keep emergency stop ready (Ctrl+C)")
    print("  - Start with small movements")
    print("  - Ensure clear workspace around robot")
    print("  - Have robot power switch easily accessible")

if __name__ == "__main__":
    main()