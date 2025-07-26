#!/usr/bin/env python3
"""
Robot Port Detection Script

This script helps you find the correct USB port for your Seeed Studio robotics arm.

Usage: python3 find_robot_port.py
"""

import os
import time
import subprocess
from pathlib import Path

# Import robot configuration
try:
    from config import SERIAL_CONFIG, update_port_config
except ImportError:
    print("❌ Could not import config.py. Make sure it's in the same directory.")
    exit(1)

def list_all_ports():
    """List all available serial ports."""
    print("🔍 All available serial ports:")
    
    # Method 1: List /dev/tty* and /dev/cu* ports
    dev_path = Path("/dev")
    patterns = ["tty.*", "cu.*"]
    
    all_ports = []
    for pattern in patterns:
        ports = list(dev_path.glob(pattern))
        all_ports.extend(ports)
    
    # Filter for likely robot ports
    robot_patterns = ["usbmodem", "USB", "ACM", "ttyS"]
    likely_robot_ports = []
    
    for port in all_ports:
        port_str = str(port)
        if any(pattern in port_str for pattern in robot_patterns):
            likely_robot_ports.append(port_str)
    
    if likely_robot_ports:
        print("📍 Likely robot ports:")
        for port in likely_robot_ports:
            print(f"  {port}")
    else:
        print("❌ No likely robot ports found")
    
    print(f"\n📋 All ports ({len(all_ports)} total):")
    for port in sorted(all_ports):
        print(f"  {port}")
    
    return likely_robot_ports

def test_port_connection(port):
    """Test if we can connect to a port."""
    try:
        import serial
        
        print(f"🧪 Testing connection to {port}...")
        
        # Try to open the port
        ser = serial.Serial(
            port=port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            timeout=1
        )
        
        if ser.is_open:
            print(f"✅ Successfully opened {port}")
            ser.close()
            return True
        else:
            print(f"❌ Failed to open {port}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {port}: {e}")
        return False

def interactive_port_selection():
    """Interactive port selection and testing."""
    print("\n🎯 Interactive Port Detection")
    print("=" * 40)
    
    # List ports
    likely_ports = list_all_ports()
    
    if not likely_ports:
        print("\n❌ No likely robot ports found.")
        print("Please ensure:")
        print("  1. Robot is connected via USB")
        print("  2. Robot is powered on")
        print("  3. USB drivers are installed")
        return None
    
    # Test each likely port
    working_ports = []
    for port in likely_ports:
        if test_port_connection(port):
            working_ports.append(port)
    
    if working_ports:
        print(f"\n✅ Found {len(working_ports)} working port(s):")
        for i, port in enumerate(working_ports):
            print(f"  {i+1}. {port}")
        
        if len(working_ports) == 1:
            selected_port = working_ports[0]
            print(f"\n🎯 Using port: {selected_port}")
        else:
            try:
                choice = int(input(f"\nSelect port (1-{len(working_ports)}): ")) - 1
                if 0 <= choice < len(working_ports):
                    selected_port = working_ports[choice]
                    print(f"🎯 Selected port: {selected_port}")
                else:
                    print("❌ Invalid choice")
                    return None
            except ValueError:
                print("❌ Invalid input")
                return None
        
        return selected_port
    else:
        print("\n❌ No working ports found.")
        return None

def update_scripts_with_port(port):
    """Update robot configuration with the detected port."""
    print(f"\n⚙️ Updating configuration to use port: {port}")
    
    try:
        # Update the centralized configuration
        update_port_config(port, "default")
        print("✅ Updated config.py with new port")
        print("✅ All scripts will now use the new port automatically")
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        print("You may need to manually update config.py")

def show_next_steps(port):
    """Show next steps after port detection."""
    print(f"\n🎉 Port detection completed!")
    print(f"📍 Robot port: {port}")
    print("\n📋 Next steps:")
    print("1. Test basic robot control:")
    print("   python3 simple_robot_control.py")
    print("")
    print("2. If that works, try advanced control:")
    print("   python3 starai_robot_controller.py --mode test")
    print("")
    print("3. For teleoperation:")
    print("   cd advX && python3 lerobot/scripts/control_robot.py --robot.type=starai --control.type=teleoperate")
    print("")
    print("⚠️  Safety reminders:")
    print("  - Keep emergency stop ready (Ctrl+C)")
    print("  - Start with small movements")
    print("  - Ensure clear workspace around robot")

def main():
    """Main function."""
    print("🔍 Robot Port Detection for Seeed Studio Robotics Arm")
    print("=" * 60)
    
    print("This script will help you find the correct USB port for your robot.")
    print("Make sure your robot is connected via USB and powered on.\n")
    
    # Interactive port detection
    detected_port = interactive_port_selection()
    
    if detected_port:
        # Update scripts
        update_scripts_with_port(detected_port)
        
        # Show next steps
        show_next_steps(detected_port)
    else:
        print("\n❌ Port detection failed.")
        print("Manual steps:")
        print("1. Connect robot via USB")
        print("2. Power on robot")
        print("3. Run: ls /dev/tty.usbmodem* /dev/cu.usbmodem*")
        print("4. Update the port in simple_robot_control.py manually")

if __name__ == "__main__":
    main()