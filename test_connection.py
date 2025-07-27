#!/usr/bin/env python3
"""
Robot Connection Test Script

This script tests the connection to the Seeed Studio robotics arm step by step.
"""

import time
import sys
import os
from pathlib import Path

# Add mainfiles to path so we can import config
sys.path.insert(0, str(Path(__file__).parent / "mainfiles"))

try:
    from config import SERIAL_CONFIG, MOTOR_CONFIG
    print("✅ Successfully imported configuration")
    print(f"📍 Default port: {SERIAL_CONFIG.DEFAULT_PORT}")
    print(f"📊 Baudrate: {SERIAL_CONFIG.BAUDRATE}")
except ImportError as e:
    print(f"❌ Failed to import configuration: {e}")
    sys.exit(1)

# Test if required modules are available
try:
    import serial
    print("✅ pyserial module available")
except ImportError:
    print("❌ pyserial not installed. Run: python3 -m pip install pyserial")
    sys.exit(1)

try:
    import fashionstar_uart_sdk as uservo
    print("✅ fashionstar_uart_sdk module available")
except ImportError:
    print("❌ fashionstar_uart_sdk not installed. Run: python3 -m pip install fashionstar_uart_sdk")
    sys.exit(1)

def test_port_exists():
    """Test if the configured port exists."""
    port_path = Path(SERIAL_CONFIG.DEFAULT_PORT)
    if port_path.exists():
        print(f"✅ Port {SERIAL_CONFIG.DEFAULT_PORT} exists")
        return True
    else:
        print(f"❌ Port {SERIAL_CONFIG.DEFAULT_PORT} does not exist")
        return False

def test_port_permissions():
    """Test if we have permission to access the port."""
    try:
        port_path = Path(SERIAL_CONFIG.DEFAULT_PORT)
        if port_path.exists():
            # Try to open for reading (should work if permissions are correct)
            with open(SERIAL_CONFIG.DEFAULT_PORT, 'rb') as f:
                print(f"✅ Port {SERIAL_CONFIG.DEFAULT_PORT} is accessible")
                return True
    except PermissionError:
        print(f"❌ Permission denied for {SERIAL_CONFIG.DEFAULT_PORT}")
        print("   Try: sudo chmod 666 {SERIAL_CONFIG.DEFAULT_PORT}")
        return False
    except Exception as e:
        print(f"⚠️  Cannot test port permissions: {e}")
        return False

def test_serial_connection():
    """Test basic serial connection."""
    print(f"\n🔌 Testing serial connection to {SERIAL_CONFIG.DEFAULT_PORT}...")
    
    try:
        # Test with tty port first
        tty_port = SERIAL_CONFIG.DEFAULT_PORT
        print(f"   Trying TTY port: {tty_port}")
        
        uart = serial.Serial(
            port=tty_port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=1,
            bytesize=8,
            timeout=0.1
        )
        
        if uart.is_open:
            print("✅ Successfully opened serial connection")
            uart.close()
            return tty_port, True
        else:
            print("❌ Failed to open serial connection")
            return tty_port, False
            
    except Exception as e:
        print(f"❌ Serial connection failed: {e}")
        
        # Try cu port as alternative
        cu_port = SERIAL_CONFIG.DEFAULT_PORT.replace('/dev/tty.', '/dev/cu.')
        print(f"   Trying CU port as alternative: {cu_port}")
        
        try:
            uart = serial.Serial(
                port=cu_port,
                baudrate=SERIAL_CONFIG.BAUDRATE,
                parity=serial.PARITY_NONE,
                stopbits=1,
                bytesize=8,
                timeout=0.1
            )
            
            if uart.is_open:
                print("✅ Successfully opened serial connection with CU port")
                uart.close()
                return cu_port, True
            else:
                print("❌ Failed to open CU port as well")
                return cu_port, False
                
        except Exception as e2:
            print(f"❌ CU port also failed: {e2}")
            return cu_port, False

def test_servo_communication(port):
    """Test communication with servo manager."""
    print(f"\n🤖 Testing servo communication...")
    
    try:
        # Open serial connection
        uart = serial.Serial(
            port=port,
            baudrate=SERIAL_CONFIG.BAUDRATE,
            parity=serial.PARITY_NONE,
            stopbits=1,
            bytesize=8,
            timeout=0.1
        )
        
        # Initialize servo manager
        servo_manager = uservo.UartServoManager(uart)
        print("✅ Servo manager initialized")
        
        # Reset multi-turn angles
        time.sleep(0.005)
        servo_manager.reset_multi_turn_angle(0xff)
        time.sleep(0.01)
        print("✅ Reset multi-turn angles")
        
        # Try to ping a servo
        test_servo_id = 0  # Joint1
        try:
            result = servo_manager.ping(test_servo_id)
            if result.is_success:
                print(f"✅ Successfully pinged servo {test_servo_id}")
                
                # Try to read servo angle
                result = servo_manager.query_servo_angle(test_servo_id)
                if result.is_success:
                    angle = result.data
                    print(f"✅ Servo {test_servo_id} current angle: {angle}°")
                else:
                    print(f"⚠️  Could not read angle from servo {test_servo_id}: {result.error_msg}")
            else:
                print(f"❌ Failed to ping servo {test_servo_id}: {result.error_msg}")
                
        except Exception as e:
            print(f"❌ Servo communication error: {e}")
        
        uart.close()
        return True
        
    except Exception as e:
        print(f"❌ Servo communication test failed: {e}")
        return False

def main():
    """Run all connection tests."""
    print("🧪 Robot Connection Diagnostic Test")
    print("=" * 50)
    
    # Test 1: Check if port exists
    print("\n1️⃣  Testing port existence...")
    if not test_port_exists():
        print("\n💡 Available USB ports:")
        os.system("ls -la /dev/*usb* 2>/dev/null || echo 'No USB ports found'")
        return False
    
    # Test 2: Check port permissions
    print("\n2️⃣  Testing port permissions...")
    test_port_permissions()
    
    # Test 3: Test serial connection
    print("\n3️⃣  Testing serial connection...")
    port, success = test_serial_connection()
    if not success:
        print("\n💡 Connection troubleshooting:")
        print("   - Make sure robot is powered on")
        print("   - Check USB cable connection")
        print("   - Try a different baudrate")
        print("   - Check if another program is using the port")
        return False
    
    # Test 4: Test servo communication
    print("\n4️⃣  Testing servo communication...")
    if test_servo_communication(port):
        print("\n🎉 All tests passed! Robot connection is working.")
        print(f"✅ Robot is connected and responsive on {port}")
    else:
        print("\n⚠️  Serial connection works but servo communication failed.")
        print("💡 This could mean:")
        print("   - Robot is not powered on")
        print("   - Wrong baudrate (try 115200 instead of 1000000)")
        print("   - Servo IDs are different than expected")
        print("   - Servos need initialization")
    
    return True

if __name__ == "__main__":
    main()
