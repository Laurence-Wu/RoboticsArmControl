#!/usr/bin/env python3
"""
Test script to verify the start position function works correctly
"""

import os
import json
import time
from auto_face_tracking import TargetTracker

def test_start_position_file_path():
    """Test if the start position file path is correct"""
    print("🧪 Testing start position file path")
    print("=" * 50)
    
    # Test the default path
    default_path = "positions/startPosition.json"
    print(f"Default path: {default_path}")
    
    # Check if file exists
    if os.path.exists(default_path):
        print(f"✅ File exists: {default_path}")
        
        # Try to read the file
        try:
            with open(default_path, 'r') as f:
                data = json.load(f)
            print(f"✅ File is valid JSON: {data}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
    else:
        print(f"❌ File does not exist: {default_path}")
        return False

def test_target_tracker_initialization():
    """Test TargetTracker initialization with start position file"""
    print("\n🧪 Testing TargetTracker initialization")
    print("=" * 50)
    
    try:
        # Create TargetTracker with default settings
        tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
        
        print(f"✅ TargetTracker created successfully")
        print(f"   Start position file: {tracker.start_position_file}")
        print(f"   Lock duration: {tracker.lock_duration}")
        print(f"   Movement threshold: {tracker.movement_threshold}")
        
        return tracker
    except Exception as e:
        print(f"❌ Error creating TargetTracker: {e}")
        return None

def test_target_tracking_simulation():
    """Simulate target tracking with start position functionality"""
    print("\n🧪 Simulating target tracking with start position")
    print("=" * 50)
    
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    # Simulate detecting a face
    print("1. Detecting face...")
    position = tracker.update_target((500, 300))
    print(f"   Returned position: {position}")
    
    # Simulate face lost for 2.5 seconds (should trigger start position)
    print("\n2. Face lost for 2.5 seconds...")
    start_time = time.time()
    for i in range(6):  # 6 iterations * 0.5 seconds = 3 seconds
        position = tracker.update_target(None)
        elapsed = time.time() - start_time
        print(f"   {i+1}. Elapsed: {elapsed:.1f}s, Position: {position}")
        time.sleep(0.5)
    
    # Simulate face detected again
    print("\n3. Face detected again...")
    position = tracker.update_target((600, 400))
    print(f"   Returned position: {position}")

def test_subprocess_call():
    """Test the subprocess call to move_to_json.py"""
    print("\n🧪 Testing subprocess call to move_to_json.py")
    print("=" * 50)
    
    import subprocess
    
    json_path = "positions/startPosition.json"
    
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    try:
        print(f"🏠 Calling: python3 move_to_json.py {json_path}")
        result = subprocess.run(
            ["python3", "move_to_json.py", json_path],
            capture_output=True, text=True, timeout=30
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        if result.returncode == 0:
            print("✅ Subprocess call successful")
            return True
        else:
            print("❌ Subprocess call failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Subprocess call timed out")
        return False
    except Exception as e:
        print(f"❌ Error in subprocess call: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Start Position Function")
    print("=" * 60)
    
    # Test 1: File path and existence
    file_ok = test_start_position_file_path()
    
    # Test 2: TargetTracker initialization
    tracker = test_target_tracker_initialization()
    
    # Test 3: Target tracking simulation
    if tracker:
        test_target_tracking_simulation()
    
    # Test 4: Subprocess call (only if file exists)
    if file_ok:
        test_subprocess_call()
    
    print("\n✅ All tests completed")

if __name__ == "__main__":
    main() 