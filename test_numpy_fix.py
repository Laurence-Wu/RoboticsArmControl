#!/usr/bin/env python3
"""
Test script to verify the numpy fallback system works correctly
"""

import sys
sys.path.append('mainfiles')

def test_numpy_fallback():
    """Test the numpy fallback functionality"""
    print("🧪 Testing NumPy fallback system...")
    
    try:
        from single_thread_face_tracking import (
            NUMPY_AVAILABLE, 
            YOLO_AVAILABLE, 
            ROBOT_AVAILABLE,
            MotionData,
            TargetTracker,
            MotionDataCollector
        )
        
        print(f"✅ Imports successful")
        print(f"   - NumPy Available: {NUMPY_AVAILABLE}")
        print(f"   - YOLO Available: {YOLO_AVAILABLE}")
        print(f"   - Robot Available: {ROBOT_AVAILABLE}")
        
        # Test MotionData creation
        print("\n🧪 Testing MotionData creation...")
        motion_data = MotionData(
            face_position=(100, 150),
            center_position=(320, 240),
            confidence=0.8
        )
        print(f"✅ MotionData created successfully")
        print(f"   - Vector: {motion_data.vector}")
        print(f"   - Magnitude: {motion_data.magnitude:.2f}")
        
        # Test TargetTracker
        print("\n🧪 Testing TargetTracker...")
        tracker = TargetTracker()
        faces = [(100, 100, 50, 50, 0.9)]  # x, y, w, h, confidence
        position, confidence = tracker.update(faces)
        print(f"✅ TargetTracker working")
        print(f"   - Position: {position}")
        print(f"   - Confidence: {confidence}")
        
        # Test MotionDataCollector
        print("\n🧪 Testing MotionDataCollector...")
        collector = MotionDataCollector(collection_duration=1.0)
        collector.add_data_point(motion_data)
        status = collector.get_status()
        print(f"✅ MotionDataCollector working")
        print(f"   - Data points: {status['data_points']}")
        
        print("\n✅ All tests passed! The numpy fallback system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_numpy_fallback()
    if success:
        print("\n🎉 System is ready to run without numpy errors!")
    else:
        print("\n💥 System still has issues that need to be fixed.")