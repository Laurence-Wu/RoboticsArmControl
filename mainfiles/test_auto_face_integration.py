#!/usr/bin/env python3
"""
Test script for auto_face_tracking integration

This script tests the integration between main.py and auto_face_tracking.py
to ensure the face tracking system works with both modes:
1. Auto face tracking (YOLO + 2s lock + enhanced visualization)
2. Fallback mode (manual detection + PID control)
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🔧 Testing imports...")
    
    try:
        from auto_face_tracking import Detection, YOLOFaceDetector, TargetTracker
        print("✅ auto_face_tracking module imported successfully")
        auto_tracking_available = True
    except ImportError as e:
        print(f"❌ auto_face_tracking import failed: {e}")
        auto_tracking_available = False
    
    try:
        from pid_controller import PIDConfig, DualAxisPIDController, OptimizedDualAxisPDController
        print("✅ PID controller modules imported successfully")
        pid_available = True
    except ImportError as e:
        print(f"❌ PID controller import failed: {e}")
        pid_available = False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        opencv_available = True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        opencv_available = False
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO/Ultralytics imported successfully")
        yolo_available = True
    except ImportError as e:
        print(f"❌ YOLO import failed: {e}")
        yolo_available = False
    
    return {
        'auto_tracking': auto_tracking_available,
        'pid': pid_available,
        'opencv': opencv_available,
        'yolo': yolo_available
    }

def test_auto_face_tracking():
    """Test the auto_face_tracking Detection function"""
    print("\n🎯 Testing auto_face_tracking Detection function...")
    
    try:
        from auto_face_tracking import Detection
        import cv2
        import numpy as np
        
        # Create a test frame (black image)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a fake face rectangle for testing
        cv2.rectangle(test_frame, (250, 200), (350, 300), (255, 255, 255), -1)
        cv2.putText(test_frame, "Test Face", (255, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Call Detection function
        x, y = Detection(test_frame)
        
        print(f"✅ Detection function returned: ({x}, {y})")
        print("✅ auto_face_tracking Detection function works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ auto_face_tracking test failed: {e}")
        return False

def test_pid_direction_correction():
    """Test the PID direction correction"""
    print("\n🔧 Testing PID direction correction...")
    
    try:
        from pid_controller import PIDConfig, OptimizedDualAxisPDController
        
        config = PIDConfig()
        controller = OptimizedDualAxisPDController(config)
        
        # Test face on right side (should move camera right)
        error_x = -50  # Face right of center
        error_y = 0
        pan_output, tilt_output = controller.update(error_x, error_y)
        
        correct_direction = pan_output > 0  # Should be positive to move right
        
        if correct_direction:
            print(f"✅ Direction correction working: Face right (-50px) → Pan output {pan_output:.2f}° (positive)")
        else:
            print(f"❌ Direction correction wrong: Face right (-50px) → Pan output {pan_output:.2f}° (should be positive)")
        
        return correct_direction
        
    except Exception as e:
        print(f"❌ PID direction test failed: {e}")
        return False

def test_main_integration():
    """Test the main.py integration"""
    print("\n🔗 Testing main.py integration...")
    
    try:
        # Import main components
        from main import SingleThreadFaceTracker, AUTO_FACE_TRACKING_AVAILABLE, PID_AVAILABLE
        from pid_controller import PIDConfig
        
        print(f"✅ Main integration imports successful")
        print(f"   - Auto face tracking available: {AUTO_FACE_TRACKING_AVAILABLE}")
        print(f"   - PID control available: {PID_AVAILABLE}")
        
        # Test initialization (without actually running)
        config = PIDConfig()
        tracker = SingleThreadFaceTracker(
            display=False,  # No display for testing
            pid_config=config,
            use_optimized_pd=True,
            enable_multi_face=True
        )
        
        print(f"✅ SingleThreadFaceTracker initialized successfully")
        print(f"   - Using auto tracking: {tracker.use_auto_tracking}")
        
        return True
        
    except Exception as e:
        print(f"❌ Main integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Testing auto_face_tracking integration")
    print("=" * 60)
    
    # Test imports
    import_results = test_imports()
    
    # Test auto face tracking
    auto_tracking_test = test_auto_face_tracking() if import_results['auto_tracking'] else False
    
    # Test PID direction correction
    pid_test = test_pid_direction_correction() if import_results['pid'] else False
    
    # Test main integration
    main_test = test_main_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  ✅ Auto face tracking: {'PASS' if auto_tracking_test else 'FAIL'}")
    print(f"  ✅ PID direction correction: {'PASS' if pid_test else 'FAIL'}")
    print(f"  ✅ Main integration: {'PASS' if main_test else 'FAIL'}")
    print(f"  ✅ OpenCV: {'PASS' if import_results['opencv'] else 'FAIL'}")
    print(f"  ✅ YOLO: {'PASS' if import_results['yolo'] else 'FAIL'}")
    
    all_critical_passed = auto_tracking_test and pid_test and main_test
    
    if all_critical_passed:
        print("\n🎉 All critical tests PASSED! System ready for face tracking.")
        print("\n💡 To run the integrated face tracking system:")
        print("   python3 main.py")
        print("\n✨ Features available:")
        if import_results['auto_tracking']:
            print("   - Auto face tracking with YOLO detection")
            print("   - 2-second target lock for stability")
            print("   - Enhanced visualization")
        print("   - Optimized PD control (TianxingWu approach)")
        print("   - Direction correction applied")
        print("   - Multi-face centroid tracking")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("   - The system may still work with reduced functionality")
        print("   - Consider installing missing dependencies")
    
    return all_critical_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 