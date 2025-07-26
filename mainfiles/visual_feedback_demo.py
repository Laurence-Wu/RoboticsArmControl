#!/usr/bin/env python3
"""
Visual Feedback System Demonstration

This script demonstrates the enhanced visual feedback system for PID face tracking.
It shows how to integrate the new display system with existing face tracking components.

Usage:
    python3 visual_feedback_demo.py
"""

import cv2
import numpy as np
import time
import sys
from typing import List, Tuple, Optional

# Import the visual feedback system
from visual_feedback_system import (
    EnhancedDisplayManager, TrackingOverlayData, PIDDisplayData, SystemStatusData,
    create_display_manager, draw_enhanced_tracking_info
)

# Import existing components
try:
    from pid_config import get_config
    from pid_controller import DualAxisPIDController
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  PID configuration not available, using mock data")
    CONFIG_AVAILABLE = False


class MockFaceDetector:
    """Mock face detector for demonstration purposes."""
    
    def __init__(self):
        self.frame_count = 0
        self.face_positions = [
            (200, 150, 100, 120),  # x, y, w, h
            (220, 160, 100, 120),
            (240, 170, 100, 120),
            (250, 180, 100, 120),
            (260, 190, 100, 120),
            (270, 200, 100, 120),
            (280, 190, 100, 120),
            (290, 180, 100, 120),
            (300, 170, 100, 120),
            (310, 160, 100, 120),
        ]
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Simulate face detection with moving face."""
        self.frame_count += 1
        
        # Simulate occasional detection failures
        if self.frame_count % 50 == 0:
            return []  # No faces detected
        
        # Get current face position
        pos_index = (self.frame_count // 5) % len(self.face_positions)
        x, y, w, h = self.face_positions[pos_index]
        
        # Add some noise
        noise_x = int(np.random.normal(0, 5))
        noise_y = int(np.random.normal(0, 5))
        
        # Return face with confidence
        confidence = 0.85 + np.random.normal(0, 0.1)
        confidence = max(0.5, min(1.0, confidence))
        
        return [(x + noise_x, y + noise_y, w, h, confidence)]


class MockMotionDataCollector:
    """Mock motion data collector for demonstration."""
    
    def __init__(self):
        self.collection_start_time = None
        self.collection_duration = 2.0
        self.is_collecting = False
    
    def start_collection(self):
        """Start data collection."""
        self.collection_start_time = time.time()
        self.is_collecting = True
    
    def get_collection_progress(self) -> float:
        """Get collection progress (0.0 to 1.0)."""
        if not self.is_collecting or self.collection_start_time is None:
            return 0.0
        
        elapsed = time.time() - self.collection_start_time
        progress = elapsed / self.collection_duration
        
        if progress >= 1.0:
            self.is_collecting = False
            return 1.0
        
        return progress
    
    def is_collection_active(self) -> bool:
        """Check if collection is active."""
        return self.is_collecting


def demonstrate_basic_display():
    """Demonstrate basic display functionality."""
    print("🎯 Demonstrating Basic Display Functionality")
    print("=" * 60)
    
    # Create display manager
    display_manager = create_display_manager("Basic Display Demo")
    
    if not display_manager.start():
        print("❌ Failed to start display system")
        return
    
    print("✅ Display system started")
    print("📋 Controls: Q/ESC=Quit, R=Reset, P=Pause, D=Debug, M=Mode, H=Help")
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Create test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (40, 40, 40)  # Dark gray background
            
            # Add some visual elements to the frame
            cv2.putText(frame, "Basic Display Demo", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Simulate simple tracking data
            center_x = 320 + int(100 * np.sin(frame_count * 0.1))
            center_y = 240 + int(50 * np.cos(frame_count * 0.15))
            
            faces = [(center_x - 50, center_y - 60, 100, 120, 0.9)]
            target_pos = (center_x, center_y)
            error_x = center_x - 320
            error_y = center_y - 240
            
            # Update display data
            display_manager.update_tracking_data(
                faces=faces,
                target_position=target_pos,
                center_position=(960, 540),  # 1920x1080
                error_vector=(error_x, error_y),
                target_locked=True,
                confidence=0.9
            )
            
            display_manager.update_pid_data(
                pan_output=error_x * 0.1,
                tilt_output=error_y * 0.1,
                pan_error=error_x,
                tilt_error=error_y,
                pan_gains=(0.1, 0.01, 0.05),
                tilt_gains=(0.1, 0.01, 0.05)
            )
            
            display_manager.update_system_status(
                tracking_active=True,
                motor_status="Simulated",
                safety_status="OK"
            )
            
            # Display frame
            key_result = display_manager.display_frame(frame)
            
            if key_result == -1:  # Quit
                break
            elif key_result == 1:  # Reset
                print("🔄 Reset requested")
                frame_count = 0
                start_time = time.time()
            
            frame_count += 1
            
            # Limit frame rate
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    
    finally:
        display_manager.stop()
        print("👋 Basic display demo finished")


def demonstrate_advanced_features():
    """Demonstrate advanced display features."""
    print("\n🚀 Demonstrating Advanced Display Features")
    print("=" * 60)
    
    # Create components
    display_manager = create_display_manager("Advanced Features Demo")
    face_detector = MockFaceDetector()
    motion_collector = MockMotionDataCollector()
    
    if CONFIG_AVAILABLE:
        config = get_config()
        pid_controller = DualAxisPIDController(config)
    else:
        pid_controller = None
    
    if not display_manager.start():
        print("❌ Failed to start display system")
        return
    
    print("✅ Advanced display system started")
    print("📋 Features: Face tracking, PID control, data collection progress")
    print("📋 Controls: Q/ESC=Quit, R=Reset, P=Pause, D=Debug, M=Mode, H=Help")
    
    try:
        frame_count = 0
        last_collection_start = 0
        
        while True:
            # Create camera-like frame
            frame = np.random.randint(20, 60, (480, 640, 3), dtype=np.uint8)
            
            # Add title
            cv2.putText(frame, "Advanced PID Face Tracking Demo", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Simulate face detection
            faces = face_detector.detect_faces(frame)
            
            # Determine target position
            target_position = None
            error_x, error_y = 0.0, 0.0
            
            if faces:
                face = faces[0]  # Use first face
                face_center_x = face[0] + face[2] // 2
                face_center_y = face[1] + face[3] // 2
                target_position = (face_center_x, face_center_y)
                
                # Calculate error from center
                error_x = face_center_x - 960  # 1920x1080
                error_y = face_center_y - 540  # 1920x1080
            
            # Update PID controller
            pan_output, tilt_output = 0.0, 0.0
            pan_gains, tilt_gains = (0.1, 0.01, 0.05), (0.1, 0.01, 0.05)
            
            if pid_controller and faces:
                pan_output, tilt_output = pid_controller.update(error_x, error_y)
                pan_stats = pid_controller.get_stats()['pan']
                tilt_stats = pid_controller.get_stats()['tilt']
                pan_gains = (pan_stats['gains']['Kp'], pan_stats['gains']['Ki'], pan_stats['gains']['Kd'])
                tilt_gains = (tilt_stats['gains']['Kp'], tilt_stats['gains']['Ki'], tilt_stats['gains']['Kd'])
            
            # Simulate data collection cycles
            if frame_count - last_collection_start > 150:  # Start collection every 5 seconds
                motion_collector.start_collection()
                last_collection_start = frame_count
            
            collection_progress = motion_collector.get_collection_progress()
            
            # Update display data
            display_manager.update_tracking_data(
                faces=faces,
                target_position=target_position,
                center_position=(960, 540),  # 1920x1080
                error_vector=(error_x, error_y),
                target_locked=len(faces) > 0,
                confidence=faces[0][4] if faces else 0.0
            )
            
            display_manager.update_pid_data(
                pan_output=pan_output,
                tilt_output=tilt_output,
                pan_error=error_x,
                tilt_error=error_y,
                pan_gains=pan_gains,
                tilt_gains=tilt_gains,
                integral_pan=pan_output * 10,  # Mock integral term
                integral_tilt=tilt_output * 10,  # Mock integral term
                derivative_pan=pan_output * 0.1,  # Mock derivative term
                derivative_tilt=tilt_output * 0.1   # Mock derivative term
            )
            
            # Simulate system status changes
            safety_status = "OK"
            error_message = None
            
            if frame_count % 200 == 100:  # Simulate warning
                safety_status = "Warning"
                error_message = "High error detected"
            elif frame_count % 300 == 200:  # Simulate error
                safety_status = "Error"
                error_message = "Face detection failed"
            
            display_manager.update_system_status(
                tracking_active=len(faces) > 0,
                data_collection_active=motion_collector.is_collection_active(),
                collection_progress=collection_progress,
                motor_status="Simulated" if len(faces) > 0 else "Idle",
                safety_status=safety_status,
                error_message=error_message
            )
            
            # Display frame
            key_result = display_manager.display_frame(frame)
            
            if key_result == -1:  # Quit
                break
            elif key_result == 1:  # Reset
                print("🔄 Reset requested")
                frame_count = 0
                last_collection_start = 0
                face_detector.frame_count = 0
            
            frame_count += 1
            
            # Limit frame rate
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    
    finally:
        display_manager.stop()
        print("👋 Advanced features demo finished")


def demonstrate_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    print("\n🔄 Demonstrating Backward Compatibility")
    print("=" * 60)
    
    print("✅ Testing draw_enhanced_tracking_info function")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)
    
    # Mock existing data structures
    class MockMotionData:
        def __init__(self):
            self.center_position = (960, 540)  # 1920x1080
            self.vector = (30.0, -15.0)
    
    faces = [(200, 150, 100, 120, 0.95), (400, 200, 80, 100, 0.87)]
    target_position = (250, 210)
    motion_data = MockMotionData()
    pid_output = (2.5, -1.2)
    pid_gains = ((0.1, 0.01, 0.05), (0.12, 0.015, 0.06))
    collection_progress = 0.75
    
    # Test the backward compatibility function
    try:
        draw_enhanced_tracking_info(
            frame=frame,
            faces=faces,
            target_position=target_position,
            motion_data=motion_data,
            pid_output=pid_output,
            pid_gains=pid_gains,
            collection_progress=collection_progress
        )
        
        print("✅ Backward compatibility function works correctly")
        
        # Display the result briefly
        cv2.imshow("Backward Compatibility Test", frame)
        print("📋 Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
    
    print("👋 Backward compatibility demo finished")


def main():
    """Main demonstration function."""
    print("🎨 Visual Feedback System Demonstration")
    print("=" * 80)
    print("This demo shows the enhanced visual feedback system for PID face tracking.")
    print("The system provides real-time display of tracking overlays, PID status,")
    print("system parameters, and user interface elements.")
    print("=" * 80)
    
    try:
        # Check if we can create a display
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("Display Test", test_frame)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        print("✅ Display system available")
        
    except Exception as e:
        print(f"❌ Display system not available: {e}")
        print("💡 This demo requires a display environment (not headless)")
        return 1
    
    # Run demonstrations
    try:
        # Basic display functionality
        demonstrate_basic_display()
        
        # Advanced features
        demonstrate_advanced_features()
        
        # Backward compatibility
        demonstrate_backward_compatibility()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n📋 Key Features Demonstrated:")
        print("  ✅ Real-time face tracking overlays")
        print("  ✅ PID controller status display")
        print("  ✅ System status monitoring")
        print("  ✅ Data collection progress bars")
        print("  ✅ Keyboard input handling")
        print("  ✅ Multiple display modes")
        print("  ✅ Error handling and recovery")
        print("  ✅ Backward compatibility")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Demonstration interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """Run the demonstration."""
    try:
        exit_code = main()
        print("\n👋 Visual feedback system demonstration finished")
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)