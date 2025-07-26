#!/usr/bin/env python3
"""
Main Robot Control with Focus-Based Activation

This script integrates focus detection with robotic arm control.
The robot starts in home position and only activates face tracking
when the user is focused. When focus is lost, it returns to home position.

Usage:
    python main.py
"""

import sys
import time
import threading
import os
from typing import Optional

# Add mainfiles to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mainfiles'))

# Import focus detection
try:
    from mainfiles.emoRobots.focus_true_false import start_focus_detector, stop_focus_detector, get_focus_status
    FOCUS_AVAILABLE = True
    print("✅ Focus detection module imported successfully")
except ImportError as e:
    FOCUS_AVAILABLE = False
    print(f"⚠️  Focus detection module not available: {e}")

# Import robot control
try:
    from mainfiles.simple_robot_control import SimpleRobotController
    from mainfiles.config import SERIAL_CONFIG, MOTOR_CONFIG
    ROBOT_AVAILABLE = True
    print("✅ Robot control module imported successfully")
except ImportError as e:
    ROBOT_AVAILABLE = False
    print(f"⚠️  Robot control module not available: {e}")

# Import face tracking
try:
    from mainfiles.auto_face_tracking import YOLOFaceDetector, TargetTracker, Detection
    FACE_TRACKING_AVAILABLE = True
    print("✅ Face tracking module imported successfully")
except ImportError as e:
    FACE_TRACKING_AVAILABLE = False
    print(f"⚠️  Face tracking module not available: {e}")

# Import camera and OpenCV
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
    print("✅ OpenCV module imported successfully")
except ImportError as e:
    OPENCV_AVAILABLE = False
    print(f"⚠️  OpenCV module not available: {e}")


class FocusControlledRobot:
    """
    Main robot controller that integrates focus detection with face tracking
    """
    
    def __init__(self):
        self.robot_controller = None
        self.face_detector = None
        self.target_tracker = None
        self.camera = None
        
        # State management
        self.is_running = False
        self.is_focused = False
        self.is_tracking = False
        self.focus_detector = None
        
        # Threading
        self.focus_thread = None
        self.tracking_thread = None
        self.camera_thread = None
        
        # Locks for thread safety
        self.state_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("\n🔧 Initializing components...")
        
        # Initialize robot controller
        if ROBOT_AVAILABLE:
            try:
                self.robot_controller = SimpleRobotController()
                if self.robot_controller.connect():
                    print("✅ Robot controller initialized and connected")
                else:
                    print("❌ Failed to connect to robot")
                    self.robot_controller = None
            except Exception as e:
                print(f"❌ Robot controller initialization failed: {e}")
                self.robot_controller = None
        
        # Initialize face detector
        if FACE_TRACKING_AVAILABLE:
            try:
                self.face_detector = YOLOFaceDetector()
                self.target_tracker = TargetTracker()
                print("✅ Face detector and target tracker initialized")
            except Exception as e:
                print(f"❌ Face detector initialization failed: {e}")
                self.face_detector = None
                self.target_tracker = None
        
        # Initialize camera
        if OPENCV_AVAILABLE:
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    # Set camera properties
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    print("✅ Camera initialized")
                else:
                    print("❌ Failed to open camera")
                    self.camera = None
            except Exception as e:
                print(f"❌ Camera initialization failed: {e}")
                self.camera = None
    
    def _focus_callback(self, is_focused: bool, score: float):
        """
        Callback function for focus detection events
        """
        with self.state_lock:
            previous_focus = self.is_focused
            self.is_focused = is_focused
            
            print(f"🧠 Focus Status: {'FOCUSED' if is_focused else 'NOT FOCUSED'} (Score: {score:.3f})")
            
            # Handle focus state change
            if is_focused and not previous_focus:
                print("🎯 Focus detected! Starting face tracking...")
                self._start_face_tracking()
            elif not is_focused and previous_focus:
                print("😴 Focus lost! Stopping face tracking and returning to home...")
                self._stop_face_tracking()
                self._return_to_home()
    
    def _start_face_tracking(self):
        """Start face tracking when focus is detected"""
        with self.state_lock:
            if not self.is_tracking and self.face_detector and self.camera:
                self.is_tracking = True
                self.tracking_thread = threading.Thread(target=self._face_tracking_loop, daemon=True)
                self.tracking_thread.start()
                print("📹 Face tracking started")
    
    def _stop_face_tracking(self):
        """Stop face tracking when focus is lost"""
        with self.state_lock:
            self.is_tracking = False
            print("📹 Face tracking stopped")
    
    def _return_to_home(self):
        """Return robot to home position"""
        if self.robot_controller and self.robot_controller.is_connected:
            with self.robot_lock:
                try:
                    print("🏠 Returning to home position...")
                    self.robot_controller.go_home()
                    time.sleep(2)  # Wait for movement to complete
                    print("✅ Robot returned to home position")
                except Exception as e:
                    print(f"❌ Error returning to home: {e}")
    
    def _face_tracking_loop(self):
        """Main face tracking loop"""
        print("🎯 Face tracking loop started")
        
        while self.is_tracking:
            try:
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                if faces:
                    # Get the largest face (closest to camera)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    face_center = (largest_face[0] + largest_face[2] // 2, 
                                 largest_face[1] + largest_face[3] // 2)
                    
                    # Update target tracker
                    self.target_tracker.update_target(face_center)
                    
                    # Execute robot movement if needed
                    if self.robot_controller and self.robot_controller.is_connected:
                        self._execute_robot_movement(frame, largest_face, face_center)
                
                # Add small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in face tracking loop: {e}")
                time.sleep(0.1)
        
        print("🎯 Face tracking loop stopped")
    
    def _execute_robot_movement(self, frame, face_bbox, face_center):
        """Execute robot movement based on face position"""
        try:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            frame_center = (frame_width // 2, frame_height // 2)
            
            # Calculate error (difference between face center and frame center)
            error_x = face_center[0] - frame_center[0]
            error_y = face_center[1] - frame_center[1]
            
            # Only move if error is significant
            threshold = 50  # pixels
            if abs(error_x) > threshold or abs(error_y) > threshold:
                with self.robot_lock:
                    # Calculate movement angles (simple proportional control)
                    pan_angle = -error_x * 0.1  # Negative for correct direction
                    tilt_angle = error_y * 0.1
                    
                    # Limit movement speed
                    max_angle = 5.0
                    pan_angle = max(-max_angle, min(max_angle, pan_angle))
                    tilt_angle = max(-max_angle, min(max_angle, tilt_angle))
                    
                    # Move robot joints
                    if abs(pan_angle) > 1.0:
                        self.robot_controller.move_joint("joint1", 
                                                       self.robot_controller.current_positions["joint1"] + pan_angle,
                                                       speed=50)
                    
                    if abs(tilt_angle) > 1.0:
                        self.robot_controller.move_joint("joint5", 
                                                       self.robot_controller.current_positions["joint5"] + tilt_angle,
                                                       speed=50)
                    
                    print(f"🤖 Robot movement: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
        
        except Exception as e:
            print(f"❌ Error executing robot movement: {e}")
    
    def start(self):
        """Start the focus-controlled robot system"""
        if not FOCUS_AVAILABLE:
            print("❌ Focus detection not available. Cannot start system.")
            return False
        
        if not self.robot_controller:
            print("❌ Robot controller not available. Cannot start system.")
            return False
        
        print("\n🚀 Starting Focus-Controlled Robot System...")
        
        # First, ensure robot is in home position
        self._return_to_home()
        
        # Start focus detection
        try:
            self.focus_detector = start_focus_detector(callback=self._focus_callback)
            print("✅ Focus detection started")
        except Exception as e:
            print(f"❌ Failed to start focus detection: {e}")
            return False
        
        self.is_running = True
        print("✅ System started successfully!")
        print("\n📋 System Status:")
        print("   - Robot: Connected and in home position")
        print("   - Focus Detection: Active")
        print("   - Face Tracking: Waiting for focus...")
        print("\n💡 Instructions:")
        print("   - Focus your attention to activate face tracking")
        print("   - Look away or lose focus to stop tracking")
        print("   - Press Ctrl+C to exit")
        
        return True
    
    def stop(self):
        """Stop the focus-controlled robot system"""
        print("\n🛑 Stopping Focus-Controlled Robot System...")
        
        self.is_running = False
        
        # Stop face tracking
        self._stop_face_tracking()
        
        # Stop focus detection
        if self.focus_detector:
            try:
                stop_focus_detector()
                print("✅ Focus detection stopped")
            except Exception as e:
                print(f"⚠️  Error stopping focus detection: {e}")
        
        # Return robot to home
        self._return_to_home()
        
        # Release camera
        if self.camera:
            self.camera.release()
            print("✅ Camera released")
        
        print("✅ System stopped successfully")
    
    def run(self):
        """Main run loop"""
        if not self.start():
            return
        
        try:
            # Keep the main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⌨️  Keyboard interrupt received")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            self.stop()


def main():
    """Main function"""
    print("🤖 Focus-Controlled Robot Arm System")
    print("=" * 50)
    
    # Check system requirements
    if not FOCUS_AVAILABLE:
        print("❌ Focus detection is required but not available")
        print("   Please ensure focus_true_false.py is properly configured")
        return 1
    
    if not ROBOT_AVAILABLE:
        print("❌ Robot control is required but not available")
        print("   Please ensure robot control modules are properly configured")
        return 1
    
    if not FACE_TRACKING_AVAILABLE:
        print("⚠️  Face tracking is not available")
        print("   The system will still respond to focus but won't track faces")
    
    if not OPENCV_AVAILABLE:
        print("⚠️  OpenCV is not available")
        print("   Face tracking will not work")
    
    # Create and run the focus-controlled robot
    robot_system = FocusControlledRobot()
    robot_system.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
