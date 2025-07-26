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
from typing import Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

# Add mainfiles to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'mainfiles'))


# =============================================================================
# ABSTRACT INTERFACES FOR DECOUPLING
# =============================================================================

class RobotController(ABC):
    """Abstract interface for robot control"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot is connected"""
        pass
    
    @abstractmethod
    def go_home(self) -> bool:
        """Move robot to home position"""
        pass
    
    @abstractmethod
    def move_joint(self, joint_name: str, angle: float, speed: int = 100) -> bool:
        """Move a specific joint"""
        pass
    
    @abstractmethod
    def get_current_positions(self) -> Dict[str, float]:
        """Get current joint positions"""
        pass


class FocusDetector(ABC):
    """Abstract interface for focus detection"""
    
    @abstractmethod
    def start_detection(self, callback: Callable[[bool, float], None]) -> bool:
        """Start focus detection with callback"""
        pass
    
    @abstractmethod
    def stop_detection(self) -> None:
        """Stop focus detection"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if detection is running"""
        pass


class FaceDetector(ABC):
    """Abstract interface for face detection"""
    
    @abstractmethod
    def detect_faces(self, frame) -> list:
        """Detect faces in frame"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if face detector is available"""
        pass


class CameraInterface(ABC):
    """Abstract interface for camera control"""
    
    @abstractmethod
    def open(self) -> bool:
        """Open camera"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close camera"""
        pass
    
    @abstractmethod
    def read(self) -> tuple:
        """Read frame from camera"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        pass


# =============================================================================
# CONCRETE IMPLEMENTATIONS WITH PROPER ERROR HANDLING
# =============================================================================

class SimpleRobotControllerWrapper(RobotController):
    """Wrapper for SimpleRobotController with error handling"""
    
    def __init__(self):
        self.controller = None
        self._available = False
        
        try:
            from mainfiles.simple_robot_control import SimpleRobotController
            self.controller = SimpleRobotController()
            self._available = True
            print("✅ Robot control module imported successfully")
        except ImportError as e:
            print(f"⚠️  Robot control module not available: {e}")
        except Exception as e:
            print(f"❌ Robot control initialization failed: {e}")
    
    def connect(self) -> bool:
        if not self._available or not self.controller:
            return False
        try:
            return self.controller.connect()
        except Exception as e:
            print(f"❌ Robot connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        if self.controller:
            try:
                self.controller.disconnect()
            except Exception as e:
                print(f"⚠️  Robot disconnect error: {e}")
    
    def is_connected(self) -> bool:
        return self.controller and self.controller.is_connected
    
    def go_home(self) -> bool:
        if not self.is_connected():
            return False
        try:
            return self.controller.go_home()
        except Exception as e:
            print(f"❌ Go home failed: {e}")
            return False
    
    def move_joint(self, joint_name: str, angle: float, speed: int = 100) -> bool:
        if not self.is_connected():
            return False
        try:
            return self.controller.move_joint(joint_name, angle, speed)
        except Exception as e:
            print(f"❌ Joint movement failed: {e}")
            return False
    
    def get_current_positions(self) -> Dict[str, float]:
        if not self.is_connected():
            return {}
        try:
            return self.controller.current_positions.copy()
        except Exception as e:
            print(f"❌ Get positions failed: {e}")
            return {}


class FocusDetectorWrapper(FocusDetector):
    """Wrapper for focus detection with error handling"""
    
    def __init__(self):
        self.detector = None
        self._available = False
        
        try:
            from mainfiles.emoRobots.focus_true_false import start_focus_detector, stop_focus_detector
            self._start_func = start_focus_detector
            self._stop_func = stop_focus_detector
            self._available = True
            print("✅ Focus detection module imported successfully")
        except ImportError as e:
            print(f"⚠️  Focus detection module not available: {e}")
        except Exception as e:
            print(f"❌ Focus detection initialization failed: {e}")
    
    def start_detection(self, callback: Callable[[bool, float], None]) -> bool:
        if not self._available:
            return False
        try:
            self.detector = self._start_func(callback=callback)
            return True
        except Exception as e:
            print(f"❌ Focus detection start failed: {e}")
            return False
    
    def stop_detection(self) -> None:
        if self._available and self.detector:
            try:
                self._stop_func()
            except Exception as e:
                print(f"⚠️  Focus detection stop error: {e}")
    
    def is_running(self) -> bool:
        return self.detector is not None


class YOLOFaceDetectorWrapper(FaceDetector):
    """Wrapper for YOLO face detection with error handling"""
    
    def __init__(self):
        self.detector = None
        self._available = False
        
        try:
            from mainfiles.auto_face_tracking import YOLOFaceDetector
            self.detector = YOLOFaceDetector()
            self._available = True
            print("✅ Face detection module imported successfully")
        except ImportError as e:
            print(f"⚠️  Face detection module not available: {e}")
        except Exception as e:
            print(f"❌ Face detection initialization failed: {e}")
    
    def detect_faces(self, frame) -> list:
        if not self._available or not self.detector:
            return []
        try:
            return self.detector.detect_faces(frame)
        except Exception as e:
            print(f"❌ Face detection failed: {e}")
            return []
    
    def is_available(self) -> bool:
        return self._available and self.detector is not None


class OpenCVCameraWrapper(CameraInterface):
    """Wrapper for OpenCV camera with error handling"""
    
    def __init__(self):
        self.camera = None
        self._available = False
        
        try:
            import cv2
            self.cv2 = cv2
            self._available = True
            print("✅ OpenCV module imported successfully")
        except ImportError as e:
            print(f"⚠️  OpenCV module not available: {e}")
        except Exception as e:
            print(f"❌ OpenCV initialization failed: {e}")
    
    def open(self) -> bool:
        if not self._available:
            return False
        try:
            self.camera = self.cv2.VideoCapture(0)
            if self.camera.isOpened():
                # Set camera properties
                self.camera.set(self.cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.camera.set(self.cv2.CAP_PROP_FPS, 30)
                return True
            return False
        except Exception as e:
            print(f"❌ Camera open failed: {e}")
            return False
    
    def close(self) -> None:
        if self.camera:
            try:
                self.camera.release()
            except Exception as e:
                print(f"⚠️  Camera close error: {e}")
    
    def read(self) -> tuple:
        if not self.camera:
            return False, None
        try:
            return self.camera.read()
        except Exception as e:
            print(f"❌ Camera read failed: {e}")
            return False, None
    
    def is_opened(self) -> bool:
        return self.camera and self.camera.isOpened()


# =============================================================================
# MAIN FOCUS-CONTROLLED ROBOT SYSTEM
# =============================================================================

class FocusControlledRobot:
    """
    Main robot controller that integrates focus detection with face tracking
    Uses dependency injection for loose coupling
    """
    
    def __init__(self, 
                 robot_controller: Optional[RobotController] = None,
                 focus_detector: Optional[FocusDetector] = None,
                 face_detector: Optional[FaceDetector] = None,
                 camera: Optional[CameraInterface] = None):
        """
        Initialize with optional dependency injection
        If not provided, will create default implementations
        """
        # Initialize components with dependency injection or defaults
        self.robot_controller = robot_controller or SimpleRobotControllerWrapper()
        self.focus_detector = focus_detector or FocusDetectorWrapper()
        self.face_detector = face_detector or YOLOFaceDetectorWrapper()
        self.camera = camera or OpenCVCameraWrapper()
        
        # State management
        self.is_running = False
        self.is_focused = False
        self.is_tracking = False
        
        # Threading
        self.tracking_thread = None
        
        # Locks for thread safety
        self.state_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all required components"""
        print("\n🔧 Initializing components...")
        
        # Initialize robot controller
        if self.robot_controller.connect():
            print("✅ Robot controller initialized and connected")
        else:
            print("❌ Failed to connect to robot")
        
        # Initialize camera
        if self.camera.open():
            print("✅ Camera initialized")
        else:
            print("❌ Failed to open camera")
        
        # Face detector is initialized in constructor
        if self.face_detector.is_available():
            print("✅ Face detector initialized")
        else:
            print("⚠️  Face detector not available")
    
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
            if not self.is_tracking and self.face_detector.is_available() and self.camera.is_opened():
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
        if self.robot_controller.is_connected():
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
                    
                    # Execute robot movement if needed
                    if self.robot_controller.is_connected():
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
                    
                    # Get current positions
                    current_positions = self.robot_controller.get_current_positions()
                    
                    # Move robot joints
                    if abs(pan_angle) > 1.0 and "joint1" in current_positions:
                        new_angle = current_positions["joint1"] + pan_angle
                        self.robot_controller.move_joint("joint1", new_angle, speed=50)
                    
                    if abs(tilt_angle) > 1.0 and "joint5" in current_positions:
                        new_angle = current_positions["joint5"] + tilt_angle
                        self.robot_controller.move_joint("joint5", new_angle, speed=50)
                    
                    print(f"🤖 Robot movement: Pan={pan_angle:.1f}°, Tilt={tilt_angle:.1f}°")
        
        except Exception as e:
            print(f"❌ Error executing robot movement: {e}")
    
    def start(self):
        """Start the focus-controlled robot system"""
        if not self.robot_controller.is_connected():
            print("❌ Robot controller not available. Cannot start system.")
            return False
        
        print("\n🚀 Starting Focus-Controlled Robot System...")
        
        # First, ensure robot is in home position
        self._return_to_home()
        
        # Start focus detection
        if not self.focus_detector.start_detection(self._focus_callback):
            print("❌ Failed to start focus detection")
            return False
        
        print("✅ Focus detection started")
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
        self.focus_detector.stop_detection()
        print("✅ Focus detection stopped")
        
        # Return robot to home
        self._return_to_home()
        
        # Release camera
        self.camera.close()
        print("✅ Camera released")
        
        # Disconnect robot
        self.robot_controller.disconnect()
        print("✅ Robot disconnected")
        
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
    
    # Create and run the focus-controlled robot
    robot_system = FocusControlledRobot()
    robot_system.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
