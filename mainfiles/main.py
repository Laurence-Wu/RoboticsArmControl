#!/usr/bin/env python3
"""
Single-Threaded Face Tracking with PID Control Pipeline (Simplified)

This system uses a single thread with a pipeline approach:
1. Collect motion data for N seconds
2. Process collected data using PID control
3. Execute motor movements based on PID output

Simplified version with reduced error handling for cleaner code.
Integrates auto_face_tracking.py algorithm perfectly.
"""

import cv2
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque
import statistics

# Import NumPy if available
try:
    import numpy as np
except ImportError:
    np = None

# Import robot configuration
try:
    from config import SERIAL_CONFIG, MOTOR_CONFIG
    from simple_robot_control import SimpleRobotController
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

# Import PID control components
try:
    from pid_controller import PIDConfig, DualAxisPIDController, OptimizedDualAxisPDController
    from pid_robot_controller import PIDRobotController, SafetyLimits
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import auto face tracking functions and classes - Perfect Integration
try:
    from auto_face_tracking import Detection, YOLOFaceDetector, TargetTracker, YOLO_AVAILABLE as AFT_YOLO_AVAILABLE, face_detector, target_tracker
    AUTO_FACE_TRACKING_AVAILABLE = True
    print("✅ Auto face tracking module imported successfully with perfect integration")
    # Use the same YOLO availability status as auto_face_tracking
    YOLO_AVAILABLE = AFT_YOLO_AVAILABLE
except ImportError as e:
    AUTO_FACE_TRACKING_AVAILABLE = False
    print(f"⚠️  Auto face tracking module not available: {e}")

# Global robot movement state to prevent command conflicts
ROBOT_MOVEMENT_LOCK = False

def set_robot_movement_lock(locked: bool):
    """Set global robot movement lock to prevent command conflicts"""
    global ROBOT_MOVEMENT_LOCK
    ROBOT_MOVEMENT_LOCK = locked
    if locked:
        print("🔒 Robot movement locked - preventing PID commands")
    else:
        print("🔓 Robot movement unlocked - PID commands resumed")

def is_robot_movement_locked() -> bool:
    """Check if robot movement is currently locked"""
    global ROBOT_MOVEMENT_LOCK
    return ROBOT_MOVEMENT_LOCK

def check_and_adjust_arm_joints(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """
    Check and adjust both joint2 and joint3 to their target positions.
    This function ensures both joints are properly positioned together.
    
    Args:
        robot_controller: SimpleRobotController instance
        joint2_target: Target angle for joint2 (default: -42.4)
        joint3_target: Target angle for joint3 (default: -132.0)
        tolerance: Acceptable deviation from target (default: 2.0 degrees)
        speed: Movement speed (default: 100)
        
    Returns:
        bool: True if both joints are at target angles or were successfully adjusted
    """
    if not robot_controller or not robot_controller.is_connected:
        print("⚠️  Robot controller not available for arm joint adjustment")
        return False
    
    try:
        print("🔧 Checking and adjusting arm joints (joint2 & joint3)...")
        
        # Read current positions for both joints
        current_joint2 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint2"])
        current_joint3 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint3"])
        
        if current_joint2 is None:
            print("⚠️  Could not read joint2 position - servo may not be responding")
            return False
        
        if current_joint3 is None:
            print("⚠️  Could not read joint3 position - servo may not be responding")
            return False
        
        # Check if both joints are within tolerance
        joint2_diff = abs(current_joint2 - joint2_target)
        joint3_diff = abs(current_joint3 - joint3_target)
        
        joint2_ok = joint2_diff <= tolerance
        joint3_ok = joint3_diff <= tolerance
        
        # Report current status
        print(f"📊 Current positions: Joint2={current_joint2:.1f}°, Joint3={current_joint3:.1f}°")
        print(f"📊 Target positions: Joint2={joint2_target}°, Joint3={joint3_target}°")
        
        if joint2_ok and joint3_ok:
            print(f"✅ Both joints are at correct positions (tolerance: ±{tolerance}°)")
            return True
        
        # Adjust joints that need adjustment
        adjustments_made = False
        
        if not joint2_ok:
            print(f"🔄 Adjusting joint2 from {current_joint2:.1f}° to {joint2_target}° (difference: {joint2_diff:.1f}°)")
            success = robot_controller.move_joint("joint2", joint2_target, speed=speed)
            if success:
                print(f"✅ Joint2 successfully adjusted to {joint2_target}°")
                adjustments_made = True
            else:
                print(f"❌ Failed to adjust joint2 to {joint2_target}°")
                return False
        
        if not joint3_ok:
            print(f"🔄 Adjusting joint3 from {current_joint3:.1f}° to {joint3_target}° (difference: {joint3_diff:.1f}°)")
            success = robot_controller.move_joint("joint3", joint3_target, speed=speed)
            if success:
                print(f"✅ Joint3 successfully adjusted to {joint3_target}°")
                adjustments_made = True
            else:
                print(f"❌ Failed to adjust joint3 to {joint3_target}°")
                return False
        
        if adjustments_made:
            # Wait a moment and verify final positions
            import time
            time.sleep(1.0)
            
            final_joint2 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint2"])
            final_joint3 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint3"])
            
            if final_joint2 is not None and final_joint3 is not None:
                final_joint2_diff = abs(final_joint2 - joint2_target)
                final_joint3_diff = abs(final_joint3 - joint3_target)
                
                if final_joint2_diff <= tolerance and final_joint3_diff <= tolerance:
                    print(f"✅ Verification: Both joints confirmed at target positions")
                    print(f"   Joint2: {final_joint2:.1f}°, Joint3: {final_joint3:.1f}°")
                else:
                    print(f"⚠️  Verification: Some joints may not have reached target positions")
                    print(f"   Joint2: {final_joint2:.1f}° (target: {joint2_target}°), Joint3: {final_joint3:.1f}° (target: {joint3_target}°)")
        
        return True
                
    except Exception as e:
        print(f"❌ Error checking/adjusting arm joints: {e}")
        return False

def ensure_arm_configuration(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """
    Ensure both joint2 and joint3 are at their target positions for proper arm configuration.
    This function checks and adjusts both joints to maintain the correct arm setup.
    
    Args:
        robot_controller: SimpleRobotController instance
        joint2_target: Target angle for joint2 (default: -42.4)
        joint3_target: Target angle for joint3 (default: -132.0)
        tolerance: Acceptable deviation from target (default: 2.0 degrees)
        speed: Movement speed (default: 100)
        
    Returns:
        bool: True if both joints are properly positioned
    """
    return check_and_adjust_arm_joints(robot_controller, joint2_target, joint3_target, tolerance, speed)

@dataclass
class MotionData:
    """Data structure for motion tracking data"""
    face_position: Tuple[int, int]  # (x, y) position of detected face
    center_position: Tuple[int, int] = (960, 540)  # Center of frame (1920x1080)
    vector: Tuple[float, float] = (0.0, 0.0)  # Vector from face to center
    magnitude: float = 0.0  # Vector magnitude
    timestamp: float = 0.0  # When this data was collected
    confidence: float = 1.0  # Confidence of detection (0-1)
    
    def __post_init__(self):
        """Calculate vector and magnitude after initialization"""
        self.vector = (
            self.center_position[0] - self.face_position[0],
            self.center_position[1] - self.face_position[1]
        )
        # Calculate magnitude
        if np:
            self.magnitude = np.sqrt(self.vector[0]**2 + self.vector[1]**2)
        else:
            self.magnitude = math.sqrt(self.vector[0]**2 + self.vector[1]**2)
        self.timestamp = time.time()

@dataclass
class ProcessedMotion:
    """Processed motion command for robot using PID control"""
    pan_angle: float  # Joint1 movement in degrees
    tilt_angle: float  # Joint5 movement in degrees
    confidence: float  # Confidence of the movement (0-1)
    data_points: int  # Number of data points used
    pid_output_x: float = 0.0  # PID output for X-axis
    pid_output_y: float = 0.0  # PID output for Y-axis
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()

class HaarFaceDetector:
    """Haar cascade fallback detector with human face filtering"""
    
    def __init__(self, face_config=None):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Face detection thresholds
        self.min_face_size = face_config.get('min_face_size', 50) if face_config else 50
        self.max_face_size = face_config.get('max_face_size', 300) if face_config else 300
        self.min_confidence = face_config.get('min_confidence', 0.7) if face_config else 0.7
        self.aspect_ratio_min = face_config.get('aspect_ratio_min', 0.6) if face_config else 0.6
        self.aspect_ratio_max = face_config.get('aspect_ratio_max', 1.8) if face_config else 1.8
        
        print(f"✅ Haar Face Detector: min_size={self.min_face_size}")
    
    def detect_faces(self, frame):
        """Detect human faces using Haar cascade with filtering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(self.min_face_size, self.min_face_size),
            maxSize=(self.max_face_size, self.max_face_size)
        )
        
        # Filter faces by aspect ratio and size
        filtered_faces = []
        for x, y, w, h in faces:
            if self._is_valid_human_face(w, h):
                # Haar doesn't provide confidence, use fixed high value
                filtered_faces.append((x, y, w, h, 0.8))
        
        return filtered_faces
    
    def _is_valid_human_face(self, width, height):
        """Validate if detected region is likely a human face"""
        # Size validation (already done by detectMultiScale)
        if width < self.min_face_size or width > self.max_face_size:
            return False
        if height < self.min_face_size or height > self.max_face_size:
            return False
        
        # Aspect ratio validation (faces are roughly square to oval)
        aspect_ratio = width / height
        if aspect_ratio < self.aspect_ratio_min or aspect_ratio > self.aspect_ratio_max:
            return False
        
        return True

class MotionDataCollector:
    """Collects and processes motion data over time using PID/PD control"""
    
    def __init__(self, max_data_points=60, pid_config=None, use_optimized_pd=True):
        # No collection duration - process data immediately
        self.max_data_points = max_data_points
        self.motion_data = deque(maxlen=max_data_points)
        self.last_processing_time = time.time()
        
        # Initialize PID configuration
        self.pid_config = pid_config or PIDConfig()
        self.use_optimized_pd = use_optimized_pd  # TianxingWu approach option
        
        # Initialize dual-axis controller (PID or optimized PD)
        if PID_AVAILABLE:
            if self.use_optimized_pd:
                # Use optimized PD controller like TianxingWu approach
                self.controller = OptimizedDualAxisPDController(self.pid_config)
                self.pid_available = True
                print("✅ Optimized dual-axis PD controller initialized (TianxingWu approach)")
            else:
                # Use traditional PID controller
                self.controller = DualAxisPIDController(self.pid_config)
                self.pid_available = True
                print("✅ Traditional dual-axis PID controller initialized")
        else:
            self.controller = None
            self.pid_available = False
            print("⚠️  PID/PD controller not available - using fallback")
        
        # Initialize robot connection for direct control
        self.robot = None
        self.robot_connected = False
        if ROBOT_AVAILABLE:
            try:
                from simple_robot_control import SimpleRobotController
                self.robot = SimpleRobotController()
                if self.robot.connect():
                    self.robot_connected = True
                    print("✅ Robot connected for direct control")
                else:
                    print("⚠️  Robot connection failed")
            except Exception as e:
                print(f"⚠️  Robot initialization failed: {e}")
        
        # Track current positions
        self.current_positions = {"joint1": 0.0, "joint5": 0.0}
    
    def add_data_point(self, motion_data: MotionData):
        """Add a new motion data point"""
        self.motion_data.append(motion_data)
    
    def should_process(self):
        """Process every frame with PID control"""
        return True  # Always process each frame
    
    def process_motion_data(self) -> Optional[ProcessedMotion]:
        """Process collected motion data and return motor commands using PID control"""
        if not self.motion_data:
            return None
        
        # Use all available data (no time filtering)
        recent_data = list(self.motion_data)
        
        if not recent_data:
            return None
        
        if self.pid_available and self.controller:
            # Use instantaneous PID control for processing
            return self._process_with_pid(recent_data)
        else:
            # Fallback to simple weighted averaging
            return self._process_with_fallback(recent_data)
    
    def _process_with_pid(self, recent_data: List[MotionData]) -> ProcessedMotion:
        """Process motion data using instantaneous PID control"""
        if not recent_data:
            return None
        
        # Use the latest data point - instantaneous error correction
        latest = recent_data[-1]
        error_x, error_y = latest.vector  # pixels from center
        
        # Feed error directly into PID controller (direction correction handled inside)
        pan_output, tilt_output = self.controller.update(error_x, error_y)  # degrees
        
        # Apply safety limits (clamp to max movement)
        try:
            pan_output = np.clip(pan_output, -self.pid_config.MAX_MOVEMENT, self.pid_config.MAX_MOVEMENT)
            tilt_output = np.clip(tilt_output, -self.pid_config.MAX_MOVEMENT, self.pid_config.MAX_MOVEMENT)
        except Exception:
            # Fallback clipping
            pan_output = max(-self.pid_config.MAX_MOVEMENT, min(self.pid_config.MAX_MOVEMENT, pan_output))
            tilt_output = max(-self.pid_config.MAX_MOVEMENT, min(self.pid_config.MAX_MOVEMENT, tilt_output))
        
        # Execute movement directly to robot
        success = self._execute_robot_movement(pan_output, tilt_output)
        
        # Update processing time
        self.last_processing_time = time.time()
        
        if success:
            print(f"🎯 PID Movement: Pan={pan_output:.2f}°, Tilt={tilt_output:.2f}° (Error: {error_x:.1f}, {error_y:.1f}px)")
            return ProcessedMotion(
                pan_angle=pan_output,
                tilt_angle=tilt_output,
                confidence=latest.confidence,
                data_points=1,  # Using single latest point
                pid_output_x=pan_output,
                pid_output_y=tilt_output
            )
        else:
            return None
    
    def _process_with_fallback(self, recent_data: List[MotionData]) -> ProcessedMotion:
        """Fallback processing without PID control"""
        # Calculate weighted averages based on confidence and recency
        total_weight = 0
        weighted_pan = 0
        weighted_tilt = 0
        
        current_time = time.time()
        
        for data in recent_data:
            # Weight based on confidence only (no time-based weighting)
            weight = data.confidence
            
            # Convert vector to motor movements
            pan_movement = -data.vector[0] * 0.1  # Scale factor
            tilt_movement = data.vector[1] * 0.1
            
            weighted_pan += pan_movement * weight
            weighted_tilt += tilt_movement * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Calculate final movements
        final_pan = weighted_pan / total_weight
        final_tilt = weighted_tilt / total_weight
        
        # Calculate overall confidence
        avg_confidence = statistics.mean([data.confidence for data in recent_data])
        
        # Update processing time
        self.last_processing_time = current_time
        
        return ProcessedMotion(
            pan_angle=final_pan,
            tilt_angle=final_tilt,
            confidence=avg_confidence,
            data_points=len(recent_data)
        )
    
    def _execute_robot_movement(self, pan_movement: float, tilt_movement: float) -> bool:
        """Execute robot movement with gravity compensation and current positions tracking"""
        # Check if robot movement is locked (e.g., during return-to-start movement)
        if is_robot_movement_locked():
            print("⏸️  Robot movement blocked - robot is executing return-to-start command")
            return False
        
        # Apply gravity compensation
        gravity_comp_pan = getattr(self.pid_config, 'GRAVITY_COMPENSATION_JOINT1', 0.0)
        gravity_comp_tilt = getattr(self.pid_config, 'GRAVITY_COMPENSATION_JOINT5', 0)
        
        # Add gravity compensation to movements
        compensated_pan = pan_movement + gravity_comp_pan
        compensated_tilt = tilt_movement + gravity_comp_tilt
        
        if not self.robot_connected:
            # Simulation mode
            self.current_positions["joint1"] += compensated_pan
            self.current_positions["joint5"] += compensated_tilt
            print(f"🎮 Simulation Movement: Pan={self.current_positions['joint1']:.1f}°, Tilt={self.current_positions['joint5']:.1f}° (gravity compensated)")
            return True
        
        try:
            # Ensure both joint2 and joint3 are at their correct positions for proper arm configuration
            if self.robot and self.robot.is_connected:
                arm_config_ok = ensure_arm_configuration(self.robot, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100)
                if not arm_config_ok:
                    print("⚠️  Arm configuration adjustment failed - movement may be affected")
                    # Continue with movement anyway, but log the issue
            
            # Calculate new target positions with gravity compensation
            new_pan = self.current_positions["joint1"] + compensated_pan
            new_tilt = self.current_positions["joint5"] + compensated_tilt
            
            # Apply joint limits if available
            if ROBOT_AVAILABLE:
                try:
                    from config import MOTOR_CONFIG
                    pan_min, pan_max = MOTOR_CONFIG.JOINT_LIMITS["joint1"]
                    tilt_min, tilt_max = MOTOR_CONFIG.JOINT_LIMITS["joint5"]
                    
                    new_pan = max(pan_min, min(pan_max, new_pan))
                    new_tilt = max(tilt_min, min(tilt_max, new_tilt))
                except Exception:
                    # Use safe defaults
                    new_pan = max(-180, min(180, new_pan))
                    new_tilt = max(-90, min(90, new_tilt))
            
            # Execute movement
            success_pan = self.robot.move_joint("joint1", new_pan, speed=150)
            success_tilt = self.robot.move_joint("joint5", new_tilt, speed=150)
            
            if success_pan and success_tilt:
                # Update current positions
                self.current_positions["joint1"] = new_pan
                self.current_positions["joint5"] = new_tilt
                print(f"🤖 Robot Movement: Pan={new_pan:.1f}°, Tilt={new_tilt:.1f}° (gravity: +{gravity_comp_tilt:.1f}°)")
                return True
            else:
                print(f"⚠️  Robot movement partially failed")
                return False
                
        except Exception as e:
            print(f"❌ Robot movement error: {e}")
            return False
    
    def get_status(self):
        """Get current collector status"""
        current_time = time.time()
        time_since_last = current_time - self.last_processing_time
        data_count = len(self.motion_data)
        
        # Get PID controller status
        pid_status = "active" if self.pid_available else "unavailable"
        robot_status = "connected" if self.robot_connected else "simulation"
        
        return {
            'data_points': data_count,
            'time_since_last_process': time_since_last,
            'ready_to_process': self.should_process(),
            'pid_available': self.pid_available,
            'pid_status': pid_status,
            'robot_status': robot_status,
            'current_pan': self.current_positions["joint1"],
            'current_tilt': self.current_positions["joint5"]
        }

class SingleThreadFaceTracker:
    """Single-threaded face tracking system with perfect auto_face_tracking integration"""
    
    def __init__(self, display=True, pid_config=None, use_optimized_pd=True, enable_multi_face=True):
        self.display = display
        self.running = False
        
        # Initialize PID configuration
        if PID_AVAILABLE and pid_config is None:
            self.pid_config = PIDConfig()
        elif PID_AVAILABLE:
            self.pid_config = pid_config
        else:
            self.pid_config = None
        
        # Get face detection configuration
        face_config = None
        if self.pid_config:
            face_config = {
                'min_face_size': getattr(self.pid_config, 'MIN_FACE_SIZE', 50),
                'max_face_size': getattr(self.pid_config, 'MAX_FACE_SIZE', 300),
                'min_confidence': getattr(self.pid_config, 'MIN_FACE_CONFIDENCE', 0.7),
                'aspect_ratio_min': getattr(self.pid_config, 'FACE_ASPECT_RATIO_MIN', 0.6),
                'aspect_ratio_max': getattr(self.pid_config, 'FACE_ASPECT_RATIO_MAX', 1.8)
            }
        
        # Initialize detector based on availability
        if AUTO_FACE_TRACKING_AVAILABLE:
            print("✅ Using auto_face_tracking Detection function with YOLO and 2s lock")
            self.use_auto_tracking = True
            # Use the imported global detector and tracker from auto_face_tracking
            self.auto_face_detector = face_detector if 'face_detector' in globals() else None
            self.auto_target_tracker = target_tracker if 'target_tracker' in globals() else None
        elif YOLO_AVAILABLE:
            try:
                print("⚠️  Auto tracking unavailable, attempting basic YOLO fallback")
                self.use_auto_tracking = False
                # Create our own YOLO detector as fallback
                self.detector = YOLOFaceDetector() if AUTO_FACE_TRACKING_AVAILABLE else None
                if not self.detector:
                    raise Exception("YOLO detector creation failed")
            except Exception as e:
                print(f"⚠️  YOLO initialization failed: {e}")
                print("🔄 Falling back to Haar cascade")
                self.use_auto_tracking = False
                self.detector = HaarFaceDetector(face_config)
        else:
            self.use_auto_tracking = False
            self.detector = HaarFaceDetector(face_config)
            print("✅ Using Haar cascade detector with human face filtering")
        
        # Initialize tracker (only if not using auto_tracking)
        if not self.use_auto_tracking:
            if AUTO_FACE_TRACKING_AVAILABLE:
                # Use TargetTracker from auto_face_tracking module
                self.tracker = TargetTracker(
                    lock_duration=2.0, 
                    movement_threshold=30
                )
                print("✅ Using TargetTracker from auto_face_tracking module")
            else:
                # Would need to implement a basic tracker here if needed
                self.tracker = None
                print("⚠️  No tracker available - fallback mode")
        
        # Initialize motion data collector
        self.collector = MotionDataCollector(
            pid_config=self.pid_config,
            use_optimized_pd=use_optimized_pd  # Use TianxingWu PD approach by default
        )
        
        # Camera will be initialized in run() method - NOT HERE
        self.cap = None
        
        # Initialize robot controller (if PID not available)
        if not PID_AVAILABLE and ROBOT_AVAILABLE:
            self.robot = SimpleRobotController()
            self.robot_connected = self.robot.connect()
        else:
            self.robot_connected = False
        
        # Control parameters
        self.dead_zone = 15  # Minimum magnitude to trigger movement
        self.max_movement = 8  # Maximum degrees per movement
        
        # Current joint positions
        self.current_positions = {
            "joint1": 0,  # Base rotation (pan)
            "joint5": 0,  # Tilt rotation (tilt)
        }
        
        print("✅ SingleThreadFaceTracker initialized (camera NOT started yet)")
    
    def _initialize_camera(self):
        """Initialize camera - called only when run() is called"""
        print("🎥 Initializing camera...")
        
        # Initialize camera
        for camera_id in [0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                print(f"✅ Camera {camera_id} opened successfully")
                return True
            else:
                self.cap.release()
        
        print("❌ Cannot open any camera")
        return False
    
    def run(self):
        """Main tracking loop - camera starts here, not in __init__"""
        # Initialize camera only when run() is called
        if not self._initialize_camera():
            raise RuntimeError("Cannot open any camera")
        
        self.running = True
        frame_count = 0
        last_status_time = time.time()
        
        if self.use_auto_tracking:
            print("🚀 Starting face tracking with auto_face_tracking module (YOLO + 2s lock)")
            print("💡 Using auto_face_tracking.Detection() function directly")
        else:
            print("🚀 Starting single-thread face tracking with fallback detection")
        
        try:
            while self.running:
                current_time = time.time()
                
                # 1. COLLECT PHASE: Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Cannot read camera frame")
                    break
                
                if self.use_auto_tracking:
                    # Use auto_face_tracking Detection function directly
                    # This handles all detection, tracking, and visualization internally
                    try:
                        target_x, target_y = Detection(frame)
                        target_position = (target_x, target_y)
                        confidence = 1.0  # auto_face_tracking doesn't return confidence
                        
                        # Check if robot is moving to start position
                        if target_tracker.is_robot_moving():
                            print("⏸️  PID control paused - robot moving to start position")
                            # Skip PID processing during movement
                            continue
                        
                        # Create motion data point
                        motion_data = MotionData(
                            face_position=target_position,
                            confidence=confidence
                        )
                        
                        # Add to collector and process
                        self.collector.add_data_point(motion_data)
                        
                        if self.collector.should_process():
                            processed_motion = self.collector.process_motion_data()
                            # Note: Movement is handled directly by collector for PID/PD control
                        
                        # Note: auto_face_tracking.Detection() handles all visualization
                        # including the cv2.imshow() call, so we don't need to do it again
                        
                    except Exception as e:
                        print(f"⚠️  Auto face tracking error: {e}")
                        target_position = (960, 540)  # Default center (1920x1080)
                        
                else:
                    # Original face detection logic (fallback)
                    if hasattr(self, 'detector') and self.detector:
                        faces = self.detector.detect_faces(frame)
                        
                        if hasattr(self, 'tracker') and self.tracker:
                            # Use TargetTracker if available
                            detected_position = None
                            if faces:
                                # Find largest face
                                biggest_face = max(faces, key=lambda f: f[2] * f[3])
                                x, y, w, h = biggest_face[:4]
                                detected_position = (x + w // 2, y + h // 2)
                            
                            target_position = self.tracker.update_target(detected_position)
                            confidence = 1.0 if detected_position else 0.5
                        else:
                            # Basic fallback without tracker
                            if faces:
                                biggest_face = max(faces, key=lambda f: f[2] * f[3])
                                x, y, w, h = biggest_face[:4]
                                target_position = (x + w // 2, y + h // 2)
                                confidence = biggest_face[4] if len(biggest_face) > 4 else 0.8
                            else:
                                target_position = (960, 540)  # 1920x1080
                                confidence = 0.0
                    else:
                        target_position = (960, 540)  # 1920x1080
                        confidence = 0.0
                        faces = []
                    
                    # Create motion data point
                    motion_data = MotionData(
                        face_position=target_position,
                        confidence=confidence
                    )
                    
                    # Add to collector
                    self.collector.add_data_point(motion_data)
                    
                    # 2. PROCESS PHASE: Check if ready to process collected data
                    if self.collector.should_process():
                        processed_motion = self.collector.process_motion_data()
                        
                        if processed_motion and not PID_AVAILABLE:
                            # 3. EXECUTE PHASE: Move robot based on processed data (fallback mode)
                            self.execute_movement(processed_motion)
                    
                    # 4. DISPLAY PHASE: Show visual feedback (only for non-auto tracking)
                    if self.display:
                        self.draw_tracking_info(frame, faces, target_position, motion_data)
                        cv2.imshow('Single-Thread Face Tracking with PID', frame)
                
                # Handle keyboard input
                if self.display:
                    try:
                        if not self.use_auto_tracking:
                            # Handle keys manually if not using auto_face_tracking
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC
                                self.stop()
                                break
                            elif key == ord('r'):  # Reset
                                if hasattr(self, 'tracker') and self.tracker:
                                    self.tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
                                self.collector = MotionDataCollector(
                                    pid_config=self.pid_config,
                                    use_optimized_pd=True
                                )
                                print("🔄 System reset")
                        else:
                            # For auto_face_tracking mode, just check for ESC to exit
                            # (auto_face_tracking handles its own key events including 'r' for reset)
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC
                                self.stop()
                                break
                    except:
                        pass
                
                frame_count += 1
                
                # Print status every 5 seconds
                if current_time - last_status_time >= 5.0:
                    status = self.collector.get_status()
                    tracking_mode = "Auto-Tracking" if self.use_auto_tracking else "Manual"
                    pid_status = "PID" if status['pid_available'] else "Fallback"
                    pid_controller_status = status.get('pid_status', 'unknown')
                    print(f"📊 Frame: {frame_count}, Mode: {tracking_mode}, "
                          f"Data points: {status['data_points']}, "
                          f"Control: {pid_status} ({pid_controller_status})")
                    last_status_time = current_time
            
        except KeyboardInterrupt:
            print("\n👋 User pressed Ctrl+C to exit")
        except Exception as e:
            print(f"❌ Error in tracking loop: {e}")
        finally:
            self.cleanup()
    
    def execute_movement(self, processed_motion: ProcessedMotion):
        """Execute robot movement based on processed motion data (fallback mode)"""
        # Check if robot movement is locked (e.g., during return-to-start movement)
        if is_robot_movement_locked():
            print("⏸️  Robot movement blocked - robot is executing return-to-start command")
            return
        
        # Check if movement is significant enough
        total_movement = abs(processed_motion.pan_angle) + abs(processed_motion.tilt_angle)
        if total_movement < self.dead_zone * 0.1:  # Convert dead_zone to angle equivalent
            return
        
        # Limit movement magnitude
        if np:
            pan_movement = np.clip(processed_motion.pan_angle, -self.max_movement, self.max_movement)
            tilt_movement = np.clip(processed_motion.tilt_angle, -self.max_movement, self.max_movement)
        else:
            pan_movement = max(-self.max_movement, min(self.max_movement, processed_motion.pan_angle))
            tilt_movement = max(-self.max_movement, min(self.max_movement, processed_motion.tilt_angle))
        
        # Calculate new positions
        new_pan = self.current_positions["joint1"] + pan_movement
        new_tilt = self.current_positions["joint5"] + tilt_movement
        
        # Apply joint limits
        if ROBOT_AVAILABLE:
            pan_min, pan_max = MOTOR_CONFIG.JOINT_LIMITS["joint1"]
            tilt_min, tilt_max = MOTOR_CONFIG.JOINT_LIMITS["joint5"]
        else:
            pan_min, pan_max = (-180, 180)
            tilt_min, tilt_max = (-90, 90)
        
        if np:
            new_pan = np.clip(new_pan, pan_min, pan_max)
            new_tilt = np.clip(new_tilt, tilt_min, tilt_max)
        else:
            new_pan = max(pan_min, min(pan_max, new_pan))
            new_tilt = max(tilt_min, min(tilt_max, new_tilt))
        
        # Execute movement
        if self.robot_connected:
            # Ensure both joint2 and joint3 are at their correct positions for proper arm configuration
            arm_config_ok = ensure_arm_configuration(self.robot, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100)
            if not arm_config_ok:
                print("⚠️  Arm configuration adjustment failed - movement may be affected")
                # Continue with movement anyway, but log the issue
            
            self.robot.move_joint("joint1", new_pan, speed=150)
            self.robot.move_joint("joint5", new_tilt, speed=150)
            
            self.current_positions["joint1"] = new_pan
            self.current_positions["joint5"] = new_tilt
            
            print(f"🤖 Robot moved (Fallback) - Pan: {new_pan:.1f}°, Tilt: {new_tilt:.1f}° "
                  f"(Confidence: {processed_motion.confidence:.2f}, "
                  f"Data points: {processed_motion.data_points})")
        else:
            # Simulation mode
            self.current_positions["joint1"] = new_pan
            self.current_positions["joint5"] = new_tilt
            
            print(f"🎮 Simulation (Fallback) - Pan: {new_pan:.1f}°, Tilt: {new_tilt:.1f}° "
                  f"(Confidence: {processed_motion.confidence:.2f}, "
                  f"Data points: {processed_motion.data_points})")
    
    def draw_tracking_info(self, frame, faces, target_position, motion_data):
        """Draw tracking information on frame"""
        # Draw detected faces
        for face in faces:
            x, y, w, h = face[:4]  # Ignore confidence for drawing
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {face[4]:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw target position (blue dot)
        cv2.circle(frame, target_position, 10, (255, 0, 0), -1)
        
        # Draw center point (yellow cross)
        center = motion_data.center_position
        cv2.line(frame, (center[0]-10, center[1]), (center[0]+10, center[1]), (0, 255, 255), 2)
        cv2.line(frame, (center[0], center[1]-10), (center[0], center[1]+10), (0, 255, 255), 2)
        
        # Draw vector arrow from face to center
        if motion_data.magnitude > 5:
            cv2.arrowedLine(frame, target_position, center, (0, 0, 255), 2, tipLength=0.1)
        
        # Get collector status
        status = self.collector.get_status()
        
        # Display information
        cv2.putText(frame, f'Target: {target_position}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Vector: ({motion_data.vector[0]:.1f}, {motion_data.vector[1]:.1f})', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Confidence: {motion_data.confidence:.2f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Data Points: {status["data_points"]}', (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show PID status
        pid_status = "Instantaneous PID" if status['pid_available'] else "Fallback Mode"
        robot_status = status.get('robot_status', 'unknown')
        status_text = f'Control: {pid_status} ({robot_status})'
        status_color = (0, 255, 0) if status['pid_available'] else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show current robot positions
        current_pan = status.get('current_pan', 0.0)
        current_tilt = status.get('current_tilt', 0.0)
        cv2.putText(frame, f'Pan: {current_pan:.1f}° Tilt: {current_tilt:.1f}°', (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show gravity compensation status
        gravity_comp = getattr(self.collector.pid_config, 'GRAVITY_COMPENSATION_JOINT5', 2.5)
        cv2.putText(frame, f'Gravity Comp: {gravity_comp:.1f}°', (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw processing status
        bar_width = 200
        bar_height = 10
        bar_x = 10
        bar_y = 240
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        if status["ready_to_process"]:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
        cv2.putText(frame, 'Processing Status', (bar_x, bar_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def stop(self):
        """Stop the tracking system"""
        self.running = False
        print("🛑 Single-thread face tracker stopping...")
    
    def cleanup(self):
        """Clean up resources"""
        if self.display:
            cv2.destroyAllWindows()
        
        if self.cap:
            self.cap.release()
            print("✅ Camera released")
        
        # Clean up robot controller if available
        if hasattr(self, 'collector') and self.collector.robot_connected:
            self.collector.robot.disconnect()
        
        # Clean up fallback robot controller
        if hasattr(self, 'robot_connected') and self.robot_connected and hasattr(self, 'robot'):
            self.robot.disconnect()
        
        tracking_mode = "auto_face_tracking" if self.use_auto_tracking else "manual"
        print(f"✅ Face tracker cleaned up (mode: {tracking_mode})")

def main():
    """Main function - automatic startup with default settings"""
    print("🤖 Single-Thread Face Tracking with Perfect Auto Face Tracking Integration")
    print("🚀 Automatic startup with default settings")
    print("=" * 80)
    
    # Check auto_face_tracking availability
    if AUTO_FACE_TRACKING_AVAILABLE:
        print("✅ Auto face tracking module available - Enhanced YOLO detection with 2s lock")
        print("📺 Will use auto_face_tracking.Detection() function with built-in visualization")
    else:
        print("⚠️  Auto face tracking module not available - Using fallback detection")
    
    # Set default preferences (no user interaction)
    enable_display = True  # Default: enable video display
    use_optimized_pd = True  # Default: use optimized PD control
    enable_multi_face = True  # Default: enable multi-face tracking
    
    # Load PID configuration if available (use defaults)
    pid_config = None
    if PID_AVAILABLE:
        print("📊 Using default PID/PD configuration")
    
    print(f"📊 Configuration (Automatic):")
    print(f"  - Processing: Immediate (no collection duration)")
    print(f"  - Display Enabled: {enable_display}")
    print(f"  - Auto Face Tracking: {'Available' if AUTO_FACE_TRACKING_AVAILABLE else 'Not Available'}")
    print(f"  - Control Algorithm: {'Optimized PD (TianxingWu)' if use_optimized_pd else 'Traditional PID'}")
    print(f"  - Multi-face Tracking: {enable_multi_face}")
    print(f"  - PID/PD Control: {'Available' if PID_AVAILABLE else 'Not Available'}")
    print(f"  - Camera Status: Will initialize automatically")
    
    print("💡 Press ESC to exit, R to reset system")
    print("=" * 80)

    # Move to home position and ensure joint3 is at correct position
    try:
        import move_to_json
        move_to_json.move_to_json_positions("positions/HOME.json", speed=100)
        
        # Ensure both joint2 and joint3 are at their correct positions for proper arm configuration
        if ROBOT_AVAILABLE:
            try:
                from simple_robot_control import SimpleRobotController
                temp_robot = SimpleRobotController()
                if temp_robot.connect():
                    print("🔧 Checking and adjusting arm configuration (joint2: -42.4°, joint3: -132°)...")
                    arm_config_ok = ensure_arm_configuration(temp_robot, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100)
                    if arm_config_ok:
                        print("✅ Arm configuration properly set for face tracking")
                    else:
                        print("⚠️  Arm configuration failed - face tracking may be affected")
                    temp_robot.disconnect()
                else:
                    print("⚠️  Could not connect to robot for arm configuration adjustment")
            except Exception as e:
                print(f"⚠️  Could not adjust arm configuration: {e}")
    except Exception as e:
        print(f"⚠️  Could not move to home position: {e}")
    
    # Create tracker with default configuration
    print("🔧 Creating face tracker with default settings...")
    tracker = SingleThreadFaceTracker(
        display=enable_display,
        pid_config=pid_config,
        use_optimized_pd=use_optimized_pd,
        enable_multi_face=enable_multi_face
    )
    
    print("🎬 Starting face tracking automatically...")
    
    # Start the tracker immediately (camera will start here)
    tracker.run()

if __name__ == "__main__":
    main() 