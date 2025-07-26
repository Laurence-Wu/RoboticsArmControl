#!/usr/bin/env python3
"""
Ultra-Optimized Single-Threaded Face Tracking with PID Control Pipeline

Key ultra-optimizations:
- Minimal error handling (only essential safety checks)
- Reduced computational overhead
- Optimized frame processing
- Minimal object creation
- Better resource management
- Configurable performance settings
"""

import cv2
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

# Import NumPy if available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Import robot configuration
try:
    from config import SERIAL_CONFIG, MOTOR_CONFIG
    from simple_robot_control import SimpleRobotController
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

# Import optimized PID control components
try:
    from pid_controller_optimized import OptimizedPIDController, PIDConfig, FastPIDController
    PID_AVAILABLE = True
except ImportError:
    PID_AVAILABLE = False

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Import auto face tracking functions and classes
try:
    from auto_face_tracking import Detection, YOLOFaceDetector, TargetTracker, YOLO_AVAILABLE as AFT_YOLO_AVAILABLE, face_detector, target_tracker
    AUTO_FACE_TRACKING_AVAILABLE = True
    print("✅ Auto face tracking module imported successfully")
    YOLO_AVAILABLE = AFT_YOLO_AVAILABLE
except ImportError:
    AUTO_FACE_TRACKING_AVAILABLE = False
    print("⚠️  Auto face tracking module not available")

# Global robot movement state to prevent command conflicts
ROBOT_MOVEMENT_LOCK = False

def set_robot_movement_lock(locked: bool):
    """Set global robot movement lock to prevent command conflicts"""
    global ROBOT_MOVEMENT_LOCK
    ROBOT_MOVEMENT_LOCK = locked

def is_robot_movement_locked() -> bool:
    """Check if robot movement is currently locked"""
    global ROBOT_MOVEMENT_LOCK
    return ROBOT_MOVEMENT_LOCK

def check_and_adjust_arm_joints(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """Ultra-optimized arm joint adjustment with minimal overhead."""
    if not robot_controller or not robot_controller.is_connected:
        return False
    
    # Read current positions for both joints
    current_joint2 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint2"])
    current_joint3 = robot_controller.servo_manager.query_servo_angle(robot_controller.motor_ids["joint3"])
    
    if current_joint2 is None or current_joint3 is None:
        return False
    
    # Check if both joints are within tolerance
    joint2_diff = abs(current_joint2 - joint2_target)
    joint3_diff = abs(current_joint3 - joint3_target)
    
    if joint2_diff <= tolerance and joint3_diff <= tolerance:
        return True
    
    # Adjust joints that need adjustment
    success = True
    if joint2_diff > tolerance:
        success &= robot_controller.move_joint("joint2", joint2_target, speed=speed)
    
    if joint3_diff > tolerance:
        success &= robot_controller.move_joint("joint3", joint3_target, speed=speed)
    
    if success:
        time.sleep(0.5)  # Reduced wait time
    
    return success

def ensure_arm_configuration(robot_controller, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100):
    """Ensure arm configuration with minimal overhead."""
    return check_and_adjust_arm_joints(robot_controller, joint2_target, joint3_target, tolerance, speed)

@dataclass
class MotionData:
    """Ultra-optimized data structure for motion tracking data"""
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
        # Ultra-optimized magnitude calculation
        if NUMPY_AVAILABLE:
            self.magnitude = np.sqrt(self.vector[0]**2 + self.vector[1]**2)
        else:
            self.magnitude = math.sqrt(self.vector[0]**2 + self.vector[1]**2)
        self.timestamp = time.time()

@dataclass
class ProcessedMotion:
    """Ultra-optimized processed motion command for robot using PID control"""
    pan_angle: float  # Joint1 movement in degrees
    tilt_angle: float  # Joint5 movement in degrees
    confidence: float  # Confidence of the movement (0-1)
    data_points: int  # Number of data points used
    pid_output_x: float = 0.0  # PID output for X-axis
    pid_output_y: float = 0.0  # PID output for Y-axis
    timestamp: float = 0.0
    
    def __post_init__(self):
        self.timestamp = time.time()

class UltraOptimizedHaarFaceDetector:
    """Ultra-optimized Haar cascade detector with minimal overhead"""
    
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
        
        # Pre-allocated variables for performance
        self._gray = None
        self._faces = []
    
    def detect_faces(self, frame):
        """Ultra-optimized face detection with pre-allocated variables"""
        if self._gray is None or self._gray.shape != frame.shape[:2]:
            self._gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._gray)
        
        self._faces = self.face_cascade.detectMultiScale(
            self._gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(self.min_face_size, self.min_face_size),
            maxSize=(self.max_face_size, self.max_face_size)
        )
        
        # Filter faces by aspect ratio and size
        filtered_faces = []
        for x, y, w, h in self._faces:
            if self._is_valid_human_face(w, h):
                filtered_faces.append((x, y, w, h, 0.8))
        
        return filtered_faces
    
    def _is_valid_human_face(self, width, height):
        """Ultra-optimized face validation"""
        if width < self.min_face_size or width > self.max_face_size:
            return False
        if height < self.min_face_size or height > self.max_face_size:
            return False
        
        aspect_ratio = width / height
        return self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max

class UltraOptimizedMotionDataCollector:
    """Ultra-optimized motion data collector with minimal overhead"""
    
    def __init__(self, max_data_points=20, pid_config=None, use_fast_controller=True):
        # Further reduced data points for maximum performance
        self.max_data_points = max_data_points
        self.motion_data = deque(maxlen=max_data_points)
        self.last_processing_time = time.time()
        
        # Initialize PID configuration
        self.pid_config = pid_config or PIDConfig()
        self.use_fast_controller = use_fast_controller
        
        # Initialize controller based on performance preference
        if PID_AVAILABLE:
            if use_fast_controller:
                # Use ultra-fast controller for maximum performance
                self.controller = FastPIDController(
                    Kp=self.pid_config.PAN_KP,
                    Kd=self.pid_config.PAN_KD,
                    dead_zone=self.pid_config.DEAD_ZONE * self.pid_config.PIXELS_TO_DEGREES,
                    max_output=self.pid_config.MAX_MOVEMENT
                )
                self.tilt_controller = FastPIDController(
                    Kp=self.pid_config.TILT_KP,
                    Kd=self.pid_config.TILT_KD,
                    dead_zone=self.pid_config.DEAD_ZONE * self.pid_config.PIXELS_TO_DEGREES,
                    max_output=self.pid_config.MAX_MOVEMENT
                )
                self.pid_available = True
                print("✅ Ultra-fast PID controllers initialized")
            else:
                # Use optimized dual-axis controller
                self.controller = OptimizedPIDController(self.pid_config)
                self.pid_available = True
                print("✅ Optimized dual-axis PID controller initialized")
        else:
            self.controller = None
            self.pid_available = False
            print("⚠️  PID/PD controller not available - using fallback")
        
        # Initialize robot connection for direct control
        self.robot = None
        self.robot_connected = False
        if ROBOT_AVAILABLE:
            from simple_robot_control import SimpleRobotController
            self.robot = SimpleRobotController()
            if self.robot.connect():
                self.robot_connected = True
                print("✅ Robot connected for direct control")
            else:
                print("⚠️  Robot connection failed")
        
        # Track current positions
        self.current_positions = {"joint1": 0.0, "joint5": 0.0}
        
        # Performance optimization: pre-compute constants
        self._dead_zone_degrees = self.pid_config.DEAD_ZONE * self.pid_config.PIXELS_TO_DEGREES
        self._max_movement = self.pid_config.MAX_MOVEMENT
        self._pixels_to_degrees = self.pid_config.PIXELS_TO_DEGREES
    
    def add_data_point(self, motion_data: MotionData):
        """Add a new motion data point"""
        self.motion_data.append(motion_data)
    
    def should_process(self):
        """Always process each frame for real-time performance"""
        return True
    
    def process_motion_data(self) -> Optional[ProcessedMotion]:
        """Ultra-optimized motion data processing"""
        if not self.motion_data:
            return None
        
        # Use the latest data point for instantaneous control
        latest = self.motion_data[-1]
        error_x, error_y = latest.vector
        
        if self.pid_available and self.controller:
            return self._process_with_pid(error_x, error_y, latest)
        else:
            return self._process_with_fallback(latest)
    
    def _process_with_pid(self, error_x: float, error_y: float, latest: MotionData) -> ProcessedMotion:
        """Ultra-optimized PID processing"""
        if self.use_fast_controller:
            # Use separate fast controllers for each axis
            pan_output = self.controller.update(error_x)
            tilt_output = self.tilt_controller.update(error_y)
        else:
            # Use unified optimized controller
            pan_output, tilt_output = self.controller.update(error_x, error_y)
        
        # Execute movement directly
        success = self._execute_robot_movement(pan_output, tilt_output)
        
        if success:
            return ProcessedMotion(
                pan_angle=pan_output,
                tilt_angle=tilt_output,
                confidence=latest.confidence,
                data_points=1,
                pid_output_x=pan_output,
                pid_output_y=tilt_output
            )
        else:
            return None
    
    def _process_with_fallback(self, latest: MotionData) -> ProcessedMotion:
        """Ultra-optimized fallback processing"""
        # Simple proportional control
        pan_movement = -latest.vector[0] * 0.1
        tilt_movement = latest.vector[1] * 0.1
        
        # Apply limits
        if NUMPY_AVAILABLE:
            pan_movement = np.clip(pan_movement, -self._max_movement, self._max_movement)
            tilt_movement = np.clip(tilt_movement, -self._max_movement, self._max_movement)
        else:
            pan_movement = max(-self._max_movement, min(self._max_movement, pan_movement))
            tilt_movement = max(-self._max_movement, min(self._max_movement, tilt_movement))
        
        return ProcessedMotion(
            pan_angle=pan_movement,
            tilt_angle=tilt_movement,
            confidence=latest.confidence,
            data_points=1
        )
    
    def _execute_robot_movement(self, pan_movement: float, tilt_movement: float) -> bool:
        """Ultra-optimized robot movement execution"""
        if is_robot_movement_locked():
            return False
        
        # Apply gravity compensation
        gravity_comp_pan = getattr(self.pid_config, 'GRAVITY_COMPENSATION_JOINT1', 0.0)
        gravity_comp_tilt = getattr(self.pid_config, 'GRAVITY_COMPENSATION_JOINT5', 0)
        
        compensated_pan = pan_movement + gravity_comp_pan
        compensated_tilt = tilt_movement + gravity_comp_tilt
        
        if not self.robot_connected:
            # Simulation mode
            self.current_positions["joint1"] += compensated_pan
            self.current_positions["joint5"] += compensated_tilt
            return True
        
        # Calculate new target positions
        new_pan = self.current_positions["joint1"] + compensated_pan
        new_tilt = self.current_positions["joint5"] + compensated_tilt
        
        # Apply joint limits
        if ROBOT_AVAILABLE:
            from config import MOTOR_CONFIG
            pan_min, pan_max = MOTOR_CONFIG.JOINT_LIMITS["joint1"]
            tilt_min, tilt_max = MOTOR_CONFIG.JOINT_LIMITS["joint5"]
            
            if NUMPY_AVAILABLE:
                new_pan = np.clip(new_pan, pan_min, pan_max)
                new_tilt = np.clip(new_tilt, tilt_min, tilt_max)
            else:
                new_pan = max(pan_min, min(pan_max, new_pan))
                new_tilt = max(tilt_min, min(tilt_max, new_tilt))
        
        # Execute movement
        success_pan = self.robot.move_joint("joint1", new_pan, speed=150)
        success_tilt = self.robot.move_joint("joint5", new_tilt, speed=150)
        
        if success_pan and success_tilt:
            self.current_positions["joint1"] = new_pan
            self.current_positions["joint5"] = new_tilt
            return True
        else:
            return False
    
    def get_status(self):
        """Get ultra-optimized collector status"""
        current_time = time.time()
        time_since_last = current_time - self.last_processing_time
        data_count = len(self.motion_data)
        
        pid_status = "ultra-fast" if self.use_fast_controller else "optimized"
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

class UltraOptimizedSingleThreadFaceTracker:
    """Ultra-optimized single-threaded face tracking system"""
    
    def __init__(self, display=True, pid_config=None, use_fast_controller=True, enable_multi_face=True):
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
            print("✅ Using auto_face_tracking Detection function")
            self.use_auto_tracking = True
            self.auto_face_detector = face_detector if 'face_detector' in globals() else None
            self.auto_target_tracker = target_tracker if 'target_tracker' in globals() else None
        elif YOLO_AVAILABLE:
            print("⚠️  Auto tracking unavailable, using basic YOLO fallback")
            self.use_auto_tracking = False
            self.detector = YOLOFaceDetector() if AUTO_FACE_TRACKING_AVAILABLE else None
            if not self.detector:
                print("🔄 Falling back to Haar cascade")
                self.detector = UltraOptimizedHaarFaceDetector(face_config)
        else:
            self.use_auto_tracking = False
            self.detector = UltraOptimizedHaarFaceDetector(face_config)
            print("✅ Using ultra-optimized Haar cascade detector")
        
        # Initialize tracker (only if not using auto_tracking)
        if not self.use_auto_tracking:
            if AUTO_FACE_TRACKING_AVAILABLE:
                self.tracker = TargetTracker(
                    lock_duration=2.0, 
                    movement_threshold=30
                )
                print("✅ Using TargetTracker from auto_face_tracking module")
            else:
                self.tracker = None
                print("⚠️  No tracker available - fallback mode")
        
        # Initialize ultra-optimized motion data collector
        self.collector = UltraOptimizedMotionDataCollector(
            pid_config=self.pid_config,
            use_fast_controller=use_fast_controller
        )
        
        # Camera will be initialized in run() method
        self.cap = None
        
        # Initialize robot controller (if PID not available)
        if not PID_AVAILABLE and ROBOT_AVAILABLE:
            self.robot = SimpleRobotController()
            self.robot_connected = self.robot.connect()
        else:
            self.robot_connected = False
        
        # Control parameters
        self.dead_zone = 15
        self.max_movement = 8
        
        # Current joint positions
        self.current_positions = {
            "joint1": 0,
            "joint5": 0,
        }
        
        # Performance optimization: pre-allocated variables
        self._frame_count = 0
        self._last_status_time = time.time()
        self._status_interval = 5.0  # Status update interval
        
        print("✅ Ultra-Optimized SingleThreadFaceTracker initialized")
    
    def _initialize_camera(self):
        """Initialize camera with ultra-optimized settings"""
        print("🎥 Initializing camera with ultra-optimized settings...")
        
        for camera_id in [0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                # Set ultra-optimized camera parameters
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Further reduced resolution
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
                
                print(f"✅ Camera {camera_id} opened with ultra-optimized settings")
                return True
            else:
                self.cap.release()
        
        print("❌ Cannot open any camera")
        return False
    
    def run(self):
        """Ultra-optimized main tracking loop"""
        if not self._initialize_camera():
            raise RuntimeError("Cannot open any camera")
        
        self.running = True
        self._frame_count = 0
        self._last_status_time = time.time()
        
        if self.use_auto_tracking:
            print("🚀 Starting ultra-optimized face tracking with auto_face_tracking module")
        else:
            print("🚀 Starting ultra-optimized single-thread face tracking")
        
        while self.running:
            current_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Cannot read camera frame")
                break
            
            if self.use_auto_tracking:
                # Use auto_face_tracking Detection function
                target_x, target_y = Detection(frame)
                target_position = (target_x, target_y)
                confidence = 1.0
                
                # Check if robot is moving to start position
                if target_tracker.is_robot_moving():
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
                
            else:
                # Fallback face detection logic
                if hasattr(self, 'detector') and self.detector:
                    faces = self.detector.detect_faces(frame)
                    
                    if hasattr(self, 'tracker') and self.tracker:
                        detected_position = None
                        if faces:
                            biggest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = biggest_face[:4]
                            detected_position = (x + w // 2, y + h // 2)
                        
                        target_position = self.tracker.update_target(detected_position)
                        confidence = 1.0 if detected_position else 0.5
                    else:
                        if faces:
                            biggest_face = max(faces, key=lambda f: f[2] * f[3])
                            x, y, w, h = biggest_face[:4]
                            target_position = (x + w // 2, y + h // 2)
                            confidence = biggest_face[4] if len(biggest_face) > 4 else 0.8
                        else:
                            target_position = (960, 540)
                            confidence = 0.0
                else:
                    target_position = (960, 540)
                    confidence = 0.0
                    faces = []
                
                # Create motion data point
                motion_data = MotionData(
                    face_position=target_position,
                    confidence=confidence
                )
                
                # Add to collector
                self.collector.add_data_point(motion_data)
                
                # Process data
                if self.collector.should_process():
                    processed_motion = self.collector.process_motion_data()
                    
                    if processed_motion and not PID_AVAILABLE:
                        self.execute_movement(processed_motion)
                
                # Display (only for non-auto tracking)
                if self.display:
                    self.draw_tracking_info(frame, faces, target_position, motion_data)
                    cv2.imshow('Ultra-Optimized Face Tracking', frame)
            
            # Handle keyboard input
            if self.display:
                if not self.use_auto_tracking:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self.stop()
                        break
                    elif key == ord('r'):  # Reset
                        if hasattr(self, 'tracker') and self.tracker:
                            self.tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
                        self.collector = UltraOptimizedMotionDataCollector(
                            pid_config=self.pid_config,
                            use_fast_controller=True
                        )
                        print("🔄 System reset")
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self.stop()
                        break
            
            self._frame_count += 1
            
            # Print status every 5 seconds
            if current_time - self._last_status_time >= self._status_interval:
                status = self.collector.get_status()
                tracking_mode = "Auto-Tracking" if self.use_auto_tracking else "Manual"
                pid_status = "Ultra-Fast PID" if status['pid_available'] else "Fallback"
                controller_status = status.get('pid_status', 'unknown')
                fps = self._frame_count / (current_time - self._last_status_time + self._status_interval)
                
                print(f"📊 Frame: {self._frame_count}, FPS: {fps:.1f}, Mode: {tracking_mode}, "
                      f"Data points: {status['data_points']}, "
                      f"Control: {pid_status} ({controller_status})")
                
                self._last_status_time = current_time
                self._frame_count = 0
        
        self.cleanup()
    
    def execute_movement(self, processed_motion: ProcessedMotion):
        """Ultra-optimized robot movement execution"""
        if is_robot_movement_locked():
            return
        
        total_movement = abs(processed_motion.pan_angle) + abs(processed_motion.tilt_angle)
        if total_movement < self.dead_zone * 0.1:
            return
        
        # Apply limits
        if NUMPY_AVAILABLE:
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
        
        if NUMPY_AVAILABLE:
            new_pan = np.clip(new_pan, pan_min, pan_max)
            new_tilt = np.clip(new_tilt, tilt_min, tilt_max)
        else:
            new_pan = max(pan_min, min(pan_max, new_pan))
            new_tilt = max(tilt_min, min(tilt_max, new_tilt))
        
        # Execute movement
        if self.robot_connected:
            self.robot.move_joint("joint1", new_pan, speed=150)
            self.robot.move_joint("joint5", new_tilt, speed=150)
            
            self.current_positions["joint1"] = new_pan
            self.current_positions["joint5"] = new_tilt
        else:
            # Simulation mode
            self.current_positions["joint1"] = new_pan
            self.current_positions["joint5"] = new_tilt
    
    def draw_tracking_info(self, frame, faces, target_position, motion_data):
        """Ultra-optimized tracking information display"""
        # Draw detected faces
        for face in faces:
            x, y, w, h = face[:4]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
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
        
        # Display information (minimal for performance)
        cv2.putText(frame, f'Target: {target_position}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show PID status
        pid_status = "Ultra-Fast PID" if status['pid_available'] else "Fallback Mode"
        robot_status = status.get('robot_status', 'unknown')
        status_text = f'Control: {pid_status} ({robot_status})'
        status_color = (0, 255, 0) if status['pid_available'] else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show current robot positions
        current_pan = status.get('current_pan', 0.0)
        current_tilt = status.get('current_tilt', 0.0)
        cv2.putText(frame, f'Pan: {current_pan:.1f}° Tilt: {current_tilt:.1f}°', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def stop(self):
        """Stop the tracking system"""
        self.running = False
        print("🛑 Ultra-optimized face tracker stopping...")
    
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
        print(f"✅ Ultra-optimized face tracker cleaned up (mode: {tracking_mode})")

def main():
    """Ultra-optimized main function with automatic startup"""
    print("🤖 Ultra-Optimized Single-Thread Face Tracking System")
    print("🚀 Automatic startup with maximum performance optimizations")
    print("=" * 80)
    
    # Check auto_face_tracking availability
    if AUTO_FACE_TRACKING_AVAILABLE:
        print("✅ Auto face tracking module available")
    else:
        print("⚠️  Auto face tracking module not available - Using fallback detection")
    
    # Set ultra-optimized preferences
    enable_display = True
    use_fast_controller = True  # Use fast controller for maximum performance
    enable_multi_face = True
    
    # Load PID configuration if available
    pid_config = None
    if PID_AVAILABLE:
        print("📊 Using ultra-optimized PID/PD configuration")
    
    print(f"📊 Ultra-Optimized Configuration:")
    print(f"  - Processing: Instantaneous (real-time)")
    print(f"  - Display Enabled: {enable_display}")
    print(f"  - Auto Face Tracking: {'Available' if AUTO_FACE_TRACKING_AVAILABLE else 'Not Available'}")
    print(f"  - Control Algorithm: {'Ultra-Fast PID' if use_fast_controller else 'Optimized PID'}")
    print(f"  - Multi-face Tracking: {enable_multi_face}")
    print(f"  - PID/PD Control: {'Available' if PID_AVAILABLE else 'Not Available'}")
    print(f"  - Camera Status: Will initialize with ultra-optimized settings")
    
    print("💡 Press ESC to exit, R to reset system")
    print("=" * 80)

    # Move to home position and ensure joint3 is at correct position
    import move_to_json
    move_to_json.move_to_json_positions("positions/HOME.json", speed=100)
    
    # Ensure arm configuration
    if ROBOT_AVAILABLE:
        from simple_robot_control import SimpleRobotController
        temp_robot = SimpleRobotController()
        if temp_robot.connect():
            print("🔧 Checking and adjusting arm configuration...")
            arm_config_ok = ensure_arm_configuration(temp_robot, joint2_target=-42.4, joint3_target=-132.0, tolerance=2.0, speed=100)
            if arm_config_ok:
                print("✅ Arm configuration properly set for face tracking")
            else:
                print("⚠️  Arm configuration failed - face tracking may be affected")
            temp_robot.disconnect()
        else:
            print("⚠️  Could not connect to robot for arm configuration adjustment")
    
    # Create ultra-optimized tracker
    print("🔧 Creating ultra-optimized face tracker...")
    tracker = UltraOptimizedSingleThreadFaceTracker(
        display=enable_display,
        pid_config=pid_config,
        use_fast_controller=use_fast_controller,
        enable_multi_face=enable_multi_face
    )
    
    print("🎬 Starting ultra-optimized face tracking...")
    
    # Start the tracker
    tracker.run()

if __name__ == "__main__":
    main() 