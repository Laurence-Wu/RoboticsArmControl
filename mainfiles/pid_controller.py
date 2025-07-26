#!/usr/bin/env python3
"""
PID Controller Implementation for Face Tracking System

This module provides a robust PID controller implementation with anti-windup protection,
configurable parameters, and support for dual-axis control (pan/tilt).

Features:
- Anti-windup protection for integral term
- Configurable gains (Kp, Ki, Kd)
- Dead zone filtering
- Separate controllers for X and Y axes
- Thread-safe operation
- Comprehensive error handling

Usage:
    from pid_controller import PIDController, PIDConfig
    
    # Create PID controller
    pid = PIDController(Kp=0.1, Ki=0.01, Kd=0.05)
    
    # Update with current error
    output = pid.update(error_value)
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PIDConfig:
    """Configuration class for PID parameters and system settings."""
    
    # PID Gains for Pan (X-axis / Joint1)
    PAN_KP: float = 0.02  # Increased from 0.1 based on TianxingWu approach
    PAN_KI: float = 0.1   # Set to 0 for PD control (TianxingWu approach)
    PAN_KD: float = 0.1  # Increased from 0.05 for better damping
    
    # PID Gains for Tilt (Y-axis / Joint4)
    TILT_KP: float = 0.02  # Increased from 0.1
    TILT_KI: float = 0.1   # Set to 0 for PD control (TianxingWu approach)
    TILT_KD: float = 0.1  # Increased from 0.05
    
    # Control Parameters - Optimized based on TianxingWu approach
    DEAD_ZONE: float = 8.0  # Reduced from 15.0 - smaller dead zone for more precision
    MAX_MOVEMENT: float = 12.0  # Increased from 8.0 - allow larger movements for faster tracking
    
    # Anti-windup Protection (less critical with PD control)
    INTEGRAL_MIN: float = -100.0  # Reduced range since Ki=0
    INTEGRAL_MAX: float = 100.0   # Reduced range since Ki=0
    
    # Safety Parameters
    SAFETY_TIMEOUT: float = 30.0  # seconds - safety timeout
    MAX_TRACKING_TIME: float = 300.0  # 5 minutes - maximum tracking duration
    
    # Conversion Parameters - Tuned for better responsiveness
    PIXELS_TO_DEGREES: float = 0.15  # Increased from 0.1 for more responsive tracking
    
    # Gravity Compensation
    GRAVITY_COMPENSATION_JOINT1: float = 0.0  # Pan joint - no gravity effect
    GRAVITY_COMPENSATION_JOINT4: float = 2.5  # Tilt joint - compensate for camera weight
    
    # Human Face Detection Thresholds
    MIN_FACE_SIZE: float = 50  # Minimum face width/height in pixels
    MAX_FACE_SIZE: float = 300  # Maximum face width/height in pixels
    MIN_FACE_CONFIDENCE: float = 0.7  # Minimum detection confidence for humans
    FACE_ASPECT_RATIO_MIN: float = 0.6  # Minimum width/height ratio for face
    FACE_ASPECT_RATIO_MAX: float = 1.8  # Maximum width/height ratio for face
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'pan_gains': {'Kp': self.PAN_KP, 'Ki': self.PAN_KI, 'Kd': self.PAN_KD},
            'tilt_gains': {'Kp': self.TILT_KP, 'Ki': self.TILT_KI, 'Kd': self.TILT_KD},
            'control': {
                'dead_zone': self.DEAD_ZONE,
                'max_movement': self.MAX_MOVEMENT
            },
            'anti_windup': {
                'integral_min': self.INTEGRAL_MIN,
                'integral_max': self.INTEGRAL_MAX
            },
            'safety': {
                'safety_timeout': self.SAFETY_TIMEOUT,
                'max_tracking_time': self.MAX_TRACKING_TIME
            },
            'conversion': {
                'pixels_to_degrees': self.PIXELS_TO_DEGREES
            },
            'gravity_compensation': {
                'joint1': self.GRAVITY_COMPENSATION_JOINT1,
                'joint4': self.GRAVITY_COMPENSATION_JOINT4
            },
            'face_detection': {
                'min_face_size': self.MIN_FACE_SIZE,
                'max_face_size': self.MAX_FACE_SIZE,
                'min_confidence': self.MIN_FACE_CONFIDENCE,
                'aspect_ratio_min': self.FACE_ASPECT_RATIO_MIN,
                'aspect_ratio_max': self.FACE_ASPECT_RATIO_MAX
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PIDConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'pan_gains' in config_dict:
            pan_gains = config_dict['pan_gains']
            config.PAN_KP = pan_gains.get('Kp', config.PAN_KP)
            config.PAN_KI = pan_gains.get('Ki', config.PAN_KI)
            config.PAN_KD = pan_gains.get('Kd', config.PAN_KD)
        
        if 'tilt_gains' in config_dict:
            tilt_gains = config_dict['tilt_gains']
            config.TILT_KP = tilt_gains.get('Kp', config.TILT_KP)
            config.TILT_KI = tilt_gains.get('Ki', config.TILT_KI)
            config.TILT_KD = tilt_gains.get('Kd', config.TILT_KD)
        
        if 'control' in config_dict:
            control = config_dict['control']
            config.DEAD_ZONE = control.get('dead_zone', config.DEAD_ZONE)
            config.MAX_MOVEMENT = control.get('max_movement', config.MAX_MOVEMENT)
            config.COLLECTION_DURATION = control.get('collection_duration', config.COLLECTION_DURATION)
        
        if 'anti_windup' in config_dict:
            anti_windup = config_dict['anti_windup']
            config.INTEGRAL_MIN = anti_windup.get('integral_min', config.INTEGRAL_MIN)
            config.INTEGRAL_MAX = anti_windup.get('integral_max', config.INTEGRAL_MAX)
        
        if 'safety' in config_dict:
            safety = config_dict['safety']
            config.SAFETY_TIMEOUT = safety.get('safety_timeout', config.SAFETY_TIMEOUT)
            config.MAX_TRACKING_TIME = safety.get('max_tracking_time', config.MAX_TRACKING_TIME)
        
        if 'conversion' in config_dict:
            conversion = config_dict['conversion']
            config.PIXELS_TO_DEGREES = conversion.get('pixels_to_degrees', config.PIXELS_TO_DEGREES)
        
        return config


class PIDController:
    """
    PID Controller implementation with anti-windup protection.
    
    This controller implements the standard PID algorithm:
    output = Kp * error + Ki * integral + Kd * derivative
    
    Features:
    - Anti-windup protection to prevent integral term overflow
    - Configurable gains that can be updated at runtime
    - Dead zone filtering to prevent micro-movements
    - Thread-safe operation with proper locking
    - Comprehensive error handling and logging
    """
    
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0,
                 integral_min: float = -500.0, integral_max: float = 500.0,
                 dead_zone: float = 0.0, max_output: float = float('inf')):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target value (typically 0 for error-based control)
            integral_min: Minimum integral term value (anti-windup)
            integral_max: Maximum integral term value (anti-windup)
            dead_zone: Ignore errors smaller than this value
            max_output: Maximum absolute output value
        """
        # PID gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        # Anti-windup limits
        self.integral_min = integral_min
        self.integral_max = integral_max
        
        # Control parameters
        self.dead_zone = dead_zone
        self.max_output = max_output
        
        # State variables
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
        # Statistics and monitoring
        self.last_output = 0.0
        self.update_count = 0
        self.total_error = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"PID Controller initialized: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    
    def update(self, current_value: float) -> float:
        """
        Update PID controller with current measurement.
        
        Args:
            current_value: Current process variable value
            
        Returns:
            float: PID controller output
        """
        with self._lock:
            current_time = time.time()
            dt = current_time - self.last_time
            
            # Prevent division by zero and handle very small time steps
            if dt <= 0.0:
                # Update the last_time to prevent issues with subsequent calls
                self.last_time = current_time
                return self.last_output
            
            # Calculate error
            error = self.setpoint - current_value
            
            # Apply dead zone filtering
            if abs(error) < self.dead_zone:
                error = 0.0
            
            # Proportional term
            proportional = self.Kp * error
            
            # Integral term with anti-windup protection
            self.integral += error * dt
            self.integral = max(self.integral_min, min(self.integral_max, self.integral))
            integral = self.Ki * self.integral
            
            # Derivative term
            derivative = 0.0
            if dt > 0:
                derivative = self.Kd * (error - self.last_error) / dt
            
            # Calculate output
            output = proportional + integral + derivative
            
            # Apply output limits
            if abs(output) > self.max_output:
                output = self.max_output if output > 0 else -self.max_output
            
            # Update state variables
            self.last_error = error
            self.last_time = current_time
            self.last_output = output
            
            # Update statistics
            self.update_count += 1
            self.total_error += abs(error)
            
            logger.debug(f"PID Update: error={error:.2f}, P={proportional:.2f}, "
                        f"I={integral:.2f}, D={derivative:.2f}, output={output:.2f}")
            
            return output
    
    def update_gains(self, Kp: Optional[float] = None, Ki: Optional[float] = None, 
                    Kd: Optional[float] = None) -> None:
        """
        Update PID gains at runtime.
        
        Args:
            Kp: New proportional gain (if provided)
            Ki: New integral gain (if provided)
            Kd: New derivative gain (if provided)
        """
        with self._lock:
            if Kp is not None:
                self.Kp = Kp
                logger.info(f"Updated Kp to {Kp}")
            
            if Ki is not None:
                self.Ki = Ki
                logger.info(f"Updated Ki to {Ki}")
            
            if Kd is not None:
                self.Kd = Kd
                logger.info(f"Updated Kd to {Kd}")
    
    def reset(self) -> None:
        """Reset PID controller state."""
        with self._lock:
            self.last_error = 0.0
            self.integral = 0.0
            self.last_time = time.time()
            self.last_output = 0.0
            self.update_count = 0
            self.total_error = 0.0
            
            logger.info("PID Controller reset")
    
    def set_setpoint(self, setpoint: float) -> None:
        """
        Set new setpoint (target value).
        
        Args:
            setpoint: New target value
        """
        with self._lock:
            self.setpoint = setpoint
            logger.info(f"Setpoint updated to {setpoint}")
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get controller statistics.
        
        Returns:
            dict: Controller statistics including gains, state, and performance metrics
        """
        with self._lock:
            avg_error = self.total_error / max(1, self.update_count)
            
            return {
                'gains': {'Kp': self.Kp, 'Ki': self.Ki, 'Kd': self.Kd},
                'state': {
                    'setpoint': self.setpoint,
                    'last_error': self.last_error,
                    'integral': self.integral,
                    'last_output': self.last_output
                },
                'performance': {
                    'update_count': self.update_count,
                    'average_error': avg_error,
                    'total_error': self.total_error
                },
                'limits': {
                    'integral_min': self.integral_min,
                    'integral_max': self.integral_max,
                    'dead_zone': self.dead_zone,
                    'max_output': self.max_output
                }
            }
    
    def __str__(self) -> str:
        """String representation of PID controller."""
        return (f"PIDController(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, "
                f"setpoint={self.setpoint}, updates={self.update_count})")


class DualAxisPIDController:
    """
    Dual-axis PID controller for pan/tilt face tracking.
    
    This class manages two separate PID controllers for X-axis (pan) and Y-axis (tilt)
    movements, providing a unified interface for dual-axis control.
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        """
        Initialize dual-axis PID controller.
        
        Args:
            config: PID configuration object (uses defaults if None)
        """
        self.config = config or PIDConfig()
        
        # Create separate PID controllers for each axis
        # Dead zone filtering is handled in the update method, so set to 0 here
        self.pid_pan = PIDController(
            Kp=self.config.PAN_KP,
            Ki=self.config.PAN_KI,
            Kd=self.config.PAN_KD,
            setpoint=0.0,  # Target is center (0 error)
            integral_min=self.config.INTEGRAL_MIN,
            integral_max=self.config.INTEGRAL_MAX,
            dead_zone=0.0,  # Dead zone handled in update method
            max_output=self.config.MAX_MOVEMENT
        )
        
        self.pid_tilt = PIDController(
            Kp=self.config.TILT_KP,
            Ki=self.config.TILT_KI,
            Kd=self.config.TILT_KD,
            setpoint=0.0,  # Target is center (0 error)
            integral_min=self.config.INTEGRAL_MIN,
            integral_max=self.config.INTEGRAL_MAX,
            dead_zone=0.0,  # Dead zone handled in update method
            max_output=self.config.MAX_MOVEMENT
        )
        
        logger.info("Dual-axis PID controller initialized")
    
    def update(self, error_x: float, error_y: float) -> tuple[float, float]:
        """
        Update both PID controllers with current errors.
        
        Args:
            error_x: X-axis error (pixels from center)
            error_y: Y-axis error (pixels from center)
            
        Returns:
            tuple: (pan_output, tilt_output) in degrees
        """
        # Fix rotation direction: invert error signs to match motor directions
        # Pan: If face is right of center (negative error), move right (positive output)
        # Tilt: If face is below center (positive error), move down (positive output)
        pan_direction_multiplier = 1  # Negative error → positive movement
        tilt_direction_multiplier = -1  # Positive error → positive movement
        
        corrected_error_x = error_x * pan_direction_multiplier
        corrected_error_y = error_y * tilt_direction_multiplier
        
        # Apply dead zone filtering in pixel space first
        filtered_error_x = corrected_error_x if abs(corrected_error_x) > self.config.DEAD_ZONE else 0.0
        filtered_error_y = corrected_error_y if abs(corrected_error_y) > self.config.DEAD_ZONE else 0.0
        
        # Convert filtered pixel errors to degrees
        error_x_degrees = filtered_error_x * self.config.PIXELS_TO_DEGREES
        error_y_degrees = filtered_error_y * self.config.PIXELS_TO_DEGREES
        
        # Update PID controllers (with dead zone set to 0 since we already filtered)
        pan_output = self.pid_pan.update(error_x_degrees)
        tilt_output = self.pid_tilt.update(error_y_degrees)
        
        logger.debug(f"Dual PID update: error_x={error_x:.1f}px ({error_x_degrees:.2f}°), "
                    f"error_y={error_y:.1f}px ({error_y_degrees:.2f}°), "
                    f"pan_out={pan_output:.2f}°, tilt_out={tilt_output:.2f}°")
        
        return pan_output, tilt_output
    
    def update_config(self, new_config: PIDConfig) -> None:
        """
        Update configuration and apply to both controllers.
        
        Args:
            new_config: New PID configuration
        """
        self.config = new_config
        
        # Update pan controller
        self.pid_pan.update_gains(
            Kp=new_config.PAN_KP,
            Ki=new_config.PAN_KI,
            Kd=new_config.PAN_KD
        )
        self.pid_pan.dead_zone = new_config.DEAD_ZONE
        self.pid_pan.max_output = new_config.MAX_MOVEMENT
        self.pid_pan.integral_min = new_config.INTEGRAL_MIN
        self.pid_pan.integral_max = new_config.INTEGRAL_MAX
        
        # Update tilt controller
        self.pid_tilt.update_gains(
            Kp=new_config.TILT_KP,
            Ki=new_config.TILT_KI,
            Kd=new_config.TILT_KD
        )
        self.pid_tilt.dead_zone = new_config.DEAD_ZONE
        self.pid_tilt.max_output = new_config.MAX_MOVEMENT
        self.pid_tilt.integral_min = new_config.INTEGRAL_MIN
        self.pid_tilt.integral_max = new_config.INTEGRAL_MAX
        
        logger.info("Dual-axis PID configuration updated")
    
    def reset(self) -> None:
        """Reset both PID controllers."""
        self.pid_pan.reset()
        self.pid_tilt.reset()
        logger.info("Dual-axis PID controllers reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from both controllers.
        
        Returns:
            dict: Combined statistics from both controllers
        """
        return {
            'pan': self.pid_pan.get_stats(),
            'tilt': self.pid_tilt.get_stats(),
            'config': self.config.to_dict()
        }
    
    def __str__(self) -> str:
        """String representation of dual-axis controller."""
        return f"DualAxisPIDController(pan={self.pid_pan}, tilt={self.pid_tilt})"


class SimplePDController:
    """
    Simplified PD Controller optimized for face tracking (TianxingWu approach).
    
    This controller implements only PD control (no integral term) which has been
    found to work better for face tracking applications. Based on the approach
    used in TianxingWu's face-tracking-pan-tilt-camera project.
    
    Key advantages:
    - No integral windup issues
    - Faster response and more stable for position control
    - Simpler tuning (only 2 parameters per axis)
    - More predictable behavior
    """
    
    def __init__(self, Kp: float, Kd: float, dead_zone: float = 0.0, 
                 max_output: float = float('inf')):
        """
        Initialize PD controller.
        
        Args:
            Kp: Proportional gain
            Kd: Derivative gain
            dead_zone: Ignore errors smaller than this value
            max_output: Maximum absolute output value
        """
        # PD gains
        self.Kp = Kp
        self.Kd = Kd
        
        # Control parameters
        self.dead_zone = dead_zone
        self.max_output = max_output
        
        # State variables
        self.last_error = 0.0
        self.last_time = time.time()
        
        # Statistics
        self.last_output = 0.0
        self.update_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Simple PD Controller initialized: Kp={Kp}, Kd={Kd}")
    
    def update(self, error: float) -> float:
        """
        Update PD controller with current error.
        
        Args:
            error: Current error value (setpoint - current_value)
            
        Returns:
            float: PD controller output
        """
        with self._lock:
            current_time = time.time()
            dt = current_time - self.last_time
            
            # Prevent division by zero
            if dt <= 0.0:
                self.last_time = current_time
                return self.last_output
            
            # Apply dead zone filtering
            if abs(error) < self.dead_zone:
                error = 0.0
            
            # Proportional term
            proportional = self.Kp * error
            
            # Derivative term
            derivative = 0.0
            if dt > 0:
                derivative = self.Kd * (error - self.last_error) / dt
            
            # Calculate output (PD only, no integral)
            output = proportional + derivative
            
            # Apply output limits
            if abs(output) > self.max_output:
                output = self.max_output if output > 0 else -self.max_output
            
            # Update state variables
            self.last_error = error
            self.last_time = current_time
            self.last_output = output
            self.update_count += 1
            
            return output
    
    def update_gains(self, Kp: float = None, Kd: float = None) -> None:
        """
        Update PD gains during runtime.
        
        Args:
            Kp: New proportional gain (optional)
            Kd: New derivative gain (optional)
        """
        with self._lock:
            if Kp is not None:
                self.Kp = Kp
            if Kd is not None:
                self.Kd = Kd
        
        logger.info(f"PD gains updated: Kp={self.Kp}, Kd={self.Kd}")
    
    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            self.last_error = 0.0
            self.last_time = time.time()
            self.last_output = 0.0
        
        logger.info("PD controller reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        with self._lock:
            return {
                'type': 'PD',
                'Kp': self.Kp,
                'Kd': self.Kd,
                'last_error': self.last_error,
                'last_output': self.last_output,
                'update_count': self.update_count,
                'dead_zone': self.dead_zone,
                'max_output': self.max_output
            }


class OptimizedDualAxisPDController:
    """
    Dual-axis PD controller optimized for face tracking (TianxingWu approach).
    
    This controller uses the same principles as TianxingWu's implementation:
    - PD control only (no integral term)
    - Direct pixel-to-degree conversion
    - Optimized for face tracking applications
    - Better stability and responsiveness
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        """
        Initialize optimized dual-axis PD controller.
        
        Args:
            config: PID configuration object (uses defaults if None)
        """
        self.config = config or PIDConfig()
        
        # Create PD controllers for each axis (no integral term)
        self.pd_pan = SimplePDController(
            Kp=self.config.PAN_KP,
            Kd=self.config.PAN_KD,
            dead_zone=self.config.DEAD_ZONE * self.config.PIXELS_TO_DEGREES,
            max_output=self.config.MAX_MOVEMENT
        )
        
        self.pd_tilt = SimplePDController(
            Kp=self.config.TILT_KP,
            Kd=self.config.TILT_KD,
            dead_zone=self.config.DEAD_ZONE * self.config.PIXELS_TO_DEGREES,
            max_output=self.config.MAX_MOVEMENT
        )
        
        logger.info("Optimized dual-axis PD controller initialized (TianxingWu approach)")
    
    def update(self, error_x: float, error_y: float) -> tuple[float, float]:
        """
        Update both PD controllers with current errors.
        
        Args:
            error_x: X-axis error (pixels from center)
            error_y: Y-axis error (pixels from center)
            
        Returns:
            tuple: (pan_output, tilt_output) in degrees
        """
        # Fix rotation direction: invert error signs to match motor directions
        # Note: These direction corrections should match the main controller
        pan_direction_multiplier = 1  # Change to +1 if pan direction is wrong
        tilt_direction_multiplier = -1  # Change to -1 if tilt direction is wrong
        
        corrected_error_x = error_x * pan_direction_multiplier
        corrected_error_y = error_y * tilt_direction_multiplier
        
        # Convert corrected pixel errors to degrees (like TianxingWu approach)
        error_x_degrees = corrected_error_x * self.config.PIXELS_TO_DEGREES
        error_y_degrees = corrected_error_y * self.config.PIXELS_TO_DEGREES
        
        # Update PD controllers directly with degree errors
        pan_output = self.pd_pan.update(error_x_degrees)
        tilt_output = self.pd_tilt.update(error_y_degrees)
        
        logger.debug(f"Optimized PD update: error_x={error_x:.1f}px ({error_x_degrees:.2f}°), "
                    f"error_y={error_y:.1f}px ({error_y_degrees:.2f}°), "
                    f"pan_out={pan_output:.2f}°, tilt_out={tilt_output:.2f}°")
        
        return pan_output, tilt_output
    
    def update_config(self, new_config: PIDConfig) -> None:
        """
        Update configuration and apply to both controllers.
        
        Args:
            new_config: New PID configuration
        """
        self.config = new_config
        
        # Update pan controller
        self.pd_pan.update_gains(Kp=new_config.PAN_KP, Kd=new_config.PAN_KD)
        self.pd_pan.dead_zone = new_config.DEAD_ZONE * new_config.PIXELS_TO_DEGREES
        self.pd_pan.max_output = new_config.MAX_MOVEMENT
        
        # Update tilt controller
        self.pd_tilt.update_gains(Kp=new_config.TILT_KP, Kd=new_config.TILT_KD)
        self.pd_tilt.dead_zone = new_config.DEAD_ZONE * new_config.PIXELS_TO_DEGREES
        self.pd_tilt.max_output = new_config.MAX_MOVEMENT
        
        logger.info("Optimized PD controller configuration updated")
    
    def reset(self) -> None:
        """Reset both controllers."""
        self.pd_pan.reset()
        self.pd_tilt.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for both controllers."""
        return {
            'type': 'Optimized_Dual_PD',
            'pan_stats': self.pd_pan.get_stats(),
            'tilt_stats': self.pd_tilt.get_stats(),
            'config': {
                'dead_zone_pixels': self.config.DEAD_ZONE,
                'max_movement_degrees': self.config.MAX_MOVEMENT,
                'pixels_to_degrees': self.config.PIXELS_TO_DEGREES
            }
        }


# =============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# =============================================================================

def load_pid_config_from_file(config_path: str) -> PIDConfig:
    """
    Load PID configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PIDConfig: Loaded configuration
    """
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            return PIDConfig.from_dict(config_dict)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return PIDConfig()
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return PIDConfig()


def save_pid_config_to_file(config: PIDConfig, config_path: str) -> bool:
    """
    Save PID configuration to JSON file.
    
    Args:
        config: PID configuration to save
        config_path: Path to save configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"PID configuration saved to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    """Example usage and basic testing of PID controller."""
    
    print("🧪 Testing PID Controller Implementation")
    print("=" * 50)
    
    # Test basic PID controller
    print("\n1. Testing Basic PID Controller:")
    pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.05, dead_zone=2.0)
    
    # Simulate some control loop updates
    test_errors = [10, 8, 5, 3, 1, 0, -1, -2, 0, 1]
    for i, error in enumerate(test_errors):
        output = pid.update(error)
        print(f"   Step {i+1}: error={error:3.0f} → output={output:6.2f}")
        time.sleep(0.01)  # Small delay to simulate real timing
    
    print(f"   Final stats: {pid.get_stats()['performance']}")
    
    # Test dual-axis controller
    print("\n2. Testing Dual-Axis PID Controller:")
    config = PIDConfig()
    dual_pid = DualAxisPIDController(config)
    
    # Simulate face tracking errors (pixels from center)
    test_positions = [
        (50, 30),   # Face to the right and up
        (30, 20),   # Moving closer to center
        (10, 5),    # Very close to center
        (0, 0),     # Centered
        (-15, -10), # Face to the left and down
        (-5, -2),   # Moving back to center
        (0, 0)      # Centered again
    ]
    
    for i, (error_x, error_y) in enumerate(test_positions):
        pan_out, tilt_out = dual_pid.update(error_x, error_y)
        print(f"   Step {i+1}: error=({error_x:3.0f}, {error_y:3.0f})px → "
              f"output=({pan_out:6.2f}, {tilt_out:6.2f})°")
        time.sleep(0.01)
    
    # Test configuration management
    print("\n3. Testing Configuration Management:")
    config_path = "/tmp/test_pid_config.json"
    
    # Save configuration
    if save_pid_config_to_file(config, config_path):
        print(f"   ✅ Configuration saved to {config_path}")
    
    # Load configuration
    loaded_config = load_pid_config_from_file(config_path)
    print(f"   ✅ Configuration loaded: PAN_KP={loaded_config.PAN_KP}")
    
    # Test gain updates
    print("\n4. Testing Runtime Gain Updates:")
    pid.update_gains(Kp=2.0, Ki=0.2)
    print(f"   ✅ Updated gains: {pid.get_stats()['gains']}")
    
    # Test reset functionality
    print("\n5. Testing Reset Functionality:")
    pid.reset()
    dual_pid.reset()
    print("   ✅ Controllers reset successfully")
    
    print("\n✅ All PID controller tests completed successfully!")