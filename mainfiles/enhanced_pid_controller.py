#!/usr/bin/env python3
"""
Enhanced PID Controller with Faster Convergence

This controller implements advanced control techniques for faster convergence:
- Adaptive gain scheduling based on error magnitude
- Feedforward control for predictive tracking
- Dead-time compensation for improved responsiveness
- Enhanced anti-windup with conditional integration
- Motion prediction and velocity estimation
- Convergence monitoring and optimization
"""

import time
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
import threading
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ConvergenceMode(Enum):
    """Convergence optimization modes."""
    STANDARD = "standard"
    FAST = "fast"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class EnhancedPIDConfig:
    """
    Enhanced PID configuration optimized for faster convergence and better tracking.
    """
    
    # =============================================================================
    # CONVERGENCE OPTIMIZATION
    # =============================================================================
    CONVERGENCE_MODE: ConvergenceMode = ConvergenceMode.ADAPTIVE
    
    # =============================================================================
    # ADAPTIVE GAIN SCHEDULING
    # =============================================================================
    
    # Base PID gains (used as minimum values)
    BASE_PAN_KP: float = 0.08    # Increased from 0.04 for faster response
    BASE_PAN_KI: float = 0.02    # Small integral for steady-state error
    BASE_PAN_KD: float = 0.25    # Increased damping for stability
    
    BASE_TILT_KP: float = 0.10   # Higher for vertical movement
    BASE_TILT_KI: float = 0.02   # Small integral for gravity compensation
    BASE_TILT_KD: float = 0.30   # Higher damping for stability
    
    # Adaptive gain multipliers based on error magnitude
    GAIN_MULTIPLIER_LARGE_ERROR: float = 2.5   # Multiply gains for large errors
    GAIN_MULTIPLIER_MEDIUM_ERROR: float = 1.5  # Multiply gains for medium errors
    GAIN_MULTIPLIER_SMALL_ERROR: float = 0.8   # Reduce gains for small errors
    
    # Error thresholds for gain scheduling (in pixels)
    LARGE_ERROR_THRESHOLD: float = 80.0
    MEDIUM_ERROR_THRESHOLD: float = 30.0
    SMALL_ERROR_THRESHOLD: float = 10.0
    
    # =============================================================================
    # FEEDFORWARD CONTROL
    # =============================================================================
    ENABLE_FEEDFORWARD: bool = True
    FEEDFORWARD_GAIN_PAN: float = 0.15     # Predictive control for pan
    FEEDFORWARD_GAIN_TILT: float = 0.18    # Predictive control for tilt
    
    # Motion prediction (based on face velocity)
    ENABLE_MOTION_PREDICTION: bool = True
    PREDICTION_HORIZON: float = 0.2  # seconds
    
    # =============================================================================
    # CONTROL PARAMETERS
    # =============================================================================
    
    # Dead zone settings
    DEAD_ZONE_ADAPTIVE: bool = True
    DEAD_ZONE_MIN: float = 5.0  # Minimum dead zone in pixels
    DEAD_ZONE_MAX: float = 15.0  # Maximum dead zone in pixels
    DEAD_ZONE_BASE: float = 8.0  # Base dead zone for calculations
    DEAD_ZONE: float = 8.0  # Simple dead zone for backward compatibility
    
    # Movement limits
    MAX_MOVEMENT_BASE: float = 12.0  # Base maximum movement in degrees
    MAX_MOVEMENT_BURST: float = 25.0  # Maximum burst movement for large errors
    BURST_ERROR_THRESHOLD: float = 100.0  # Error threshold for burst movement
    MAX_MOVEMENT: float = 12.0  # Simple max movement for backward compatibility
    
    # Conversion parameters
    PIXELS_TO_DEGREES: float = 0.08  # Pixel to degree conversion factor
    
    # =============================================================================
    # ADVANCED CONTROL FEATURES
    # =============================================================================
    
    # Enhanced anti-windup
    ENHANCED_ANTI_WINDUP: bool = True
    INTEGRAL_DECAY_FACTOR: float = 0.98  # Decay integral over time
    INTEGRAL_RESET_THRESHOLD: float = 50.0  # Reset integral for large errors
    CONDITIONAL_INTEGRATION: bool = True  # Only integrate when beneficial
    
    # Dead-time compensation
    ENABLE_DEAD_TIME_COMPENSATION: bool = True
    DEAD_TIME_STEPS: int = 2  # Number of steps to predict ahead
    
    # Derivative improvements
    ENABLE_DERIVATIVE_ON_MEASUREMENT: bool = True  # Prevent derivative kick
    
    # Setpoint ramping
    ENABLE_SETPOINT_RAMPING: bool = True
    SETPOINT_RAMP_RATE: float = 50.0  # pixels/second maximum setpoint change
    
    # Performance monitoring
    CONVERGENCE_MONITORING: bool = True
    CONVERGENCE_THRESHOLD: float = 10.0  # pixels
    CONVERGENCE_TIME_WINDOW: float = 2.0  # seconds
    
    # =============================================================================
    # VELOCITY ESTIMATION
    # =============================================================================
    VELOCITY_HISTORY_SIZE: int = 5  # Number of positions to track for velocity
    VELOCITY_SMOOTHING_FACTOR: float = 0.3  # Smoothing factor for velocity estimation
    
    # =============================================================================
    # GRAVITY COMPENSATION
    # =============================================================================
    GRAVITY_COMPENSATION_JOINT1: float = 0.0  # Pan joint - no gravity effect
    GRAVITY_COMPENSATION_JOINT4: float = 2.5  # Tilt joint - compensate for camera weight
    
    # =============================================================================
    # DEAD-TIME COMPENSATION
    # =============================================================================
    ESTIMATED_DEAD_TIME: float = 0.1  # Estimated system dead time in seconds
    DEAD_TIME_PREDICTION_GAIN: float = 0.5  # Gain for dead-time prediction
    
    def __post_init__(self):
        """Post-initialization to compute derived parameters."""
        # Pre-compute conversion factors
        self._dead_zone_base_degrees = self.DEAD_ZONE_BASE * self.PIXELS_TO_DEGREES
        self._dead_zone_min_degrees = self.DEAD_ZONE_MIN * self.PIXELS_TO_DEGREES
        self._dead_zone_max_degrees = self.DEAD_ZONE_MAX * self.PIXELS_TO_DEGREES
        
        # Pre-compute thresholds in degrees
        self._large_error_degrees = self.LARGE_ERROR_THRESHOLD * self.PIXELS_TO_DEGREES
        self._medium_error_degrees = self.MEDIUM_ERROR_THRESHOLD * self.PIXELS_TO_DEGREES
        self._small_error_degrees = self.SMALL_ERROR_THRESHOLD * self.PIXELS_TO_DEGREES
    
    def get_adaptive_gains(self, error_magnitude: float, velocity_magnitude: float = 0.0) -> Tuple[float, float, float]:
        """Get adaptive gain multipliers based on error and velocity."""
        if error_magnitude >= self.LARGE_ERROR_THRESHOLD:
            return (self.GAIN_MULTIPLIER_LARGE_ERROR, 
                   self.GAIN_MULTIPLIER_LARGE_ERROR * 0.8,  # Reduce Ki slightly
                   self.GAIN_MULTIPLIER_LARGE_ERROR * 1.2)  # Increase Kd for stability
        elif error_magnitude >= self.MEDIUM_ERROR_THRESHOLD:
            return (self.GAIN_MULTIPLIER_MEDIUM_ERROR,
                   self.GAIN_MULTIPLIER_MEDIUM_ERROR * 0.9,
                   self.GAIN_MULTIPLIER_MEDIUM_ERROR * 1.1)
        elif error_magnitude <= self.SMALL_ERROR_THRESHOLD:
            return (self.GAIN_MULTIPLIER_SMALL_ERROR,
                   self.GAIN_MULTIPLIER_SMALL_ERROR * 1.1,  # Slightly increase Ki for small errors
                   self.GAIN_MULTIPLIER_SMALL_ERROR * 0.9)  # Reduce Kd for smoothness
        else:
            return (1.0, 1.0, 1.0)  # No modification for medium errors
    
    def get_adaptive_dead_zone(self, error_magnitude: float, convergence_quality: float = 1.0) -> float:
        """Get adaptive dead zone based on error magnitude and convergence quality."""
        if not self.DEAD_ZONE_ADAPTIVE:
            return self.DEAD_ZONE_BASE
        
        # Scale dead zone based on error magnitude and convergence
        if error_magnitude < self.SMALL_ERROR_THRESHOLD:
            # Small errors - larger dead zone to prevent jitter
            base_zone = self.DEAD_ZONE_MAX
        elif error_magnitude > self.LARGE_ERROR_THRESHOLD:
            # Large errors - smaller dead zone for responsiveness
            base_zone = self.DEAD_ZONE_MIN
        else:
            # Medium errors - interpolate
            factor = (error_magnitude - self.SMALL_ERROR_THRESHOLD) / (self.LARGE_ERROR_THRESHOLD - self.SMALL_ERROR_THRESHOLD)
            base_zone = self.DEAD_ZONE_MAX - factor * (self.DEAD_ZONE_MAX - self.DEAD_ZONE_MIN)
        
        # Adjust based on convergence quality
        return base_zone * convergence_quality
    
    def calculate_feedforward(self, velocity: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate feedforward terms based on target velocity."""
        if not self.ENABLE_FEEDFORWARD:
            return (0.0, 0.0)
        
        vx, vy = velocity
        
        # Convert velocity to degrees/second
        vx_deg = vx * self.PIXELS_TO_DEGREES
        vy_deg = vy * self.PIXELS_TO_DEGREES
        
        # Apply feedforward gains
        ff_pan = self.FEEDFORWARD_GAIN_PAN * vx_deg
        ff_tilt = self.FEEDFORWARD_GAIN_TILT * vy_deg
        
        return (ff_pan, ff_tilt)
    
    def get_convergence_quality(self) -> float:
        """Get current convergence quality (placeholder - implement based on performance tracking)."""
        # This would be implemented based on actual performance metrics
        # For now, return a default value
        return 1.0
    
    def get_max_movement(self, error_magnitude: float) -> float:
        """Get maximum movement based on error magnitude."""
        if error_magnitude >= self.BURST_ERROR_THRESHOLD:
            return self.MAX_MOVEMENT_BURST
        else:
            # Interpolate between base and burst based on error
            factor = min(1.0, error_magnitude / self.BURST_ERROR_THRESHOLD)
            return self.MAX_MOVEMENT_BASE + factor * (self.MAX_MOVEMENT_BURST - self.MAX_MOVEMENT_BASE)
    
    def update_convergence_history(self, error_magnitude: float, convergence_time: float):
        """Update convergence history (placeholder for performance tracking)."""
        # This would track convergence performance over time
        # For now, just pass
        pass
    
    def is_converged(self, error_magnitude: float) -> bool:
        """Check if the system has converged based on error magnitude."""
        return error_magnitude < self.CONVERGENCE_THRESHOLD
    
    def print_summary(self):
        """Print a summary of the enhanced PID configuration."""
        print("📊 Enhanced PID Configuration Summary")
        print("=" * 50)
        print(f"Convergence Mode:    {self.CONVERGENCE_MODE.value}")
        print(f"Pan Gains:           Kp={self.BASE_PAN_KP:.3f}, Ki={self.BASE_PAN_KI:.3f}, Kd={self.BASE_PAN_KD:.3f}")
        print(f"Tilt Gains:          Kp={self.BASE_TILT_KP:.3f}, Ki={self.BASE_TILT_KI:.3f}, Kd={self.BASE_TILT_KD:.3f}")
        print(f"Dead Zone Range:     {self.DEAD_ZONE_MIN}-{self.DEAD_ZONE_MAX} pixels")
        print(f"Movement Range:      Base={self.MAX_MOVEMENT_BASE}°, Burst={self.MAX_MOVEMENT_BURST}°")
        print(f"Pixels to Degrees:   {self.PIXELS_TO_DEGREES:.3f}")
        print(f"Feedforward:         {'Enabled' if self.ENABLE_FEEDFORWARD else 'Disabled'}")
        print(f"Motion Prediction:   {'Enabled' if self.ENABLE_MOTION_PREDICTION else 'Disabled'}")
        print(f"Anti-windup:         {'Enhanced' if self.ENHANCED_ANTI_WINDUP else 'Standard'}")
        print(f"Dead-time Comp:      {'Enabled' if self.ENABLE_DEAD_TIME_COMPENSATION else 'Disabled'}")
        print("=" * 50)


def create_enhanced_config(mode: ConvergenceMode = ConvergenceMode.ADAPTIVE) -> EnhancedPIDConfig:
    """Create an enhanced PID configuration with specified convergence mode."""
    config = EnhancedPIDConfig()
    config.CONVERGENCE_MODE = mode
    
    if mode == ConvergenceMode.AGGRESSIVE:
        # More aggressive settings for fastest convergence
        config.BASE_PAN_KP = 0.12
        config.BASE_TILT_KP = 0.15
        config.GAIN_MULTIPLIER_LARGE_ERROR = 3.0
        config.MAX_MOVEMENT_BURST = 30.0
        config.DEAD_ZONE_MIN = 3.0
        
    elif mode == ConvergenceMode.FAST:
        # Balanced fast convergence
        config.BASE_PAN_KP = 0.10
        config.BASE_TILT_KP = 0.12
        config.GAIN_MULTIPLIER_LARGE_ERROR = 2.8
        config.MAX_MOVEMENT_BURST = 28.0
        
    elif mode == ConvergenceMode.ADAPTIVE:
        # Default adaptive settings (already set in class)
        pass
        
    elif mode == ConvergenceMode.STANDARD:
        # Conservative settings for stability
        config.BASE_PAN_KP = 0.06
        config.BASE_TILT_KP = 0.08
        config.GAIN_MULTIPLIER_LARGE_ERROR = 2.0
        config.MAX_MOVEMENT_BURST = 20.0
        config.DEAD_ZONE_MIN = 8.0
    
    # Re-run post-init to update computed parameters
    config.__post_init__()
    
    return config


@dataclass
class MotionData:
    """Structure for tracking motion and velocity."""
    timestamp: float
    position: Tuple[float, float]  # (x, y) in pixels
    velocity: Optional[Tuple[float, float]] = None  # (vx, vy) in pixels/sec
    acceleration: Optional[Tuple[float, float]] = None  # (ax, ay) in pixels/sec²


class VelocityEstimator:
    """Estimates target velocity for feedforward control."""
    
    def __init__(self, history_size: int = 5, smoothing_factor: float = 0.7):
        self.history_size = history_size
        self.smoothing_factor = smoothing_factor
        self.position_history = deque(maxlen=history_size)
        self.velocity_estimate = (0.0, 0.0)
        
    def update(self, position: Tuple[float, float], timestamp: float) -> Tuple[float, float]:
        """Update velocity estimate with new position."""
        self.position_history.append((position, timestamp))
        
        if len(self.position_history) < 2:
            return self.velocity_estimate
        
        # Calculate velocity using multiple points for smoothness
        velocities = []
        for i in range(1, len(self.position_history)):
            (x1, y1), t1 = self.position_history[i-1]
            (x2, y2), t2 = self.position_history[i]
            
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                velocities.append((vx, vy))
        
        if velocities:
            # Average velocities and apply smoothing
            avg_vx = sum(v[0] for v in velocities) / len(velocities)
            avg_vy = sum(v[1] for v in velocities) / len(velocities)
            
            # Apply exponential smoothing
            self.velocity_estimate = (
                self.smoothing_factor * self.velocity_estimate[0] + (1 - self.smoothing_factor) * avg_vx,
                self.smoothing_factor * self.velocity_estimate[1] + (1 - self.smoothing_factor) * avg_vy
            )
        
        return self.velocity_estimate
    
    def predict_position(self, horizon: float) -> Tuple[float, float]:
        """Predict future position based on current velocity."""
        if not self.position_history:
            return (0.0, 0.0)
        
        current_pos, _ = self.position_history[-1]
        vx, vy = self.velocity_estimate
        
        predicted_x = current_pos[0] + vx * horizon
        predicted_y = current_pos[1] + vy * horizon
        
        return (predicted_x, predicted_y)


class EnhancedPIDController:
    """
    Enhanced PID controller with adaptive features for faster convergence.
    """
    
    def __init__(self, config: EnhancedPIDConfig):
        self.config = config
        self._lock = threading.RLock()
        
        # Initialize velocity estimator
        self.velocity_estimator = VelocityEstimator(
            history_size=config.VELOCITY_HISTORY_SIZE,
            smoothing_factor=config.VELOCITY_SMOOTHING_FACTOR
        )
        
        # Initialize controller state
        self._reset_controller_state()
        
        # Performance tracking
        self._initialize_performance_tracking()
        
        # Dead-time compensation buffer
        self._output_history = deque(maxlen=10)
        
        logger.info(f"Enhanced PID Controller initialized with {config.CONVERGENCE_MODE.value} mode")
    
    def _reset_controller_state(self):
        """Reset all controller state variables."""
        current_time = time.time()
        
        # Pan controller state
        self._pan_state = {
            'last_error': 0.0,
            'last_measurement': 0.0,  # For derivative on measurement
            'integral': 0.0,
            'last_time': current_time,
            'last_output': 0.0,
            'error_sum_time': 0.0  # For conditional integration
        }
        
        # Tilt controller state  
        self._tilt_state = {
            'last_error': 0.0,
            'last_measurement': 0.0,
            'integral': 0.0,
            'last_time': current_time,
            'last_output': 0.0,
            'error_sum_time': 0.0
        }
        
        # Convergence tracking
        self._convergence_start_time = None
        self._last_setpoint = (0.0, 0.0)
        self._setpoint_filter = (0.0, 0.0)
    
    def _initialize_performance_tracking(self):
        """Initialize performance monitoring."""
        self._update_count = 0
        self._total_compute_time = 0.0
        self._last_update_time = time.time()
        self._performance_history = deque(maxlen=100)
    
    def update(self, error_x: float, error_y: float, 
               current_position: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Enhanced PID update with adaptive features.
        
        Args:
            error_x: Pan error in pixels (positive = target right of center)
            error_y: Tilt error in pixels (positive = target below center)
            current_position: Current target position for velocity estimation
            
        Returns:
            Control outputs (pan_degrees, tilt_degrees)
        """
        start_time = time.time()
        
        with self._lock:
            current_time = time.time()
            
            # Update velocity estimation if position provided
            if current_position is not None:
                target_velocity = self.velocity_estimator.update(current_position, current_time)
            else:
                target_velocity = (0.0, 0.0)
            
            # Calculate error magnitude for adaptive control
            error_magnitude = math.sqrt(error_x**2 + error_y**2)
            
            # Get adaptive parameters
            convergence_quality = self.config.get_convergence_quality()
            kp_mult, ki_mult, kd_mult = self.config.get_adaptive_gains(error_magnitude, 
                                                                     math.sqrt(target_velocity[0]**2 + target_velocity[1]**2))
            
            # Apply setpoint ramping for smooth transitions
            if self.config.ENABLE_SETPOINT_RAMPING:
                error_x, error_y = self._apply_setpoint_ramping(error_x, error_y, current_time)
            
            # Calculate adaptive dead zones
            dead_zone_x = self.config.get_adaptive_dead_zone(abs(error_x), convergence_quality)
            dead_zone_y = self.config.get_adaptive_dead_zone(abs(error_y), convergence_quality)
            
            # Apply dead zones
            if abs(error_x) < dead_zone_x:
                error_x = 0.0
            if abs(error_y) < dead_zone_y:
                error_y = 0.0
            
            # Calculate feedforward terms
            ff_pan, ff_tilt = self.config.calculate_feedforward(target_velocity)
            
            # Update pan controller
            pan_output = self._update_enhanced_axis(
                self._pan_state, error_x, current_time,
                self.config.BASE_PAN_KP * kp_mult,
                self.config.BASE_PAN_KI * ki_mult,
                self.config.BASE_PAN_KD * kd_mult,
                feedforward=ff_pan,
                gravity_compensation=self.config.GRAVITY_COMPENSATION_JOINT1
            )
            
            # Update tilt controller
            tilt_output = self._update_enhanced_axis(
                self._tilt_state, error_y, current_time,
                self.config.BASE_TILT_KP * kp_mult,
                self.config.BASE_TILT_KI * ki_mult,
                self.config.BASE_TILT_KD * kd_mult,
                feedforward=ff_tilt,
                gravity_compensation=self.config.GRAVITY_COMPENSATION_JOINT4
            )
            
            # Apply movement limits with burst capability
            max_movement = self.config.get_max_movement(error_magnitude)
            pan_output = self._limit_output(pan_output, max_movement)
            tilt_output = self._limit_output(tilt_output, max_movement)
            
            # Dead-time compensation
            if self.config.ENABLE_DEAD_TIME_COMPENSATION:
                pan_output, tilt_output = self._apply_dead_time_compensation(pan_output, tilt_output)
            
            # Update performance tracking
            self._update_performance_tracking(start_time, error_magnitude, 
                                            math.sqrt(pan_output**2 + tilt_output**2))
            
            # Update convergence monitoring
            self.config.update_convergence_history(error_magnitude, 
                                                 math.sqrt(pan_output**2 + tilt_output**2))
            
            return pan_output, tilt_output
    
    def _update_enhanced_axis(self, state: Dict, error: float, current_time: float,
                            kp: float, ki: float, kd: float,
                            feedforward: float = 0.0, gravity_compensation: float = 0.0) -> float:
        """Enhanced single-axis PID update with advanced features."""
        
        dt = current_time - state['last_time']
        if dt <= 0.0:
            state['last_time'] = current_time
            return state['last_output']
        
        # Proportional term
        proportional = kp * error
        
        # Enhanced integral term with conditional integration
        integral = 0.0
        if ki != 0.0:
            # Conditional integration: only integrate when error is decreasing or small
            error_decreasing = (error * state['last_error']) <= 0  # Sign change or zero
            error_small = abs(error) < self.config.LARGE_ERROR_THRESHOLD
            
            if self.config.CONDITIONAL_INTEGRATION and not (error_decreasing or error_small):
                # Don't integrate when error is large and increasing
                pass
            else:
                state['integral'] += error * dt
                state['error_sum_time'] += dt
                
                # Enhanced anti-windup
                if self.config.ENHANCED_ANTI_WINDUP:
                    # Decay integral over time
                    state['integral'] *= self.config.INTEGRAL_DECAY_FACTOR
                    
                    # Reset integral for large errors
                    if abs(error) > self.config.INTEGRAL_RESET_THRESHOLD:
                        state['integral'] = 0.0
                        state['error_sum_time'] = 0.0
                
                # Standard anti-windup limits
                max_integral = self.config.MAX_MOVEMENT_BASE / max(ki, 0.001)  # Prevent division by zero
                state['integral'] = max(-max_integral, min(max_integral, state['integral']))
            
            integral = ki * state['integral']
        
        # Enhanced derivative term
        derivative = 0.0
        if kd != 0.0 and dt > 0.0:
            if self.config.ENABLE_DERIVATIVE_ON_MEASUREMENT:
                # Derivative on measurement to prevent derivative kick
                measurement = -error  # Negative error represents measurement
                derivative = kd * (measurement - state['last_measurement']) / dt
                state['last_measurement'] = measurement
            else:
                # Traditional derivative on error
                derivative = kd * (error - state['last_error']) / dt
        
        # Combine PID terms
        pid_output = proportional + integral + derivative
        
        # Add feedforward and gravity compensation
        total_output = pid_output + feedforward + gravity_compensation
        
        # Update state
        state['last_error'] = error
        state['last_time'] = current_time
        state['last_output'] = total_output
        
        return total_output
    
    def _apply_setpoint_ramping(self, error_x: float, error_y: float, current_time: float) -> Tuple[float, float]:
        """Apply setpoint ramping for smooth transitions."""
        dt = current_time - self._last_update_time
        if dt <= 0:
            return error_x, error_y
        
        # Calculate desired setpoint change
        setpoint_x = -error_x  # Convert error to setpoint
        setpoint_y = -error_y
        
        # Calculate maximum allowed change
        max_change = self.config.SETPOINT_RAMP_RATE * dt
        
        # Ramp setpoint gradually
        delta_x = setpoint_x - self._setpoint_filter[0]
        delta_y = setpoint_y - self._setpoint_filter[1]
        
        # Limit rate of change
        if abs(delta_x) > max_change:
            delta_x = max_change if delta_x > 0 else -max_change
        if abs(delta_y) > max_change:
            delta_y = max_change if delta_y > 0 else -max_change
        
        # Update filtered setpoint
        self._setpoint_filter = (
            self._setpoint_filter[0] + delta_x,
            self._setpoint_filter[1] + delta_y
        )
        
        # Convert back to error
        ramped_error_x = -self._setpoint_filter[0]
        ramped_error_y = -self._setpoint_filter[1]
        
        self._last_update_time = current_time
        
        return ramped_error_x, ramped_error_y
    
    def _apply_dead_time_compensation(self, pan_output: float, tilt_output: float) -> Tuple[float, float]:
        """Apply dead-time compensation for improved responsiveness."""
        
        # Store current output in history
        self._output_history.append((pan_output, tilt_output, time.time()))
        
        # Find output from dead_time ago
        compensation_time = time.time() - self.config.ESTIMATED_DEAD_TIME
        
        # Find closest historical output
        compensated_pan = pan_output
        compensated_tilt = tilt_output
        
        for hist_pan, hist_tilt, hist_time in reversed(self._output_history):
            if hist_time <= compensation_time:
                # Calculate compensation
                compensation_pan = (pan_output - hist_pan) * self.config.DEAD_TIME_PREDICTION_GAIN
                compensation_tilt = (tilt_output - hist_tilt) * self.config.DEAD_TIME_PREDICTION_GAIN
                
                compensated_pan = pan_output + compensation_pan
                compensated_tilt = tilt_output + compensation_tilt
                break
        
        return compensated_pan, compensated_tilt
    
    def _limit_output(self, output: float, max_movement: float) -> float:
        """Apply output limits with saturation."""
        return max(-max_movement, min(max_movement, output))
    
    def _update_performance_tracking(self, start_time: float, error_magnitude: float, output_magnitude: float):
        """Update performance monitoring."""
        compute_time = time.time() - start_time
        self._total_compute_time += compute_time
        self._update_count += 1
        
        # Track performance metrics
        self._performance_history.append({
            'timestamp': time.time(),
            'compute_time': compute_time,
            'error_magnitude': error_magnitude,
            'output_magnitude': output_magnitude
        })
        
        # Log performance warnings
        if compute_time > 0.010:  # More than 10ms
            logger.warning(f"Slow PID update: {compute_time*1000:.1f}ms")
    
    def is_converged(self, error_x: float, error_y: float) -> bool:
        """Check if the system has converged."""
        error_magnitude = math.sqrt(error_x**2 + error_y**2)
        return self.config.is_converged(error_magnitude)
    
    def reset(self):
        """Reset controller to initial state."""
        with self._lock:
            self._reset_controller_state()
            self.velocity_estimator = VelocityEstimator(
                history_size=self.config.VELOCITY_HISTORY_SIZE,
                smoothing_factor=self.config.VELOCITY_SMOOTHING_FACTOR
            )
            self._output_history.clear()
            logger.info("Enhanced PID controller reset")
    
    def update_config(self, new_config: EnhancedPIDConfig):
        """Update controller configuration."""
        with self._lock:
            self.config = new_config
            logger.info(f"Updated to {new_config.CONVERGENCE_MODE.value} mode")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        with self._lock:
            avg_compute_time = self._total_compute_time / max(1, self._update_count)
            
            recent_performance = list(self._performance_history)[-20:]  # Last 20 updates
            
            if recent_performance:
                recent_avg_error = sum(p['error_magnitude'] for p in recent_performance) / len(recent_performance)
                recent_avg_compute = sum(p['compute_time'] for p in recent_performance) / len(recent_performance)
            else:
                recent_avg_error = 0.0
                recent_avg_compute = avg_compute_time
            
            current_velocity = self.velocity_estimator.velocity_estimate
            convergence_quality = self.config.get_convergence_quality()
            
            return {
                'mode': self.config.CONVERGENCE_MODE.value,
                'performance': {
                    'update_count': self._update_count,
                    'avg_compute_time_ms': avg_compute_time * 1000,
                    'recent_avg_error': recent_avg_error,
                    'recent_compute_time_ms': recent_avg_compute * 1000,
                    'convergence_quality': convergence_quality,
                    'update_rate_hz': self._update_count / max(1, time.time() - self._last_update_time)
                },
                'control_state': {
                    'pan_integral': self._pan_state['integral'],
                    'tilt_integral': self._tilt_state['integral'],
                    'pan_last_output': self._pan_state['last_output'],
                    'tilt_last_output': self._tilt_state['last_output'],
                    'target_velocity': current_velocity
                },
                'adaptive_params': {
                    'current_dead_zone': self.config.get_adaptive_dead_zone(recent_avg_error, convergence_quality),
                    'current_max_movement': self.config.get_max_movement(recent_avg_error),
                    'gain_multipliers': self.config.get_adaptive_gains(recent_avg_error)
                }
            }
    
    def tune_automatically(self, target_error: float = 5.0, max_iterations: int = 50):
        """
        Automatic tuning procedure to optimize gains for target error.
        
        Args:
            target_error: Target error magnitude in pixels
            max_iterations: Maximum tuning iterations
        """
        logger.info(f"Starting automatic tuning for target error {target_error} pixels")
        
        # Simple gradient-based tuning
        learning_rate = 0.1
        best_performance = float('inf')
        best_gains = (self.config.BASE_PAN_KP, self.config.BASE_TILT_KP)
        
        for iteration in range(max_iterations):
            # Test current gains
            recent_performance = list(self._performance_history)[-10:]
            if recent_performance:
                avg_error = sum(p['error_magnitude'] for p in recent_performance) / len(recent_performance)
                performance_metric = abs(avg_error - target_error)
                
                if performance_metric < best_performance:
                    best_performance = performance_metric
                    best_gains = (self.config.BASE_PAN_KP, self.config.BASE_TILT_KP)
                
                # Adjust gains based on error
                if avg_error > target_error:
                    # Increase gains
                    self.config.BASE_PAN_KP *= (1 + learning_rate)
                    self.config.BASE_TILT_KP *= (1 + learning_rate)
                else:
                    # Decrease gains slightly
                    self.config.BASE_PAN_KP *= (1 - learning_rate * 0.5)
                    self.config.BASE_TILT_KP *= (1 - learning_rate * 0.5)
                
                # Ensure gains stay within reasonable bounds
                self.config.BASE_PAN_KP = max(0.01, min(1.0, self.config.BASE_PAN_KP))
                self.config.BASE_TILT_KP = max(0.01, min(1.0, self.config.BASE_TILT_KP))
                
                logger.debug(f"Tuning iteration {iteration}: error={avg_error:.1f}, gains=({self.config.BASE_PAN_KP:.3f}, {self.config.BASE_TILT_KP:.3f})")
        
        # Restore best gains
        self.config.BASE_PAN_KP, self.config.BASE_TILT_KP = best_gains
        logger.info(f"Tuning complete. Best gains: Kp_pan={self.config.BASE_PAN_KP:.3f}, Kp_tilt={self.config.BASE_TILT_KP:.3f}")


class UltraFastEnhancedController:
    """
    Ultra-fast version of enhanced controller for maximum performance.
    Sacrifices some features for speed but keeps core enhancements.
    """
    
    def __init__(self, config: EnhancedPIDConfig):
        self.config = config
        
        # Pre-compute frequently used values
        self._dead_zone_base = config.DEAD_ZONE_BASE * config.PIXELS_TO_DEGREES
        self._max_movement = config.MAX_MOVEMENT_BASE
        self._pixels_to_degrees = config.PIXELS_TO_DEGREES
        
        # Minimal state tracking
        self.pan_state = {'last_error': 0.0, 'last_time': time.time(), 'last_output': 0.0}
        self.tilt_state = {'last_error': 0.0, 'last_time': time.time(), 'last_output': 0.0}
        
        # Simple velocity estimation
        self._last_position = (0.0, 0.0)
        self._last_position_time = time.time()
        self._velocity = (0.0, 0.0)
    
    def update(self, error_x: float, error_y: float) -> Tuple[float, float]:
        """Ultra-fast update with minimal overhead."""
        current_time = time.time()
        
        # Simple dead zone
        if abs(error_x) < self._dead_zone_base:
            error_x = 0.0
        if abs(error_y) < self._dead_zone_base:
            error_y = 0.0
        
        # Fast adaptive gains based on error magnitude
        error_mag = abs(error_x) + abs(error_y)  # Faster than sqrt
        if error_mag > self.config.LARGE_ERROR_THRESHOLD:
            gain_mult = self.config.GAIN_MULTIPLIER_LARGE_ERROR
        elif error_mag > self.config.MEDIUM_ERROR_THRESHOLD:
            gain_mult = self.config.GAIN_MULTIPLIER_MEDIUM_ERROR
        else:
            gain_mult = 1.0
        
        # Fast PD control
        pan_output = self._fast_pd_update(self.pan_state, error_x, current_time, 
                                        self.config.BASE_PAN_KP * gain_mult,
                                        self.config.BASE_PAN_KD * gain_mult)
        
        tilt_output = self._fast_pd_update(self.tilt_state, error_y, current_time,
                                         self.config.BASE_TILT_KP * gain_mult,
                                         self.config.BASE_TILT_KD * gain_mult)
        
        # Apply limits
        pan_output = max(-self._max_movement, min(self._max_movement, pan_output))
        tilt_output = max(-self._max_movement, min(self._max_movement, tilt_output))
        
        return pan_output, tilt_output
    
    def _fast_pd_update(self, state: Dict, error: float, current_time: float, kp: float, kd: float) -> float:
        """Ultra-fast PD update."""
        dt = current_time - state['last_time']
        
        if dt <= 0:
            state['last_time'] = current_time
            return state['last_output']
        
        # PD control
        proportional = kp * error
        derivative = kd * (error - state['last_error']) / dt if dt > 0 else 0.0
        
        output = proportional + derivative
        
        # Update state
        state['last_error'] = error
        state['last_time'] = current_time
        state['last_output'] = output
        
        return output


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhanced_controller(mode: ConvergenceMode = ConvergenceMode.ADAPTIVE, 
                             ultra_fast: bool = False) -> Any:
    """
    Factory function to create enhanced PID controllers.
    
    Args:
        mode: Convergence mode for optimization
        ultra_fast: If True, creates ultra-fast controller with minimal features
        
    Returns:
        Enhanced PID controller instance
    """
    config = create_enhanced_config(mode)
    
    if ultra_fast:
        return UltraFastEnhancedController(config)
    else:
        return EnhancedPIDController(config)


if __name__ == "__main__":
    # Demonstrate enhanced controller
    config = create_enhanced_config(ConvergenceMode.ADAPTIVE)
    controller = EnhancedPIDController(config)
    
    print("🚀 Enhanced PID Controller Demo")
    config.print_summary()
    
    # Test controller response
    print(f"\nController Response Test:")
    test_errors = [(100, 50), (30, 20), (10, 5), (3, 2)]
    
    for i, (error_x, error_y) in enumerate(test_errors):
        pan_out, tilt_out = controller.update(error_x, error_y)
        print(f"Step {i+1}: Error=({error_x}, {error_y}) → Output=({pan_out:.2f}°, {tilt_out:.2f}°)")
    
    # Show performance stats
    stats = controller.get_performance_stats()
    print(f"\nPerformance: {stats['performance']['avg_compute_time_ms']:.2f}ms avg compute time") 