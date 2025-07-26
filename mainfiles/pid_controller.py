#!/usr/bin/env python3
"""
Optimized PID Controller Implementation for Face Tracking System

Key optimizations:
- Reduced computational overhead
- Minimal memory allocations  
- Optimized for real-time performance
- Unified controller architecture
- Configurable logging levels
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import logging
from enum import Enum

# Configure logging - optimized for performance
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO
logger = logging.getLogger(__name__)


class ControllerMode(Enum):
    """Controller operation modes."""
    PID = "pid"
    PD = "pd"


@dataclass
class PIDConfig:
    """Optimized configuration class for PID parameters."""
    
    # Controller Mode
    MODE: ControllerMode = ControllerMode.PD  # Default to PD for face tracking
    
    # PID Gains for Pan (X-axis / Joint1)
    PAN_KP: float = 0.02
    PAN_KI: float = 0.0  # Set to 0 for PD control
    PAN_KD: float = 0.15
    
    # PID Gains for Tilt (Y-axis / Joint4)
    TILT_KP: float = 0.02
    TILT_KI: float = 0.0  # Set to 0 for PD control
    TILT_KD: float = 0.15
    
    # Control Parameters
    DEAD_ZONE: float = 15.0
    MAX_MOVEMENT: float = 8.0
    
    # Anti-windup Protection
    INTEGRAL_MIN: float = -100.0
    INTEGRAL_MAX: float = 100.0
    
    # Safety Parameters
    SAFETY_TIMEOUT: float = 30.0
    MAX_TRACKING_TIME: float = 300.0
    
    # Conversion Parameters
    PIXELS_TO_DEGREES: float = 0.15
    
    # Gravity Compensation
    GRAVITY_COMPENSATION_JOINT1: float = 0.0
    GRAVITY_COMPENSATION_JOINT4: float = 2.5
    
    # Performance Settings
    ENABLE_DEBUG_LOGGING: bool = False  # Disable debug logging by default
    MIN_UPDATE_INTERVAL: float = 0.001  # Minimum time between updates (1ms)
    
    # Face Detection Thresholds
    MIN_FACE_SIZE: float = 50
    MAX_FACE_SIZE: float = 300
    MIN_FACE_CONFIDENCE: float = 0.7
    FACE_ASPECT_RATIO_MIN: float = 0.6
    FACE_ASPECT_RATIO_MAX: float = 1.8
    
    def __post_init__(self):
        """Post-initialization validation and optimization."""
        # Pre-compute dead zone in degrees for performance
        self._dead_zone_degrees = self.DEAD_ZONE * self.PIXELS_TO_DEGREES
        
        # Pre-compute direction multipliers
        self._pan_direction = 1.0
        self._tilt_direction = -1.0
        
        # Validate configuration
        if self.MODE == ControllerMode.PD:
            self.PAN_KI = 0.0
            self.TILT_KI = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mode': self.MODE.value,
            'pan_gains': {'Kp': self.PAN_KP, 'Ki': self.PAN_KI, 'Kd': self.PAN_KD},
            'tilt_gains': {'Kp': self.TILT_KP, 'Ki': self.TILT_KI, 'Kd': self.TILT_KD},
            'control': {
                'dead_zone': self.DEAD_ZONE,
                'max_movement': self.MAX_MOVEMENT,
                'pixels_to_degrees': self.PIXELS_TO_DEGREES
            },
            'performance': {
                'enable_debug_logging': self.ENABLE_DEBUG_LOGGING,
                'min_update_interval': self.MIN_UPDATE_INTERVAL
            }
        }


class OptimizedPIDController:
    """
    Highly optimized PID/PD controller for real-time face tracking.
    
    Key optimizations:
    - Conditional logging (disabled by default)
    - Minimal object creation
    - Pre-computed constants
    - Optimized update loop
    - Unified PD/PID implementation
    """
    
    def __init__(self, config: Optional[PIDConfig] = None):
        """Initialize optimized PID controller."""
        self.config = config or PIDConfig()
        
        # Pre-compute constants for performance
        self._dead_zone_degrees = self.config._dead_zone_degrees
        self._max_movement = self.config.MAX_MOVEMENT
        self._pixels_to_degrees = self.config.PIXELS_TO_DEGREES
        
        # Initialize controllers
        self._pan_controller = self._create_single_axis_controller(
            self.config.PAN_KP, self.config.PAN_KI, self.config.PAN_KD
        )
        self._tilt_controller = self._create_single_axis_controller(
            self.config.TILT_KP, self.config.TILT_KI, self.config.TILT_KD
        )
        
        # Performance monitoring
        self._last_update_time = time.time()
        self._update_count = 0
        self._total_error = 0.0
        
        # Thread safety (minimal locking)
        self._lock = threading.RLock()  # Reentrant lock for better performance
        
        if self.config.ENABLE_DEBUG_LOGGING:
            logger.info(f"Optimized PID Controller initialized: mode={self.config.MODE.value}")
    
    def _create_single_axis_controller(self, Kp: float, Ki: float, Kd: float) -> Dict[str, float]:
        """Create optimized single-axis controller state."""
        return {
            'Kp': Kp,
            'Ki': Ki,
            'Kd': Kd,
            'last_error': 0.0,
            'integral': 0.0,
            'last_time': time.time(),
            'last_output': 0.0
        }
    
    def update(self, error_x: float, error_y: float) -> Tuple[float, float]:
        """
        Optimized dual-axis update with minimal overhead.
        
        Args:
            error_x: X-axis error (pixels from center)
            error_y: Y-axis error (pixels from center)
            
        Returns:
            tuple: (pan_output, tilt_output) in degrees
        """
        # Performance check - skip if too frequent updates
        current_time = time.time()
        if current_time - self._last_update_time < self.config.MIN_UPDATE_INTERVAL:
            return self._pan_controller['last_output'], self._tilt_controller['last_output']
        
        with self._lock:
            # Apply direction corrections and convert to degrees
            error_x_degrees = error_x * self.config._pan_direction * self._pixels_to_degrees
            error_y_degrees = error_y * self.config._tilt_direction * self._pixels_to_degrees
            
            # Apply dead zone filtering
            if abs(error_x_degrees) < self._dead_zone_degrees:
                error_x_degrees = 0.0
            if abs(error_y_degrees) < self._dead_zone_degrees:
                error_y_degrees = 0.0
            
            # Update controllers
            pan_output = self._update_single_axis(self._pan_controller, error_x_degrees, current_time)
            tilt_output = self._update_single_axis(self._tilt_controller, error_y_degrees, current_time)
            
            # Update performance metrics
            self._last_update_time = current_time
            self._update_count += 1
            self._total_error += abs(error_x_degrees) + abs(error_y_degrees)
            
            # Conditional debug logging
            if self.config.ENABLE_DEBUG_LOGGING:
                logger.debug(f"Update: error=({error_x:.1f}, {error_y:.1f})px → "
                           f"output=({pan_output:.2f}, {tilt_output:.2f})°")
            
            return pan_output, tilt_output
    
    def _update_single_axis(self, controller: Dict[str, float], error: float, current_time: float) -> float:
        """Optimized single-axis controller update."""
        dt = current_time - controller['last_time']
        
        # Handle very small time steps
        if dt <= 0.0:
            controller['last_time'] = current_time
            return controller['last_output']
        
        # Proportional term
        proportional = controller['Kp'] * error
        
        # Integral term (only for PID mode)
        integral = 0.0
        if self.config.MODE == ControllerMode.PID and controller['Ki'] != 0.0:
            controller['integral'] += error * dt
            controller['integral'] = max(
                self.config.INTEGRAL_MIN,
                min(self.config.INTEGRAL_MAX, controller['integral'])
            )
            integral = controller['Ki'] * controller['integral']
        
        # Derivative term
        derivative = 0.0
        if dt > 0.0:
            derivative = controller['Kd'] * (error - controller['last_error']) / dt
        
        # Calculate output
        output = proportional + integral + derivative
        
        # Apply output limits
        if abs(output) > self._max_movement:
            output = self._max_movement if output > 0 else -self._max_movement
        
        # Update state
        controller['last_error'] = error
        controller['last_time'] = current_time
        controller['last_output'] = output
        
        return output
    
    def update_gains(self, pan_gains: Optional[Dict[str, float]] = None,
                    tilt_gains: Optional[Dict[str, float]] = None) -> None:
        """Update controller gains with minimal locking."""
        with self._lock:
            if pan_gains:
                self._pan_controller.update(pan_gains)
            if tilt_gains:
                self._tilt_controller.update(tilt_gains)
    
    def reset(self) -> None:
        """Reset controller state."""
        with self._lock:
            self._pan_controller = self._create_single_axis_controller(
                self.config.PAN_KP, self.config.PAN_KI, self.config.PAN_KD
            )
            self._tilt_controller = self._create_single_axis_controller(
                self.config.TILT_KP, self.config.TILT_KI, self.config.TILT_KD
            )
            self._update_count = 0
            self._total_error = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimized controller statistics."""
        with self._lock:
            avg_error = self._total_error / max(1, self._update_count)
            
            return {
                'mode': self.config.MODE.value,
                'performance': {
                    'update_count': self._update_count,
                    'average_error': avg_error,
                    'total_error': self._total_error,
                    'update_rate': self._update_count / max(1, time.time() - self._pan_controller['last_time'])
                },
                'pan_controller': {
                    'gains': {'Kp': self._pan_controller['Kp'], 'Ki': self._pan_controller['Ki'], 'Kd': self._pan_controller['Kd']},
                    'last_error': self._pan_controller['last_error'],
                    'last_output': self._pan_controller['last_output']
                },
                'tilt_controller': {
                    'gains': {'Kp': self._tilt_controller['Kp'], 'Ki': self._tilt_controller['Ki'], 'Kd': self._tilt_controller['Kd']},
                    'last_error': self._tilt_controller['last_error'],
                    'last_output': self._tilt_controller['last_output']
                }
            }
    
    def set_debug_logging(self, enabled: bool) -> None:
        """Enable/disable debug logging."""
        self.config.ENABLE_DEBUG_LOGGING = enabled


class FastPIDController:
    """
    Ultra-fast PID controller for maximum performance.
    
    This controller sacrifices some features for maximum speed:
    - No logging
    - No thread safety
    - Minimal error checking
    - Pre-allocated buffers
    """
    
    def __init__(self, Kp: float, Kd: float, dead_zone: float = 0.0, max_output: float = float('inf')):
        """Initialize fast controller."""
        self.Kp = Kp
        self.Kd = Kd
        self.dead_zone = dead_zone
        self.max_output = max_output
        
        # Pre-allocated state
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_output = 0.0
    
    def update(self, error: float) -> float:
        """Ultra-fast update with minimal overhead."""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0.0:
            self.last_time = current_time
            return self.last_output
        
        # Apply dead zone
        if abs(error) < self.dead_zone:
            error = 0.0
        
        # PD control only
        proportional = self.Kp * error
        derivative = self.Kd * (error - self.last_error) / dt if dt > 0 else 0.0
        
        output = proportional + derivative
        
        # Apply limits
        if abs(output) > self.max_output:
            output = self.max_output if output > 0 else -self.max_output
        
        # Update state
        self.last_error = error
        self.last_time = current_time
        self.last_output = output
        
        return output


# =============================================================================
# CONFIGURATION MANAGEMENT (Optimized)
# =============================================================================

def load_config_from_file(config_path: str) -> PIDConfig:
    """Load configuration with error handling."""
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            config = PIDConfig()
            
            # Apply configuration updates
            if 'mode' in config_dict:
                config.MODE = ControllerMode(config_dict['mode'])
            
            if 'pan_gains' in config_dict:
                gains = config_dict['pan_gains']
                config.PAN_KP = gains.get('Kp', config.PAN_KP)
                config.PAN_KI = gains.get('Ki', config.PAN_KI)
                config.PAN_KD = gains.get('Kd', config.PAN_KD)
            
            if 'tilt_gains' in config_dict:
                gains = config_dict['tilt_gains']
                config.TILT_KP = gains.get('Kp', config.TILT_KP)
                config.TILT_KI = gains.get('Ki', config.TILT_KI)
                config.TILT_KD = gains.get('Kd', config.TILT_KD)
            
            if 'control' in config_dict:
                control = config_dict['control']
                config.DEAD_ZONE = control.get('dead_zone', config.DEAD_ZONE)
                config.MAX_MOVEMENT = control.get('max_movement', config.MAX_MOVEMENT)
                config.PIXELS_TO_DEGREES = control.get('pixels_to_degrees', config.PIXELS_TO_DEGREES)
            
            if 'performance' in config_dict:
                perf = config_dict['performance']
                config.ENABLE_DEBUG_LOGGING = perf.get('enable_debug_logging', config.ENABLE_DEBUG_LOGGING)
                config.MIN_UPDATE_INTERVAL = perf.get('min_update_interval', config.MIN_UPDATE_INTERVAL)
            
            # Trigger post-init
            config.__post_init__()
            return config
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return PIDConfig()
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        return PIDConfig()


def save_config_to_file(config: PIDConfig, config_path: str) -> bool:
    """Save configuration with error handling."""
    import json
    from pathlib import Path
    
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        return False


# =============================================================================
# PERFORMANCE BENCHMARKING
# =============================================================================

def benchmark_controllers():
    """Benchmark different controller implementations."""
    import timeit
    
    print("🏃‍♂️ Controller Performance Benchmark")
    print("=" * 50)
    
    # Test configurations
    config = PIDConfig()
    config.ENABLE_DEBUG_LOGGING = False  # Disable logging for fair comparison
    
    # Create controllers
    optimized_pid = OptimizedPIDController(config)
    fast_pid = FastPIDController(Kp=0.02, Kd=0.15, dead_zone=15.0, max_output=8.0)
    
    # Benchmark parameters
    test_errors = [(50, 30), (30, 20), (10, 5), (0, 0), (-15, -10), (-5, -2), (0, 0)]
    iterations = 10000
    
    # Benchmark optimized controller
    def test_optimized():
        for _ in range(iterations):
            for error_x, error_y in test_errors:
                optimized_pid.update(error_x, error_y)
    
    # Benchmark fast controller
    def test_fast():
        for _ in range(iterations):
            for error_x, error_y in test_errors:
                fast_pid.update(error_x)
                fast_pid.update(error_y)
    
    # Run benchmarks
    optimized_time = timeit.timeit(test_optimized, number=1)
    fast_time = timeit.timeit(test_fast, number=1)
    
    print(f"Optimized PID Controller: {optimized_time:.4f}s")
    print(f"Fast PID Controller: {fast_time:.4f}s")
    print(f"Speedup: {optimized_time/fast_time:.2f}x")
    
    # Test accuracy
    optimized_pid.reset()
    fast_pid = FastPIDController(Kp=0.02, Kd=0.15, dead_zone=15.0, max_output=8.0)
    
    print("\n📊 Accuracy Test:")
    for i, (error_x, error_y) in enumerate(test_errors):
        opt_pan, opt_tilt = optimized_pid.update(error_x, error_y)
        fast_pan = fast_pid.update(error_x)
        fast_tilt = fast_pid.update(error_y)
        
        print(f"Step {i+1}: Optimized=({opt_pan:.3f}, {opt_tilt:.3f}), "
              f"Fast=({fast_pan:.3f}, {fast_tilt:.3f})")


if __name__ == "__main__":
    benchmark_controllers() 