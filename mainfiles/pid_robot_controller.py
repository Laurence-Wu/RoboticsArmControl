#!/usr/bin/env python3
"""
PID Robot Controller Integration

This module integrates PID control output with the SimpleRobotController to provide
smooth, safe face tracking movements. It handles pixel-to-angle conversion,
joint limit checking, movement clamping, and dead zone filtering.

Features:
- Integration with existing SimpleRobotController
- Pixel-to-angle conversion for camera coordinates
- Safe movement execution with joint limit checking
- Movement clamping to prevent excessive movements
- Dead zone filtering to prevent micro-movements
- Thread-safe operation
- Comprehensive error handling and logging

Usage:
    from pid_robot_controller import PIDRobotController
    
    # Create controller
    controller = PIDRobotController()
    
    # Execute PID-based movement
    success = controller.execute_pid_movement(error_x=50, error_y=-30)
"""

import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Import existing components
try:
    from simple_robot_control import SimpleRobotController
    from config import MOTOR_CONFIG, ROBOT_CONFIG
    from pid_controller import DualAxisPIDController, PIDConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure simple_robot_control.py, config.py, and pid_controller.py are available")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """States for the PID robot controller."""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRACKING = "tracking"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits for robot movement."""
    max_movement_per_update: float = 8.0  # degrees
    max_velocity: float = 50.0  # degrees per second
    joint_limit_buffer: float = 5.0  # degrees buffer from joint limits
    emergency_stop_threshold: float = 45.0  # degrees - emergency stop if exceeded
    
    # Camera to robot coordinate conversion
    pixels_to_degrees_x: float = 0.1  # X-axis conversion factor
    pixels_to_degrees_y: float = 0.1  # Y-axis conversion factor
    
    # Dead zone filtering
    pixel_dead_zone: float = 15.0  # pixels
    angle_dead_zone: float = 1.5   # degrees


@dataclass
class MovementCommand:
    """Represents a movement command with safety validation."""
    pan_angle: float  # Joint1 movement in degrees
    tilt_angle: float  # Joint4 movement in degrees
    confidence: float = 1.0  # Confidence in the movement (0-1)
    timestamp: float = 0.0
    is_safe: bool = True
    safety_notes: str = ""
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PIDRobotController:
    """
    PID Robot Controller that integrates PID output with SimpleRobotController.
    
    This class provides safe, smooth robot movements based on PID control output,
    with comprehensive safety checks and error handling.
    """
    
    def __init__(self, config: Optional[PIDConfig] = None, 
                 safety_limits: Optional[SafetyLimits] = None,
                 simulation_mode: bool = False):
        """
        Initialize PID Robot Controller.
        
        Args:
            config: PID configuration (uses defaults if None)
            safety_limits: Safety limits configuration (uses defaults if None)
            simulation_mode: If True, operates in simulation mode without hardware
        """
        # Configuration
        self.config = config or PIDConfig()
        self.safety_limits = safety_limits or SafetyLimits()
        self.simulation_mode = simulation_mode
        
        # Robot controller
        self.robot = None
        if not simulation_mode:
            self.robot = SimpleRobotController()
        
        # PID controllers
        self.dual_pid = DualAxisPIDController(self.config)
        
        # State management
        self.state = ControllerState.DISCONNECTED
        self.current_positions = {"joint1": 0.0, "joint4": 0.0}
        self.last_movement_time = time.time()
        self.total_movements = 0
        self.emergency_stop_active = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'movements_executed': 0,
            'movements_blocked': 0,
            'safety_violations': 0,
            'total_error_x': 0.0,
            'total_error_y': 0.0,
            'max_movement_x': 0.0,
            'max_movement_y': 0.0
        }
        
        logger.info(f"PID Robot Controller initialized (simulation={simulation_mode})")
    
    def connect(self) -> bool:
        """
        Connect to the robot hardware.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        with self._lock:
            if self.simulation_mode:
                self.state = ControllerState.CONNECTED
                logger.info("Connected to robot (simulation mode)")
                return True
            
            if self.robot is None:
                logger.error("Robot controller not initialized")
                return False
            
            try:
                success = self.robot.connect()
                if success:
                    self.state = ControllerState.CONNECTED
                    # Read initial positions
                    self._update_current_positions()
                    logger.info("Connected to robot hardware")
                else:
                    self.state = ControllerState.ERROR
                    logger.error("Failed to connect to robot hardware")
                
                return success
                
            except Exception as e:
                self.state = ControllerState.ERROR
                logger.error(f"Error connecting to robot: {e}")
                return False
    
    def disconnect(self) -> None:
        """Disconnect from the robot hardware."""
        with self._lock:
            if not self.simulation_mode and self.robot:
                self.robot.disconnect()
            
            self.state = ControllerState.DISCONNECTED
            logger.info("Disconnected from robot")
    
    def _update_current_positions(self) -> None:
        """Update current joint positions from robot."""
        if self.simulation_mode or not self.robot:
            return
        
        try:
            self.robot.read_positions()
            # Update our tracking of pan/tilt positions
            self.current_positions["joint1"] = self.robot.current_positions.get("joint1", 0.0)
            self.current_positions["joint4"] = self.robot.current_positions.get("joint4", 0.0)
        except Exception as e:
            logger.warning(f"Failed to update current positions: {e}")
    
    def pixel_to_angle_conversion(self, error_x: float, error_y: float) -> Tuple[float, float]:
        """
        Convert pixel errors to angle commands.
        
        Args:
            error_x: X-axis error in pixels (positive = face to the right)
            error_y: Y-axis error in pixels (positive = face down)
            
        Returns:
            tuple: (pan_angle, tilt_angle) in degrees
        """
        # Apply dead zone filtering in pixel space
        filtered_error_x = error_x if abs(error_x) > self.safety_limits.pixel_dead_zone else 0.0
        filtered_error_y = error_y if abs(error_y) > self.safety_limits.pixel_dead_zone else 0.0
        
        # Convert to angles
        # Note: Pan (joint1) should move in the same direction as error to center the face
        # Tilt (joint4) should move in the same direction as error to center the face
        pan_angle = filtered_error_x * self.safety_limits.pixels_to_degrees_x
        tilt_angle = filtered_error_y * self.safety_limits.pixels_to_degrees_y
        
        logger.info(f"Pixel conversion: ({error_x:.1f}, {error_y:.1f})px → "
                    f"({pan_angle:.2f}, {tilt_angle:.2f})°")
        
        return pan_angle, tilt_angle
    
    def validate_movement_safety(self, pan_movement: float, tilt_movement: float) -> MovementCommand:
        """
        Validate movement for safety and apply necessary limits.
        
        Args:
            pan_movement: Desired pan movement in degrees
            tilt_movement: Desired tilt movement in degrees
            
        Returns:
            MovementCommand: Validated and potentially modified movement command
        """
        safety_notes = []
        is_safe = True
        
        # Check for emergency stop conditions
        if (abs(pan_movement) > self.safety_limits.emergency_stop_threshold or
            abs(tilt_movement) > self.safety_limits.emergency_stop_threshold):
            safety_notes.append("Emergency stop: movement too large")
            is_safe = False
            pan_movement = 0.0
            tilt_movement = 0.0
        
        # Apply movement limits
        original_pan = pan_movement
        original_tilt = tilt_movement
        
        pan_movement = max(-self.safety_limits.max_movement_per_update,
                          min(self.safety_limits.max_movement_per_update, pan_movement))
        tilt_movement = max(-self.safety_limits.max_movement_per_update,
                           min(self.safety_limits.max_movement_per_update, tilt_movement))
        
        if abs(original_pan - pan_movement) > 0.01:
            safety_notes.append(f"Pan movement clamped: {original_pan:.2f}° → {pan_movement:.2f}°")
        
        if abs(original_tilt - tilt_movement) > 0.01:
            safety_notes.append(f"Tilt movement clamped: {original_tilt:.2f}° → {tilt_movement:.2f}°")
        
        # Apply dead zone filtering in angle space
        if abs(pan_movement) < self.safety_limits.angle_dead_zone:
            pan_movement = 0.0
        
        if abs(tilt_movement) < self.safety_limits.angle_dead_zone:
            tilt_movement = 0.0
        
        # Check joint limits
        current_pan = self.current_positions.get("joint1", 0.0)
        current_tilt = self.current_positions.get("joint4", 0.0)
        
        new_pan = current_pan + pan_movement
        new_tilt = current_tilt + tilt_movement
        
        # Get joint limits from config
        pan_limits = MOTOR_CONFIG.JOINT_LIMITS.get("joint1", (-180, 180))
        tilt_limits = MOTOR_CONFIG.JOINT_LIMITS.get("joint4", (-90, 90))
        
        # Apply buffer to limits
        pan_min = pan_limits[0] + self.safety_limits.joint_limit_buffer
        pan_max = pan_limits[1] - self.safety_limits.joint_limit_buffer
        tilt_min = tilt_limits[0] + self.safety_limits.joint_limit_buffer
        tilt_max = tilt_limits[1] - self.safety_limits.joint_limit_buffer
        
        # Check and clamp to limits
        if new_pan < pan_min:
            pan_movement = pan_min - current_pan
            safety_notes.append(f"Pan limited by minimum joint limit")
        elif new_pan > pan_max:
            pan_movement = pan_max - current_pan
            safety_notes.append(f"Pan limited by maximum joint limit")
        
        if new_tilt < tilt_min:
            tilt_movement = tilt_min - current_tilt
            safety_notes.append(f"Tilt limited by minimum joint limit")
        elif new_tilt > tilt_max:
            tilt_movement = tilt_max - current_tilt
            safety_notes.append(f"Tilt limited by maximum joint limit")
        
        return MovementCommand(
            pan_angle=pan_movement,
            tilt_angle=tilt_movement,
            is_safe=is_safe,
            safety_notes="; ".join(safety_notes) if safety_notes else "OK"
        )
    
    def execute_safe_movement(self, movement_cmd: MovementCommand) -> bool:
        """
        Execute a validated movement command.
        
        Args:
            movement_cmd: Validated movement command
            
        Returns:
            bool: True if movement executed successfully, False otherwise
        """
        if not movement_cmd.is_safe:
            logger.warning(f"Unsafe movement blocked: {movement_cmd.safety_notes}")
            self.stats['movements_blocked'] += 1
            return False
        
        # Check if movement is significant enough to execute
        if (abs(movement_cmd.pan_angle) < 0.1 and 
            abs(movement_cmd.tilt_angle) < 0.1):
            logger.debug("Movement too small, skipping")
            return True
        
        try:
            if self.simulation_mode:
                # Simulate movement
                self.current_positions["joint1"] += movement_cmd.pan_angle
                self.current_positions["joint4"] += movement_cmd.tilt_angle
                logger.info(f"Simulated movement: pan={movement_cmd.pan_angle:.2f}°, "
                           f"tilt={movement_cmd.tilt_angle:.2f}°")
            else:
                # Execute real movement
                if not self.robot or self.state not in [ControllerState.CONNECTED, ControllerState.TRACKING]:
                    logger.error("Robot not connected")
                    return False
                
                # Calculate target positions
                current_pan = self.current_positions.get("joint1", 0.0)
                current_tilt = self.current_positions.get("joint4", 0.0)
                
                target_pan = current_pan + movement_cmd.pan_angle
                target_tilt = current_tilt + movement_cmd.tilt_angle
                
                # Execute movements
                success = True
                
                if abs(movement_cmd.pan_angle) > 0.1:
                    if not self.robot.move_joint("joint1", target_pan, speed=100):
                        logger.error("Failed to move joint1 (pan)")
                        success = False
                
                if abs(movement_cmd.tilt_angle) > 0.1:
                    if not self.robot.move_joint("joint4", target_tilt, speed=100):
                        logger.error("Failed to move joint4 (tilt)")
                        success = False
                
                if success:
                    # Update our position tracking
                    self.current_positions["joint1"] = target_pan
                    self.current_positions["joint4"] = target_tilt
                    logger.debug(f"Executed movement: pan={movement_cmd.pan_angle:.2f}°, "
                               f"tilt={movement_cmd.tilt_angle:.2f}°")
                else:
                    return False
            
            # Update statistics
            self.stats['movements_executed'] += 1
            self.stats['max_movement_x'] = max(self.stats['max_movement_x'], 
                                             abs(movement_cmd.pan_angle))
            self.stats['max_movement_y'] = max(self.stats['max_movement_y'], 
                                             abs(movement_cmd.tilt_angle))
            self.last_movement_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing movement: {e}")
            self.stats['safety_violations'] += 1
            return False
    
    def execute_pid_movement(self, error_x: float, error_y: float) -> bool:
        """
        Execute PID-based movement from pixel errors.
        
        This is the main interface method that combines all the safety and control logic.
        
        Args:
            error_x: X-axis error in pixels (positive = face to the right)
            error_y: Y-axis error in pixels (positive = face down)
            
        Returns:
            bool: True if movement executed successfully, False otherwise
        """
        with self._lock:
            # Check if we're in a valid state
            if self.emergency_stop_active:
                logger.warning("Emergency stop active - movement blocked")
                return False
            
            if self.state not in [ControllerState.CONNECTED, ControllerState.TRACKING]:
                logger.warning(f"Invalid state for movement: {self.state}")
                return False
            
            # Update state to tracking
            self.state = ControllerState.TRACKING
            
            # Update statistics
            self.stats['total_error_x'] += abs(error_x)
            self.stats['total_error_y'] += abs(error_y)
            
            try:
                # Get PID output
                pan_output, tilt_output = self.dual_pid.update(error_x, error_y)
                
                # Validate movement safety
                movement_cmd = self.validate_movement_safety(pan_output, tilt_output)
                
                # Log movement details
                if movement_cmd.safety_notes != "OK":
                    logger.info(f"Movement safety: {movement_cmd.safety_notes}")
                
                # Execute movement
                success = self.execute_safe_movement(movement_cmd)
                
                if success:
                    logger.debug(f"PID movement executed: error=({error_x:.1f}, {error_y:.1f})px, "
                               f"movement=({movement_cmd.pan_angle:.2f}, {movement_cmd.tilt_angle:.2f})°")
                
                return success
                
            except Exception as e:
                logger.error(f"Error in PID movement execution: {e}")
                self.state = ControllerState.ERROR
                return False
    
    def emergency_stop(self) -> None:
        """Activate emergency stop - blocks all movements."""
        with self._lock:
            self.emergency_stop_active = True
            self.state = ControllerState.EMERGENCY_STOP
            logger.warning("EMERGENCY STOP ACTIVATED")
    
    def reset_emergency_stop(self) -> None:
        """Reset emergency stop condition."""
        with self._lock:
            self.emergency_stop_active = False
            if self.state == ControllerState.EMERGENCY_STOP:
                self.state = ControllerState.CONNECTED
            logger.info("Emergency stop reset")
    
    def reset_pid_controllers(self) -> None:
        """Reset PID controllers to clear accumulated state."""
        self.dual_pid.reset()
        logger.info("PID controllers reset")
    
    def update_pid_config(self, new_config: PIDConfig) -> None:
        """
        Update PID configuration.
        
        Args:
            new_config: New PID configuration
        """
        self.config = new_config
        self.dual_pid.update_config(new_config)
        
        # Update safety limits from config
        self.safety_limits.pixel_dead_zone = new_config.DEAD_ZONE
        self.safety_limits.max_movement_per_update = new_config.MAX_MOVEMENT
        self.safety_limits.pixels_to_degrees_x = new_config.PIXELS_TO_DEGREES
        self.safety_limits.pixels_to_degrees_y = new_config.PIXELS_TO_DEGREES
        
        logger.info("PID configuration updated")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current controller status and statistics.
        
        Returns:
            dict: Controller status and statistics
        """
        with self._lock:
            return {
                'state': self.state.value,
                'simulation_mode': self.simulation_mode,
                'emergency_stop_active': self.emergency_stop_active,
                'current_positions': self.current_positions.copy(),
                'last_movement_time': self.last_movement_time,
                'statistics': self.stats.copy(),
                'pid_stats': self.dual_pid.get_stats(),
                'safety_limits': {
                    'max_movement_per_update': self.safety_limits.max_movement_per_update,
                    'pixel_dead_zone': self.safety_limits.pixel_dead_zone,
                    'angle_dead_zone': self.safety_limits.angle_dead_zone,
                    'joint_limit_buffer': self.safety_limits.joint_limit_buffer
                }
            }
    
    def __str__(self) -> str:
        """String representation of the controller."""
        return (f"PIDRobotController(state={self.state.value}, "
                f"simulation={self.simulation_mode}, "
                f"movements={self.stats['movements_executed']})")


# =============================================================================
# MOCK ROBOT CONTROLLER FOR TESTING
# =============================================================================

class MockRobotController:
    """Mock robot controller for testing without hardware."""
    
    def __init__(self):
        """Initialize mock controller."""
        self.positions = {"joint1": 0.0, "joint4": 0.0, "joint3": -90.0, 
                         "joint4": 0.0, "joint5": 0.0, "joint6": 0.0, "gripper": 0.0}
        self.connected = False
        self.current_positions = self.positions.copy()
        
        logger.info("Mock robot controller initialized")
    
    def connect(self) -> bool:
        """Mock connection."""
        self.connected = True
        logger.info("Mock robot connected")
        return True
    
    def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False
        logger.info("Mock robot disconnected")
    
    def move_joint(self, joint: str, angle: float, speed: int = 100) -> bool:
        """Mock joint movement."""
        if not self.connected:
            return False
        
        if joint in self.positions:
            self.positions[joint] = angle
            self.current_positions[joint] = angle
            logger.debug(f"Mock moved {joint} to {angle}°")
            return True
        
        return False
    
    def read_positions(self) -> None:
        """Mock position reading."""
        logger.debug(f"Mock positions: {self.current_positions}")


# =============================================================================
# TESTING AND EXAMPLE USAGE
# =============================================================================

def test_pid_robot_controller():
    """Test the PID robot controller functionality."""
    print("🧪 Testing PID Robot Controller")
    print("=" * 50)
    
    # Test in simulation mode
    print("\n1. Testing Simulation Mode:")
    controller = PIDRobotController(simulation_mode=True)
    
    # Connect
    if controller.connect():
        print("   ✅ Connected to simulated robot")
    
    # Test movements
    test_errors = [
        (50, 30),   # Face to right and down
        (25, 15),   # Moving toward center
        (10, 5),    # Close to center
        (0, 0),     # Centered
        (-20, -10), # Face to left and up
        (0, 0)      # Centered again
    ]
    
    print("   Testing PID movements:")
    for i, (error_x, error_y) in enumerate(test_errors):
        success = controller.execute_pid_movement(error_x, error_y)
        status = "✅" if success else "❌"
        print(f"   {status} Step {i+1}: error=({error_x:3.0f}, {error_y:3.0f})px")
        time.sleep(0.1)
    
    # Test safety limits
    print("\n2. Testing Safety Limits:")
    
    # Test large movement (should be clamped)
    success = controller.execute_pid_movement(500, -300)
    print(f"   {'✅' if success else '❌'} Large movement test (should be clamped)")
    
    # Test emergency stop
    controller.emergency_stop()
    success = controller.execute_pid_movement(10, 10)
    print(f"   {'❌' if not success else '✅'} Emergency stop test (should block movement)")
    
    # Reset emergency stop
    controller.reset_emergency_stop()
    success = controller.execute_pid_movement(10, 10)
    print(f"   {'✅' if success else '❌'} Emergency stop reset test")
    
    # Get status
    print("\n3. Controller Status:")
    status = controller.get_status()
    print(f"   State: {status['state']}")
    print(f"   Movements executed: {status['statistics']['movements_executed']}")
    print(f"   Movements blocked: {status['statistics']['movements_blocked']}")
    print(f"   Current positions: {status['current_positions']}")
    
    controller.disconnect()
    print("   ✅ Disconnected from simulated robot")
    
    print("\n✅ All PID robot controller tests completed!")


if __name__ == "__main__":
    """Example usage and testing."""
    test_pid_robot_controller()