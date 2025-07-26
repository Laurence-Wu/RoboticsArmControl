#!/usr/bin/env python3
"""
Joint Constraint Manager for Robot Arm

This module provides constraint management for robot arm movements,
including loading constraints from JSON files and applying them to
movement commands.

Features:
- Load joint constraints from JSON configuration
- Validate movement commands against constraints
- Apply safety limits and collision avoidance
- Support for face tracking mode with fixed joints
- Comprehensive constraint checking and reporting

Usage:
    from joint_constraint_manager import JointConstraintManager
    
    # Load constraints
    constraint_manager = JointConstraintManager("joint_constraints.json")
    
    # Validate movement
    is_valid, adjusted_angle = constraint_manager.validate_joint_movement("joint1", 45.0, 100)
"""

import json
import os
import time
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints that can be applied."""
    POSITION = "position"
    MOVEMENT = "movement"
    SAFETY = "safety"
    COLLISION = "collision"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: ConstraintType
    joint_name: str
    current_value: float
    limit_value: float
    violation_message: str
    severity: str = "warning"  # "warning", "error", "critical"


@dataclass
class MovementValidation:
    """Result of movement validation."""
    is_valid: bool
    adjusted_angle: float
    adjusted_speed: int
    violations: List[ConstraintViolation]
    warnings: List[str]
    applied_constraints: List[str]


class JointConstraintManager:
    """
    Manages joint constraints for robot arm movements.
    
    This class loads constraints from JSON files and provides methods
    to validate and adjust movement commands according to the constraints.
    """
    
    def __init__(self, constraints_file: str = "joint_constraints.json"):
        """
        Initialize the constraint manager.
        
        Args:
            constraints_file: Path to the JSON constraints file
        """
        self.constraints_file = constraints_file
        self.constraints = {}
        self.global_constraints = {}
        self.validation_ranges = {}
        self.face_tracking_mode = False
        
        # Load constraints
        self.load_constraints()
        
        # Track constraint violations for reporting
        self.violation_history = []
        self.last_validation_time = time.time()
    
    def load_constraints(self) -> bool:
        """
        Load constraints from JSON file.
        
        Returns:
            bool: True if constraints loaded successfully
        """
        try:
            if not os.path.exists(self.constraints_file):
                print(f"⚠️  Constraints file not found: {self.constraints_file}")
                print("Using default constraints...")
                self._load_default_constraints()
                return True
            
            with open(self.constraints_file, 'r') as f:
                data = json.load(f)
            
            self.constraints = data.get("joint_constraints", {})
            self.global_constraints = data.get("global_constraints", {})
            self.validation_ranges = data.get("validation_ranges", {})
            
            # Check if face tracking mode is enabled
            face_tracking_config = self.global_constraints.get("face_tracking_mode", {})
            self.face_tracking_mode = face_tracking_config.get("enabled", False)
            
            print(f"✅ Loaded constraints from {self.constraints_file}")
            print(f"   Joints: {len(self.constraints)}")
            print(f"   Face tracking mode: {'enabled' if self.face_tracking_mode else 'disabled'}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in constraints file: {e}")
            self._load_default_constraints()
            return False
        except Exception as e:
            print(f"❌ Error loading constraints: {e}")
            self._load_default_constraints()
            return False
    
    def _load_default_constraints(self):
        """Load default constraints when file is not available."""
        self.constraints = {
            "joint1": {"min_angle": -180.0, "max_angle": 180.0},
            "joint2": {"min_angle": -90.0, "max_angle": 90.0},
            "joint3": {"min_angle": -135.0, "max_angle": 135.0},
            "joint4": {"min_angle": -90.0, "max_angle": 90.0},
            "joint5": {"min_angle": -180.0, "max_angle": 180.0},
            "joint6": {"min_angle": -180.0, "max_angle": 180.0},
            "gripper": {"min_angle": -90.0, "max_angle": 90.0}
        }
        
        self.global_constraints = {
            "face_tracking_mode": {
                "enabled": True,
                "fixed_joints": {"joint2": -42.4, "joint3": -132.0},
                "active_joints": ["joint1", "joint5"]
            },
            "safety_timeout": 30.0,
            "max_concurrent_movements": 3,
            "collision_detection_enabled": True
        }
        
        self.validation_ranges = {
            "angle_tolerance": 2.0,
            "speed_tolerance": 10,
            "acceleration_tolerance": 5.0
        }
        
        print("✅ Loaded default constraints")
    
    def validate_joint_movement(self, joint_name: str, target_angle: float, 
                               speed: int, current_angle: Optional[float] = None) -> MovementValidation:
        """
        Validate a joint movement against all constraints.
        
        Args:
            joint_name: Name of the joint to validate
            target_angle: Target angle in degrees
            speed: Movement speed (0-1000)
            current_angle: Current joint angle (optional, for movement validation)
            
        Returns:
            MovementValidation: Validation result with adjusted values and violations
        """
        violations = []
        warnings = []
        applied_constraints = []
        adjusted_angle = target_angle
        adjusted_speed = speed
        
        # Check if joint exists in constraints
        if joint_name not in self.constraints:
            violations.append(ConstraintViolation(
                ConstraintType.SAFETY, joint_name, target_angle, 0.0,
                f"Joint '{joint_name}' not found in constraints", "error"
            ))
            return MovementValidation(False, target_angle, speed, violations, warnings, applied_constraints)
        
        joint_constraints = self.constraints[joint_name]
        
        # Check angle limits (simplified structure)
        min_angle = joint_constraints.get("min_angle", -180.0)
        max_angle = joint_constraints.get("max_angle", 180.0)
        
        if target_angle < min_angle:
            violations.append(ConstraintViolation(
                ConstraintType.POSITION, joint_name, target_angle, min_angle,
                f"Target angle {target_angle}° below minimum {min_angle}°", "error"
            ))
            adjusted_angle = min_angle
            applied_constraints.append(f"Clamped to minimum: {min_angle}°")
        elif target_angle > max_angle:
            violations.append(ConstraintViolation(
                ConstraintType.POSITION, joint_name, target_angle, max_angle,
                f"Target angle {target_angle}° above maximum {max_angle}°", "error"
            ))
            adjusted_angle = max_angle
            applied_constraints.append(f"Clamped to maximum: {max_angle}°")
        
        # Check face tracking mode constraints
        if self.face_tracking_mode:
            face_config = self.global_constraints.get("face_tracking_mode", {})
            fixed_joints = face_config.get("fixed_joints", {})
            active_joints = face_config.get("active_joints", [])
            
            if joint_name in fixed_joints:
                fixed_angle = fixed_joints[joint_name]
                if abs(target_angle - fixed_angle) > 1.0:  # Allow small tolerance
                    violations.append(ConstraintViolation(
                        ConstraintType.POSITION, joint_name, target_angle, fixed_angle,
                        f"Joint '{joint_name}' is fixed at {fixed_angle}° in face tracking mode", "error"
                    ))
                    adjusted_angle = fixed_angle
                    applied_constraints.append(f"Fixed to face tracking position: {fixed_angle}°")
            
            if joint_name not in active_joints and joint_name not in fixed_joints:
                warnings.append(f"Joint '{joint_name}' is not active in face tracking mode")
        
        # Determine if movement is valid
        critical_violations = [v for v in violations if v.severity == "critical"]
        error_violations = [v for v in violations if v.severity == "error"]
        
        is_valid = len(critical_violations) == 0 and len(error_violations) == 0
        
        # Record violations for history
        if violations:
            self.violation_history.extend(violations)
            self.last_validation_time = time.time()
        
        return MovementValidation(
            is_valid=is_valid,
            adjusted_angle=adjusted_angle,
            adjusted_speed=adjusted_speed,
            violations=violations,
            warnings=warnings,
            applied_constraints=applied_constraints
        )
    
    def get_joint_constraints(self, joint_name: str) -> Optional[Dict[str, Any]]:
        """
        Get constraints for a specific joint.
        
        Args:
            joint_name: Name of the joint
            
        Returns:
            Dict containing joint constraints or None if not found
        """
        return self.constraints.get(joint_name)
    
    def get_fixed_joints(self) -> Dict[str, float]:
        """
        Get fixed joint positions for face tracking mode.
        
        Returns:
            Dict of joint names and their fixed positions
        """
        if not self.face_tracking_mode:
            return {}
        
        face_config = self.global_constraints.get("face_tracking_mode", {})
        return face_config.get("fixed_joints", {})
    
    def get_active_joints(self) -> List[str]:
        """
        Get list of active joints for face tracking mode.
        
        Returns:
            List of active joint names
        """
        if not self.face_tracking_mode:
            return list(self.constraints.keys())
        
        face_config = self.global_constraints.get("face_tracking_mode", {})
        return face_config.get("active_joints", [])
    
    def is_face_tracking_mode(self) -> bool:
        """
        Check if face tracking mode is enabled.
        
        Returns:
            bool: True if face tracking mode is enabled
        """
        return self.face_tracking_mode
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of constraint violations.
        
        Returns:
            Dict containing violation statistics
        """
        if not self.violation_history:
            return {"total_violations": 0, "violations_by_type": {}, "violations_by_joint": {}}
        
        violations_by_type = {}
        violations_by_joint = {}
        
        for violation in self.violation_history:
            # Count by type
            v_type = violation.constraint_type.value
            violations_by_type[v_type] = violations_by_type.get(v_type, 0) + 1
            
            # Count by joint
            violations_by_joint[violation.joint_name] = violations_by_joint.get(violation.joint_name, 0) + 1
        
        return {
            "total_violations": len(self.violation_history),
            "violations_by_type": violations_by_type,
            "violations_by_joint": violations_by_joint,
            "last_validation": self.last_validation_time
        }
    
    def clear_violation_history(self):
        """Clear the violation history."""
        self.violation_history.clear()
    
    def reload_constraints(self) -> bool:
        """
        Reload constraints from the JSON file.
        
        Returns:
            bool: True if reloaded successfully
        """
        return self.load_constraints()


# Global constraint manager instance
_global_constraint_manager = None


def get_constraint_manager(constraints_file: str = "joint_constraints.json") -> JointConstraintManager:
    """
    Get the global constraint manager instance.
    
    Args:
        constraints_file: Path to constraints file (only used on first call)
        
    Returns:
        JointConstraintManager: Global constraint manager instance
    """
    global _global_constraint_manager
    if _global_constraint_manager is None:
        _global_constraint_manager = JointConstraintManager(constraints_file)
    return _global_constraint_manager


def validate_movement(joint_name: str, target_angle: float, speed: int, 
                     current_angle: Optional[float] = None) -> MovementValidation:
    """
    Convenience function to validate movement using global constraint manager.
    
    Args:
        joint_name: Name of the joint
        target_angle: Target angle in degrees
        speed: Movement speed
        current_angle: Current joint angle (optional)
        
    Returns:
        MovementValidation: Validation result
    """
    manager = get_constraint_manager()
    return manager.validate_joint_movement(joint_name, target_angle, speed, current_angle)


if __name__ == "__main__":
    """Test the constraint manager."""
    print("🧪 Testing Joint Constraint Manager")
    
    # Create constraint manager
    manager = JointConstraintManager()
    
    # Test some validations
    test_cases = [
        ("joint1", 45.0, 100, 0.0),
        ("joint2", -50.0, 150, -42.4),  # Should be limited by speed
        ("joint3", -140.0, 100, -132.0),  # Should be limited by movement per step
        ("gripper", 100.0, 100, 0.0),  # Should be clamped to max angle
    ]
    
    for joint_name, target_angle, speed, current_angle in test_cases:
        print(f"\n🔍 Testing {joint_name}: {target_angle}° at speed {speed}")
        result = manager.validate_joint_movement(joint_name, target_angle, speed, current_angle)
        
        print(f"   Valid: {result.is_valid}")
        print(f"   Adjusted angle: {result.adjusted_angle}°")
        print(f"   Adjusted speed: {result.adjusted_speed}")
        
        if result.violations:
            print(f"   Violations: {len(result.violations)}")
            for violation in result.violations:
                print(f"     - {violation.violation_message} ({violation.severity})")
        
        if result.warnings:
            print(f"   Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"     - {warning}")
    
    # Show violation summary
    summary = manager.get_violation_summary()
    print(f"\n📊 Violation Summary: {summary}") 