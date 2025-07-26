#!/usr/bin/env python3
"""
PID Configuration Management System for Face Tracking

This module provides comprehensive configuration management for the PID-based
face tracking system, including parameter validation, file loading, environment
variable support, and runtime parameter adjustment.

Requirements addressed:
- 6.1: Load PID parameters from configuration
- 6.4: Operators can adjust Kp, Ki, and Kd values
- 6.5: Apply changes without requiring restart
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source types."""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


@dataclass
class PIDGains:
    """PID gain parameters for a single axis."""
    Kp: float = 0.1
    Ki: float = 0.01
    Kd: float = 0.05
    
    def validate(self) -> Tuple[bool, str]:
        """Validate PID gains are within acceptable ranges."""
        if self.Kp < 0 or self.Kp > 10.0:
            return False, f"Kp ({self.Kp}) must be between 0 and 10.0"
        if self.Ki < 0 or self.Ki > 1.0:
            return False, f"Ki ({self.Ki}) must be between 0 and 1.0"
        if self.Kd < 0 or self.Kd > 1.0:
            return False, f"Kd ({self.Kd}) must be between 0 and 1.0"
        return True, "Valid"


@dataclass
class ControlParameters:
    """Control system parameters."""
    dead_zone: float = 15.0  # pixels
    max_movement: float = 8.0  # degrees per update
    pixels_to_degrees: float = 0.1  # conversion factor
    
    def validate(self) -> Tuple[bool, str]:
        """Validate control parameters."""
        if self.dead_zone < 0 or self.dead_zone > 100:
            return False, f"dead_zone ({self.dead_zone}) must be between 0 and 100 pixels"
        if self.max_movement < 0.1 or self.max_movement > 45.0:
            return False, f"max_movement ({self.max_movement}) must be between 0.1 and 45.0 degrees"
        if self.pixels_to_degrees <= 0 or self.pixels_to_degrees > 1.0:
            return False, f"pixels_to_degrees ({self.pixels_to_degrees}) must be between 0 and 1.0"
        return True, "Valid"


@dataclass
class AntiWindupParameters:
    """Anti-windup protection parameters."""
    integral_min: float = -500.0
    integral_max: float = 500.0
    
    def validate(self) -> Tuple[bool, str]:
        """Validate anti-windup parameters."""
        if self.integral_min >= self.integral_max:
            return False, f"integral_min ({self.integral_min}) must be less than integral_max ({self.integral_max})"
        if abs(self.integral_min) > 10000 or abs(self.integral_max) > 10000:
            return False, "Integral limits must be within ±10000"
        return True, "Valid"


@dataclass
class SafetyParameters:
    """Safety system parameters."""
    safety_timeout: float = 30.0  # seconds
    max_tracking_time: float = 300.0  # seconds (5 minutes)
    emergency_stop_enabled: bool = True
    joint_limit_enforcement: bool = True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate safety parameters."""
        if self.safety_timeout < 1.0 or self.safety_timeout > 300.0:
            return False, f"safety_timeout ({self.safety_timeout}) must be between 1.0 and 300.0 seconds"
        if self.max_tracking_time < 10.0 or self.max_tracking_time > 3600.0:
            return False, f"max_tracking_time ({self.max_tracking_time}) must be between 10.0 and 3600.0 seconds"
        if self.safety_timeout >= self.max_tracking_time:
            return False, "safety_timeout must be less than max_tracking_time"
        return True, "Valid"


@dataclass
class DisplayParameters:
    """Display and visualization parameters."""
    show_pid_values: bool = True
    show_error_vectors: bool = True
    show_target_lock: bool = True
    show_data_collection_progress: bool = True
    overlay_transparency: float = 0.7
    
    def validate(self) -> Tuple[bool, str]:
        """Validate display parameters."""
        if self.overlay_transparency < 0.0 or self.overlay_transparency > 1.0:
            return False, f"overlay_transparency ({self.overlay_transparency}) must be between 0.0 and 1.0"
        return True, "Valid"


@dataclass
class PIDConfig:
    """
    Comprehensive PID configuration for face tracking system.
    
    This class contains all tunable parameters with default values,
    validation, and runtime adjustment capabilities.
    """
    
    # PID Gains
    pan_gains: PIDGains = field(default_factory=PIDGains)
    tilt_gains: PIDGains = field(default_factory=PIDGains)
    
    # Control Parameters
    control: ControlParameters = field(default_factory=ControlParameters)
    
    # Anti-windup Protection
    anti_windup: AntiWindupParameters = field(default_factory=AntiWindupParameters)
    
    # Safety Parameters
    safety: SafetyParameters = field(default_factory=SafetyParameters)
    
    # Display Parameters
    display: DisplayParameters = field(default_factory=DisplayParameters)
    
    # Metadata
    config_version: str = "1.0"
    last_modified: Optional[str] = None
    source: ConfigSource = ConfigSource.DEFAULT
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate all configuration parameters.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Validate each section
        sections = [
            ("pan_gains", self.pan_gains),
            ("tilt_gains", self.tilt_gains),
            ("control", self.control),
            ("anti_windup", self.anti_windup),
            ("safety", self.safety),
            ("display", self.display)
        ]
        
        for section_name, section in sections:
            is_valid, message = section.validate()
            if not is_valid:
                return False, f"{section_name}: {message}"
        
        return True, "All parameters valid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Convert enum to string for serialization
        config_dict['source'] = self.source.value
        return config_dict
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration updates
        """
        # Update pan gains
        if "pan_gains" in config_dict:
            for key, value in config_dict["pan_gains"].items():
                if hasattr(self.pan_gains, key):
                    setattr(self.pan_gains, key, value)
        
        # Update tilt gains
        if "tilt_gains" in config_dict:
            for key, value in config_dict["tilt_gains"].items():
                if hasattr(self.tilt_gains, key):
                    setattr(self.tilt_gains, key, value)
        
        # Update control parameters
        if "control" in config_dict:
            for key, value in config_dict["control"].items():
                if hasattr(self.control, key):
                    setattr(self.control, key, value)
        
        # Update anti-windup parameters
        if "anti_windup" in config_dict:
            for key, value in config_dict["anti_windup"].items():
                if hasattr(self.anti_windup, key):
                    setattr(self.anti_windup, key, value)
        
        # Update safety parameters
        if "safety" in config_dict:
            for key, value in config_dict["safety"].items():
                if hasattr(self.safety, key):
                    setattr(self.safety, key, value)
        
        # Update display parameters
        if "display" in config_dict:
            for key, value in config_dict["display"].items():
                if hasattr(self.display, key):
                    setattr(self.display, key, value)
        
        # Update metadata
        self.source = ConfigSource.RUNTIME
        import datetime
        self.last_modified = datetime.datetime.now().isoformat()


class PIDConfigManager:
    """
    Configuration manager for PID face tracking system.
    
    Handles loading from files, environment variables, validation,
    and runtime parameter adjustment.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to search for configuration files
        """
        self.config_dir = config_dir or Path.cwd()
        self.config = PIDConfig()
        self._config_files = [
            "pid_config.json",
            "pid_config.yaml",
            "pid_config.yml",
            ".pid_config.json",
            ".pid_config.yaml"
        ]
        self._env_prefix = "PID_"
    
    def load_from_file(self, file_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            file_path: Specific file path, or None to search for default files
            
        Returns:
            bool: True if configuration was loaded successfully
        """
        if file_path:
            return self._load_specific_file(Path(file_path))
        
        # Search for configuration files
        for config_file in self._config_files:
            config_path = self.config_dir / config_file
            if config_path.exists():
                if self._load_specific_file(config_path):
                    logger.info(f"Loaded configuration from {config_path}")
                    return True
        
        logger.info("No configuration file found, using defaults")
        return False
    
    def _load_specific_file(self, file_path: Path) -> bool:
        """Load configuration from a specific file."""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            self.config.update_from_dict(config_dict)
            self.config.source = ConfigSource.FILE
            
            # Validate after loading
            is_valid, message = self.config.validate()
            if not is_valid:
                logger.error(f"Invalid configuration loaded from {file_path}: {message}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            return False
    
    def load_from_environment(self) -> int:
        """Load configuration from environment variables."""
        print("METHOD CALLED!")
        loaded_count = 0
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(self._env_prefix):
                continue
            
            # Parse environment variable name
            parts = env_var[len(self._env_prefix):].lower().split('_')
            if len(parts) < 2:
                continue
            
            try:
                # Convert string value to appropriate type
                if value.lower() in ['true', 'false']:
                    parsed_value = value.lower() == 'true'
                elif '.' in value:
                    parsed_value = float(value)
                else:
                    try:
                        parsed_value = int(value)
                    except ValueError:
                        parsed_value = value
                
                # Apply the configuration
                if self._set_nested_parameter(parts, parsed_value):
                    loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to parse environment variable {env_var}={value}: {e}")
        
        if loaded_count > 0:
            self.config.source = ConfigSource.ENVIRONMENT
            logger.info(f"Loaded {loaded_count} parameters from environment variables")
        
        return loaded_count
    
    def _set_nested_parameter(self, parts: list, value: Any) -> bool:
        """Set a nested parameter using dot notation."""
        try:
            print(f"DEBUG: Setting parameter: parts={parts}, value={value}")
            
            if len(parts) == 3:
                section_name, subsection_name, param_name = parts
                # Handle nested structure like pan_gains_kp -> pan_gains.Kp
                if subsection_name == "gains":
                    section_name = f"{section_name}_gains"
                    param_name = param_name.upper()  # Convert to uppercase for Kp, Ki, Kd
                    print(f"DEBUG: Gains parameter: section={section_name}, param={param_name}")
                else:
                    # Handle direct section parameters like control_dead_zone
                    param_name = f"{subsection_name}_{param_name}"
                    print(f"DEBUG: Section parameter: section={section_name}, param={param_name}")
                
                section = getattr(self.config, section_name, None)
                print(f"DEBUG: Section object: {section}, has_attr: {hasattr(section, param_name) if section else False}")
                
                if section and hasattr(section, param_name):
                    setattr(section, param_name, value)
                    print(f"DEBUG: Successfully set {section_name}.{param_name} = {value}")
                    return True
                    
            elif len(parts) == 2:
                section_name, param_name = parts
                section = getattr(self.config, section_name, None)
                if section and hasattr(section, param_name):
                    setattr(section, param_name, value)
                    return True
                    
            elif len(parts) == 4:
                # Handle cases like safety_emergency_stop_enabled
                section_name, param1, param2, param3 = parts
                param_name = f"{param1}_{param2}_{param3}"
                section = getattr(self.config, section_name, None)
                if section and hasattr(section, param_name):
                    setattr(section, param_name, value)
                    return True
                    
        except Exception as e:
            print(f"DEBUG: Exception in _set_nested_parameter: {e}")
            
        print(f"DEBUG: Failed to set parameter: parts={parts}")
        return False
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "json") -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format ("json" or "yaml")
            
        Returns:
            bool: True if saved successfully
        """
        try:
            file_path = Path(file_path)
            config_dict = self.config.to_dict()
            
            with open(file_path, 'w') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            return False
    
    def update_parameter(self, parameter_path: str, value: Any) -> bool:
        """
        Update a single parameter at runtime.
        
        Args:
            parameter_path: Dot-separated path to parameter (e.g., "pan_gains.Kp")
            value: New value for the parameter
            
        Returns:
            bool: True if parameter was updated successfully
        """
        try:
            parts = parameter_path.split('.')
            
            if len(parts) == 2:
                section_name, param_name = parts
                section = getattr(self.config, section_name, None)
                if section and hasattr(section, param_name):
                    # Store old value for validation
                    old_value = getattr(section, param_name)
                    setattr(section, param_name, value)
                    
                    # Validate the change
                    is_valid, message = section.validate()
                    if not is_valid:
                        # Revert the change
                        setattr(section, param_name, old_value)
                        logger.error(f"Invalid parameter value: {message}")
                        return False
                    
                    # Update metadata
                    self.config.source = ConfigSource.RUNTIME
                    import datetime
                    self.config.last_modified = datetime.datetime.now().isoformat()
                    
                    logger.info(f"Updated {parameter_path} = {value}")
                    return True
            
            logger.error(f"Invalid parameter path: {parameter_path}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to update parameter {parameter_path}: {e}")
            return False
    
    def get_parameter(self, parameter_path: str) -> Any:
        """
        Get a parameter value by path.
        
        Args:
            parameter_path: Dot-separated path to parameter
            
        Returns:
            Parameter value or None if not found
        """
        try:
            parts = parameter_path.split('.')
            
            if len(parts) == 2:
                section_name, param_name = parts
                section = getattr(self.config, section_name, None)
                if section and hasattr(section, param_name):
                    return getattr(section, param_name)
            
            return None
            
        except Exception:
            return None
    
    def validate_configuration(self) -> Tuple[bool, str]:
        """
        Validate current configuration.
        
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        return self.config.validate()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = PIDConfig()
        logger.info("Configuration reset to defaults")
    
    def print_configuration(self) -> None:
        """Print current configuration in a readable format."""
        print("🎛️  PID Configuration Summary")
        print("=" * 60)
        
        print(f"📊 Pan Gains:     Kp={self.config.pan_gains.Kp:.3f}, Ki={self.config.pan_gains.Ki:.3f}, Kd={self.config.pan_gains.Kd:.3f}")
        print(f"📊 Tilt Gains:    Kp={self.config.tilt_gains.Kp:.3f}, Ki={self.config.tilt_gains.Ki:.3f}, Kd={self.config.tilt_gains.Kd:.3f}")
        print(f"🎯 Dead Zone:     {self.config.control.dead_zone:.1f} pixels")
        print(f"🚀 Max Movement:  {self.config.control.max_movement:.1f} degrees")
        print(f"🔄 Conversion:    {self.config.control.pixels_to_degrees:.3f} deg/pixel")
        print(f"🛡️  Safety Timeout: {self.config.safety.safety_timeout:.1f} seconds")
        print(f"⏰ Max Tracking:  {self.config.safety.max_tracking_time:.1f} seconds")
        print(f"📍 Source:        {self.config.source.value}")
        
        if self.config.last_modified:
            print(f"🕐 Last Modified: {self.config.last_modified}")
        
        print("=" * 60)


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> PIDConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = PIDConfigManager()
        # Try to load configuration on first access
        _config_manager.load_from_file()
        _config_manager.load_from_environment()
    return _config_manager


def get_config() -> PIDConfig:
    """Get the current PID configuration."""
    return get_config_manager().config


# Convenience functions for common operations
def update_pid_gains(axis: str, Kp: float = None, Ki: float = None, Kd: float = None) -> bool:
    """
    Update PID gains for a specific axis.
    
    Args:
        axis: "pan" or "tilt"
        Kp, Ki, Kd: New gain values (None to keep current value)
        
    Returns:
        bool: True if all updates were successful
    """
    manager = get_config_manager()
    success = True
    
    if axis not in ["pan", "tilt"]:
        logger.error(f"Invalid axis: {axis}. Must be 'pan' or 'tilt'")
        return False
    
    for gain_name, value in [("Kp", Kp), ("Ki", Ki), ("Kd", Kd)]:
        if value is not None:
            parameter_path = f"{axis}_gains.{gain_name}"
            if not manager.update_parameter(parameter_path, value):
                success = False
    
    return success


if __name__ == "__main__":
    """Example usage and testing."""
    print("🧪 Testing PID Configuration Management System")
    
    # Create configuration manager
    manager = PIDConfigManager()
    
    # Print default configuration
    manager.print_configuration()
    
    # Test parameter updates
    print("\n🔧 Testing parameter updates...")
    success = manager.update_parameter("pan_gains.Kp", 0.2)
    print(f"Update pan Kp: {'✅' if success else '❌'}")
    
    success = manager.update_parameter("control.dead_zone", 20.0)
    print(f"Update dead zone: {'✅' if success else '❌'}")
    
    # Test validation
    print("\n🔍 Testing validation...")
    is_valid, message = manager.validate_configuration()
    print(f"Configuration valid: {'✅' if is_valid else '❌'} - {message}")
    
    # Test invalid parameter
    success = manager.update_parameter("pan_gains.Kp", -1.0)  # Invalid negative value
    print(f"Invalid parameter rejected: {'✅' if not success else '❌'}")
    
    # Print final configuration
    print("\n📋 Final configuration:")
    manager.print_configuration()


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> PIDConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = PIDConfigManager()
        # Try to load configuration on first access
        _config_manager.load_from_file()
        _config_manager.load_from_environment()
    return _config_manager


def get_config() -> PIDConfig:
    """Get the current PID configuration."""
    return get_config_manager().config