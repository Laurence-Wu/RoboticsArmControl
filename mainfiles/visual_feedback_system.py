#!/usr/bin/env python3
"""
Visual Feedback and Display System for PID Face Tracking

This module provides comprehensive visual feedback including tracking overlays,
PID status displays, real-time parameter monitoring, progress bars, and
keyboard input handling for system control.

Requirements addressed:
- 4.1: Display camera feed with tracking overlays
- 4.2: Draw bounding boxes around detected faces
- 4.3: Display target position and error vectors
- 4.4: Show current PID parameters and output values
- 4.5: Display progress bar for data collection phases
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue

from pid_config import get_config, DisplayParameters


class DisplayMode(Enum):
    """Display mode options."""
    FULL = "full"
    MINIMAL = "minimal"
    DEBUG = "debug"
    OFF = "off"


@dataclass
class DisplayState:
    """Current display state information."""
    mode: DisplayMode = DisplayMode.FULL
    show_overlays: bool = True
    show_info_panel: bool = True
    show_debug_info: bool = False
    paused: bool = False
    recording: bool = False
    frame_count: int = 0
    fps: float = 0.0
    last_fps_update: float = 0.0


@dataclass
class TrackingOverlayData:
    """Data for tracking overlays."""
    faces: List[Tuple[int, int, int, int, float]] = None  # x, y, w, h, confidence
    target_position: Tuple[int, int] = None
    center_position: Tuple[int, int] = (960, 540)  # 1920x1080
    error_vector: Tuple[float, float] = (0.0, 0.0)
    target_locked: bool = False
    confidence: float = 0.0


@dataclass
class PIDDisplayData:
    """Data for PID status display."""
    pan_output: float = 0.0
    tilt_output: float = 0.0
    pan_error: float = 0.0
    tilt_error: float = 0.0
    pan_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Kp, Ki, Kd
    tilt_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Kp, Ki, Kd
    integral_pan: float = 0.0
    integral_tilt: float = 0.0
    derivative_pan: float = 0.0
    derivative_tilt: float = 0.0


@dataclass
class SystemStatusData:
    """Data for system status display."""
    tracking_active: bool = False
    data_collection_active: bool = False
    collection_progress: float = 0.0
    collection_duration: float = 2.0
    motor_status: str = "Unknown"
    safety_status: str = "OK"
    error_message: Optional[str] = None


class VisualFeedbackSystem:
    """
    Comprehensive visual feedback system for PID face tracking.
    
    Provides real-time display of tracking information, PID status,
    system parameters, and user interface elements.
    """
    
    def __init__(self, window_name: str = "PID Face Tracking", 
                 display_config: Optional[DisplayParameters] = None):
        """
        Initialize the visual feedback system.
        
        Args:
            window_name: Name of the display window
            display_config: Display configuration parameters
        """
        self.window_name = window_name
        self.display_config = display_config or get_config().display
        
        # Display state
        self.state = DisplayState()
        self.window_created = False
        
        # Color scheme
        self.colors = {
            'face_box': (0, 255, 0),      # Green
            'target': (255, 0, 0),        # Blue
            'center': (0, 255, 255),      # Yellow
            'error_vector': (0, 0, 255),  # Red
            'locked_target': (255, 0, 255), # Magenta
            'text': (255, 255, 255),      # White
            'background': (0, 0, 0),      # Black
            'progress_bg': (64, 64, 64),  # Dark gray
            'progress_fg': (0, 255, 0),   # Green
            'warning': (0, 165, 255),     # Orange
            'error': (0, 0, 255),         # Red
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 1
        self.large_font_scale = 0.8
        self.large_font_thickness = 2
        
        # Layout settings
        self.info_panel_width = 300
        self.info_panel_height = 400
        self.margin = 10
        self.line_height = 25
        
        # FPS calculation
        self.fps_frames = []
        self.fps_window = 30  # Calculate FPS over last 30 frames
        
        # Keyboard input handling
        self.key_handlers = {
            ord('q'): self._handle_quit,
            ord('Q'): self._handle_quit,
            27: self._handle_escape,  # ESC key
            ord('r'): self._handle_reset,
            ord('R'): self._handle_reset,
            ord('p'): self._handle_pause,
            ord('P'): self._handle_pause,
            ord('d'): self._handle_debug_toggle,
            ord('D'): self._handle_debug_toggle,
            ord('m'): self._handle_mode_cycle,
            ord('M'): self._handle_mode_cycle,
            ord('h'): self._handle_help,
            ord('H'): self._handle_help,
        }
        
        # Help text
        self.help_text = [
            "Keyboard Controls:",
            "Q/ESC - Quit system",
            "R - Reset tracking",
            "P - Pause/Resume",
            "D - Toggle debug info",
            "M - Cycle display modes",
            "H - Show/Hide help",
        ]
        
        self.show_help = False
    
    def create_window(self) -> bool:
        """
        Create the display window.
        
        Returns:
            bool: True if window was created successfully
        """
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
            return True
        except Exception as e:
            print(f"❌ Failed to create display window: {e}")
            return False
    
    def destroy_window(self) -> None:
        """Destroy the display window and clean up resources."""
        try:
            if self.window_created:
                cv2.destroyWindow(self.window_name)
                cv2.destroyAllWindows()
                self.window_created = False
        except Exception as e:
            print(f"⚠️  Error destroying window: {e}")
    
    def update_display(self, frame: np.ndarray,
                      tracking_data: Optional[TrackingOverlayData] = None,
                      pid_data: Optional[PIDDisplayData] = None,
                      status_data: Optional[SystemStatusData] = None) -> int:
        """
        Update the display with current frame and data.
        
        Args:
            frame: Current camera frame
            tracking_data: Face tracking overlay data
            pid_data: PID controller status data
            status_data: System status data
            
        Returns:
            int: Key code pressed (0 if no key pressed, -1 for quit)
        """
        if not self.window_created:
            if not self.create_window():
                return -1
        
        if self.state.mode == DisplayMode.OFF:
            return 0
        
        # Update FPS calculation
        self._update_fps()
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw tracking overlays
        if tracking_data and self.display_config.show_error_vectors:
            self._draw_tracking_overlays(display_frame, tracking_data)
        
        # Draw info panel
        if self.state.show_info_panel and self.state.mode != DisplayMode.MINIMAL:
            self._draw_info_panel(display_frame, tracking_data, pid_data, status_data)
        
        # Draw debug information
        if self.state.show_debug_info and self.state.mode == DisplayMode.DEBUG:
            self._draw_debug_info(display_frame, pid_data)
        
        # Draw help overlay
        if self.show_help:
            self._draw_help_overlay(display_frame)
        
        # Draw status bar
        self._draw_status_bar(display_frame, status_data)
        
        # Show the frame
        try:
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            return self._handle_keyboard_input(key)
            
        except Exception as e:
            print(f"⚠️  Display error: {e}")
            return -1
    
    def _draw_tracking_overlays(self, frame: np.ndarray, 
                               tracking_data: TrackingOverlayData) -> None:
        """Draw face tracking overlays on the frame."""
        if not tracking_data:
            return
        
        # Draw detected faces
        if tracking_data.faces:
            for i, face in enumerate(tracking_data.faces):
                x, y, w, h, confidence = face
                
                # Choose color based on whether this is the target
                if (tracking_data.target_position and 
                    abs(x + w//2 - tracking_data.target_position[0]) < 20 and
                    abs(y + h//2 - tracking_data.target_position[1]) < 20):
                    color = self.colors['locked_target'] if tracking_data.target_locked else self.colors['target']
                    thickness = 3
                else:
                    color = self.colors['face_box']
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
                
                # Draw confidence score
                conf_text = f'Face {confidence:.2f}'
                cv2.putText(frame, conf_text, (x, y - 10), 
                           self.font, self.font_scale, color, self.font_thickness)
        
        # Draw center point (yellow cross)
        center = tracking_data.center_position
        cross_size = 15
        cv2.line(frame, (center[0] - cross_size, center[1]), 
                (center[0] + cross_size, center[1]), self.colors['center'], 2)
        cv2.line(frame, (center[0], center[1] - cross_size), 
                (center[0], center[1] + cross_size), self.colors['center'], 2)
        
        # Draw target position
        if tracking_data.target_position:
            target = tracking_data.target_position
            
            # Draw target circle
            color = self.colors['locked_target'] if tracking_data.target_locked else self.colors['target']
            cv2.circle(frame, target, 8, color, -1)
            cv2.circle(frame, target, 12, color, 2)
            
            # Draw error vector arrow
            if tracking_data.error_vector and (abs(tracking_data.error_vector[0]) > 5 or 
                                             abs(tracking_data.error_vector[1]) > 5):
                end_point = (int(center[0] + tracking_data.error_vector[0] * 2),
                           int(center[1] + tracking_data.error_vector[1] * 2))
                cv2.arrowedLine(frame, target, end_point, 
                              self.colors['error_vector'], 2, tipLength=0.3)
                
                # Draw error magnitude
                magnitude = np.sqrt(tracking_data.error_vector[0]**2 + tracking_data.error_vector[1]**2)
                error_text = f'Error: {magnitude:.1f}px'
                cv2.putText(frame, error_text, (target[0] + 15, target[1] - 15),
                           self.font, self.font_scale, self.colors['error_vector'], self.font_thickness)
    
    def _draw_info_panel(self, frame: np.ndarray,
                        tracking_data: Optional[TrackingOverlayData],
                        pid_data: Optional[PIDDisplayData],
                        status_data: Optional[SystemStatusData]) -> None:
        """Draw the information panel overlay."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        panel_x = width - self.info_panel_width - self.margin
        panel_y = self.margin
        
        # Draw background rectangle
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + self.info_panel_width, panel_y + self.info_panel_height),
                     self.colors['background'], -1)
        
        # Apply transparency
        alpha = self.display_config.overlay_transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw panel content
        y_offset = panel_y + 25
        
        # Title
        cv2.putText(frame, "PID Face Tracking", (panel_x + 10, y_offset),
                   self.font, self.large_font_scale, self.colors['text'], self.large_font_thickness)
        y_offset += 35
        
        # System status
        if status_data:
            status_color = self.colors['text']
            if status_data.error_message:
                status_color = self.colors['error']
            elif status_data.safety_status != "OK":
                status_color = self.colors['warning']
            
            cv2.putText(frame, f"Status: {status_data.safety_status}", 
                       (panel_x + 10, y_offset), self.font, self.font_scale, status_color, self.font_thickness)
            y_offset += self.line_height
            
            cv2.putText(frame, f"Motor: {status_data.motor_status}", 
                       (panel_x + 10, y_offset), self.font, self.font_scale, self.colors['text'], self.font_thickness)
            y_offset += self.line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {self.state.fps:.1f}", 
                   (panel_x + 10, y_offset), self.font, self.font_scale, self.colors['text'], self.font_thickness)
        y_offset += self.line_height + 10
        
        # PID information
        if pid_data and self.display_config.show_pid_values:
            cv2.putText(frame, "PID Controllers:", (panel_x + 10, y_offset),
                       self.font, self.font_scale, self.colors['text'], self.font_thickness)
            y_offset += self.line_height
            
            # Pan controller
            cv2.putText(frame, f"Pan Output: {pid_data.pan_output:6.2f}°", 
                       (panel_x + 15, y_offset), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
            y_offset += self.line_height - 5
            
            cv2.putText(frame, f"Pan Error:  {pid_data.pan_error:6.1f}px", 
                       (panel_x + 15, y_offset), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
            y_offset += self.line_height - 5
            
            # Tilt controller
            cv2.putText(frame, f"Tilt Output: {pid_data.tilt_output:5.2f}°", 
                       (panel_x + 15, y_offset), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
            y_offset += self.line_height - 5
            
            cv2.putText(frame, f"Tilt Error:  {pid_data.tilt_error:5.1f}px", 
                       (panel_x + 15, y_offset), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
            y_offset += self.line_height + 5
        
        # Data collection progress
        if status_data and status_data.data_collection_active and self.display_config.show_data_collection_progress:
            self._draw_progress_bar(frame, panel_x + 10, y_offset, 
                                  self.info_panel_width - 20, 15,
                                  status_data.collection_progress, "Data Collection")
            y_offset += 35
        
        # Target lock status
        if tracking_data and self.display_config.show_target_lock:
            lock_status = "LOCKED" if tracking_data.target_locked else "SEARCHING"
            lock_color = self.colors['locked_target'] if tracking_data.target_locked else self.colors['warning']
            cv2.putText(frame, f"Target: {lock_status}", (panel_x + 10, y_offset),
                       self.font, self.font_scale, lock_color, self.font_thickness)
            y_offset += self.line_height
            
            if tracking_data.confidence > 0:
                cv2.putText(frame, f"Confidence: {tracking_data.confidence:.2f}", 
                           (panel_x + 10, y_offset), self.font, self.font_scale, self.colors['text'], self.font_thickness)
    
    def _draw_progress_bar(self, frame: np.ndarray, x: int, y: int, 
                          width: int, height: int, progress: float, label: str) -> None:
        """Draw a progress bar with label."""
        # Draw background
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     self.colors['progress_bg'], -1)
        
        # Draw progress
        progress_width = int(width * min(1.0, max(0.0, progress)))
        if progress_width > 0:
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height), 
                         self.colors['progress_fg'], -1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), 
                     self.colors['text'], 1)
        
        # Draw label and percentage
        label_text = f"{label}: {progress * 100:.0f}%"
        cv2.putText(frame, label_text, (x, y - 5),
                   self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
    
    def _draw_debug_info(self, frame: np.ndarray, pid_data: Optional[PIDDisplayData]) -> None:
        """Draw detailed debug information."""
        if not pid_data:
            return
        
        height, width = frame.shape[:2]
        debug_x = 10
        debug_y = height - 150
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (debug_x - 5, debug_y - 25), 
                     (debug_x + 400, debug_y + 120), self.colors['background'], -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Debug title
        cv2.putText(frame, "Debug Information:", (debug_x, debug_y),
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        debug_y += self.line_height
        
        # PID gains
        cv2.putText(frame, f"Pan Gains:  Kp={pid_data.pan_gains[0]:.3f} Ki={pid_data.pan_gains[1]:.3f} Kd={pid_data.pan_gains[2]:.3f}", 
                   (debug_x, debug_y), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
        debug_y += self.line_height - 5
        
        cv2.putText(frame, f"Tilt Gains: Kp={pid_data.tilt_gains[0]:.3f} Ki={pid_data.tilt_gains[1]:.3f} Kd={pid_data.tilt_gains[2]:.3f}", 
                   (debug_x, debug_y), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
        debug_y += self.line_height - 5
        
        # Internal PID states
        cv2.putText(frame, f"Pan I-term: {pid_data.integral_pan:6.2f}  D-term: {pid_data.derivative_pan:6.2f}", 
                   (debug_x, debug_y), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
        debug_y += self.line_height - 5
        
        cv2.putText(frame, f"Tilt I-term: {pid_data.integral_tilt:5.2f}  D-term: {pid_data.derivative_tilt:5.2f}", 
                   (debug_x, debug_y), self.font, self.font_scale - 0.1, self.colors['text'], self.font_thickness)
    
    def _draw_help_overlay(self, frame: np.ndarray) -> None:
        """Draw help text overlay."""
        height, width = frame.shape[:2]
        help_x = width // 2 - 150
        help_y = height // 2 - len(self.help_text) * self.line_height // 2
        
        # Create semi-transparent background
        overlay = frame.copy()
        bg_width = 300
        bg_height = len(self.help_text) * self.line_height + 20
        cv2.rectangle(overlay, (help_x - 10, help_y - 15), 
                     (help_x + bg_width, help_y + bg_height), self.colors['background'], -1)
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw help text
        for i, text in enumerate(self.help_text):
            y_pos = help_y + i * self.line_height
            color = self.colors['warning'] if i == 0 else self.colors['text']
            cv2.putText(frame, text, (help_x, y_pos),
                       self.font, self.font_scale, color, self.font_thickness)
    
    def _draw_status_bar(self, frame: np.ndarray, status_data: Optional[SystemStatusData]) -> None:
        """Draw status bar at the bottom of the frame."""
        height, width = frame.shape[:2]
        status_y = height - 25
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, status_y - 5), (width, height), self.colors['background'], -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Status text
        status_text = f"Mode: {self.state.mode.value.upper()}"
        if self.state.paused:
            status_text += " | PAUSED"
        if status_data and status_data.tracking_active:
            status_text += " | TRACKING"
        if status_data and status_data.error_message:
            status_text += f" | ERROR: {status_data.error_message}"
        
        cv2.putText(frame, status_text, (10, status_y + 15),
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        
        # Frame counter
        frame_text = f"Frame: {self.state.frame_count}"
        text_size = cv2.getTextSize(frame_text, self.font, self.font_scale, self.font_thickness)[0]
        cv2.putText(frame, frame_text, (width - text_size[0] - 10, status_y + 15),
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
    
    def _update_fps(self) -> None:
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_frames.append(current_time)
        
        # Keep only recent frames
        while len(self.fps_frames) > self.fps_window:
            self.fps_frames.pop(0)
        
        # Calculate FPS
        if len(self.fps_frames) > 1:
            time_span = self.fps_frames[-1] - self.fps_frames[0]
            if time_span > 0:
                self.state.fps = (len(self.fps_frames) - 1) / time_span
        
        self.state.frame_count += 1
    
    def _handle_keyboard_input(self, key: int) -> int:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            int: 0 for continue, -1 for quit, 1 for reset
        """
        if key == 255 or key == 0:  # No key pressed
            return 0
        
        # Check for registered key handlers
        if key in self.key_handlers:
            return self.key_handlers[key]()
        
        return 0
    
    def _handle_quit(self) -> int:
        """Handle quit command."""
        return -1
    
    def _handle_escape(self) -> int:
        """Handle escape key."""
        return -1
    
    def _handle_reset(self) -> int:
        """Handle reset command."""
        return 1
    
    def _handle_pause(self) -> int:
        """Handle pause/resume toggle."""
        self.state.paused = not self.state.paused
        return 0
    
    def _handle_debug_toggle(self) -> int:
        """Handle debug info toggle."""
        self.state.show_debug_info = not self.state.show_debug_info
        if self.state.show_debug_info:
            self.state.mode = DisplayMode.DEBUG
        else:
            self.state.mode = DisplayMode.FULL
        return 0
    
    def _handle_mode_cycle(self) -> int:
        """Handle display mode cycling."""
        modes = [DisplayMode.FULL, DisplayMode.MINIMAL, DisplayMode.DEBUG]
        current_index = modes.index(self.state.mode)
        next_index = (current_index + 1) % len(modes)
        self.state.mode = modes[next_index]
        
        # Update related flags
        self.state.show_info_panel = self.state.mode != DisplayMode.MINIMAL
        self.state.show_debug_info = self.state.mode == DisplayMode.DEBUG
        
        return 0
    
    def _handle_help(self) -> int:
        """Handle help toggle."""
        self.show_help = not self.show_help
        return 0
    
    def set_display_mode(self, mode: DisplayMode) -> None:
        """Set the display mode."""
        self.state.mode = mode
        self.state.show_info_panel = mode != DisplayMode.MINIMAL
        self.state.show_debug_info = mode == DisplayMode.DEBUG
    
    def is_paused(self) -> bool:
        """Check if display is paused."""
        return self.state.paused
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.state.fps
    
    def get_frame_count(self) -> int:
        """Get current frame count."""
        return self.state.frame_count


class EnhancedDisplayManager:
    """
    Enhanced display manager that integrates with the PID face tracking system.
    
    Provides a high-level interface for updating displays with tracking data,
    PID status, and system information.
    """
    
    def __init__(self, window_name: str = "PID Face Tracking System"):
        """Initialize the enhanced display manager."""
        self.visual_system = VisualFeedbackSystem(window_name)
        self.active = False
        
        # Data containers
        self.tracking_data = TrackingOverlayData()
        self.pid_data = PIDDisplayData()
        self.status_data = SystemStatusData()
    
    def start(self) -> bool:
        """
        Start the display system.
        
        Returns:
            bool: True if started successfully
        """
        if self.visual_system.create_window():
            self.active = True
            return True
        return False
    
    def stop(self) -> None:
        """Stop the display system."""
        self.visual_system.destroy_window()
        self.active = False
    
    def update_tracking_data(self, faces: List[Tuple], target_position: Optional[Tuple[int, int]] = None,
                           center_position: Tuple[int, int] = (960, 540),  # 1920x1080
                           error_vector: Tuple[float, float] = (0.0, 0.0),
                           target_locked: bool = False, confidence: float = 0.0) -> None:
        """Update tracking overlay data."""
        self.tracking_data.faces = faces
        self.tracking_data.target_position = target_position
        self.tracking_data.center_position = center_position
        self.tracking_data.error_vector = error_vector
        self.tracking_data.target_locked = target_locked
        self.tracking_data.confidence = confidence
    
    def update_pid_data(self, pan_output: float = 0.0, tilt_output: float = 0.0,
                       pan_error: float = 0.0, tilt_error: float = 0.0,
                       pan_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       tilt_gains: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       integral_pan: float = 0.0, integral_tilt: float = 0.0,
                       derivative_pan: float = 0.0, derivative_tilt: float = 0.0) -> None:
        """Update PID display data."""
        self.pid_data.pan_output = pan_output
        self.pid_data.tilt_output = tilt_output
        self.pid_data.pan_error = pan_error
        self.pid_data.tilt_error = tilt_error
        self.pid_data.pan_gains = pan_gains
        self.pid_data.tilt_gains = tilt_gains
        self.pid_data.integral_pan = integral_pan
        self.pid_data.integral_tilt = integral_tilt
        self.pid_data.derivative_pan = derivative_pan
        self.pid_data.derivative_tilt = derivative_tilt
    
    def update_system_status(self, tracking_active: bool = False,
                           data_collection_active: bool = False,
                           collection_progress: float = 0.0,
                           collection_duration: float = 2.0,
                           motor_status: str = "Unknown",
                           safety_status: str = "OK",
                           error_message: Optional[str] = None) -> None:
        """Update system status data."""
        self.status_data.tracking_active = tracking_active
        self.status_data.data_collection_active = data_collection_active
        self.status_data.collection_progress = collection_progress
        self.status_data.collection_duration = collection_duration
        self.status_data.motor_status = motor_status
        self.status_data.safety_status = safety_status
        self.status_data.error_message = error_message
    
    def display_frame(self, frame: np.ndarray) -> int:
        """
        Display a frame with all current data.
        
        Args:
            frame: Camera frame to display
            
        Returns:
            int: Key code (0 for continue, -1 for quit, 1 for reset)
        """
        if not self.active:
            return 0
        
        return self.visual_system.update_display(
            frame, self.tracking_data, self.pid_data, self.status_data
        )
    
    def is_paused(self) -> bool:
        """Check if display is paused."""
        return self.visual_system.is_paused()
    
    def get_fps(self) -> float:
        """Get current display FPS."""
        return self.visual_system.get_fps()


# Convenience functions for integration with existing code
def create_display_manager(window_name: str = "PID Face Tracking") -> EnhancedDisplayManager:
    """Create and return a new display manager."""
    return EnhancedDisplayManager(window_name)


def draw_enhanced_tracking_info(frame: np.ndarray, faces: List[Tuple], 
                               target_position: Optional[Tuple[int, int]],
                               motion_data: Any, pid_output: Tuple[float, float] = (0.0, 0.0),
                               pid_gains: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                               collection_progress: float = 0.0) -> None:
    """
    Enhanced version of draw_tracking_info for backward compatibility.
    
    This function provides an enhanced version of the existing draw_tracking_info
    function with additional PID status and progress information.
    """
    # Create a temporary visual system for one-time use
    visual_system = VisualFeedbackSystem()
    
    # Prepare data structures
    tracking_data = TrackingOverlayData(
        faces=faces,
        target_position=target_position,
        center_position=getattr(motion_data, 'center_position', (960, 540)),  # 1920x1080
        error_vector=(getattr(motion_data, 'vector', (0.0, 0.0))),
        target_locked=True,  # Assume locked if we have a target
        confidence=1.0
    )
    
    pid_data = PIDDisplayData(
        pan_output=pid_output[0],
        tilt_output=pid_output[1],
        pan_error=getattr(motion_data, 'vector', (0.0, 0.0))[0],
        tilt_error=getattr(motion_data, 'vector', (0.0, 0.0))[1],
        pan_gains=pid_gains[0],
        tilt_gains=pid_gains[1]
    )
    
    status_data = SystemStatusData(
        tracking_active=True,
        data_collection_active=collection_progress > 0,
        collection_progress=collection_progress,
        motor_status="Active",
        safety_status="OK"
    )
    
    # Draw overlays directly on frame
    visual_system._draw_tracking_overlays(frame, tracking_data)
    
    # Draw a simplified info panel
    if visual_system.display_config.show_pid_values:
        # Draw basic PID info in top-left corner
        y_offset = 30
        cv2.putText(frame, f"Pan: {pid_output[0]:5.2f}° Tilt: {pid_output[1]:5.2f}°", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if collection_progress > 0:
            y_offset += 25
            progress_text = f"Collection: {collection_progress*100:.0f}%"
            cv2.putText(frame, progress_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)


if __name__ == "__main__":
    """Test the visual feedback system."""
    import numpy as np
    
    print("🧪 Testing Visual Feedback System")
    
    # Create test display manager
    display_manager = create_display_manager("Test Display")
    
    if not display_manager.start():
        print("❌ Failed to start display system")
        exit(1)
    
    print("✅ Display system started")
    print("📋 Controls: Q/ESC=Quit, R=Reset, P=Pause, D=Debug, M=Mode, H=Help")
    
    try:
        # Simulate tracking loop
        frame_count = 0
        while True:
            # Create test frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50, 50, 50)  # Dark gray background
            
            # Simulate face detection
            faces = [(200, 150, 100, 120, 0.95), (400, 200, 80, 100, 0.87)]
            target_pos = (250, 210)
            
            # Update display data
            display_manager.update_tracking_data(
                faces=faces,
                target_position=target_pos,
                error_vector=(30.0, -15.0),
                target_locked=True,
                confidence=0.95
            )
            
            display_manager.update_pid_data(
                pan_output=2.5,
                tilt_output=-1.2,
                pan_error=30.0,
                tilt_error=-15.0,
                pan_gains=(0.1, 0.01, 0.05),
                tilt_gains=(0.12, 0.015, 0.06)
            )
            
            display_manager.update_system_status(
                tracking_active=True,
                data_collection_active=frame_count % 100 < 50,
                collection_progress=(frame_count % 100) / 100.0,
                motor_status="Active",
                safety_status="OK"
            )
            
            # Display frame
            key_result = display_manager.display_frame(frame)
            
            if key_result == -1:  # Quit
                break
            elif key_result == 1:  # Reset
                print("🔄 Reset requested")
                frame_count = 0
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    
    finally:
        display_manager.stop()
        print("👋 Display system stopped")