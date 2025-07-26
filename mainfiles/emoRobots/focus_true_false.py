#!/usr/bin/env python3
"""
Focus Detection Module - Reusable Function

This module provides a simple function to check if a person is focused.
Can be used as a standalone script or imported as a module.

Usage as module:
    from focus_true_false import get_focus_status, start_focus_detector
    
    # Get single focus reading
    is_focused = get_focus_status()  # Returns True/False
    
    # Start continuous monitoring with callback
    def on_focus_change(is_focused, score):
        print(f"Focus: {is_focused}, Score: {score}")
    
    detector = start_focus_detector(callback=on_focus_change)

Usage as script:
    python focus_true_false.py
"""

import sys
import time
import os
import numpy as np
from cortex import Cortex
from dotenv import load_dotenv
from collections import deque
import threading


class FocusDetector:
    def __init__(self, app_client_id, app_client_secret, callback=None, **kwargs):
        # Algorithm weights
        self.weights = {
            'attention': 0.35, 'engagement': 0.25, 'excitement': 0.15,
            'interest': 0.15, 'stress': 0.15, 'relaxation': -0.10,
        }
        
        self.performance_buffer = deque(maxlen=200)
        self.classification_interval = 2.0
        self.last_classification_time = time.time()
        self.focus_threshold = 0.44
        self.collection_start_time = None
        self.label_mapping = {}
        self.callback = callback
        self.current_focus_status = None
        self.current_score = 0.0
        self.is_running = False
        
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=False, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_met_data=self.on_new_met_data)
        self.c.bind(inform_error=self.on_inform_error)
        
    def start_detection(self):
        """Start the focus detection"""
        self.is_running = True
        self.c.open()
        
    def stop_detection(self):
        """Stop the focus detection"""
        self.is_running = False
        try:
            self.c.unsub_request(['met'])
            self.c.close()
        except:
            pass
        
    def get_current_status(self):
        """Get the current focus status"""
        return self.current_focus_status, self.current_score
        
    def parse_met_labels(self, labels):
        label_mapping = {}
        for i, label in enumerate(labels):
            label_str = str(label).lower()
            if label_str == 'attention':
                label_mapping['attention'] = i
                label_mapping['attention_active'] = i - 1
            elif label_str == 'eng':
                label_mapping['engagement'] = i
                label_mapping['engagement_active'] = i - 1
            elif label_str == 'exc':
                label_mapping['excitement'] = i
                label_mapping['excitement_active'] = i - 1
            elif label_str == 'int':
                label_mapping['interest'] = i
                label_mapping['interest_active'] = i - 1
            elif label_str == 'str':
                label_mapping['stress'] = i
                label_mapping['stress_active'] = i - 1
            elif label_str == 'rel':
                label_mapping['relaxation'] = i
                label_mapping['relaxation_active'] = i - 1
        return label_mapping
        
    def extract_metrics(self, met_values):
        if not self.label_mapping:
            return None
            
        metrics = {'attention': 0.0, 'relaxation': 0.0, 'engagement': 0.0, 
                  'excitement': 0.0, 'stress': 0.0, 'interest': 0.0}
        
        for metric_name in metrics.keys():
            value_index = self.label_mapping.get(metric_name)
            active_index = self.label_mapping.get(f'{metric_name}_active')
            
            if value_index is not None and value_index < len(met_values):
                is_active = True
                if active_index is not None and active_index < len(met_values):
                    active_value = met_values[active_index]
                    is_active = str(active_value).lower() == 'true'
                
                if is_active:
                    value = met_values[value_index]
                    if value is not None and str(value).strip() != '':
                        try:
                            metrics[metric_name] = float(value)
                        except (ValueError, TypeError):
                            metrics[metric_name] = 0.0
        return metrics
        
    def calculate_composite_score(self, metrics):
        return sum(metrics[k] * self.weights[k] for k in metrics.keys())
        
    def check_focus_status(self):
        current_time = time.time()
        
        if current_time - self.last_classification_time >= self.classification_interval:
            if len(self.performance_buffer) >= 10:
                recent_data = list(self.performance_buffer)[-40:]
                
                if recent_data:
                    metrics_arrays = {k: [] for k in ['attention', 'relaxation', 'engagement', 'excitement', 'stress', 'interest']}
                    
                    for timestamp, metrics in recent_data:
                        if metrics:
                            for metric_name in metrics_arrays.keys():
                                metrics_arrays[metric_name].append(metrics[metric_name])
                    
                    if any(len(values) > 0 for values in metrics_arrays.values()):
                        avg_metrics = {}
                        for metric_name, values in metrics_arrays.items():
                            avg_metrics[metric_name] = np.mean(values) if values else 0.0
                        
                        composite_score = self.calculate_composite_score(avg_metrics)
                        is_focused = composite_score > self.focus_threshold
                        
                        # Update current status
                        self.current_focus_status = is_focused
                        self.current_score = composite_score
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(is_focused, composite_score)
                        else:
                            # Default behavior: print TRUE/FALSE
                            print("TRUE" if is_focused else "FALSE")
                        
            self.last_classification_time = current_time

    def on_create_session_done(self, *args, **kwargs):
        self.c.sub_request(['met'])
        self.collection_start_time = time.time()
        self.last_classification_time = self.collection_start_time
        
    def on_new_data_labels(self, *args, **kwargs):
        data = kwargs.get('data')
        if data['streamName'] == 'met':
            self.label_mapping = self.parse_met_labels(data['labels'])
    
    def on_new_met_data(self, *args, **kwargs):
        if not self.is_running:
            return
            
        data = kwargs.get('data')
        metrics = self.extract_metrics(data['met'])
        if metrics:
            self.performance_buffer.append((data['time'], metrics))
            self.check_focus_status()
    
    def on_inform_error(self, *args, **kwargs):
        pass  # Silent error handling


# Global detector instance
_global_detector = None


def start_focus_detector(callback=None, client_id=None, client_secret=None):
    """
    Start a focus detector that runs in the background
    
    Args:
        callback: Function to call when focus status changes. 
                 Signature: callback(is_focused: bool, score: float)
        client_id: Emotiv app client ID (optional, will load from .env if not provided)
        client_secret: Emotiv app client secret (optional, will load from .env if not provided)
    
    Returns:
        FocusDetector instance
    """
    global _global_detector
    
    # Load credentials if not provided
    if not client_id or not client_secret:
        load_dotenv()
        client_id = client_id or os.getenv('CLIENT_ID')
        client_secret = client_secret or os.getenv('CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError("Missing credentials. Provide client_id/client_secret or set CLIENT_ID/CLIENT_SECRET in .env file")
    
    _global_detector = FocusDetector(client_id, client_secret, callback=callback)
    _global_detector.start_detection()
    
    return _global_detector


def get_focus_status(timeout=10):
    """
    Get a single focus status reading
    
    Args:
        timeout: Maximum time to wait for a reading (seconds)
    
    Returns:
        tuple: (is_focused: bool, score: float) or (None, None) if timeout
    """
    global _global_detector
    
    if not _global_detector:
        # Start detector if not already running
        start_focus_detector()
    
    # Wait for first reading
    start_time = time.time()
    while time.time() - start_time < timeout:
        status, score = _global_detector.get_current_status()
        if status is not None:
            return status, score
        time.sleep(0.1)
    
    return None, None


def stop_focus_detector():
    """Stop the global focus detector"""
    global _global_detector
    if _global_detector:
        _global_detector.stop_detection()
        _global_detector = None


def is_focused(timeout=10):
    """
    Simple function that returns True if focused, False if not
    
    Args:
        timeout: Maximum time to wait for a reading (seconds)
    
    Returns:
        bool: True if focused, False if not focused, None if timeout
    """
    status, _ = get_focus_status(timeout)
    return status


def main():
    """Main function for standalone script usage"""
    # Load credentials
    load_dotenv()
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("ERROR: Missing credentials", file=sys.stderr)
        sys.exit(1)
    
    try:
        detector = start_focus_detector()
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        stop_focus_detector()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()