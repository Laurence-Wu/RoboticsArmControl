# Advanced Robotic Arm Control System

[![AdventureX 2024](https://img.shields.io/badge/AdventureX-2024-blue.svg)](https://adventurex.com)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🏆 About AdventureX

This project was developed for **AdventureX**, the biggest Chinese hackathon that brings together the most innovative minds in technology. AdventureX challenges participants to push the boundaries of what's possible in 48 hours, and this advanced robotic arm control system represents our contribution to the future of robotics.

## 📖 Project Overview

An advanced control system for Seeed Studio robotic arms featuring:

- **PID Control System** - Precise motor positioning with real-time feedback
- **Visual Feedback Integration** - Computer vision-based object tracking and manipulation
- **Face Tracking** - Automatic face detection and following capabilities
- **Emergency Safety Systems** - Multiple failsafe mechanisms for safe operation
- **Modular Architecture** - Easy to extend and customize for different use cases

## 🛠 Hardware Requirements

- Seeed Studio Robotic Arm (6-DOF recommended)
- USB connection to control computer
- Camera (for visual feedback features)
- Python-compatible operating system (Windows, macOS, Linux)

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd advXRobotControl
pip install -r requirements.txt
```

### 2. Hardware Setup

1. Connect your robotic arm via USB
2. Power on the robotic arm
3. Find the correct serial port:

   ```bash
   python find_robot_port.py
   ```

### 3. Configuration

1. Update `config.py` with your robot's settings:

   ```python
   # Edit the DEFAULT_PORT if needed
   SERIAL_CONFIG.DEFAULT_PORT = "/dev/ttyUSB0"  # Linux/macOS
   # or
   SERIAL_CONFIG.DEFAULT_PORT = "COM3"  # Windows
   ```

### 4. First Run

Enable motors and test basic functionality:

```bash
python enable_motors.py
python simple_robot_control.py
```

## 📋 Available Scripts

### Core Control Scripts

| Script | Description |
|--------|-------------|
| `simple_robot_control.py` | Basic robot arm control and testing |
| `enable_motors.py` | Enable all motors for operation |
| `disable_all_motors.py` | Safe shutdown of all motors |
| `emergency_stop.py` | Emergency stop functionality |

### Advanced Features

| Script | Description |
|--------|-------------|
| `mainfiles/main.py` | Main integrated control system |
| `mainfiles/auto_face_tracking.py` | Automatic face detection and tracking |
| `mainfiles/visual_feedback_system.py` | Computer vision integration |
| `mainfiles/pid_controller.py` | PID control implementation |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `find_robot_port.py` | Automatically detect robot connection |
| `quick_robot_setup.py` | Quick setup and calibration |
| `test_numpy_fix.py` | Test NumPy compatibility |

## 🎮 Usage Examples

### Basic Motor Control

```bash
# Enable all motors
python enable_motors.py

# Run basic control interface
python simple_robot_control.py

# Emergency stop (if needed)
python emergency_stop.py
```

### Advanced PID Control

```bash
cd mainfiles
python main.py
```

### Face Tracking Demo

```bash
cd mainfiles
python auto_face_tracking.py
```

### Visual Feedback System

```bash
cd mainfiles
python visual_feedback_demo.py
```

## ⚙️ Configuration

### Main Configuration (`config.py`)

```python
# Serial Communication Settings
SERIAL_CONFIG.DEFAULT_PORT = "/dev/ttyUSB0"  # Adjust for your system
SERIAL_CONFIG.BAUDRATE = 115200

# Motor IDs (adjust based on your robot configuration)
MOTOR_CONFIG.MOTOR_IDS = {
    "base": 0,
    "shoulder": 1,
    "elbow": 2,
    "wrist_pitch": 3,
    "wrist_roll": 4,
    "gripper": 5
}
```

### PID Configuration (`mainfiles/pid_config.py`)

```python
# Fine-tune PID parameters for your specific robot
PID_PARAMS = {
    "base": {"kp": 1.0, "ki": 0.1, "kd": 0.05},
    "shoulder": {"kp": 1.2, "ki": 0.15, "kd": 0.08},
    # ... adjust for each joint
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Robot not responding**

   ```bash
   python find_robot_port.py  # Check connection
   python emergency_stop.py   # Reset if needed
   ```

2. **Permission denied on USB port (Linux/macOS)**

   ```bash
   sudo chmod 666 /dev/ttyUSB0
   # or add user to dialout group
   sudo usermod -a -G dialout $USER
   ```

3. **Import errors**

   ```bash
   pip install -r requirements.txt
   python test_numpy_fix.py
   ```

### Emergency Procedures

If the robot becomes unresponsive:

1. Run `python emergency_stop.py`
2. If that fails, manually power off the robot
3. Check all connections before restarting

## 📁 Project Structure

```text
advXRobotControl/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.py                   # Main configuration
├── .gitignore                  # Git ignore rules
│
├── Core Scripts/
│   ├── simple_robot_control.py # Basic control interface
│   ├── enable_motors.py        # Motor initialization
│   ├── disable_all_motors.py   # Safe shutdown
│   └── emergency_stop.py       # Emergency procedures
│
├── mainfiles/                  # Advanced features
│   ├── main.py                 # Integrated control system
│   ├── pid_controller.py       # PID implementation
│   ├── auto_face_tracking.py   # Face tracking
│   ├── visual_feedback_system.py # Computer vision
│   └── positions/              # Saved positions
│
└── Utilities/
    ├── find_robot_port.py      # Port detection
    ├── quick_robot_setup.py    # Quick setup
    └── test_numpy_fix.py       # Compatibility testing
```

## 🤝 Contributing

This project was created during AdventureX 2024. Feel free to contribute by:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏅 AdventureX Team

Developed with ❤️ by our team during AdventureX 2024 - China's premier hackathon where innovation meets execution in just 48 hours.

---

**⚠️ Safety Notice**: Always ensure the robot arm has adequate clearance and never leave it running unattended. Use emergency stop procedures if anything goes wrong.

**📞 Support**: For issues specific to AdventureX or this implementation, please open an issue in this repository.
