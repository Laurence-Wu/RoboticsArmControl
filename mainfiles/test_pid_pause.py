#!/usr/bin/env python3
"""
测试PID控制暂停功能
"""

import time
from auto_face_tracking import TargetTracker

def test_pid_pause():
    """测试PID控制暂停功能"""
    print("🧪 测试PID控制暂停功能")
    print("=" * 50)
    
    # 创建目标跟踪器
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    print(f"起始位置文件: {tracker.start_position_file}")
    
    # 模拟检测到目标
    print("\n1. 检测到目标...")
    position = tracker.update_target((500, 300))
    print(f"   返回位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    print(f"   机器人移动中: {tracker.is_robot_moving()}")
    
    # 模拟目标丢失2.5秒（应该触发移动到起始位置）
    print("\n2. 目标丢失2.5秒...")
    start_time = time.time()
    for i in range(6):  # 6次 * 0.5秒 = 3秒
        position = tracker.update_target(None)
        elapsed = time.time() - start_time
        status = tracker.get_status()
        is_moving = tracker.is_robot_moving()
        print(f"   第{i+1}次: 经过{elapsed:.1f}秒, 位置: {position}, 状态: {status}, 移动中: {is_moving}")
        
        # 检查是否在移动中
        if is_moving:
            print(f"   ⏸️  PID控制应该暂停，机器人移动中...")
        
        time.sleep(0.5)
    
    # 等待移动完成
    print("\n3. 等待移动完成...")
    while tracker.is_robot_moving():
        position = tracker.update_target(None)
        status = tracker.get_status()
        is_moving = tracker.is_robot_moving()
        print(f"   移动中... 状态: {status}, 移动中: {is_moving}")
        time.sleep(0.5)
    
    # 模拟重新检测到目标
    print("\n4. 重新检测到目标...")
    position = tracker.update_target((600, 400))
    print(f"   返回位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    print(f"   机器人移动中: {tracker.is_robot_moving()}")

def test_main_integration():
    """测试与main.py的集成"""
    print("\n🧪 测试与main.py的集成")
    print("=" * 50)
    
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    # 模拟main.py中的检查逻辑
    print("模拟main.py中的PID控制检查:")
    
    # 正常状态
    tracker.update_target((100, 100))
    if tracker.is_robot_moving():
        print("❌ 错误：正常跟踪时不应该暂停PID")
    else:
        print("✅ 正常：正常跟踪时PID控制继续")
    
    # 移动状态
    tracker.is_moving_to_start = True
    tracker.movement_start_time = time.time()
    if tracker.is_robot_moving():
        print("✅ 正常：移动时PID控制应该暂停")
    else:
        print("❌ 错误：移动时应该暂停PID")
    
    # 移动完成
    tracker.is_moving_to_start = False
    tracker.movement_start_time = None
    if tracker.is_robot_moving():
        print("❌ 错误：移动完成后不应该暂停PID")
    else:
        print("✅ 正常：移动完成后PID控制恢复")

if __name__ == "__main__":
    print("🚀 测试PID控制暂停功能")
    print("=" * 60)
    
    # 测试基本功能
    test_pid_pause()
    
    # 测试集成
    test_main_integration()
    
    print("\n✅ 测试完成") 