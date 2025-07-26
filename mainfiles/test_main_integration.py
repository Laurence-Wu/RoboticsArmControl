#!/usr/bin/env python3
"""
测试main.py中的PID暂停集成
"""

import time
from auto_face_tracking import target_tracker

def test_main_integration():
    """测试main.py中的PID暂停集成"""
    print("🧪 测试main.py中的PID暂停集成")
    print("=" * 50)
    
    # 模拟main.py中的检查逻辑
    print("模拟main.py中的PID控制检查:")
    
    # 正常状态
    target_tracker.update_target((100, 100))
    if target_tracker.is_robot_moving():
        print("❌ 错误：正常跟踪时不应该暂停PID")
    else:
        print("✅ 正常：正常跟踪时PID控制继续")
    
    # 手动设置移动状态（模拟触发移动）
    print("\n手动设置移动状态...")
    target_tracker.is_moving_to_start = True
    target_tracker.movement_start_time = time.time()
    
    if target_tracker.is_robot_moving():
        print("✅ 正常：移动时PID控制应该暂停")
        print(f"   状态: {target_tracker.get_status()}")
    else:
        print("❌ 错误：移动时应该暂停PID")
    
    # 等待一段时间
    print("\n等待2秒...")
    time.sleep(2)
    
    if target_tracker.is_robot_moving():
        print("✅ 正常：移动中状态持续")
        print(f"   状态: {target_tracker.get_status()}")
    else:
        print("❌ 错误：移动状态应该持续")
    
    # 移动完成
    print("\n模拟移动完成...")
    target_tracker.is_moving_to_start = False
    target_tracker.movement_start_time = None
    
    if target_tracker.is_robot_moving():
        print("❌ 错误：移动完成后不应该暂停PID")
    else:
        print("✅ 正常：移动完成后PID控制恢复")
        print(f"   状态: {target_tracker.get_status()}")

if __name__ == "__main__":
    print("🚀 测试main.py中的PID暂停集成")
    print("=" * 60)
    
    test_main_integration()
    
    print("\n✅ 测试完成") 