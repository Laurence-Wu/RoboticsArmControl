#!/usr/bin/env python3
"""
测试简化的流程：2秒后直接移动到起始位置，然后重新开始识别
"""

import time
from auto_face_tracking import TargetTracker

def test_simplified_flow():
    """测试简化的流程"""
    print("🧪 测试简化流程：2秒后直接移动到起始位置")
    print("=" * 60)
    
    # 创建目标跟踪器
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    print(f"起始位置文件: {tracker.start_position_file}")
    
    # 模拟检测到目标
    print("\n1. 检测到目标...")
    position = tracker.update_target((500, 300))
    print(f"   返回位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    
    # 模拟目标丢失2.5秒（应该触发移动到起始位置）
    print("\n2. 目标丢失2.5秒...")
    start_time = time.time()
    for i in range(6):  # 6次 * 0.5秒 = 3秒
        position = tracker.update_target(None)
        elapsed = time.time() - start_time
        status = tracker.get_status()
        print(f"   第{i+1}次: 经过{elapsed:.1f}秒, 位置: {position}, 状态: {status}")
        time.sleep(0.5)
    
    # 模拟重新检测到目标
    print("\n3. 重新检测到目标...")
    position = tracker.update_target((600, 400))
    print(f"   返回位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    
    # 验证状态重置
    print("\n4. 验证状态重置...")
    print(f"   target_lost: {tracker.target_lost}")
    print(f"   is_locked: {tracker.is_locked}")
    print(f"   locked_time: {tracker.locked_time}")
    print(f"   return_to_center_mode: {tracker.return_to_center_mode}")

def test_immediate_restart():
    """测试立即重新开始识别"""
    print("\n🧪 测试立即重新开始识别")
    print("=" * 50)
    
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    # 模拟完整流程
    print("1. 检测到目标")
    tracker.update_target((100, 100))
    
    print("2. 目标丢失2.5秒")
    for i in range(5):
        tracker.update_target(None)
        time.sleep(0.5)
    
    print("3. 移动完成后立即检测新目标")
    position = tracker.update_target((200, 200))
    print(f"   新目标位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    
    # 验证可以立即响应新目标
    print("4. 验证立即响应")
    position = tracker.update_target((300, 300))
    print(f"   响应新位置: {position}")
    print(f"   状态: {tracker.get_status()}")

if __name__ == "__main__":
    print("🚀 测试简化流程")
    print("=" * 60)
    
    # 测试基本流程
    test_simplified_flow()
    
    # 测试立即重新开始
    test_immediate_restart()
    
    print("\n✅ 测试完成") 