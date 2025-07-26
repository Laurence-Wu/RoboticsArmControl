#!/usr/bin/env python3
"""
测试视觉识别暂停功能
"""

import time
from auto_face_tracking import TargetTracker

def test_visual_pause():
    """测试视觉识别暂停功能"""
    print("🧪 测试视觉识别暂停功能")
    print("=" * 50)
    
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
        
        # 检查是否在移动中
        if tracker.is_moving_to_start:
            print(f"   ⏸️  视觉识别已暂停，等待机器人移动完成...")
        
        time.sleep(0.5)
    
    # 等待移动完成
    print("\n3. 等待移动完成...")
    while tracker.is_moving_to_start:
        position = tracker.update_target(None)
        status = tracker.get_status()
        print(f"   移动中... 状态: {status}")
        time.sleep(0.5)
    
    # 模拟重新检测到目标
    print("\n4. 重新检测到目标...")
    position = tracker.update_target((600, 400))
    print(f"   返回位置: {position}")
    print(f"   状态: {tracker.get_status()}")
    
    # 测试超时机制
    print("\n5. 测试超时机制...")
    tracker.is_moving_to_start = True
    tracker.movement_start_time = time.time() - 11  # 设置超过10秒
    
    position = tracker.update_target((700, 500))
    print(f"   超时后位置: {position}")
    print(f"   状态: {tracker.get_status()}")

def test_status_display():
    """测试状态显示"""
    print("\n🧪 测试状态显示")
    print("=" * 50)
    
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    # 正常跟踪状态
    tracker.update_target((100, 100))
    print(f"正常跟踪: {tracker.get_status()}")
    
    # 锁定状态
    tracker.update_target(None)
    print(f"锁定状态: {tracker.get_status()}")
    
    # 移动状态
    tracker.is_moving_to_start = True
    tracker.movement_start_time = time.time()
    print(f"移动状态: {tracker.get_status()}")
    
    # 移动中状态（经过时间）
    time.sleep(1)
    print(f"移动中状态: {tracker.get_status()}")

if __name__ == "__main__":
    print("🚀 测试视觉识别暂停功能")
    print("=" * 60)
    
    # 测试状态显示
    test_status_display()
    
    # 测试完整流程
    test_visual_pause()
    
    print("\n✅ 测试完成") 