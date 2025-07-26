#!/usr/bin/env python3
"""
测试直接调用move_to_json函数
"""

import time
from auto_face_tracking import TargetTracker

def test_direct_move_to_json():
    """测试直接调用move_to_json函数"""
    print("🧪 测试直接调用move_to_json函数")
    print("=" * 50)
    
    # 创建目标跟踪器
    tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
    
    print(f"起始位置文件: {tracker.start_position_file}")
    
    # 模拟检测到目标
    print("\n1. 检测到目标...")
    position = tracker.update_target((500, 300))
    print(f"   返回位置: {position}")
    
    # 模拟目标丢失2.5秒（应该触发移动到起始位置）
    print("\n2. 目标丢失2.5秒...")
    start_time = time.time()
    for i in range(6):  # 6次 * 0.5秒 = 3秒
        position = tracker.update_target(None)
        elapsed = time.time() - start_time
        print(f"   第{i+1}次: 经过{elapsed:.1f}秒, 位置: {position}")
        time.sleep(0.5)
    
    # 模拟重新检测到目标
    print("\n3. 重新检测到目标...")
    position = tracker.update_target((600, 400))
    print(f"   返回位置: {position}")

def test_move_to_json_import():
    """测试move_to_json模块导入"""
    print("\n🧪 测试move_to_json模块导入")
    print("=" * 50)
    
    try:
        from move_to_json import move_to_json_positions
        print("✅ 成功导入move_to_json模块")
        
        # 测试函数调用
        json_path = "positions/startPosition.json"
        print(f"测试调用: move_to_json_positions({json_path})")
        
        # 注意：这里只是测试导入，不实际移动机器人
        print("✅ 函数导入成功，可以正常调用")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    print("🚀 测试直接调用move_to_json功能")
    print("=" * 60)
    
    # 测试模块导入
    import_ok = test_move_to_json_import()
    
    if import_ok:
        # 测试直接调用功能
        test_direct_move_to_json()
    
    print("\n✅ 测试完成") 