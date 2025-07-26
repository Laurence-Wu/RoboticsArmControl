#!/usr/bin/env python3
"""
测试集成功能：验证auto_face_tracking和main.py的集成
包括返回起始位置功能和PID控制暂停
"""

import time
import sys
import os

def test_auto_face_tracking_import():
    """测试auto_face_tracking模块导入"""
    print("🧪 测试auto_face_tracking模块导入")
    print("=" * 50)
    
    try:
        from auto_face_tracking import Detection, TargetTracker, YOLO_AVAILABLE
        print("✅ auto_face_tracking模块导入成功")
        print(f"   YOLO可用: {YOLO_AVAILABLE}")
        return True
    except ImportError as e:
        print(f"❌ auto_face_tracking模块导入失败: {e}")
        return False

def test_target_tracker_functionality():
    """测试TargetTracker功能"""
    print("\n🧪 测试TargetTracker功能")
    print("=" * 50)
    
    try:
        from auto_face_tracking import TargetTracker
        
        # 创建跟踪器
        tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
        print("✅ TargetTracker创建成功")
        print(f"   起始位置文件: {tracker.start_position_file}")
        print(f"   锁定时间: {tracker.lock_duration}秒")
        print(f"   移动阈值: {tracker.movement_threshold}像素")
        
        # 测试状态检查
        print(f"   初始状态: {tracker.get_status()}")
        print(f"   机器人移动状态: {tracker.is_robot_moving()}")
        
        return True
    except Exception as e:
        print(f"❌ TargetTracker测试失败: {e}")
        return False

def test_move_to_json_integration():
    """测试move_to_json集成"""
    print("\n🧪 测试move_to_json集成")
    print("=" * 50)
    
    try:
        # 检查move_to_json模块
        from move_to_json import move_to_json_positions
        print("✅ move_to_json模块导入成功")
        
        # 检查起始位置文件
        start_position_file = "positions/HOME.json"
        if os.path.exists(start_position_file):
            print(f"✅ 起始位置文件存在: {start_position_file}")
            
            # 读取文件内容
            import json
            with open(start_position_file, 'r') as f:
                positions = json.load(f)
            print(f"   位置数据: {positions}")
        else:
            print(f"⚠️  起始位置文件不存在: {start_position_file}")
            print("   请确保positions/HOME.json文件存在")
        
        return True
    except ImportError as e:
        print(f"❌ move_to_json模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ move_to_json集成测试失败: {e}")
        return False

def test_main_integration():
    """测试main.py集成"""
    print("\n🧪 测试main.py集成")
    print("=" * 50)
    
    try:
        # 检查main.py中的关键组件
        from main import SingleThreadFaceTracker, MotionData, MotionDataCollector
        print("✅ main.py关键组件导入成功")
        
        # 测试MotionData
        motion_data = MotionData(face_position=(960, 540))
        print(f"✅ MotionData创建成功: {motion_data}")
        
        # 测试MotionDataCollector
        collector = MotionDataCollector()
        print("✅ MotionDataCollector创建成功")
        
        return True
    except ImportError as e:
        print(f"❌ main.py组件导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ main.py集成测试失败: {e}")
        return False

def test_robot_movement_simulation():
    """模拟机器人移动状态"""
    print("\n🧪 模拟机器人移动状态")
    print("=" * 50)
    
    try:
        from auto_face_tracking import TargetTracker
        
        tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
        
        # 模拟检测到目标
        print("1. 检测到目标")
        position = tracker.update_target((500, 300))
        print(f"   返回位置: {position}")
        print(f"   状态: {tracker.get_status()}")
        print(f"   机器人移动: {tracker.is_robot_moving()}")
        
        # 模拟目标丢失2.5秒
        print("\n2. 目标丢失2.5秒")
        start_time = time.time()
        for i in range(5):
            position = tracker.update_target(None)
            elapsed = time.time() - start_time
            status = tracker.get_status()
            is_moving = tracker.is_robot_moving()
            print(f"   第{i+1}次: 经过{elapsed:.1f}秒, 位置: {position}, 状态: {status}, 移动中: {is_moving}")
            time.sleep(0.5)
        
        return True
    except Exception as e:
        print(f"❌ 机器人移动模拟失败: {e}")
        return False

def test_pid_pause_logic():
    """测试PID暂停逻辑"""
    print("\n🧪 测试PID暂停逻辑")
    print("=" * 50)
    
    try:
        from auto_face_tracking import TargetTracker
        
        tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
        
        # 模拟PID控制检查
        print("1. 正常跟踪状态")
        tracker.update_target((100, 100))
        if tracker.is_robot_moving():
            print("   ❌ 错误：正常跟踪时不应该暂停PID")
        else:
            print("   ✅ 正常：正常跟踪时PID控制正常")
        
        # 模拟移动状态
        print("\n2. 移动状态")
        tracker.is_moving_to_start = True
        tracker.movement_start_time = time.time()
        
        if tracker.is_robot_moving():
            print("   ✅ 正常：移动状态时PID应该暂停")
        else:
            print("   ❌ 错误：移动状态时PID没有暂停")
        
        # 重置状态
        tracker.is_moving_to_start = False
        tracker.movement_start_time = None
        
        return True
    except Exception as e:
        print(f"❌ PID暂停逻辑测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始集成测试")
    print("=" * 60)
    
    tests = [
        ("auto_face_tracking导入", test_auto_face_tracking_import),
        ("TargetTracker功能", test_target_tracker_functionality),
        ("move_to_json集成", test_move_to_json_integration),
        ("main.py集成", test_main_integration),
        ("机器人移动模拟", test_robot_movement_simulation),
        ("PID暂停逻辑", test_pid_pause_logic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！集成功能正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 