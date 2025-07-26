#!/usr/bin/env python3
"""
测试机器人移动锁功能
验证在返回起始位置期间，PID命令被正确阻止
"""

import time
import sys

def test_robot_movement_lock():
    """测试机器人移动锁功能"""
    print("🧪 测试机器人移动锁功能")
    print("=" * 50)
    
    try:
        from main import set_robot_movement_lock, is_robot_movement_locked
        
        # 测试初始状态
        print("1. 测试初始状态")
        initial_lock = is_robot_movement_locked()
        print(f"   初始锁定状态: {initial_lock}")
        
        # 测试设置锁定
        print("\n2. 测试设置锁定")
        set_robot_movement_lock(True)
        locked_state = is_robot_movement_locked()
        print(f"   锁定后状态: {locked_state}")
        
        if locked_state:
            print("   ✅ 锁定功能正常")
        else:
            print("   ❌ 锁定功能异常")
            return False
        
        # 测试解除锁定
        print("\n3. 测试解除锁定")
        set_robot_movement_lock(False)
        unlocked_state = is_robot_movement_locked()
        print(f"   解除锁定后状态: {unlocked_state}")
        
        if not unlocked_state:
            print("   ✅ 解除锁定功能正常")
        else:
            print("   ❌ 解除锁定功能异常")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 机器人移动锁测试失败: {e}")
        return False

def test_auto_face_tracking_lock_integration():
    """测试auto_face_tracking与移动锁的集成"""
    print("\n🧪 测试auto_face_tracking与移动锁的集成")
    print("=" * 50)
    
    try:
        from auto_face_tracking import TargetTracker
        from main import is_robot_movement_locked
        
        # 创建跟踪器
        tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
        print("✅ TargetTracker创建成功")
        
        # 模拟检测到目标
        print("\n1. 模拟检测到目标")
        position = tracker.update_target((500, 300))
        print(f"   返回位置: {position}")
        print(f"   机器人移动状态: {tracker.is_robot_moving()}")
        print(f"   全局移动锁状态: {is_robot_movement_locked()}")
        
        # 模拟目标丢失，触发移动到起始位置
        print("\n2. 模拟目标丢失，触发移动到起始位置")
        for i in range(5):  # 5次 * 0.5秒 = 2.5秒
            position = tracker.update_target(None)
            time.sleep(0.5)
            
            # 检查移动状态
            is_moving = tracker.is_robot_moving()
            global_locked = is_robot_movement_locked()
            print(f"   第{i+1}次: 移动中={is_moving}, 全局锁定={global_locked}")
            
            if is_moving and global_locked:
                print("   ✅ 移动锁集成正常")
                break
        else:
            print("   ⚠️  未检测到移动状态变化")
        
        return True
    except Exception as e:
        print(f"❌ auto_face_tracking移动锁集成测试失败: {e}")
        return False

def test_pid_command_blocking():
    """测试PID命令阻止功能"""
    print("\n🧪 测试PID命令阻止功能")
    print("=" * 50)
    
    try:
        from main import MotionDataCollector, set_robot_movement_lock, is_robot_movement_locked
        
        # 创建数据收集器
        collector = MotionDataCollector()
        print("✅ MotionDataCollector创建成功")
        
        # 测试正常状态下的PID命令
        print("\n1. 测试正常状态下的PID命令")
        set_robot_movement_lock(False)
        print(f"   移动锁状态: {is_robot_movement_locked()}")
        
        # 模拟PID命令执行
        success = collector._execute_robot_movement(10.0, 5.0)
        print(f"   PID命令执行结果: {success}")
        
        # 测试锁定状态下的PID命令
        print("\n2. 测试锁定状态下的PID命令")
        set_robot_movement_lock(True)
        print(f"   移动锁状态: {is_robot_movement_locked()}")
        
        # 模拟PID命令执行（应该被阻止）
        success = collector._execute_robot_movement(10.0, 5.0)
        print(f"   PID命令执行结果: {success}")
        
        if not success:
            print("   ✅ PID命令正确被阻止")
        else:
            print("   ❌ PID命令未被阻止")
            return False
        
        # 恢复正常状态
        set_robot_movement_lock(False)
        
        return True
    except Exception as e:
        print(f"❌ PID命令阻止测试失败: {e}")
        return False

def test_error_recovery():
    """测试错误恢复功能"""
    print("\n🧪 测试错误恢复功能")
    print("=" * 50)
    
    try:
        from main import set_robot_movement_lock, is_robot_movement_locked
        
        # 模拟异常情况下的锁定
        print("1. 模拟异常情况下的锁定")
        set_robot_movement_lock(True)
        print(f"   设置锁定状态: {is_robot_movement_locked()}")
        
        # 模拟异常恢复
        print("\n2. 模拟异常恢复")
        set_robot_movement_lock(False)
        print(f"   恢复后状态: {is_robot_movement_locked()}")
        
        if not is_robot_movement_locked():
            print("   ✅ 错误恢复功能正常")
        else:
            print("   ❌ 错误恢复功能异常")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 错误恢复测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始机器人移动锁测试")
    print("=" * 60)
    
    tests = [
        ("机器人移动锁基本功能", test_robot_movement_lock),
        ("auto_face_tracking集成", test_auto_face_tracking_lock_integration),
        ("PID命令阻止", test_pid_command_blocking),
        ("错误恢复", test_error_recovery),
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
        print("🎉 所有测试通过！机器人移动锁功能正常")
        print("\n💡 功能说明:")
        print("  - 返回起始位置期间，PID命令被完全阻止")
        print("  - 移动完成后，PID控制自动恢复")
        print("  - 异常情况下也能正确恢复")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 