# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:03:22 2018
Modified to use YOLO for face detection with 2s lock

@author: James Wu
"""

import cv2
import numpy as np
import os
import time
import subprocess

# YOLO模型相关导入
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO库可用")
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO库不可用，请安装: pip install ultralytics")

#==============================================================================
#   YOLO人脸检测器初始化
#==============================================================================
class YOLOFaceDetector:
    def __init__(self):
        """初始化YOLO人脸检测器"""
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO库未安装，请运行: pip install ultralytics")
        
        try:
            # 使用YOLOv8n模型，可以检测人脸（person类别）
            self.model = YOLO('yolov8n.pt')  # 自动下载预训练模型
            print("✅ YOLO模型加载成功")
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            raise
    
    def detect_faces(self, frame):
        """
        使用YOLO检测人脸
        返回: [(x, y, w, h), ...] 人脸边界框列表
        """
        try:
            # YOLO推理
            results = self.model(frame, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取类别ID和置信度
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # 检测人（class_id=0）且置信度足够高
                        if class_id == 0 and confidence > 0.5:  # 0是person类别
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            
                            # 估算人脸区域（通常在人体上部1/4区域）
                            face_h = h // 4
                            face_w = w // 2
                            face_x = x + (w - face_w) // 2
                            face_y = y + h // 8  # 稍微往下一点
                            
                            if face_w > 30 and face_h > 30:  # 过滤太小的检测
                                faces.append((face_x, face_y, face_w, face_h))
            
            return faces
            
        except Exception as e:
            print(f"YOLO检测出错: {e}")
            return []

#==============================================================================
#   目标跟踪状态管理类（修改为2秒）
#==============================================================================
class TargetTracker:
    def __init__(self, lock_duration=2.0, movement_threshold=30, start_position_file="positions/startPosition.json"):
        """
        目标跟踪管理器
        :param lock_duration: 目标丢失后的锁定时间（秒）- 改为2秒
        :param movement_threshold: 目标移动的最小阈值（像素）
        """
        self.lock_duration = lock_duration
        self.movement_threshold = movement_threshold
        
        # 当前锁定的目标位置
        self.locked_position = (960, 540)  # 默认中心位置 (1920x1080)
        self.locked_time = None  # 目标锁定开始时间
        self.last_detection_time = time.time()  # 最后一次检测到目标的时间
        
        # 目标状态
        self.target_lost = False
        self.is_locked = False
        
        # 智能回中参数
        self.center_position = (960, 540)  # 1920x1080的中心点
        self.return_to_center_mode = False
        self.center_return_speed = 0.1  # 回中速度系数
        
        self.start_position_file = start_position_file
        
        # 移动状态控制
        self.is_moving_to_start = False  # 是否正在移动到起始位置
        self.movement_start_time = None  # 移动开始时间
        self.movement_timeout = 10.0  # 移动超时时间（秒）
        
        print(f"🎯 目标跟踪器初始化 - 锁定时间: {lock_duration}秒, 移动阈值: {movement_threshold}像素")
    
    def _move_to_start_position(self):
        json_path = self.start_position_file
        try:
            print(f"🏠 移动到起始位置: {json_path}")
            
            # 直接导入并调用move_to_json函数
            from move_to_json import move_to_json_positions
            
            # 调用函数
            success = move_to_json_positions(json_path, speed=100, verbose=False)
            
            if success:
                print("✅ 成功移动到起始位置")
                return True
            else:
                print("❌ 移动到起始位置失败")
                return False
        except ImportError as e:
            print(f"❌ 无法导入move_to_json模块: {e}")
            return False
        except Exception as e:
            print(f"❌ 移动机器人时出错: {e}")
            return False

    def update_target(self, detected_position):
        """
        更新目标位置
        :param detected_position: 检测到的位置 (x, y) 或 None
        :return: 最终的目标位置 (x, y)
        """
        current_time = time.time()
        
        if detected_position is not None:
            # 检测到目标
            self.last_detection_time = current_time
            
            if self.target_lost:
                # 从丢失状态恢复
                print("🎯 目标重新找到，解除锁定")
                self.target_lost = False
                self.is_locked = False
                self.locked_time = None
                self.return_to_center_mode = False
            
            # 检查是否需要移动（避免小幅度抖动）
            distance = self._calculate_distance(detected_position, self.locked_position)
            
            if distance > self.movement_threshold or not self.is_locked:
                print(f"🎯 目标移动 {distance:.1f}像素 -> 更新位置: {detected_position}")
                self.locked_position = detected_position
                return detected_position
            else:
                print(f"🎯 目标移动较小 ({distance:.1f}像素) -> 保持锁定位置: {self.locked_position}")
                return self.locked_position
                
        else:
            # 没有检测到目标
            if not self.target_lost:
                # 刚刚丢失目标
                print(f"⚠️  目标丢失，开始锁定 {self.lock_duration}秒")
                self.target_lost = True
                self.is_locked = True
                self.locked_time = current_time
            
            # 检查锁定时间是否过期
            if self.locked_time and (current_time - self.locked_time) > self.lock_duration:
                if (current_time - self.last_detection_time) > self.lock_duration:
                    print("🔓 锁定时间到期，目标确实丢失，移动到起始位置")
                    self._move_to_start_position()
                    # 重置所有状态，立即重新开始识别
                    self.target_lost = False
                    self.is_locked = False
                    self.locked_time = None
                    self.return_to_center_mode = False
                    print("🔄 状态重置，重新开始识别和运动")
            
            if self.return_to_center_mode:
                # 智能回中：逐渐向中心移动，而不是直接跳到中心
                return self._smart_return_to_center()
            else:
                # 锁定期间保持最后位置
                remaining_time = max(0, self.lock_duration - (current_time - self.locked_time if self.locked_time else 0))
                print(f"🔒 目标锁定中... 剩余时间: {remaining_time:.1f}秒")
                return self.locked_position
    
    def _smart_return_to_center(self):
        """
        智能回中：逐渐向中心移动，保持闭环控制
        """
        # 计算当前位置到中心的向量
        dx = self.center_position[0] - self.locked_position[0]
        dy = self.center_position[1] - self.locked_position[1]
        
        # 计算距离
        distance = self._calculate_distance(self.locked_position, self.center_position)
        
        if distance < 5:  # 接近中心，停止回中
            print("🎯 已接近中心位置，停止回中")
            self.return_to_center_mode = False
            self.locked_position = self.center_position
            return self.center_position
        
        # 逐渐向中心移动
        new_x = self.locked_position[0] + dx * self.center_return_speed
        new_y = self.locked_position[1] + dy * self.center_return_speed
        
        self.locked_position = (new_x, new_y)
        print(f"🎯 智能回中: 距离中心 {distance:.1f}像素 -> 移动到 ({new_x:.1f}, {new_y:.1f})")
        
        return self.locked_position
    
    def _calculate_distance(self, pos1, pos2):
        """计算两点之间的距离"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_status(self):
        """获取当前状态"""
        if self.return_to_center_mode:
            return "RETURNING_TO_CENTER"
        elif self.target_lost:
            remaining_time = max(0, self.lock_duration - (time.time() - self.locked_time if self.locked_time else 0))
            return f"LOCKED ({remaining_time:.1f}s)"
        else:
            return "TRACKING"
    


# 初始化检测器和跟踪器
if YOLO_AVAILABLE:
    face_detector = YOLOFaceDetector()
    print("🚀 使用YOLO检测器")
else:
    # 备用Haar分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("🚀 使用Haar分类器作为备用")

# 全局目标跟踪器（改为2秒）
target_tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)

#==============================================================================
#   人脸检测函数（YOLO版本）
#==============================================================================
def Detection(frame):
    global target_tracker
    
    # 图像镜像处理 - 水平翻转
    frame = cv2.flip(frame, 1)  # 1表示水平翻转
    
    detected_position = None
    
    if YOLO_AVAILABLE:
        # 使用YOLO检测
        faces = face_detector.detect_faces(frame)
    else:
        # 使用Haar分类器备用检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(30, 30))
    
    if len(faces) > 0:
        # 过滤并选择最大的人脸
        filtered_faces = []
        for (x, y, w, h) in faces:
            if w > 50 and h > 50:  # 只保留较大的检测
                filtered_faces.append((x, y, w, h))
        
        if len(filtered_faces) > 0:
            # 找到最大的人脸（按面积）
            biggest_face = max(filtered_faces, key=lambda face: face[2] * face[3])
            x, y, w, h = biggest_face
            
            # 检测到的位置
            detected_x = x + w // 2
            detected_y = y + h // 2
            detected_position = (detected_x, detected_y)
            
            # 绘制检测到的人脸
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (detected_x, detected_y), 8, (0, 255, 0), -1)  # 绿色圆点：检测位置
            
            # 添加检测方法标识
            method = "YOLO" if YOLO_AVAILABLE else "Haar"
            cv2.putText(frame, method, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"✅ {method}检测到人脸: 位置({x}, {y}), 尺寸({w}x{h}), 中心坐标({detected_x}, {detected_y})")
    
    # 通过目标跟踪器获取最终目标位置
    final_position = target_tracker.update_target(detected_position)
    centroid_X, centroid_Y = final_position
    
    # 绘制最终目标位置（跟踪目标）
    cv2.circle(frame, (centroid_X, centroid_Y), 12, (255, 0, 0), -1)  # 蓝色大圆点：跟踪目标
    
    # 如果检测位置和跟踪位置不同，绘制连接线
    if detected_position and detected_position != final_position:
        cv2.line(frame, detected_position, final_position, (255, 255, 0), 2)  # 黄色连接线
    
    #==========================================================================
    #     绘制参考线和状态信息
    #==========================================================================
    # 绘制画面中心十字线 (1920x1080)
    cv2.line(frame, (950, 540), (970, 540), (0, 255, 255), 2)  # 水平线
    cv2.line(frame, (960, 530), (960, 550), (0, 255, 255), 2)  # 垂直线
    
    # 绘制参考矩形 (1920x1080)
    x = 0; y = 0; w = 960; h = 540;
    rectangle_pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.int32)
    cv2.polylines(frame, [rectangle_pts], True, (0,255,0), 2)
    
    x2 = 960; y2 = 540;
    rectangle_pts2 = np.array([[x2,y2],[x2+w,y2],[x2+w,y2+h],[x2,y2+h]], np.int32)
    cv2.polylines(frame, [rectangle_pts2], True, (0,255,0), 2)

    # 显示状态信息
    status = target_tracker.get_status()
    detector_type = "YOLO" if YOLO_AVAILABLE else "Haar"
    
    cv2.putText(frame, f'Target: ({centroid_X}, {centroid_Y})', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Status: {status}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Detector: {detector_type}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 图例
    cv2.putText(frame, 'Green: Detection', (10, frame.shape[0] - 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, 'Blue: Tracking Target', (10, frame.shape[0] - 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, 'Yellow: Center Cross', (10, frame.shape[0] - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, 'Cyan: Connection Line', (10, frame.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f'Lock Duration: 2.0s', (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # 尝试显示
    try:
        cv2.imshow('YOLO Face Tracking (2s Lock + Mirror)', frame)
    except cv2.error:
        pass
    
    return centroid_X, centroid_Y

#==============================================================================
#   主函数入口
#==============================================================================
if __name__  == "__main__":
    print("🚀 启动YOLO人脸跟踪系统（2秒锁定 + 镜像显示）")
    print("=" * 60)
    print("💡 功能说明：")
    print("  - 使用YOLO算法进行人脸检测")
    print("  - 图像已水平镜像处理，显示更直观")
    print("  - 绿色圆点：实时检测到的人脸位置")
    print("  - 蓝色圆点：最终跟踪目标位置")
    print("  - 黄色连线：检测位置与跟踪位置的连接")
    print("  - 目标丢失后锁定2秒，避免频繁移动")
    print("=" * 60)

    # 如果YOLO不可用，显示安装说明
    if not YOLO_AVAILABLE:
        print("⚠️  YOLO不可用，使用Haar分类器")
        print("💡 要使用YOLO，请运行:")
        print("   pip install ultralytics")
        print("=" * 60)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    # Check and print resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头分辨率: {int(width)} x {int(height)}")

    if not cap.isOpened():
        print("尝试摄像头1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        exit(1)
        
    print("✅ 摄像头打开成功")
    print("💡 按ESC键退出程序，R键重置跟踪器")
    print("=" * 60)

    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            X, Y = Detection(frame)  # 执行人脸检测（YOLO + 2秒锁定）
            
            # 每60帧显示一次检测结果统计
            if frame_count % 60 == 0:
                status = target_tracker.get_status()
                detector_type = "YOLO" if YOLO_AVAILABLE else "Haar"
                print(f"📊 帧数: {frame_count}, 跟踪坐标: ({X}, {Y}), 状态: {status}, 检测器: {detector_type}")
            
            frame_count += 1
            
            # 检查退出按键
            try:
                key = cv2.waitKey(5) & 0xFF
                if key == 27:  # ESC键退出
                    print("\n👋 用户按ESC键退出")
                    break
                elif key == ord('r'):  # R键重置目标
                    target_tracker = TargetTracker(lock_duration=2.0, movement_threshold=30)
                    print("🔄 目标跟踪器已重置（2秒锁定）")
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n👋 用户按Ctrl+C退出")
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
    finally:
        print("\n🔧 清理资源...")
        try:
            cv2.destroyAllWindows()
        except:
            pass
        cap.release()
        print("✅ 摄像头已关闭")
        print("✅ 程序退出完成")