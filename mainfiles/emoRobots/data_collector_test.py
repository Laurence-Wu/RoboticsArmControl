# Windows版本 - Emotiv Cortex Performance Metrics数据收集脚本
"""
Performance Metrics Data Collector for Emotiv Cortex API with WebSocket Server

This script collects only performance metrics data and outputs in the required format:
time, attention, relaxation, engagement, excitement, stress, interest, composite

Usage:
    python data_collector_test.py

Requirements:
    - Emotiv headset connected via Bluetooth or USB dongle
    - Emotiv Launcher running
    - Valid app credentials (client ID and secret)
    - pip install websockets matplotlib
"""

import sys
import time
import json
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime
from cortex import Cortex
from dotenv import load_dotenv
from collections import deque
import threading
from scipy.signal import find_peaks
import asyncio
import websockets
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetricsCollector:
    """
    Performance Metrics数据收集器，集成实时算法分析和WebSocket服务器
    只收集met数据流，输出标准化格式
    """
    
    def __init__(self, app_client_id, app_client_secret, **kwargs):
        print("=" * 60)
        print("初始化 Emotiv Performance Metrics 数据收集器")
        print("=" * 60)
        
        # 算法权重配置（与alg.py保持一致）
        self.weights = {
            'attention'  :  0.35,
            'engagement' :  0.25,
            'excitement' :  0.15,
            'interest'   :  0.15,
            'stress'     :  0.15,
            'relaxation' : -0.10,
        }
        
        # 数据缓冲区
        self.performance_buffer = deque(maxlen=200)
        self.classification_results = []
        
        # 时间和分类配置
        self.classification_interval = 2.0  # 2秒分类一次
        self.last_classification_time = time.time()
        
        # 文件输出配置
        self.collection_start_time = None
        self.collection_duration = 60  # 默认60秒
        self.save_to_file = True
        self.output_directory = "collected_data"
        
        # 创建输出目录
        if self.save_to_file and not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # CSV文件设置
        self.csv_file = None
        self.csv_writer = None
        self.sample_count = 0
        self.data_labels = {}
        
        # WebSocket服务器配置
        self.websocket_clients = set()
        self.websocket_server = None
        self.websocket_port = 8765
        
        # 实时数据存储
        self.realtime_results = []
        self.frontend_data = []
        
        # 初始化Cortex连接
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=True, **kwargs)
        self._bind_event_handlers()
        
    def _bind_event_handlers(self):
        """绑定事件处理器"""
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(inform_error=self.on_inform_error)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_met_data=self.on_new_met_data)
        
    def start_collection(self, duration=60, headset_id=''):
        """开始数据收集"""
        self.collection_duration = duration
        
        print("\n🚀 启动 Performance Metrics 数据收集:")
        print(f"  - 数据流: met (performance metrics)")
        print(f"  - 收集时长: {duration} 秒")
        print(f"  - 分类间隔: {self.classification_interval} 秒")
        print(f"  - WebSocket端口: {self.websocket_port}")
        print(f"  - 头盔ID: {headset_id if headset_id else 'Auto-detect'}")
        
        if headset_id:
            self.c.set_wanted_headset(headset_id)
        
        # 启动WebSocket服务器
        self.start_websocket_server()
        
        print("\n⏳ 正在建立连接...")
        self.c.open()
        
    def start_websocket_server(self):
        """启动WebSocket服务器"""
        async def handle_client(websocket, path):
            
            """处理WebSocket客户端连接"""
            self.websocket_clients.add(websocket)
            print(f"🔗 WebSocket客户端已连接: {websocket.remote_address}")
            
            try:
                # 发送欢迎消息
                welcome_message = {
                    'type': 'connection',
                    'status': 'connected',
                    'message': 'Connected to Emotiv Data Collector'
                }
                await websocket.send(json.dumps(welcome_message))
                
                # 保持连接
                await websocket.wait_closed()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.discard(websocket)
                print(f"❌ WebSocket客户端已断开: {websocket.remote_address}")
        
        async def start_server():
            """启动服务器"""
            self.websocket_server = await websockets.serve(
                    handle_client, 
                    "0.0.0.0",  # 改为 0.0.0.0 允许外部访问
                    self.websocket_port
                )
            print(f"🌐 WebSocket服务器启动在 ws://0.0.0.0:{self.websocket_port}")
            
        # 在新线程中运行WebSocket服务器
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_server())
            loop.run_forever()
            
        websocket_thread = threading.Thread(target=run_websocket, daemon=True)
        websocket_thread.start()
        
    async def send_to_clients(self, message):
        """向所有WebSocket客户端发送消息"""
        if self.websocket_clients:
            # 创建消息副本以避免并发修改
            clients_copy = self.websocket_clients.copy()
            disconnected_clients = set()
            
            for client in clients_copy:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.warning(f"发送消息失败: {e}")
                    disconnected_clients.add(client)
            
            # 移除断开的客户端
            for client in disconnected_clients:
                self.websocket_clients.discard(client)
        
    def send_realtime_update(self, data):
        """发送实时数据更新"""
        message = {
            'type': 'attention_update',
            'timestamp': data['time'],
            'focused': data['focused'],
            'composite_score': data['composite_score'],
            'metrics': data['metrics']
        }
        
        # 在新的事件循环中发送
        def send_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_to_clients(message))
                loop.close()
            except Exception as e:
                logger.warning(f"异步发送失败: {e}")
        
        threading.Thread(target=send_async, daemon=True).start()
        
    def _setup_csv_writer(self):
        """设置CSV文件写入器"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.output_directory, f"performance_metrics_{timestamp}.csv")
        
        # 打开文件并创建写入器
        self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 写入标准化头部
        headers = ['time', 'attention', 'relaxation', 'engagement', 'excitement', 'stress', 'interest', 'composite']
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
        
        print(f"✓ Performance Metrics CSV文件创建: {filename}")
        
    def parse_met_labels(self, labels):
        """
        解析performance metrics标签，映射到标准化索引
        实际标签格式: ['attention.isActive', 'attention', 'eng.isActive', 'eng', 'exc.isActive', 'exc', 'lex', 'str.isActive', 'str', 'rel.isActive', 'rel', 'int.isActive', 'int']
        """
        label_mapping = {}
        
        for i, label in enumerate(labels):
            label_str = str(label).lower()
            
            # 映射各个指标的数值索引（跳过isActive字段）
            if label_str == 'attention':
                label_mapping['attention'] = i
                label_mapping['attention_active'] = i - 1  # isActive在前一列
            elif label_str == 'eng':  # engagement的缩写
                label_mapping['engagement'] = i
                label_mapping['engagement_active'] = i - 1
            elif label_str == 'exc':  # excitement的缩写
                label_mapping['excitement'] = i
                label_mapping['excitement_active'] = i - 1
            elif label_str == 'int':  # interest的缩写
                label_mapping['interest'] = i
                label_mapping['interest_active'] = i - 1
            elif label_str == 'str':  # stress的缩写
                label_mapping['stress'] = i
                label_mapping['stress_active'] = i - 1
            elif label_str == 'rel':  # relaxation的缩写
                label_mapping['relaxation'] = i
                label_mapping['relaxation_active'] = i - 1
                
        print(f"✓ 标签映射解析完成: {label_mapping}")
        return label_mapping
        
    def extract_metrics(self, timestamp, met_values):
        """
        从performance metrics数据中提取标准化指标
        """
        if not hasattr(self, 'label_mapping'):
            return None
            
        metrics = {
            'time': timestamp,
            'attention': 0.0,
            'relaxation': 0.0,
            'engagement': 0.0,
            'excitement': 0.0,
            'stress': 0.0,
            'interest': 0.0
        }
        
        # 根据标签映射提取数据
        for metric_name in ['attention', 'relaxation', 'engagement', 'excitement', 'stress', 'interest']:
            value_index = self.label_mapping.get(metric_name)
            active_index = self.label_mapping.get(f'{metric_name}_active')
            
            if value_index is not None and value_index < len(met_values):
                # 检查是否激活
                is_active = True  # 默认激活
                if active_index is not None and active_index < len(met_values):
                    active_value = met_values[active_index]
                    is_active = str(active_value).lower() == 'true'
                
                # 只有激活时才提取数值
                if is_active:
                    value = met_values[value_index]
                    if value is not None and str(value).strip() != '':
                        try:
                            metrics[metric_name] = float(value)
                        except (ValueError, TypeError):
                            metrics[metric_name] = 0.0
                    else:
                        metrics[metric_name] = 0.0
                else:
                    metrics[metric_name] = 0.0
            else:
                metrics[metric_name] = 0.0
        
        return metrics
        
    def calculate_composite_score(self, metrics):
        """
        计算综合唤醒度分数（与alg.py算法一致）
        """
        composite = (
            metrics['attention'] * self.weights['attention'] +
            metrics['engagement'] * self.weights['engagement'] +
            metrics['excitement'] * self.weights['excitement'] +
            metrics['interest'] * self.weights['interest'] +
            metrics['stress'] * self.weights['stress'] +
            metrics['relaxation'] * self.weights['relaxation']
        )
        return composite
        
    def write_metrics_to_csv(self, metrics):
        """将指标数据写入CSV文件"""
        if self.csv_writer and metrics:
            # 计算综合分数
            composite = self.calculate_composite_score(metrics)
            
            # 按照标准格式写入: time, attention, relaxation, engagement, excitement, stress, interest, composite
            row = [
                metrics['time'],
                metrics['attention'],
                metrics['relaxation'],
                metrics['engagement'],
                metrics['excitement'],
                metrics['stress'],
                metrics['interest'],
                composite
            ]
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.sample_count += 1
            
            # 显示有效数据
            valid_metrics = {k: v for k, v in metrics.items() if k != 'time' and v > 0}
            if valid_metrics:
                print(f"📊 样本 {self.sample_count}: 综合分数={composite:.3f}, 有效指标={valid_metrics}")
        
    def perform_classification(self):
        """执行实时分类分析 - 先平均再计算分数"""
        current_time = time.time()
        
        if current_time - self.last_classification_time >= self.classification_interval:
            if len(self.performance_buffer) >= 10:  # 至少需要0.5秒的数据
                # 获取最近2秒的数据
                recent_data = list(self.performance_buffer)[-40:]  # 假设20Hz采样率
                
                if recent_data:
                    # 收集所有指标值用于计算平均值
                    metrics_arrays = {
                        'attention': [],
                        'relaxation': [],
                        'engagement': [],
                        'excitement': [],
                        'stress': [],
                        'interest': []
                    }
                    
                    # 从缓冲区提取所有指标值
                    for timestamp, metrics in recent_data:
                        if metrics:
                            for metric_name in metrics_arrays.keys():
                                metrics_arrays[metric_name].append(metrics[metric_name])
                    
                    # 计算每个指标在2秒内的平均值
                    if any(len(values) > 0 for values in metrics_arrays.values()):
                        avg_metrics = {}
                        for metric_name, values in metrics_arrays.items():
                            if values:
                                avg_metrics[metric_name] = np.mean(values)
                            else:
                                avg_metrics[metric_name] = 0.0
                        
                        # 用平均值计算综合分数
                        composite_score = self.calculate_composite_score(avg_metrics)
                        
                        # 简单的阈值判断
                        threshold = 0.44  # 可调整的阈值
                        undisturbed = 1 if composite_score > threshold else 0
                        status = "🎯 专注" if undisturbed == 1 else "😴 分散"
                        
                        elapsed = current_time - self.collection_start_time
                        print(f"\n{status} | 时间: {elapsed:.1f}s | 综合分数: {composite_score:.3f}")
                        print(f"  平均指标 - 注意力: {avg_metrics['attention']:.2f} | 参与度: {avg_metrics['engagement']:.2f} | "
                            f"兴奋度: {avg_metrics['excitement']:.2f} | 兴趣: {avg_metrics['interest']:.2f} | "
                            f"压力: {avg_metrics['stress']:.2f} | 放松: {avg_metrics['relaxation']:.2f}")
                        
                        # 保存实时数据
                        realtime_data = {
                            'time': elapsed,
                            'focused': undisturbed,
                            'composite_score': composite_score,
                            'metrics': avg_metrics
                        }
                        self.realtime_results.append(realtime_data)
                        
                        # 发送到前端
                        self.send_realtime_update(realtime_data)
                        
            self.last_classification_time = current_time

    def generate_attention_plot(self):
        """生成专注状态条形图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            print("\n📊 生成专注状态分析图...")
            
            if not self.realtime_results:
                print("⚠️ 无数据可绘制")
                return
            
            # 提取数据
            times = [entry['time'] for entry in self.realtime_results]
            focused_states = [entry['focused'] for entry in self.realtime_results]
            scores = [entry['composite_score'] for entry in self.realtime_results]
            
            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 上图：专注状态条形图
            colors = ['#ff6b6b' if state == 0 else '#4ecdc4' for state in focused_states]
            bars = ax1.bar(times, [1]*len(times), color=colors, width=1.8, alpha=0.7)
            
            # 标注专注区域
            focused_segments = []
            start_time = None
            
            for i, (time, state) in enumerate(zip(times, focused_states)):
                if state == 1 and start_time is None:
                    start_time = time
                elif state == 0 and start_time is not None:
                    focused_segments.append((start_time, time))
                    start_time = None
            
            # 处理结尾的专注状态
            if start_time is not None:
                focused_segments.append((start_time, times[-1] + 2))
            
            # 在图上标注专注区域
            for start, end in focused_segments:
                duration = end - start
                ax1.add_patch(patches.Rectangle((start-1, 0), duration+2, 1.2, 
                                              facecolor='green', alpha=0.3, 
                                              edgecolor='green', linewidth=2))
                ax1.text(start + duration/2, 1.1, f'Focused {duration:.0f}s', 
                        ha='center', va='center', fontsize=10, fontweight='bold')
            
            ax1.set_ylim(0, 1.3)
            ax1.set_ylabel('Focus State', fontsize=12)
            ax1.set_title('Real-time Attention Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 添加图例
            legend_elements = [patches.Patch(color='#4ecdc4', label='Focused'),
                              patches.Patch(color='#ff6b6b', label='Distracted'),
                              patches.Patch(color='green', alpha=0.3, label='Focus Period')]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # 下图：综合分数曲线
            ax2.plot(times, scores, color='#2c3e50', linewidth=2, marker='o', markersize=4)
            ax2.axhline(y=0.44, color='red', linestyle='--', alpha=0.7, label='Focus Threshold')
            ax2.fill_between(times, scores, alpha=0.3, color='#3498db')
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Composite Score', fontsize=12)
            ax2.set_title('Composite Arousal Score Changes', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 保存图片
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(self.output_directory, f"attention_analysis_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 专注状态分析图已保存: {plot_filename}")
            
            # 打印统计信息
            total_time = times[-1] if times else 0
            focused_time = sum(end - start for start, end in focused_segments)
            focused_percentage = (focused_time / total_time * 100) if total_time > 0 else 0
            
            print(f"📈 专注统计:")
            print(f"  - 总时长: {total_time:.1f}秒")
            print(f"  - 专注时长: {focused_time:.1f}秒")
            print(f"  - 专注比例: {focused_percentage:.1f}%")
            print(f"  - 专注区间数: {len(focused_segments)}")
            
        except ImportError:
            print("⚠️ 需要安装matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"❌ 绘图失败: {e}")

    def subscribe_streams(self):
        """订阅performance metrics数据流"""
        print(f"\n📡 订阅数据流: met (performance metrics)")
        
        if self.save_to_file:
            self._setup_csv_writer()
        
        self.c.sub_request(['met'])
        
        self.collection_start_time = time.time()
        self.last_classification_time = self.collection_start_time
        
        print(f"📊 数据收集开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔄 实时算法分析已启动")
        print(f"🌐 前端网站: http://localhost:8000")
        
        # 定时停止收集
        stop_timer = threading.Timer(self.collection_duration, self.stop_collection)
        stop_timer.start()
        
    def stop_collection(self):
        """停止数据收集"""
        print(f"\n🛑 停止数据收集 ({self.collection_duration}秒)")
        
        self.c.unsub_request(['met'])
        
        if self.csv_file:
            self.csv_file.close()
            print(f"✓ CSV文件已保存: {self.sample_count} 个样本")
        
        # 生成图表
        if self.realtime_results:
            self.generate_attention_plot()
        
        self.print_collection_summary()
        self.c.close()
        
    def print_collection_summary(self):
        """打印收集总结"""
        print("\n" + "=" * 60)
        print("Performance Metrics 数据收集总结")
        print("=" * 60)
        
        print(f"  样本总数: {self.sample_count}")
        print(f"  收集时长: {self.collection_duration} 秒")
        
        if self.sample_count > 0:
            avg_rate = self.sample_count / self.collection_duration
            print(f"  平均采样率: {avg_rate:.1f} 样本/秒")
        
        print(f"  输出格式: time, attention, relaxation, engagement, excitement, stress, interest, composite")
    
    # 事件处理器
    def on_create_session_done(self, *args, **kwargs):
        print("✓ 会话创建成功")
        self.subscribe_streams()
        
    def on_new_data_labels(self, *args, **kwargs):
        data = kwargs.get('data')
        stream_name = data['streamName']
        labels = data['labels']
        
        if stream_name == 'met':
            self.data_labels[stream_name] = labels
            self.label_mapping = self.parse_met_labels(labels)
            print(f"✓ 接收到 {stream_name} 标签: {len(labels)} 通道")
    
    def on_new_met_data(self, *args, **kwargs):
        """处理新的performance metrics数据"""
        data = kwargs.get('data')
        timestamp = data['time']
        met_values = data['met']
        
        # 提取标准化指标
        metrics = self.extract_metrics(timestamp, met_values)
        
        if metrics:
            # 写入CSV文件
            self.write_metrics_to_csv(metrics)
            
            # 添加到缓冲区用于实时分析
            self.performance_buffer.append((timestamp, metrics))
            
            # 执行分类分析
            self.perform_classification()
    
    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(f"❌ 错误: {error_data}")
        
        if isinstance(error_data, dict):
            error_code = error_data.get('code', 0)
            error_message = error_data.get('message', '')
            
            if error_code == -32142:
                print("\n💡 解决建议：应用程序需要设置为 'Published' 状态")
            elif "headset" in error_message.lower():
                print("\n💡 解决建议：检查头盔连接状态")


def create_web_files():
    """创建前端网站文件"""
    web_dir = "web"
    if not os.path.exists(web_dir):
        os.makedirs(web_dir)
# 在create_web_files()函数中修改html_content部分：

    html_content = '''<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
        <title>Focus Status</title>
        <link rel="stylesheet" href="style.css">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black">
    </head>
    <body>
        <!-- 全屏按钮 -->
        <div id="fullscreenBtn" class="fullscreen-btn" title="Toggle Fullscreen">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
            </svg>
        </div>
        
        <div class="container">
            <!-- 左侧状态指示区域 -->
            <div class="status-section">
                <div id="statusIndicator" class="status-indicator focused"></div>
                <div id="statusText" class="status-text focused">Do Not Disturb</div>
                <div id="connectionStatus" class="connection-status">Connecting...</div>
            </div>
            
            <!-- 右侧数据显示区域 -->
            <div class="metrics-section">
                <div class="metrics-grid">
                    <div class="metric-card attention">
                        <span class="metric-label">Attention</span>
                        <span id="attentionValue" class="metric-value">0.00</span>
                    </div>
                    <div class="metric-card engagement">
                        <span class="metric-label">Engagement</span>
                        <span id="engagementValue" class="metric-value">0.00</span>
                    </div>
                    <div class="metric-card composite">
                        <span class="metric-label">Composite</span>
                        <span id="compositeValue" class="metric-value">0.000</span>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="script.js"></script>
    </body>
    </html>'''        
    # 在css_content的开头添加全屏按钮样式：

    css_content = '''* {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        height: 100vh;
        overflow: hidden;
        color: white;
    }

    /* 全屏按钮样式 */
    .fullscreen-btn {
        position: fixed;
        top: 20px;
        left: 20px;
        width: 44px;
        height: 44px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        z-index: 1000;
        color: rgba(255, 255, 255, 0.8);
    }

    .fullscreen-btn:hover {
        background: rgba(255, 255, 255, 0.25);
        border-color: rgba(255, 255, 255, 0.5);
        color: rgba(255, 255, 255, 1);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    .fullscreen-btn:active {
        transform: translateY(0);
        background: rgba(255, 255, 255, 0.2);
    }

    .fullscreen-btn svg {
        transition: transform 0.2s ease;
    }

    .fullscreen-btn:hover svg {
        transform: scale(1.1);
    }

    /* 全屏状态下的样式调整 */
    :fullscreen .fullscreen-btn,
    :-webkit-full-screen .fullscreen-btn,
    :-moz-full-screen .fullscreen-btn {
        background: rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .container {
        display: flex;
        height: 100vh;
        width: 100vw;
    }

    /* 左侧状态指示区域 */
    .status-section {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: rgba(0, 0, 0, 0.2);
        position: relative;
    }

    .status-indicator {
        width: 200px;
        height: 200px;
        border-radius: 50%;
        margin-bottom: 30px;
        transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }

    .status-indicator.focused {
        background: radial-gradient(circle, #ff4757, #ff3742);
        box-shadow: 0 0 80px rgba(255, 71, 87, 0.8), 0 20px 60px rgba(0, 0, 0, 0.4);
        animation: focusedPulse 2s infinite;
    }

    .status-indicator.distracted {
        background: radial-gradient(circle, #2ed573, #17c0eb);
        box-shadow: 0 0 80px rgba(46, 213, 115, 0.8), 0 20px 60px rgba(0, 0, 0, 0.4);
        animation: distractedPulse 3s infinite;
    }

    @keyframes focusedPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 80px rgba(255, 71, 87, 0.8), 0 20px 60px rgba(0, 0, 0, 0.4);
        }
        50% { 
            transform: scale(1.1);
            box-shadow: 0 0 120px rgba(255, 71, 87, 1), 0 20px 60px rgba(0, 0, 0, 0.4);
        }
    }

    @keyframes distractedPulse {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 0 80px rgba(46, 213, 115, 0.8), 0 20px 60px rgba(0, 0, 0, 0.4);
        }
        50% { 
            transform: scale(1.1);
            box-shadow: 0 0 120px rgba(46, 213, 115, 1), 0 20px 60px rgba(0, 0, 0, 0.4);
        }
    }

    .status-text {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    .status-text.focused {
        color: #ff4757;
        text-shadow: 0 0 30px rgba(255, 71, 87, 0.8);
    }

    .status-text.distracted {
        color: #2ed573;
        text-shadow: 0 0 30px rgba(46, 213, 115, 0.8);
    }

    .connection-status {
        position: absolute;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 1.1rem;
        opacity: 0.8;
        transition: all 0.3s ease;
        padding: 8px 16px;
        border-radius: 20px;
        background: rgba(0, 0, 0, 0.3);
    }

    .connection-status.connected {
        color: #2ed573;
        border-left: 4px solid #2ed573;
    }

    .connection-status.disconnected {
        color: #ff4757;
        border-left: 4px solid #ff4757;
    }

    /* 右侧数据显示区域 */
    .metrics-section {

        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 40px;
        background: rgba(255, 255, 255, 0.05);
    }

    .metrics-grid {
        display: grid;
        grid-template-rows: repeat(3, 1fr);
        gap: 25px;
        height: 100%;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }

    .metric-card:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateX(5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }

    .metric-label {
        font-size: 1.8rem;
        font-weight: 600;
        opacity: 0.9;
        color: #ffffff;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'SF Mono', Monaco, 'Consolas', monospace;
        color: #00d4ff;
        text-shadow: 0 2px 4px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }

    .metric-value.updating {
        animation: valueUpdate 0.5s ease;
    }

    @keyframes valueUpdate {
        0% { 
            transform: scale(1);
            color: #00d4ff;
        }
        50% { 
            transform: scale(1.1);
            color: #ff6b6b;
        }
        100% { 
            transform: scale(1);
            color: #00d4ff;
        }
    }

    /* 特殊状态样式 */
    .metric-card.attention {
        border-left: 4px solid #ff6b6b;
    }

    .metric-card.engagement {
        border-left: 4px solid #4ecdc4;
    }

    .metric-card.composite {
        border-left: 4px solid #ffa726;
        background: rgba(255, 167, 38, 0.1);
    }

    .metric-card.composite .metric-value {
        color: #ffa726;
        font-size: 2.5rem;
    }

    /* 加载动画 */
    .loading {
        animation: loadingPulse 1.5s infinite;
    }

    @keyframes loadingPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* 横屏专用媒体查询 */
    @media (orientation: landscape) and (max-height: 600px) {
        .fullscreen-btn {
            width: 36px;
            height: 36px;
            top: 15px;
            left: 15px;
        }
        
        .fullscreen-btn svg {
            width: 16px;
            height: 16px;
        }
        
        .status-indicator {
            width: 160px;
            height: 160px;
        }
        
        .status-text {
            font-size: 2rem;
        }
        
        .metric-label {
            font-size: 1.5rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
        }
        
        .metrics-grid {
            gap: 20px;
        }
        
        .metric-card {
            padding: 20px 25px;
        }
    }

    /* 超小屏幕适配 */
    @media (max-height: 500px) {
        .fullscreen-btn {
            width: 32px;
            height: 32px;
            top: 10px;
            left: 10px;
        }
        
        .fullscreen-btn svg {
            width: 14px;
            height: 14px;
        }
        
        .container {
            padding: 10px;
        }
        
        .status-indicator {
            width: 120px;
            height: 120px;
        }
        
        .status-text {
            font-size: 1.5rem;
            margin-bottom: 15px;
        }
        
        .metrics-section {
            padding: 20px;
        }
        
        .metric-card {
            padding: 15px 20px;
        }
        
        .metric-label {
            font-size: 1.2rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }'''# 在js_content的FocusStatusApp类中添加全屏功能：
# 在create_web_files()函数中修改js_content的connectWebSocket方法：

    js_content = '''class FocusStatusApp {
        constructor() {
            this.ws = null;
            this.reconnectAttempts = 0;
            this.maxReconnectAttempts = 10;
            this.reconnectDelay = 1000;
            
            // 添加数据缓存用于平滑过渡
            this.lastUpdateTime = 0;
            this.updateThreshold = 100; // 最小更新间隔100ms
            
            this.elements = {
                statusText: document.getElementById('statusText'),
                statusIndicator: document.getElementById('statusIndicator'),
                connectionStatus: document.getElementById('connectionStatus'),
                attentionValue: document.getElementById('attentionValue'),
                engagementValue: document.getElementById('engagementValue'),
                compositeValue: document.getElementById('compositeValue'),
                fullscreenBtn: document.getElementById('fullscreenBtn')
            };
            
            this.init();
        }
        
        init() {
            this.connectWebSocket();
            this.updateConnectionStatus('Connecting...', 'connecting');
            this.setupFullscreenButton();
        }
        
        setupFullscreenButton() {
            this.elements.fullscreenBtn.addEventListener('click', () => {
                this.toggleFullscreen();
            });
            
            // 监听全屏状态变化
            document.addEventListener('fullscreenchange', () => {
                this.updateFullscreenButton();
            });
            
            document.addEventListener('webkitfullscreenchange', () => {
                this.updateFullscreenButton();
            });
            
            document.addEventListener('mozfullscreenchange', () => {
                this.updateFullscreenButton();
            });
        }
        
        toggleFullscreen() {
            if (!document.fullscreenElement && 
                !document.webkitFullscreenElement && 
                !document.mozFullScreenElement) {
                // 进入全屏
                if (document.documentElement.requestFullscreen) {
                    document.documentElement.requestFullscreen();
                } else if (document.documentElement.webkitRequestFullscreen) {
                    document.documentElement.webkitRequestFullscreen();
                } else if (document.documentElement.mozRequestFullScreen) {
                    document.documentElement.mozRequestFullScreen();
                }
            } else {
                // 退出全屏
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                } else if (document.mozCancelFullScreen) {
                    document.mozCancelFullScreen();
                }
            }
        }
        
        updateFullscreenButton() {
            const isFullscreen = document.fullscreenElement || 
                            document.webkitFullscreenElement || 
                            document.mozFullScreenElement;
            
            const svg = this.elements.fullscreenBtn.querySelector('svg');
            
            if (isFullscreen) {
                // 显示退出全屏图标
                svg.innerHTML = '<path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>';
                this.elements.fullscreenBtn.title = 'Exit Fullscreen';
            } else {
                // 显示进入全屏图标
                svg.innerHTML = '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>';
                this.elements.fullscreenBtn.title = 'Enter Fullscreen';
            }
        }
        
        connectWebSocket() {
            try {
                // 自动适配WebSocket地址
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const wsUrl = `${protocol}//${host}:8765`;
                
                console.log('Attempting to connect to:', wsUrl);
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket connected to:', wsUrl);
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus('Connected', 'connected');
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket disconnected from:', wsUrl);
                    this.updateConnectionStatus('Disconnected', 'disconnected');
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus('Connection Error', 'disconnected');
                };
                
            } catch (error) {
                console.error('Failed to create WebSocket connection:', error);
                this.updateConnectionStatus('Connection Failed', 'disconnected');
                this.attemptReconnect();
            }
        }
        
        attemptReconnect() {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                this.updateConnectionStatus(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`, 'connecting');
                
                setTimeout(() => {
                    this.connectWebSocket();
                }, this.reconnectDelay * this.reconnectAttempts);
            } else {
                this.updateConnectionStatus('Connection Failed', 'disconnected');
            }
        }
        
        handleMessage(data) {
            switch (data.type) {
                case 'connection':
                    console.log('Connection established:', data.message);
                    break;
                    
                case 'attention_update':
                    this.updateFocusStatus(data);
                    break;
                    
                default:
                    console.log('Unknown message type:', data);
            }
        }
        
        updateFocusStatus(data) {
            const currentTime = Date.now();
            
            // 限制更新频率，避免过于频繁的DOM操作
            if (currentTime - this.lastUpdateTime < this.updateThreshold) {
                return;
            }
            this.lastUpdateTime = currentTime;
            
            const isFocused = data.focused === 1;
            const statusText = isFocused ? 'Do Not Disturb' : 'Welcome to Disturb';
            
            // 更新状态文本（添加平滑过渡）
            if (this.elements.statusText.textContent !== statusText) {
                this.elements.statusText.style.opacity = '0.7';
                setTimeout(() => {
                    this.elements.statusText.textContent = statusText;
                    this.elements.statusText.className = `status-text ${isFocused ? 'focused' : 'distracted'}`;
                    this.elements.statusText.style.opacity = '1';
                }, 150);
            }
            
            // 更新状态指示器
            this.elements.statusIndicator.className = `status-indicator ${isFocused ? 'focused' : 'distracted'}`;
            
            // 更新指标值（使用动画数字过渡）
            if (data.metrics) {
                this.animateValue(this.elements.attentionValue, data.metrics.attention, 2);
                this.animateValue(this.elements.engagementValue, data.metrics.engagement, 2);
            }
            
            this.animateValue(this.elements.compositeValue, data.composite_score, 3);
            
            // 添加页面标题更新
            document.title = `${statusText} - Focus Status`;
            
            console.log(`Focus status: ${statusText}, Score: ${data.composite_score.toFixed(3)}`);
        }
        
        // 添加数字动画方法
        animateValue(element, targetValue, decimals) {
            const currentValue = parseFloat(element.textContent) || 0;
            const difference = targetValue - currentValue;
            
            if (Math.abs(difference) < 0.01) return; // 变化太小不执行动画
            
            // 添加更新动画类
            element.classList.add('updating');
            setTimeout(() => element.classList.remove('updating'), 500);
            
            const increment = difference / 15; // 15步完成动画
            let currentStep = 0;
            
            const timer = setInterval(() => {
                currentStep++;
                const newValue = currentValue + (increment * currentStep);
                element.textContent = newValue.toFixed(decimals);
                
                if (currentStep >= 15) {
                    clearInterval(timer);
                    element.textContent = targetValue.toFixed(decimals);
                }
            }, 16); // 约60fps
        }
        
        updateConnectionStatus(message, status) {
            this.elements.connectionStatus.textContent = message;
            this.elements.connectionStatus.className = `connection-status ${status}`;
            
            // 如果正在连接，添加加载动画
            if (status === 'connecting') {
                this.elements.connectionStatus.classList.add('loading');
            } else {
                this.elements.connectionStatus.classList.remove('loading');
            }
        }
    }

    // 页面加载完成后初始化应用
    document.addEventListener('DOMContentLoaded', () => {
        new FocusStatusApp();
    });

    // 页面可见性API - 当页面重新获得焦得焦点时尝试重连
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && window.focusApp) {
            if (!window.focusApp.ws || window.focusApp.ws.readyState === WebSocket.CLOSED) {
                window.focusApp.connectWebSocket();
            }
        }
    });'''
    
    # 写入文件
    with open(os.path.join(web_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    with open(os.path.join(web_dir, 'style.css'), 'w', encoding='utf-8') as f:
        f.write(css_content)
    
    with open(os.path.join(web_dir, 'script.js'), 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"✓ 网站文件已创建在 {web_dir} 目录")


# 修改start_http_server()函数：

def start_http_server():
    """启动HTTP服务器提供网站服务"""
    import http.server
    import socketserver
    import socket
    import os
    
    web_dir = os.path.abspath("web")
    port = 8000
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)
    
    def run_server():
        with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:  # 改为0.0.0.0
            print(f"🌐 HTTP服务器启动在 http://0.0.0.0:{port}")
            
            # 获取并显示本机IP地址
            try:
                # 获取本机IP
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                print(f"📱 手机访问地址: http://{local_ip}:{port}")
                print(f"💻 本机访问地址: http://localhost:{port}")
            except:
                print("📱 请使用电脑的实际IP地址访问")
                
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()


def main():
    """主函数"""
    print("EMOTIV CORTEX Performance Metrics 专用数据收集器 + 实时网站")
    print("=" * 60)
    print("输出格式: time, attention, relaxation, engagement, excitement, stress, interest, composite")
    print("实时网站: Apple风格专注状态显示")
    print("=" * 60)

    # 加载环境变量
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=env_path)
    
    your_app_client_id = os.getenv('CLIENT_ID')
    your_app_client_secret = os.getenv('CLIENT_SECRET')
    
    if not your_app_client_id or not your_app_client_secret:
        print("❌ 错误: 请先配置应用程序凭证")
        print("   - 创建 .env 文件并填入 CLIENT_ID 和 CLIENT_SECRET")
        sys.exit(1)
    
    # 创建网站文件
    create_web_files()
    
    # 启动HTTP服务器
    start_http_server()
    
    print("\n🔧 检查清单：")
    print("1. ✅ Emotiv应用程序状态为'Published'")
    print("2. ✅ Emotiv Launcher正在运行")  
    print("3. ✅ 头盔已连接且电量充足")
    print("4. ✅ 使用应用程序所有者账户登录")
    print("5. ✅ 网站将在 http://localhost:8000 打开")
    print("\n按Enter键开始收集数据并启动实时网站...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n👋 用户取消")
        sys.exit(0)
    
    try:
        print("\n🔄 初始化Performance Metrics收集器...")
        collector = PerformanceMetricsCollector(your_app_client_id, your_app_client_secret)
        
        # 配置
        collection_duration = 240  # 2分钟测试
        
        print(f"\n📋 配置:")
        print(f"  - 收集时长: {collection_duration}秒")
        print(f"  - 实时分类: 每2秒输出一次状态")
        print(f"  - WebSocket服务器: ws://localhost:8765")
        print(f"  - 网站地址: http://localhost:8000")
        print(f"  - 算法权重: attention(0.35), engagement(0.25), excitement(0.15), interest(0.15), stress(0.15), relaxation(-0.10)")
        
        # 自动打开网站
        try:
            import webbrowser
            webbrowser.open('http://localhost:8000')
        except:
            pass
        
        # 开始收集
        collector.start_collection(duration=collection_duration)
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断收集")
            collector.stop_collection()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断收集")
    except Exception as e:
        print(f"❌ 收集过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()