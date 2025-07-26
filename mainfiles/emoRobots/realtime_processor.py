# realtime_processor.py
"""
实时数据处理器 - 监控采集文件并应用算法分析

此脚本监控data_collector_test.py生成的CSV文件，
每2秒读取新数据，应用alg.py的算法逻辑，
并在终端输出分类结果。

使用方法:
1. 先运行 data_collector_test.py 开始数据采集
2. 再运行此脚本进行实时处理
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from collections import deque
import threading
from scipy.signal import find_peaks

class RealtimeProcessor:
    """
    实时处理器类
    监控CSV文件变化并应用算法分析
    """
    
    def __init__(self, monitor_directory="collected_data"):
        """
        初始化实时处理器
        
        Parameters:
        -----------
        monitor_directory : str
            监控的目录路径
        """
        print("=" * 60)
        print("初始化实时数据处理器")
        print("=" * 60)
        
        self.monitor_directory = monitor_directory
        self.met_file_path = None
        self.last_row_processed = 0
        self.processing_interval = 2.0  # 2秒处理间隔
        self.is_running = False
        
        # 算法权重配置（与alg.py保持一致）
        self.weights = {
            'attention'  :  0.35,
            'engagement' :  0.25,
            'excitement' :  0.15,
            'interest'   :  0.15,
            'stress'     :  0.15,
            'relaxation' : -0.10,
        }
        
        # 数据缓冲区存储最近的数据用于算法处理
        self.data_buffer = deque(maxlen=100)  # 存储最近100个数据点
        
        # 分析结果记录
        self.analysis_results = []
        
    def find_latest_met_file(self):
        """
        查找最新的performance metrics CSV文件
        
        Returns:
        --------
        str or None: 最新文件路径，如果没找到返回None
        """
        pattern = os.path.join(self.monitor_directory, "realtime_met_*.csv")
        met_files = glob.glob(pattern)
        
        if not met_files:
            return None
            
        # 找到最新的文件（按修改时间排序）
        latest_file = max(met_files, key=os.path.getmtime)
        return latest_file
        
    def wait_for_data_file(self, timeout=60):
        """
        等待数据文件出现
        
        Parameters:
        -----------
        timeout : int
            等待超时时间（秒）
            
        Returns:
        --------
        bool: 是否成功找到文件
        """
        print(f"📡 正在等待数据采集文件...")
        print(f"   监控目录: {os.path.abspath(self.monitor_directory)}")
        print(f"   查找模式: realtime_met_*.csv")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.met_file_path = self.find_latest_met_file()
            if self.met_file_path:
                print(f"✓ 找到数据文件: {os.path.basename(self.met_file_path)}")
                return True
            time.sleep(1)
            
        print(f"❌ 等待超时！未找到数据文件")
        print(f"💡 请确保先运行 data_collector_test.py 开始数据采集")
        return False
        
    def parse_met_labels(self, header_row):
        """
        解析performance metrics标签
        
        Parameters:
        -----------
        header_row : list
            CSV文件的头部行
            
        Returns:
        --------
        dict: 字段名到索引的映射
        """
        indices = {}
        for i, label in enumerate(header_row):
            if 'attention' in label.lower() and 'isactive' not in label.lower():
                indices['attention'] = i
            elif 'eng' in label.lower() and 'isactive' not in label.lower():
                indices['engagement'] = i
            elif 'exc' in label.lower() and 'isactive' not in label.lower():
                indices['excitement'] = i
            elif 'int' in label.lower() and 'isactive' not in label.lower():
                indices['interest'] = i
            elif 'str' in label.lower() and 'isactive' not in label.lower():
                indices['stress'] = i
            elif 'rel' in label.lower() and 'isactive' not in label.lower():
                indices['relaxation'] = i
                
        return indices
        
    def read_new_data(self):
        """
        读取文件中的新数据
        
        Returns:
        --------
        list: 新数据行列表
        """
        if not self.met_file_path or not os.path.exists(self.met_file_path):
            return []
            
        try:
            # 读取CSV文件
            df = pd.read_csv(self.met_file_path)
            
            # 如果是第一次读取，获取列索引映射
            if not hasattr(self, 'column_indices'):
                self.column_indices = self.parse_met_labels(df.columns.tolist())
                print(f"✓ 解析列映射: {self.column_indices}")
                
            # 获取新数据（从上次处理位置开始）
            new_rows = df.iloc[self.last_row_processed:].copy()
            self.last_row_processed = len(df)
            
            return new_rows
            
        except Exception as e:
            print(f"⚠️ 读取文件错误: {e}")
            return []
            
    def extract_metrics(self, row):
        """
        从数据行中提取performance metrics
        
        Parameters:
        -----------
        row : pandas.Series
            数据行
            
        Returns:
        --------
        dict: 提取的指标字典
        """
        metrics = {}
        
        for metric_name, col_index in self.column_indices.items():
            if col_index < len(row):
                value = row.iloc[col_index] if hasattr(row, 'iloc') else row[col_index]
                # 处理None值和无效值
                metrics[metric_name] = float(value) if pd.notna(value) else 0.0
            else:
                metrics[metric_name] = 0.0
                
        # 确保所有需要的指标都有值
        for weight_key in self.weights.keys():
            if weight_key not in metrics:
                metrics[weight_key] = 0.0
                
        return metrics
        
    def calculate_composite_score(self, metrics):
        """
        计算综合唤醒度分数（与alg.py算法一致）
        
        Parameters:
        -----------
        metrics : dict
            performance metrics字典
            
        Returns:
        --------
        float: 综合分数
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
        
    def apply_algorithm(self, data_window):
        """
        应用算法进行分析（基于alg.py逻辑）
        
        Parameters:
        -----------
        data_window : list
            数据窗口
            
        Returns:
        --------
        dict: 分析结果
        """
        if len(data_window) < 3:
            return None
            
        # 提取综合分数序列
        composite_scores = []
        latest_metrics = None
        
        for timestamp, metrics in data_window:
            score = self.calculate_composite_score(metrics)
            composite_scores.append(score)
            latest_metrics = metrics
            
        # 转换为numpy数组进行处理
        comp_array = np.array(composite_scores)
        
        # 归一化（与alg.py一致）
        if comp_array.max() > comp_array.min():
            normalized = (comp_array - comp_array.min()) / (comp_array.max() - comp_array.min())
        else:
            normalized = np.ones_like(comp_array) * 0.5
            
        # 平滑处理（3点移动平均）
        if len(normalized) >= 3:
            smooth_series = pd.Series(normalized).rolling(3, center=True).mean().fillna(method='bfill')
            smooth_scores = smooth_series.values
        else:
            smooth_scores = normalized
            
        # 当前分数
        current_score = smooth_scores[-1]
        
        # 简化的分类判断（基于阈值0.65）
        undisturbed = 1 if current_score > 0.65 else 0
        
        # 构建结果
        result = {
            'timestamp': time.time(),
            'composite_score': comp_array[-1],
            'normalized_score': current_score,
            'undisturbed_classification': undisturbed,
            'metrics': latest_metrics,
            'buffer_size': len(data_window)
        }
        
        return result
        
    def process_data_batch(self, new_data):
        """
        处理一批新数据
        
        Parameters:
        -----------
        new_data : pandas.DataFrame
            新数据批次
        """
        if new_data.empty:
            return
            
        # 将新数据添加到缓冲区
        for _, row in new_data.iterrows():
            timestamp = row.iloc[0] if 'timestamp' in row.index or len(row) > 0 else time.time()
            metrics = self.extract_metrics(row)
            self.data_buffer.append((timestamp, metrics))
            
        # 应用算法分析
        if len(self.data_buffer) >= 3:
            result = self.apply_algorithm(list(self.data_buffer))
            
            if result:
                # 保存结果
                self.analysis_results.append(result)
                
                # 输出到终端
                self.display_result(result)
                
    def display_result(self, result):
        """
        在终端显示分析结果
        
        Parameters:
        -----------
        result : dict
            分析结果
        """
        # 状态图标和文字
        status = "🎯 专注状态" if result['undisturbed_classification'] == 1 else "😴 分散状态"
        
        # 格式化输出
        current_time = datetime.now().strftime('%H:%M:%S')
        metrics = result['metrics']
        
        print(f"\n{status} | {current_time}")
        print(f"  综合分数: {result['normalized_score']:.3f}")
        print(f"  注意力: {metrics['attention']:.2f} | 参与度: {metrics['engagement']:.2f}")
        print(f"  兴奋度: {metrics['excitement']:.2f} | 兴趣度: {metrics['interest']:.2f}")
        print(f"  压力值: {metrics['stress']:.2f} | 放松度: {metrics['relaxation']:.2f}")
        print(f"  数据点: {result['buffer_size']}")
        print("-" * 50)
        
    def start_processing(self):
        """
        开始实时处理
        """
        print(f"\n🚀 启动实时处理器")
        print(f"  处理间隔: {self.processing_interval} 秒")
        print(f"  算法权重: {self.weights}")
        
        # 等待数据文件
        if not self.wait_for_data_file():
            return False
            
        self.is_running = True
        print(f"\n📊 开始实时数据处理: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        try:
            while self.is_running:
                # 读取新数据
                new_data = self.read_new_data()
                
                if not new_data.empty:
                    self.process_data_batch(new_data)
                    
                # 等待下个处理周期
                time.sleep(self.processing_interval)
                
        except KeyboardInterrupt:
            print(f"\n⚠️ 用户中断处理")
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_processing()
            
    def stop_processing(self):
        """
        停止处理
        """
        self.is_running = False
        print(f"\n🛑 停止实时处理")
        self.print_summary()
        
    def print_summary(self):
        """
        打印处理总结
        """
        if not self.analysis_results:
            print("📊 没有生成分析结果")
            return
            
        print("\n" + "=" * 60)
        print("实时处理总结")
        print("=" * 60)
        
        total_analyses = len(self.analysis_results)
        undisturbed_count = sum(1 for r in self.analysis_results if r['undisturbed_classification'] == 1)
        undisturbed_percentage = (undisturbed_count / total_analyses) * 100
        
        print(f"  总分析次数: {total_analyses}")
        print(f"  专注状态: {undisturbed_count} 次 ({undisturbed_percentage:.1f}%)")
        print(f"  分散状态: {total_analyses - undisturbed_count} 次 ({100-undisturbed_percentage:.1f}%)")
        
        if self.analysis_results:
            avg_attention = np.mean([r['metrics']['attention'] for r in self.analysis_results])
            avg_engagement = np.mean([r['metrics']['engagement'] for r in self.analysis_results])
            avg_composite = np.mean([r['normalized_score'] for r in self.analysis_results])
            
            print(f"  平均注意力: {avg_attention:.2f}")
            print(f"  平均参与度: {avg_engagement:.2f}")
            print(f"  平均综合分数: {avg_composite:.3f}")


def main():
    """
    主函数
    """
    print("EMOTIV 实时数据处理器")
    print("=" * 60)
    print("📋 使用说明:")
    print("1. 请先运行 data_collector_test.py 开始数据采集")
    print("2. 然后运行此脚本进行实时处理")
    print("3. 处理器将每2秒输出一次分析结果")
    print("4. 按 Ctrl+C 可以停止处理")
    
    print("\n⚠️  确认数据采集器已经运行？按 Enter 继续，Ctrl+C 退出...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n👋 用户取消")
        return
    
    try:
        # 创建处理器
        processor = RealtimeProcessor()
        
        # 开始处理
        processor.start_processing()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()