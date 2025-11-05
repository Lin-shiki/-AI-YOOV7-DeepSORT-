#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机跟踪算法性能测试

此脚本用于测量和评估跟踪算法的性能指标，包括：
1. 处理帧率 (FPS)
2. 内存使用情况
3. CPU利用率
4. 不同分辨率和目标数量下的性能表现
"""

import os
import sys
import time
import psutil
import numpy as np
import cv2
import argparse
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tracker_integrator import DroneTracker

class PerformanceTester:
    """
    性能测试类
    """
    
    def __init__(self, config_path, output_dir='../output'):
        """
        初始化性能测试器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化跟踪器
        self.tracker = DroneTracker(config_path)
        
        # 性能指标
        self.performance_results = {}
    
    def create_test_image(self, width, height, num_objects=1):
        """
        创建测试图像
        
        Args:
            width: 图像宽度
            height: 图像高度
            num_objects: 目标数量
            
        Returns:
            numpy数组: 测试图像
        """
        # 创建空白图像
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加目标
        object_size = min(width, height) // 10
        for i in range(num_objects):
            # 在图像中均匀分布目标
            cols = int(np.ceil(np.sqrt(num_objects)))
            rows = int(np.ceil(num_objects / cols))
            col = i % cols
            row = i // cols
            
            x = int((col + 0.5) * width / cols - object_size/2)
            y = int((row + 0.5) * height / rows - object_size/2)
            
            # 确保目标在图像范围内
            x = max(0, min(x, width - object_size))
            y = max(0, min(y, height - object_size))
            
            # 绘制目标（不同颜色）
            color = ((i * 50) % 255, (i * 100) % 255, (i * 150) % 255)
            cv2.rectangle(image, (x, y), (x + object_size, y + object_size), color, -1)
        
        return image
    
    def measure_fps(self, width, height, num_objects, duration=10):
        """
        测量帧率
        
        Args:
            width: 图像宽度
            height: 图像高度
            num_objects: 目标数量
            duration: 测量持续时间（秒）
            
        Returns:
            dict: 帧率和处理时间统计
        """
        # 重置跟踪器
        self.tracker.reset_tracking()
        
        # 创建测试图像
        image = self.create_test_image(width, height, num_objects)
        
        # 预热
        for _ in range(10):
            self.tracker.process_frame(image.copy(), mode='multi')
        
        # 测量处理时间
        start_time = time.time()
        frame_count = 0
        process_times = []
        
        while time.time() - start_time < duration:
            frame_start = time.time()
            self.tracker.process_frame(image.copy(), mode='multi')
            frame_end = time.time()
            
            process_time = (frame_end - frame_start) * 1000  # 毫秒
            process_times.append(process_time)
            frame_count += 1
        
        total_time = time.time() - start_time
        
        # 计算统计数据
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_process_time = np.mean(process_times) if process_times else 0
        max_process_time = np.max(process_times) if process_times else 0
        min_process_time = np.min(process_times) if process_times else 0
        
        result = {
            'width': width,
            'height': height,
            'num_objects': num_objects,
            'duration': total_time,
            'total_frames': frame_count,
            'avg_fps': avg_fps,
            'avg_process_time_ms': avg_process_time,
            'max_process_time_ms': max_process_time,
            'min_process_time_ms': min_process_time
        }
        
        print(f"分辨率 {width}x{height}, {num_objects} 个目标: "
              f"平均FPS: {avg_fps:.2f}, "
              f"平均处理时间: {avg_process_time:.2f}ms")
        
        return result
    
    def measure_memory_usage(self, width, height, num_objects, frames=100):
        """
        测量内存使用情况
        
        Args:
            width: 图像宽度
            height: 图像高度
            num_objects: 目标数量
            frames: 处理的帧数
            
        Returns:
            dict: 内存使用统计
        """
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 重置跟踪器
        self.tracker.reset_tracking()
        
        # 创建测试图像
        image = self.create_test_image(width, height, num_objects)
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理多帧
        memory_usages = []
        
        for i in range(frames):
            # 每处理10帧记录一次内存使用
            if i % 10 == 0:
                memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usages.append(memory)
            
            self.tracker.process_frame(image.copy(), mode='multi')
        
        # 记录最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = {
            'width': width,
            'height': height,
            'num_objects': num_objects,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'peak_memory_mb': max(memory_usages) if memory_usages else 0,
            'memory_growth_mb': final_memory - initial_memory
        }
        
        print(f"分辨率 {width}x{height}, {num_objects} 个目标: "
              f"初始内存: {initial_memory:.2f}MB, "
              f"最终内存: {final_memory:.2f}MB, "
              f"内存增长: {result['memory_growth_mb']:.2f}MB")
        
        return result
    
    def run_resolution_test(self, resolutions, num_objects=1):
        """
        测试不同分辨率下的性能
        
        Args:
            resolutions: 分辨率列表 [(width1, height1), (width2, height2), ...]
            num_objects: 目标数量
        """
        print(f"\n开始分辨率测试 (每个分辨率测试10秒，目标数量: {num_objects})...")
        
        results = []
        for width, height in resolutions:
            result = self.measure_fps(width, height, num_objects)
            results.append(result)
        
        self.performance_results['resolution_test'] = results
        return results
    
    def run_object_count_test(self, base_resolution=(640, 480), object_counts=[1, 2, 5, 10]):
        """
        测试不同目标数量下的性能
        
        Args:
            base_resolution: 基础分辨率 (width, height)
            object_counts: 目标数量列表
        """
        width, height = base_resolution
        print(f"\n开始目标数量测试 (分辨率: {width}x{height})...")
        
        results = []
        for count in object_counts:
            result = self.measure_fps(width, height, count)
            results.append(result)
        
        self.performance_results['object_count_test'] = results
        return results
    
    def run_memory_test(self, resolutions, num_objects_list):
        """
        测试内存使用情况
        
        Args:
            resolutions: 分辨率列表
            num_objects_list: 目标数量列表
        """
        print(f"\n开始内存使用测试...")
        
        results = []
        for width, height in resolutions:
            for count in num_objects_list:
                result = self.measure_memory_usage(width, height, count)
                results.append(result)
        
        self.performance_results['memory_test'] = results
        return results
    
    def run_stability_test(self, width, height, num_objects, frames=1000):
        """
        稳定性测试 - 长时间运行测试
        
        Args:
            width: 图像宽度
            height: 图像高度
            num_objects: 目标数量
            frames: 测试帧数
        """
        print(f"\n开始稳定性测试 (分辨率: {width}x{height}, 目标: {num_objects}, 帧数: {frames})...")
        
        # 重置跟踪器
        self.tracker.reset_tracking()
        
        # 创建测试图像
        image = self.create_test_image(width, height, num_objects)
        
        # 开始测试
        start_time = time.time()
        process_times = []
        errors = 0
        
        for i in range(frames):
            try:
                frame_start = time.time()
                self.tracker.process_frame(image.copy(), mode='multi')
                frame_end = time.time()
                
                process_time = (frame_end - frame_start) * 1000  # 毫秒
                process_times.append(process_time)
                
                # 每100帧输出一次进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  已处理 {i+1}/{frames} 帧, 耗时: {elapsed:.2f}秒")
                    
            except Exception as e:
                print(f"  处理第 {i} 帧时出错: {str(e)}")
                errors += 1
        
        total_time = time.time() - start_time
        avg_fps = frames / total_time if total_time > 0 else 0
        
        result = {
            'width': width,
            'height': height,
            'num_objects': num_objects,
            'frames': frames,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'errors': errors,
            'success_rate': (frames - errors) / frames * 100 if frames > 0 else 0
        }
        
        print(f"稳定性测试完成: 总耗时 {total_time:.2f}秒, "
              f"平均FPS: {avg_fps:.2f}, "
              f"错误数: {errors}, "
              f"成功率: {result['success_rate']:.2f}%")
        
        self.performance_results['stability_test'] = result
        return result
    
    def save_results(self):
        """
        保存性能测试结果到文件
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(self.output_dir, f"performance_results_{timestamp}.yaml")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.performance_results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\n性能测试结果已保存到: {result_file}")
        return result_file
    
    def generate_report(self):
        """
        生成性能测试报告（图表）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 生成分辨率与FPS关系图
        if 'resolution_test' in self.performance_results:
            plt.figure(figsize=(10, 6))
            results = self.performance_results['resolution_test']
            labels = [f"{r['width']}x{r['height']}" for r in results]
            fps_values = [r['avg_fps'] for r in results]
            
            plt.bar(labels, fps_values, color='skyblue')
            plt.xlabel('分辨率')
            plt.ylabel('平均FPS')
            plt.title('不同分辨率下的处理性能')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_file = os.path.join(self.output_dir, f"resolution_performance_{timestamp}.png")
            plt.savefig(chart_file)
            print(f"分辨率性能图表已保存到: {chart_file}")
        
        # 2. 生成目标数量与FPS关系图
        if 'object_count_test' in self.performance_results:
            plt.figure(figsize=(10, 6))
            results = self.performance_results['object_count_test']
            object_counts = [r['num_objects'] for r in results]
            fps_values = [r['avg_fps'] for r in results]
            
            plt.plot(object_counts, fps_values, marker='o', linestyle='-', color='green')
            plt.xlabel('目标数量')
            plt.ylabel('平均FPS')
            plt.title('不同目标数量下的处理性能')
            plt.grid(True)
            plt.tight_layout()
            
            chart_file = os.path.join(self.output_dir, f"object_count_performance_{timestamp}.png")
            plt.savefig(chart_file)
            print(f"目标数量性能图表已保存到: {chart_file}")
        
        # 3. 生成内存使用情况图
        if 'memory_test' in self.performance_results:
            plt.figure(figsize=(12, 6))
            results = self.performance_results['memory_test']
            
            # 按分辨率分组
            resolutions = set((r['width'], r['height']) for r in results)
            
            for width, height in resolutions:
                res_data = [r for r in results if r['width'] == width and r['height'] == height]
                object_counts = [d['num_objects'] for d in res_data]
                memory_growth = [d['memory_growth_mb'] for d in res_data]
                
                plt.plot(object_counts, memory_growth, marker='s', linestyle='-', 
                         label=f"{width}x{height}")
            
            plt.xlabel('目标数量')
            plt.ylabel('内存增长 (MB)')
            plt.title('不同分辨率和目标数量下的内存增长')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            chart_file = os.path.join(self.output_dir, f"memory_usage_{timestamp}.png")
            plt.savefig(chart_file)
            print(f"内存使用图表已保存到: {chart_file}")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='无人机跟踪算法性能测试')
    parser.add_argument('--config', type=str, 
                        default='tests/test_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, 
                        default='../output',
                        help='输出目录')
    parser.add_argument('--resolution', action='store_true', 
                        help='运行分辨率测试')
    parser.add_argument('--objects', action='store_true', 
                        help='运行目标数量测试')
    parser.add_argument('--memory', action='store_true', 
                        help='运行内存使用测试')
    parser.add_argument('--stability', action='store_true', 
                        help='运行稳定性测试')
    parser.add_argument('--all', action='store_true', 
                        help='运行所有测试')
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_arguments()
    
    # 创建性能测试器
    tester = PerformanceTester(args.config, args.output)
    
    # 定义测试配置
    resolutions = [(320, 240), (640, 480), (1280, 720)]
    object_counts = [1, 2, 5, 10]
    
    # 运行测试
    run_all = args.all or (not any([args.resolution, args.objects, args.memory, args.stability]))
    
    if args.resolution or run_all:
        tester.run_resolution_test(resolutions, num_objects=1)
    
    if args.objects or run_all:
        tester.run_object_count_test(base_resolution=(640, 480), object_counts=object_counts)
    
    if args.memory or run_all:
        tester.run_memory_test(resolutions[:2], [1, 5])  # 只测试部分配置以节省时间
    
    if args.stability or run_all:
        tester.run_stability_test(640, 480, 1, frames=500)  # 运行500帧的稳定性测试
    
    # 保存结果
    tester.save_results()
    
    # 生成报告
    tester.generate_report()
    
    print("\n性能测试完成！")

if __name__ == '__main__':
    main()