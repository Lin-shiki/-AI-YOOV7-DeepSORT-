#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机跟踪算法演示脚本
演示如何使用YOLOv7和DeepSORT实现单目标和多目标跟踪
"""

# 解决OpenMP重复初始化问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import cv2
from src.tracker_integrator import DroneTracker

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='无人机跟踪算法演示')
    parser.add_argument('--video', type=str, default='input/test_video.mp4',
                      help='输入视频文件路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出视频文件路径')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                      help='跟踪模式: single(单目标) 或 multi(多目标)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='配置文件路径')
    parser.add_argument('--display', action='store_true',
                      help='是否显示结果')
    return parser.parse_args()

def demo_tracking(args):
    """
    演示跟踪功能
    """
    print("===== 无人机跟踪算法演示 =====")
    print(f"配置文件: {args.config}")
    print(f"输入视频: {args.video}")
    print(f"跟踪模式: {args.mode}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.video):
        print(f"错误: 视频文件 {args.video} 不存在！")
        print("请先创建测试视频或提供有效的视频文件路径。")
        print("可以运行以下命令创建测试视频:")
        print("  python -m src.tracker_integrator")
        return
    
    # 初始化跟踪器
    tracker = DroneTracker(args.config)
    
    # 覆盖显示设置
    if args.display:
        tracker.display = True
    
    # 根据模式执行跟踪
    if args.mode == 'single':
        tracker.track_single_object(args.video, args.output)
    else:
        tracker.track_multi_objects(args.video, args.output)

def create_sample_video():
    """
    创建示例视频（简单版本）
    """
    import numpy as np
    
    output_path = 'input/sample_video.mp4'
    duration = 10  # 10秒
    fps = 30
    width, height = 1280, 720
    
    # 确保输入目录存在
    os.makedirs('input', exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"正在创建示例视频: {output_path}")
    
    # 创建多个移动对象
    num_objects = 5
    object_data = []
    
    # 初始化每个对象的参数
    for i in range(num_objects):
        # 随机初始位置和速度
        x = np.random.randint(100, width - 100)
        y = np.random.randint(100, height - 100)
        vx = np.random.uniform(-3, 3)
        vy = np.random.uniform(-3, 3)
        size = np.random.randint(20, 50)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        object_data.append({
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'size': size,
            'color': color
        })
    
    # 生成视频帧
    for frame_idx in range(duration * fps):
        # 创建背景（浅蓝色天空）
        frame = np.ones((height, width, 3), dtype=np.uint8) * np.array([220, 230, 255], dtype=np.uint8)
        
        # 添加地面
        cv2.rectangle(frame, (0, int(height * 0.7)), (width, height), (0, 150, 0), -1)
        
        # 更新和绘制每个对象
        for obj in object_data:
            # 更新位置
            obj['x'] += obj['vx']
            obj['y'] += obj['vy']
            
            # 边界反弹
            if obj['x'] < obj['size'] or obj['x'] > width - obj['size']:
                obj['vx'] *= -1
            if obj['y'] < obj['size'] or obj['y'] > height * 0.7 - obj['size']:
                obj['vy'] *= -1
            
            # 绘制对象（作为无人机的简化表示）
            # 主体
            cv2.rectangle(frame,
                         (int(obj['x'] - obj['size'] // 2), int(obj['y'] - obj['size'] // 2)),
                         (int(obj['x'] + obj['size'] // 2), int(obj['y'] + obj['size'] // 2)),
                         obj['color'], -1)
            
            # 添加一些无人机特征
            # 螺旋桨
            cv2.circle(frame, (int(obj['x'] - obj['size']), int(obj['y'] - obj['size'])),
                      obj['size'] // 3, (200, 200, 200), 2)
            cv2.circle(frame, (int(obj['x'] + obj['size']), int(obj['y'] - obj['size'])),
                      obj['size'] // 3, (200, 200, 200), 2)
            cv2.circle(frame, (int(obj['x'] - obj['size']), int(obj['y'] + obj['size'])),
                      obj['size'] // 3, (200, 200, 200), 2)
            cv2.circle(frame, (int(obj['x'] + obj['size']), int(obj['y'] + obj['size'])),
                      obj['size'] // 3, (200, 200, 200), 2)
        
        # 添加一些随机背景元素（云）
        if frame_idx % 50 == 0:
            cloud_x = np.random.randint(0, width)
            cloud_y = np.random.randint(0, int(height * 0.3))
            cloud_size = np.random.randint(50, 200)
            cv2.ellipse(frame, (cloud_x, cloud_y), (cloud_size, cloud_size // 2),
                       0, 0, 360, (255, 255, 255), -1)
        
        # 添加时间戳
        timestamp = f'Time: {frame_idx/fps:.1f}s'
        cv2.putText(frame, timestamp, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # 添加标题
        cv2.putText(frame, 'Drone Tracking Demo', (width // 2 - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        
        # 写入帧
        out.write(frame)
    
    out.release()
    print(f"示例视频创建完成！")
    print(f"视频路径: {output_path}")
    print(f"视频长度: {duration}秒，{fps} FPS")
    print(f"分辨率: {width}x{height}")
    print(f"包含 {num_objects} 个模拟无人机")
    
    return output_path

def main():
    """
    主函数
    """
    # 创建示例视频（如果不存在）
    if not os.path.exists('input/sample_video.mp4'):
        print("检测到示例视频不存在，将自动创建...")
        sample_video_path = create_sample_video()
    else:
        sample_video_path = 'input/sample_video.mp4'
    
    # 解析参数
    args = parse_arguments()
    
    # 如果用户没有指定视频，使用示例视频
    if args.video == 'input/test_video.mp4' and os.path.exists(sample_video_path):
        args.video = sample_video_path
    
    # 执行跟踪演示
    demo_tracking(args)

if __name__ == '__main__':
    main()