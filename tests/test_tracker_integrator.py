#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跟踪器集成模块的单元测试
"""

import unittest
import cv2
import numpy as np
import os
import sys
import tempfile

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tracker_integrator import DroneTracker

class TestDroneTracker(unittest.TestCase):
    """
    DroneTracker集成测试类
    """
    
    def setUp(self):
        """
        每个测试方法执行前的设置
        """
        # 创建测试配置
        self.config = {
            'yolo': {
                'model_path': 'models/yolov7-tiny.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu',
                'img_size': 640,
                'half': False
            },
            'deepsort': {
                'model_path': 'models/deepsort/ckpt.t7',
                'max_dist': 0.2,
                'min_confidence': 0.3,
                'nms_max_overlap': 0.5,
                'max_iou_distance': 0.7,
                'max_age': 70,
                'n_init': 3,
                'nn_budget': 100,
                'max_cosine_distance': 0.2
            },
            'tracking': {
                'display': False,
                'save_output': False,
                'output_path': 'output',
                'classes_to_track': [0],
                'track_color_map': {0: [0, 0, 255]}
            },
            'video': {
                'codec': 'mp4v'
            }
        }
        
        # 创建临时配置文件
        self.temp_config_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        self.temp_config_file.close()
        
        # 写入配置内容
        with open(self.temp_config_file.name, 'w') as f:
            import yaml
            yaml.dump(self.config, f)
        
        # 创建测试图像
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 在图像中绘制几个目标
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 0, 0), -1)
        cv2.rectangle(self.test_image, (300, 200), (400, 300), (0, 255, 0), -1)
        
        # 初始化跟踪器（模拟模式）
        # 我们需要修改DroneTracker的初始化方式以支持mock模式
        # 这里我们先创建实例，然后手动替换detector和tracker为模拟对象
        self.tracker = DroneTracker(self.temp_config_file.name)
        
        # 由于DroneTracker类内部没有直接支持mock模式，我们需要使用monkey patching
        # 这里我们创建简单的模拟函数来替代真实的检测和跟踪功能
        self._setup_mock_functions()
    
    def tearDown(self):
        """
        每个测试方法执行后的清理
        """
        # 删除临时文件
        try:
            os.unlink(self.temp_config_file.name)
        except:
            pass
    
    def _setup_mock_functions(self):
        """
        设置模拟函数来替代真实的检测和跟踪功能
        """
        # 模拟检测函数
        def mock_detect(image):
            return [
                {'bbox': [100, 100, 100, 100], 'confidence': 0.9, 'class_id': 0},
                {'bbox': [300, 200, 100, 100], 'confidence': 0.85, 'class_id': 0}
            ]
        
        # 模拟更新函数
        def mock_update(detections, image):
            # 这里不做实际操作，只是记录检测结果
            self._last_detections = detections
        
        # 模拟获取跟踪结果函数
        def mock_get_tracks():
            return [
                {'track_id': 1, 'bbox': [102, 102, 100, 100], 'confidence': 0.9, 'class_id': 0},
                {'track_id': 2, 'bbox': [302, 202, 100, 100], 'confidence': 0.85, 'class_id': 0}
            ]
        
        # 应用mock
        self.tracker.detector.detect = mock_detect
        self.tracker.tracker.update = mock_update
        self.tracker.tracker.get_tracks = mock_get_tracks
        self._last_detections = None
    
    def test_initialization(self):
        """
        测试初始化
        """
        self.assertIsNotNone(self.tracker)
        self.assertIsNotNone(self.tracker.detector)
        self.assertIsNotNone(self.tracker.tracker)
        self.assertEqual(self.tracker.target_track_id, None)
    
    def test_reset_tracking(self):
        """
        测试重置跟踪状态
        """
        # 先设置一些状态
        self.tracker.target_track_id = 1
        self.tracker.track_histories = {1: [(100, 100)]}
        self.tracker.active_track_ids = {1, 2}
        
        # 重置
        self.tracker.reset_tracking()
        
        # 验证状态已重置
        self.assertIsNone(self.tracker.target_track_id)
        self.assertEqual(self.tracker.track_histories, {})
        self.assertEqual(self.tracker.active_track_ids, set())
        self.assertEqual(self.tracker.total_tracked_objects, 0)
    
    def test_process_frame_single_mode(self):
        """
        测试单帧处理（单目标模式）
        """
        # 处理帧
        result = self.tracker.process_frame(self.test_image, mode='single')
        
        # 验证返回结果格式
        self.assertIn('detections', result)
        self.assertIn('tracks', result)
        self.assertIn('target', result)
        self.assertIn('frame', result)
        
        # 验证检测和跟踪结果
        self.assertEqual(len(result['detections']), 2)
        self.assertEqual(len(result['tracks']), 2)
        
        # 在单目标模式下应该选择一个目标
        self.assertIsNotNone(result['target'])
    
    def test_process_frame_multi_mode(self):
        """
        测试单帧处理（多目标模式）
        """
        # 处理帧
        result = self.tracker.process_frame(self.test_image, mode='multi')
        
        # 验证返回结果格式
        self.assertIn('detections', result)
        self.assertIn('tracks', result)
        self.assertIn('target', result)
        self.assertIn('frame', result)
        self.assertIn('total_objects', result)
        
        # 验证检测和跟踪结果
        self.assertEqual(len(result['detections']), 2)
        self.assertEqual(len(result['tracks']), 2)
        
        # 在多目标模式下目标应该为None
        self.assertIsNone(result['target'])
        
        # 验证轨迹历史被更新
        self.assertEqual(len(self.tracker.track_histories), 2)
    
    def test_filter_detections(self):
        """
        测试检测结果过滤
        """
        # 创建检测结果
        detections = [
            {'bbox': [100, 100, 100, 100], 'confidence': 0.9, 'class_id': 0},
            {'bbox': [200, 200, 50, 50], 'confidence': 0.4, 'class_id': 0},  # 低置信度
            {'bbox': [300, 300, 80, 80], 'confidence': 0.8, 'class_id': 1}   # 不在跟踪类别中
        ]
        
        # 过滤检测结果
        self.tracker.classes_to_track = [0]
        self.tracker.target_confidence_threshold = 0.5
        filtered = self.tracker.filter_detections(detections)
        
        # 应该只保留高置信度且在跟踪类别中的检测
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['bbox'], [100, 100, 100, 100])
    
    def test_select_target(self):
        """
        测试目标选择
        """
        # 创建检测结果
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.8, 'class_id': 0},
            {'bbox': [200, 200, 100, 100], 'confidence': 0.9, 'class_id': 0}  # 更大且置信度更高
        ]
        
        # 选择目标
        target = self.tracker.select_target(detections)
        
        # 应该选择置信度最高的目标
        self.assertEqual(target['bbox'], [200, 200, 100, 100])
    
    def test_calculate_iou(self):
        """
        测试IOU计算
        """
        # 创建两个边界框
        bbox1 = [100, 100, 100, 100]  # x, y, w, h
        bbox2 = [120, 120, 100, 100]  # 部分重叠
        bbox3 = [300, 300, 100, 100]  # 不重叠
        
        # 计算IOU
        iou1_2 = self.tracker._calculate_iou(bbox1, bbox2)
        iou1_3 = self.tracker._calculate_iou(bbox1, bbox3)
        
        # 验证IOU值范围和关系
        self.assertGreater(iou1_2, 0.0)
        self.assertLessEqual(iou1_2, 1.0)
        self.assertEqual(iou1_3, 0.0)  # 无重叠时IOU为0
    
    def test_generate_track_color(self):
        """
        测试跟踪颜色生成
        """
        # 为相同ID生成颜色
        color1 = self.tracker._generate_track_color(1)
        color1_again = self.tracker._generate_track_color(1)
        
        # 为不同ID生成颜色
        color2 = self.tracker._generate_track_color(2)
        
        # 验证相同ID生成相同颜色
        self.assertEqual(color1, color1_again)
        
        # 验证不同ID生成不同颜色
        self.assertNotEqual(color1, color2)
        
        # 验证颜色格式
        self.assertEqual(len(color1), 3)  # BGR颜色
        self.assertTrue(all(0 <= c <= 255 for c in color1))

class TestVideoProcessing(unittest.TestCase):
    """
    视频处理相关功能的测试
    """
    
    def setUp(self):
        """
        创建测试视频
        """
        # 创建临时视频文件
        self.temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        self.temp_video.close()
        
        # 创建一个简单的测试视频
        width, height = 640, 480
        fps = 10
        frames = 20
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video.name, fourcc, fps, (width, height))
        
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # 创建一个移动的目标
            x = 100 + i * 5
            y = 100
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
    
    def tearDown(self):
        """
        清理临时文件
        """
        try:
            os.unlink(self.temp_video.name)
        except:
            pass
    
    def test_video_creation(self):
        """
        测试创建的视频文件是否有效
        """
        # 检查文件是否存在
        self.assertTrue(os.path.exists(self.temp_video.name))
        
        # 检查文件大小
        self.assertGreater(os.path.getsize(self.temp_video.name), 0)

if __name__ == '__main__':
    unittest.main()