#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSORT跟踪器单元测试
"""

import unittest
import cv2
import numpy as np
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from deep_sort_tracker import DeepSORTTracker, KalmanBoxTracker

class TestKalmanBoxTracker(unittest.TestCase):
    """
    卡尔曼滤波器跟踪器的单元测试
    """
    
    def setUp(self):
        """
        每个测试方法执行前的设置
        """
        # 创建一个初始边界框 [x, y, width, height]
        self.initial_bbox = [100, 100, 50, 50]
        # 创建卡尔曼跟踪器
        self.tracker = KalmanBoxTracker(self.initial_bbox, 1)
    
    def test_initialization(self):
        """
        测试初始化
        """
        self.assertEqual(self.tracker.id, 1)
        self.assertEqual(self.tracker.time_since_update, 0)
        self.assertEqual(self.tracker.hits, 1)
    
    def test_predict(self):
        """
        测试预测功能
        """
        # 获取初始预测
        pred_bbox = self.tracker.predict()
        self.assertEqual(len(pred_bbox), 4)  # x, y, w, h
        
        # 连续预测几次，应该有轻微变化
        prev_bbox = pred_bbox
        for _ in range(5):
            new_bbox = self.tracker.predict()
            # 预测结果应该与前一结果有轻微差异
            self.assertNotEqual(new_bbox, prev_bbox)
            prev_bbox = new_bbox
    
    def test_update(self):
        """
        测试更新功能
        """
        # 创建新的检测边界框（稍微移动）
        new_bbox = [105, 105, 50, 50]
        
        # 更新跟踪器
        updated_bbox = self.tracker.update(new_bbox)
        
        # 验证更新后的值
        self.assertEqual(self.tracker.time_since_update, 0)
        self.assertEqual(self.tracker.hits, 2)
        
        # 更新后的边界框应该接近新的检测
        for i in range(4):
            self.assertAlmostEqual(updated_bbox[i], new_bbox[i], delta=10)
    
    def test_time_since_update(self):
        """
        测试时间戳更新
        """
        # 初始状态
        self.assertEqual(self.tracker.time_since_update, 0)
        
        # 预测一次，不更新
        self.tracker.predict()
        self.assertEqual(self.tracker.time_since_update, 1)
        
        # 再预测一次
        self.tracker.predict()
        self.assertEqual(self.tracker.time_since_update, 2)
        
        # 更新后重置计数器
        self.tracker.update(self.initial_bbox)
        self.assertEqual(self.tracker.time_since_update, 0)

class TestDeepSORTTracker(unittest.TestCase):
    """
    DeepSORT跟踪器的单元测试
    """
    
    def setUp(self):
        """
        每个测试方法执行前的设置
        """
        # 创建配置
        self.config = {
            'model_path': 'models/deepsort/ckpt.t7',
            'max_dist': 0.2,
            'min_confidence': 0.3,
            'nms_max_overlap': 0.5,
            'max_iou_distance': 0.7,
            'max_age': 70,
            'n_init': 3,
            'nn_budget': 100,
            'max_cosine_distance': 0.2
        }
        
        # 创建测试图像
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # 创建跟踪器（使用mock模式）
        self.tracker = DeepSORTTracker(self.config, use_mock=True)
    
    def test_initialization(self):
        """
        测试初始化
        """
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.trackers), 0)
        self.assertEqual(self.tracker.next_id, 1)
    
    def test_update_empty_detections(self):
        """
        测试空检测更新
        """
        # 使用空检测更新
        self.tracker.update([], self.test_image)
        
        # 应该没有跟踪器
        self.assertEqual(len(self.tracker.trackers), 0)
    
    def test_update_with_detections(self):
        """
        测试有检测结果的更新
        """
        # 创建模拟检测结果
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.9, 'class_id': 0},
            {'bbox': [200, 200, 60, 60], 'confidence': 0.8, 'class_id': 0}
        ]
        
        # 更新跟踪器
        self.tracker.update(detections, self.test_image)
        
        # 应该有新的跟踪器创建
        self.assertEqual(len(self.tracker.trackers), 2)
    
    def test_get_tracks(self):
        """
        测试获取跟踪结果
        """
        # 创建模拟检测
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.9, 'class_id': 0}
        ]
        
        # 更新两次，确保跟踪器被初始化（n_init=3，可能需要多次更新）
        for _ in range(3):
            self.tracker.update(detections, self.test_image)
        
        # 获取跟踪结果
        tracks = self.tracker.get_tracks()
        
        # 验证结果
        self.assertIsInstance(tracks, list)
        # 应该至少有一个有效的跟踪结果
        self.assertGreaterEqual(len(tracks), 1)
        
        # 验证每个跟踪结果的字段
        for track in tracks:
            self.assertIn('track_id', track)
            self.assertIn('bbox', track)
            self.assertIn('confidence', track)
            self.assertIn('class_id', track)
    
    def test_iou_distance(self):
        """
        测试IOU距离计算
        """
        # 创建两个边界框
        bbox1 = np.array([[100, 100, 200, 200]])  # [x1, y1, x2, y2]
        bbox2 = np.array([[120, 120, 220, 220]])  # 与第一个有重叠
        bbox3 = np.array([[300, 300, 400, 400]])  # 与第一个无重叠
        
        # 计算IOU距离
        # 注意：在DeepSORT中，距离是1-IOU
        # 高重叠应该有小距离，低重叠应该有大距离
        dist1_2 = self.tracker._iou_distance(bbox1, bbox2)[0, 0]
        dist1_3 = self.tracker._iou_distance(bbox1, bbox3)[0, 0]
        
        # 验证距离关系
        self.assertLess(dist1_2, dist1_3)
        self.assertGreaterEqual(dist1_2, 0)
        self.assertLessEqual(dist1_2, 1)
    
    def test_track_lifecycle(self):
        """
        测试跟踪器的生命周期
        """
        # 初始状态
        self.assertEqual(len(self.tracker.trackers), 0)
        
        # 创建初始检测
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.9, 'class_id': 0}
        ]
        
        # 多次更新，应该创建并稳定跟踪器
        for _ in range(5):
            self.tracker.update(detections, self.test_image)
        
        # 应该有跟踪器
        self.assertEqual(len(self.tracker.trackers), 1)
        
        # 停止检测（模拟目标消失）
        for _ in range(5):
            self.tracker.update([], self.test_image)
        
        # 跟踪器还应该存在（因为max_age设置较大）
        self.assertEqual(len(self.tracker.trackers), 1)

if __name__ == '__main__':
    unittest.main()