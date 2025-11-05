#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv7检测器单元测试
"""

import unittest
import cv2
import numpy as np
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from yolo_detector import YOLOv7Detector

class TestYOLOv7Detector(unittest.TestCase):
    """
    YOLOv7检测器的单元测试类
    """
    
    def setUp(self):
        """
        每个测试方法执行前的设置
        """
        # 使用mock模式进行测试，避免实际模型加载
        self.config = {
            'model_path': 'models/yolov7-tiny.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'img_size': 640,
            'half': False
        }
        
        # 创建测试图像 (640x640 RGB)
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        # 在图像中绘制一个白色矩形作为目标
        cv2.rectangle(self.test_image, (200, 200), (400, 400), (255, 255, 255), -1)
        
        # 创建检测器实例（使用mock模式）
        self.detector = YOLOv7Detector(self.config, use_mock=True)
    
    def test_initialization(self):
        """
        测试检测器初始化
        """
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.iou_threshold, 0.45)
        self.assertTrue(hasattr(self.detector, 'model'))
    
    def test_detect(self):
        """
        测试检测功能
        """
        detections = self.detector.detect(self.test_image)
        
        # 验证返回结果格式
        self.assertIsInstance(detections, list)
        
        # Mock模式下应该返回一些模拟的检测结果
        self.assertTrue(len(detections) > 0)
        
        # 验证每个检测结果的字段
        for detection in detections:
            self.assertIn('bbox', detection)
            self.assertIn('confidence', detection)
            self.assertIn('class_id', detection)
            
            # 验证边界框格式
            bbox = detection['bbox']
            self.assertEqual(len(bbox), 4)  # x, y, w, h
            self.assertTrue(all(isinstance(coord, (int, float)) for coord in bbox))
            
            # 验证置信度范围
            confidence = detection['confidence']
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
    
    def test_filter_confidence(self):
        """
        测试基于置信度的过滤
        """
        # 设置较低的置信度阈值
        self.detector.confidence_threshold = 0.3
        low_threshold_detections = self.detector.detect(self.test_image)
        
        # 设置较高的置信度阈值
        self.detector.confidence_threshold = 0.8
        high_threshold_detections = self.detector.detect(self.test_image)
        
        # 较高的阈值应该返回较少或相等数量的检测结果
        self.assertLessEqual(len(high_threshold_detections), len(low_threshold_detections))
    
    def test_preprocess_image(self):
        """
        测试图像预处理
        """
        # 创建一个不同大小的图像
        test_img = np.zeros((320, 480, 3), dtype=np.uint8)
        
        # 预处理图像
        processed_img, ratio, pad = self.detector._preprocess_image(test_img)
        
        # 验证输出尺寸
        self.assertEqual(processed_img.shape[2], 3)  # 通道数
        self.assertEqual(processed_img.shape[1], 640)  # 高度
        self.assertEqual(processed_img.shape[0], 640)  # 宽度
        
        # 验证类型
        self.assertEqual(processed_img.dtype, np.float32)
    
    def test_postprocess_detections(self):
        """
        测试检测结果后处理
        """
        # 创建模拟的模型输出
        mock_output = np.array([
            [200, 200, 400, 400, 0.9, 0],  # [x1, y1, x2, y2, conf, class_id]
            [100, 100, 150, 150, 0.6, 0]
        ])
        
        # 后处理
        detections = self.detector._postprocess_detections(mock_output, self.test_image.shape)
        
        # 验证结果
        self.assertEqual(len(detections), 2)
        
        # 验证第一个检测的边界框（应该转换为x, y, w, h格式）
        self.assertEqual(detections[0]['bbox'], (200, 200, 200, 200))
        self.assertEqual(detections[0]['confidence'], 0.9)
        self.assertEqual(detections[0]['class_id'], 0)
    
    def test_different_image_sizes(self):
        """
        测试不同尺寸的图像
        """
        # 测试小图像
        small_img = np.zeros((224, 224, 3), dtype=np.uint8)
        small_detections = self.detector.detect(small_img)
        self.assertIsInstance(small_detections, list)
        
        # 测试大图像
        large_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        large_detections = self.detector.detect(large_img)
        self.assertIsInstance(large_detections, list)

if __name__ == '__main__':
    unittest.main()