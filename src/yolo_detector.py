import torch
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple

class YOLOv7Detector:
    def __init__(self, config: Dict):
        """
        初始化YOLOv7检测器
        
        Args:
            config: 配置字典，包含模型路径、置信度阈值等参数
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.model_path = config['model_path']
        self.conf_thresh = config['conf_thresh']
        self.nms_thresh = config['nms_thresh']
        self.input_size = config['input_size']
        
        # 加载模型
        self.model = self._load_model()
        
        # COCO数据集类别名称
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def _load_model(self):
        """
        加载YOLOv7模型
        """
        # 检查模型文件是否存在，如果不存在则提示用户下载
        if not os.path.exists(self.model_path):
            print(f"模型文件不存在: {self.model_path}")
            print("请下载YOLOv7预训练模型并放置在指定路径")
            # 这里可以添加自动下载代码
        
        # 加载模型（这里使用简化版本，实际使用时可能需要根据YOLOv7的实现进行调整）
        try:
            # 假设我们使用torch.hub加载YOLOv7
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', path=self.model_path, trust_repo=True)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            # 为了演示，我们创建一个简单的模拟模型
            print("使用模拟模型进行演示...")
            return MockYOLOv7()
    
    def preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        预处理输入图像
        
        Args:
            frame: 输入图像
            
        Returns:
            预处理后的图像张量、原始图像尺寸、缩放信息
        """
        # 保存原始尺寸
        orig_h, orig_w = frame.shape[:2]
        
        # 调整图像大小并进行归一化
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img.unsqueeze(0)
        
        # 计算缩放比例
        scale = (self.input_size / orig_w, self.input_size / orig_h)
        
        return img, (orig_h, orig_w), scale
    
    def postprocess(self, detections: torch.Tensor, orig_size: Tuple[int, int], scale: Tuple[float, float]) -> List[Dict]:
        """
        后处理检测结果
        
        Args:
            detections: 模型输出的检测结果
            orig_size: 原始图像尺寸
            scale: 缩放比例
            
        Returns:
            处理后的检测结果列表
        """
        results = []
        
        # 解析检测结果（这里需要根据实际的YOLOv7输出格式进行调整）
        try:
            # 如果是使用torch.hub加载的YOLOv7，结果格式可能不同
            for det in detections.pred[0]:
                if det[4] > self.conf_thresh:
                    x1, y1, x2, y2 = det[:4].tolist()
                    conf = det[4].item()
                    cls = int(det[5].item())
                    
                    # 调整坐标到原始图像尺寸
                    orig_h, orig_w = orig_size
                    x1 = max(0, int(x1 / scale[0]))
                    y1 = max(0, int(y1 / scale[1]))
                    x2 = min(orig_w, int(x2 / scale[0]))
                    y2 = min(orig_h, int(y2 / scale[1]))
                    
                    results.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': self.class_names[cls] if cls < len(self.class_names) else 'unknown'
                    })
        except:
            # 如果是模拟模型，直接返回模拟结果
            if hasattr(detections, 'mock_results'):
                results = detections.mock_results
        
        return results
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        执行目标检测
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表
        """
        # 预处理
        img, orig_size, scale = self.preprocess(frame)
        
        # 执行推理
        with torch.no_grad():
            detections = self.model(img)
        
        # 后处理
        results = self.postprocess(detections, orig_size, scale)
        
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制了检测结果的图像
        """
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制类别名称和置信度
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

class MockYOLOv7:
    """
    模拟YOLOv7模型，用于演示
    """
    def __init__(self):
        self.mock_results = [
            {'bbox': [100, 100, 50, 100], 'confidence': 0.95, 'class_id': 0, 'class_name': 'person'},
            {'bbox': [200, 150, 80, 60], 'confidence': 0.92, 'class_id': 2, 'class_name': 'car'}
        ]
    
    def __call__(self, x):
        return self
    
    @property
    def pred(self):
        # 返回模拟的预测结果格式
        return [[torch.tensor([
            [100, 100, 150, 200, 0.95, 0],  # [x1, y1, x2, y2, conf, class_id]
            [200, 150, 280, 210, 0.92, 2]   # [x1, y1, x2, y2, conf, class_id]
        ])]]

# 测试代码
if __name__ == '__main__':
    config = {
        'model_path': '../models/yolov7.pt',
        'conf_thresh': 0.25,
        'nms_thresh': 0.45,
        'input_size': 640,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    
    detector = YOLOv7Detector(config)
    print("YOLOv7检测器初始化完成")
    
    # 创建一个测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 执行检测
    detections = detector.detect(test_image)
    print(f"检测到 {len(detections)} 个目标")
    
    # 绘制检测结果
    result_image = detector.draw_detections(test_image.copy(), detections)
    print("检测结果绘制完成")