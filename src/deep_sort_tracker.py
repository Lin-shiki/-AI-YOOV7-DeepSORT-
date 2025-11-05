import numpy as np
import torch
import os
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

class KalmanBoxTracker:
    """
    使用卡尔曼滤波器跟踪单个目标
    """
    count = 0
    
    def __init__(self, bbox: List[float]):
        """
        初始化卡尔曼滤波器
        
        Args:
            bbox: 边界框 [x, y, width, height]
        """
        # 初始化卡尔曼滤波器
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 状态转移矩阵
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # 过程噪声协方差
        self.kf.Q[4:, 4:] *= 1e-1
        
        # 观测噪声协方差
        self.kf.R[2:, 2:] *= 1e-1
        
        # 初始状态
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # 初始协方差矩阵
        self.kf.P[4:, 4:] *= 10.0
        self.kf.P *= 10.0
        
        # 跟踪信息
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def _convert_bbox_to_z(self, bbox: List[float]) -> np.ndarray:
        """
        将边界框转换为状态向量
        
        Args:
            bbox: [x, y, width, height]
            
        Returns:
            状态向量 [x_center, y_center, width, height]
        """
        w = bbox[2]
        h = bbox[3]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        return np.array([x, y, w, h]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x: np.ndarray) -> List[float]:
        """
        将状态向量转换为边界框
        
        Args:
            x: 状态向量
            
        Returns:
            边界框 [x_min, y_min, width, height]
        """
        x_center = x[0]
        y_center = x[1]
        w = x[2]
        h = x[3]
        x_min = x_center - w / 2.0
        y_min = y_center - h / 2.0
        return [x_min, y_min, w, h]
    
    def update(self, bbox: List[float]):
        """
        使用观测值更新状态
        
        Args:
            bbox: 观测到的边界框
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        预测下一时刻的状态
        """
        # 更新速度
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        if (self.kf.x[7] + self.kf.x[3]) <= 0:
            self.kf.x[7] *= 0.0
        
        # 预测
        self.kf.predict()
        self.age += 1
        
        # 更新未更新时间
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # 记录历史
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> List[float]:
        """
        获取当前状态的边界框
        """
        return self._convert_x_to_bbox(self.kf.x)

class ReIDExtractor:
    """
    目标重识别特征提取器
    """
    def __init__(self, model_path: str, device='cuda:0'):
        """
        初始化特征提取器
        
        Args:
            model_path: 模型路径
            device: 运行设备
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        """
        加载特征提取模型
        """
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            print(f"特征提取模型文件不存在: {self.model_path}")
            print("使用简化版特征提取器进行演示")
            return SimpleExtractor()
        
        try:
            # 加载预训练模型
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"加载特征提取模型失败: {e}")
            print("使用简化版特征提取器进行演示")
            return SimpleExtractor()
    
    def extract(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        提取目标特征
        
        Args:
            image: 输入图像
            bbox: 目标边界框 [x, y, width, height]
            
        Returns:
            特征向量
        """
        # 裁剪目标区域
        x, y, w, h = map(int, bbox)
        x = max(0, x)
        y = max(0, y)
        w = max(1, w)
        h = max(1, h)
        
        # 确保边界框不超出图像
        h_img, w_img = image.shape[:2]
        x2 = min(x + w, w_img)
        y2 = min(y + h, h_img)
        x = x2 - w if x2 - w > 0 else 0
        y = y2 - h if y2 - h > 0 else 0
        
        # 裁剪并预处理
        crop = image[y:y2, x:x2]
        crop = cv2.resize(crop, (128, 64))  # 调整到固定大小
        crop = crop[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        crop = np.ascontiguousarray(crop)
        crop = torch.from_numpy(crop).to(self.device).float() / 255.0
        crop = crop.unsqueeze(0)
        
        # 提取特征
        with torch.no_grad():
            feature = self.model(crop).cpu().numpy().flatten()
        
        # 归一化特征向量
        feature = feature / np.linalg.norm(feature)
        
        return feature

class SimpleExtractor:
    """
    简化版特征提取器，用于演示
    """
    def __call__(self, x):
        # 返回随机特征向量作为演示
        return torch.randn(1, 128)  # 假设特征维度为128

class DeepSORTTracker:
    """
    DeepSORT多目标跟踪器
    """
    def __init__(self, config: Dict):
        """
        初始化DeepSORT跟踪器
        
        Args:
            config: 配置字典
        """
        self.max_age = config['max_age']
        self.n_init = config['n_init']
        self.nn_budget = config['nn_budget']
        self.max_iou_distance = config['max_iou_distance']
        
        # 强制使用CPU，避免CUDA相关错误
        self.device = torch.device('cpu')
        
        # 初始化特征提取器
        self.extractor = ReIDExtractor(
            config['model_path'],
            device=self.device
        )
        
        # 跟踪器列表
        self.trackers = []
        # 特征库 {track_id: [features]}
        self.feature_sets = {}
    
    def _iou_distance(self, tracks: List[KalmanBoxTracker], detections: List[Dict]) -> np.ndarray:
        """
        计算跟踪器和检测之间的IOU距离矩阵
        
        Args:
            tracks: 跟踪器列表
            detections: 检测列表
            
        Returns:
            距离矩阵
        """
        # 获取所有跟踪器的预测边界框
        track_boxes = np.array([track.get_state() for track in tracks])
        # 获取所有检测的边界框
        det_boxes = np.array([det['bbox'] for det in detections])
        
        # 计算IOU矩阵
        iou_matrix = np.zeros((len(track_boxes), len(det_boxes)))
        
        for i, (x1, y1, w1, h1) in enumerate(track_boxes):
            for j, (x2, y2, w2, h2) in enumerate(det_boxes):
                # 计算交集
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                
                # 计算并集
                union_area = w1 * h1 + w2 * h2 - inter_area
                
                # 计算IOU
                iou = inter_area / union_area if union_area > 0 else 0
                
                # 距离 = 1 - IOU
                iou_matrix[i, j] = 1 - iou
        
        return iou_matrix
    
    def _nn_distance(self, tracks: List[KalmanBoxTracker], detections: List[Dict], 
                    image: np.ndarray) -> np.ndarray:
        """
        计算最近邻特征距离
        
        Args:
            tracks: 跟踪器列表
            detections: 检测列表
            image: 输入图像
            
        Returns:
            距离矩阵
        """
        n_track = len(tracks)
        n_detection = len(detections)
        dist_matrix = np.zeros((n_track, n_detection))
        
        # 提取所有检测的特征
        det_features = []
        for det in detections:
            feature = self.extractor.extract(image, det['bbox'])
            det_features.append(feature)
        
        # 计算特征距离
        for i, track in enumerate(tracks):
            track_id = track.id
            # 获取该跟踪器的特征集
            if track_id not in self.feature_sets:
                self.feature_sets[track_id] = []
            
            # 如果没有特征，使用IOU距离作为替代
            if len(self.feature_sets[track_id]) == 0:
                dist_matrix[i, :] = np.ones(n_detection) * float('inf')
                continue
            
            # 计算特征距离
            track_features = np.array(self.feature_sets[track_id])
            for j, det_feature in enumerate(det_features):
                # 计算最近邻距离
                dists = np.linalg.norm(track_features - det_feature, axis=1)
                min_dist = np.min(dists)
                dist_matrix[i, j] = min_dist
        
        return dist_matrix
    
    def update(self, detections: List[Dict], image: np.ndarray):
        """
        更新跟踪器
        
        Args:
            detections: 检测列表
            image: 输入图像
        """
        # 预测所有跟踪器的下一个状态
        for tracker in self.trackers:
            tracker.predict()
        
        # 如果没有检测，只需要预测
        if len(detections) == 0:
            # 移除长时间未更新的跟踪器
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return
        
        # 如果没有跟踪器，初始化新的跟踪器
        if len(self.trackers) == 0:
            for det in detections:
                tracker = KalmanBoxTracker(det['bbox'])
                self.trackers.append(tracker)
                # 提取特征
                feature = self.extractor.extract(image, det['bbox'])
                self.feature_sets[tracker.id] = [feature]
            return
        
        # 计算距离矩阵
        iou_dist = self._iou_distance(self.trackers, detections)
        nn_dist = self._nn_distance(self.trackers, detections, image)
        
        # 组合距离
        # 对于已确认的跟踪器，使用特征距离；对于未确认的跟踪器，使用IOU距离
        dist_matrix = np.zeros_like(iou_dist)
        for i, tracker in enumerate(self.trackers):
            if tracker.hit_streak >= self.n_init:
                # 已确认的跟踪器，使用特征距离
                dist_matrix[i, :] = nn_dist[i, :]
            else:
                # 未确认的跟踪器，使用IOU距离
                dist_matrix[i, :] = iou_dist[i, :]
        
        # 使用匈牙利算法进行关联
        row_indices, col_indices = linear_sum_assignment(dist_matrix)
        
        # 处理匹配的跟踪器和检测
        matches = []
        unmatched_tracks = list(range(len(self.trackers)))
        unmatched_detections = list(range(len(detections)))
        
        # 移除匹配的索引
        for r, c in zip(row_indices, col_indices):
            if dist_matrix[r, c] <= self.max_iou_distance:
                matches.append((r, c))
                unmatched_tracks.remove(r)
                unmatched_detections.remove(c)
        
        # 更新匹配的跟踪器
        for r, c in matches:
            tracker = self.trackers[r]
            det = detections[c]
            tracker.update(det['bbox'])
            
            # 更新特征集
            feature = self.extractor.extract(image, det['bbox'])
            if tracker.id not in self.feature_sets:
                self.feature_sets[tracker.id] = []
            self.feature_sets[tracker.id].append(feature)
            
            # 保持特征库大小
            if self.nn_budget > 0 and len(self.feature_sets[tracker.id]) > self.nn_budget:
                self.feature_sets[tracker.id].pop(0)
        
        # 初始化新的跟踪器用于未匹配的检测
        for c in unmatched_detections:
            det = detections[c]
            tracker = KalmanBoxTracker(det['bbox'])
            self.trackers.append(tracker)
            
            # 提取特征
            feature = self.extractor.extract(image, det['bbox'])
            self.feature_sets[tracker.id] = [feature]
        
        # 移除长时间未更新的跟踪器
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
    
    def get_tracks(self) -> List[Dict]:
        """
        获取当前所有活跃的跟踪结果
        
        Returns:
            跟踪结果列表
        """
        tracks = []
        for tracker in self.trackers:
            # 只返回已确认的跟踪器
            if tracker.hit_streak >= self.n_init:
                bbox = tracker.get_state()
                tracks.append({
                    'bbox': bbox,
                    'track_id': tracker.id,
                    'age': tracker.age,
                    'hits': tracker.hits
                })
        
        return tracks
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """
        在图像上绘制跟踪结果
        
        Args:
            frame: 输入图像
            tracks: 跟踪结果列表
            
        Returns:
            绘制了跟踪结果的图像
        """
        for track in tracks:
            x, y, w, h = track['bbox']
            track_id = track['track_id']
            
            # 为每个跟踪ID生成唯一的颜色
            color = self._get_track_color(track_id)
            
            # 绘制边界框
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # 绘制跟踪ID
            label = f'ID: {track_id}'
            cv2.putText(frame, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        根据跟踪ID生成唯一的颜色
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            BGR颜色
        """
        # 简单的颜色生成算法
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color

# 导入OpenCV
import cv2

# 测试代码
if __name__ == '__main__':
    config = {
        'model_path': '../models/deepsort/model.pth',
        'max_age': 30,
        'n_init': 3,
        'nn_budget': 100,
        'max_iou_distance': 0.7,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
    }
    
    tracker = DeepSORTTracker(config)
    print("DeepSORT跟踪器初始化完成")
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟检测结果
    mock_detections = [
        {'bbox': [100, 100, 50, 100], 'confidence': 0.95, 'class_id': 0},
        {'bbox': [200, 150, 80, 60], 'confidence': 0.92, 'class_id': 2}
    ]
    
    # 更新跟踪器
    tracker.update(mock_detections, test_image)
    
    # 获取跟踪结果
    tracks = tracker.get_tracks()
    print(f"跟踪到 {len(tracks)} 个目标")
    
    # 绘制跟踪结果
    result_image = tracker.draw_tracks(test_image.copy(), tracks)
    print("跟踪结果绘制完成")