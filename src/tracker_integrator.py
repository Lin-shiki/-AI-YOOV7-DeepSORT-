import cv2
import yaml
import os
import torch
import numpy as np
from typing import Dict, List, Optional
from src.yolo_detector import YOLOv7Detector
from src.deep_sort_tracker import DeepSORTTracker

class DroneTracker:
    """
    无人机跟踪器，集成YOLOv7目标检测和DeepSORT跟踪，支持单目标和多目标跟踪
    """
    def __init__(self, config_path: str):
        """
        初始化无人机跟踪器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化YOLOv7检测器
        self.detector = YOLOv7Detector(self.config['yolo'])
        
        # 初始化DeepSORT跟踪器
        self.tracker = DeepSORTTracker(self.config['deepsort'])
        
        # 跟踪配置
        self.display = self.config['tracking']['display']
        self.save_output = self.config['tracking']['save_output']
        self.output_path = self.config['tracking']['output_path']
        self.classes_to_track = self.config['tracking']['classes_to_track']
        self.track_color_map = self.config['tracking']['track_color_map']
        
        # 确保输出目录存在
        if self.save_output:
            os.makedirs(self.output_path, exist_ok=True)
        
        # 单目标跟踪的目标ID
        self.target_track_id = None
        self.target_confidence_threshold = 0.5
        
        # 多目标跟踪相关变量
        self.max_track_history = 30  # 保存的历史轨迹长度
        self.track_histories = {}  # 保存每个目标的轨迹 {track_id: [(x, y)]}
        self.total_tracked_objects = 0  # 总共跟踪过的目标数量
        self.active_track_ids = set()  # 当前活跃的跟踪ID集合
        self.track_colors = {}  # 每个跟踪ID的颜色映射 {track_id: (r, g, b)}
    
    def _load_config(self, config_path: str) -> Dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        根据配置过滤检测结果
        
        Args:
            detections: 原始检测结果列表
            
        Returns:
            过滤后的检测结果列表
        """
        filtered = []
        for det in detections:
            # 检查是否是需要跟踪的类别
            if det['class_id'] in self.classes_to_track:
                # 检查置信度
                if det['confidence'] >= self.target_confidence_threshold:
                    filtered.append(det)
        return filtered
    
    def select_target(self, detections: List[Dict]) -> Optional[Dict]:
        """
        选择要跟踪的目标（单目标跟踪）
        
        Args:
            detections: 检测结果列表
            
        Returns:
            选定的目标，如果没有则返回None
        """
        if not detections:
            return None
        
        # 按置信度排序，选择置信度最高的目标
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[0]
    
    def process_frame(self, frame: cv2.Mat, frame_idx: int = 0, mode: str = 'single') -> Dict:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像
            frame_idx: 帧索引
            mode: 跟踪模式，'single' 或 'multi'
            
        Returns:
            处理结果字典，包含检测结果、跟踪结果和处理后的图像
        """
        # 执行目标检测
        detections = self.detector.detect(frame)
        
        # 过滤检测结果
        filtered_detections = self.filter_detections(detections)
        
        # 更新跟踪器
        self.tracker.update(filtered_detections, frame)
        
        # 获取跟踪结果
        tracks = self.tracker.get_tracks()
        
        # 更新轨迹历史（多目标跟踪需要）
        self._update_track_histories(tracks)
        
        # 选择目标（单目标跟踪模式）
        selected_target = None
        if mode == 'single' and tracks:
            # 如果没有指定目标ID，选择第一个出现的目标
            if self.target_track_id is None and filtered_detections:
                selected_detection = self.select_target(filtered_detections)
                # 查找与选定检测最匹配的跟踪
                if selected_detection:
                    min_dist = float('inf')
                    for track in tracks:
                        # 计算IOU
                        iou = self._calculate_iou(selected_detection['bbox'], track['bbox'])
                        if iou > 0.5 and (1 - iou) < min_dist:  # 1-iou作为距离
                            min_dist = 1 - iou
                            self.target_track_id = track['track_id']
            
            # 获取目标跟踪
            for track in tracks:
                if track['track_id'] == self.target_track_id:
                    selected_target = track
                    break
        
        # 绘制检测和跟踪结果
        result_frame = frame.copy()
        
        # 多目标模式：绘制轨迹
        if mode == 'multi':
            self._draw_track_trajectories(result_frame)
        
        # 绘制所有检测结果（半透明）
        for det in filtered_detections:
            x, y, w, h = det['bbox']
            # 绘制半透明边界框
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.3, result_frame, 0.7, 0, result_frame)
        
        # 绘制所有跟踪结果
        for track in tracks:
            x, y, w, h = track['bbox']
            track_id = track['track_id']
            
            # 根据模式生成颜色
            if mode == 'single':
                color = (0, 0, 255) if track_id == self.target_track_id else (255, 0, 0)
            else:  # multi
                color = self._generate_track_color(track_id)
            
            # 绘制边界框
            cv2.rectangle(result_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # 绘制跟踪ID和年龄
            label = f'ID: {track_id}'
            cv2.putText(result_frame, label, (int(x), int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 单目标模式：标记目标跟踪
        if mode == 'single' and selected_target:
            x, y, w, h = selected_target['bbox']
            # 绘制特殊标记
            cv2.rectangle(result_frame, (int(x) - 2, int(y) - 2), 
                         (int(x + w) + 2, int(y + h) + 2), (0, 0, 255), 3)
            cv2.putText(result_frame, 'TARGET', (int(x), int(y) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
        
        # 显示统计信息
        if mode == 'single':
            info_text = f'Detections: {len(filtered_detections)}, Tracks: {len(tracks)}'
            if selected_target:
                info_text += f', Target ID: {selected_target["track_id"]}'
        else:  # multi
            info_text = f'Detections: {len(filtered_detections)}, Tracks: {len(tracks)}, Total: {self.total_tracked_objects}'
        
        cv2.putText(result_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 添加模式标识
        mode_text = f'Mode: {mode.upper()}'
        cv2.putText(result_frame, mode_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return {
            'detections': filtered_detections,
            'tracks': tracks,
            'target': selected_target,
            'frame': result_frame,
            'total_objects': self.total_tracked_objects
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算两个边界框的IOU
        
        Args:
            bbox1: 第一个边界框 [x, y, width, height]
            bbox2: 第二个边界框 [x, y, width, height]
            
        Returns:
            IOU值
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
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
        
        return iou
    
    def reset_tracking(self):
        """
        重置跟踪状态，用于开始新的跟踪
        """
        # 重置单目标跟踪相关
        self.target_track_id = None
        
        # 重置多目标跟踪相关
        self.track_histories = {}
        self.total_tracked_objects = 0
        self.active_track_ids = set()
        self.track_colors = {}
        
        # 清除跟踪器中的所有跟踪
        self.tracker.trackers = []
        self.tracker.feature_sets = {}
    
    def _generate_track_color(self, track_id: int) -> tuple:
        """
        为跟踪ID生成唯一的颜色
        
        Args:
            track_id: 跟踪ID
            
        Returns:
            BGR颜色元组
        """
        if track_id not in self.track_colors:
            # 使用固定种子生成一致的颜色
            np.random.seed(track_id)
            self.track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.track_colors[track_id]
    
    def _update_track_histories(self, tracks: List[Dict]):
        """
        更新跟踪历史轨迹
        
        Args:
            tracks: 当前的跟踪结果列表
        """
        current_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            current_ids.add(track_id)
            
            # 如果是新出现的跟踪ID，更新总数
            if track_id not in self.active_track_ids:
                self.total_tracked_objects += 1
                self.active_track_ids.add(track_id)
            
            # 获取边界框中心
            x, y, w, h = track['bbox']
            center_x = x + w / 2
            center_y = y + h / 2
            
            # 更新轨迹历史
            if track_id not in self.track_histories:
                self.track_histories[track_id] = []
            
            self.track_histories[track_id].append((center_x, center_y))
            
            # 限制历史长度
            if len(self.track_histories[track_id]) > self.max_track_history:
                self.track_histories[track_id].pop(0)
        
        # 更新活跃ID集合
        self.active_track_ids = current_ids
    
    def _draw_track_trajectories(self, frame: cv2.Mat):
        """
        绘制跟踪轨迹
        
        Args:
            frame: 要绘制的图像
        """
        for track_id, history in self.track_histories.items():
            # 只绘制活跃目标的轨迹
            if track_id in self.active_track_ids and len(history) > 1:
                color = self._generate_track_color(track_id)
                
                # 绘制轨迹线段
                for i in range(1, len(history)):
                    # 根据时间调整透明度
                    alpha = i / len(history)
                    # 创建半透明图层
                    overlay = frame.copy()
                    cv2.line(overlay, 
                            (int(history[i-1][0]), int(history[i-1][1])),
                            (int(history[i][0]), int(history[i][1])),
                            color, 2)
                    # 应用透明度
                    cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, mode: str = 'single') -> None:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径，如果不提供则使用配置中的路径
            mode: 跟踪模式，'single' 或 'multi'
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        print(f"跟踪模式: {mode}")
        
        # 设置输出视频
        out = None
        if self.save_output or output_path:
            if output_path is None:
                # 生成输出文件名
                video_name = os.path.basename(video_path)
                output_path = os.path.join(self.output_path, f'tracked_{mode}_{video_name}')
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.config['video']['codec'])
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"输出视频将保存到: {output_path}")
        
        # 重置跟踪状态
        self.reset_tracking()
        
        # 处理每一帧
        frame_idx = 0
        max_objects = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            result = self.process_frame(frame, frame_idx, mode)
            
            # 更新最大目标数
            current_tracks = len(result['tracks'])
            if current_tracks > max_objects:
                max_objects = current_tracks
            
            # 显示结果
            if self.display:
                cv2.imshow(f'Drone Tracking - {mode.upper()}', result['frame'])
                # 按ESC键退出
                if cv2.waitKey(1) == 27:
                    break
            
            # 保存结果
            if out:
                out.write(result['frame'])
            
            # 显示进度
            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100
                print(f"处理进度: {frame_idx}/{total_frames} ({progress:.1f}%) - 当前跟踪: {current_tracks}, 最大跟踪: {max_objects}")
        
        # 释放资源
        cap.release()
        if out:
            out.release()
        if self.display:
            cv2.destroyAllWindows()
        
        print(f"视频处理完成")
        print(f"跟踪统计: 总共跟踪目标数: {self.total_tracked_objects}, 最大同时跟踪数: {max_objects}")
    
    def track_single_object(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        单目标跟踪模式
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
        """
        print("===== 开始单目标跟踪 ====")
        # 确保只跟踪一个目标（通过过滤检测结果实现）
        self.target_confidence_threshold = 0.3  # 稍微降低阈值以确保能检测到目标
        
        # 处理视频
        self.process_video(video_path, output_path, mode='single')
        
        print("===== 单目标跟踪完成 ====")
    
    def track_multi_objects(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        多目标跟踪模式
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
        """
        print("===== 开始多目标跟踪 ====")
        # 多目标跟踪时使用较低的置信度阈值以捕获更多目标
        self.target_confidence_threshold = 0.25
        
        # 处理视频
        self.process_video(video_path, output_path, mode='multi')
        
        print("===== 多目标跟踪完成 ====")

# 测试代码
if __name__ == '__main__':
    # 创建测试视频
    def create_test_video(output_path='../input/test_video.mp4', duration=5, fps=30, num_objects=3):
        """
        创建测试视频用于演示，支持多目标
        
        Args:
            output_path: 输出视频路径
            duration: 视频时长（秒）
            fps: 帧率
            num_objects: 对象数量
        """
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 创建移动的对象
        for i in range(duration * fps):
            # 创建黑色背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 为每个对象创建不同的运动轨迹
            for obj_id in range(num_objects):
                # 不同的运动模式
                if obj_id == 0:
                    # 圆形运动
                    center_x = width // 2 + int(100 * np.cos(i * 0.1 + obj_id))
                    center_y = height // 2 + int(100 * np.sin(i * 0.1 + obj_id))
                    size = 20
                elif obj_id == 1:
                    # 直线运动
                    center_x = 50 + int((width - 100) * i / (duration * fps))
                    center_y = height // 3
                    size = 15
                else:
                    # 正弦运动
                    center_x = width // 2
                    center_y = height // 2 + int(100 * np.sin(i * 0.15 + obj_id))
                    size = 18
                
                # 根据对象ID设置不同颜色
                colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)]
                color = colors[obj_id % len(colors)]
                
                # 绘制对象（矩形）
                cv2.rectangle(frame, 
                             (center_x - size, center_y - size * 2), 
                             (center_x + size, center_y + size * 2), 
                             color, -1)
                
                # 添加对象ID标签
                cv2.putText(frame, f'Obj {obj_id}', 
                            (center_x - size, center_y - size * 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加一些随机干扰点
            for _ in range(10):
                rand_x = np.random.randint(0, width)
                rand_y = np.random.randint(0, height)
                cv2.circle(frame, (rand_x, rand_y), 2, (100, 100, 100), -1)
            
            # 添加时间戳
            timestamp = f'Time: {i/fps:.1f}s'
            cv2.putText(frame, timestamp, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 写入帧
            out.write(frame)
        
        out.release()
        print(f"测试视频创建完成: {output_path}")
        print(f"视频包含 {num_objects} 个移动对象")
        return output_path
    
    # 创建测试视频
    import numpy as np
    test_video_path = create_test_video(num_objects=4)
    
    # 初始化跟踪器
    tracker = DroneTracker('../config.yaml')
    
    # 先测试单目标跟踪
    print("\n=== 测试单目标跟踪 ===")
    tracker.track_single_object(test_video_path, '../output/tracked_single_test.mp4')
    
    # 再测试多目标跟踪
    print("\n=== 测试多目标跟踪 ===")
    tracker.track_multi_objects(test_video_path, '../output/tracked_multi_test.mp4')
    
    print("\n所有测试完成！请查看output目录下的结果视频。")