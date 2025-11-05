import os
import gdown
import zipfile
import shutil

def download_yolov7_model(model_dir='models', model_name='yolov7.pt'):
    """
    下载YOLOv7预训练模型
    
    Args:
        model_dir: 模型保存目录
        model_name: 模型文件名
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"模型文件已存在: {model_path}")
        return model_path
    
    print("正在下载YOLOv7预训练模型...")
    
    try:
        # YOLOv7预训练模型的Google Drive链接
        # 注意：实际使用时可能需要更新这个链接
        url = 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt'
        
        # 使用gdown下载模型
        gdown.download(url, model_path, quiet=False)
        
        print(f"YOLOv7模型下载完成: {model_path}")
        return model_path
    except Exception as e:
        print(f"下载YOLOv7模型失败: {e}")
        print("请手动从以下链接下载模型并放置在models目录下:")
        print("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
        return None

def download_deepsort_model(model_dir='models/deepsort', model_name='model.pth'):
    """
    下载DeepSORT预训练模型
    
    Args:
        model_dir: 模型保存目录
        model_name: 模型文件名
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_path):
        print(f"DeepSORT模型文件已存在: {model_path}")
        return model_path
    
    print("正在下载DeepSORT预训练模型...")
    
    try:
        # DeepSORT预训练模型的下载链接
        # 这里使用一个示例链接，实际使用时可能需要更新
        url = 'https://drive.google.com/uc?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp'
        zip_path = os.path.join(model_dir, 'deepsort.zip')
        
        # 下载并解压模型
        gdown.download(url, zip_path, quiet=False)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # 清理zip文件
        os.remove(zip_path)
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试找到解压后的模型文件
            for root, _, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pth'):
                        found_path = os.path.join(root, file)
                        shutil.move(found_path, model_path)
                        print(f"DeepSORT模型已找到并移动到: {model_path}")
                        return model_path
            
            print(f"解压后未找到模型文件: {model_path}")
            return None
        
        print(f"DeepSORT模型下载完成: {model_path}")
        return model_path
    except Exception as e:
        print(f"下载DeepSORT模型失败: {e}")
        print("请手动下载DeepSORT模型并放置在models/deepsort目录下")
        return None

def download_all_models():
    """
    下载所有必要的模型
    """
    print("===== 开始下载所有模型 =====")
    
    # 下载YOLOv7模型
    yolo_path = download_yolov7_model()
    
    # 下载DeepSORT模型
    deepsort_path = download_deepsort_model()
    
    print("===== 模型下载完成 =====")
    
    return {
        'yolov7': yolo_path,
        'deepsort': deepsort_path
    }

if __name__ == '__main__':
    download_all_models()
    print("\n请确保模型下载成功后再运行主程序。")
    print("如果自动下载失败，请手动下载模型并放置在相应目录。")