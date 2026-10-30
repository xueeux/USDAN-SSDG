import os
import sys
import json
import random
from glob import glob
from PIL import Image
import numpy as np

# 添加MXNet MTCNN路径
mtcnn_path = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\mxnet_mtcnn_face_detection'
sys.path.insert(0, mtcnn_path)

try:
    #from mtcnn_detector import MtcnnDetector
    import mxnet as mx
    MXNET_MTCNN_AVAILABLE = True
    print("MXNet MTCNN 已加载")
except ImportError as e:
    MXNET_MTCNN_AVAILABLE = False
    print(f"警告: MXNet MTCNN 不可用: {e}")
    print("将使用简单的居中裁剪（不推荐，但可运行）")

class AIFacePreprocessor:
    def __init__(self, data_root, output_image_dir, json_output_dir, image_size=256, margin=20, train_ratio=0.5, val_ratio=0.25, seed=666):
        """
        data_root: AI-face-detection-Dataset 目录路径
        output_image_dir: 处理后图像的输出目录（如 E:\SSDG_Processed_Data_MXNet\data\ai_256）
        json_output_dir: 标注JSON文件的输出目录（如 D:\...\data_label\AI_dataset）
        image_size: 最终图像尺寸
        margin: 人脸框边距
        train_ratio: 训练集比例（剩余为测试集）
        seed: 随机种子
        """
        self.data_root = data_root
        self.output_image_dir = output_image_dir
        self.json_output_dir = json_output_dir
        self.image_size = image_size
        self.margin = margin
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio   # 自动计算测试比例
        self.seed = seed
        
        # 初始化MTCNN
        self.mtcnn = None
        #self._init_mtcnn()
        
        # 创建输出目录
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.json_output_dir, exist_ok=True)
        
    # def _init_mtcnn(self):
    #     if not MXNET_MTCNN_AVAILABLE:
    #         return
    #     try:
    #         model_dir = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\mxnet_mtcnn_face_detection\model'
    #         ctx = mx.cpu()
    #         self.mtcnn = MtcnnDetector(
    #             model_folder=model_dir,
    #             ctx=ctx,
    #             num_worker=1,
    #             accurate_landmark=True,
    #             threshold=[0.6, 0.7, 0.8]
    #         )
    #         print("✅ MTCNN初始化成功")
    #     except Exception as e:
    #         print(f"MTCNN初始化失败: {e}")
    #         self.mtcnn = None
    
    # def detect_and_crop(self, image):
    #     """检测人脸，返回裁剪并缩放后的PIL图像"""
    #     try:
    #         if self.mtcnn is None:
    #             # 备选：直接缩放
    #             return image.resize((self.image_size, self.image_size), Image.BILINEAR)
            
    #         img_np = np.array(image)
    #         results = self.mtcnn.detect_face(img_np)
    #         if results is not None and len(results[0]) > 0:
    #             boxes = results[0]
    #             best_idx = np.argmax(boxes[:, 4])
    #             x1, y1, x2, y2 = boxes[best_idx][:4].astype(int)
    #             x1 = max(0, x1 - self.margin)
    #             y1 = max(0, y1 - self.margin)
    #             x2 = min(img_np.shape[1], x2 + self.margin)
    #             y2 = min(img_np.shape[0], y2 + self.margin)
    #             face = image.crop((x1, y1, x2, y2))
    #             face = face.resize((self.image_size, self.image_size), Image.BILINEAR)
    #             return face
    #         else:
    #             return image.resize((self.image_size, self.image_size), Image.BILINEAR)
    #     except Exception as e:
    #         print(f"人脸检测出错: {e}")
    #         return image.resize((self.image_size, self.image_size), Image.BILINEAR)
    
    # def process_image(self, src_path, dst_path):
    #     """处理单张图片：检测人脸、缩放、保存"""
    #     try:
    #         img = Image.open(src_path).convert('RGB')
    #         face = self.detect_and_crop(img)
    #         face.save(dst_path, 'PNG')
    #         return True
    #     except Exception as e:
    #         print(f"处理失败 {src_path}: {e}")
    #         return False
    
    def run(self):
        print("开始处理AI数据集...")
        # real_dir = os.path.join(self.data_root, 'real')
        # ai_dir = os.path.join(self.data_root, 'AI')
        
        # all_items = []  # 存储 (src_path, label, dst_path)
        # video_id = 0
        
        # # 处理 real (label=1)
        # for img_path in glob(os.path.join(real_dir, '*.*')):
        #     if img_path.lower().endswith(('.jpg','.png','.jpeg')):
        #         # 生成输出路径
        #         rel_path = os.path.relpath(img_path, self.data_root)
        #         dst_name = rel_path.replace('\\', '_').replace('/', '_').replace('.', '_') + '.png'
        #         dst_path = os.path.join(self.output_image_dir, dst_name)
        #         if self.process_image(img_path, dst_path):
        #             all_items.append({
        #                 'photo_path': dst_path,
        #                 'photo_label': 1,
        #                 'photo_belong_to_video_ID': video_id
        #             })
        #             video_id += 1
        
        # # 处理 AI (label=0)
        # for img_path in glob(os.path.join(ai_dir, '*.*')):
        #     if img_path.lower().endswith(('.jpg','.png','.jpeg')):
        #         rel_path = os.path.relpath(img_path, self.data_root)
        #         dst_name = rel_path.replace('\\', '_').replace('/', '_').replace('.', '_') + '.png'
        #         dst_path = os.path.join(self.output_image_dir, dst_name)
        #         if self.process_image(img_path, dst_path):
        #             all_items.append({
        #                 'photo_path': dst_path,
        #                 'photo_label': 0,
        #                 'photo_belong_to_video_ID': video_id
        #             })
        #             video_id += 1
        
        # print(f"总共处理 {len(all_items)} 张图片，{video_id} 个独立视频")
                # 新逻辑：直接扫描输出图像目录中的 .png 文件
        image_dir = self.output_image_dir
        all_image_paths = glob(os.path.join(image_dir, '*.png'))
        if not all_image_paths:
            print(f"错误: 在 {image_dir} 中没有找到任何 .png 文件，请确保图像已处理完毕。")
            return False
        
        all_items = []
        video_id = 0
        for img_path in all_image_paths:
            # 从文件名推断标签：包含 'real' 的为真实（label=1），否则为伪造（label=0）
            filename = os.path.basename(img_path).lower()
            if 'real' in filename:
                label = 1
            else:
                label = 0
            all_items.append({
                'photo_path': img_path,
                'photo_label': label,
                'photo_belong_to_video_ID': video_id
            })
            video_id += 1
        
        print(f"总共找到 {len(all_items)} 张图片，{video_id} 个独立视频")
        # 随机打乱并划分训练/测试集
        random.seed(self.seed)
        random.shuffle(all_items)
        n_total = len(all_items)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        # 测试集取剩余（保证总数一致）
        n_test = n_total - n_train - n_val
        
        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train+n_val]
        test_items = all_items[n_train+n_val:]
        
        # 重新分配视频 ID（每个样本独立视频，ID 全局唯一）
        for i, item in enumerate(train_items):
            item['photo_belong_to_video_ID'] = i
        for i, item in enumerate(val_items):
            item['photo_belong_to_video_ID'] = n_train + i
        for i, item in enumerate(test_items):
            item['photo_belong_to_video_ID'] = n_train + n_val + i
        
        # 保存 JSON 文件
        with open(os.path.join(self.json_output_dir, 'all_label.json'), 'w') as f:
            json.dump(all_items, f, indent=4)
        with open(os.path.join(self.json_output_dir, 'train_label.json'), 'w') as f:
            json.dump(train_items, f, indent=4)
        with open(os.path.join(self.json_output_dir, 'val_label.json'), 'w') as f:
            json.dump(val_items, f, indent=4)
        with open(os.path.join(self.json_output_dir, 'test_label.json'), 'w') as f:
            json.dump(test_items, f, indent=4)
        
        # 按标签分类保存（可选）
        real_items = [x for x in all_items if x['photo_label'] == 1]
        fake_items = [x for x in all_items if x['photo_label'] == 0]
        with open(os.path.join(self.json_output_dir, 'real_label.json'), 'w') as f:
            json.dump(real_items, f, indent=4)
        with open(os.path.join(self.json_output_dir, 'fake_label.json'), 'w') as f:
            json.dump(fake_items, f, indent=4)
        
        print(f"标注文件已保存至: {self.json_output_dir}")
        print(f"训练集: {len(train_items)} 张, 验证集: {len(val_items)} 张, 测试集: {len(test_items)} 张")
        return True

if __name__ == '__main__':
    #data_root = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\AI-face-detection-Dataset'
    data_root = r"D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\AI_dataset"
    output_image_dir = r'E:\SSDG_Processed_Data_MXNet\data\ai_256'   # 处理后图像存放位置
    json_output_dir = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\data_label\AI_dataset'
    
    preprocessor = AIFacePreprocessor(data_root, output_image_dir, json_output_dir, train_ratio=0.5, val_ratio=0.25)
    preprocessor.run()