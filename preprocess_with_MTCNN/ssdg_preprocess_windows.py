"""
数据集目录结构：
E:\SSDG_Dataset\
├─CASIS_FASD\
├─Idiap_Replay_Attack\
├─MSU_MFSD\
└─OULU_NPU\
以下是采取小样本的数据预处理实现
"""

import os
import sys
import cv2
import glob
import json
import math
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings
warnings.filterwarnings('ignore')

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加MXNet MTCNN路径
mtcnn_path = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\mxnet_mtcnn_face_detection'
sys.path.insert(0, mtcnn_path)

try:
    from mtcnn_detector import MtcnnDetector
    MXNET_MTCNN_AVAILABLE = True
    print("MXNet MTCNN 已加载")
except ImportError as e:
    MXNET_MTCNN_AVAILABLE = False
    print(f"MXNet MTCNN 加载失败: {e}")
    print("将使用备用人脸检测方法")

class SSDGPreprocessor_MXNet:
    """SSDG预处理类 - 使用MXNet MTCNN"""
    
    def __init__(self, base_dir="E:\\SSDG_Dataset", config_file=None):
        self.base_dir = base_dir
        
        # # 默认配置 - 小样本1500张
        # self.default_config = {
        #     "target_total_images": 1900,
            
        #     # MSU-MFSD配置
        #     "MSU_MFSD": {
        #         "enabled": True,
        #         "real_videos": 12,
        #         "attack_videos": 12,
        #         "frames_per_video": 15,
        #         "total_frames": 360
        #     },
            
        #     # Replay-Attack配置
        #     "Replay_Attack": {
        #         "enabled": True,
        #         "videos_per_set": 5,
        #         "frames_per_video": 10,
        #         "total_frames": 600
        #     },
            
        #     # CASIA-FASD配置
        #     "CASIA_FASD": {
        #         "enabled": True,
        #         "train_live": 80,
        #         "train_spoof": 80,
        #         "test_live": 80,
        #         "test_spoof": 80,
        #         "frames_per_video": 1,
        #         "total_frames": 320
        #     },
            
        #     # OULU-NPU配置
        #     "OULU_NPU": {
        #         "enabled": True,
        #         "folders_per_set": 20,
        #         "images_per_folder": 10,
        #         "total_frames": 600
        #     },
                # 默认配置 - 小样本1500张
        self.default_config = {
            "target_total_images": 13400,
            
            # MSU-MFSD配置
            "MSU_MFSD": {
                "enabled": True,
                "real_videos": 70,
                "attack_videos": 70,
                "frames_per_video": 15,
                "total_frames": 2100
            },
            
            # Replay-Attack配置
            "Replay_Attack": {
                "enabled": True,
                "videos_per_set": 60,
                "frames_per_video": 10,
                "total_frames": 3600
            },
            
            # CASIA-FASD配置
            "CASIA_FASD": {
                "enabled": True,
                "train_live": 800,
                "train_spoof": 800,
                "test_live": 800,
                "test_spoof": 800,
                "frames_per_video": 1,
                "total_frames": 3200
            },
            
            # OULU-NPU配置
            "OULU_NPU": {
                "enabled": True,
                "folders_per_set": 500,
                "images_per_folder": 3,
                "total_frames": 4500
            },
            
            # 输出目录
            "output": {
                "output_dir": "E:\\SSDG_Processed_Data_MXNet",
                "create_symbolic_links": False
            },
            
            # 处理参数
            "processing": {
                "image_size": 256,
                "margin": 20,
                "min_face_size": 20,
                "save_quality": 95
            }
        }
        
        # 加载自定义配置
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                self._merge_configs(self.default_config, user_config)
        
        self.config = self.default_config
        self.output_dir = self.config["output"]["output_dir"]
        
        # 初始化MTCNN
        self.mtcnn = None
        self._init_mtcnn()
        
        # 创建输出目录
        self.create_directories()
    
    def _merge_configs(self, default, user):
        """合并配置"""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _init_mtcnn(self):
        """初始化MXNet MTCNN检测器"""
        print("正在初始化MXNet MTCNN人脸检测器...")
        
        if not MXNET_MTCNN_AVAILABLE:
            print("警告: MXNet MTCNN不可用，将使用备用人脸检测")
            return
        
        try:
            # 检查模型文件是否存在
            model_dir = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\mxnet_mtcnn_face_detection\model'
            if not os.path.exists(model_dir):
                print(f"错误: MTCNN模型目录不存在: {model_dir}")
                print("请确保已下载mxnet_mtcnn_face_detection仓库")
                return
            
            # 导入MXNet
            import mxnet as mx
            
            # 初始化检测器（使用CPU）
            ctx = mx.cpu()  # 也可以使用GPU: mx.gpu(0)
            self.mtcnn = MtcnnDetector(
                model_folder=model_dir,
                ctx=ctx,
                num_worker=1,
                accurate_landmark=True,
                threshold=[0.6, 0.7, 0.8]
            )
            print("✅ MXNet MTCNN初始化成功")
            
        except Exception as e:
            print(f"MXNet MTCNN初始化失败: {e}")
            print("将使用备用人脸检测方法")
    
    def detect_face(self, image):
        """检测人脸并返回对齐后的面部"""
        try:
            # 如果没有MTCNN，使用备选方法
            if self.mtcnn is None:
                # 简单居中裁剪
                img_resized = image.resize((256, 256), Image.BILINEAR)
                return img_resized, False
            
            # 将PIL图像转换为numpy数组
            img_np = np.array(image)
            
            # MTCNN检测人脸
            results = self.mtcnn.detect_face(img_np)
            
            if results is not None and len(results[0]) > 0:
                boxes = results[0]  # 人脸框
                landmarks = results[1]  # 关键点
                
                # 取置信度最高的人脸
                best_idx = np.argmax(boxes[:, 4])
                box = boxes[best_idx]
                
                # 提取人脸坐标
                x1, y1, x2, y2 = box[:4].astype(int)
                
                # 添加边距
                margin = self.config["processing"]["margin"]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(img_np.shape[1], x2 + margin)
                y2 = min(img_np.shape[0], y2 + margin)
                
                # 裁剪人脸
                face_img = image.crop((x1, y1, x2, y2))
                face_img = face_img.resize((256, 256), Image.BILINEAR)
                return face_img, True
            else:
                # 未检测到人脸，居中裁剪
                img_resized = image.resize((256, 256), Image.BILINEAR)
                return img_resized, False
                
        except Exception as e:
            print(f"人脸检测失败: {e}")
            # 备选方案：居中裁剪
            img_resized = image.resize((256, 256), Image.BILINEAR)
            return img_resized, False
    
    def process_video(self, video_path, output_dir, max_frames=10):
        """处理单个视频"""
        if not os.path.exists(video_path):
            print(f"  视频不存在: {video_path}")
            return 0
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开视频: {video_path}")
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样帧
        frame_indices = []
        if max_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]
        
        processed_count = 0
        
        for idx, frame_num in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            try:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # 检测人脸
                face_img, _ = self.detect_face(pil_img)
                
                # 保存图片
                output_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
                face_img.save(output_path, "PNG", 
                             quality=self.config["processing"]["save_quality"])
                
                processed_count += 1
                
            except Exception as e:
                print(f"    帧处理失败: {e}")
                continue
        
        cap.release()
        print(f"    完成: {processed_count}/{len(frame_indices)} 帧")
        return processed_count
    
    def process_image(self, img_path, output_path):
        """处理单张图片"""
        try:
            img = Image.open(img_path)
            face_img, _ = self.detect_face(img)
            face_img.save(output_path, "PNG", 
                         quality=self.config["processing"]["save_quality"])
            return True
        except Exception as e:
            print(f"  处理失败 {img_path}: {e}")
            return False
    
    def create_directories(self):
        """创建输出目录结构"""
        print("创建目录结构...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        required_dirs = [
            os.path.join(self.output_dir, "data"),
            os.path.join(self.output_dir, "data", "msu_256"),
            os.path.join(self.output_dir, "data", "replay_256"),
            os.path.join(self.output_dir, "data", "casia_256"),
            os.path.join(self.output_dir, "data", "oulu_256"),
            os.path.join(self.output_dir, "data_label"),
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print("✅ 目录结构创建完成")
        return True
    
    def process_msu_mfsd(self):
        """处理MSU-MFSD数据集"""
        if not self.config["MSU_MFSD"]["enabled"]:
            return True
            
        print("\n" + "="*60)
        print("处理 MSU-MFSD 数据集")
        print("="*60)
        
        config = self.config["MSU_MFSD"]
        msu_path = os.path.join(self.base_dir, "MSU_MFSD")
        output_path = os.path.join(self.output_dir, "data", "msu_256")
        
        if not os.path.exists(msu_path):
            print(f"错误：MSU-MFSD路径不存在: {msu_path}")
            return False
        
        real_videos_path = os.path.join(msu_path, "scene01", "real")
        attack_videos_path = os.path.join(msu_path, "scene01", "attack")
        
        os.makedirs(os.path.join(output_path, "real"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "attack"), exist_ok=True)
        
        total_frames = 0
        
        # 处理真实视频
        if os.path.exists(real_videos_path):
            real_videos = [f for f in os.listdir(real_videos_path) 
                          if f.endswith(('.mov', '.mp4'))]
            
            selected_real = real_videos[:config["real_videos"]]
            
            for i, video_file in enumerate(selected_real):
                video_path = os.path.join(real_videos_path, video_file)
                output_folder = f"real_{i+1:03d}"
                output_dir = os.path.join(output_path, "real", output_folder)
                os.makedirs(output_dir, exist_ok=True)
                
                frames_processed = self.process_video(
                    video_path, output_dir, 
                    max_frames=config["frames_per_video"]
                )
                total_frames += frames_processed
        
        # 处理攻击视频
        if os.path.exists(attack_videos_path):
            attack_videos = [f for f in os.listdir(attack_videos_path) 
                           if f.endswith(('.mov', '.mp4'))]
            
            selected_attack = attack_videos[:config["attack_videos"]]
            
            for i, video_file in enumerate(selected_attack):
                video_path = os.path.join(attack_videos_path, video_file)
                output_folder = f"attack_{i+1:03d}"
                output_dir = os.path.join(output_path, "attack", output_folder)
                os.makedirs(output_dir, exist_ok=True)
                
                frames_processed = self.process_video(
                    video_path, output_dir, 
                    max_frames=config["frames_per_video"]
                )
                total_frames += frames_processed
        
        print(f"MSU-MFSD处理完成! 总帧数: {total_frames}")
        return True
    
    def process_replay_attack(self):
        """处理Replay-Attack数据集"""
        if not self.config["Replay_Attack"]["enabled"]:
            return True
            
        print("\n" + "="*60)
        print("处理 Replay-Attack 数据集")
        print("="*60)
        
        config = self.config["Replay_Attack"]
        replay_path = os.path.join(self.base_dir, "Idiap_Replay_Attack")
        output_path = os.path.join(self.output_dir, "data", "replay_256")
        
        if not os.path.exists(replay_path):
            print(f"错误：Replay-Attack路径不存在: {replay_path}")
            return False
        
        dataset_mapping = [
            ("train", "replayattack-train"),
            ("devel", "replayattack-devel"),
            ("test", "replayattack-test")
        ]
        
        total_frames = 0
        
        for src_type, dst_type in dataset_mapping:
            print(f"\n处理 {src_type} 集...")
            
            # === 1. 处理真实视频 ===
            real_src = os.path.join(replay_path, src_type, "real")
            real_dst = os.path.join(output_path, dst_type, "real")
            
            if os.path.exists(real_src):
                real_videos = [f for f in os.listdir(real_src) 
                            if f.lower().endswith('.mov')]
                
                selected_real = real_videos[:config["videos_per_set"]]
                
                for video_file in selected_real:
                    video_path = os.path.join(real_src, video_file)
                    video_name = os.path.splitext(video_file)[0]
                    output_dir = os.path.join(real_dst, video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 使用改进的 process_video 函数
                    frames = self.process_video(
                        video_path, output_dir, 
                        max_frames=config["frames_per_video"]
                    )
                    total_frames += frames
            else:
                print(f"  警告: {real_src} 不存在")
            
                # === 2. 处理攻击视频（修改为直接读取 attack 根目录下的 .mov 文件） ===
            attack_src = os.path.join(replay_path, src_type, "attack")
            attack_dst = os.path.join(output_path, dst_type, "attack")
            
            if os.path.exists(attack_src):
                # 直接列出 attack 目录下的所有 .mov 文件（不再遍历子文件夹）
                attack_videos = [f for f in os.listdir(attack_src) 
                                if f.lower().endswith('.mov')]
                
                if not attack_videos:
                    print(f"  警告: {attack_src} 中没有找到 .mov 视频文件")
                    continue
                
                # 选择指定数量的视频
                selected_attack = attack_videos[:config["videos_per_set"]]
                print(f"    找到 {len(attack_videos)} 个攻击视频，选择前 {len(selected_attack)} 个")
                
                for video_file in selected_attack:
                    video_path = os.path.join(attack_src, video_file)
                    video_name = os.path.splitext(video_file)[0]  # 直接使用文件名（不含扩展名）
                    output_dir = os.path.join(attack_dst, video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    frames = self.process_video(
                        video_path, output_dir, 
                        max_frames=config["frames_per_video"]
                    )
                    if frames == 0:
                        print(f"    警告: 未提取到帧 - {video_file}")
                    total_frames += frames
            else:
                print(f"  警告: {attack_src} 不存在")
        
        print(f"\nReplay-Attack处理完成! 总帧数: {total_frames}")
        return True
    
    def process_casia_fasd(self):
        """处理CASIA-FASD数据集"""
        if not self.config["CASIA_FASD"]["enabled"]:
            return True
            
        print("\n" + "="*60)
        print("处理 CASIA-FASD 数据集")
        print("="*60)
        
        config = self.config["CASIA_FASD"]
        casia_path = os.path.join(self.base_dir, "CASIS_FASD")
        output_path = os.path.join(self.output_dir, "data", "casia_256")
        
        if not os.path.exists(casia_path):
            print(f"错误：CASIA-FASD路径不存在: {casia_path}")
            return False
        
        # 创建SSDG需要的目录结构
        required_dirs = ["1", "2", "HR_1", "3", "HR_3", "train_release", "test_release"]
        for dir_name in required_dirs:
            os.makedirs(os.path.join(output_path, dir_name), exist_ok=True)
        
        total_images = 0
        
        # 处理训练集
        train_live_path = os.path.join(casia_path, "train", "live")
        train_spoof_path = os.path.join(casia_path, "train", "spoof")
        
        if os.path.exists(train_live_path):
            live_images = [f for f in os.listdir(train_live_path) 
                          if f.lower().endswith('.png')]
            selected_live = live_images[:config["train_live"]]
            
            for idx, img_file in enumerate(selected_live):
                img_path = os.path.join(train_live_path, img_file)
                
                # 创建虚拟视频文件夹
                video_folder = f"train_live_{idx+1:03d}"
                video_path = os.path.join(output_path, "train_release", video_folder)
                os.makedirs(video_path, exist_ok=True)
                
                # 处理并保存图片
                output_img_path = os.path.join(video_path, f"frame_000.png")
                if self.process_image(img_path, output_img_path):
                    total_images += 1
                    
                    # 复制到对应目录
                    if idx % 3 == 0:
                        link_dir = "1"
                    elif idx % 3 == 1:
                        link_dir = "2"
                    else:
                        link_dir = "HR_1"
                    
                    link_path = os.path.join(output_path, link_dir, img_file)
                    shutil.copy2(output_img_path, link_path)
        
        if os.path.exists(train_spoof_path):
            spoof_images = [f for f in os.listdir(train_spoof_path) 
                           if f.lower().endswith('.png')]
            selected_spoof = spoof_images[:config["train_spoof"]]
            
            for idx, img_file in enumerate(selected_spoof):
                img_path = os.path.join(train_spoof_path, img_file)
                video_folder = f"train_spoof_{idx+1:03d}"
                video_path = os.path.join(output_path, "train_release", video_folder)
                os.makedirs(video_path, exist_ok=True)
                
                output_img_path = os.path.join(video_path, f"frame_000.png")
                if self.process_image(img_path, output_img_path):
                    total_images += 1
        
        # 处理测试集
        test_live_path = os.path.join(casia_path, "test", "live")
        test_spoof_path = os.path.join(casia_path, "test", "spoof")
        
        if os.path.exists(test_live_path):
            test_live_images = [f for f in os.listdir(test_live_path) 
                               if f.lower().endswith('.png')]
            selected_test_live = test_live_images[:config["test_live"]]
            
            for idx, img_file in enumerate(selected_test_live):
                img_path = os.path.join(test_live_path, img_file)
                video_folder = f"test_live_{idx+1:03d}"
                video_path = os.path.join(output_path, "test_release", video_folder)
                os.makedirs(video_path, exist_ok=True)
                
                output_img_path = os.path.join(video_path, f"frame_000.png")
                if self.process_image(img_path, output_img_path):
                    total_images += 1
                    
                    # 复制到对应目录
                    if idx % 2 == 0:
                        link_dir = "3"
                    else:
                        link_dir = "HR_3"
                    
                    link_path = os.path.join(output_path, link_dir, img_file)
                    shutil.copy2(output_img_path, link_path)
        
        if os.path.exists(test_spoof_path):
            test_spoof_images = [f for f in os.listdir(test_spoof_path) 
                                if f.lower().endswith('.png')]
            selected_test_spoof = test_spoof_images[:config["test_spoof"]]
            
            for idx, img_file in enumerate(selected_test_spoof):
                img_path = os.path.join(test_spoof_path, img_file)
                video_folder = f"test_spoof_{idx+1:03d}"
                video_path = os.path.join(output_path, "test_release", video_folder)
                os.makedirs(video_path, exist_ok=True)
                
                output_img_path = os.path.join(video_path, f"frame_000.png")
                if self.process_image(img_path, output_img_path):
                    total_images += 1
        
        print(f"CASIA-FASD处理完成! 总图片数: {total_images}")
        return True
    
    def process_oulu_npu(self):
        """处理OULU-NPU数据集"""
        if not self.config["OULU_NPU"]["enabled"]:
            return True
            
        print("\n" + "="*60)
        print("处理 OULU-NPU 数据集")
        print("="*60)
        
        config = self.config["OULU_NPU"]
        oulu_path = os.path.join(self.base_dir, "OULU_NPU")
        output_path = os.path.join(self.output_dir, "data", "oulu_256")
        
        if not os.path.exists(oulu_path):
            print(f"错误：OULU-NPU路径不存在: {oulu_path}")
            return False
        
        # 创建目录
        for dir_name in ["Train_files", "Dev_files", "Test_files"]:
            os.makedirs(os.path.join(output_path, dir_name), exist_ok=True)
        
        total_images = 0
        
        # 数据集映射
        dataset_mapping = [
            ("train", "Train_files"),
            ("dev", "Dev_files"),
            ("test", "Test_files")
        ]
        
        for src_type, dst_type in dataset_mapping:
            src_dir = os.path.join(oulu_path, src_type)
            
            if not os.path.exists(src_dir):
                continue
            
            # 获取文件夹
            all_folders = [f for f in os.listdir(src_dir) 
                          if os.path.isdir(os.path.join(src_dir, f))]
            
            # 选择文件夹（确保有真实和伪造）
            real_folders = [f for f in all_folders if f.endswith('_1')]
            fake_folders = [f for f in all_folders if not f.endswith('_1')]
            
            # 选择指定数量的文件夹
            num_real = min(config["folders_per_set"] // 2, len(real_folders))
            num_fake = config["folders_per_set"] - num_real
            
            selected_folders = real_folders[:num_real] + fake_folders[:num_fake]
            
            for folder_name in selected_folders:
                folder_path = os.path.join(src_dir, folder_name)
                output_folder = os.path.join(output_path, dst_type, folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # 获取图片
                jpg_files = [f for f in os.listdir(folder_path) 
                            if f.lower().endswith('.jpg')]
                
                # 按数字排序
                jpg_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
                selected_jpgs = jpg_files[:config["images_per_folder"]]
                
                for jpg_file in selected_jpgs:
                    jpg_path = os.path.join(folder_path, jpg_file)
                    png_name = os.path.splitext(jpg_file)[0] + ".png"
                    output_path_png = os.path.join(output_folder, png_name)
                    
                    if self.process_image(jpg_path, output_path_png):
                        total_images += 1
        
        print(f"OULU-NPU处理完成! 总图片数: {total_images}")
        return True
    
    def run_all(self):
        """运行所有预处理"""
        print("="*70)
        print("SSDG小样本预处理 - MXNet MTCNN版本")
        print("="*70)
        print(f"原始数据目录: {self.base_dir}")
        print(f"输出数据目录: {self.output_dir}")
        print("="*70)
        
        # 创建目录
        self.create_directories()
        
        # 处理数据集
        try:
            print("\n开始处理数据集...")
            
            # if self.config["MSU_MFSD"]["enabled"]:
            #     self.process_msu_mfsd()
            
            if self.config["Replay_Attack"]["enabled"]:
                self.process_replay_attack()
            
            # if self.config["CASIA_FASD"]["enabled"]:
            #     self.process_casia_fasd()
            
            # if self.config["OULU_NPU"]["enabled"]:
            #     self.process_oulu_npu()
                
        except Exception as e:
            print(f"\n❌ 预处理过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "="*70)
        print("✅ 预处理完成!")
        print("="*70)
        
        # 生成统计信息
        total_count = self.count_images()
        print(f"\n📊 处理结果统计:")
        print(f"总图片数: {total_count}")
        print(f"目标图片数: {self.config['target_total_images']}")
        print(f"完成度: {(total_count/self.config['target_total_images'])*100:.1f}%")
        
        print("\n📋 下一步操作:")
        print(f"1. 数据已保存在: {self.output_dir}")
        print("2. 复制或链接数据到SSDG代码目录")
        print("3. 运行SSDG的训练脚本")
        
        return True
    
    def count_images(self):
        """统计处理后的图片数量"""
        total = 0
        data_dir = os.path.join(self.output_dir, "data")
        
        if not os.path.exists(data_dir):
            return 0
        
        for root, dirs, files in os.walk(data_dir):
            total += len([f for f in files if f.lower().endswith('.png')])
        
        return total


def main():
    """主函数"""
    print("SSDG小样本预处理工具")
    print("-" * 50)
    print("使用MXNet MTCNN，兼容PyTorch 0.4 + CUDA 8.0")
    print("-" * 50)
    
    # 简单配置选择
    print("\n请选择预处理规模:")
    print("1. 极小样本 (~100张)")
    print("2. 小样本 (~13400张) [推荐]")
    print("3. 自定义")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        print("使用极小样本配置")
        # 可以在config中调整参数
        preprocessor = SSDGPreprocessor_MXNet()
    elif choice == "2":
        print("使用小样本配置 (~500张)")
        preprocessor = SSDGPreprocessor_MXNet()
    elif choice == "3":
        base_dir = input("请输入原始数据目录 (默认: E:\\SSDG_Dataset): ").strip()
        if not base_dir:
            base_dir = "E:\\SSDG_Dataset"
        preprocessor = SSDGPreprocessor_MXNet(base_dir=base_dir)
    else:
        print("无效选择，使用默认配置")
        preprocessor = SSDGPreprocessor_MXNet()
    
    # 运行预处理
    success = preprocessor.run_all()
    
    if success:
        print("\n🎉 预处理成功完成！")
        print("可以开始SSDG训练了！")
    else:
        print("\n❌ 预处理失败，请检查错误信息")

if __name__ == "__main__":
    main()