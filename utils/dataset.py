import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import os
class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        # 添加路径检查
        print(f"数据集大小: {len(self.photo_path)}")
        for i, path in enumerate(self.photo_path[:3]):  # 检查前3个路径
            fixed_path = path.replace('\\', '/')
            if not os.path.exists(fixed_path):
                print(f"警告: 路径不存在 - {fixed_path}")
            else:
                print(f"路径正常: {fixed_path}")
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            # 修复路径：将Windows路径分隔符统一为正斜杠
            img_path = img_path.replace('\\', '/')
            # 检查路径是否存在，如果不存在则尝试相对路径
            if not os.path.exists(img_path):
                # 尝试相对于项目根目录的路径
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                # 可能的数据存储位置
                possible_paths = [
                    os.path.join(base_dir, '..', '..', 'SSDG_Processed_Data_MXNet', 'data'),
                    os.path.join(base_dir, '..', 'data'),
                    os.path.join(base_dir, 'data'),
                    'E:/SSDG_Processed_Data_MXNet/data'  # 绝对路径
                ]
                
                for base in possible_paths:
                    # 提取相对路径部分
                    path_parts = img_path.split('/')
                    # 查找数据集名称
                    for part in path_parts:
                        if 'casia' in part or 'replay' in part or 'msu' in part or 'oulu' in part:
                            dataset_name = part
                            break
                    
                    # 构建新路径
                    new_path = os.path.join(base, dataset_name, *path_parts[-2:])
                    if os.path.exists(new_path):
                        img_path = new_path.replace('\\', '/')
                        break
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            # 同样修复路径
            img_path = img_path.replace('\\', '/')
            
            # 检查路径是否存在
            if not os.path.exists(img_path):
                # 同上添加路径检查逻辑
                pass
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label, videoID
