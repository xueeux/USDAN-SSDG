import os
import random
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from utils.dataset import YunpeiDataset
from utils.utils import sample_frames
import pandas as pd

def get_dataset(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
                tgt_data, tgt_test_num_frames, batch_size):
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = sample_frames(flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data)
    src1_train_data_real = sample_frames(flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data)
    print('Source Data: ', src2_data)
    src2_train_data_fake = sample_frames(flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data)
    src2_train_data_real = sample_frames(flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data)
    print('Source Data: ', src3_data)
    src3_train_data_fake = sample_frames(flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data)
    src3_train_data_real = sample_frames(flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames(flag=2, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

    src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
                                            batch_size=batch_size, shuffle=True)
    src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
                                            batch_size=batch_size, shuffle=True)
    tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)
    return src1_train_dataloader_fake, src1_train_dataloader_real, \
           src2_train_dataloader_fake, src2_train_dataloader_real, \
           src3_train_dataloader_fake, src3_train_dataloader_real, \
           tgt_dataloader


def get_target_labeled_unlabeled_loaders(target_data, labeled_ratio, batch_size, num_frames_labeled, num_frames_unlabeled):
    label_path = os.path.join('..', '..', 'data_label', target_data, 'train_label.json')
    with open(label_path, 'r') as f:
        all_data = json.load(f)

    # 使用 JSON 中的 photo_belong_to_video_ID 分组（若不存在则回退）
    video_dict = {}
    for item in all_data:
        vid = item.get('photo_belong_to_video_ID')
        if vid is None:
            # 兼容旧数据：从路径倒数第二级提取
            path = item['photo_path']
            vid = path.replace('\\', '/').split('/')[-2]
        if vid not in video_dict:
            video_dict[vid] = {'paths': [], 'label': item['photo_label']}
        video_dict[vid]['paths'].append(item['photo_path'])

    video_ids = list(video_dict.keys())
    random.shuffle(video_ids)
    num_labeled = int(len(video_ids) * labeled_ratio)
    labeled_videos = video_ids[:num_labeled]
    unlabeled_videos = video_ids[num_labeled:]

    labeled_data = []
    for vid in labeled_videos:
        paths = video_dict[vid]['paths']
        sampled_paths = random.sample(paths, min(num_frames_labeled, len(paths)))
        for p in sampled_paths:
            labeled_data.append({
                'photo_path': p,
                'photo_label': video_dict[vid]['label'],
                'photo_belong_to_video_ID': vid
            })

    unlabeled_data = []
    for vid in unlabeled_videos:
        paths = video_dict[vid]['paths']
        sampled_paths = random.sample(paths, min(num_frames_unlabeled, len(paths)))
        for p in sampled_paths:
            unlabeled_data.append({
                'photo_path': p,
                'photo_label': -1,
                'photo_belong_to_video_ID': vid
            })

    labeled_dataset = YunpeiDataset(pd.DataFrame(labeled_data), train=True)
    unlabeled_dataset = YunpeiDataset(pd.DataFrame(unlabeled_data), train=True)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return labeled_loader, unlabeled_loader







