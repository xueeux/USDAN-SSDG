import sys
sys.path.append('../../')

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from config import config
from utils.utils import sample_frames
from utils.dataset import YunpeiDataset
from models.DGFAS import DG_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predictions(model_path, test_dataloader):
    """加载模型，返回视频级概率列表和标签列表"""
    net = DG_model(config.model).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    
    prob_dict = {}
    label_dict = {}
    with torch.no_grad():
        for input, target, videoID in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            cls_out, _ = net(input, config.norm_flag)
            prob = F.softmax(cls_out, dim=1).cpu().numpy()[:, 1]
            label = target.cpu().numpy()
            videoID = videoID.cpu().numpy()
            for i in range(len(prob)):
                vid = videoID[i]
                prob_dict.setdefault(vid, []).append(prob[i])
                label_dict.setdefault(vid, []).append(label[i])
    
    # 视频级平均
    prob_list = [np.mean(prob_dict[vid]) for vid in prob_dict]
    label_list = [label_dict[vid][0] for vid in label_dict]  # 同一视频标签一致
    return np.array(prob_list), np.array(label_list)

def main():
    # 六个模型的权重路径和对应的显示名称
    model_configs = [
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.206_14.pth.tar", "Baseline"),
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.096_18.pth.tar", "L"),
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.108_23.pth.tar", "L+E"),
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.076_100.pth.tar", "L+T"),
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.104_17.pth.tar", "L+C"),
        ("./oulu_checkpoint/resnet18/best_model/model_best_0.064_88.pth.tar", "Full (USDAN-SSDG)")
    ]
    
    # 加载测试数据（与训练时使用的测试集一致）
    # flag=3 对应 test_label.json，具体请根据你的 sample_frames 定义调整
    test_data = sample_frames(flag=3, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=1, shuffle=False)
    
    # 存储所有预测结果
    all_probs = []
    all_labels = None
    model_names = []
    
    for model_path, name in model_configs:
        print(f"Processing {name}...")
        probs, labels = get_predictions(model_path, test_dataloader)
        all_probs.append(probs)
        if all_labels is None:
            all_labels = labels  # 所有模型的标签应相同
        model_names.append(name)
    
    # 绘图
    plt.figure(figsize=(9, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for i, name in enumerate(model_names):
        fpr, tpr, _ = roc_curve(all_labels, all_probs[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i], lw=2,
                 label=f'{name} (AUC={roc_auc:.3f})')
    
    # 随机分类器参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves for Different Configurations', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_all_models.png', dpi=300)
    plt.show()
    print("ROC curves saved as roc_curves_all_models.png")

if __name__ == '__main__':
    main()