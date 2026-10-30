import sys
sys.path.append('../../')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import json
import random
import os

from config import config
from models.DGFAS import DG_model

def set_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(666)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. 加载模型 ----------
def load_model(model_path, model_name='resnet18'):
    net = DG_model(model_name).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    return net

baseline_ckpt = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\oulu_checkpoint\resnet18\best_model\model_best_0.206_14.pth.tar'   # 修改为实际路径
#full_ckpt = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\oulu_checkpoint\resnet18\best_model\model_best_0.064_88.pth.tar'
full_ckpt = r'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\experiment\I_C_M_to_O\oulu_checkpoint\resnet18\best_model\model_best_0.08_100.pth.tar'
# ---------- 2. 获取源域训练样本（已采样后的图片）----------
def get_training_samples_from_source(src_name, root_path='../../data_label/'):
    real_json = os.path.join(root_path, src_name, 'choose_real_label.json')
    fake_json = os.path.join(root_path, src_name, 'choose_fake_label.json')
    with open(real_json, 'r') as f:
        real_data = json.load(f)
    with open(fake_json, 'r') as f:
        fake_data = json.load(f)
    paths = [item['photo_path'] for item in real_data] + [item['photo_path'] for item in fake_data]
    labels = [item['photo_label'] for item in real_data] + [item['photo_label'] for item in fake_data]
    return paths, labels

source_configs = [('casia', 0), ('replay', 1), ('msu', 2)]
source_paths = []
source_labels = []
source_domains = []
for src_name, dom_id in source_configs:
    paths, labels = get_training_samples_from_source(src_name)
    source_paths.extend(paths)
    source_labels.extend(labels)
    source_domains.extend([dom_id] * len(paths))

print(f"源域总样本数: {len(source_paths)} (真实: {sum(source_labels)}, 伪造: {len(source_labels)-sum(source_labels)})")

# ---------- 3. 获取目标域样本（测试集，用于可视化对比）----------
with open('../../data_label/oulu/all_label.json', 'r') as f:
    test_data = json.load(f)
tgt_real = [item for item in test_data if item['photo_label'] == 1]
tgt_fake = [item for item in test_data if item['photo_label'] == 0]
# 各取200个（可调整）
num_samples = 1000
tgt_real_choose = random.sample(tgt_real, min(num_samples, len(tgt_real)))
tgt_fake_choose = random.sample(tgt_fake, min(num_samples, len(tgt_fake)))
tgt_paths = [item['photo_path'] for item in tgt_real_choose + tgt_fake_choose]
tgt_labels = [1]*len(tgt_real_choose) + [0]*len(tgt_fake_choose)
tgt_domains = [3] * len(tgt_paths)   # 3 代表目标域

# ---------- 4. 合并所有样本 ----------
all_paths = source_paths + tgt_paths
all_labels = source_labels + tgt_labels
all_domains = source_domains + tgt_domains
assert len(all_paths) == len(all_labels) == len(all_domains), "长度不一致"

# ---------- 5. 特征提取 ----------
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(model, paths, batch_size=32):
    features = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p).convert('RGB')
                img = val_transform(img).unsqueeze(0)
                batch_imgs.append(img)
            batch_tensor = torch.cat(batch_imgs, dim=0).to(device)
            _, feat = model(batch_tensor, config.norm_flag)
            features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)

print("Loading Baseline model...")
model_baseline = load_model(baseline_ckpt, config.model)
print("Extracting features with Baseline...")
feat_all_baseline = extract_features(model_baseline, all_paths)

print("Loading Full model...")
model_full = load_model(full_ckpt, config.model)
print("Extracting features with Full model...")
feat_all_full = extract_features(model_full, all_paths)

# ---------- 6. t-SNE 降维 ----------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
print("Running t-SNE on Baseline...")
feat_tsne_baseline = tsne.fit_transform(feat_all_baseline)
print("Running t-SNE on Full model...")
feat_tsne_full = tsne.fit_transform(feat_all_full)

# ---------- 7. 绘图函数（包含源域真实/伪造 + 目标域真实/伪造）----------
def plot_tsne_full(feat_tsne, labels, domains, title, save_path):
    plt.figure(figsize=(12, 8))
    source_fake_colors = {0: 'blue', 1: 'green', 2: '#ffea00'}
    source_shapes = {0: 'o', 1: 's', 2: 'D'}
    target_shape = '^'
    target_real_color = '#ff7f0e'
    target_fake_color = '#00ced1'

    # 源域伪造
    for dom in [0,1,2]:
        idx = (domains == dom) & (labels == 0)
        if idx.any():
            color = source_fake_colors[dom]
            marker = source_shapes[dom]
            src_name = {0:'CASIA',1:'Replay',2:'MSU'}[dom]
            plt.scatter(feat_tsne[idx,0], feat_tsne[idx,1],
                        c=color, marker=marker, s=30, alpha=0.6, label=f'{src_name} Fake')
    # 源域真实（统一红色）
    for dom in [0,1,2]:
        idx = (domains == dom) & (labels == 1)
        if idx.any():
            marker = source_shapes[dom]
            src_name = {0:'CASIA',1:'Replay',2:'MSU'}[dom]
            plt.scatter(feat_tsne[idx,0], feat_tsne[idx,1],
                        c='red', marker=marker, s=30, alpha=0.6, label=f'{src_name} Real')
    # 目标域真实
    idx_tgt_real = (domains == 3) & (labels == 1)
    if idx_tgt_real.any():
        plt.scatter(feat_tsne[idx_tgt_real,0], feat_tsne[idx_tgt_real,1],
                    c=target_real_color, marker=target_shape, s=30, alpha=0.6, label='Target Real')
    # 目标域伪造
    idx_tgt_fake = (domains == 3) & (labels == 0)
    if idx_tgt_fake.any():
        plt.scatter(feat_tsne[idx_tgt_fake,0], feat_tsne[idx_tgt_fake,1],
                    c=target_fake_color, marker=target_shape, s=30, alpha=0.6, label='Target Fake')

    plt.legend(loc='best', fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

#plot_tsne_full(feat_tsne_baseline, np.array(all_labels), np.array(all_domains),
#               'Baseline: Source Training Features + Target Real/Fake',
#               'tsne_baseline.png')
# plot_tsne_full(feat_tsne_full, np.array(all_labels), np.array(all_domains),
#                'Full Model: Source Training Features + Target Real/Fake',
#                'tsne_full.png')
plot_tsne_full(feat_tsne_full, np.array(all_labels), np.array(all_domains),
               'Full Model: Source Training Features + Target Real/Fake',
               'tsne_full_shuffled.png')
print("Done.")