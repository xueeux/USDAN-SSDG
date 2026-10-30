import sys
sys.path.append('../../')

import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config import config
from models.DGFAS import DG_model
from utils.utils import sample_frames
from utils.dataset import YunpeiDataset

def load_model(model_path, device):
    net = DG_model(config.model).to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    net.load_state_dict(checkpoint["state_dict"])
    threshold = checkpoint.get("threshold", 0.5)
    net.eval()
    return net, threshold

def preprocess_image(img_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor, img

def predict(model, img_tensor, device, threshold):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        cls_out, _ = model(img_tensor, config.norm_flag)
        prob = F.softmax(cls_out, dim=1)[0, 1].item()
    pred = 1 if prob >= threshold else 0
    return prob, pred

def draw_annotation_on_ax(ax, img, prob, pred, true_label, threshold):
    ax.imshow(img)
    ax.axis('off')
    is_correct = (pred == true_label)
    border_color = 'green' if is_correct else 'red'
    # 绘制边框
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), img.width, img.height, linewidth=3, edgecolor=border_color, facecolor='none')
    ax.add_patch(rect)
    # 可选：固定坐标范围
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    # 添加文本
    text = f"True: {'R' if true_label==1 else 'F'}\nPred: {'R' if pred==1 else 'F'} ({prob:.2f})\nThr: {threshold:.2f}"
    ax.text(5, 5, text, bbox=dict(facecolor='black', alpha=0.7, pad=2),
            color='white', fontsize=8, verticalalignment='top')

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model_path = os.path.join(config.best_model_path, config.tgt_best_model_name)
    print(f"Loading model from {model_path}")
    net, threshold = load_model(model_path, device)
    net = net.to(device)
    
    # 加载测试数据集
    test_data = sample_frames(flag=4, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)
    test_dataset = YunpeiDataset(test_data, train=False)
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:12]
    
    # 准备网格
    cols, rows = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        img_path = test_dataset.photo_path[idx]
        true_label = test_dataset.photo_label[idx]
        img_tensor, original_img = preprocess_image(img_path)
        prob, pred = predict(net, img_tensor, device, threshold)
        draw_annotation_on_ax(axes[i], original_img, prob, pred, true_label, threshold)
        axes[i].set_title(f"Sample {i+1}", fontsize=10)
    
    # 隐藏多余的子图（如果少于12个）
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    output_path = "./sample_predictions/grid_4x3_AI_II.png"
    os.makedirs("./sample_predictions", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grid image saved to {output_path}")

if __name__ == '__main__':
    main()