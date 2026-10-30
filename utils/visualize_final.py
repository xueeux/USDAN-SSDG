import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F

def plot_confusion_matrix(labels, preds, save_path, class_names=['Fake', 'Real']):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(labels, probs, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, 'b-', label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pred_distribution(labels, probs, save_path):
    """预测概率分布直方图（按真实标签分组）"""
    plt.figure(figsize=(8,5))
    plt.hist(probs[labels==1], bins=50, alpha=0.6, label='Real', color='green')
    plt.hist(probs[labels==0], bins=50, alpha=0.6, label='Fake', color='red')
    plt.xlabel('Predicted probability of Real')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_samples(error_items, save_path, num_samples=16):
    """
    展示错误样本的图像网格
    error_items: list of (image_tensor, true_label, pred_label, prob)
    image_tensor: 形状 (C, H, W) 或 (1,C,H,W)
    """
    if not error_items:
        print("No error samples to display.")
        return
    # 取前 num_samples 个
    error_items = error_items[:num_samples]
    images = [item[0] for item in error_items]
    # 确保每个图像是 (C,H,W) 并归一化到 [0,1] 以便显示
    # 注意：原始图像可能已经归一化，需要反归一化（如果需要）
    # 这里假设 images 是经过反归一化或原始像素值（0-1）的 tensor
    grid = make_grid(images, nrow=4, padding=2, normalize=True)
    plt.figure(figsize=(12,12))
    plt.imshow(grid.permute(1,2,0).cpu().numpy())
    plt.axis('off')
    # 添加文本标签（可选，较复杂，可简单保存）
    plt.title('Error Samples (Red: Fake, Green: Real)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne_features(features, labels, preds, save_path):
    """t-SNE特征可视化，按真实标签和预测正确/错误着色"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    feat_2d = tsne.fit_transform(features)
    plt.figure(figsize=(10,8))
    # 按真实标签+预测正确性着色
    for i, (label, pred) in enumerate(zip(labels, preds)):
        if label == pred:
            color = 'green' if label == 1 else 'blue'
            marker = 'o'
        else:
            color = 'red'
            marker = 'x'
        plt.scatter(feat_2d[i,0], feat_2d[i,1], c=color, marker=marker, alpha=0.6)
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Correct Real', markerfacecolor='green', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='Correct Fake', markerfacecolor='blue', markersize=8),
                       Line2D([0], [0], marker='x', color='w', label='Wrong', markerfacecolor='red', markersize=8)]
    plt.legend(handles=legend_elements)
    plt.title('t-SNE of Features (Color: correctness)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def collect_test_info(test_dataloader, model, device, norm_flag, threshold):
    """
    运行测试集，收集预测概率、标签、特征、错误样本信息
    返回: 
        prob_list: 每个视频的平均概率
        label_list: 每个视频的真实标签
        pred_list: 每个视频的预测标签（基于threshold）
        features_list: 每个视频的平均特征（可选，如果内存大可以收集所有帧特征然后平均）
        error_items: 每个错误样本的 (image_tensor, true_label, pred_label, prob)
        video_ids: 视频ID列表
    """
    model.eval()
    prob_dict = {}
    label_dict = {}
    feature_dict = {}
    # 存储每个视频的第一帧图像（用于错误样本展示）
    first_frame_dict = {}
    with torch.no_grad():
        for batch in test_dataloader:
            # 假设 dataloader 返回 (input, target, videoID) 且 train=False
            input, target, videoID = batch
            input = input.to(device)
            target = target.to(device)
            cls_out, feature = model(input, norm_flag)
            prob = F.softmax(cls_out, dim=1)[:, 1].cpu().numpy()
            label = target.cpu().numpy()
            videoID = videoID.cpu().numpy()
            feature_np = feature.cpu().numpy()
            for i in range(len(prob)):
                vid = videoID[i]
                if vid not in prob_dict:
                    prob_dict[vid] = []
                    label_dict[vid] = []
                    feature_dict[vid] = []
                    first_frame_dict[vid] = input[i].cpu()  # 保存第一帧图像
                prob_dict[vid].append(prob[i])
                label_dict[vid].append(label[i])
                feature_dict[vid].append(feature_np[i])
    # 对每个视频求平均
    prob_list = []
    label_list = []
    features_list = []
    error_items = []
    for vid in prob_dict.keys():
        avg_prob = np.mean(prob_dict[vid])
        avg_label = np.mean(label_dict[vid])  # 所有帧标签一致
        avg_feature = np.mean(feature_dict[vid], axis=0)
        prob_list.append(avg_prob)
        label_list.append(avg_label)
        features_list.append(avg_feature)
        pred = 1 if avg_prob >= threshold else 0
        if pred != avg_label:
            # 错误样本：保存该视频的第一帧图像
            img_tensor = first_frame_dict[vid]  # 形状 (C,H,W)
            error_items.append((img_tensor, avg_label, pred, avg_prob))
    prob_list = np.array(prob_list)
    label_list = np.array(label_list)
    features_list = np.array(features_list)
    pred_list = (prob_list >= threshold).astype(int)
    return prob_list, label_list, pred_list, features_list, error_items

def visualize_final_results(test_dataloader, model, device, norm_flag, threshold, save_dir):
    """主函数：生成所有可视化图表"""
    os.makedirs(save_dir, exist_ok=True)
    print("Collecting test information...")
    probs, labels, preds, features, error_items = collect_test_info(
        test_dataloader, model, device, norm_flag, threshold
    )
    print(f"Test set size: {len(probs)} videos")
    
    # 1. 混淆矩阵
    plot_confusion_matrix(labels, preds, os.path.join(save_dir, 'confusion_matrix.png'))
    print("Saved confusion matrix")
    
    # 2. ROC曲线
    plot_roc_curve(labels, probs, os.path.join(save_dir, 'roc_curve.png'))
    print("Saved ROC curve")
    
    # 3. 预测概率分布
    plot_pred_distribution(labels, probs, os.path.join(save_dir, 'pred_distribution.png'))
    print("Saved prediction distribution")
    
    # 4. 错误样本图像网格
    if error_items:
        plot_error_samples(error_items, os.path.join(save_dir, 'error_samples.png'), num_samples=16)
        print(f"Saved error samples ({len(error_items)} errors)")
    else:
        print("No error samples found.")
    
    # 5. t-SNE特征可视化（如果特征维度不太高且样本数适中）
    if len(features) > 1:
        # 可选：降采样以避免t-SNE过慢（比如最多2000个点）
        n_samples = min(len(features), 2000)
        indices = np.random.choice(len(features), n_samples, replace=False)
        plot_tsne_features(features[indices], labels[indices], preds[indices], 
                           os.path.join(save_dir, 'tsne_features.png'))
        print("Saved t-SNE visualization")
    else:
        print("Not enough samples for t-SNE")
    
    print(f"All visualizations saved to {save_dir}")