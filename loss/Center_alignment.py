import torch
import torch.nn as nn

def Center_alignment(source_data, source_label, labeled_target_data, target_label):
    source_size = source_data.size(0)
    target_size = labeled_target_data.size(0)

    # 收集源域真实样本
    source_real_list = []
    for i in range(source_size):
        if source_label[i].item() == 1:   # 使用 .item() 获取标量
            source_real_list.append(source_data[i].view(1, -1))

    # 收集有标签目标域真实样本
    target_real_list = []
    for i in range(target_size):
        if target_label[i].item() == 1:
            target_real_list.append(labeled_target_data[i].view(1, -1))

    # 如果任一方缺失真实样本，返回零损失
    if len(source_real_list) == 0 or len(target_real_list) == 0:
        return torch.tensor(0.0, device=source_data.device)

    # 拼接并计算中心
    source_real = torch.cat(source_real_list, dim=0)
    target_real = torch.cat(target_real_list, dim=0)

    source_center = source_real.mean(dim=0).view(1, -1)
    target_center = target_real.mean(dim=0).view(1, -1)

    pdist = nn.PairwiseDistance(p=2)
    distance = pdist(source_center, target_center)
    return distance.mean()