import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import re
def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
    i = 0
    for param_group in optimizer.param_groups:
        init_lr = init_param_lr[i]
        i += 1
        if(epoch <= lr_epoch_1):
            param_group['lr'] = init_lr * 0.1 ** 0
        elif(epoch <= lr_epoch_2):
            param_group['lr'] = init_lr * 0.1 ** 1
        else:
            param_group['lr'] = init_lr * 0.1 ** 2

import matplotlib.pyplot as plt
def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)

# def sample_frames(flag, num_frames, dataset_name):
#     '''
#         from every video (frames) to sample num_frames to test
#         return: the choosen frames' path and label
#     '''
#     # The process is a litter cumbersome, you can change to your way for convenience
#     root_path = '../../data_label/' + dataset_name
#     if(flag == 0): # select the fake images
#         label_path = root_path + '/fake_label.json'
#         save_label_path = root_path + '/choose_fake_label.json'
#     elif(flag == 1): # select the real images
#         label_path = root_path + '/real_label.json'
#         save_label_path = root_path + '/choose_real_label.json'
#     else: # select all the real and fake images
#         label_path = root_path + '/all_label.json'
#         save_label_path = root_path + '/choose_all_label.json'

#     all_label_json = json.load(open(label_path, 'r'))
#     f_sample = open(save_label_path, 'w')
#     length = len(all_label_json)
#     # three componets: frame_prefix, frame_num, png
#     # 添加路径统一处理
#     def normalize_path(path):
#         """统一路径分隔符，并确保路径格式正确"""
#         return path.replace('\\', '/')
    
#     if length == 0:
#         print(f"警告: {label_path} 为空")
#         f_sample.close()
#         return pd.DataFrame(columns=['photo_path', 'photo_label', 'photo_belong_to_video_ID'])
#     #saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
#     final_json = []
#     video_number = 0

#     # 第一个条目的路径前缀
#     first_path = normalize_path(all_label_json[0]['photo_path'])
#     saved_frame_prefix = '/'.join(first_path.split('/')[:-1])
    
#     # 改为存储 (frame_number, filename, full_path, label) 元组
#     single_video_items = []
#     single_video_label = 0

#     import re
    
#     def extract_frame_number(filename):
#         """从文件名中提取帧号"""
#         try:
#             # 处理数字文件名
#             return int(filename)
#         except ValueError:
#             # 处理 s10v1f0 格式
#             numbers = re.findall(r'\d+', filename)
#             if numbers:
#                 return int(numbers[-1])  # 取最后一个数字
#             # 处理 frame_000 格式
#             elif filename.startswith('frame_'):
#                 return int(filename.split('_')[1])
#             else:
#                 return 0
            
#     #single_video_frame_list = []
#     #single_video_frame_num = 0
#     #single_video_label = 0
#     for i in range(length):
#         #photo_path = all_label_json[i]['photo_path']
#         #photo_label = all_label_json[i]['photo_label']
#         raw_path = all_label_json[i]['photo_path']
#         photo_label = all_label_json[i]['photo_label']
        
#         # 统一路径格式
#         photo_path = normalize_path(raw_path)

#         frame_prefix = '/'.join(photo_path.split('/')[:-1])

#         # 提取文件名（不带扩展名）
#         filename = os.path.basename(photo_path).split('.')[0]
        
#         # 提取帧号
#         photo_frame = extract_frame_number(filename)

#         # the last frame
#         if (i == length - 1):
#             #photo_frame = int(photo_path.split('/')[-1].split('.')[0])
#             #single_video_frame_list.append(photo_frame)
#             #single_video_frame_num += 1

#             single_video_items.append((photo_frame, filename, photo_path, photo_label))
#             single_video_label = photo_label
#         # a new video, so process the saved one
#         if (frame_prefix != saved_frame_prefix or i == length - 1):
#             single_video_items.sort(key=lambda x: x[0])
            
#             available_frames = len(single_video_items)
#             if available_frames > 0:
#                 # 计算采样间隔
#                 actual_num_frames = min(num_frames, available_frames)
#                 if actual_num_frames > 0:
#                     frame_interval = max(1, available_frames // actual_num_frames)
                    
#                     # 确定起始位置
#                     start_idx = min(6, available_frames - actual_num_frames * frame_interval)
#                     start_idx = max(0, start_idx)
                    
#                     for j in range(actual_num_frames):
#                         dict = {}
#                         idx = start_idx + j * frame_interval
#                         if idx < available_frames:
#                             _, chosen_filename, chosen_path, chosen_label = single_video_items[idx]
#                             dict['photo_path'] = chosen_path  # 使用完整路径，而不是重新构造
#                             dict['photo_label'] = chosen_label
#                             dict['photo_belong_to_video_ID'] = video_number
#                             final_json.append(dict)
            
#             video_number += 1
#             saved_frame_prefix = frame_prefix
#             single_video_items.clear()
#             # [1, 2, 3, 4,.....]
#         #     single_video_frame_list.sort()
#         #     frame_interval = math.floor(single_video_frame_num / num_frames)
#         #     for j in range(num_frames):
#         #         dict = {}
#         #         #起始帧调整
#         #         #dict['photo_path'] = saved_frame_prefix + '/' + str(
#         #         #    single_video_frame_list[6 + j * frame_interval]) + '.png'
#         #         start_idx = min(6, len(single_video_frame_list) - num_frames)
#         #         start_idx = max(0, start_idx)  # 确保不小于0
#         #         dict['photo_path'] = saved_frame_prefix + '/' + str(
#         #             single_video_frame_list[start_idx + j * frame_interval]) + '.png'
#         #         dict['photo_label'] = single_video_label
#         #         dict['photo_belong_to_video_ID'] = video_number
#         #         final_json.append(dict)
#         #     video_number += 1
#         #     saved_frame_prefix = frame_prefix
#         #     single_video_frame_list.clear()
#         #     single_video_frame_num = 0
#         # # get every frame information
#         # #photo_frame = int(photo_path.split('/')[-1].split('.')[0])
#         # ## 改为：
        
#         # filename = os.path.basename(photo_path).split('.')[0]
#         # photo_frame = extract_frame_number(filename)
#         # single_video_frame_list.append(photo_frame)
#         # #single_video_frame_num += 1

            
#         # 添加到当前视频帧列表
#         single_video_items.append((photo_frame, filename, photo_path, photo_label))
#         single_video_label = photo_label
#     if(flag == 0):
#         print("Total video number(fake): ", video_number, dataset_name)
#     elif(flag == 1):
#         print("Total video number(real): ", video_number, dataset_name)
#     else:
#         print("Total video number(target): ", video_number, dataset_name)
#     json.dump(final_json, f_sample, indent=4)
#     f_sample.close()

#     f_json = open(save_label_path)
#     sample_data_pd = pd.read_json(f_json)
#     return sample_data_pd
def sample_frames(flag, num_frames, dataset_name):
    '''
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path and label
    '''
    # The process is a litter cumbersome, you can change to your way for convenience
    root_path = '../../data_label/' + dataset_name
    if(flag == 0): # select the fake images
        label_path = root_path + '/fake_label.json'
        save_label_path = root_path + '/choose_fake_label.json'
    elif(flag == 1): # select the real images
        label_path = root_path + '/real_label.json'
        save_label_path = root_path + '/choose_real_label.json'
    elif flag == 2:   # 验证集优先使用 test_label.json
        test_path = root_path + '/test_label.json'
        if os.path.exists(test_path):
            label_path = test_path
            save_label_path = root_path + '/choose_test_label.json'
        else:
            test_path = root_path + '/test_label.json'
            if os.path.exists(test_path):
                label_path = test_path
                save_label_path = root_path + '/choose_test_label.json'
            else:
                label_path = root_path + '/all_label.json'
                save_label_path = root_path + '/choose_all_label.json'
    elif flag == 3:   # 测试集使用 valid_label.json
        label_path = root_path + '/valid_label.json'
        save_label_path = root_path + '/choose_valid_label.json'
    else:
        label_path = root_path + '/all_label.json'
        save_label_path = root_path + '/choose_all_label.json'

    all_label_json = json.load(open(label_path, 'r'))
    f_sample = open(save_label_path, 'w')
    length = len(all_label_json)
    
    # 检查数据是否为空
    if length == 0:
        print(f"警告: {label_path} 为空")
        f_sample.close()
        return pd.DataFrame(columns=['photo_path', 'photo_label', 'photo_belong_to_video_ID'])
    
# 检查是否已有 photo_belong_to_video_ID
    has_video_id = all('photo_belong_to_video_ID' in item for item in all_label_json)

    final_json = []
    video_number = 0

    if has_video_id:
        # 按已有的 video_id 分组
        from collections import defaultdict
        video_dict = defaultdict(list)
        for item in all_label_json:
            video_dict[item['photo_belong_to_video_ID']].append(item)

        for items in video_dict.values():
            # 按文件名排序，保证顺序一致
            items.sort(key=lambda x: os.path.basename(x['photo_path']))
            available = len(items)
            sample_count = min(num_frames, available)
            if sample_count > 0:
                step = max(1, available // sample_count)
                for j in range(sample_count):
                    idx = j * step
                    if idx < available:
                        item = items[idx]
                        final_json.append({
                            'photo_path': item['photo_path'],
                            'photo_label': item['photo_label'],
                            'photo_belong_to_video_ID': video_number
                        })
            video_number += 1
    
    else:
        # 存储当前视频的所有帧 (frame_num, filename, full_path, label)
        current_video_frames = []
        current_video_prefix = None
        
        # 定义文件名提取函数
        def extract_frame_number(filename):
            """从文件名中提取帧号，支持所有四个数据集的格式"""
            # 移除扩展名
            if '.' in filename:
                filename = filename.split('.')[0]
            
            # 1. 尝试直接转换数字 (OULU: 1, 2, 3...)
            try:
                return int(filename)
            except ValueError:
                pass
            
            # 2. 处理 CASIA 格式: s10v1f0, bs10v1f100
            # 提取最后一个数字作为帧号
            numbers = re.findall(r'\d+', filename)
            if numbers:
                # 对于 s10v1f0，取最后一个数字 0
                return int(numbers[-1])
            
            # 3. 处理 frame_000, frame_001 格式 (MSU, Replay)
            if filename.startswith('frame_'):
                try:
                    return int(filename.split('_')[1])
                except:
                    pass
            
            # 4. 其他格式，返回0
            return 0
        
        for i in range(length):
            raw_path = all_label_json[i]['photo_path']
            photo_label = all_label_json[i]['photo_label']
            
            # 统一路径格式（Windows和Linux兼容）
            photo_path = raw_path.replace('\\', '/')
            
            # 提取目录前缀
            frame_prefix = '/'.join(photo_path.split('/')[:-1])
            
            # 提取文件名
            filename = os.path.basename(photo_path)
            
            # 提取帧号
            frame_num = extract_frame_number(filename)
            
            # 如果是第一个帧，初始化
            if current_video_prefix is None:
                current_video_prefix = frame_prefix
            
            # 检查是否是新视频
            if frame_prefix != current_video_prefix:
                # 处理上一个视频
                if current_video_frames:
                    # 按帧号排序
                    current_video_frames.sort(key=lambda x: x[0])
                    
                    # 确定要采样的帧数（不能超过可用帧数）
                    available_frames = len(current_video_frames)
                    frames_to_sample = min(num_frames, available_frames)
                    
                    if frames_to_sample > 0:
                        # 计算采样间隔
                        step = max(1, available_frames // frames_to_sample)
                        
                        # 采样并保存（从小样本数据实际出发，从第0帧开始采样）
                        for j in range(frames_to_sample):
                            idx = j * step
                            if idx < available_frames:
                                _, chosen_filename, chosen_path, chosen_label = current_video_frames[idx]
                                dict_item = {
                                    'photo_path': chosen_path,
                                    'photo_label': chosen_label,
                                    'photo_belong_to_video_ID': video_number
                                }
                                final_json.append(dict_item)
                    
                    video_number += 1
                
                # 重置为新的视频
                current_video_prefix = frame_prefix
                current_video_frames = []
            
            # 添加到当前视频列表
            current_video_frames.append((frame_num, filename, photo_path, photo_label))
        
        # 处理最后一个视频
        if current_video_frames:
            # 按帧号排序
            current_video_frames.sort(key=lambda x: x[0])
            
            available_frames = len(current_video_frames)
            frames_to_sample = min(num_frames, available_frames)
            
            if frames_to_sample > 0:
                # 计算采样间隔
                step = max(1, available_frames // frames_to_sample)
                
                # 采样并保存
                for j in range(frames_to_sample):
                    idx = j * step
                    if idx < available_frames:
                        _, chosen_filename, chosen_path, chosen_label = current_video_frames[idx]
                        dict_item = {
                            'photo_path': chosen_path,
                            'photo_label': chosen_label,
                            'photo_belong_to_video_ID': video_number
                        }
                        final_json.append(dict_item)
            
            video_number += 1
    
    if(flag == 0):
        print("Total video number(fake): ", video_number, dataset_name)
    elif(flag == 1):
        print("Total video number(real): ", video_number, dataset_name)
    else:
        print("Total video number(target): ", video_number, dataset_name)
    
    json.dump(final_json, f_sample, indent=4)
    f_sample.close()

    f_json = open(save_label_path)
    sample_data_pd = pd.read_json(f_json)
    return sample_data_pd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "valid_arg": valid_args,
            "best_model_EER": best_model_HTER,
            "best_model_ACER": best_model_ACER,
            "best_model_ACC": best_model_ACC,
            "threshold": threshold
        }
    filepath = checkpoint_path + filename
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

def patch_shuffle_batch(images, patch_size=16):
    """
    images: torch.Tensor of shape (B, C, H, W)
    patch_size: int, 必须能整除 H 和 W
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    n_h, n_w = H // patch_size, W // patch_size
    # 切块 (B, C, n_h, patch, n_w, patch) -> (B, n_h, n_w, C, patch, patch)
    patches = images.view(B, C, n_h, patch_size, n_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(B, n_h * n_w, C, patch_size, patch_size)
    # 随机打乱每个 batch 的块顺序
    shuffled = []
    for b in range(B):
        perm = torch.randperm(n_h * n_w, device=images.device)
        shuffled.append(patches[b, perm])
    shuffled = torch.stack(shuffled, dim=0)
    # 重组回图像
    shuffled = shuffled.view(B, n_h, n_w, C, patch_size, patch_size)
    shuffled = shuffled.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled = shuffled.view(B, C, H, W)
    return shuffled