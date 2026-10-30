import sys
sys.path.append('../../')
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str, patch_shuffle_batch
from utils.evaluate import eval
from utils.get_loader import get_dataset,get_target_labeled_unlabeled_loaders
from models.DGFAS import DG_model, Discriminator
from loss.hard_triplet_loss import HardTripletLoss
from loss.AdLoss import Real_AdLoss, Fake_AdLoss
import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer
from loss.Adptive_Softmax import Adaptive_Softmax
from loss.Entropy import EntropyLoss
from loss.Center_alignment import Center_alignment
import torch.nn.functional as F
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim 
from datetime import datetime

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
#os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#device = 'cuda'
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")
def train():
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='default')
    args = parser.parse_args()
    log_dir = os.path.join(config.logs, 'tensorboard', args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)

    # load data数据加载，每个源域都有独立的 真实样本加载器 和 伪造样本加载器。
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    tgt_valid_dataloader = get_dataset(config.src1_data, config.src1_train_num_frames, 
                                       config.src2_data, config.src2_train_num_frames, 
                                       config.src3_data, config.src3_train_num_frames,
                                       config.tgt_data, config.tgt_test_num_frames, config.batch_size)
    # 新增：加载目标域有标签和无标签数据
    if config.use_semi_supervised:
        tgt_labeled_loader, tgt_unlabeled_loader = get_target_labeled_unlabeled_loaders(
            target_data=config.tgt_data,
            labeled_ratio=config.tgt_labeled_ratio,
            batch_size=config.batch_size,
            num_frames_labeled=config.tgt_train_num_frames,
            num_frames_unlabeled=config.tgt_unlabeled_num_frames
        )
        tgt_labeled_iter = iter(tgt_labeled_loader)
        tgt_unlabeled_iter = iter(tgt_unlabeled_loader)
        tgt_labeled_len = len(tgt_labeled_loader)
        tgt_unlabeled_len = len(tgt_unlabeled_loader)
    else:
        tgt_labeled_iter = tgt_unlabeled_iter = None
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    # #用CPU
    # device = torch.device('cpu')
    # print(f"Using device: {device}")

    #模型结构
    net = DG_model(config.model).to(device)# 特征提取器 + 分类器，输入图像，输出分类 logits 和中间特征向量。
    ad_net_real = Discriminator().to(device)# 域判别器（仅用于真实样本）二分类器（判断样本来自哪个源域）。这里只对真实样本做域判别。
    ad_net_fake = Discriminator().to(device)# 定义但未使用

    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    #定义损失函数
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).to(device),
        'adaptive_softmax': Adaptive_Softmax().to(device),
        'bce': nn.BCELoss().to(device)
    }
    # 获取 net 中除 domain_disc 以外的参数（domain_disc 单独添加）
    net_params = [p for n, p in net.named_parameters() if 'domain_disc' not in n]
    if config.use_semi_supervised:
        optimizer_dict = [
            {"params": net_params, "lr": config.init_lr},
            {"params": ad_net_real.parameters(), "lr": config.init_lr},
            {"params": net.domain_disc.parameters(), "lr": config.init_lr},
            # {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
            # {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
        ]
    else:
        optimizer_dict = [
            {"params": net_params, "lr": config.init_lr},
            {"params": ad_net_real.parameters(), "lr": config.init_lr},
        ]
    optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)

    max_iter = config.max_iter
    epoch = 1
    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).to(device)

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)

        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        #if (iter_num % src2_iter_per_epoch_fake == 0):
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)
        ######### data prepare #########
        src1_img_real, src1_label_real = next(src1_train_iter_real)#每次迭代一个batch的图像
        src1_img_real = src1_img_real.to(device)
        src1_label_real = src1_label_real.to(device)
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_label_real = next(src2_train_iter_real)
        src2_img_real = src2_img_real.to(device)
        src2_label_real = src2_label_real.to(device)
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_label_real = next(src3_train_iter_real)
        src3_img_real = src3_img_real.to(device)
        src3_label_real = src3_label_real.to(device)
        input3_real_shape = src3_img_real.shape[0]

        src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
        src1_img_fake = src1_img_fake.to(device)
        src1_label_fake = src1_label_fake.to(device)
        input1_fake_shape = src1_img_fake.shape[0]
        
        src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
        src2_img_fake = src2_img_fake.to(device)
        src2_label_fake = src2_label_fake.to(device)
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_label_fake = next(src3_train_iter_fake)
        src3_img_fake = src3_img_fake.to(device)
        src3_label_fake = src3_label_fake.to(device)
        input3_fake_shape = src3_img_fake.shape[0]
        input_src = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)

        if config.use_semi_supervised:
        # 获取目标域有标签 batch
            if iter_num % tgt_labeled_len == 0:
                tgt_labeled_iter = iter(tgt_labeled_loader)
            if iter_num % tgt_unlabeled_len == 0:
                tgt_unlabeled_iter = iter(tgt_unlabeled_loader)

        
            # 获取目标域无标签 batch
            input_labeled_tgt, label_labeled_tgt = next(tgt_labeled_iter)
            input_unlabeled_tgt, _ = next(tgt_unlabeled_iter)   # 标签占位，不使用
            input_labeled_tgt = input_labeled_tgt.to(device)
            label_labeled_tgt = label_labeled_tgt.to(device)
            input_unlabeled_tgt = input_unlabeled_tgt.to(device)
            batch_src_size = input_src.size(0)
            batch_labeled_tgt_size = input_labeled_tgt.size(0)
            batch_unlabeled_tgt_size = input_unlabeled_tgt.size(0)
            #内容破坏
            if config.use_content_destruction and config.use_semi_supervised and epoch < 10:
                input_unlabeled_tgt_shuffled = patch_shuffle_batch(input_unlabeled_tgt, patch_size=config.patch_size)
                # 原始特征不计算梯度（节省显存）
                with torch.no_grad():
                    _, feat_unlabeled_orig = net(input_unlabeled_tgt, config.norm_flag)
                _, feat_unlabeled_shuffled = net(input_unlabeled_tgt_shuffled, config.norm_flag)
                consistency_loss = F.mse_loss(feat_unlabeled_shuffled, feat_unlabeled_orig)
            input_data = torch.cat([input_src, input_labeled_tgt, input_unlabeled_tgt], dim=0)

        else:
                # 仅使用源域数据
            input_data = input_src
            batch_src_size = input_src.size(0)
            batch_labeled_tgt_size = 0
            batch_unlabeled_tgt_size = 0

        source_label = torch.cat([src1_label_real, src1_label_fake,
                                  src2_label_real, src2_label_fake,
                                  src3_label_real, src3_label_fake], dim=0)

        ######### forward #########
        classifier_label_out, feature = net(input_data, config.norm_flag)

        ######### single side adversarial learning #########
        input1_shape = input1_real_shape + input1_fake_shape
        input2_shape = input2_real_shape + input2_fake_shape
        feature_real_1 = feature.narrow(0, 0, input1_real_shape)#切片操作。这里取 feature 的第 0 维（样本维）上，从索引 0 开始，取 input1_real_shape 个样本 → 即 src1 真实样本的特征。
        feature_real_2 = feature.narrow(0, input1_shape, input2_real_shape)
        feature_real_3 = feature.narrow(0, input1_shape+input2_shape, input3_real_shape)
        feature_real = torch.cat([feature_real_1, feature_real_2, feature_real_3], dim=0)
        discriminator_out_real = ad_net_real(feature_real)

        ######### unbalanced triplet loss #########
        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).to(device)
        real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0).to(device)
        real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0).to(device)
        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1).to(device)
        fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(2).to(device)
        fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(3).to(device)
        source_domain_label = torch.cat([real_domain_label_1, fake_domain_label_1,
                                         real_domain_label_2, fake_domain_label_2,
                                         real_domain_label_3, fake_domain_label_3], dim=0).view(-1)
        #1三元组损失
        feature_src = feature.narrow(0, 0, batch_src_size)
        triplet = criterion["triplet"](feature_src, source_domain_label)

        ######### cross-entropy loss #########
        real_shape_list = []
        real_shape_list.append(input1_real_shape)
        real_shape_list.append(input2_real_shape)
        real_shape_list.append(input3_real_shape)
        #2单边对抗损失
        real_adloss = Real_AdLoss(discriminator_out_real, criterion["softmax"], real_shape_list)
        #3分类损失（源域）
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, batch_src_size), source_label)

        if config.use_semi_supervised:
            # ========== 新增半监督损失 ==========
            # 4. 有标签目标域分类损失（使用 Adaptive_Softmax）
            labeled_tgt_cls_loss = criterion["adaptive_softmax"](
                classifier_label_out.narrow(0, batch_src_size, batch_labeled_tgt_size),
                label_labeled_tgt
            )
            
            # 5. 无标签目标域熵损失
            if config.use_entropy:
                softmax_out = F.softmax(classifier_label_out.narrow(0, batch_src_size + batch_labeled_tgt_size, batch_unlabeled_tgt_size), dim=1)
                entropy_loss = EntropyLoss(softmax_out)
            else:
                entropy_loss = torch.tensor(0.0, device=device)
            
            # 6. 域对抗损失（源域 + 无标签目标域）
            # 取源域和无标签目标域的特征
            if config.use_domain_adaptation:
                feature_src_unlabeled = torch.cat([
                    feature.narrow(0, 0, batch_src_size),
                    feature.narrow(0, batch_src_size + batch_labeled_tgt_size, batch_unlabeled_tgt_size)
                ], dim=0)
                domain_label = torch.cat([
                    torch.ones(batch_src_size, 1),
                    torch.zeros(batch_unlabeled_tgt_size, 1)
                ], dim=0).to(device)
                domain_out = net.domain_disc(feature_src_unlabeled)   # 输出 [0,1] 概率
                transfer_loss = criterion["bce"](domain_out, domain_label)
            else:
                transfer_loss = torch.tensor(0.0, device=device)
            
            # 7. 中心对齐损失（源域 vs 有标签目标域）
            if config.use_center_alignment:
                alignment_loss = Center_alignment(
                    feature.narrow(0, 0, batch_src_size), source_label,
                    feature.narrow(0, batch_src_size, batch_labeled_tgt_size), label_labeled_tgt
                )
            else:
                alignment_loss = torch.tensor(0.0, device=device)
            ######### backward #########
            #total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss 
            # ========== 总损失 ==========
            total_loss = (cls_loss +
                        config.lambda_triplet * triplet +
                        config.lambda_adreal * real_adloss +
                        config.lambda_labeled_tgt * labeled_tgt_cls_loss +
                        config.lambda_entropy * entropy_loss +
                        config.lambda_transfer * transfer_loss +
                        config.lambda_center * alignment_loss)
            if config.use_content_destruction and config.use_semi_supervised and epoch < 10:
                total_loss += config.lambda_consistency * consistency_loss
                writer.add_scalar('train/consistency_loss', consistency_loss.item(), iter_num)
        else:
            total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(cls_loss.item())
        #acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        if config.use_semi_supervised:
            labeled_pred = classifier_label_out.narrow(0, 0, batch_src_size + batch_labeled_tgt_size)
            labeled_target = torch.cat([source_label, label_labeled_tgt], dim=0)
            acc = accuracy(labeled_pred, labeled_target, topk=(1,))
        else:
            acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        # 记录训练指标（每 log_interval 次迭代）
        if iter_num % config.log_interval == 0:
            writer.add_scalar('train/total_loss', total_loss.item(), iter_num)
            writer.add_scalar('train/cls_loss', cls_loss.item(), iter_num)
            writer.add_scalar('train/triplet_loss', triplet.item(), iter_num)
            writer.add_scalar('train/real_adloss', real_adloss.item(), iter_num)
            if config.use_semi_supervised:
                writer.add_scalar('train/labeled_tgt_cls_loss', labeled_tgt_cls_loss.item(), iter_num)
                writer.add_scalar('train/entropy_loss', entropy_loss.item(), iter_num)
                writer.add_scalar('train/transfer_loss', transfer_loss.item(), iter_num)
                writer.add_scalar('train/alignment_loss', alignment_loss.item(), iter_num)
            writer.add_scalar('train/top1_acc', classifer_top1.avg, iter_num)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)

        #验证时记录
        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            valid_args = eval(tgt_valid_dataloader, net, config.norm_flag)
            writer.add_scalar('valid/loss', valid_args[0], iter_num)
            writer.add_scalar('valid/top1_acc', valid_args[1], iter_num)
            writer.add_scalar('valid/EER', valid_args[2], iter_num)
            writer.add_scalar('valid/HTER', valid_args[3], iter_num)
            writer.add_scalar('valid/AUC', valid_args[4], iter_num)
            writer.add_scalar('valid/ACC_threshold', valid_args[6], iter_num)
        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
            valid_args = eval(tgt_valid_dataloader, net, config.norm_flag)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                param_lr_tmp[0]))
            log.write('\n')
            time.sleep(0.01)
        
    writer.close()
if __name__ == '__main__':
    train()




















