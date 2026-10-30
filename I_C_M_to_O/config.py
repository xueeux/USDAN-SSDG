class DefaultConfigs(object):
    seed = 888
    # SGD

    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    #init_lr = 0.01
    init_lr = 0.01# 降低初始学习率
    #lr_epoch_1 = 0
    lr_epoch_1 = 10    # 原为10，提前降低学习率
    lr_epoch_2 = 20    # 原为150，
    # model
    pretrained = True
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = "3"
    batch_size = 10
    norm_flag = True
    max_iter = 1000 
    log_interval = 10   # 每 10 次迭代记录一次 TensorBoard
    lambda_triplet = 2
    lambda_adreal = 0.1

    # test model name
    # tgt_best_model_name = "model_best_0.125_20.pth.tar"
    # tgt_best_model_name = "model_best_0.2_14.pth.tar"
    # tgt_best_model_name = "model_best_0.1_20.pth.tar"
    #tgt_best_model_name = "model_best_0.05_51.pth.tar"  #SSDG_with_USDAN
    #tgt_best_model_name = "model_best_0.206_14.pth.tar"   #baseline
    # tgt_best_model_name = "model_best_0.084_100.pth.tar"   #labeled
    # tgt_best_model_name = "model_best_0.1_100.pth.tar"  #labeled_entropy
    # tgt_best_model_name = "model_best_0.07_90.pth.tar"    #labeled_transfer
    # tgt_best_model_name = "model_best_0.084_100.pth.tar"  #labeled_center 
    #tgt_best_model_name = "model_best_0.05675_82.pth.tar"  #AI

    #不带训练集验证（seed=888）
    #tgt_best_model_name = "model_best_0.064_88.pth.tar"  #SSDG+USDAN_all
    #tgt_best_model_name = "model_best_0.206_14.pth.tar"   #baseline
    # tgt_best_model_name = "model_best_0.096_18.pth.tar"   #labeled
    # tgt_best_model_name = "model_best_0.108_23.pth.tar"  #labeled_entropy
    # tgt_best_model_name = "model_best_0.076_100.pth.tar"    #labeled_transfer
    # tgt_best_model_name = "model_best_0.104_17.pth.tar"  #labeled_center 
    #tgt_best_model_name = "model_best_0.07447_47.pth.tar"  #AI
    #tgt_best_model_name = "model_best_0.06014_85.pth.tar"  #AI_II
    # source data information
    #tgt_best_model_name = "model_best_0.08168_93.pth.tar"#AI_I
    #tgt_best_model_name = "model_best_0.13334_87.pth.tar"#AI_I_shuffle
    #tgt_best_model_name = "model_best_0.08687_98.pth.tar"#SSDG+USDAN_all_AI_I_shuffle
    #tgt_best_model_name = "model_best_0.08_100.pth.tar" #SSDG+USDAN_all+shuffle
    #tgt_best_model_name = "model_best_0.05296_94.pth.tar"#SSDG+USDAN_all_AI_II_shuffle
    src1_data = 'casia'
    src1_train_num_frames = 1
    src2_data = 'replay'
    src2_train_num_frames = 5

    src3_data = 'msu'
    src3_train_num_frames = 10
    # target data information
    tgt_data = 'oulu'
    #tgt_data1 = 'oulu_merged_II'
    tgt_test_num_frames = 2

    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

    # 半监督相关配置

    tgt_labeled_ratio = 0.2                # 从目标域训练集中抽取多少比例作为有标签数据 (0~1)
    tgt_train_num_frames = 3           # 每个视频采帧数
    tgt_unlabeled_num_frames = 3           # 无标签视频采样帧数
    lambda_labeled_tgt = 0.5               # 有标签目标域分类损失权重
    lambda_entropy = 0.1               # 无标签目标域熵损失权重
    use_semi_supervised = True   # 改为 False 即可关闭半监督
    lambda_transfer = 1.0                  # 源域 vs 无标签目标域对抗损失权重
    use_entropy = True                # 是否使用熵最小化
    use_domain_adaptation = True     # 是否使用源-目标域对抗
    lambda_center = 0.05                    # 中心对齐损失权重
    use_center_alignment = True
    # 内容破坏（因果偏差减轻）
    use_content_destruction = True   # 是否启用
    patch_size = 4                  # 打乱块大小
    lambda_consistency = 0.001         # 一致性损失权重
config = DefaultConfigs()
