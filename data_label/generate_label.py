'''生成的JSON文件保存在当前工作目录下的子文件夹中'''
import os
import json
import sys
import glob

# change your data path
data_dir = r'E:\SSDG_Processed_Data_MXNet\data'

def msu_process():
 # 1. 读取训练/测试列表，去除换行符并转换为整数（忽略前导零）
    with open(r'E:\SSDG_Dataset\MSU_MFSD\train_sub_list.txt', 'r') as f:
        train_list = [int(line.strip()) for line in f if line.strip()]
    with open(r'E:\SSDG_Dataset\MSU_MFSD\test_sub_list.txt', 'r') as f:
        test_list = [int(line.strip()) for line in f if line.strip()]

    print("Train IDs:", train_list)
    print("Test IDs:", test_list)
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    #dataset_path = data_dir + 'msu_256/'
    #path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    #path_list.sort()
    # 修改：使用 os.path.join 和正确的路径模式
    dataset_path = os.path.join(data_dir, 'msu_256')
    path_list = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    print(f"Found {len(path_list)} files in {dataset_path}")
    for path in path_list:
        normalized_path = path.replace('\\', '/')

        # 标签：路径中包含 '/real/' 则为 1（真实），否则 0（伪造）
        label = 1 if '/real/' in normalized_path else 0

        # 提取视频编号：路径结构为 .../msu_256/real/real_001/frame_000.png
        # 倒数第二级目录名如 'real_001'，取 '_' 后面的部分并转为整数
        dir_name = normalized_path.split('/')[-2]          # e.g., 'real_001'
        video_num_str = dir_name.split('_')[-1]            # e.g., '001'
        video_num = int(video_num_str)                     # 转换为整数 1

        dict_item = {
            'photo_path': path,
            'photo_label': label
        }

        # 划分训练/测试集
        if video_num in train_list:
            train_final_json.append(dict_item)
        else:
            test_final_json.append(dict_item)

        all_final_json.append(dict_item)
        if label == 1:
            real_final_json.append(dict_item)
        else:
            fake_final_json.append(dict_item)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def casia_process():
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    #dataset_path = data_dir + 'casia_256/'
    #path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    #path_list.sort()
    dataset_path = os.path.join(data_dir, 'casia_256')
    path_list = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    print(f"Found {len(path_list)} files in {dataset_path}")
    for i in range(len(path_list)):
        #flag = path_list[i].split('/')[-2]
        normalized_path = path_list[i].replace('\\', '/')
        flag = normalized_path.split('/')[-2]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        #flag = path_list[i].find('/train_release/')
        flag = normalized_path.find('/train_release/')
        if (flag != -1):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './replay/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    #dataset_path = data_dir + 'replay_256/'
    #path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    #path_list.sort()
    dataset_path = os.path.join(data_dir, 'replay_256')
    path_list = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    print(f"Found {len(path_list)} files in {dataset_path}")
    for i in range(len(path_list)):
        #flag = path_list[i].find('/real/')
        normalized_path = path_list[i].replace('\\', '/')
        flag = normalized_path.find('/real/')
        if (flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        #if (path_list[i].find('/replayattack-train/') != -1):
        #    train_final_json.append(dict)
        #elif(path_list[i].find('/replayattack-devel/') != -1):
        #    valid_final_json.append(dict)
        #else:
        #    test_final_json.append(dict)
        #if(path_list[i].find('/replayattack-devel/') != -1):
        #   continue
        if (normalized_path.find('/replayattack-train/') != -1):
            train_final_json.append(dict)
        elif(normalized_path.find('/replayattack-devel/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(normalized_path.find('/replayattack-devel/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def oulu_process():
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = './oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    #dataset_path = data_dir + 'oulu_256/'
    #path_list = glob.glob(dataset_path + '**/*.png', recursive=True)
    #path_list.sort()
    dataset_path = os.path.join(data_dir, 'oulu_256')
    path_list = glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)
    
    print(f"Found {len(path_list)} files in {dataset_path}")
    for i in range(len(path_list)):
        #flag = int(path_list[i].split('/')[-2].split('_')[-1])
        normalized_path = path_list[i].replace('\\', '/')
        flag = int(normalized_path.split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        #if (path_list[i].find('/Train_files/') != -1):
        #    train_final_json.append(dict)
        #elif(path_list[i].find('/Dev_files/') != -1):
        #    valid_final_json.append(dict)
        #else:
        #    test_final_json.append(dict)
        #if(path_list[i].find('/Dev_files/') != -1):
        #    continue
        if (normalized_path.find('/Train_files/') != -1):
            train_final_json.append(dict)
        elif(normalized_path.find('/Dev_files/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(normalized_path.find('/Dev_files/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


if __name__=="__main__":
    msu_process()
    casia_process()
    replay_process()
    oulu_process()