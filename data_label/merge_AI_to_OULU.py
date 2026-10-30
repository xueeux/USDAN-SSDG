import os
import json

def merge_json_files(oulu_dir, ai_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 需要合并的文件列表
    for fname in ['all_label.json', 'real_label.json', 'fake_label.json', 'train_label.json', 'test_label.json']:
        oulu_path = os.path.join(oulu_dir, fname)
        ai_path = os.path.join(ai_dir, fname)
        
        if not os.path.exists(oulu_path):
            print(f"Missing {oulu_path}, skip")
            continue
        
        with open(oulu_path, 'r') as f:
            oulu_data = json.load(f)
        
        # 为 OULU 数据补上 photo_belong_to_video_ID（如果不存在）
        for idx, item in enumerate(oulu_data):
            if 'photo_belong_to_video_ID' not in item:
                item['photo_belong_to_video_ID'] = idx  # 按顺序分配视频ID
        
        if os.path.exists(ai_path):
            with open(ai_path, 'r') as f:
                ai_data = json.load(f)
            
            # 找到OULU中最大的视频ID
            max_vid = max([item['photo_belong_to_video_ID'] for item in oulu_data]) if oulu_data else -1
            # 为AI数据重新分配视频ID（避免冲突）
            for item in ai_data:
                max_vid += 1
                item['photo_belong_to_video_ID'] = max_vid
            
            merged = oulu_data + ai_data
        else:
            merged = oulu_data
        
        with open(os.path.join(output_dir, fname), 'w') as f:
            json.dump(merged, f, indent=4)
        print(f"{fname}: OULU({len(oulu_data)}) + AI({len(ai_data) if os.path.exists(ai_path) else 0}) = {len(merged)}")

if __name__ == '__main__':
    oulu_dir = 'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\data_label\oulu'          # 原OULU标注目录
    ai_dir   = 'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\data_label\AI_dataset_II'    # AI标注目录
    output_dir = 'D:\subject\semester4\Graduation_project\SSDG_CVPR2020_master\data_label\oulu_merged_II' # 合并后存放目录
    merge_json_files(oulu_dir, ai_dir, output_dir)