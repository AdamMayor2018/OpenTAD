import pandas as pd
import os
import argparse
import random
from collections import defaultdict
import cv2
from datetime import datetime, time
import json
import numpy as np
from tqdm import tqdm

# video_path = '/data/ysp_public_data/sport-editing/basketball_videos'
# label_path = '/data/ysp_public_data/sport-editing/basketball_labeled'
# train_val_test_file_path = '/data/ysp_public_data/sport-editing/split_file.txt'

import os
import json
import subprocess

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration, frame_count


def split_videos(anno_path, video_dir, output_dir, label_list, train_val_test,extend_time=12):
    # 读取动作定位字典
    with open(os.path.join(anno_path, 'base_anno_1030.json'), 'r') as f:
        action_dict = json.load(f)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    new_action_dict = {"database": {}}

    # 遍历每个视频
    for video_name, video_info in action_dict['database'].items():
        print("处理视频: ", video_name)
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            continue
        if video_info['subset'] not in train_val_test:
            continue
        for i in range(3):
            # 遍历视频中的每个标注
            for idx, anno in tqdm(enumerate(video_info['annotations']), total=len(video_info['annotations'])):
                label = anno['label']
                # extend_time 将extend_time 设置为 从1到5之中的一个随机值
                # if video_info['subset'] == 'training':
                #     start_extend_time = random.randint(5, 17)
                #     end_extend_time = 17-start_extend_time
                # else:
                #     start_extend_time = random.randint(1, 6)
                #     end_extend_time = 7-start_extend_time
                # 检查标签是否在label_list中
                if label in label_list:
                    start_time, end_time = anno['segment']
                    if video_info['subset'] == 'training':
                        start_extend_time = random.randint(5, 17)
                        end_extend_time = 17-start_extend_time
                    else:
                        action_time = end_time - start_time
                        if action_time > 10:
                            continue
                        start_extend_time = random.randint(0, 10-action_time)
                        end_extend_time = 11-start_extend_time-action_time
                    
                    # 将时间段前后各延长3秒，
                    # 默认：对于3分、2分、扣篮、盖帽等动作，时间间隔一定会大于3秒。
                    # 默认2：由于有开头和结尾，默认start_time和end_time 不会取到0 或者 duration。
                    start_time = max(0, start_time - start_extend_time)
                    end_time = min(video_info['duration']-1, end_time + end_extend_time)
                    
                    # 构建输出文件名
                    output_filename = f"{video_name}-{idx+1}-{start_time}-{end_time}-{i}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    cmd = f"ffmpeg -ss {str(start_time)} -t {str(end_time-start_time)} -i {video_path} {output_path} >/data/ysp_public_data/sport-editing/basketball_annotation/video_ffmpeg.log 2>&1"
                    try:
                        # print(f"成功裁剪并保存视频: {output_filename}")
                        os.system(cmd)
                        # 获取新视频的信息
                        new_duration, new_frame_count = get_video_info(output_path)
                        if new_duration - end_extend_time <= start_extend_time:
                            continue
                        
                        # 更新新的动作定位字典
                        new_action_dict["database"][output_filename.split('.')[0]] = {
                            "subset": video_info["subset"],
                            "duration": new_duration,
                            "frame": new_frame_count,
                            "annotations": [{
                                "segment": [start_extend_time, new_duration - end_extend_time],  # 假设原始动作在新视频中的位置
                                "label": label
                            }]
                        }
                    except Exception as e:
                        
                        print(f"裁剪视频时出错: {output_filename}")
                        print(f"错误信息: {e}")
                        import pdb;pdb.set_trace()
    
    # 保存新的动作定位字典
    new_action_dict_path = os.path.join(anno_path, "short_anno_1030.json")
    with open(new_action_dict_path, 'w') as f:
        con = json.dumps(new_action_dict, indent=4)
        f.write(con)
        # json.dump(new_action_dict, f, indent=4)
# 使用示例
anno_path = '/data/ysp_public_data/sport-editing/basketball_annotation/'
video_dir = '/data/ysp_public_data/sport-editing/basketball_video'
output_dir = '/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s_1030'
label_list = ['Three', 'MidRangeShot', 'BreakthroughLayup', 'Dunk', 'BlockShot']



def new_anno(path, subset):
    old_path = os.path.join(path, "short_anno_1030.json")
    add_path = os.path.join(path, "add_short_anno_1030.json")
    new_path = os.path.join(path, "new_short_anno_1030.json")
    with open(old_path, 'r') as f:
        anno_dict = json.load(f)
    anno_dict = anno_dict['database']
    new_anno_dict = {
        "database": {},
    }

    for video_name, video_info in anno_dict.items():
        if video_info['subset'] == subset:
            new_anno_dict['database'][video_name] = video_info

    with open(add_path, 'r') as f:
        add_dict = json.load(f)
        add_dict = add_dict['database']
    new_anno_dict['database'].update(add_dict)    
    with open(new_path, 'w') as f:
        con = json.dumps(new_anno_dict, indent=4)
        f.write(con)

def duplicate_videos(directory, copies=3):
    import shutil
    # 获取目录下所有的文件
    for filename in tqdm(os.listdir(directory), total=len(os.listdir(directory))):
        # 检查文件是否是 mp4 格式
        if filename.endswith(".mp4"):
            base_name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
            for i in range(0, copies):
                # 构造新的文件名
                new_filename = f"{base_name}_{i}{ext}"
                # 复制文件
                shutil.copyfile(os.path.join(directory, filename), os.path.join(directory, new_filename))
                # print(f"复制 {filename} 为 {new_filename}")
    
split_videos(anno_path, video_dir, output_dir, label_list, ['training', 'validation', 'test'])
# new_anno(anno_path, 'training')
# duplicate_videos(output_dir)