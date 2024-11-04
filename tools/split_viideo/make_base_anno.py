import pandas as pd
import os
import argparse
import random
from collections import defaultdict
import cv2
from datetime import datetime, time
import json
from tqdm import tqdm
import shutil

# label_dict = {
#     '1': 'Replay',
#     '2': 'Three',
#     '3': 'MidRangeShot',
#     '4': 'BreakthroughLayup',
#     '5': 'Dunk',
#     '6': 'FreeThrows',
#     '7': 'BlockShot'
# }
label_dict = {
    '2': 'Three',
    '3': 'MidRangeShot',
    '4': 'BreakthroughLayup',
    '5': 'Dunk',
    '7': 'BlockShot'
}


def read_files(train_val_test_file_path):
    train_file_list = []
    val_file_list = []
    test_file_list = []
    with open(train_val_test_file_path, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            assert len(line) == 2
            if line[1].strip() == 'train':
                train_file_list.append(line[0].strip())
            elif line[1].strip() == 'test':
                test_file_list.append(line[0].strip())
            elif line[1].strip() == 'val':
                val_file_list.append(line[0].strip())
            else:
                print(f"train/val/test format error: {line[1].strip()}")
                continue
    return train_file_list, val_file_list, test_file_list

# 检查xlsx 中 起始时间和结束时间的格式是否正确
def check_format(time_str):

    if not isinstance(time_str, time):
        return False
    h = time_str.hour
    m = time_str.minute
    s = time_str.second
    if not (h < 10 and m < 60 and s < 60):
        return False
    return True

# 将时间格式转换为 视频的秒数
def std_time_sec(time_str):
    h = time_str.hour
    m = time_str.minute
    s = time_str.second
    return h * 3600 + m * 60 + s

# 利用原始的 打标xlsx 文件，制作初始的anno 文件
def make_single_video_anno_dict(file, subset, video_path, label_path):
    video_dict = {}
    video_dict['subset'] = subset
    video_file_path = os.path.join(video_path, file+'.mp4')
    label_file_path = os.path.join(label_path, file+'.xlsx')
    if not os.path.exists(video_file_path) or not os.path.exists(label_file_path):
        raise FileNotFoundError(f"file not exist: {video_file_path}, {label_file_path}")
    # 查看mp4 视频文件 的时长 和 总帧数
    # 打开视频文件
    video = cv2.VideoCapture(video_file_path)
    # 获取视频的帧速率
    fps = video.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # 计算视频的时长（秒）
    duration = total_frames / fps
    video_dict['duration'] = duration
    video_dict['frame'] = int(total_frames)
    # 关闭视频文件
    video.release()
    # 读取xlsx文件
    df = pd.read_excel(label_file_path)
    if not df.columns.values.tolist() == ['开始时间', '结束时间', '标签']:
        print(f"label file format error: {label_file_path}")
        import pdb;pdb.set_trace()
    assert df.columns.values.tolist() == ['开始时间', '结束时间', '标签']
    # 按行遍历df
    anno_lst = []
    for index, row in df.iterrows():
        action_dict = {}
        start_time = row['开始时间']
        end_time = row['结束时间']
        if check_format(start_time) and check_format(end_time):
            # 将时间格式转换为秒
            start_time = std_time_sec(start_time)
            end_time = std_time_sec(end_time)
            label = row['标签']
            if str(label) in label_dict.keys():
                action_dict['segment'] = [start_time, end_time]
                try:
                    action_dict['label'] = label_dict[str(label)]
                except Exception as e:
                    import pdb;pdb.set_trace()
                anno_lst.append(action_dict)
        else:
            print(f"time format error: {start_time}, {end_time}, in file {file}, line {index + 2}")
            continue
    video_dict['annotations'] = anno_lst
    return video_dict


def convert_grounding_annotation(train_file, val_file, test_file, video_path, label_path):
    anno_dict = {
        'database': {},
        'taxonomy': {},
        'version': {}
    }
    for file in train_file:
        video_dict = make_single_video_anno_dict(file, 'training', video_path, label_path)
        video_name = file.split('/')[-1].split('.')[0]
        anno_dict['database'][video_name] = video_dict
    for file in val_file:
        video_dict = make_single_video_anno_dict(file, 'validation', video_path, label_path)
        video_name = file.split('/')[-1].split('.')[0]
        anno_dict['database'][video_name] = video_dict
    for file in test_file:
        video_dict = make_single_video_anno_dict(file, 'test', video_path, label_path)
        video_name = file.split('/')[-1].split('.')[0]
        anno_dict['database'][video_name] = video_dict
    return anno_dict

# 根据anno_dict ,来自定义的切分视频。这里是将视频切分成120秒。
def split_video_file(anno_dict):
    video_split_dict = {}
    for k,v in tqdm(anno_dict['database'].items()):
        # if v['subset']=='test':
        #     video_split_dict[k] = v
        #     continue
        anno_fino = v['annotations']
        sorted_anno = sorted(anno_fino, key=lambda x: x['segment'][0])
        start = int((sorted_anno[0]['segment'][0] + sorted_anno[0]['segment'][1]+1) /2)
        end = sorted_anno[0]['segment'][1]+1
        ret = []
        anno = []
        split_point_lst = []
        count = 0
        if end - start > 120:
            raise ValueError('video duration is too long', start, end, k)
        for idx in range(len(sorted_anno) - 1):
            if idx == 0:

                start = max(0, sorted_anno[0]['segment'][0] - int((sorted_anno[0]['segment'][1] + 1 - sorted_anno[0]['segment'][0]) /2))
                if sorted_anno[idx]['segment'][0] - start < 1:
                    import pdb;pdb.set_trace()
                anno = [{'segment': [sorted_anno[idx]['segment'][0] - start, sorted_anno[idx]['segment'][1] + 1 - start], 'label': sorted_anno[idx]['label']}]
            else:
                if sorted_anno[idx]['segment'][1]-start >= 120:
                    # 找到结束 ，上个动作结束点 + 动作时间的一半 或者 两个动作的中点的小值
                    end = min(int((sorted_anno[idx]['segment'][0] + sorted_anno[idx-1]['segment'][1] + 1) / 2),
                                sorted_anno[idx-1]['segment'][1] + 1 + int((sorted_anno[idx-1]['segment'][1] + 1 - sorted_anno[idx-1]['segment'][0]) / 2),
                              sorted_anno[idx-1]['segment'][1] + 1 + 10)
                    # anno.append({'segments': [sorted_anno[idx]['segment'][0] - start, sorted_anno[idx]['segment'][1] - start], 'label': sorted_anno[idx]['label']})
                    ret.append({'annotations': anno, 'start': start, 'end': end, 'subset': v['subset']})
                    # ret[k+'-' +str(count)] = {'annotations': anno, 'start': start, 'end': end, 'subset': v['subset'], 'duration': (end - start) * 25, 'frame': end - start}
                    # split_point_lst.append({'start': start, 'end': end})
                    # count += 1
                    anno = []
                    # 找到开始点， 两个动作中点或者当前动作开始点 -  当前动作时间的一半
                    start = max(int((sorted_anno[idx]['segment'][0] + sorted_anno[idx-1]['segment'][1]) / 2),
                                sorted_anno[idx]['segment'][0] - int((sorted_anno[idx]['segment'][1] + 1 - sorted_anno[idx]['segment'][0]) / 2))
                    if sorted_anno[idx]['segment'][0] - start < 1:
                        import pdb;pdb.set_trace()
                    # 切割视频
                    anno.append({'segment': [sorted_anno[idx]['segment'][0] - start, sorted_anno[idx]['segment'][1] + 1 - start], 'label': sorted_anno[idx]['label']})
                    # import pdb;pdb.set_trace()
                else:
                    if sorted_anno[idx]['segment'][0] - start < 0:
                        import pdb;pdb.set_trace()
                    anno.append({'segment': [sorted_anno[idx]['segment'][0] - start, sorted_anno[idx]['segment'][1] + 1 - start], 'label': sorted_anno[idx]['label']})

        video_split_dict[k] = ret
    return video_split_dict

# def seconds_to_hhmmss(seconds):
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     seconds = seconds % 60
#     return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# # 按照 video_split_dict 对原视频进行切分。并生成最终的anno 文件。
# def video_split_and_make_anno(video_split_dict, video_path):
#     anno_dict = {
#         'database': {},
#         'taxonomy': {},
#         'version': {}
#     }
#     print('gegin to split video')
#     for k,v in tqdm(video_split_dict.items()):
#         video_full_path = os.path.join(video_path, k+'.mp4')
#         dest_path = os.path.join('/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s', k+'.mp4')
#         # if isinstance(v, dict):
#         #     import pdb;pdb.set_trace()
#         #     if v['subset'] == "test":
#         #         anno_dict['database'][k] = v
#         #         shutil.copy(video_full_path, dest_path)
#         #     continue
#         # video_path = os.path.join(video_path, k+'.mp4')
#         for info in v:
#             start = info['start']
#             end = info['end']
#             name = k + '_' + str(start) + '_' + str(end) + '.mp4'
#             output_path = os.path.join('/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s', name)
#             # 判断下文件是否已经存在
#             if os.path.exists(output_path):
#                 print("{} exist, skip ".format(name))
#                 continue
#             if end - start < 20:
#                 continue
#             cmd = f"ffmpeg -ss {seconds_to_hhmmss(start)} -t {seconds_to_hhmmss(end-start)} -i {video_full_path} -c copy {output_path} >/data/ysp_public_data/sport-editing/basketball_annotation/video_ffmpeg.log 2>&1"
#             os.system(cmd)
#             # 查看切割后的视频时长，帧数
#             video = cv2.VideoCapture(output_path)
#             fps = video.get(cv2.CAP_PROP_FPS)
#             total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
#             if fps == 0 or total_frames == 0:
#                 import pdb;pdb.set_trace()
#                 continue
#             duration = total_frames / fps
#             anno_info = {
#                 'annotations': info['annotations'],
#                 'duration': duration,
#                 'frame': total_frames,
#                 'subset': info['subset']
#             }
#             print(name, duration, total_frames,seconds_to_hhmmss(start), seconds_to_hhmmss(end-start), start, end)
#             anno_dict['database'][k + '_' + str(start) + '_' + str(end)] = anno_info
#     return anno_dict


if __name__ == '__main__':
    '''
    python 
    '''
    parser = argparse.ArgumentParser(description="grounding annotation conversion")
    parser.add_argument(
        "-opath", "--output_path", default='/data/ysp_public_data/sport-editing/basketball_annotation', help="select ontology type"
    )
    parser.add_argument(
        "-ofile", "--base_file", default='base_anno_1030.json', help="select ontology type"
    )
    # parser.add_argument(
    #     "-afile", "--anno_file", default='debug_final_anno.json'
    # )
    parser.add_argument(
        "-vpath", "--video_path", default='/data/ysp_public_data/sport-editing/basketball_video'
    )
    parser.add_argument(
        "-lpath", "--label_path", default="/data/ysp_public_data/sport-editing/basketball_labeled"
    )
    args = parser.parse_args()
    # 一些路径
    video_path = args.video_path
    label_path = args.label_path
    output_path = args.output_path
    output_default_anno_file = args.base_file
    # final_anno_file = os.path.join(output_path, args.anno_file)
    train_val_test_file_path = '/data/ysp_public_data/sport-editing/split_file.txt'


    train_file_list, val_file_list, test_file_list = read_files(train_val_test_file_path)
    anno_dict =convert_grounding_annotation(train_file_list, val_file_list, test_file_list,video_path, label_path)
    # 将初始的 anno 文件保存起来
    output_file_path = os.path.join(output_path, output_default_anno_file)
    json_str = json.dumps(anno_dict, indent=4)
    with open(output_file_path, 'w') as f:
        f.write(json_str)
    
    
    # split_video_dict = split_video_file(anno_dict)
    # print('finish new anno')
    # video_split_dict = video_split_and_make_anno(split_video_dict, video_path)
    # with open(final_anno_file, 'w') as f:
    #     json.dump(video_split_dict, f, indent=4)
    # print()

