import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from pprint import pprint

import ffmpeg
from tqdm import tqdm


def load_action_data(json_file):
    """
    加载动作数据从JSON文件中。

    参数:
        json_file (str): JSON文件路径。

    返回:
        dict: 动作结果数据。
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["results"]


def filter_action_data(action_data, conf=0.1, min_duration=3.0):
    for video, segments in action_data.items():
        action_data[video] = [
            segment
            for segment in segments
            if segment["score"] > conf
            and (segment["segment"][1] - segment["segment"][0]) >= min_duration
        ]
    return action_data


def merge_segments(segments, iou_threshold=0.5):
    # 按照分数降序排序
    segments = sorted(segments, key=lambda x: x["score"], reverse=True)
    merged = []
    while segments:
        current = segments.pop(0)
        merged.append(current)
        segments = [
            s
            for s in segments
            if compute_iou(current["segment"], s["segment"]) < iou_threshold
        ]
    return merged


def compute_iou(seg1, seg2):
    start1, end1 = seg1
    start2, end2 = seg2
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union != 0 else 0


def apply_nms(action_data, iou_threshold=0.5):
    for video, segments in action_data.items():
        action_data[video] = merge_segments(segments, iou_threshold)
    return action_data


def select_segments_by_time(videos, desired_actions, action_priorities, max_duration):
    """
    根据动作优先级选择视频片段。

    参数:
        videos (dict): 视频及其对应的动作数据。
        desired_actions (list): 所需的动作列表。
        action_priorities (dict): 动作的优先级。
        max_duration (float): 最长时长（秒）。

    返回:
        list: 选择的片段列表，包括视频名和动作信息。
    """
    selected_segments = []
    total_duration = 0
    action_counts = defaultdict(int)
    selected_segments_set = set()

    # 根据优先级排序动作
    sorted_actions = sorted(action_priorities.items(), key=lambda x: x[1], reverse=True)

    # 计算每个动作的目标时长
    total_priority = sum(action_priorities.values())
    target_durations = {
        action: (priority / total_priority) * max_duration
        for action, priority in action_priorities.items()
    }

    # 优先选择高优先级的动作片段
    for action, _ in sorted_actions:
        for action_info in videos[action]:
            duration = action_info["segment"][1] - action_info["segment"][0]
            if (
                total_duration + duration <= max_duration
                and action_counts[action] + duration <= target_durations[action]
            ):
                selected_segments.append(action_info)
                total_duration += duration
                action_counts[action] += 1
                selected_segments_set.add(action)
                action_info["is_used"] = True
                if total_duration >= max_duration:
                    break
            if total_duration >= max_duration:
                break
        if total_duration >= max_duration:
            break

    for action in desired_actions:
        if action not in selected_segments_set:
            for action_info in videos[action]:
                if not action_info["is_used"]:
                    selected_segments.append(action_info)
                    action_counts[action] += 1
                    break
            if action in selected_segments_set:
                break

    return selected_segments


def select_segments_by_action(videos, desired_actions, action_priorities, max_num):
    """
    根据动作优先级选择视频片段。

    参数:
        videos (dict): 视频及其对应的动作数据。
        desired_actions (list): 所需的动作列表。
        action_priorities (dict): 动作的优先级。
        max_duration (float): 最长时长（秒）。

    返回:
        list: 选择的片段列表，包括视频名和动作信息。
    """
    selected_segments = []

    # 根据优先级排序动作
    sorted_actions = sorted(action_priorities.items(), key=lambda x: x[1], reverse=True)

    # 计算每个动作的目标时长
    total_priority = sum(action_priorities.values())
    target_count = {
        action: (round(priority / total_priority) * max_num)
        for action, priority in action_priorities.items()
    }

    difference = max_num - sum(target_count.values())

    # 如果总和不等于max_num, 则将差值分配到每个动作中
    while difference != 0:
        for act in range(sorted_actions.keys()):
            if difference == 0:
                break
            if difference > 0:
                target_count[act] += 1
                difference -= 1
            if difference < 0 and target_count[act] > 0:
                target_count[act] -= 1
                difference += 1

    # 优先选择高优先级的动作片段
    for action, max_cnt in target_count.items():
        cur_count = 0
        for action_info in videos[action]:
            selected_segments.append(action_info)
            cur_count += 1
            if cur_count >= max_cnt:
                break

    return selected_segments


def split_and_concat_videos(selected_segments, start_offset, end_offset):
    """
    分割并拼接选定的片段生成精彩视频。

    参数:
        selected_segments (list): 选择的片段列表，包括视频名和动作信息。
    """
    # 创建 'videos' 文件夹如果它不存在
    output_dir = Path.cwd() / "videos"
    output_dir.mkdir(exist_ok=True)

    # 使用时间戳生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"highlight_reel_{timestamp}.mp4"
    output_path = output_dir / output_file

    trimmed_clips = []

    VIDEO_BASE_DIR = Path(
        "/data/ysp_public_data/sport-editing/basketball_video_split_10s"
    )

    def trim_clip(action_info, index, start_offset, end_offset):
        start, end = action_info["segment"]
        start = max(0, start - start_offset)
        end += end_offset
        input_video_path = str(VIDEO_BASE_DIR / action_info["video_name"]) + ".mp4"
        temp_output = output_dir / f"clip_{index}.mp4"

        # 构建文字内容
        text = f"视频: {action_info['video_name']}, 时间: {start}-{end}秒"

        # 使用 ffmpeg 添加文字叠加
        try:
            (
                ffmpeg.input(input_video_path, ss=start, to=end)
                .filter(
                    "drawtext",
                    text=text,
                    fontfile="./simsun.ttc",  # 替换为实际的字体文件路径
                    fontsize=24,
                    fontcolor="white",
                    x=10,
                    y=10,
                    box=True,
                    boxcolor="black@0.5",
                    boxborderw=5,
                )
                .output(str(temp_output))
                .run(overwrite_output=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"ffmpeg 错误: {e.stderr.decode('utf8')}")
            raise e

        return str(temp_output)

    # 使用线程池进行多线程处理
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_index = {
            executor.submit(trim_clip, video_info, idx, start_offset, end_offset): idx
            for idx, video_info in enumerate(selected_segments)
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(selected_segments),
            desc="视频切片中",
        ):
            try:
                trimmed_clip = future.result()
                trimmed_clips.append(trimmed_clip)
            except Exception as e:
                print(f"剪辑失败: {e}")

    if not trimmed_clips:
        print("没有成功剪辑的片段来生成精彩视频。")
        return

    # 拼接所有剪辑的片段
    try:
        # 创建一个临时文件列表
        list_file = output_dir / "clips.txt"
        with open(list_file, "w") as f:
            for clip in trimmed_clips:
                f.write(f"file '{clip}'\n")

        with tqdm(total=1, desc="视频拼接中") as pbar:
            # 使用 ffmpeg 进行拼接
            ffmpeg.input(str(list_file), format="concat", safe=0).output(
                str(output_path), c="copy"
            ).run(overwrite_output=True, quiet=True)
            pbar.update(1)
        print(f"精彩视频已生成: {output_path}")
    except Exception as e:
        print(f"视频拼接失败: {e}")
    finally:
        # 清理临时剪辑文件
        for clip in trimmed_clips:
            try:
                os.remove(clip)
            except OSError as e:
                print(f"无法删除临时文件 {clip}: {e}")
        try:
            os.remove(list_file)
        except OSError as e:
            print(f"无法删除列表文件 {list_file}: {e}")


def transform_annotation(nms_data, video_list):
    """
    根据nms后的数据，进行倒排索引
    将videos中数据结构由{video_name: [{segment: [start, end], score: 0.9, label: 'Three'}]} 转换为 {label: [{video_name: video_name, segment: [start, end], score: 0.9}]}, 并根据置信度排序

    参数:
        nms_data (dict): nms后的数据
        video_list (list): 视频列表
    """
    # 根据视频列表获取每个视频对应的动作
    # 如果video_list为空则选择全部视频
    if video_list:
        videos = {video: nms_data.get(video, []) for video in video_list}
    else:
        videos = nms_data
    # 将videos中数据结构由{video_name: [{segment: [start, end], score: 0.9, label: 'Three'}]} 转换为 {label: [{video_name: video_name, segment: [start, end], score: 0.9}]}, 并根据置信度排序
    transformed_videos = defaultdict(list)
    for video, actions in videos.items():
        for action in actions:
            transformed_videos[action["label"]].append(
                {
                    "video_name": video,
                    "segment": action["segment"],
                    "score": action["score"],
                }
            )
    # 每个value根据score排序
    for label, actions in transformed_videos.items():
        transformed_videos[label] = sorted(
            actions, key=lambda x: x["score"], reverse=True
        )
    return transformed_videos


def gen_by_time(
    video_list,
    desired_actions,
    action_priorities,
    max_duration,
    json_file,
    start_offset=1,
    end_offset=3,
):
    """
    根据视频时间生成集锦，负责加载数据，选择视频片段，并生成精彩视频。

    参数:
        video_list (list): 视频列表。
        desired_actions (list): 所需的动作列表。
        action_priorities (dict): 动作的优先级。
        max_duration (float): 最大时长（秒）。
        json_file (str): JSON数据文件路径。
        start_offset (float): 开始偏移（秒）。
        end_offset (float): 结束偏移（秒）。
    """
    # 如果最大时长小于30秒，设置为30秒
    if max_duration < 30:
        max_duration = 30

    # 加载动作数据
    action_data = load_action_data(json_file)

    filtered_data = filter_action_data(action_data, conf=0, min_duration=3.0)
    nms_data = apply_nms(filtered_data, iou_threshold=0.3)

    transformed_videos = transform_annotation(nms_data, video_list)
    # 选择符合条件的片段
    selected_segments = select_segments_by_time(
        transformed_videos, desired_actions, action_priorities, max_duration
    )

    pprint(selected_segments)
    # 分割并拼接选择的片段生成输出视频
    split_and_concat_videos(selected_segments, start_offset, end_offset)


def gen_by_action(
    video_list,
    desired_actions,
    action_priorities,
    max_num,
    json_file,
    start_offset=1,
    end_offset=2,
):
    """
    根据视频片段数量生成集锦，负责加载数据，选择视频片段，并生成精彩视频。

    参数:
        video_list (list): 视频列表。
        desired_actions (list): 所需的动作列表。
        action_priorities (dict): 动作的优先级。
        json_file (str): JSON数据文件路径。
        start_offset (float): 开始偏移（秒）。
        end_offset (float): 结束偏移（秒）。
    """

    # 加载动作数据
    action_data = load_action_data(json_file)

    filtered_data = filter_action_data(action_data, conf=0, min_duration=3.0)
    nms_data = apply_nms(filtered_data, iou_threshold=0.3)

    transformed_videos = transform_annotation(nms_data, video_list)
    pprint(transformed_videos)
    # 选择符合条件的片段
    selected_segments = select_segments_by_action(
        transformed_videos, desired_actions, action_priorities, max_num
    )

    pprint(selected_segments)
    # 分割并拼接选择的片段生成输出视频
    split_and_concat_videos(selected_segments, start_offset, end_offset)


if __name__ == "__main__":
    video_list = []  # 示例视频列表
    desired_actions = ["Three"]  # 示例所需动作
    action_priorities = {"Three": 1.0}  # 示例动作优先级
    max_duration = 60  # 示例最大时长（秒）
    json_file = "/data/zzm/aigc/OpenTAD/exps/thumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/basketball-1010-freeze/gpu8_id0/result_detection.json"  # 假设的JSON文件名

    # 根据时间生成视频
    # gen_by_time(
    #     video_list,
    #     desired_actions,
    #     action_priorities,
    #     max_duration,
    #     json_file,
    #     start_offset=0,
    #     end_offset=3,
    # )

    # 根据片段个数生成视频
    gen_by_action(
        video_list,
        desired_actions,
        action_priorities,
        5,
        json_file,
        start_offset=3,
        end_offset=3,
    )
