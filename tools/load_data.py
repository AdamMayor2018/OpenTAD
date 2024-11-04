import json
import os
import numpy as np
from mmengine.dataset import Compose
from opentad.datasets.builder import DATASETS, get_class_index, build_dataset
from opentad.datasets.base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from opentad.utils.logger import setup_logger
import argparse
from mmengine.config import Config, DictAction
import time
import torch
import json
import threading
import queue
import concurrent.futures
from tqdm import tqdm

@DATASETS.register_module()
class LoadPadddingVideoData(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return {}
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)
    def worker(self, video_queue, pt_path):
        while not video_queue.empty():
            try:
                video_name = video_queue.get_nowait()  # 尝试从队列中获取一个视频名称
                self.get_single_data(video_name, pt_path)  # 处理视频
                video_queue.task_done()  # 标记任务完成
            except queue.Empty:
                break

    def get_data(self, pt_path):


        print("Loading video data in padding mode...")
        count = 0
        count_all = 0
        assert os.path.exists(pt_path)
        pos_neg_count = 0
        for idx in tqdm(range(len(self.data_list)), total=len(self.data_list)):
            video_name, video_info, video_anno = self.data_list[idx]
            if video_anno != {}:
                video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
                video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride
            times = 1
            # start_time = time.time()
            video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            # if os.path.exists(video_pt_path):
            #     print(f"{video_name} already exists, skip it.")
            #     count += 1
            #     continue        
            results = self.pipeline(
                dict(
                    video_name=video_name,
                    data_path=self.data_path,
                    sample_stride=self.sample_stride,
                    snippet_stride=self.snippet_stride,
                    fps=video_info["frame"] / video_info["duration"],
                    duration=video_info["duration"],
                    offset_frames=self.offset_frames,
                    **video_anno,
                )
            )
            
            pos_neg_count += results['have_action']
            count_all += 1
            torch.save(results, video_pt_path, _use_new_zipfile_serialization=False)
            # end_time = time.time()
            
            # print(f"Processing video {video_name} takes {end_time - start_time:.2f} seconds.")
            if count_all % 50 == 0:
                print(f"Processed {count_all} videos, {pos_neg_count} positive and negative samples.")
        
        print("Loading train video data done. There are {} videos have been processed before. {} videos have been loaded.".format(count, count_all))
        return results

    def get_single_data(self, data_idx, pt_path):
        video_name, video_info, video_anno = data_idx
        if video_anno != {}:
                video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
                video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride
            
        video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
        if os.path.exists(video_pt_path):
            # count += 1
            print(f"{video_name} already exists, skip it.")
        else:
            start_time = time.time()
            results = self.pipeline(
                dict(
                    video_name=video_name,
                    data_path=self.data_path,
                    sample_stride=self.sample_stride,
                    snippet_stride=self.snippet_stride,
                    fps=video_info["frame"] / video_info["duration"],
                    duration=video_info["duration"],
                    offset_frames=self.offset_frames,
                    **video_anno,
                )
            )
            end_time = time.time()
            # count_all += 1
            print(f"Processing video {video_name} takes {end_time - start_time:.2f} seconds.")
            ss = time.time()
            torch.save(results, video_pt_path, _use_new_zipfile_serialization=False)
            ee = time.time()
            print(f"Save video {video_name} takes {ee - ss:.2f} seconds.")
    
    def load_data(self, pt_path):
        assert os.path.exists(pt_path)
        for idx in range(len(self.data_list)):
            video_name, video_info, video_anno = self.data_list[idx]
            video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            if os.path.exists(video_pt_path):
                results = torch.load(video_pt_path)
        return results
        
@DATASETS.register_module()
class LoadComposeVideoData(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)


    def get_data(self, pt_path):
        print("Loading decord data...")
        count = 0
        count_all = 0
        assert os.path.exists(pt_path)
        for idx in range(len(self.data_list)):
            video_name, video_info, video_anno = self.data_list[idx]
            if video_anno != {}:
                video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
                video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride
            
            video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            # if os.path.exists(video_pt_path):
            #     count += 1
            #     continue
            #     print(f"{video_name} already exists, skip it.")
            if os.path.exists(video_pt_path):
                results = torch.load(video_pt_path)

            start = time.time()
            keys = ['video_name', 'data_path', 'sample_stride', 'snippet_stride', 'fps', 'duration', 'offset_frames', 'gt_segments', 'gt_labels', 'modality', 'filename', 'total_frames', 'video_reader', 'avg_fps', 'frame_inds', 'num_clips', 'clip_len', 'masks', 'imgs', 'original_shape', 'img_shape']
            new_results = {k: results[k] for k in keys}
            # new_results['imgs'] = new_results['imgs'].numpy()
            new_results['masks'] = new_results['masks'].numpy()
            ret = self.pipeline(
                new_results
            )
            end = time.time()
            count_all += 1
            print(f"Processing video {video_name} takes {end - start:.2f} seconds.")
            torch.save(results, video_pt_path, _use_new_zipfile_serialization=False)
        print("Loading aug video data done. There are {} videos have been processed before. {} videos have been loaded.".format(count, count_all))
        # return results
    def load_data(self, pt_path):
        assert os.path.exists(pt_path)
        for idx in range(len(self.data_list)):
            video_name, video_info, video_anno = self.data_list[idx]
            video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            if os.path.exists(video_pt_path):
                results = torch.load(video_pt_path)
        return results


@DATASETS.register_module()
class LoadSlidingVideoData(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            print("have no valid gt")
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def get_data(self, pt_path):
        print("Loading video data in sliding mode...")
        count = 0
        count_all = 0
        pos_neg_count = 0
        assert os.path.exists(pt_path)
        for idx in tqdm(range(len(self.data_list)), total=len(self.data_list)):
            video_name, video_info, video_anno, window_snippet_centers = self.data_list[idx]
            if video_anno != {}:
                video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
                video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride
            times = 1
            video_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            # if os.path.exists(video_pt_path):
            #     count += 1
            #     print(f"{video_name} already exists, skip it.")
            #     continue
            results = self.pipeline(
                dict(
                    video_name=video_name,
                    data_path=self.data_path,
                    window_size=self.window_size,
                    # trunc window setting
                    feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                    feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                    sample_stride=self.sample_stride,
                    # sliding post process setting
                    fps=video_info["frame"] / video_info["duration"],
                    snippet_stride=self.snippet_stride,
                    window_start_frame=window_snippet_centers[0],
                    duration=video_info["duration"],
                    offset_frames=self.offset_frames,
                    # training setting
                    **video_anno,
                )
            )
            pos_neg_count += results['have_action']

            count_all += 1
            torch.save(results, video_pt_path, _use_new_zipfile_serialization=False)
                # if count_all % 50 == 0:
                #     print(f"Processed {count_all} videos, {pos_neg_count} positive and negative samples.")
            # end_time = time.time()
            # print(f"Processing video {video_name} takes {end_time - start_time:.2f} seconds.")
        print("Loading val video data done. There are {} videos have been processed before. {} videos have been loaded.".format(count, count_all))
        # return results
    def load_data(self, pt_path):
        assert os.path.exists(pt_path)
        for idx in range(len(self.data_list)):
            video_name, video_info, video_anno = self.data_list[idx]
            data_pt_path = os.path.join(pt_path, f"{video_name}.pt")
            if os.path.exists(data_pt_path):
                results = torch.load(data_pt_path)
        return results
    def check_data_list(self):
        return self.data_list
    

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--path", type=str, default='/data/ysp_public_data/sport-editing/pt_file/', help="the dir to save video pt files")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # load config
    cfg = Config.fromfile(args.config)
    args.rank = 0
    pt_path = args.path
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)

    print('start load val')
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_dataset.get_data(pt_path)
    print('start load test')
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_dataset.get_data(pt_path)
    print('start load train')
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_dataset.get_data(pt_path)


    # aug_dataset = build_dataset(cfg.dataset.aug, default_args=dict(logger=logger))
    # aug_dataset.get_data(pt_path)

    




if __name__ == "__main__":
    
    main()

    print("Done!")