## 1. 更新视频切分脚本。
### 1.1 基于原始视频和原始标注文件，生成annotation.json文件。
```
python tools/split_viideo/make_base_anno.py --base_file=base_anno_1104.json
会在 basketball_annotation 文件夹下生成 base_anno_1104.json 作为原始标注文件。
```
### 1.2 基于annotation.json文件，生成视频切分文件。
```
python tools/make_data.py 
注意每次生成前检查 output_dir。
会在 /data/ysp_public_data/sport-editing/basketball_video_split_5class_120s_1030 文件夹下生成切分好的视频。
```
### 1.3 基于已有的切分文件，生成pt 文件用于训练。
```
python tools/load_data.py configs/ysp/self_decord_video_s_768x1_160_adapter.py --path=/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s_1030/pt
```

## 2. 模型训练修改。
### 2.1 train 代码
```
CUDA_VISIBLE_DEVICES=1,2,6,7 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py configs/ysp/self_aug_video_s_768x1_160_adapter.py
注意检查 config 文件的 路径是否都能对上。
注意检查 window 大小是否符合要求。是否和 生成pt 文件的pt 一致。

```