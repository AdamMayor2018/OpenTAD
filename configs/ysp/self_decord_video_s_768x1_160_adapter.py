_base_ = [
    # "../_base_/datasets/thumos-14/self_train_trunc_test_sw_256x224x224.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

annotation_path = "/data/ysp_public_data/sport-editing/basketball_annotation/short_anno_1030.json"
class_map = "/data/ysp_public_data/sport-editing/basketball_annotation/less_category_idx.txt"
data_path = "/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s_1030"
pt_path = "/data/ysp_public_data/sport-editing/basketball_video_split_5class_120s_1030/pt"

block_list = None
window_size = 64
scale_factor = 1
chunk_num = window_size * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        type="LoadPadddingVideoData",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        pt_path=pt_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
        ],
    ),
    val=dict(
        type="LoadSlidingVideoData",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        pt_path=pt_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
        ],
    ),
    test=dict(
        type="LoadSlidingVideoData",
        ann_file=annotation_path,
        subset_name="test",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        pt_path=pt_path,
        filter_gt=False,
        # test_mode=True,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
        ],
    ),
    aug=dict(
        pipeline=[
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
)



model = dict(
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,  # enable activation checkpointing
            total_frames=window_size * scale_factor,
            adapter_index=list(range(12)),
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
            norm_eval=False,  # also update the norm layers
            freeze_backbone=True,  # unfreeze the backbone
        ),
    ),
    projection=dict(
        in_channels=384,
        max_seq_len=window_size,
        attn_cfg=dict(n_mha_win_size=-1),
    ),
)

solver = dict(
    train=dict(batch_size=32, num_workers=4),
    val=dict(batch_size=32, num_workers=4),
    test=dict(batch_size=32, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=100)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=True,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=1,
    end_epoch=200,
)

work_dir = "exps/self/dataset/1029"
