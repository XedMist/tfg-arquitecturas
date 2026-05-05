_base_ = [
    "mmdet::_base_/models/faster_rcnn_r50_fpn.py",
    "mmdet::_base_/datasets/coco_detection.py",
    "mmdet::_base_/schedules/schedule_1x.py",
    "mmdet::_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["detection.backbones"],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="MetaFormerBackbone",
        arch="gated_cnn",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        pretrained=None,
    ),
    neck=dict(
        type="FPN",
        in_channels=[96, 192, 384, 576],
        out_channels=256,
        num_outs=5,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
    ),
)

data_root = "/data/coco/"

train_dataloader = dict(
    batch_size=2,  # por GPU — ajusta según VRAM disponible
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    classwise=True,
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        # No aplicar weight decay a norm layers ni biases
        custom_keys={
            "norm": dict(decay_mult=0.0),
            ".bias": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,  # 500 iteraciones de warmup
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(800, 1333), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
    ),
    logger=dict(type="LoggerHook", interval=50),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
