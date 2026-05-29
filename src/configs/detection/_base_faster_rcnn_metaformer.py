_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
    "mmdet::_base_/datasets/coco_detection.py",
    "mmdet::_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["detection.backbones"],
    allow_failed_imports=False,
)

# 40 clases seleccionadas para el TFG (la mitad de COCO)
my_classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle'
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
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=40,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
)

data_root = "/data/coco/"

auto_scale_lr = dict(enable=False, base_batch_size=2)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=my_classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=my_classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(
        _delete_=True,
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
                        (480, 1024),
                        (544, 1024),
                        (608, 1024),
                        (672, 1024),
                        (736, 1024),
                        (800, 1024),
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
    dict(type="Resize", scale=(800, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader['dataset']['pipeline'] = train_pipeline
val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=4,
        max_keep_ckpts=2,
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

