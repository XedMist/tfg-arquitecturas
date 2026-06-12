# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN + FPN — Backbone: ResNet-50  [BASELINE]
# Schedule: 2x (24 épocas), COCO 2017 — 20 clases subset TFG
#
# Configuración idéntica a faster_rcnn_gated_cnn_fpn_2x_20classes:
#   - Mismo optimizador (AdamW + AMP)
#   - Mismo pipeline de augmentación y escala (800x800)
#   - Mismo batch_size (24 train, 8 val)
#   - Mismas 20 clases
#   - Mismo schedule 2x (milestones 16, 22)
#
# Pre-entrenamiento backbone: ImageNet-1k (pesos oficiales de torchvision/MMDet).
# NO se necesita custom_imports porque R50 ya está en MMDetection.
# ─────────────────────────────────────────────────────────────────────────────

_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
    "mmdet::_base_/datasets/coco_detection.py",
    "mmdet::_base_/default_runtime.py",
]

# Mismas 20 clases que los MetaFormers entrenados
my_classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow',
)

# Solo ajustar cabeza a 20 clases; backbone R50 cargando los pesos customizados
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='checkpoints/detection/resnet50.pth'
        )
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=20)
    )
)

data_root = "/data/coco/"

# LR se escala automáticamente con el batch real (base=32, batch=24 → lr*0.75)
auto_scale_lr = dict(enable=True, base_batch_size=32)

# ── Schedule 2x ──────────────────────────────────────────────────────────────
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=4)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500,
    ),
    dict(
        type="MultiStepLR", begin=0, end=24, by_epoch=True,
        milestones=[16, 22], gamma=0.1,
    ),
]

# ── Optimizador — igual que MetaFormers ──────────────────────────────────────
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale="dynamic",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=0.5, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            ".bias": dict(decay_mult=0.0),
        }
    ),
)

# ── Pipelines — igual que MetaFormers entrenados ─────────────────────────────
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
                        (480, 800), (544, 800), (608, 800),
                        (672, 800), (736, 800), (800, 800),
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
    dict(type="Resize", scale=(800, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# ── Dataloaders ───────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=my_classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    dataset=dict(
        type="CocoDataset",
        metainfo=dict(classes=my_classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        pipeline=test_pipeline,
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

# ── Hooks y visualización ─────────────────────────────────────────────────────
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

work_dir = "outputs/detection/faster_rcnn_r50_fpn_2x_20classes"
