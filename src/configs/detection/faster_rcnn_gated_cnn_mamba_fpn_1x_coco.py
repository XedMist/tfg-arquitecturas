# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN + FPN — Backbone: MetaFormer GatedCNN-Mamba
# Schedule: 1x (12 épocas), COCO 2017
#
# Stages 0-1: GatedCNN | Stages 2-3: Mamba SSM
# Canales: [96, 192, 384, 576]
# ─────────────────────────────────────────────────────────────────────────────

_base_ = ["./_base_faster_rcnn_metaformer.py", "./_schedule_1x.py"]

_out_channels = [96, 192, 384, 576]

model = dict(
    backbone=dict(
        _delete_=True,
        type="MetaFormerBackbone",
        arch="gated_cnn_mamba",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        pretrained="checkpoints/detection/gated_cnn_mamba_backbone.pth",  # path tras convert_checkpoint.py
        drop_path_rate=0.1,
    ),
    neck=dict(
        in_channels=_out_channels,
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = "outputs/detection/faster_rcnn_gated_cnn_mamba_fpn_1x"
