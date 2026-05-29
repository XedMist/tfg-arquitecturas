# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN + FPN — Backbone: MetaFormer GatedCNN-DAT
# Schedule: 1x (12 épocas), COCO 2017
#
# Stages 0-1: GatedCNN | Stages 2-3: Deformable Attention (DAT)
# Canales: [96, 192, 320, 512]
# ─────────────────────────────────────────────────────────────────────────────

_base_ = ["./_base_faster_rcnn_metaformer.py", "./_schedule_1x.py"]

# DAT tiene canales distintos en stages 2-3
_out_channels = [96, 192, 320, 512]

model = dict(
    backbone=dict(
        _delete_=True,
        type="MetaFormerBackbone",
        arch="gated_cnn_dat",
        out_indices=(0, 1, 2, 3),
        frozen_stages=2,
        pretrained="checkpoints/detection/gated_cnn_dat_backbone.pth",
        drop_path_rate=0.1,
    ),
    neck=dict(
        in_channels=_out_channels,
        out_channels=256,
        num_outs=5,
    ),
)

work_dir = "outputs/detection/faster_rcnn_gated_cnn_dat_fpn_1x"
