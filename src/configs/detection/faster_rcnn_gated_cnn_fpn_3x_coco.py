_base_ = [
    "./_base_faster_rcnn_metaformer.py",
    "./_schedule_3x.py",
]

_out_channels = [96, 192, 384, 576]

model = dict(
    backbone=dict(
        _delete_=True,
        type="MetaFormerBackbone",
        arch="gated_cnn",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        pretrained="checkpoints/detection/gated_cnn.pth",
        drop_path_rate=0.1,
    ),
    neck=dict(in_channels=_out_channels, out_channels=256, num_outs=5),
)

work_dir = "outputs/detection/faster_rcnn_gated_cnn_fpn_3x"
