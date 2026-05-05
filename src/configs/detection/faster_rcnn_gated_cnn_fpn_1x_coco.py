# ─────────────────────────────────────────────────────────────────────────────
# Faster R-CNN + FPN — Backbone: MetaFormer GatedCNN
# Schedule: 1x (12 épocas), COCO 2017
#
# Comparado con Swin-T Faster R-CNN 1x (mAP ~46.0):
#   - Mismo FPN, mismo head, mismo schedule, misma augmentación
#   - Solo varía el backbone
# ─────────────────────────────────────────────────────────────────────────────

_base_ = ["./_base_faster_rcnn_metaformer.py"]

# Canales de salida de GatedCNN: [96, 192, 384, 576]
_out_channels = [96, 192, 384, 576]

model = dict(
    backbone=dict(
        _delete_=True,
        type="MetaFormerBackbone",
        arch="gated_cnn",
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,          # congelar stem para fine-tune
        pretrained=None,           # rellenar con path tras convert_checkpoint.py
        drop_path_rate=0.1,
    ),
    neck=dict(
        in_channels=_out_channels,
        out_channels=256,
        num_outs=5,
    ),
)

# Directorio de trabajo y nombre del experimento
work_dir = "outputs/detection/faster_rcnn_gated_cnn_fpn_1x"
