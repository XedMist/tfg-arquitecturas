# Schedule 2x — 24 épocas
# Drop LR en epochs 16 y 22 (proporcional a 1x: 8→16, 11→22)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=4)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1,
    ),
]
