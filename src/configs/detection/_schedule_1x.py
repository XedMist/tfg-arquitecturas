# Schedule 1x — 12 épocas

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=3)

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
