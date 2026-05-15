from omegaconf import DictConfig
import torch.nn as nn

from models.metaformer import BlockConfig, MetaFormerConfig, Metaformer, StageConfig
from models.mixer.gated_cnn import GatedCNNMixerConfig
from models.module.drop_path import calculate_drop_path_rates


def build_small_gcnn(cfg: DictConfig) -> nn.Module:
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    depths = [3, 3, 9, 3]
    dp_rates = calculate_drop_path_rates(drop_path_rate, depths)

    config = MetaFormerConfig(
        num_classes=cfg.num_classes,
        stages=[
            StageConfig(
                in_dim=48,
                out_dim=96,
                mixer_configs=[GatedCNNMixerConfig(48)] * depths[0],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:0]), sum(depths[:1]))
                ],
            ),
            StageConfig(
                in_dim=96,
                out_dim=192,
                mixer_configs=[GatedCNNMixerConfig(96)] * depths[1],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:1]), sum(depths[:2]))
                ],
            ),
            StageConfig(
                in_dim=192,
                out_dim=384,
                mixer_configs=[GatedCNNMixerConfig(192)] * depths[2],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:2]), sum(depths[:3]))
                ],
            ),
            StageConfig(
                in_dim=384,
                out_dim=384,
                mixer_configs=[GatedCNNMixerConfig(384)] * depths[3],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:3]), sum(depths[:4]))
                ],
            ),
        ],
    )

    model = Metaformer(config)
    return model
