from omegaconf import DictConfig
import torch.nn as nn

from models.metaformer import BlockConfig, MetaFormerConfig, Metaformer, StageConfig
from models.mixer.gated_cnn import GatedCNNMixerConfig
from models.module.drop_path import calculate_drop_path_rates


def build_small_gcnn(cfg: DictConfig) -> nn.Module:
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    depths = [3, 3, 9, 3]
    dp_rates = calculate_drop_path_rates(drop_path_rate, depths)

    in_dims = [40, 80, 160, 320]
    out_dims = [80, 160, 320, 320]

    config = MetaFormerConfig(
        num_classes=cfg.num_classes,
        stages=[
            StageConfig(
                in_dim=in_dims[0],
                out_dim=out_dims[0],
                mixer_configs=[GatedCNNMixerConfig(in_dims[0])] * depths[0],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:0]), sum(depths[:1]))
                ],
            ),
            StageConfig(
                in_dim=in_dims[1],
                out_dim=out_dims[1],
                mixer_configs=[GatedCNNMixerConfig(in_dims[1])] * depths[1],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:1]), sum(depths[:2]))
                ],
            ),
            StageConfig(
                in_dim=in_dims[2],
                out_dim=out_dims[2],
                mixer_configs=[GatedCNNMixerConfig(in_dims[2])] * depths[2],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:2]), sum(depths[:3]))
                ],
            ),
            StageConfig(
                in_dim=in_dims[3],
                out_dim=out_dims[3],
                mixer_configs=[GatedCNNMixerConfig(in_dims[3])] * depths[3],
                block_cfgs=[
                    BlockConfig(use_mlp=False, drop_path=dp_rates[i])
                    for i in range(sum(depths[:3]), sum(depths[:4]))
                ],
            ),
        ],
    )

    model = Metaformer(config)
    return model
