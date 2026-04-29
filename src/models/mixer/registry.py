from typing import Dict, Type

import torch.nn as nn

from models.mixer.mixer_config import BaseMixerConfig

MIXER_REGISTRY: Dict[Type[BaseMixerConfig], Type[nn.Module]] = {}


def register_mixer(config_class: Type[BaseMixerConfig]):
    def decorator(mixer_class: Type[nn.Module]):
        MIXER_REGISTRY[config_class] = mixer_class
        return mixer_class

    return decorator


def build_mixer(config: BaseMixerConfig) -> nn.Module:
    mixer_class = MIXER_REGISTRY.get(type(config))
    if mixer_class is None:
        raise ValueError(
            f"No hay ningún mixer registrado para la configuración: {type(config).__name__}"
        )
    return mixer_class(config)
