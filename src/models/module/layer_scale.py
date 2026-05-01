import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(self, d_model: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(d_model), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
