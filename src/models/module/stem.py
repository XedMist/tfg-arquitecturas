import torch
import torch.nn as nn


class StemLayer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=96,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1
        )
        self.norm1 = nn.LayerNorm(out_channels // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, C, H, W = X.shape
        X = self.conv1(X)  # (B, C/2, H/2, W/2)
        X = X.permute(0, 2, 3, 1)  # BHWC
        X = self.norm1(X)
        X = X.permute(0, 3, 1, 2)  # BCHW
        X = self.act(X)
        X = self.conv2(X)  # (B, C, H/4, W/4)
        X = X.permute(0, 2, 3, 1)  # BHWC
        X = self.norm2(X)
        return X
