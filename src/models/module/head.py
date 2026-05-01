import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
