"""SwiGLU feed forward network: used both as one MoE expert and as layer 0's dense FFN."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import IndicPrayogConfig


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_expert(config: IndicPrayogConfig) -> SwiGLU:
    return SwiGLU(config.hidden_size, config.moe_intermediate_size)


def build_dense_ffn(config: IndicPrayogConfig) -> SwiGLU:
    return SwiGLU(config.hidden_size, config.intermediate_size)
