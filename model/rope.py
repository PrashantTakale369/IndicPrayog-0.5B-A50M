"""RoPE: rotate queries and keys by an angle proportional to their position."""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float, max_position_embeddings: int) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        self.register_buffer("inv_freq", 1.0 / (rope_theta**exponent), persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-3).to(q.dtype)
    sin = sin.unsqueeze(-3).to(q.dtype)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out
