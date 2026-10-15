"""RMS normalization, rotary positions, grouped-query attention and SwiGLU."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import IndicPrayogConfig

KVCache = tuple[Tensor, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Reduce in fp32 even when weights/activations are bf16.
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, q: Tensor, k: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # Compute on demand, rather than allocating a max_context squared table.
        # Keep the phase calculation out of autocast for long-position precision.
        with torch.autocast(device_type=q.device.type, enabled=False):
            inv_freq = self.theta ** (
                -torch.arange(0, self.head_dim, 2, device=q.device, dtype=torch.float32)
                / self.head_dim
            )
            angles = position_ids.float().unsqueeze(-1) * inv_freq
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            cos, sin = angles.cos(), angles.sin()

        def rotate(x: Tensor) -> Tensor:
            left, right = x.float().chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
            return (x.float() * cos + rotated * sin).to(x.dtype)

        return rotate(q), rotate(k)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: IndicPrayogConfig):
        super().__init__()
        self.config = config
        h, d = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(h, config.num_attention_heads * d, bias=False)
        self.k_proj = nn.Linear(h, config.num_key_value_heads * d, bias=False)
        self.v_proj = nn.Linear(h, config.num_key_value_heads * d, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * d, h, bias=False)
        self.q_norm = RMSNorm(d, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.rotary = RotaryEmbedding(d, config.rope_theta)

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        is_causal: bool,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        batch, length, _ = x.shape
        c = self.config

        def project(layer: nn.Linear, heads: int) -> Tensor:
            return layer(x).view(batch, length, heads, c.head_dim).transpose(1, 2)

        q = self.q_norm(project(self.q_proj, c.num_attention_heads))
        k = self.k_norm(project(self.k_proj, c.num_key_value_heads))
        v = project(self.v_proj, c.num_key_value_heads)
        q, k = self.rotary(q, k, position_ids)
        if past_key_value is not None:
            if past_key_value[0].dtype != k.dtype or past_key_value[1].dtype != v.dtype:
                raise ValueError("cache dtype must match the current attention dtype")
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None
        # Store only KV heads in the cache. Repeat here for portable CPU/CUDA SDPA.
        k = k.repeat_interleave(c.queries_per_kv_head, dim=1)
        v = v.repeat_interleave(c.queries_per_kv_head, dim=1)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=c.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, c.hidden_size)
        return self.o_proj(attended), present


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
