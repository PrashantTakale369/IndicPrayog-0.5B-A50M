"""Grouped query attention with QK-Norm, RoPE and causal masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import IndicPrayogConfig
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    if n_repeat == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, num_kv_heads, n_repeat, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_repeat, seq_len, head_dim)


class Attention(nn.Module):
    def __init__(self, config: IndicPrayogConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.queries_per_kv_head = config.queries_per_kv_head
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # normalize before rotating: RoPE preserves vector length, so the learned
        # per-dim scale must be applied in the unrotated frame
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.queries_per_kv_head)
        v = repeat_kv(v, self.queries_per_kv_head)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn)
