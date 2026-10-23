"""IndicPrayog-0.5B-A50M: embed -> dense layer 0 -> 18 MoE layers -> final norm -> lm_head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import Attention
from .config import IndicPrayogConfig
from .experts import build_dense_ffn
from .moe import MoELayer
from .norm import RMSNorm
from .rope import RotaryEmbedding
from .router import Router

INIT_STD = 0.02
IGNORE_INDEX = -100


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    expert_load: torch.Tensor | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: IndicPrayogConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe_layer(layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MoELayer(config) if self.is_moe else build_dense_ffn(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        attn_out = self.self_attn(self.input_layernorm(hidden_states), cos, sin, attention_mask)
        hidden_states = hidden_states + attn_out

        normed = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            moe = self.mlp(normed)
            return hidden_states + moe.hidden_states, moe.aux_loss, moe.load_fraction
        return hidden_states + self.mlp(normed), None, None


class IndicPrayogModel(nn.Module):
    def __init__(self, config: IndicPrayogConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(position_ids)

        hidden_states = self.embed_tokens(input_ids)
        aux_losses: list[torch.Tensor] = []
        loads: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, aux_loss, load_fraction = layer(hidden_states, cos, sin, attention_mask)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
                loads.append(load_fraction)

        return self.norm(hidden_states), torch.stack(aux_losses).sum(), torch.stack(loads)


class IndicPrayogForCausalLM(nn.Module):
    def __init__(self, config: IndicPrayogConfig) -> None:
        super().__init__()
        self.config = config
        self.model = IndicPrayogModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # residual output projections shrink with depth, else activations grow layer by layer
        residual_std = INIT_STD / math.sqrt(2 * config.num_hidden_layers)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        elif isinstance(module, Router):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
            if module.expert_bias is not None:
                nn.init.zeros_(module.expert_bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        hidden_states, aux_loss, expert_load = self.model(input_ids, position_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutput(
            logits=logits, loss=loss, aux_loss=aux_loss, expert_load=expert_load
        )

    @torch.no_grad()
    def update_expert_bias(self, expert_load: torch.Tensor, update_rate: float = 1e-3) -> None:
        moe_layers = [layer for layer in self.model.layers if layer.is_moe]
        for layer, load_fraction in zip(moe_layers, expert_load):
            layer.mlp.router.update_expert_bias(load_fraction, update_rate)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def expected_total_parameters(config: IndicPrayogConfig) -> int:
    embedding = config.vocab_size * config.hidden_size
    lm_head = 0 if config.tie_word_embeddings else embedding

    query = config.hidden_size * config.num_attention_heads * config.head_dim
    key_value = 2 * config.hidden_size * config.num_key_value_heads * config.head_dim
    output = config.num_attention_heads * config.head_dim * config.hidden_size
    qk_norm = 2 * config.head_dim if config.use_qk_norm else 0
    attention = config.num_hidden_layers * (query + key_value + output + qk_norm)

    dense_ffn = config.num_dense_layers * 3 * config.hidden_size * config.intermediate_size

    moe = config.num_moe_layers * (
        config.experts_per_moe_layer * _expert_params(config) + _router_params(config)
    )

    norms = (2 * config.num_hidden_layers + 1) * config.hidden_size

    return embedding + lm_head + attention + dense_ffn + moe + norms


def expected_active_parameters(config: IndicPrayogConfig) -> int:
    """Params that run for a single token. Embedding and lm_head excluded, per the A50M name."""
    query = config.hidden_size * config.num_attention_heads * config.head_dim
    key_value = 2 * config.hidden_size * config.num_key_value_heads * config.head_dim
    output = config.num_attention_heads * config.head_dim * config.hidden_size
    qk_norm = 2 * config.head_dim if config.use_qk_norm else 0
    attention = config.num_hidden_layers * (query + key_value + output + qk_norm)

    dense_ffn = config.num_dense_layers * 3 * config.hidden_size * config.intermediate_size

    moe = config.num_moe_layers * (
        config.active_experts_per_token * _expert_params(config) + _router_params(config)
    )

    norms = (2 * config.num_hidden_layers + 1) * config.hidden_size

    return attention + dense_ffn + moe + norms


def _expert_params(config: IndicPrayogConfig) -> int:
    return 3 * config.hidden_size * config.moe_intermediate_size


def _router_params(config: IndicPrayogConfig) -> int:
    bias = config.num_experts if config.moe_router_enable_expert_bias else 0
    return config.hidden_size * config.num_experts + bias
