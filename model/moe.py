"""MoE feed forward layer: top-k routed experts, scaled, plus an always-on shared expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import IndicPrayogConfig
from .experts import build_expert
from .router import Router


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    load_fraction: torch.Tensor


class MoELayer(nn.Module):
    def __init__(self, config: IndicPrayogConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.router = Router(config)
        self.experts = nn.ModuleList(build_expert(config) for _ in range(config.num_experts))
        self.shared_experts = nn.ModuleList(
            build_expert(config) for _ in range(config.num_shared_experts)
        )

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        batch, seq_len, hidden_size = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_size)
        routing = self.router(tokens)

        assignments = routing.expert_indices.reshape(-1)
        weights = routing.expert_weights.reshape(-1).to(tokens.dtype)

        # group assignments by expert so each expert runs one batched matmul
        order = torch.argsort(assignments)
        counts = torch.bincount(assignments, minlength=self.num_experts)
        boundaries = counts.cumsum(0).tolist()

        routed = torch.zeros_like(tokens)
        start = 0
        for expert_idx, end in enumerate(boundaries):
            if end > start:
                slots = order[start:end]
                token_idx = slots // self.top_k
                expert_out = self.experts[expert_idx](tokens[token_idx])
                routed.index_add_(0, token_idx, expert_out * weights[slots].unsqueeze(-1))
            start = end

        out = routed * self.routed_scaling_factor
        for shared_expert in self.shared_experts:
            out = out + shared_expert(tokens)

        return MoEOutput(
            hidden_states=out.view(batch, seq_len, hidden_size),
            aux_loss=routing.aux_loss,
            load_fraction=routing.load_fraction,
        )
