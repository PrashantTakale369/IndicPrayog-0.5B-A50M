"""MoE router: independent per-expert scores, expert bias for load balance, top-k select."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import IndicPrayogConfig

ROUTER_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class RouterOutput:
    expert_indices: torch.Tensor  # [tokens, top_k] long
    expert_weights: torch.Tensor  # [tokens, top_k], rows sum to 1
    aux_loss: torch.Tensor  # scalar, 1.0 when load is perfectly uniform
    load_fraction: torch.Tensor  # [num_experts], sums to 1


class Router(nn.Module):
    def __init__(self, config: IndicPrayogConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.score_function = config.score_function
        self.router_dtype = ROUTER_DTYPES[config.router_dtype]
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        if config.moe_router_enable_expert_bias:
            # updated by update_expert_bias, never by backprop
            self.expert_bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        else:
            self.expert_bias = None

    def forward(self, tokens: torch.Tensor) -> RouterOutput:
        with torch.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.to(self.router_dtype), self.weight.to(self.router_dtype))

            if self.score_function == "sigmoid":
                scores = torch.sigmoid(logits)
            else:
                scores = torch.softmax(logits, dim=-1)

            # the bias steers selection only; the gate value stays the raw affinity
            selection_scores = scores if self.expert_bias is None else scores + self.expert_bias
            expert_indices = torch.topk(selection_scores, self.top_k, dim=-1).indices

            expert_weights = scores.gather(-1, expert_indices)
            expert_weights = expert_weights / expert_weights.sum(-1, keepdim=True).clamp_min(1e-9)

            assignments = torch.bincount(expert_indices.reshape(-1), minlength=self.num_experts)
            load_fraction = assignments.to(scores.dtype) / assignments.sum().clamp_min(1)

            mean_score = (scores / scores.sum(-1, keepdim=True).clamp_min(1e-9)).mean(0)
            aux_loss = self.num_experts * (load_fraction * mean_score).sum()

        return RouterOutput(expert_indices, expert_weights, aux_loss, load_fraction)

    @torch.no_grad()
    def update_expert_bias(self, load_fraction: torch.Tensor, update_rate: float) -> None:
        if self.expert_bias is None:
            return
        uniform_share = 1.0 / self.num_experts
        error = uniform_share - load_fraction.to(self.expert_bias.dtype)
        self.expert_bias.add_(update_rate * torch.sign(error))
