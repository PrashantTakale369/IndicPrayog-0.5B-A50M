"""Dropless sparse MoE; selection bias is separate from mixture weights."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from .config import IndicPrayogConfig
from .layers import SwiGLU


class TopKRouter(nn.Module):
    def __init__(self, config: IndicPrayogConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # This is algorithm state, not a learned parameter or an auxiliary loss.
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_experts, dtype=torch.float32)
            if config.moe_router_enable_expert_bias
            else None,
        )
        self.register_buffer(
            "token_counts", torch.zeros(config.num_experts, dtype=torch.int64), persistent=False
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.expert_bias is not None:
            self.expert_bias = self.expert_bias.float()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
            self.config.router_dtype
        ]
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.to(dtype), self.gate.weight.to(dtype)).float()
            scores = (
                logits.sigmoid() if self.config.score_function == "sigmoid" else logits.softmax(-1)
            )
            selection_scores = scores if self.expert_bias is None else scores + self.expert_bias
            indices = selection_scores.topk(self.config.num_experts_per_tok, dim=-1).indices
            # Bias affects selection only. Backpropagate through the original scores.
            weights = scores.gather(-1, indices)
            if self.config.num_experts_per_tok > 1:
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(
                    torch.finfo(scores.dtype).tiny
                )
            weights = weights * self.config.routed_scaling_factor
        if self.training and self.expert_bias is not None:
            with torch.no_grad():
                self.token_counts.add_(
                    torch.bincount(indices.reshape(-1), minlength=self.config.num_experts)
                )
        return indices, weights

    @torch.no_grad()
    def update_bias(self, update_rate: float = 1e-3, synchronize: bool = True) -> None:
        """Call once after an optimizer step, after all accumulated microbatches.

        In DDP every rank must call this, including ranks with no local valid tokens.
        No state changes occur during evaluation. Counts reset after the update.
        """
        if not math.isfinite(update_rate) or update_rate < 0:
            raise ValueError("update_rate must be finite and nonnegative")
        if self.expert_bias is None:
            return
        counts = self.token_counts.clone()
        if synchronize and dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        counts = counts.float()
        self.expert_bias.add_(update_rate * (counts.mean() - counts).sign())
        self.token_counts.zero_()


class SparseMoE(nn.Module):
    def __init__(self, config: IndicPrayogConfig):
        super().__init__()
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        # Concatenating shared experts into a wider SwiGLU gives the same summed output.
        self.shared_experts = (
            SwiGLU(config.hidden_size, config.moe_intermediate_size * config.num_shared_experts)
            if config.num_shared_experts
            else None
        )

    def forward(self, x: Tensor, token_mask: Tensor | None = None) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        valid_indices = None
        if token_mask is not None:
            valid_indices = token_mask.reshape(-1).nonzero(as_tuple=True)[0]
            flat = flat.index_select(0, valid_indices)
        selected, weights = self.router(flat)
        output = torch.zeros_like(flat)
        # Dispatch only selected tokens; no capacity limit and no dropped assignments.
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot = torch.where(selected == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_idx))
            contribution = (expert_output.float() * weights[token_idx, slot, None]).to(output.dtype)
            output = output.index_add(0, token_idx, contribution)
        if self.shared_experts is not None:
            output = output + self.shared_experts(flat)
        if valid_indices is not None:
            output = torch.zeros_like(x.reshape(-1, shape[-1])).index_copy(0, valid_indices, output)
        return output.reshape(shape)
