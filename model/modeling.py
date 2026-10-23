"""From-scratch decoder-only sparse MoE language model, using PyTorch ops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import IndicPrayogConfig
from .layers import GroupedQueryAttention, KVCache, RMSNorm, SwiGLU
from .moe import SparseMoE
from .parameter_count import parameter_counts

PastKeyValues = tuple[KVCache, ...]


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: PastKeyValues | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: IndicPrayogConfig, layer_idx: int):
        super().__init__()
        self.input_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.use_rmsnorm
            else nn.Identity()
        )
        self.post_attention_norm = (
            RMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.use_rmsnorm
            else nn.Identity()
        )
        self.attention = GroupedQueryAttention(config)
        self.ffn = (
            SparseMoE(config)
            if config.is_moe_layer(layer_idx)
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )

    def forward(
        self,
        x: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        is_causal: bool,
        token_mask: Tensor | None,
        past_key_value: KVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, KVCache | None]:
        attended, present = self.attention(
            self.input_norm(x),
            position_ids,
            attention_mask,
            is_causal,
            past_key_value,
            use_cache,
        )
        x = x + attended
        normalized = self.post_attention_norm(x)
        ffn_output = (
            self.ffn(normalized, token_mask)
            if isinstance(self.ffn, SparseMoE)
            else self.ffn(normalized)
        )
        return x + ffn_output, present


class IndicPrayogForCausalLM(nn.Module):
    def __init__(self, config: IndicPrayogConfig | None = None):
        super().__init__()
        self.config = config if config is not None else IndicPrayogConfig()
        c = self.config
        c.validate()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps) if c.use_rmsnorm else nn.Identity()
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Scale residual branches for depth while retaining the config's base init.
        residual_std = c.initializer_range / math.sqrt(2 * c.num_hidden_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(("o_proj", "down_proj")):
                nn.init.normal_(module.weight, std=residual_std)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _past_length(self, past: PastKeyValues | None, batch: int, device: torch.device) -> int:
        if past is None:
            return 0
        if not isinstance(past, (tuple, list)) or len(past) != len(self.layers):
            raise ValueError("past_key_values must contain one (key, value) pair per layer")
        length = None
        for entry in past:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("each cache entry must be a (key, value) pair")
            key, value = entry
            expected = (batch, self.config.num_key_value_heads)
            if (
                not isinstance(key, Tensor)
                or not isinstance(value, Tensor)
                or key.ndim != 4
                or key.shape != value.shape
                or tuple(key.shape[:2]) != expected
                or key.shape[-1] != self.config.head_dim
            ):
                raise ValueError(
                    "cache tensors must have shape [batch, kv_heads, past_length, head_dim]"
                )
            if key.device != device or value.device != device or key.dtype != value.dtype:
                raise ValueError("cache tensors must match input device and each other's dtype")
            if length is not None and key.shape[-2] != length:
                raise ValueError("all layers must have the same cache length")
            length = key.shape[-2]
        return length

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """labels are UNshifted token IDs, normally input_ids.clone().

        A 2D attention_mask uses 1/True for valid tokens and 0/False for padding.
        With a cache it covers the full prefix plus current tokens. Positions are
        inferred from this mask, or can be supplied for the current tokens only.
        EOS separates documents; attention is causal across packed documents.
        """
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be an int32/int64 tensor of shape [batch, sequence]")
        batch, length = input_ids.shape
        if batch == 0 or length == 0:
            raise ValueError("input_ids must have a nonempty batch and sequence")
        past_length = self._past_length(past_key_values, batch, input_ids.device)
        total_length = past_length + length
        if total_length > self.config.max_position_embeddings:
            raise ValueError("sequence plus cache exceeds max_position_embeddings")
        if labels is not None:
            if past_key_values is not None or use_cache:
                raise ValueError("labels cannot be combined with KV caching")
            if labels.shape != input_ids.shape or labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("labels must be integer IDs with the same shape as input_ids")
            if labels.device != input_ids.device or length < 2:
                raise ValueError("labels must match input device and contain at least two tokens")

        token_mask = None
        if attention_mask is not None:
            if (
                attention_mask.shape != (batch, total_length)
                or attention_mask.device != input_ids.device
            ):
                raise ValueError(
                    "attention_mask must cover [batch, past_length + sequence] on the input device"
                )
            if not ((attention_mask == 0) | (attention_mask == 1)).all():
                raise ValueError("attention_mask must contain only 0/1 values")
            attention_mask = attention_mask.bool()
            token_mask = attention_mask[:, -length:]

        if position_ids is None:
            if attention_mask is not None:
                position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -length:]
            else:
                position_ids = torch.arange(past_length, total_length, device=input_ids.device)[
                    None, :
                ]
        if (
            position_ids.shape not in ((batch, length), (1, length))
            or position_ids.dtype not in (torch.int32, torch.int64)
            or position_ids.device != input_ids.device
        ):
            raise ValueError(
                "position_ids must be integer [batch or 1, sequence] on the input device"
            )
        if (position_ids < 0).any() or (position_ids >= self.config.max_position_embeddings).any():
            raise ValueError("position_ids are outside the configured context range")

        # SDPA's non-square is_causal mask aligns at the upper left. Cached
        # chunks need an explicit offset so each new query can see its prefix.
        is_causal = past_length == 0 and attention_mask is None
        causal_mask = None
        if attention_mask is not None or (past_length > 0 and length > 1):
            q_positions = torch.arange(past_length, total_length, device=input_ids.device)
            k_positions = torch.arange(total_length, device=input_ids.device)
            causal_mask = (k_positions[None, :] <= q_positions[:, None])[None, None, :, :]
            if attention_mask is not None:
                causal_mask = causal_mask & attention_mask[:, None, None, :]

        hidden = self.embed_tokens(input_ids)
        present = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            hidden, layer_cache = layer(
                hidden,
                position_ids,
                causal_mask,
                is_causal,
                token_mask,
                None if past_key_values is None else past_key_values[i],
                use_cache,
            )
            if present is not None:
                present.append(layer_cache)
        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            targets = labels[:, 1:].long().clone()
            if attention_mask is not None:
                # Do not learn pad->first-token or last-token->pad transitions.
                valid_pairs = attention_mask[:, :-1] & attention_mask[:, 1:]
                targets.masked_fill_(~valid_pairs, -100)
            loss = (
                F.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                if (targets != -100).any()
                else logits.float().sum() * 0.0
            )
        return CausalLMOutput(logits, loss, tuple(present) if present is not None else None)

    @torch.no_grad()
    def update_router_bias(self, update_rate: float = 1e-3, synchronize: bool = True) -> None:
        """Apply once per optimizer step, after accumulating every microbatch."""
        for layer in self.layers:
            if isinstance(layer.ffn, SparseMoE):
                layer.ffn.router.update_bias(update_rate, synchronize)

    def parameter_counts(self) -> dict[str, int]:
        counts = parameter_counts(self.config)
        counts["actual_parameters"] = sum(p.numel() for p in self.parameters())
        return counts

    def save_pretrained(self, directory: str | Path) -> None:
        """Save this package's config and weights (not a Transformers checkpoint)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.to_json(directory / "config.json")
        torch.save(self.state_dict(), directory / "model.pt")

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "IndicPrayogForCausalLM":
        directory = Path(directory)
        state = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model = cls(IndicPrayogConfig.from_json(directory / "config.json")).to(
            device=map_location,
            dtype=state["embed_tokens.weight"].dtype,
        )
        model.load_state_dict(state)
        return model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
        attention_mask: Tensor | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int = 0,
    ) -> Tensor:
        """Greedy or temperature/top-k sampling, with unpadded or left-padded prompts."""
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (
            type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be between 1 and vocab_size")
        for name, value in (("pad_token_id", pad_token_id), ("eos_token_id", eos_token_id)):
            if value is not None and (
                type(value) is not int or not 0 <= value < self.config.vocab_size
            ):
                raise ValueError(f"{name} must be a valid vocabulary ID")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [nonempty batch, nonempty sequence]")
        if input_ids.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("requested generation exceeds max_position_embeddings")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape or not (attention_mask[:, -1] == 1).all():
                raise ValueError("generation requires a valid final prompt token; use left padding")
            if (attention_mask[:, :-1] > attention_mask[:, 1:]).any():
                raise ValueError("generation supports left padding only")
        if max_new_tokens == 0:
            return input_ids
        was_training = self.training
        self.eval()
        try:
            sequence, current, past = input_ids, input_ids, None
            mask = attention_mask
            finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
            for _ in range(max_new_tokens):
                output = self(current, attention_mask=mask, past_key_values=past, use_cache=True)
                scores = output.logits[:, -1].float() / temperature
                if top_k is not None:
                    threshold = scores.topk(top_k, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                next_token = (
                    torch.multinomial(scores.softmax(-1), 1)
                    if do_sample
                    else scores.argmax(-1, keepdim=True)
                )
                next_token = next_token.masked_fill(finished[:, None], pad_token_id)
                sequence = torch.cat((sequence, next_token), dim=-1)
                if mask is not None:
                    mask = torch.cat((mask, (~finished).to(mask.dtype)[:, None]), dim=-1)
                if eos_token_id is not None:
                    finished |= next_token[:, 0] == eos_token_id
                    if finished.all():
                        break
                current, past = next_token, output.past_key_values
            return sequence
        finally:
            self.train(was_training)
