"""IndicPrayog-0.5B-A50M architecture config. Stdlib only, no torch/transformers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_TYPE = "sarvam_moe"

SUPPORTED_ACTIVATIONS = ("silu",)
SUPPORTED_SCORE_FUNCTIONS = ("sigmoid", "softmax")
SUPPORTED_ROUTER_DTYPES = ("fp32", "bf16", "fp16")

# Packed dataset (dataset/final_packed/*.bin) stores token ids as uint16.
UINT16_VOCAB_LIMIT = 65536


@dataclass(frozen=True)
class IndicPrayogConfig:
    model_type: str = MODEL_TYPE

    vocab_size: int = 64000
    hidden_size: int = 768
    num_hidden_layers: int = 19

    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 64

    max_position_embeddings: int = 131072
    rope_theta: float = 8000000.0

    intermediate_size: int = 1536
    moe_intermediate_size: int = 192

    num_experts: int = 46
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    routed_scaling_factor: float = 2.5

    score_function: str = "sigmoid"
    router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True

    hidden_act: str = "silu"
    use_rmsnorm: bool = True
    use_qk_norm: bool = True
    rms_norm_eps: float = 1e-6

    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        # Annotations do not enforce runtime types (including after JSON loading).
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "moe_intermediate_size",
            "max_position_embeddings",
            "num_experts",
            "num_experts_per_tok",
            "num_shared_experts",
            "first_k_dense_replace",
        )
        for name in integer_fields:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer, got {getattr(self, name)!r}")
        for name in (
            "use_rmsnorm",
            "use_qk_norm",
            "tie_word_embeddings",
            "moe_router_enable_expert_bias",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        self._require_choice("model_type", self.model_type, (MODEL_TYPE,))
        self._require_positive("vocab_size", self.vocab_size)
        self._require_positive("hidden_size", self.hidden_size)
        self._require_positive("num_hidden_layers", self.num_hidden_layers)
        self._require_positive("num_attention_heads", self.num_attention_heads)
        self._require_positive("num_key_value_heads", self.num_key_value_heads)
        self._require_positive("head_dim", self.head_dim)
        self._require_positive("intermediate_size", self.intermediate_size)
        self._require_positive("moe_intermediate_size", self.moe_intermediate_size)
        self._require_positive("max_position_embeddings", self.max_position_embeddings)
        self._require_positive("rope_theta", self.rope_theta)
        self._require_positive("rms_norm_eps", self.rms_norm_eps)
        self._require_positive("routed_scaling_factor", self.routed_scaling_factor)
        self._require_positive("initializer_range", self.initializer_range)

        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary position embeddings")

        if self.vocab_size > UINT16_VOCAB_LIMIT:
            raise ValueError(
                f"vocab_size={self.vocab_size} exceeds {UINT16_VOCAB_LIMIT}: the packed "
                "dataset stores token ids as uint16, so a larger vocab would be "
                "silently truncated when the .bin files are read back"
            )

        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                f"num_attention_heads * head_dim must equal hidden_size: got "
                f"{self.num_attention_heads} * {self.head_dim} = "
                f"{self.num_attention_heads * self.head_dim}, expected {self.hidden_size}"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads}) so each KV head serves "
                "an equal group of query heads"
            )

        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"num_key_value_heads ({self.num_key_value_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        if not 0 <= self.first_k_dense_replace < self.num_hidden_layers:
            raise ValueError(
                f"first_k_dense_replace ({self.first_k_dense_replace}) must be in "
                f"[0, num_hidden_layers={self.num_hidden_layers}): the model needs at "
                "least one MoE layer"
            )

        if self.num_experts < 1:
            raise ValueError(f"num_experts must be at least 1, got {self.num_experts}")

        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) must be in "
                f"[1, num_experts={self.num_experts}]"
            )

        if self.num_shared_experts < 0:
            raise ValueError(
                f"num_shared_experts cannot be negative, got {self.num_shared_experts}"
            )

        if (
            type(self.attention_dropout) not in (float, int)
            or not math.isfinite(self.attention_dropout)
            or not 0.0 <= self.attention_dropout < 1.0
        ):
            raise ValueError(
                f"attention_dropout must be in [0.0, 1.0), got {self.attention_dropout}"
            )

        self._require_choice("hidden_act", self.hidden_act, SUPPORTED_ACTIVATIONS)
        self._require_choice("score_function", self.score_function, SUPPORTED_SCORE_FUNCTIONS)
        self._require_choice("router_dtype", self.router_dtype, SUPPORTED_ROUTER_DTYPES)

    @staticmethod
    def _require_positive(name: str, value: float) -> None:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    @staticmethod
    def _require_choice(name: str, value: Any, allowed: tuple[Any, ...]) -> None:
        if value not in allowed:
            raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")

    # -- derived values --------------------------------------------------

    @property
    def num_dense_layers(self) -> int:
        return self.first_k_dense_replace

    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.first_k_dense_replace

    @property
    def experts_per_moe_layer(self) -> int:
        return self.num_experts + self.num_shared_experts

    @property
    def active_experts_per_token(self) -> int:
        return self.num_experts_per_tok + self.num_shared_experts

    @property
    def queries_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def is_moe_layer(self, layer_idx: int) -> bool:
        if type(layer_idx) is not int or not 0 <= layer_idx < self.num_hidden_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for {self.num_hidden_layers} layers"
            )
        return layer_idx >= self.first_k_dense_replace

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if not key.startswith("_")}

    def to_json(self, path: str | Path, indent: int = 2) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=indent, allow_nan=False) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "IndicPrayogConfig":
        if not isinstance(values, dict):
            raise ValueError("config must be a JSON object")
        known = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}. Known keys: {sorted(known)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "IndicPrayogConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
