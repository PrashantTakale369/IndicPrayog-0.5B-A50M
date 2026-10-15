"""Exact parameter counts for this implementation; no torch dependency.

Run: python -m model.parameter_count
"""

import json

from .config import IndicPrayogConfig


def parameter_counts(config: IndicPrayogConfig) -> dict[str, int]:
    c = config
    h, d, n = c.hidden_size, c.head_dim, c.num_hidden_layers
    embedding = c.vocab_size * h
    attention = n * (2 * h * c.num_attention_heads * d + 2 * h * c.num_key_value_heads * d)
    norms = (n * 2 * h + h) if c.use_rmsnorm else 0
    norms += n * 2 * d if c.use_qk_norm else 0
    dense = c.num_dense_layers * 3 * h * c.intermediate_size
    router = c.num_moe_layers * h * c.num_experts
    expert = 3 * h * c.moe_intermediate_size
    all_experts = c.num_moe_layers * c.experts_per_moe_layer * expert
    active_experts = c.num_moe_layers * c.active_experts_per_token * expert
    always_active = attention + norms + dense + router
    transformer = always_active + all_experts
    active = always_active + active_experts
    return {
        "total_parameters": transformer + embedding * (1 if c.tie_word_embeddings else 2),
        "transformer_parameters": transformer,
        "active_transformer_parameters_per_token": active,
        "active_parameters_including_lm_head": active + embedding,
        "input_embedding_parameters": embedding,
        "lm_head_parameters": embedding,
        "input_embedding_values_looked_up_per_token": h,
        "attention_parameters": attention,
        "normalization_parameters": norms,
        "dense_ffn_parameters": dense,
        "router_parameters": router,
        "routed_and_shared_expert_parameters": all_experts,
        "expert_bias_buffer_values": c.num_moe_layers * c.num_experts
        if c.moe_router_enable_expert_bias
        else 0,
    }


if __name__ == "__main__":
    print(json.dumps(parameter_counts(IndicPrayogConfig()), indent=2))
