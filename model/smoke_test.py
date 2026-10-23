"""Forward/backward smoke run: python -m model.smoke_test [--full]."""

import argparse
import json

import torch

from .config import IndicPrayogConfig
from .modeling import IndicPrayogForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="instantiate the full 502.9M model")
    parser.add_argument("--device", default="cpu", help="e.g. cpu or cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = (
        IndicPrayogConfig()
        if args.full
        else IndicPrayogConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_experts=4,
        )
    )
    # Construct directly in the requested dtype/device, avoiding a full fp32 copy.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, args.dtype))
        with torch.device(args.device):
            model = IndicPrayogForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    ids = torch.randint(4, config.vocab_size, (1, 8), device=args.device)
    output = model(ids, labels=ids)
    if not torch.isfinite(output.loss):
        raise RuntimeError("non-finite loss")
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient: {name}")
    model.update_router_bias()
    counts = model.parameter_counts()
    if counts["actual_parameters"] != counts["total_parameters"]:
        raise RuntimeError("parameter count does not match config")
    print(
        json.dumps(
            {
                "full_model": args.full,
                "device": args.device,
                "dtype": args.dtype,
                "loss": output.loss.item(),
                "logits_shape": list(output.logits.shape),
                "forward_backward": "passed",
                **counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
