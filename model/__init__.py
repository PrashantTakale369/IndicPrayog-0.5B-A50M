"""Scratch IndicPrayog MoE. Importing the config alone does not require PyTorch."""

from .config import IndicPrayogConfig

__all__ = ["IndicPrayogConfig", "IndicPrayogForCausalLM", "CausalLMOutput"]


def __getattr__(name):
    if name in ("IndicPrayogForCausalLM", "CausalLMOutput"):
        from .modeling import CausalLMOutput, IndicPrayogForCausalLM

        return {"IndicPrayogForCausalLM": IndicPrayogForCausalLM, "CausalLMOutput": CausalLMOutput}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
