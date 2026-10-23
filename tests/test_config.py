import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from model import IndicPrayogConfig
from model.parameter_count import parameter_counts


class ConfigTests(unittest.TestCase):
    def test_defaults_and_layer_boundaries(self):
        c = IndicPrayogConfig()
        self.assertEqual((c.num_dense_layers, c.num_moe_layers), (1, 18))
        self.assertEqual((c.experts_per_moe_layer, c.active_experts_per_token), (47, 3))
        self.assertEqual(c.queries_per_kv_head, 6)
        self.assertFalse(c.is_moe_layer(0))
        self.assertTrue(c.is_moe_layer(18))
        for idx in (-1, 19, 0.5, True):
            with self.assertRaises(IndexError):
                c.is_moe_layer(idx)

    def test_roundtrip_and_immutable_config(self):
        c = IndicPrayogConfig()
        with tempfile.TemporaryDirectory() as td:
            path = c.to_json(Path(td) / "nested" / "config.json")
            self.assertEqual(c, IndicPrayogConfig.from_json(path))
            self.assertEqual(c.to_dict(), json.loads(path.read_text()))
        with self.assertRaises(FrozenInstanceError):
            c.hidden_size = 123
        self.assertEqual(replace(c, num_experts=32).num_experts, 32)

    def test_invalid_configs_rejected(self):
        cases = [
            {"vocab_size": 65537},
            {"hidden_size": 0},
            {"head_dim": 63, "hidden_size": 756},
            {"num_key_value_heads": 5},
            {"first_k_dense_replace": 19},
            {"num_experts_per_tok": 47},
            {"num_experts_per_tok": 0},
            {"num_shared_experts": -1},
            {"num_experts": 0},
            {"attention_dropout": 1},
            {"attention_dropout": True},
            {"hidden_act": "relu"},
            {"score_function": "other"},
            {"router_dtype": "fp64"},
            {"model_type": "other"},
            {"use_qk_norm": 1},
        ]
        for field in (
            "num_hidden_layers",
            "num_experts",
            "num_experts_per_tok",
            "first_k_dense_replace",
            "hidden_size",
            "vocab_size",
        ):
            cases.extend([{field: 1.5}, {field: True}])
        for field in ("rope_theta", "rms_norm_eps", "routed_scaling_factor", "initializer_range"):
            cases.extend({field: v} for v in (float("nan"), float("inf"), -1, 0, True, "1"))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                IndicPrayogConfig(**case)
        for data in ({"typo": 1}, [], None):
            with self.assertRaises(ValueError):
                IndicPrayogConfig.from_dict(data)

    def test_config_import_without_site_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "from model import IndicPrayogConfig; IndicPrayogConfig()",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_default_parameter_budget(self):
        counts = parameter_counts(IndicPrayogConfig())
        self.assertEqual(counts["total_parameters"], 502_903_424)
        self.assertEqual(counts["active_transformer_parameters_per_token"], 54_243_968)
        self.assertEqual(counts["active_parameters_including_lm_head"], 103_395_968)
        self.assertEqual(counts["expert_bias_buffer_values"], 828)
