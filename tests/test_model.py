import copy
import math
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from model import IndicPrayogConfig, IndicPrayogForCausalLM
from model.layers import GroupedQueryAttention, RMSNorm, RotaryEmbedding
from model.moe import SparseMoE, TopKRouter
from model.parameter_count import parameter_counts

torch.set_num_threads(1)


def tiny_config(**overrides):
    values = dict(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        intermediate_size=48,
        moe_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
    )
    values.update(overrides)
    return IndicPrayogConfig(**values)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.config = tiny_config()
        self.model = IndicPrayogForCausalLM(self.config)
        self.ids = torch.randint(4, self.config.vocab_size, (2, 8))

    def test_forward_loss_backward_and_optimization(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        output = self.model(self.ids, labels=self.ids)
        self.assertEqual(output.logits.shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(output.loss))
        self.assertLess(abs(output.loss.item() - math.log(32)), 0.5)
        manual = F.cross_entropy(output.logits[:, :-1].reshape(-1, 32), self.ids[:, 1:].reshape(-1))
        torch.testing.assert_close(output.loss, manual)
        before = self.model.lm_head.weight.detach().clone()
        output.loss.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all(), name)
        router = self.model.layers[1].ffn.router
        self.assertGreater(router.gate.weight.grad.abs().sum().item(), 0)
        opt.step()
        self.model.update_router_bias()
        self.assertFalse(torch.equal(before, self.model.lm_head.weight))
        self.assertEqual(router.token_counts.sum().item(), 0)
        self.assertLess(self.model(self.ids, labels=self.ids).loss.item(), output.loss.item())

    def test_causality_future_tokens_cannot_change_prefix(self):
        self.model.eval()
        changed = self.ids.clone()
        changed[:, 4:] = (changed[:, 4:] + 7) % 32
        with torch.no_grad():
            original = self.model(self.ids).logits
            perturbed = self.model(changed).logits
        torch.testing.assert_close(original[:, :4], perturbed[:, :4], rtol=0, atol=0)

    def test_single_and_chunked_kv_cache_match_full_forward(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.ids).logits
            for chunks in ([1] * 8, [3, 2, 3]):
                outputs, past, offset = [], None, 0
                for length in chunks:
                    result = self.model(
                        self.ids[:, offset : offset + length], past_key_values=past, use_cache=True
                    )
                    past = result.past_key_values
                    offset += length
                    self.assertEqual(past[0][0].shape, (2, 2, offset, 8))
                    outputs.append(result.logits)
                torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-6)

    def test_left_and_right_padding_match_unpadded(self):
        self.model.eval()
        ids = self.ids[:1, :5]
        with torch.no_grad():
            expected = self.model(ids).logits
            for left in (True, False):
                padded = F.pad(ids, (3, 0) if left else (0, 3))
                mask = F.pad(torch.ones_like(ids), (3, 0) if left else (0, 3))
                actual = self.model(padded, attention_mask=mask).logits
                valid = actual[:, 3:] if left else actual[:, :5]
                torch.testing.assert_close(valid, expected, rtol=1e-5, atol=1e-6)
                loss = self.model(padded, attention_mask=mask, labels=padded).loss
                torch.testing.assert_close(loss, self.model(ids, labels=ids).loss)

    def test_padding_changes_do_not_leak_into_valid_logits(self):
        self.model.eval()
        mask = torch.ones_like(self.ids)
        mask[:, :3] = 0
        changed = self.ids.clone()
        changed[:, :3] = (changed[:, :3] + 11) % 32
        with torch.no_grad():
            first = self.model(self.ids, attention_mask=mask).logits[:, 3:]
            second = self.model(changed, attention_mask=mask).logits[:, 3:]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_cached_left_padded_batch_matches_full(self):
        self.model.eval()
        mask = torch.ones_like(self.ids)
        mask[0, :3] = 0
        mask[1, :1] = 0
        with torch.no_grad():
            expected = self.model(self.ids, attention_mask=mask).logits
            prefix = self.model(self.ids[:, :5], attention_mask=mask[:, :5], use_cache=True)
            actual = self.model(
                self.ids[:, 5:], attention_mask=mask, past_key_values=prefix.past_key_values
            ).logits
            torch.testing.assert_close(actual, expected[:, 5:], rtol=1e-5, atol=1e-6)

    def test_masked_loss_and_no_targets_are_finite(self):
        labels = self.ids.clone()
        labels[:, 4:] = -100
        output = self.model(self.ids, labels=labels)
        expected = F.cross_entropy(output.logits[:, :3].reshape(-1, 32), labels[:, 1:4].reshape(-1))
        torch.testing.assert_close(output.loss, expected)
        for kwargs in (
            {"labels": torch.full_like(self.ids, -100)},
            {"labels": self.ids, "attention_mask": torch.zeros_like(self.ids)},
        ):
            output = self.model(self.ids, **kwargs)
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertEqual(output.loss.item(), 0)
            output.loss.backward()

    def test_parameter_formula_matches_real_model_and_variants(self):
        for changes in (
            {},
            {"tie_word_embeddings": True},
            {"num_shared_experts": 0},
            {"num_shared_experts": 2, "first_k_dense_replace": 0},
            {"use_rmsnorm": False, "use_qk_norm": False},
            {"moe_router_enable_expert_bias": False},
        ):
            with self.subTest(changes=changes):
                config = replace(self.config, **changes)
                model = IndicPrayogForCausalLM(config)
                self.assertEqual(
                    sum(p.numel() for p in model.parameters()),
                    parameter_counts(config)["total_parameters"],
                )
                self.assertTrue(torch.isfinite(model(self.ids, labels=self.ids).loss))
        with torch.device("meta"):
            default_model = IndicPrayogForCausalLM()
        self.assertEqual(default_model.parameter_counts()["actual_parameters"], 502_903_424)

    def test_tied_embeddings_share_parameter_and_gradient(self):
        model = IndicPrayogForCausalLM(replace(self.config, tie_word_embeddings=True))
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)
        model(self.ids, labels=self.ids).loss.backward()
        self.assertIs(model.lm_head.weight.grad, model.embed_tokens.weight.grad)

    def test_bfloat16_checkpoint_and_cache(self):
        model = self.model.bfloat16().eval()
        with tempfile.TemporaryDirectory() as td:
            model.save_pretrained(td)
            restored = IndicPrayogForCausalLM.from_pretrained(td)
        self.assertEqual(restored.embed_tokens.weight.dtype, torch.bfloat16)
        with torch.no_grad():
            torch.testing.assert_close(restored(self.ids).logits, model(self.ids).logits)
            prefix = restored(self.ids[:, :4], use_cache=True)
            cached = restored(self.ids[:, 4:], past_key_values=prefix.past_key_values)
            torch.testing.assert_close(cached.logits, restored(self.ids).logits[:, 4:])
            invalid = tuple((k.float(), v.float()) for k, v in prefix.past_key_values)
            with self.assertRaises(ValueError):
                restored(self.ids[:, 4:], past_key_values=invalid)

    def test_save_load_roundtrip_includes_bias_state(self):
        self.model.eval()
        self.model.layers[1].ffn.router.expert_bias[0] = 0.5
        with tempfile.TemporaryDirectory() as td:
            self.model.save_pretrained(td)
            restored = IndicPrayogForCausalLM.from_pretrained(td)
        self.assertEqual(restored.config, self.model.config)
        torch.testing.assert_close(
            restored(self.ids).logits, self.model(self.ids).logits, rtol=0, atol=0
        )
        self.assertFalse(restored.training)

    def test_eval_does_not_mutate_router_state_and_disables_dropout(self):
        model = IndicPrayogForCausalLM(replace(self.config, attention_dropout=0.5)).eval()
        before = {name: buf.clone() for name, buf in model.named_buffers()}
        first, second = model(self.ids).logits, model(self.ids).logits
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, buf in model.named_buffers():
            torch.testing.assert_close(buf, before[name], rtol=0, atol=0)

    def test_bfloat16_and_autocast_forward_backward(self):
        for converted in (True, False):
            model = copy.deepcopy(self.model)
            if converted:
                model.bfloat16()
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not converted):
                result = model(self.ids, labels=self.ids)
            self.assertEqual(result.logits.dtype, torch.bfloat16)
            self.assertEqual(result.loss.dtype, torch.float32)
            result.loss.backward()
            self.assertTrue(torch.isfinite(result.loss))
            router = model.layers[1].ffn.router
            self.assertEqual(router.expert_bias.dtype, torch.float32)
            self.assertTrue(torch.isfinite(router.gate.weight.grad).all())

    def test_greedy_generation_matches_full_recomputation(self):
        self.model.eval()
        expected = self.ids[:, :3]
        with torch.no_grad():
            for _ in range(4):
                token = self.model(expected).logits[:, -1].argmax(-1, keepdim=True)
                expected = torch.cat((expected, token), -1)
        self.model.train()
        actual = self.model.generate(self.ids[:, :3], max_new_tokens=4)
        self.assertTrue(self.model.training)
        torch.testing.assert_close(actual, expected)

    def test_generation_left_padding_sampling_and_eos(self):
        self.model.eval()
        ids = self.ids[:1, :3]
        padded, mask = F.pad(ids, (2, 0)), F.pad(torch.ones_like(ids), (2, 0))
        expected = self.model.generate(ids, max_new_tokens=3)
        actual = self.model.generate(padded, attention_mask=mask, max_new_tokens=3)
        torch.testing.assert_close(actual[:, 2:], expected)
        sampled = self.model.generate(ids, max_new_tokens=3, do_sample=True, top_k=1)
        torch.testing.assert_close(sampled, expected)
        eos = self.model(ids).logits[:, -1].argmax(-1).item()
        self.assertEqual(self.model.generate(ids, max_new_tokens=5, eos_token_id=eos).shape[1], 4)

    def test_invalid_inputs_and_cache_rejected(self):
        cases = [
            {"input_ids": self.ids.float()},
            {"input_ids": self.ids[0]},
            {"input_ids": self.ids[:, :0]},
            {"attention_mask": torch.ones(2, 7)},
            {"attention_mask": torch.full_like(self.ids, 2)},
            {"position_ids": torch.full_like(self.ids, 64)},
            {"position_ids": torch.full_like(self.ids, -1)},
            {"position_ids": self.ids.float()},
            {"labels": self.ids[:, :2]},
            {"labels": self.ids, "use_cache": True},
            {"past_key_values": ()},
            {"input_ids": torch.ones(1, 65, dtype=torch.long)},
        ]
        for change in cases:
            kwargs = {"input_ids": self.ids, **change}
            with self.subTest(keys=list(change)), self.assertRaises(ValueError):
                self.model(**kwargs)
        cache = self.model(self.ids, use_cache=True).past_key_values
        bad_cache = ((cache[0][0][..., :-1, :], cache[0][1][..., :-1, :]), *cache[1:])
        with self.assertRaises(ValueError):
            self.model(self.ids, past_key_values=bad_cache)


class RoutingAndLayerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = tiny_config()

    def test_gqa_matches_manual_attention(self):
        attention = GroupedQueryAttention(self.config).eval()
        x = torch.randn(2, 5, 32)
        # At position zero RoPE is the identity, isolating GQA/mask/scale behavior.
        positions = torch.zeros(2, 5, dtype=torch.long)
        actual, _ = attention(x, positions, attention_mask=None, is_causal=True)
        q = attention.q_norm(attention.q_proj(x).view(2, 5, 4, 8).transpose(1, 2))
        k = attention.k_norm(attention.k_proj(x).view(2, 5, 2, 8).transpose(1, 2))
        v = attention.v_proj(x).view(2, 5, 2, 8).transpose(1, 2)
        per_head = []
        causal = torch.ones(5, 5, dtype=torch.bool).tril()
        for head in range(4):
            kv_head = head // 2
            scores = q[:, head] @ k[:, kv_head].transpose(-2, -1) / math.sqrt(8)
            probs = scores.masked_fill(~causal, -torch.inf).softmax(-1)
            per_head.append(probs @ v[:, kv_head])
        expected = attention.o_proj(torch.stack(per_head, 2).reshape(2, 5, 32))
        torch.testing.assert_close(actual, expected)

    def test_top_one_keeps_router_gradient(self):
        router = TopKRouter(replace(self.config, num_experts_per_tok=1))
        _, weights = router(torch.randn(8, 32))
        weights.sum().backward()
        self.assertGreater(router.gate.weight.grad.abs().sum().item(), 0)

    def test_sparse_dispatch_matches_dense_reference_and_gradients(self):
        sparse = SparseMoE(self.config).eval()
        dense = copy.deepcopy(sparse)
        x = torch.randn(2, 5, 32, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        actual = sparse(x)
        flat = x_ref.reshape(-1, 32)
        selected, weights = dense.router(flat)
        mixture = torch.zeros(flat.shape[0], 4).scatter_add(1, selected, weights)
        expected = sum(
            expert(flat).float() * mixture[:, i : i + 1] for i, expert in enumerate(dense.experts)
        )
        expected = (expected + dense.shared_experts(flat)).reshape_as(x)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        for (name, p), (_, ref) in zip(sparse.named_parameters(), dense.named_parameters()):
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            reference_grad = ref.grad if ref.grad is not None else torch.zeros_like(ref)
            torch.testing.assert_close(grad, reference_grad, rtol=1e-4, atol=1e-6, msg=name)

    def test_only_selected_experts_execute(self):
        moe = SparseMoE(self.config).eval()
        with torch.no_grad():
            moe.router.gate.weight.zero_()
            moe.router.expert_bias.copy_(torch.tensor([4.0, 3.0, 0.0, -1.0]))
        calls = []
        handles = [
            expert.register_forward_hook(lambda _m, _i, _o, idx=i: calls.append(idx))
            for i, expert in enumerate(moe.experts)
        ]
        moe(torch.randn(2, 4, 32))
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [0, 1])

    def test_bias_changes_selection_but_not_weights(self):
        router = TopKRouter(self.config).eval()
        x = torch.randn(6, 32)
        with torch.no_grad():
            router.expert_bias.copy_(torch.tensor([-100.0, -100.0, 100.0, 100.0]))
        indices, weights = router(x)
        self.assertTrue(((indices == 2) | (indices == 3)).all())
        scores = F.linear(x, router.gate.weight).sigmoid().gather(-1, indices)
        expected = scores / scores.sum(-1, keepdim=True) * self.config.routed_scaling_factor
        torch.testing.assert_close(weights, expected)

    def test_router_counts_padding_accumulation_and_bias_update(self):
        moe = SparseMoE(self.config).train()
        x = torch.randn(2, 4, 32)
        mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
        output = moe(x, mask)
        self.assertEqual(moe.router.token_counts.sum().item(), 10)
        self.assertEqual(output[~mask].abs().sum().item(), 0)
        moe(x, mask)
        self.assertEqual(moe.router.token_counts.sum().item(), 20)
        with torch.no_grad():
            moe.router.token_counts.copy_(torch.tensor([20, 0, 0, 0]))
        moe.router.update_bias(0.01)
        torch.testing.assert_close(moe.router.expert_bias, torch.tensor([-0.01, 0.01, 0.01, 0.01]))
        self.assertEqual(moe.router.token_counts.sum().item(), 0)
        before = moe.router.expert_bias.clone()
        moe.router.update_bias()
        torch.testing.assert_close(moe.router.expert_bias, before)

    def test_distributed_update_sums_counts_before_bias_update(self):
        router = TopKRouter(self.config)
        router.token_counts.copy_(torch.tensor([4, 0, 0, 0]))

        def mock_all_reduce(counts, op):
            counts.add_(torch.tensor([0, 4, 4, 4]))

        with (
            patch("model.moe.dist.is_initialized", return_value=True),
            patch("model.moe.dist.all_reduce", side_effect=mock_all_reduce) as reduce,
        ):
            router.update_bias(0.1)
        reduce.assert_called_once()
        torch.testing.assert_close(router.expert_bias, torch.zeros(4))

    def test_softmax_router_and_optional_bias(self):
        for score in ("sigmoid", "softmax"):
            router = TopKRouter(
                replace(self.config, score_function=score, moe_router_enable_expert_bias=False)
            )
            indices, weights = router(torch.randn(5, 32))
            self.assertEqual(indices.shape, (5, 2))
            torch.testing.assert_close(weights.sum(-1), torch.full((5,), 2.5))
            self.assertTrue((weights >= 0).all())
            weights[:, 0].sum().backward()
            self.assertTrue(torch.isfinite(router.gate.weight.grad).all())
            router.update_bias()
            self.assertIsNone(router.expert_bias)

    def test_rmsnorm_matches_fp32_reference(self):
        norm = RMSNorm(32)
        x = (torch.randn(4, 32) * 100).bfloat16()
        expected = (
            x.float() / torch.sqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
        ).bfloat16()
        torch.testing.assert_close(norm(x), expected)
        self.assertEqual(norm(x).dtype, x.dtype)

    def test_rope_preserves_norm_and_relative_position(self):
        rope = RotaryEmbedding(8, 8_000_000.0)
        q = torch.randn(1, 4, 5, 8)
        k = torch.randn(1, 2, 5, 8)
        positions = torch.arange(5)[None]
        rotated_q, rotated_k = rope(q, k, positions)
        torch.testing.assert_close(rotated_q.norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(rotated_k.norm(dim=-1), k.norm(dim=-1))
        shifted_q, shifted_k = rope(q, k, positions + 100_000)
        original_dot = rotated_q[:, :2] @ rotated_k.transpose(-2, -1)
        shifted_dot = shifted_q[:, :2] @ shifted_k.transpose(-2, -1)
        torch.testing.assert_close(original_dot, shifted_dot, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
