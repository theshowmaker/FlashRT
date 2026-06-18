import json

import numpy as np
import pytest
import torch

from flash_rt.core.utils.dvt2_policy import (
    PROFILE_DVT2_0605,
    DVT2Profile,
    clamp_h10w_dvt2_actions,
    exist_prediction_torch,
    exist_prediction_numpy,
    h10w_dual_action_joint_bounds,
    make_stage_fusion_tokens_numpy,
    make_stage_fusion_tokens_torch,
    normalize_stage,
    resolve_policy_profile,
    stage_logits_numpy,
    stage_logits_torch,
    task_category_from_prompt,
)


def _write_train_config(path, model_overrides=None):
    model = {
        "use_stage_prediction": True,
        "use_stage_fusion": True,
        "use_stage_tracking": True,
        "use_exist_prediction": True,
        "use_exist_mlp": True,
        "use_stage_mlp": True,
        "task_categories": ["pick", "place", "give"],
        "category_num_stages": [5, 5, 6],
        "max_num_stages": 6,
        "stage_fusion_num_tokens": 4,
    }
    if model_overrides:
        model.update(model_overrides)
    path.mkdir(parents=True, exist_ok=True)
    (path / "train_config_full.json").write_text(json.dumps({"model": model}), encoding="utf-8")


def _fake_weights(dim=8, sub_dim=4, task_dim=2, hidden=5):
    gen = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return {
        "stage_task_embed": rand(3, task_dim),
        "stage_mlp_1_w": rand(dim + task_dim, hidden),
        "stage_mlp_1_b": rand(hidden),
        "stage_mlp_2_w": rand(hidden, 6),
        "stage_mlp_2_b": rand(6),
        "exist_mlp_1_w": rand(dim * 2, hidden),
        "exist_mlp_1_b": rand(hidden),
        "exist_mlp_2_w": rand(hidden, 1),
        "exist_mlp_2_b": rand(1),
        "stage_class_embeddings": rand(6, sub_dim),
        "task_stage_embeddings": rand(16, sub_dim),
        "gate_sincos_w": rand(dim + sub_dim * 2, sub_dim),
        "gate_sincos_b": rand(sub_dim),
        "gate_task_stage_w": rand(dim + sub_dim * 2, sub_dim),
        "gate_task_stage_b": rand(sub_dim),
        "gate_task_w": rand(dim + sub_dim * 2, dim),
        "gate_task_b": rand(dim),
        "fusion_layer1_w": rand(dim + sub_dim * 2, dim * 2),
        "fusion_layer1_b": rand(dim * 2),
        "fusion_layer2_w": rand(dim * 2, dim),
        "fusion_layer2_b": rand(dim),
        "stage_projection_w": rand(sub_dim * 2, dim),
        "stage_projection_b": rand(dim),
    }


def test_resolve_policy_profile_auto_detects_dvt2_train_config(tmp_path):
    _write_train_config(tmp_path)
    profile = resolve_policy_profile("auto", tmp_path)
    assert profile is not None
    assert profile.task_categories == ("pick", "place", "give")
    assert profile.category_num_stages == (5, 5, 6)
    assert profile.stage_fusion_num_tokens == 4


def test_resolve_policy_profile_auto_ignores_non_dvt2_config(tmp_path):
    _write_train_config(tmp_path, {"use_stage_fusion": False})
    assert resolve_policy_profile("auto", tmp_path) is None
    assert resolve_policy_profile("none", tmp_path) is None
    assert resolve_policy_profile(PROFILE_DVT2_0605, tmp_path) is not None


def test_task_category_and_stage_normalization_match_openpi_defaults():
    profile = DVT2Profile()
    assert task_category_from_prompt("pick the cube", profile) == 0
    assert task_category_from_prompt("place block in bin", profile) == 1
    assert task_category_from_prompt("give it to me", profile) == 2
    assert task_category_from_prompt("unknown verb", profile) == 0
    assert normalize_stage(2, 0, profile) == pytest.approx(0.5)
    assert normalize_stage(5, 2, profile) == pytest.approx(1.0)
    assert normalize_stage(-1, 2, profile) == -1


def test_stage_logits_are_task_masked_and_finite_for_valid_stages():
    weights = _fake_weights()
    prompt = torch.randn(8)
    logits = stage_logits_torch(prompt, 0, weights, DVT2Profile())
    assert logits.shape == (6,)
    assert torch.isfinite(logits[:5]).all()
    assert torch.isneginf(logits[5])
    give_logits = stage_logits_torch(prompt, 2, weights, DVT2Profile())
    assert torch.isfinite(give_logits).all()


def test_stage_fusion_tokens_and_exist_mlp_shapes():
    weights = _fake_weights()
    prompt_embeds = torch.randn(7, 8, dtype=torch.bfloat16)
    tokens = make_stage_fusion_tokens_torch(prompt_embeds, 2, 5, weights, DVT2Profile())
    assert tokens.shape == (4, 8)
    assert tokens.dtype == torch.bfloat16
    exist, prob = exist_prediction_torch(torch.randn(8), torch.randn(8), weights)
    assert exist.shape == ()
    assert prob.shape == ()
    assert int(exist.item()) in (0, 1)
    assert 0.0 <= float(prob.item()) <= 1.0


def test_numpy_dvt2_heads_match_torch_helpers():
    weights_t = _fake_weights()
    weights_np = {k: v.numpy() for k, v in weights_t.items()}
    prompt = torch.randn(7, 8, dtype=torch.bfloat16)
    prompt_np = prompt.float().numpy().astype(np.float16)

    pooled = torch.randn(8)
    logits_t = stage_logits_torch(pooled, 1, weights_t, DVT2Profile())
    logits_np = stage_logits_numpy(pooled.numpy(), 1, weights_np, DVT2Profile())
    assert logits_np.shape == (6,)
    assert np.isneginf(logits_np[5])
    np.testing.assert_allclose(logits_np[:5], logits_t.numpy()[:5], rtol=1e-5, atol=1e-5)

    tokens_t = make_stage_fusion_tokens_torch(prompt, 2, 5, weights_t, DVT2Profile())
    tokens_np = make_stage_fusion_tokens_numpy(prompt_np, 2, 5, weights_np, DVT2Profile())
    assert tokens_np.shape == (4, 8)
    np.testing.assert_allclose(tokens_np.astype(np.float32), tokens_t.float().numpy(), rtol=5e-3, atol=5e-3)

    base = torch.randn(8)
    prompt_pooled = torch.randn(8)
    exist_t, prob_t = exist_prediction_torch(base, prompt_pooled, weights_t)
    exist_np, prob_np = exist_prediction_numpy(base.numpy(), prompt_pooled.numpy(), weights_np)
    assert int(exist_np.item()) == int(exist_t.item())
    np.testing.assert_allclose(prob_np, prob_t.numpy(), rtol=1e-6, atol=1e-6)


def test_h10w_dvt2_clamp_preserves_gripper_columns():
    lower, upper = h10w_dual_action_joint_bounds()
    actions = np.full((2, 16), 999.0, dtype=np.float32)
    actions[:, 7] = -123.0
    actions[:, 15] = 456.0
    clamped = clamp_h10w_dvt2_actions(actions)
    np.testing.assert_allclose(clamped[:, :7], np.broadcast_to(upper[:7], (2, 7)))
    np.testing.assert_allclose(clamped[:, 8:15], np.broadcast_to(upper[8:15], (2, 7)))
    np.testing.assert_allclose(clamped[:, 7], actions[:, 7])
    np.testing.assert_allclose(clamped[:, 15], actions[:, 15])

    actions = np.full((1, 16), -999.0, dtype=np.float32)
    actions[:, 7] = 0.25
    actions[:, 15] = 0.75
    clamped = clamp_h10w_dvt2_actions(actions)
    np.testing.assert_allclose(clamped[:, :7], np.broadcast_to(lower[:7], (1, 7)))
    np.testing.assert_allclose(clamped[:, 8:15], np.broadcast_to(lower[8:15], (1, 7)))
    np.testing.assert_allclose(clamped[:, 7], actions[:, 7])
    np.testing.assert_allclose(clamped[:, 15], actions[:, 15])


def test_dvt2_weight_collection_reports_missing_keys():
    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

    pipe = object.__new__(Pi05TorchFrontendRtx)
    pipe._dvt2_enabled = True
    pipe._ckpt_bf16 = _fake_weights()
    assert pipe._collect_dvt2_weights() is not None
    del pipe._ckpt_bf16["stage_projection_b"]
    assert pipe._collect_dvt2_weights() is None
