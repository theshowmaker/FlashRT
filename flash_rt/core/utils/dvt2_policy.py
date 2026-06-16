"""Utilities for the H10W DVT2 Pi0.5 policy profile.

This module intentionally stays small and framework-light.  The websocket
server and Pi0.5 RTX frontends share the pure policy/profile helpers, while
the optional head computations operate on torch tensors when a DVT2 checkpoint
is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


PROFILE_NONE = "none"
PROFILE_AUTO = "auto"
PROFILE_DVT2_0605 = "pi05_dvt2_fft_0605"


@dataclass(frozen=True)
class DVT2Profile:
    task_categories: tuple[str, ...] = ("pick", "place", "give")
    category_num_stages: tuple[int, ...] = (5, 5, 6)
    stage_fusion_num_tokens: int = 4
    max_num_stages: int = 6
    include_no_action_stage: bool = False


DVT2_DEFAULT_PROFILE = DVT2Profile()


LEFT_ARM_JOINT_LIMITS = np.array(
    [
        [-1.5708, 3.7176],
        [-0.4363, 2.2689],
        [-0.7854, 3.9270],
        [-1.5708, 1.6232],
        [-2.9671, 2.9671],
        [-1.2217, 1.2217],
        [-1.2217, 1.2217],
    ],
    dtype=np.float32,
)
RIGHT_ARM_JOINT_LIMITS = np.array(
    [
        [-1.5708, 3.7176],
        [-2.2689, 0.4363],
        [-3.9270, 0.7854],
        [-1.5708, 1.6232],
        [-2.9671, 2.9671],
        [-1.2217, 1.2217],
        [-1.2217, 1.2217],
    ],
    dtype=np.float32,
)


def h10w_dual_action_joint_bounds(
    left_arm_joint_limits=LEFT_ARM_JOINT_LIMITS,
    right_arm_joint_limits=RIGHT_ARM_JOINT_LIMITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build DVT2 bounds for the 16-dim H10W dual action layout."""
    left_limits = np.asarray(left_arm_joint_limits, dtype=np.float32)
    right_limits = np.asarray(right_arm_joint_limits, dtype=np.float32)
    if left_limits.shape != (7, 2):
        raise ValueError(f"left arm joint limits must be (7, 2), got {left_limits.shape}")
    if right_limits.shape != (7, 2):
        raise ValueError(f"right arm joint limits must be (7, 2), got {right_limits.shape}")
    if np.any(left_limits[:, 0] > left_limits[:, 1]):
        raise ValueError("left arm joint limits must satisfy min <= max")
    if np.any(right_limits[:, 0] > right_limits[:, 1]):
        raise ValueError("right arm joint limits must satisfy min <= max")
    lower = np.full(16, -np.inf, dtype=np.float32)
    upper = np.full(16, np.inf, dtype=np.float32)
    lower[:7], upper[:7] = left_limits[:, 0], left_limits[:, 1]
    lower[8:15], upper[8:15] = right_limits[:, 0], right_limits[:, 1]
    return lower, upper


def clamp_h10w_dvt2_actions(actions: np.ndarray) -> np.ndarray:
    """Clamp DVT2 arm joints and preserve gripper columns."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 0 or arr.shape[-1] < 16:
        raise ValueError(f"expected actions with last dim >= 16, got {arr.shape}")
    lower, upper = h10w_dual_action_joint_bounds()
    out = np.array(arr, copy=True)
    out[..., :16] = np.clip(out[..., :16], lower, upper)
    return out


def load_train_config(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    path = Path(checkpoint_dir) / "train_config_full.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def profile_from_train_config(config: Mapping[str, Any] | None) -> DVT2Profile | None:
    if not config:
        return None
    model = config.get("model")
    if not isinstance(model, Mapping):
        return None
    required = (
        model.get("use_stage_prediction"),
        model.get("use_stage_fusion"),
        model.get("use_stage_tracking"),
        model.get("use_exist_prediction"),
        model.get("use_exist_mlp"),
    )
    if not all(bool(x) for x in required):
        return None
    categories = tuple(str(x) for x in model.get("task_categories", DVT2_DEFAULT_PROFILE.task_categories))
    counts = tuple(int(x) for x in model.get("category_num_stages", DVT2_DEFAULT_PROFILE.category_num_stages))
    if not categories or len(categories) != len(counts):
        categories = DVT2_DEFAULT_PROFILE.task_categories
        counts = DVT2_DEFAULT_PROFILE.category_num_stages
    max_stages = int(model.get("max_num_stages", max(counts)))
    return DVT2Profile(
        task_categories=categories,
        category_num_stages=counts,
        stage_fusion_num_tokens=int(model.get("stage_fusion_num_tokens", 4)),
        max_num_stages=max(max_stages, max(counts)),
        include_no_action_stage=bool(model.get("include_no_action_stage", False)),
    )


def resolve_policy_profile(
    requested: str,
    checkpoint_dir: str | Path,
) -> DVT2Profile | None:
    requested = str(requested or PROFILE_AUTO)
    if requested == PROFILE_NONE:
        return None
    if requested == PROFILE_DVT2_0605:
        return profile_from_train_config(load_train_config(checkpoint_dir)) or DVT2_DEFAULT_PROFILE
    if requested != PROFILE_AUTO:
        raise ValueError(
            f"policy_profile must be one of "
            f"{PROFILE_AUTO!r}, {PROFILE_NONE!r}, {PROFILE_DVT2_0605!r}; got {requested!r}"
        )
    return profile_from_train_config(load_train_config(checkpoint_dir))


def task_category_from_prompt(prompt: str, profile: DVT2Profile = DVT2_DEFAULT_PROFILE) -> int:
    words = str(prompt or "").strip().lower().split()
    first = words[0] if words else ""
    for i, category in enumerate(profile.task_categories):
        if first == category.lower():
            return i
    return 0


def num_stages_for_category(task_category: int, profile: DVT2Profile = DVT2_DEFAULT_PROFILE) -> int:
    idx = int(np.clip(task_category, 0, len(profile.category_num_stages) - 1))
    return max(int(profile.category_num_stages[idx]), 1)


def normalize_stage(stage: int, task_category: int, profile: DVT2Profile = DVT2_DEFAULT_PROFILE) -> float | int:
    if int(stage) == -1:
        return -1
    denom = num_stages_for_category(task_category, profile) - 1
    if denom <= 0:
        return 0.0
    return float(stage) / float(denom)


def _linear_torch(x, weight, bias):
    return x @ weight.float() + bias.float()


def make_stage_fusion_tokens_torch(
    prompt_embeds,
    task_category: int,
    stage: int,
    weights: Mapping[str, Any],
    profile: DVT2Profile = DVT2_DEFAULT_PROFILE,
):
    """Return the 4 openpi stage-fusion tokens as a CUDA/CPU torch tensor."""
    import torch
    import torch.nn.functional as F

    if prompt_embeds.ndim != 2:
        raise ValueError(f"prompt_embeds must be [tokens, dim], got {tuple(prompt_embeds.shape)}")
    device = prompt_embeds.device
    task_embedding = prompt_embeds.float().mean(dim=0, keepdim=True)
    task_id = int(np.clip(task_category, 0, len(profile.category_num_stages) - 1))
    class_state = int(stage) + 1 if profile.include_no_action_stage else int(stage)
    class_state = int(np.clip(class_state, 0, profile.max_num_stages - 1))
    counts = [int(x) + (1 if profile.include_no_action_stage else 0) for x in profile.category_num_stages]
    offsets = np.cumsum([0] + counts[:-1])
    safe_class = int(np.clip(class_state, 0, counts[task_id] - 1))
    task_stage_idx = int(offsets[task_id] + safe_class)

    stage_encoding = weights["stage_class_embeddings"][class_state : class_state + 1].float()
    task_stage_embedding = weights["task_stage_embeddings"][task_stage_idx : task_stage_idx + 1].float()
    all_inputs = torch.cat([task_embedding, stage_encoding, task_stage_embedding], dim=-1)

    gate_sincos = torch.sigmoid(_linear_torch(all_inputs, weights["gate_sincos_w"], weights["gate_sincos_b"]))
    gate_task_stage = torch.sigmoid(
        _linear_torch(all_inputs, weights["gate_task_stage_w"], weights["gate_task_stage_b"])
    )
    gate_task = torch.sigmoid(_linear_torch(all_inputs, weights["gate_task_w"], weights["gate_task_b"]))
    task_gated = task_embedding * gate_task
    balanced = _linear_torch(
        F.relu(_linear_torch(all_inputs, weights["fusion_layer1_w"], weights["fusion_layer1_b"])),
        weights["fusion_layer2_w"],
        weights["fusion_layer2_b"],
    )
    gated_stage = torch.cat(
        [stage_encoding * gate_sincos, task_stage_embedding * gate_task_stage],
        dim=-1,
    )
    stage_dominant = _linear_torch(gated_stage, weights["stage_projection_w"], weights["stage_projection_b"])
    pure_stage = torch.cat([stage_encoding, task_stage_embedding], dim=-1)
    return torch.stack([task_gated, balanced, stage_dominant, pure_stage], dim=1).squeeze(0).to(
        device=device, dtype=prompt_embeds.dtype
    )


def stage_logits_torch(
    prompt_pooled,
    task_category: int,
    weights: Mapping[str, Any],
    profile: DVT2Profile = DVT2_DEFAULT_PROFILE,
):
    """Compute masked DVT2 stage logits from pooled prompt prefix output."""
    import torch
    import torch.nn.functional as F

    task_id = int(np.clip(task_category, 0, len(profile.category_num_stages) - 1))
    task_emb = weights["stage_task_embed"][task_id : task_id + 1].float()
    x = torch.cat([prompt_pooled.float().reshape(1, -1), task_emb], dim=-1)
    hidden = F.gelu(_linear_torch(x, weights["stage_mlp_1_w"], weights["stage_mlp_1_b"]))
    logits = _linear_torch(hidden, weights["stage_mlp_2_w"], weights["stage_mlp_2_b"]).reshape(-1)
    valid = num_stages_for_category(task_id, profile)
    masked = logits.clone()
    masked[valid:] = -float("inf")
    return masked


def exist_prediction_torch(base_pooled, prompt_pooled, weights: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return (exist_pred, exist_prob) from the DVT2 exist MLP."""
    import torch
    import torch.nn.functional as F

    x = torch.cat([base_pooled.float().reshape(1, -1), prompt_pooled.float().reshape(1, -1)], dim=-1)
    hidden = F.gelu(_linear_torch(x, weights["exist_mlp_1_w"], weights["exist_mlp_1_b"]))
    logit = _linear_torch(hidden, weights["exist_mlp_2_w"], weights["exist_mlp_2_b"]).reshape(())
    prob = torch.sigmoid(logit)
    pred = (prob > 0.5).to(torch.int32)
    return pred, prob
