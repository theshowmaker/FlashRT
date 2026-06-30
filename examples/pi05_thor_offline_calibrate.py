#!/usr/bin/env python3
"""Build a Thor Pi0.5 FP8 calibration cache from recorded observations."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger("pi05_thor_offline_calibrate")


def _load_obs(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    if "obs" in data:
        obs = dict(data["obs"].item())
        passthrough_keys = {
            "episode_index",
            "frame_index",
            "task_index",
            "task_category",
            "subtask_state",
            "exist_label",
        }
        for key in data.files:
            if key == "obs" or key in obs or key not in passthrough_keys:
                continue
            obs[key] = data[key].item() if data[key].dtype == object and data[key].shape == () else data[key]
        return obs
    obs = {}
    for key in data.files:
        obs[key] = data[key].item() if data[key].dtype == object and data[key].shape == () else data[key]
    return obs


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"image must be rank-3, got shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.nanmax(arr)) if arr.size else 0.0
            if max_v <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _first_present(obs: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obs:
            return obs[key]
    return None


def _extract_images(obs: dict, num_views: int) -> list[np.ndarray]:
    if "images" in obs:
        images = list(obs["images"])
    else:
        main = _first_present(obs, ("observation/image", "image"))
        left = _first_present(obs, ("observation/wrist_image_left", "wrist_image", "observation/wrist_image"))
        right = _first_present(obs, ("observation/wrist_image_right", "wrist_image_right"))
        if main is None:
            raise KeyError("missing main image key: expected observation/image or image")
        images = [main]
        if num_views >= 2:
            images.append(left if left is not None else main)
        if num_views >= 3:
            images.append(right if right is not None else images[-1])
    if len(images) < num_views:
        raise ValueError(f"need {num_views} image views, got {len(images)}")
    return [_to_hwc_uint8(img) for img in images[:num_views]]


def _extract_state(obs: dict) -> np.ndarray | None:
    state = _first_present(obs, ("observation/state", "state"))
    if state is None:
        return None
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _extract_prompt(obs: dict, default: str) -> str:
    prompt = obs.get("prompt", default)
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    arr = np.asarray(prompt)
    if arr.shape == ():
        return str(arr.item())
    return str(prompt)


def _calibration_obs(obs: dict, *, num_views: int, prompt: str) -> dict:
    out = {
        "images": _extract_images(obs, num_views),
        "prompt": _extract_prompt(obs, prompt),
    }
    state = _extract_state(obs)
    if state is not None:
        out["state"] = state
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs-glob", required=True,
                        help="Recorded observation .npz glob, e.g. 'tmp/robot_obs_record/*.npz'.")
    parser.add_argument("--framework", default="jax", choices=["jax"])
    parser.add_argument("--config", default="pi05")
    parser.add_argument("--hardware", default="thor", choices=["thor"])
    parser.add_argument("--num-views", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--autotune", type=int, default=0)
    parser.add_argument("--use-fp4", action="store_true")
    parser.add_argument("--prompt", default=None,
                        help="Prompt used to build state-prompt buffers. Defaults to first obs prompt.")
    parser.add_argument("--ignore-state", action="store_true",
                        help="Do not use the first observation state when setting the prompt.")
    parser.add_argument("--fixed-state-prompt-len", type=int, default=200)
    parser.add_argument("--prompt-mode", default="openpi_masked_fixed200",
                        choices=["bucketed", "fixed", "openpi_masked_fixed200"])
    parser.add_argument("--policy-profile", default="pi05_dvt2_fft_0605",
                        choices=["auto", "none", "pi05_dvt2_fft_0605"])
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clear-existing", action="store_true",
                        help="Clear existing calibration cache for this checkpoint before loading.")
    parser.add_argument("--debug-calibration", action="store_true",
                        help="Print finite/nonfinite reduction diagnostics for multi-frame calibration.")
    parser.add_argument("--calibration-debug-topk", type=int, default=8,
                        help="How many worst quantization points to show with --debug-calibration.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    paths = sorted(Path(p) for p in glob.glob(args.obs_glob))
    if args.max_samples is not None:
        paths = paths[:max(0, args.max_samples)]
    if not paths:
        raise FileNotFoundError(f"no obs files matched: {args.obs_glob}")
    if len(paths) < 2:
        raise ValueError("multi-frame offline calibration needs at least 2 observations")

    if args.clear_existing:
        from flash_rt.core.quant.calibrator import clear_calibration
        clear_calibration(args.checkpoint)

    raw_first = _load_obs(paths[0])
    prompt = args.prompt or _extract_prompt(raw_first, "do something")
    state = None if args.ignore_state else _extract_state(raw_first)

    import flash_rt

    logger.info("Loading model: checkpoint=%s", args.checkpoint)
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework=args.framework,
        config=args.config,
        hardware=args.hardware,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        autotune=args.autotune,
        use_fp4=args.use_fp4,
        fixed_state_prompt_len=args.fixed_state_prompt_len,
        prompt_mode=args.prompt_mode,
        policy_profile=args.policy_profile,
    )
    logger.info("Setting prompt for calibration: %r state=%s",
                prompt, None if state is None else tuple(state.shape))
    model.set_prompt(prompt, state=state)
    if args.debug_calibration:
        pipe = getattr(model, "_pipe", None)
        if pipe is not None:
            pipe._calibration_debug_topk = max(1, int(args.calibration_debug_topk))

    obs_list = []
    for path in paths:
        obs_list.append(_calibration_obs(
            _load_obs(path),
            num_views=args.num_views,
            prompt=prompt,
        ))
    logger.info("Calibrating from %d observations, percentile=%.3f",
                len(obs_list), args.percentile)
    model.calibrate(
        obs_list,
        percentile=args.percentile,
        max_samples=None,
        verbose=args.verbose,
    )
    logger.info("Offline multi-frame calibration complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
