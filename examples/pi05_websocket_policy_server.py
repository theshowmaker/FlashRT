#!/usr/bin/env python3
"""
FlashRT Pi0.5 websocket policy server.

Compatible with openpi_client.websocket_client_policy.WebsocketClientPolicy.
The server sends metadata immediately after a websocket connects, then accepts
msgpack-numpy observation dicts and returns msgpack-numpy action dicts.

4090 smoke example:
    FVK_PI05_RTX_FORCE_BF16=1 python examples/pi05_websocket_policy_server.py \
        --checkpoint /path/to/pi05_orbax/29999 \
        --framework jax \
        --hardware rtx_sm89 \
        --num-views 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import msgpack
import numpy as np
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("pi05_policy_server")


def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        shape = obj.shape
        arr = np.ascontiguousarray(obj)
        return {
            b"__ndarray__": True,
            b"data": arr.tobytes(),
            b"dtype": arr.dtype.str,
            b"shape": shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _numpy_scalar(value, dtype):
    return np.asarray(value, dtype=dtype).reshape(())


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_array)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FlashRT Pi0.5 websocket policy server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--framework", default="jax", choices=["torch", "jax"])
    parser.add_argument("--config", default="pi05")
    parser.add_argument("--hardware", default="rtx_sm89",
                        choices=["auto", "thor", "rtx_sm120", "rtx_sm89", "rtx_sm87"])
    parser.add_argument("--num-views", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Action chunk length. Your H10W/OpenPI policy expects 50.")
    parser.add_argument("--autotune", type=int, default=5)
    parser.add_argument("--use-fp4", action="store_true",
                        help="Enable Thor NVFP4 production preset for Pi0.5.")
    parser.add_argument("--cache-frames", type=int, default=None,
                        help="Pi0.5 temporal KV cache period. None keeps frontend default.")
    parser.add_argument("--vision-pool-factor", type=int, default=None, choices=(1, 2, 4),
                        help="Pi0.5 vision token spatial pooling factor.")
    parser.add_argument("--vision-num-layers", type=int, default=None,
                        help="Pi0.5 number of SigLIP vision layers to run.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default="do something",
                        help="Warmup/default prompt when an observation has no prompt.")
    parser.add_argument("--state-dim", type=int, default=16,
                        help="Dummy warmup state dim. Set 0 to warm without state.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Dummy warmup predict calls after model load.")
    parser.add_argument("--force-bf16", action="store_true", default=False,
                        help="Set FVK_PI05_RTX_FORCE_BF16=1 before loading the model.")
    parser.add_argument("--use-fp8", action="store_true",
                        help="Do not disable FP8 in load_model. On 4090 this may hit cuBLASLt code=15.")
    parser.add_argument("--ignore-state", action="store_true",
                        help="Do not pass observation state into model.predict().")
    parser.add_argument("--fixed-state-prompt-len", type=int, default=None,
                        help="Pi0.5 RTX: use one fixed state-prompt runtime length "
                             "(for OpenPI-like fixed-shape serving, use 200).")
    parser.add_argument("--prompt-mode", default="bucketed",
                        choices=["bucketed", "fixed", "openpi_masked_fixed200"],
                        help="Pi0.5 prompt runtime mode. openpi_masked_fixed200 "
                             "uses fixed 200-token state prompts plus OpenPI-style "
                             "prefix padding masks on supported RTX/Thor builds.")
    parser.add_argument("--policy-profile", default="auto",
                        choices=["auto", "none", "pi05_dvt2_fft_0605"],
                        help="Policy-side profile. auto enables DVT2/System2 heads "
                             "when detected in train_config_full.json.")
    parser.add_argument("--robot-type", default="auto",
                        choices=["auto", "none", "dvt2"],
                        help="DVT2 enables H10W dual joint-limit clamping. auto "
                             "uses dvt2 when the DVT2 policy profile is active.")
    parser.add_argument("--no-h10w-dual-absolute-actions", action="store_true",
                        help="Return raw normalized/unnormalized model actions without H10W dual AbsoluteActions().")
    parser.add_argument("--log-obs-keys-once", action="store_true", default=True)
    return parser.parse_args()


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"image must be rank-3, got shape={arr.shape}")
    # openpi ALOHA sample uses CHW; FlashRT wants HWC.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.nanmax(arr)) if arr.size else 0.0
            if max_v <= 1.5:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_images(obs: dict, num_views: int) -> list[np.ndarray]:
    candidates: list[Any] = []

    # H10W / LIBERO / DROID-style flat keys.
    for key in (
        "observation/image",
        "observation/exterior_image_1_left",
        "image",
    ):
        if key in obs:
            candidates.append(obs[key])
            break
    for key in (
        "observation/wrist_image_left",
        "observation/wrist_image",
        "observation/wrist_image_1",
        "wrist_image",
    ):
        if key in obs:
            candidates.append(obs[key])
            break
    for key in (
        "observation/wrist_image_right",
        "observation/wrist_image_2",
        "wrist_image_right",
    ):
        if key in obs:
            candidates.append(obs[key])
            break

    # ALOHA-style nested image dict.
    images_dict = obs.get("images")
    if isinstance(images_dict, dict):
        for key in ("cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low"):
            if key in images_dict:
                candidates.append(images_dict[key])
    elif isinstance(images_dict, (list, tuple)):
        candidates.extend(images_dict)

    if len(candidates) < num_views:
        raise ValueError(
            f"expected at least {num_views} image(s), found {len(candidates)}. "
            f"obs keys={list(obs.keys())}")
    return [_to_hwc_uint8(img) for img in candidates[:num_views]]


def _extract_state(obs: dict) -> np.ndarray | None:
    for key in (
        "observation/state",
        "state",
        "observation/joint_position",
    ):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).reshape(-1)
    if "observation/joint_position" in obs or "observation/gripper_position" in obs:
        parts = []
        if "observation/joint_position" in obs:
            parts.append(np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1))
        if "observation/gripper_position" in obs:
            parts.append(np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1))
        if parts:
            return np.concatenate(parts)
    return None


def _dummy_images(num_views: int) -> list[np.ndarray]:
    return [
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for _ in range(num_views)
    ]


def _h10w_dual_absolute_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Match openpi H10W dual AbsoluteActions(make_bool_mask(7, -1, 7, -1))."""
    arr = np.asarray(actions, dtype=np.float32).copy()
    st = np.asarray(state, dtype=np.float32).reshape(-1)
    mask = np.array([True] * 7 + [False] + [True] * 7 + [False], dtype=bool)
    dims = min(mask.size, arr.shape[-1], st.size)
    if dims > 0:
        arr[..., :dims] += np.where(mask[:dims], st[:dims], 0.0)
    return arr


class FlashRTPi05Policy:
    def __init__(self, args: argparse.Namespace):
        if args.force_bf16:
            os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"

        import flash_rt

        self.args = args
        self.flash_rt = flash_rt
        t0 = time.perf_counter()
        self.model = flash_rt.load_model(
            checkpoint=args.checkpoint,
            framework=args.framework,
            config=args.config,
            hardware=args.hardware,
            num_views=args.num_views,
            chunk_size=args.chunk_size,
            autotune=args.autotune,
            use_fp4=args.use_fp4,
            cache_frames=args.cache_frames,
            vision_pool_factor=args.vision_pool_factor,
            vision_num_layers=args.vision_num_layers,
            use_fp8=bool(args.use_fp8),
            fixed_state_prompt_len=args.fixed_state_prompt_len,
            prompt_mode=args.prompt_mode,
            policy_profile=args.policy_profile,
        )
        self.load_s = time.perf_counter() - t0
        self._printed_obs_keys = False
        self.action_shape: tuple[int, ...] | None = None
        logger.info("Model loaded in %.2fs", self.load_s)

    def warmup(self) -> None:
        state = None
        if self.args.state_dim > 0 and not self.args.ignore_state:
            state = np.zeros((self.args.state_dim,), dtype=np.float32)
        pipe = getattr(self.model, "_pipe", None)
        for i in range(max(0, self.args.warmup)):
            t0 = time.perf_counter()
            actions = self.model.predict(
                images=_dummy_images(self.args.num_views),
                prompt=self.args.prompt,
                state=state,
            )
            logger.info(
                "Warmup %d/%d: %.2f ms, actions=%s finite=%s",
                i + 1,
                self.args.warmup,
                (time.perf_counter() - t0) * 1000,
                actions.shape,
                bool(np.isfinite(actions).all()),
            )
            self.action_shape = tuple(int(v) for v in actions.shape)
        if pipe is not None and hasattr(pipe, "reset_dvt2_tracker"):
            pipe.reset_dvt2_tracker()
        if pipe is not None and hasattr(pipe, "_real_data_calibrated"):
            # Warmup uses synthetic images. Do not let Thor's lazy real-data
            # recalibration treat that dummy pass as representative of the
            # first real observation stream.
            pipe._real_data_calibrated = False
            logger.info("Reset Thor real-data calibration after dummy warmup")

    def metadata(self) -> dict:
        pipe = getattr(self.model, "_pipe", None)
        fixed_len = getattr(pipe, "fixed_state_prompt_len", self.args.fixed_state_prompt_len)
        prompt_capacity = getattr(pipe, "max_prompt_len", fixed_len)
        prompt_mode = getattr(pipe, "prompt_mode", self.args.prompt_mode)
        openpi_masked = bool(getattr(pipe, "openpi_masked_prefix", False))
        prompt_mask_supported = bool(getattr(pipe, "prompt_mask_supported", openpi_masked))
        dvt2_enabled = bool(getattr(pipe, "_dvt2_enabled", False))
        pipeline = getattr(pipe, "pipeline", None)
        materialize_encoder_output = (
            bool(getattr(pipeline, "materialize_encoder_output", False))
            if pipeline is not None else dvt2_enabled
        )
        fast_state_tokenizer = bool(getattr(pipe, "debug_prompt_stats", lambda: {})().get(
            "fast_state_tokenizer", os.environ.get("FLASH_RT_PI05_FAST_STATE_TOKENIZER", "1") != "0"))
        return {
            "model": "pi05",
            "framework": self.args.framework,
            "hardware": self.args.hardware,
            "policy_profile": getattr(pipe, "policy_profile_name", self.args.policy_profile),
            "dvt2_profile": dvt2_enabled,
            "robot_type": self._effective_robot_type(),
            "num_views": self.args.num_views,
            "checkpoint": self.args.checkpoint,
            "force_bf16": os.environ.get("FVK_PI05_RTX_FORCE_BF16") == "1",
            "use_fp4": self.args.use_fp4,
            "cache_frames": self.args.cache_frames,
            "fixed_state_prompt_len": fixed_len,
            "prompt_mode": prompt_mode,
            "prompt_capacity": prompt_capacity,
            "openpi_masked_prefix": openpi_masked,
            "prompt_mask_supported": prompt_mask_supported,
            "dvt2_materialize_encoder_output": materialize_encoder_output,
            "dvt2_openpi_fixed_hole_rope": dvt2_enabled,
            "fast_state_tokenizer": fast_state_tokenizer,
            "load_s": self.load_s,
            "action_shape": list(self.action_shape or (self.args.chunk_size, -1)),
        }

    def _effective_robot_type(self) -> str:
        robot_type = str(self.args.robot_type)
        if robot_type == "auto":
            pipe = getattr(self.model, "_pipe", None)
            return "dvt2" if bool(getattr(pipe, "_dvt2_enabled", False)) else "none"
        return robot_type

    def infer(self, obs: dict) -> dict:
        if self.args.log_obs_keys_once and not self._printed_obs_keys:
            logger.info("OBS keys: %s", list(obs.keys()))
            self._printed_obs_keys = True

        prep_t0 = time.perf_counter()
        prompt = str(obs.get("prompt") or self.args.prompt)
        images = _extract_images(obs, self.args.num_views)
        state = None if self.args.ignore_state else _extract_state(obs)
        pipe = getattr(self.model, "_pipe", None)
        # Match OpenPI StageTracker semantics: frame_index/episode_index are
        # metadata and do not reset tracker state. DVT2 tracker reset happens
        # in the frontend only when the task category changes.
        prep_ms = (time.perf_counter() - prep_t0) * 1000

        infer_t0 = time.perf_counter()
        actions = self.model.predict(images=images, prompt=prompt, state=state)
        model_result = getattr(self.model, "last_result", None) or {}
        model_timing = getattr(self.model, "last_timing", {}) or {}
        if state is not None and not self.args.no_h10w_dual_absolute_actions:
            actions = _h10w_dual_absolute_actions(actions, state)
        actions = np.asarray(actions, dtype=np.float32)
        if bool(getattr(pipe, "_dvt2_enabled", False)):
            actions = actions[:, :16]
        if self._effective_robot_type() == "dvt2":
            from flash_rt.core.utils.dvt2_policy import clamp_h10w_dvt2_actions
            actions = clamp_h10w_dvt2_actions(actions)
        infer_ms = (time.perf_counter() - infer_t0) * 1000

        policy_timing = {
            "prep_ms": prep_ms,
            "predict_ms": infer_ms,
        }
        policy_timing.update(model_timing)
        response = {
            "actions": actions,
            "action": actions,
            "policy_timing": policy_timing,
        }
        if "exist" in model_result:
            response["exist"] = _numpy_scalar(model_result["exist"], np.int32)
        if "exist_prob" in model_result:
            response["exist_prob"] = _numpy_scalar(model_result["exist_prob"], np.float32)
        for key in ("subtask_logits", "predicted_stage", "stage"):
            if key in model_result:
                response[key] = np.asarray(model_result[key])
        if "dvt2_debug" in model_result:
            response["dvt2_debug"] = model_result["dvt2_debug"]
        return response


async def _handler(websocket, policy: FlashRTPi05Policy, api_key: str | None):
    if api_key:
        auth = websocket.request.headers.get("Authorization")
        if auth != f"Api-Key {api_key}":
            await websocket.close(code=1008, reason="invalid api key")
            return

    logger.info("Connection from %s opened", websocket.remote_address)
    await websocket.send(packb(policy.metadata()))
    prev_total_ms = None

    while True:
        try:
            start = time.perf_counter()
            message = await websocket.recv()
            if isinstance(message, str):
                await websocket.send("expected binary msgpack message")
                continue
            obs = unpackb(message)
            infer_t0 = time.perf_counter()
            response = policy.infer(obs)
            server_infer_ms = (time.perf_counter() - infer_t0) * 1000
            response.setdefault("server_timing", {})
            response["server_timing"]["infer_ms"] = server_infer_ms
            if prev_total_ms is not None:
                response["server_timing"]["prev_total_ms"] = prev_total_ms
            await websocket.send(packb(response))
            prev_total_ms = (time.perf_counter() - start) * 1000
        except ConnectionClosed:
            logger.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            tb = traceback.format_exc()
            logger.error("Inference failed:\n%s", tb)
            try:
                await websocket.send(tb)
                await websocket.close(code=1011, reason="internal server error")
            except ConnectionClosed:
                logger.info("Connection from %s closed while reporting an error", websocket.remote_address)
            break


async def _run_server(args: argparse.Namespace, policy: FlashRTPi05Policy) -> None:
    import websockets.asyncio.server as ws_server

    async def handler(websocket):
        await _handler(websocket, policy, args.api_key)

    async with ws_server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    ):
        logger.info("Serving websocket policy on ws://%s:%s", args.host, args.port)
        await asyncio.Future()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy = FlashRTPi05Policy(args)
    policy.warmup()
    asyncio.run(_run_server(args, policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
