#!/usr/bin/env python3
"""Compare actions returned by two openpi-compatible websocket policies.

This is intended for checking a FlashRT service against the original openpi
service with the same observation payload. Pi0/Pi0.5 sampling is stochastic
unless both runtimes are explicitly driven with the same diffusion noise, so
this script reports numerical distance/statistics rather than requiring exact
equality.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(obj):
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


def packb(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_array)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_unpack_array)


class PolicyClient:
    def __init__(self, host: str, port: int):
        uri = host if host.startswith("ws://") or host.startswith("wss://") else f"ws://{host}:{port}"
        self._ws = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=None,
        )
        self.metadata = unpackb(self._ws.recv())

    def infer(self, obs: dict) -> dict:
        self._ws.send(packb(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(response)
        return unpackb(response)

    def close(self) -> None:
        self._ws.close()


def _random_h10w_obs(seed: int, prompt: str) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.uniform(-1, 1, size=(16,)).astype(np.float32),
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


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


def _actions(out: dict) -> np.ndarray:
    if "actions" in out:
        return np.asarray(out["actions"], dtype=np.float32)
    if "action" in out:
        return np.asarray(out["action"], dtype=np.float32)
    raise KeyError(f"response has no actions/action key: {list(out.keys())}")


def _maybe_exist(out: dict):
    if "exist" not in out:
        return None
    arr = np.asarray(out["exist"], dtype=np.int32)
    if arr.size == 1:
        return arr.reshape(())
    return arr


def _maybe_array(out: dict, key: str, dtype=None):
    if key not in out:
        return None
    return np.asarray(out[key], dtype=dtype)


def _format_optional(name: str, value) -> str:
    if value is None:
        return f"{name}=None"
    arr = np.asarray(value)
    if arr.size == 1:
        return f"{name}={arr.reshape(-1)[0]}"
    return f"{name}=shape{arr.shape}"


def _stats(name: str, arr: np.ndarray) -> str:
    finite = bool(np.isfinite(arr).all())
    return (
        f"{name}: shape={arr.shape}, finite={finite}, "
        f"range=[{arr.min():.6f}, {arr.max():.6f}], "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}"
    )


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_match": False}
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        mask = np.isfinite(a) & np.isfinite(b)
        if not bool(mask.any()):
            return {"shape_match": True, "finite": False}
        a_cmp = a[mask]
        b_cmp = b[mask]
    else:
        a_cmp = a
        b_cmp = b
    diff = a_cmp - b_cmp
    denom = float(np.linalg.norm(a_cmp.ravel()) * np.linalg.norm(b_cmp.ravel()))
    cosine = float(np.dot(a_cmp.ravel(), b_cmp.ravel()) / denom) if denom > 0 else float("nan")
    return {
        "shape_match": True,
        "finite": finite,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine,
    }


def _latency_stats(name: str, values: list[float]) -> str:
    if not values:
        return f"{name}: no samples"
    arr = np.asarray(values, dtype=np.float64)
    return (
        f"{name}: min={arr.min():.2f} ms, p50={np.percentile(arr, 50):.2f} ms, "
        f"p90={np.percentile(arr, 90):.2f} ms, p95={np.percentile(arr, 95):.2f} ms, "
        f"mean={arr.mean():.2f} ms, max={arr.max():.2f} ms"
    )


def _numeric_timing(out: dict) -> dict[str, float]:
    timing = out.get("policy_timing") or out.get("server_timing") or {}
    result = {}
    if isinstance(timing, dict):
        for k, v in timing.items():
            try:
                result[str(k)] = float(v)
            except Exception:
                pass
    return result


def _print_timing_summary(prefix: str, rows: list[dict[str, float]]) -> None:
    keys = sorted({k for row in rows for k in row})
    for key in keys:
        values = [row[key] for row in rows if key in row]
        print(_latency_stats(f"{prefix} {key}", values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare openpi and FlashRT websocket policy outputs.")
    p.add_argument("--openpi-host", default="127.0.0.1")
    p.add_argument("--openpi-port", type=int, default=8000)
    p.add_argument("--flashrt-host", default="127.0.0.1")
    p.add_argument("--flashrt-port", type=int, default=8001)
    p.add_argument("--prompt", default="do something")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--obs-npz", type=Path, default=None,
                   help="Optional .npz observation. If omitted, random H10W observations are generated.")
    p.add_argument("--obs-glob", default=None,
                   help="Glob of .npz observations to replay in sorted order. "
                        "Use quotes so the shell does not expand it.")
    p.add_argument("--fixed-obs", action="store_true",
                   help="Reuse the exact same observation for every step.")
    p.add_argument("--require-exist", type=int, choices=(0, 1), default=None,
                   help="Fail unless both services return this exist value.")
    p.add_argument("--require-exist-match", action="store_true",
                   help="Fail if both services return exist but values differ.")
    p.add_argument("--require-stage-match", action="store_true",
                   help="Fail if both services return predicted_stage but values differ.")
    p.add_argument("--require-action-shape", default=None,
                   help="Optional required action shape, e.g. 50,16.")
    p.add_argument("--summary-skip", type=int, default=0,
                   help="Skip the first N calls in latency summaries.")
    p.add_argument("--save", type=Path, default=None,
                   help="Optional output .npz containing actions and last obs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.obs_npz is not None and args.obs_glob is not None:
        raise ValueError("--obs-npz and --obs-glob are mutually exclusive")

    openpi = PolicyClient(args.openpi_host, args.openpi_port)
    flashrt = PolicyClient(args.flashrt_host, args.flashrt_port)
    print(f"openpi metadata:  {openpi.metadata}")
    print(f"flashrt metadata: {flashrt.metadata}")

    obs_paths = [Path(p) for p in sorted(glob.glob(args.obs_glob))] if args.obs_glob else []
    if args.obs_glob and not obs_paths:
        raise FileNotFoundError(f"--obs-glob matched no files: {args.obs_glob}")
    if obs_paths:
        print(f"replaying {min(args.steps, len(obs_paths))} obs files from: {args.obs_glob}")

    base_obs = (
        _load_obs(obs_paths[0]) if obs_paths
        else _load_obs(args.obs_npz) if args.obs_npz
        else _random_h10w_obs(args.seed, args.prompt)
    )
    openpi_actions = []
    flashrt_actions = []
    last_obs = base_obs
    openpi_exist = []
    flashrt_exist = []
    openpi_exist_prob = []
    flashrt_exist_prob = []
    openpi_predicted_stage = []
    flashrt_predicted_stage = []
    openpi_stage = []
    flashrt_stage = []
    openpi_logits = []
    flashrt_logits = []
    openpi_times = []
    flashrt_times = []
    openpi_policy_timings = []
    flashrt_policy_timings = []
    used_obs_paths = []

    for i in range(args.steps):
        if args.fixed_obs:
            obs = base_obs
            obs_path = obs_paths[0] if obs_paths else args.obs_npz
        elif obs_paths:
            if i >= len(obs_paths):
                break
            obs_path = obs_paths[i]
            obs = _load_obs(obs_path)
        elif args.obs_npz:
            obs_path = args.obs_npz
            obs = _load_obs(args.obs_npz)
        else:
            obs_path = None
            obs = _random_h10w_obs(args.seed + i, args.prompt)
        last_obs = obs
        if obs_path is not None:
            used_obs_paths.append(str(obs_path))
        t0 = time.perf_counter()
        out_a = openpi.infer(obs)
        t_a = (time.perf_counter() - t0) * 1000
        openpi_times.append(t_a)
        openpi_policy_timings.append(_numeric_timing(out_a))
        t0 = time.perf_counter()
        out_b = flashrt.infer(obs)
        t_b = (time.perf_counter() - t0) * 1000
        flashrt_times.append(t_b)
        flashrt_policy_timings.append(_numeric_timing(out_b))
        act_a = _actions(out_a)
        act_b = _actions(out_b)
        if args.require_action_shape:
            expected_shape = tuple(int(x) for x in args.require_action_shape.split(","))
            if act_a.shape != expected_shape or act_b.shape != expected_shape:
                raise RuntimeError(
                    f"expected action shape {expected_shape}, got openpi={act_a.shape}, "
                    f"flashrt={act_b.shape}"
                )
        openpi_actions.append(act_a)
        flashrt_actions.append(act_b)
        cmp = _compare(act_a, act_b)
        exist_a = _maybe_exist(out_a)
        exist_b = _maybe_exist(out_b)
        exist_prob_a = _maybe_array(out_a, "exist_prob", np.float32)
        exist_prob_b = _maybe_array(out_b, "exist_prob", np.float32)
        pred_stage_a = _maybe_array(out_a, "predicted_stage", np.int32)
        pred_stage_b = _maybe_array(out_b, "predicted_stage", np.int32)
        stage_a = _maybe_array(out_a, "stage")
        stage_b = _maybe_array(out_b, "stage")
        logits_a = _maybe_array(out_a, "subtask_logits", np.float32)
        logits_b = _maybe_array(out_b, "subtask_logits", np.float32)
        if exist_prob_a is not None or exist_prob_b is not None:
            openpi_exist_prob.append(exist_prob_a)
            flashrt_exist_prob.append(exist_prob_b)
        if pred_stage_a is not None or pred_stage_b is not None:
            openpi_predicted_stage.append(pred_stage_a)
            flashrt_predicted_stage.append(pred_stage_b)
            if args.require_stage_match and pred_stage_a is not None and pred_stage_b is not None:
                if int(np.asarray(pred_stage_a).reshape(-1)[0]) != int(np.asarray(pred_stage_b).reshape(-1)[0]):
                    debug_a = out_a.get("dvt2_debug")
                    debug_b = out_b.get("dvt2_debug")
                    raise RuntimeError(
                        f"predicted_stage mismatch: openpi={pred_stage_a}, flashrt={pred_stage_b}; "
                        f"openpi_debug={debug_a}, flashrt_debug={debug_b}; "
                        f"openpi_logits={logits_a}, flashrt_logits={logits_b}"
                    )
        if stage_a is not None or stage_b is not None:
            openpi_stage.append(stage_a)
            flashrt_stage.append(stage_b)
        if logits_a is not None or logits_b is not None:
            openpi_logits.append(logits_a)
            flashrt_logits.append(logits_b)
        exist_msg = ""
        if exist_a is not None or exist_b is not None:
            openpi_exist.append(exist_a)
            flashrt_exist.append(exist_b)
            exist_msg = f" exist_openpi={exist_a} exist_flashrt={exist_b}"
            if args.require_exist_match and exist_a is not None and exist_b is not None:
                if int(np.asarray(exist_a).reshape(-1)[0]) != int(np.asarray(exist_b).reshape(-1)[0]):
                    raise RuntimeError(f"exist mismatch: openpi={exist_a}, flashrt={exist_b}")
            if args.require_exist is not None:
                required = int(args.require_exist)
                if exist_a is None or exist_b is None:
                    raise RuntimeError(
                        f"--require-exist={required} but one service did not return exist: "
                        f"openpi={exist_a}, flashrt={exist_b}"
                    )
                if int(np.asarray(exist_a).reshape(-1)[0]) != required:
                    raise RuntimeError(
                        f"openpi exist={exist_a}, expected {required}"
                    )
                if int(np.asarray(exist_b).reshape(-1)[0]) != required:
                    raise RuntimeError(
                        f"flashrt exist={exist_b}, expected {required}"
                    )
        extras = " ".join(
            [
                _format_optional("stage_openpi", pred_stage_a),
                _format_optional("stage_flashrt", pred_stage_b),
                _format_optional("exist_prob_openpi", exist_prob_a),
                _format_optional("exist_prob_flashrt", exist_prob_b),
            ]
        )
        logits_msg = ""
        if logits_a is not None and logits_b is not None:
            logits_msg = f" logits_compare={_compare(logits_a, logits_b)}"
        print(
            f"[{i:03d}] openpi={t_a:.2f} ms flashrt={t_b:.2f} ms "
            f"compare={cmp}{exist_msg} {extras}{logits_msg}"
        )

    a = np.stack(openpi_actions)
    b = np.stack(flashrt_actions)
    print(_stats("openpi actions", a))
    print(_stats("flashrt actions", b))
    print(f"aggregate compare: {_compare(a, b)}")
    skip = max(0, int(args.summary_skip))
    if skip:
        print(f"latency summary skips first {skip} call(s)")
    print(_latency_stats("openpi latency", openpi_times[skip:]))
    print(_latency_stats("flashrt latency", flashrt_times[skip:]))
    _print_timing_summary("openpi policy", openpi_policy_timings[skip:])
    _print_timing_summary("flashrt policy", flashrt_policy_timings[skip:])
    if openpi_exist or flashrt_exist:
        print(f"openpi exist:  {openpi_exist}")
        print(f"flashrt exist: {flashrt_exist}")
    if openpi_predicted_stage or flashrt_predicted_stage:
        print(f"openpi predicted_stage:  {openpi_predicted_stage}")
        print(f"flashrt predicted_stage: {flashrt_predicted_stage}")
    if openpi_stage or flashrt_stage:
        print(f"openpi stage:  {openpi_stage}")
        print(f"flashrt stage: {flashrt_stage}")
    if openpi_exist_prob and flashrt_exist_prob and all(x is not None for x in openpi_exist_prob + flashrt_exist_prob):
        ep_a = np.stack([np.asarray(x, dtype=np.float32) for x in openpi_exist_prob])
        ep_b = np.stack([np.asarray(x, dtype=np.float32) for x in flashrt_exist_prob])
        print(f"aggregate exist_prob compare: {_compare(ep_a, ep_b)}")
    if openpi_logits and flashrt_logits and all(x is not None for x in openpi_logits + flashrt_logits):
        l_a = np.stack([np.asarray(x, dtype=np.float32) for x in openpi_logits])
        l_b = np.stack([np.asarray(x, dtype=np.float32) for x in flashrt_logits])
        print(f"aggregate subtask_logits compare: {_compare(l_a, l_b)}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            openpi_actions=a,
            flashrt_actions=b,
            openpi_exist=np.array(openpi_exist, dtype=object),
            flashrt_exist=np.array(flashrt_exist, dtype=object),
            openpi_exist_prob=np.array(openpi_exist_prob, dtype=object),
            flashrt_exist_prob=np.array(flashrt_exist_prob, dtype=object),
            openpi_predicted_stage=np.array(openpi_predicted_stage, dtype=object),
            flashrt_predicted_stage=np.array(flashrt_predicted_stage, dtype=object),
            openpi_stage=np.array(openpi_stage, dtype=object),
            flashrt_stage=np.array(flashrt_stage, dtype=object),
            openpi_logits=np.array(openpi_logits, dtype=object),
            flashrt_logits=np.array(flashrt_logits, dtype=object),
            openpi_times_ms=np.asarray(openpi_times, dtype=np.float32),
            flashrt_times_ms=np.asarray(flashrt_times, dtype=np.float32),
            openpi_policy_timings=np.array(openpi_policy_timings, dtype=object),
            flashrt_policy_timings=np.array(flashrt_policy_timings, dtype=object),
            obs_paths=np.asarray(used_obs_paths, dtype=object),
            obs=np.array(last_obs, dtype=object),
        )
        print(f"saved: {args.save}")

    openpi.close()
    flashrt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
