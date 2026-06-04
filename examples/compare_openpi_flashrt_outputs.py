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
        return data["obs"].item()
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
    diff = a - b
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    cosine = float(np.dot(a.ravel(), b.ravel()) / denom) if denom > 0 else float("nan")
    return {
        "shape_match": True,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine,
    }


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
    p.add_argument("--fixed-obs", action="store_true",
                   help="Reuse the exact same observation for every step.")
    p.add_argument("--save", type=Path, default=None,
                   help="Optional output .npz containing actions and last obs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    openpi = PolicyClient(args.openpi_host, args.openpi_port)
    flashrt = PolicyClient(args.flashrt_host, args.flashrt_port)
    print(f"openpi metadata:  {openpi.metadata}")
    print(f"flashrt metadata: {flashrt.metadata}")

    base_obs = _load_obs(args.obs_npz) if args.obs_npz else _random_h10w_obs(args.seed, args.prompt)
    openpi_actions = []
    flashrt_actions = []
    last_obs = base_obs
    openpi_exist = []
    flashrt_exist = []

    for i in range(args.steps):
        obs = base_obs if args.fixed_obs else (
            _load_obs(args.obs_npz) if args.obs_npz else _random_h10w_obs(args.seed + i, args.prompt)
        )
        last_obs = obs
        t0 = time.perf_counter()
        out_a = openpi.infer(obs)
        t_a = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        out_b = flashrt.infer(obs)
        t_b = (time.perf_counter() - t0) * 1000
        act_a = _actions(out_a)
        act_b = _actions(out_b)
        openpi_actions.append(act_a)
        flashrt_actions.append(act_b)
        cmp = _compare(act_a, act_b)
        exist_a = _maybe_exist(out_a)
        exist_b = _maybe_exist(out_b)
        exist_msg = ""
        if exist_a is not None or exist_b is not None:
            openpi_exist.append(exist_a)
            flashrt_exist.append(exist_b)
            exist_msg = f" exist_openpi={exist_a} exist_flashrt={exist_b}"
        print(f"[{i:03d}] openpi={t_a:.2f} ms flashrt={t_b:.2f} ms compare={cmp}{exist_msg}")

    a = np.stack(openpi_actions)
    b = np.stack(flashrt_actions)
    print(_stats("openpi actions", a))
    print(_stats("flashrt actions", b))
    print(f"aggregate compare: {_compare(a, b)}")
    if openpi_exist or flashrt_exist:
        print(f"openpi exist:  {openpi_exist}")
        print(f"flashrt exist: {flashrt_exist}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            openpi_actions=a,
            flashrt_actions=b,
            openpi_exist=np.array(openpi_exist, dtype=object),
            flashrt_exist=np.array(flashrt_exist, dtype=object),
            obs=np.array(last_obs, dtype=object),
        )
        print(f"saved: {args.save}")

    openpi.close()
    flashrt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
