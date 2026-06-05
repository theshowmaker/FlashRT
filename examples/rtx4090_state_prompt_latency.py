#!/usr/bin/env python3
"""Check RTX Pi0.5 state-in-prompt latency with real H10W states.

The RTX path should update language embeddings when the state changes, while
reusing the already captured pipeline for the same prompt token length. This
script fixes the images and varies only state so any repeated latency spikes
come from prompt/state handling rather than image content.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flash_rt
from flash_rt.core.utils.pi05_prompt import PI05_STATE_PROMPT_MAX_LEN, format_pi05_prompt
from flash_rt.utils.paligemma_tokenizer import load_paligemma_sentencepiece


IMAGE_KEYS = ("image", "left_wrist_image", "right_wrist_image")


def _read_arrow(path: Path):
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all()


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    candidates = [dataset_root / "meta" / "tasks.jsonl"]
    cache_info = dataset_root / "meta" / "cache_info.json"
    if cache_info.exists():
        try:
            source_root = Path(json.loads(cache_info.read_text())["source_root"])
            candidates.insert(0, source_root / "meta" / "tasks.jsonl")
        except Exception:
            pass

    tasks: dict[int, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tasks[int(item["task_index"])] = str(item["task"])
        if tasks:
            break
    return tasks


def _episode_path(dataset: Path, episode: int) -> Path:
    path = dataset / "episodes" / f"episode_{episode:06d}.arrow"
    if path.exists():
        return path
    paths = sorted((dataset / "episodes").glob("episode_*.arrow"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.arrow files under {dataset / 'episodes'}")
    if episode >= len(paths):
        raise FileNotFoundError(f"episode {episode} not found under {dataset / 'episodes'}")
    return paths[episode]


def _image(row_value) -> np.ndarray:
    data = row_value.as_py()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != 224 * 224 * 3:
        raise ValueError(f"expected raw 224x224x3 RGB bytes, got {arr.size} bytes")
    return arr.reshape(224, 224, 3).copy()


def _state(row_value) -> np.ndarray:
    return np.asarray(row_value.as_py(), dtype=np.float32)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _make_prompt_len_fn():
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer

        tokenizer = PaligemmaTokenizer(max_len=PI05_STATE_PROMPT_MAX_LEN)

        def _openpi_len(prompt: str, state: np.ndarray) -> int:
            _, mask = tokenizer.tokenize(prompt, state=state)
            return int(np.asarray(mask).sum())

        return _openpi_len
    except Exception:
        sp = load_paligemma_sentencepiece()

        def _sp_len(prompt: str, state: np.ndarray) -> int:
            text = format_pi05_prompt(prompt, state)
            tokens = sp.encode(text, out_type=int, add_bos=True)
            return min(len(tokens), PI05_STATE_PROMPT_MAX_LEN)

        return _sp_len


def _tokenizer_len(prompt_len_fn, prompt: str, state: np.ndarray) -> int:
    return int(prompt_len_fn(prompt, state))


def _select_rows(rows_by_len: dict[int, list[int]], mode: str, count: int) -> list[int]:
    if mode == "same-bucket":
        _, rows = max(rows_by_len.items(), key=lambda kv: len(kv[1]))
        return rows[:count]

    selected: list[int] = []
    ordered = sorted(rows_by_len.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for _, rows in ordered:
        if rows:
            selected.append(rows[0])
        if len(selected) >= count:
            return selected
    i = 1
    while len(selected) < count:
        added = False
        for _, rows in ordered:
            if i < len(rows):
                selected.append(rows[i])
                added = True
                if len(selected) >= count:
                    return selected
        if not added:
            break
        i += 1
    return selected


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(values)
    p50 = sorted_values[len(sorted_values) // 2]
    p90 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.9))]
    p95 = sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))]
    return {
        "min": min(values),
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _print_stats(name: str, values: list[float]) -> None:
    s = _stats(values)
    if not s:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: min={s['min']:.2f} ms, p50={s['p50']:.2f} ms, "
        f"p90={s['p90']:.2f} ms, p95={s['p95']:.2f} ms, "
        f"mean={s['mean']:.2f} ms, max={s['max']:.2f} ms"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/peng.song/vla/small-vla/spi/datasets/MERGED_0303_posneg_224_arrow"),
    )
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--prompt", default=None, help="Override dataset task prompt.")
    p.add_argument("--mode", choices=("same-bucket", "mixed-bucket"), default="same-bucket")
    p.add_argument("--num-states", type=int, default=40)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num-views", type=int, default=3)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--autotune", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fixed-state-prompt-len", type=int, default=None,
                   help="Use one fixed runtime prompt length for state prompts, e.g. 200.")
    p.add_argument(
        "--threshold-ms",
        type=float,
        default=500.0,
        help="Fail if a steady-state same-bucket call exceeds this latency.",
    )
    p.add_argument("--fp8", action="store_true", help="Enable FP8. Default is BF16 for RTX 4090.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    episode_path = _episode_path(args.dataset, args.episode)
    table = _read_arrow(episode_path)
    tasks = _load_tasks(args.dataset)
    task_index = int(table["task_index"][0].as_py()) if "task_index" in table.column_names else -1
    prompt = args.prompt or tasks.get(task_index, "do something")
    prompt_len_fn = _make_prompt_len_fn()

    candidate_rows = list(range(0, table.num_rows, max(1, int(args.stride))))
    rows_by_len: dict[int, list[int]] = defaultdict(list)
    prompt_lens_by_row: dict[int, int] = {}
    for row in candidate_rows:
        state = _state(table["state"][row])
        prompt_len = _tokenizer_len(prompt_len_fn, prompt, state)
        rows_by_len[prompt_len].append(row)
        prompt_lens_by_row[row] = prompt_len

    selected = _select_rows(rows_by_len, args.mode, args.num_states)
    if not selected:
        raise RuntimeError(f"no states selected from {episode_path}")
    if len(selected) < args.num_states:
        print(f"warning: selected only {len(selected)} states, requested {args.num_states}")

    fixed_row = selected[0]
    images = [_image(table[key][fixed_row]) for key in IMAGE_KEYS[: args.num_views]]
    states = [_state(table["state"][row]) for row in selected]
    selected_lens = [prompt_lens_by_row[row] for row in selected]

    dist = Counter(selected_lens)
    print(f"episode: {episode_path}")
    print(f"prompt: {prompt!r}")
    print(f"mode: {args.mode}, selected={len(selected)}, fixed_image_row={fixed_row}")
    print(f"selected prompt_len distribution: {dict(sorted(dist.items()))}")

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="jax",
        config="pi05",
        hardware="rtx_sm89",
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        autotune=args.autotune,
        use_fp8=args.fp8,
        fixed_state_prompt_len=args.fixed_state_prompt_len,
    )
    _sync_cuda()
    print(f"model loaded in {(time.perf_counter() - t0):.2f}s")

    pipe = getattr(model, "_pipe", None)

    def debug_stats() -> dict[str, Any]:
        if pipe is not None and hasattr(pipe, "debug_prompt_stats"):
            return pipe.debug_prompt_stats()
        return {}

    print(f"initial prompt debug: {debug_stats()}")

    for i in range(min(args.warmup, len(states))):
        model.predict(images=images, prompt=prompt, state=states[i])
        _sync_cuda()
    print(f"after warmup prompt debug: {debug_stats()}")

    seen_lens: set[int] = set(debug_stats().get("cached_prompt_lens", []))
    records: list[dict[str, Any]] = []
    for i, (row, state, prompt_len) in enumerate(zip(selected, states, selected_lens)):
        was_seen = prompt_len in seen_lens
        before = debug_stats()
        start = time.perf_counter()
        actions = model.predict(images=images, prompt=prompt, state=state)
        _sync_cuda()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        after = debug_stats()
        seen_lens.add(prompt_len)
        build_delta = after.get("pipeline_build_count", 0) - before.get("pipeline_build_count", 0)
        hit_delta = after.get("pipeline_cache_hit_count", 0) - before.get("pipeline_cache_hit_count", 0)
        upload_delta = after.get("language_embed_upload_count", 0) - before.get("language_embed_upload_count", 0)
        finite = bool(np.isfinite(actions).all())
        record = {
            "i": i,
            "row": row,
            "prompt_len": prompt_len,
            "seen_bucket": was_seen,
            "ms": elapsed_ms,
            "build_delta": build_delta,
            "cache_hit_delta": hit_delta,
            "upload_delta": upload_delta,
            "finite": finite,
            "shape": tuple(np.asarray(actions).shape),
        }
        records.append(record)
        print(
            f"[{i:03d}] row={row:06d} len={prompt_len:3d} "
            f"{'reuse' if was_seen else 'new  '} {elapsed_ms:8.2f} ms "
            f"build+{build_delta} cache+{hit_delta} upload+{upload_delta} "
            f"shape={record['shape']} finite={finite}"
        )

    all_ms = [float(r["ms"]) for r in records]
    steady_ms = [float(r["ms"]) for r in records if r["seen_bucket"]]
    new_bucket_ms = [float(r["ms"]) for r in records if not r["seen_bucket"]]
    _print_stats("all timed calls", all_ms)
    _print_stats("new prompt_len buckets", new_bucket_ms)
    _print_stats("same-bucket steady calls", steady_ms)
    print(f"final prompt debug: {debug_stats()}")

    bad = [
        r for r in records
        if r["seen_bucket"] and (not r["finite"] or float(r["ms"]) > args.threshold_ms)
    ]
    if bad:
        print(f"FAIL: {len(bad)} same-bucket calls exceeded {args.threshold_ms:.1f} ms or returned non-finite actions")
        for r in bad[:10]:
            print(f"  bad[{r['i']}]: row={r['row']} len={r['prompt_len']} ms={r['ms']:.2f} finite={r['finite']}")
        return 1

    if args.mode == "same-bucket":
        builds = debug_stats().get("pipeline_build_count", 0)
        if builds > 1:
            print(f"FAIL: same-bucket mode built {builds} pipelines; expected 1")
            return 1

    print("PASS: RTX state-in-prompt updates reused cached same-length pipeline without recurring high latency.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
