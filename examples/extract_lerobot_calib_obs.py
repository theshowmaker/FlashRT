#!/usr/bin/env python3
"""Extract a small, task-balanced LeRobot observation set for calibration.

The output is one OpenPI/FlashRT-compatible observation ``.npz`` per frame:

    obs["observation/state"]
    obs["observation/image"]
    obs["observation/wrist_image_left"]
    obs["observation/wrist_image_right"]
    obs["prompt"]

It is designed for large LeRobot datasets: it reads metadata first, samples a
small set of frame indices across tasks/episodes/time, then decodes only the
selected frames.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("extract_lerobot_calib_obs")


PREFERRED_IMAGE_KEYS = (
    "observation.image",
    "observation.images.image",
    "image",
    "observation.exterior_image_1_left",
    "observation.wrist_image_left",
    "observation.wrist_image",
    "observation.images.wrist_image",
    "wrist_image",
    "left_wrist_image",
    "observation.wrist_image_right",
    "observation.images.wrist_image_right",
    "wrist_image_right",
    "right_wrist_image",
)

PREFERRED_STATE_KEYS = (
    "observation.state",
    "state",
    "observation/joint_position",
)

OUT_IMAGE_KEYS = (
    "observation/image",
    "observation/wrist_image_left",
    "observation/wrist_image_right",
)


@dataclass(frozen=True)
class Episode:
    episode_index: int
    task_index: int
    task_name: str
    length: int
    chunk_index: int | None
    file_index: int | None
    dataset_from_index: int | None
    dataset_to_index: int | None


@dataclass(frozen=True)
class Sample:
    episode: Episode
    frame_index: int
    global_index: int | None
    local_row: int | None
    data_path: Path


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_tasks(root: Path) -> dict[int, str]:
    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        out = {}
        for item in _read_jsonl(tasks_jsonl):
            task_index = int(item.get("task_index", len(out)))
            task = item.get("task", item.get("name", item.get("tasks", "")))
            out[task_index] = str(task)
        return out

    tasks_parquet = root / "meta" / "tasks.parquet"
    if tasks_parquet.exists():
        df = pq.read_table(tasks_parquet).to_pandas().reset_index()
        if "task" in df.columns:
            return {int(k): str(v) for k, v in zip(df["task_index"], df["task"])}
        if "tasks" in df.columns:
            return {int(k): str(v) for k, v in zip(df["task_index"], df["tasks"])}
        return {int(k): str(v) for k, v in zip(df["task_index"], df["index"])}

    return {}


def _task_name_to_index(tasks: dict[int, str], task_name: str) -> int:
    for task_index, name in tasks.items():
        if name == task_name:
            return int(task_index)
    if not task_name:
        return -1
    new_index = max(tasks) + 1 if tasks else 0
    tasks[new_index] = task_name
    return new_index


def _scalar(row: Any, key: str, default=None):
    if key not in row:
        return default
    return row[key]


def _first_task_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value)


def _load_episodes(root: Path, info: dict, tasks: dict[int, str]) -> list[Episode]:
    episodes: list[Episode] = []
    ep_dir = root / "meta" / "episodes"
    if ep_dir.exists():
        for parquet in sorted(ep_dir.rglob("*.parquet")):
            df = pq.read_table(parquet).to_pandas()
            for _, row in df.iterrows():
                task_name = _first_task_name(row.get("tasks", None))
                task_index = int(row["task_index"]) if "task_index" in row else _task_name_to_index(tasks, task_name)
                if not task_name:
                    task_name = tasks.get(task_index, "do something")
                episodes.append(Episode(
                    episode_index=int(row["episode_index"]),
                    task_index=task_index,
                    task_name=task_name,
                    length=int(row["length"]),
                    chunk_index=int(row["data/chunk_index"]) if "data/chunk_index" in row else None,
                    file_index=int(row["data/file_index"]) if "data/file_index" in row else None,
                    dataset_from_index=int(row["dataset_from_index"]) if "dataset_from_index" in row else None,
                    dataset_to_index=int(row["dataset_to_index"]) if "dataset_to_index" in row else None,
                ))
    else:
        ep_jsonl = root / "meta" / "episodes.jsonl"
        if not ep_jsonl.exists():
            raise FileNotFoundError(
                f"neither meta/episodes/*.parquet nor meta/episodes.jsonl exists under {root}")
        for item in _read_jsonl(ep_jsonl):
            task_values = item.get("tasks", [])
            task_name = _first_task_name(task_values)
            task_index = int(item.get("task_index", _task_name_to_index(tasks, task_name)))
            if not task_name:
                task_name = tasks.get(task_index, "do something")
            episodes.append(Episode(
                episode_index=int(item["episode_index"]),
                task_index=task_index,
                task_name=task_name,
                length=int(item["length"]),
                chunk_index=item.get("data/chunk_index"),
                file_index=item.get("data/file_index"),
                dataset_from_index=item.get("dataset_from_index"),
                dataset_to_index=item.get("dataset_to_index"),
            ))

    episodes.sort(key=lambda e: e.episode_index)
    if not episodes:
        raise RuntimeError(f"no episodes found under {root}")
    return episodes


def _format_data_path(root: Path, info: dict, episode: Episode) -> Path:
    template = str(info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    ))
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode.episode_index // chunks_size
    kwargs = {
        "episode_chunk": episode_chunk,
        "episode_index": episode.episode_index,
        "chunk_index": episode.chunk_index if episode.chunk_index is not None else episode_chunk,
        "file_index": episode.file_index if episode.file_index is not None else episode.episode_index,
    }
    return root / template.format(**kwargs)


def _compute_local_rows(root: Path, info: dict, episodes: list[Episode]) -> dict[int, int]:
    """Return local row base for each episode inside its data file."""
    by_file: dict[Path, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_file[_format_data_path(root, info, ep)].append(ep)

    bases: dict[int, int] = {}
    for eps in by_file.values():
        eps.sort(key=lambda e: (
            e.dataset_from_index if e.dataset_from_index is not None else e.episode_index,
            e.episode_index,
        ))
        cursor = 0
        for ep in eps:
            bases[ep.episode_index] = cursor
            cursor += ep.length
    return bases


def _image_feature_keys(info: dict) -> list[str]:
    feats = info.get("features", {})
    keys = [k for k, v in feats.items() if isinstance(v, dict) and v.get("dtype") == "image"]
    if not keys:
        return []
    order = {key: i for i, key in enumerate(PREFERRED_IMAGE_KEYS)}
    return sorted(keys, key=lambda k: (order.get(k, 10_000), k))


def _choose_image_keys(info: dict, requested: str | None, num_views: int) -> list[str]:
    if requested:
        keys = [k.strip() for k in requested.split(",") if k.strip()]
    else:
        keys = _image_feature_keys(info)
    if not keys:
        raise ValueError(
            "could not auto-detect image columns from meta/info.json; pass --image-keys")
    if len(keys) < num_views:
        logger.warning(
            "Only %d image columns found for num_views=%d; repeating the last view",
            len(keys), num_views)
        keys = keys + [keys[-1]] * (num_views - len(keys))
    return keys[:num_views]


def _choose_state_key(info: dict, requested: str | None) -> str:
    if requested:
        return requested
    feats = info.get("features", {})
    for key in PREFERRED_STATE_KEYS:
        if key in feats:
            return key
    for key, value in feats.items():
        if "state" in key and isinstance(value, dict):
            return key
    raise ValueError("could not auto-detect state column; pass --state-key")


def _decode_image(cell: Any, *, root: Path, image_size: int) -> np.ndarray:
    raw = None
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw is None and cell.get("path"):
            image_path = Path(cell["path"])
            if not image_path.is_absolute():
                image_path = root / image_path
            raw = image_path.read_bytes()
    elif isinstance(cell, (bytes, bytearray)):
        raw = bytes(cell)
    elif isinstance(cell, np.ndarray):
        arr = cell
        if arr.ndim == 3:
            return _to_hwc_uint8(arr, image_size=image_size)

    if raw is None:
        raise TypeError(f"unsupported image cell type: {type(cell).__name__}")

    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _to_hwc_uint8(arr: np.ndarray, *, image_size: int) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim != 3:
        raise ValueError(f"image must be rank-3, got shape={out.shape}")
    if out.shape[0] in (1, 3) and out.shape[-1] not in (1, 3):
        out = np.transpose(out, (1, 2, 0))
    if out.dtype != np.uint8:
        if np.issubdtype(out.dtype, np.floating):
            max_v = float(np.nanmax(out)) if out.size else 0.0
            if max_v <= 1.5:
                out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.shape[:2] != (image_size, image_size):
        from PIL import Image
        img = Image.fromarray(out).resize((image_size, image_size), Image.BILINEAR)
        out = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(out)


def _sample_frames(
    episodes: list[Episode],
    *,
    count: int,
    max_episodes_per_task: int,
    seed: int,
) -> list[tuple[Episode, int]]:
    rng = np.random.default_rng(seed)
    by_task: dict[int, list[Episode]] = defaultdict(list)
    for ep in episodes:
        if ep.length > 0:
            by_task[ep.task_index].append(ep)

    tasks = sorted(by_task)
    rng.shuffle(tasks)
    if not tasks:
        raise RuntimeError("no non-empty episodes to sample")

    per_task = {task: count // len(tasks) for task in tasks}
    for task in tasks[:count % len(tasks)]:
        per_task[task] += 1

    samples: list[tuple[Episode, int]] = []
    for task in tasks:
        target = per_task[task]
        if target <= 0:
            continue
        eps = sorted(by_task[task], key=lambda e: e.episode_index)
        if len(eps) > max_episodes_per_task:
            positions = np.linspace(0, len(eps) - 1, max_episodes_per_task)
            eps = [eps[int(round(p))] for p in positions]
        rng.shuffle(eps)
        frames_per_ep = max(1, math.ceil(target / len(eps)))
        task_samples: list[tuple[Episode, int]] = []
        for ep in eps:
            k = min(frames_per_ep, ep.length)
            if k <= 0:
                continue
            # Stratify within the episode timeline and jitter inside each bin.
            bins = (np.arange(k) + rng.random(k)) / k
            frames = np.unique(np.clip((bins * ep.length).astype(np.int64), 0, ep.length - 1))
            for frame in frames.tolist():
                task_samples.append((ep, int(frame)))
                if len(task_samples) >= target:
                    break
            if len(task_samples) >= target:
                break
        samples.extend(task_samples)

    # Fill any shortfall from all episodes.
    seen = {(s[0].episode_index, s[1]) for s in samples}
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        ep = episodes[int(rng.integers(0, len(episodes)))]
        if ep.length <= 0:
            attempts += 1
            continue
        frame = int(rng.integers(0, ep.length))
        key = (ep.episode_index, frame)
        if key not in seen:
            samples.append((ep, frame))
            seen.add(key)
        attempts += 1

    samples = samples[:count]
    samples.sort(key=lambda s: (s[0].task_index, s[0].episode_index, s[1]))
    return samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("tmp/lerobot_calib_obs"))
    p.add_argument("--count", type=int, default=256)
    p.add_argument("--num-views", type=int, default=3, choices=(1, 2, 3))
    p.add_argument("--image-keys", default=None,
                   help="Comma-separated source image columns. Auto-detected from meta/info.json by default.")
    p.add_argument("--state-key", default=None,
                   help="Source state column. Auto-detected from meta/info.json by default.")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-episodes-per-task", type=int, default=8)
    p.add_argument("--require-state-dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    root = args.dataset
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found: {info_path}")
    info = json.loads(info_path.read_text())
    tasks = _load_tasks(root)
    episodes = _load_episodes(root, info, tasks)
    image_keys = _choose_image_keys(info, args.image_keys, args.num_views)
    state_key = _choose_state_key(info, args.state_key)
    local_bases = _compute_local_rows(root, info, episodes)

    logger.info("dataset=%s episodes=%d tasks=%d", root, len(episodes), len({e.task_index for e in episodes}))
    logger.info("image_keys=%s state_key=%s", image_keys, state_key)

    planned_pairs = _sample_frames(
        episodes,
        count=args.count,
        max_episodes_per_task=max(1, args.max_episodes_per_task),
        seed=args.seed,
    )
    samples: list[Sample] = []
    for ep, frame in planned_pairs:
        data_path = _format_data_path(root, info, ep)
        if not data_path.exists():
            raise FileNotFoundError(f"data parquet not found for episode {ep.episode_index}: {data_path}")
        global_index = (
            ep.dataset_from_index + frame
            if ep.dataset_from_index is not None else None
        )
        local_row = local_bases.get(ep.episode_index)
        if local_row is not None:
            local_row += frame
        samples.append(Sample(ep, frame, global_index, local_row, data_path))

    manifest = {
        "source_dataset": str(root),
        "count": len(samples),
        "num_views": args.num_views,
        "image_keys": image_keys,
        "state_key": state_key,
        "image_size": args.image_size,
        "seed": args.seed,
        "max_episodes_per_task": args.max_episodes_per_task,
        "entries": [],
    }

    if args.dry_run:
        by_task = defaultdict(int)
        for sample in samples:
            by_task[sample.episode.task_index] += 1
        logger.info("dry run: selected %d samples over %d tasks", len(samples), len(by_task))
        for task_index, n in sorted(by_task.items()):
            logger.info("  task %s: %d samples prompt=%r", task_index, n, tasks.get(task_index, ""))
        return 0

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.out_dir} is not empty; pass --overwrite or choose another --out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples_by_file: dict[Path, list[Sample]] = defaultdict(list)
    for sample in samples:
        samples_by_file[sample.data_path].append(sample)

    saved = 0
    needed_cols = list(dict.fromkeys([state_key, *image_keys]))
    for data_path, file_samples in sorted(samples_by_file.items()):
        logger.info("reading %s (%d selected frames)", data_path, len(file_samples))
        df = pq.read_table(data_path, columns=needed_cols).to_pandas()
        for sample in file_samples:
            if sample.local_row is None:
                raise RuntimeError(
                    "cannot locate local row for this dataset layout; "
                    "please use LeRobot v3 episode metadata with dataset_from_index")
            row = df.iloc[int(sample.local_row)]
            state = np.asarray(row[state_key], dtype=np.float32).reshape(-1)
            if args.require_state_dim is not None and state.size != args.require_state_dim:
                raise ValueError(
                    f"state dim mismatch at episode={sample.episode.episode_index} "
                    f"frame={sample.frame_index}: got {state.size}, "
                    f"expected {args.require_state_dim}")
            images = [
                _decode_image(row[key], root=root, image_size=args.image_size)
                for key in image_keys
            ]
            prompt = sample.episode.task_name or tasks.get(sample.episode.task_index, "do something")
            obs = {
                "observation/state": state,
                "prompt": prompt,
            }
            for out_key, image in zip(OUT_IMAGE_KEYS, images):
                obs[out_key] = image

            out = args.out_dir / (
                f"task{sample.episode.task_index:04d}_"
                f"episode{sample.episode.episode_index:06d}_"
                f"frame{sample.frame_index:06d}.npz"
            )
            np.savez_compressed(
                out,
                obs=np.asarray(obs, dtype=object),
                episode_index=np.asarray(sample.episode.episode_index, dtype=np.int64),
                frame_index=np.asarray(sample.frame_index, dtype=np.int64),
                task_index=np.asarray(sample.episode.task_index, dtype=np.int64),
                prompt=np.asarray(prompt),
                source_path=np.asarray(str(data_path)),
                source_local_row=np.asarray(sample.local_row, dtype=np.int64),
                source_global_index=np.asarray(-1 if sample.global_index is None else sample.global_index, dtype=np.int64),
            )
            manifest["entries"].append({
                "path": str(out),
                "episode_index": sample.episode.episode_index,
                "frame_index": sample.frame_index,
                "task_index": sample.episode.task_index,
                "prompt": prompt,
                "source_path": str(data_path),
                "source_local_row": int(sample.local_row),
                "source_global_index": sample.global_index,
            })
            saved += 1

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("saved %d observations to %s", saved, args.out_dir)
    logger.info("manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
