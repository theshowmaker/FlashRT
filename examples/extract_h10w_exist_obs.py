#!/usr/bin/env python3
"""Extract H10W observations with a chosen exist_label from Arrow episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    candidates = [
        dataset_root / "meta" / "tasks.jsonl",
        Path("/home/peng.song/nfs-share/peng.song/hl-policy/MERGED_0303_posneg/meta/tasks.jsonl"),
    ]
    cache_info = dataset_root / "meta" / "cache_info.json"
    if cache_info.exists():
        try:
            source_root = Path(json.loads(cache_info.read_text())["source_root"])
            candidates.insert(0, source_root / "meta" / "tasks.jsonl")
        except Exception:
            pass

    for path in candidates:
        if not path.exists():
            continue
        tasks = {}
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                tasks[int(item["task_index"])] = str(item["task"])
        if tasks:
            return tasks
    return {}


def _read_arrow(path: Path):
    with pa.memory_map(str(path), "r") as source:
        try:
            return ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return ipc.open_stream(source).read_all()


def _image(row_value) -> np.ndarray:
    data = row_value.as_py()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != 224 * 224 * 3:
        raise ValueError(f"expected raw 224x224x3 RGB bytes, got {arr.size} bytes")
    return arr.reshape(224, 224, 3).copy()


def _state(row_value) -> np.ndarray:
    return np.asarray(row_value.as_py(), dtype=np.float32)


def _obs_from_row(table, row: int, prompt: str) -> dict:
    return {
        "observation/state": _state(table["state"][row]),
        "observation/image": _image(table["image"][row]),
        "observation/wrist_image_left": _image(table["left_wrist_image"][row]),
        "observation/wrist_image_right": _image(table["right_wrist_image"][row]),
        "prompt": prompt,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/peng.song/vla/small-vla/spi/datasets/MERGED_0303_posneg_224_arrow"),
    )
    p.add_argument("--exist", default="1", choices=("0", "1", "any"),
                   help="Filter by exist_label, or use 'any' for unfiltered real sequences.")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=Path("tmp/h10w_exist_obs"))
    p.add_argument("--prompt", default=None, help="Override prompt for all samples.")
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--stride", type=int, default=25,
                   help="Keep at most one matching frame every N frames per episode.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    episodes_dir = args.dataset / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"episodes dir not found: {episodes_dir}")

    tasks = _load_tasks(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    saved = 0
    exist_filter = None if args.exist == "any" else int(args.exist)
    exist_name = "any" if exist_filter is None else str(exist_filter)

    for path in sorted(episodes_dir.glob("episode_*.arrow")):
        try:
            episode_number = int(path.stem.split("_")[-1])
        except ValueError:
            episode_number = -1
        if episode_number < args.start_episode:
            continue

        table = _read_arrow(path)
        labels = np.asarray(table["exist_label"].to_pylist(), dtype=np.int8)
        rows = np.arange(len(labels), dtype=np.int64) if exist_filter is None else np.flatnonzero(labels == exist_filter)
        if args.stride > 1 and len(rows):
            kept = []
            last = -args.stride
            for row in rows:
                if int(row) - last >= args.stride:
                    kept.append(row)
                    last = int(row)
            rows = np.asarray(kept, dtype=np.int64)
        for row in rows:
            task_index = int(table["task_index"][int(row)].as_py())
            prompt = args.prompt or tasks.get(task_index, "do something")
            obs = _obs_from_row(table, int(row), prompt)
            exist_label = int(labels[int(row)])
            episode_index = int(table["episode_index"][int(row)].as_py())
            frame_index = int(table["frame_index"][int(row)].as_py())
            out = args.out_dir / (
                f"exist{exist_label}_episode{episode_index:06d}_frame{frame_index:06d}.npz"
            )
            np.savez_compressed(
                out,
                obs=np.array(obs, dtype=object),
                exist_label=np.asarray(exist_label, dtype=np.int32),
                episode_index=np.asarray(episode_index, dtype=np.int64),
                frame_index=np.asarray(frame_index, dtype=np.int64),
                task_index=np.asarray(task_index, dtype=np.int64),
                prompt=np.asarray(prompt),
            )
            manifest.append(
                {
                    "path": str(out),
                    "exist_label": exist_label,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "task_index": task_index,
                    "prompt": prompt,
                }
            )
            saved += 1
            print(f"saved {out} prompt={prompt!r}")
            if saved >= args.count:
                manifest_path = args.out_dir / f"manifest_exist{exist_name}.json"
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"manifest: {manifest_path}")
                return 0

    label_msg = "any exist_label" if exist_filter is None else f"exist_label={exist_filter}"
    raise SystemExit(f"only found {saved} samples with {label_msg}")


if __name__ == "__main__":
    raise SystemExit(main())
