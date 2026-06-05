#!/usr/bin/env python3
"""
FlashRT RTX 4090 JAX Pi0.5 smoke/benchmark CLI.

This is intentionally narrower than examples/quickstart.py:
it is for testing a Pi0.5 Orbax checkpoint on an RTX 4090 before moving
the same checkpoint to Jetson Thor.

Example:
    python examples/rtx4090_jax_pi05_cli.py \
        --checkpoint /path/to/pi05_orbax \
        --benchmark 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a Pi0.5 JAX/Orbax checkpoint on RTX 4090 (SM89)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Pi0.5 Orbax checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the red block and place it in the tray",
        help="Prompt used for the first prediction.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        choices=(1, 2, 3),
        help="Number of camera views to synthesize for the smoke test.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Action chunk length. Your H10W/OpenPI policy expects 50.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square dummy image size. Pi0.5 defaults to 224.",
    )
    parser.add_argument(
        "--autotune",
        type=int,
        default=5,
        help="CUDA Graph autotune trials. JAX usually benefits from 5.",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Timed predict() iterations after warmup. 0 disables timing.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup predict() iterations before benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for generated dummy images/state.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=0,
        help="Optional robot state dimension. 0 means do not pass state.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Clear FlashRT calibration and JAX weight cache before load.",
    )
    parser.add_argument(
        "--no-weight-cache",
        action="store_true",
        help="Disable the JAX FP8 weight cache for this run.",
    )
    parser.add_argument(
        "--no-fp8",
        action="store_true",
        help="Disable FP8 kernels and run the BF16 fallback path.",
    )
    parser.add_argument(
        "--fixed-state-prompt-len",
        type=int,
        default=None,
        help="Pi0.5 RTX only. Fixed runtime length for state prompts.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="bucketed",
        choices=("bucketed", "fixed", "openpi_masked_fixed200"),
        help="Pi0.5 RTX prompt runtime mode.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check imports, CUDA arch dispatch, and tokenizer presence.",
    )
    parser.add_argument(
        "--allow-non-4090",
        action="store_true",
        help="Do not fail if the detected FlashRT arch is not rtx_sm89.",
    )
    return parser.parse_args()


def configure_jax_env() -> None:
    # Keep JAX from grabbing almost all VRAM before FlashRT builds graphs.
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0",
    )


def tokenizer_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("FLASH_RT_PALIGEMMA_TOKENIZER")
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.extend(
        [
            Path("~/.cache/flash_rt/paligemma_tokenizer.model").expanduser(),
            Path("~/.cache/openpi/big_vision/paligemma_tokenizer.model").expanduser(),
            Path("/workspace/paligemma_tokenizer.model"),
        ]
    )
    return paths


def find_tokenizer() -> Path | None:
    for path in tokenizer_paths():
        if path.is_file():
            return path
    return None


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def main() -> int:
    args = parse_args()
    configure_jax_env()

    import numpy as np
    import flash_rt
    from flash_rt.hardware import _PIPELINE_MAP, detect_arch

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        print(f"ERROR: checkpoint does not exist: {checkpoint}", file=sys.stderr)
        return 2

    arch = detect_arch()
    print(f"flash_rt: {flash_rt.__version__}")
    print(f"checkpoint: {checkpoint}")
    print(f"detected arch: {arch}")

    if arch != "rtx_sm89":
        msg = "This CLI is intended for RTX 4090 / SM89 (rtx_sm89)."
        if not args.allow_non_4090:
            print(f"ERROR: {msg} Pass --allow-non-4090 to continue.", file=sys.stderr)
            return 3
        print(f"WARNING: {msg} Continuing because --allow-non-4090 was set.")

    dispatch = _PIPELINE_MAP.get(("pi05", "jax", arch))
    print(f"pi05/jax dispatch: {dispatch}")
    if dispatch is None:
        print(f"ERROR: no pi05/jax frontend registered for arch={arch}", file=sys.stderr)
        return 4

    tokenizer = find_tokenizer()
    if tokenizer is None:
        print("WARNING: paligemma_tokenizer.model was not found.")
        print("         Run: bash scripts/download_paligemma_tokenizer.sh")
    else:
        print(f"tokenizer: {tokenizer}")

    try:
        from flash_rt import flash_rt_kernels as fvk
        print(f"flash_rt_kernels: OK, has_cutlass_sm100={fvk.has_cutlass_sm100()}")
    except Exception as exc:
        print(f"ERROR: cannot import flash_rt_kernels: {exc}", file=sys.stderr)
        return 5

    try:
        from flash_rt import flash_rt_fa2 as fa2
        print(
            "flash_rt_fa2: OK, "
            f"fwd_fp16={callable(fa2.fwd_fp16)}, "
            f"fwd_bf16={callable(fa2.fwd_bf16)}"
        )
    except Exception as exc:
        print(f"ERROR: cannot import flash_rt_fa2 on RTX: {exc}", file=sys.stderr)
        return 6

    if args.check_only:
        print("check-only: PASS")
        return 0

    rng = np.random.default_rng(args.seed)
    image_shape = (args.image_size, args.image_size, 3)
    images = [
        rng.integers(0, 255, image_shape, dtype=np.uint8)
        for _ in range(args.num_views)
    ]
    state = None
    if args.state_dim > 0:
        state = rng.uniform(-1.0, 1.0, size=(args.state_dim,)).astype(np.float32)

    print("loading model...")
    load_t0 = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=str(checkpoint),
        framework="jax",
        config="pi05",
        hardware="rtx_sm89" if arch == "rtx_sm89" else arch,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        autotune=args.autotune,
        recalibrate=args.recalibrate,
        weight_cache=not args.no_weight_cache,
        use_fp8=not args.no_fp8,
        fixed_state_prompt_len=args.fixed_state_prompt_len,
        prompt_mode=args.prompt_mode,
    )
    sync_cuda()
    print(f"model loaded in {time.perf_counter() - load_t0:.2f}s")

    print("running first predict()...")
    pred_t0 = time.perf_counter()
    try:
        actions = model.predict(images=images, prompt=args.prompt, state=state)
    except RuntimeError as exc:
        msg = str(exc)
        if "cuBLAS error" in msg and "code=15" in msg and not args.no_fp8:
            print(
                "ERROR: cuBLASLt rejected an FP8 GEMM shape on this GPU/toolkit.\n"
                "       Re-run with --no-fp8 to validate the JAX checkpoint via "
                "the BF16 fallback path.",
                file=sys.stderr,
            )
        raise
    sync_cuda()
    first_ms = (time.perf_counter() - pred_t0) * 1000.0
    finite = bool(np.isfinite(actions).all())
    print(
        "first predict: "
        f"{first_ms:.2f} ms, shape={actions.shape}, "
        f"finite={finite}, range=[{actions.min():.5f}, {actions.max():.5f}]"
    )
    if not finite:
        return 7

    reuse_t0 = time.perf_counter()
    actions2 = model.predict(images=images, state=state)
    sync_cuda()
    reuse_ms = (time.perf_counter() - reuse_t0) * 1000.0
    print(
        "reuse prompt: "
        f"{reuse_ms:.2f} ms, shape={actions2.shape}, "
        f"finite={bool(np.isfinite(actions2).all())}"
    )

    if args.benchmark <= 0:
        print("smoke: PASS")
        return 0

    print(f"warming up {args.warmup} iterations...")
    for _ in range(args.warmup):
        model.predict(images=images, state=state)
    sync_cuda()

    print(f"benchmarking {args.benchmark} iterations...")
    times_ms: list[float] = []
    for _ in range(args.benchmark):
        t0 = time.perf_counter()
        model.predict(images=images, state=state)
        sync_cuda()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    p50 = percentile(times_ms, 0.50)
    p90 = percentile(times_ms, 0.90)
    p95 = percentile(times_ms, 0.95)
    mean = sum(times_ms) / len(times_ms)
    print(
        "benchmark: "
        f"min={times_ms[0]:.2f} ms, p50={p50:.2f} ms, "
        f"p90={p90:.2f} ms, p95={p95:.2f} ms, "
        f"mean={mean:.2f} ms, max={times_ms[-1]:.2f} ms"
    )
    print("benchmark: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
