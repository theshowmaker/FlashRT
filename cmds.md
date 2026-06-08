# FlashRT / OpenPI validation commands

## 4090 local build

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-4

/usr/local/cuda-12.4/bin/nvcc --version

然后只在当前 shell 里设置，不写进全局 bashrc：

cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

cmake -B build -S . -DGPU_ARCH=89 \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

cmake --build build -j$(nproc)
```

```bash
cd /home/peng.song/vla/FlashRT

python3.12 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install pybind11 cmake "numpy>=1.24" safetensors sentencepiece "transformers<4.56" pandas pillow pyarrow msgpack websockets
pip install -e ".[torch,jax,server]"
pip install -U "jax[cuda12]"

git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass

bash scripts/download_paligemma_tokenizer.sh

cmake -B build -S . -DGPU_ARCH=89
cmake --build build -j"$(nproc)"
```

## 4090 quick model check

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 FVK_PI05_RTX_FORCE_BF16=1 \
python examples/quickstart.py \
  --checkpoint /home/peng.song/ksyun_server/models/exp_0303_posneg/29999 \
  --framework jax \
  --config pi05 \
  --hardware rtx_sm89 \
  --num_views 3 \
  --chunk_size 50 \
  --autotune 5 \
  --benchmark 20 \
  --warmup 50
```

## OpenPI baseline service on 4090

```bash
cd /home/peng.song/vla/openpi

CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py \
  --env H10W_0303 \
  --port 8000
```

## Thor environment setup

Run this on Thor from a fresh checkout. Use Python 3.12, not Python 3.13.

```bash
cd ~/vla/FlashRT

python3.12 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install pybind11 cmake "numpy>=1.24" safetensors sentencepiece "transformers<4.56" pandas pillow pyarrow msgpack websockets
pip install -e ".[jax,server]"
pip install -U "jax[cuda13]"

git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass

./scripts/download_paligemma_tokenizer.sh

cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j"$(nproc)"
```

Quick checks on Thor:

```bash
cd ~/vla/FlashRT
source .venv/bin/activate

python - <<'PY'
import jax
import flash_rt.flash_rt_kernels as k
import flash_rt.flash_rt_fp4 as fp4
print("jax devices:", jax.devices())
print("prefix masked attention:", hasattr(k, "attention_qkv_fp16_prefix_masked"))
print("fp4 available:", fp4.has_nvfp4())
PY
```

For Pi0.5 Thor FP4, no extra pip package is needed beyond the normal FlashRT
environment. `flash_rt_fp4.so` is built by CMake when `GPU_ARCH=110`. The
optional `.[thor-fa4]` dependency is for LingBot FA4 attention experiments, not
for Pi0.5 `--use-fp4`.

## Thor FlashRT service

Run this on Thor.

```bash
cd ~/vla/FlashRT
source .venv/bin/activate

python examples/pi05_websocket_policy_server.py \
  --checkpoint ~/vla/models/exp_0303_posneg/29999 \
  --framework jax \
  --hardware thor \
  --num-views 3 \
  --chunk-size 50 \
  --prompt-mode openpi_masked_fixed200 \
  --autotune 5 \
  --host 0.0.0.0 \
  --port 8001
```

Expected metadata includes:

```text
prompt_mode=openpi_masked_fixed200
openpi_masked_prefix=True
prompt_mask_supported=True
fast_state_tokenizer=True
action_shape=[50, 17]
```

## Thor FlashRT service, FP4 candidate

Run this on Thor if `flash_rt_fp4.has_nvfp4()` is `True`.

```bash
cd ~/vla/FlashRT
source .venv/bin/activate

python examples/pi05_websocket_policy_server.py \
  --checkpoint ~/vla/models/exp_0303_posneg/29999 \
  --framework jax \
  --hardware thor \
  --num-views 3 \
  --chunk-size 50 \
  --prompt-mode openpi_masked_fixed200 \
  --use-fp4 \
  --autotune 5 \
  --host 0.0.0.0 \
  --port 8001
```

Expected metadata additionally includes:

```text
use_fp4=True
```

## OpenPI client latency test against Thor

Run this on the 4090 workstation. Use Thor wired IP for lower jitter.

```bash
cd /home/peng.song/vla/openpi

uv run examples/simple_client/main.py \
  --env H10W_0303 \
  --host 10.8.26.61 \
  --port 8001
```

## Extract exist=0 observations

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

./.venv/bin/python examples/extract_h10w_exist_obs.py \
  --dataset /home/peng.song/vla/small-vla/spi/datasets/MERGED_0303_posneg_224_arrow/ \
  --exist 0 \
  --count 100 \
  --out-dir tmp/h10w_exist0_obs
```

## Extract exist=1 observations

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

./.venv/bin/python examples/extract_h10w_exist_obs.py \
  --dataset /home/peng.song/vla/small-vla/spi/datasets/MERGED_0303_posneg_224_arrow/ \
  --exist 1 \
  --count 100 \
  --out-dir tmp/h10w_exist1_obs
```

## Compare OpenPI vs FlashRT Thor, exist=1 short run

Requires OpenPI service on `127.0.0.1:8000` and FlashRT Thor service on `10.8.26.61:8001`.

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

./.venv/bin/python examples/compare_openpi_flashrt_outputs.py \
  --openpi-host 127.0.0.1 \
  --openpi-port 8000 \
  --flashrt-host 10.8.26.61 \
  --flashrt-port 8001 \
  --obs-glob 'tmp/h10w_longrun_obs/*.npz' \
  --steps 100 \
  --require-exist 1 \
  --summary-skip 1 \
  --save tmp/openpi_flashrt_exist1_thor_fixed200_mask_compare.npz
```

## Compare OpenPI vs FlashRT Thor, exist=0

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

./.venv/bin/python examples/compare_openpi_flashrt_outputs.py \
  --openpi-host 127.0.0.1 \
  --openpi-port 8000 \
  --flashrt-host 10.8.26.61 \
  --flashrt-port 8001 \
  --obs-glob 'tmp/h10w_exist0_obs/*.npz' \
  --steps 100 \
  --require-exist 0 \
  --summary-skip 1 \
  --save tmp/openpi_flashrt_exist0_thor_fixed200_mask_compare.npz
```

## Long-run stability test

Use this after preparing at least 500 or 1000 real observation files. The
current retained long-run observation set is `tmp/h10w_longrun_obs`.

```bash
cd /home/peng.song/vla/FlashRT
source .venv/bin/activate

./.venv/bin/python examples/compare_openpi_flashrt_outputs.py \
  --openpi-host 127.0.0.1 \
  --openpi-port 8000 \
  --flashrt-host 10.8.26.61 \
  --flashrt-port 8001 \
  --obs-glob 'tmp/h10w_longrun_obs/*.npz' \
  --steps 1000 \
  --summary-skip 1 \
  --save tmp/openpi_flashrt_longrun_fp8_thor_fixed200_mask_compare.npz
```

For the FP4 candidate service, use the same command but save to:

```bash
./.venv/bin/python examples/compare_openpi_flashrt_outputs.py \
  --openpi-host 127.0.0.1 \
  --openpi-port 8000 \
  --flashrt-host 10.8.26.61 \
  --flashrt-port 8001 \
  --obs-glob 'tmp/h10w_longrun_obs/*.npz' \
  --steps 1000 \
  --summary-skip 1 \
  --save tmp/openpi_flashrt_longrun_fp4_thor_fixed200_mask_compare.npz
```

Check that:

```text
policy_calibrate_ms = 0
policy_set_prompt_ms is about 1-2 ms
no 800 ms / second-level latency spikes
exist labels match
```

## Sync code to Thor

Do not copy local build outputs or x86 `.so` files to Thor.

```bash
rsync -av \
  --exclude .venv \
  --exclude build \
  --exclude '*.so' \
  --exclude '__pycache__' \
  --exclude tmp \
  /home/peng.song/vla/FlashRT/ \
  diana@10.8.26.61:~/vla/FlashRT/
```

Then rebuild on Thor.

```bash
cd ~/vla/FlashRT
source .venv/bin/activate

cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j"$(nproc)"
```

After rebuilding, verify the new masked attention binding exists:

```bash
cd ~/vla/FlashRT
source .venv/bin/activate

python - <<'PY'
import flash_rt.flash_rt_kernels as k
print(hasattr(k, "attention_qkv_fp16_prefix_masked"))
PY
```
