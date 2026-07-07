# How to setup VLA on a thor


## env

mkdir vla
cd vla
mkdir FlashRT
mkdir models

rsync -av --delete  --exclude .venv --exclude .git   --exclude build   --exclude '*.so'   --exclude '__pycache__'   --exclude tmp   --exclude .agents   --exclude .codex   ~/vla/FlashRT/   diana@10.8.24.139:~/vla/FlashRT/

rsync -av ~/vla/FlashRT/tmp/flash_rt    diana@10.8.24.139:~/.cache/

rsync -av --progress ~/vla/FlashRT/tmp/models/0629_dvt2_all  diana@10.8.24.139:~/vla/models/

export PATH=/usr/local/cuda-13.0/bin:$PATH
nvcc --version

cd ~/vla/FlashRT/
python3.12 -m venv .venv  (可选 sudo apt update， sudo apt install python3.12-venv)
source .venv/bin/activate
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

pip install -U pip setuptools wheel
pip install pybind11 cmake ninja "numpy>=1.24" safetensors sentencepiece \
  "transformers<4.56" pandas pillow pyarrow msgpack websockets
pip install -U "jax[cuda13]"
pip install -e ".[jax,server]"

cmake -B build -S . -DGPU_ARCH=110
cmake --build build -j"$(nproc)"

optional:
(重新编译前，可以手动删除旧有的编译结果)
ls -lh flash_rt/*.so build/*.so 2>/dev/null
rm -rf build
rm -f flash_rt/*.so


# offline calib
rsync -av --progress tmp/0629_all_calib_obs   diana@10.8.26.61:~/vla/FlashRT/tmp

cd ~/vla/FlashRT
source .venv/bin/activate
python examples/pi05_thor_offline_calibrate.py   --checkpoint ~/vla/models/0629_dvt2_all/89999   --obs-glob "tmp/0629_all_calib_obs/*.npz"   --num-views 3   --chunk-size 50   --prompt-mode openpi_masked_fixed200   --fixed-state-prompt-len 200   --policy-profile pi05_dvt2_fft_0605   --percentile 99.9   --max-samples 256   --clear-existing   --verbose



# launch service

cd ~/vla/FlashRT
source .venv/bin/activate
python examples/pi05_websocket_policy_server.py \
  --checkpoint ~/vla/models/0629_dvt2_all/89999  \
  --framework jax \
  --hardware thor \
  --num-views 3 \
  --chunk-size 50 \
  --prompt-mode openpi_masked_fixed200 \
  --fixed-state-prompt-len 200 \
  --policy-profile pi05_dvt2_fft_0605 \
  --robot-type dvt2 \
  --host 0.0.0.0 \
  --port 8001
