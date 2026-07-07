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

# check

.venv/bin/python examples/compare_openpi_flashrt_outputs.py   --openpi-host 127.0.0.1   --openpi-port 8000   --flashrt-host 10.8.24.114   --flashrt-port 8001   --obs-glob "tmp/0603_dvt2_sofa_episode_000000_obs_npz/*.npz"   --steps 157   --require-action-shape 50,16   --require-exist-match   --require-stage-match   --summary-skip 1   --no-early-fail   --save tmp/openpi_vs_flashrt_79999_robot_obs_nodebug.npz

.venv/bin/python examples/compare_openpi_flashrt_outputs.py   --openpi-host 127.0.0.1   --openpi-port 8000   --flashrt-host 10.8.24.183   --flashrt-port 8001   --obs-glob "tmp/robot_obs_record/*.npz"   --steps 19   --require-action-shape 50,16   --require-exist-match   --require-stage-match   --summary-skip 1   --no-early-fail   --save tmp/openpi_vs_flashrt_79999_robot_obs_nodebug.npz



# 0704 models
rsync -av --delete   --exclude .git   --exclude .venv   --exclude build   --exclude '*.so'   --exclude '__pycache__'   --exclude tmp   --exclude .agents   --exclude .codex   /home/peng.song/vla/FlashRT/   diana@10.8.26.61:~/vla/FlashRT/

peng.song@ubuntu-22-peng-song:/home/peng.song/ksyun_server/models/dvt2$ rsync -av --exclude='train_state' peng@120.92.116.251:/DATA/disk0/yuhao.song/checkpoints/pi05_dvt2_fft_0704/0704_dvt2_all/50000 ./0704_dvt2_all/

peng.song@ubuntu-22-peng-song:/home/peng.song/ksyun_server/models/dvt2$ rsync -av --progress ./0704_dvt2_all  diana@10.8.26.61:~/vla/models/

rsync -av --progress  peng@120.92.116.251:~/peng.song/high-level-policy/tmp4FlashRT/tmp/0704_all_calib_obs ~/vla/FlashRT/tmp/

rsync -av --progress /home/peng.song/vla/FlashRT/tmp/0704_all_calib_obs   diana@10.8.26.61:~/vla/FlashRT/tmp/



(.venv) diana@localhost:~/vla/FlashRT$ CUDA_VISIBLE_DEVICES=0 python examples/pi05_thor_offline_calibrate.py   --checkpoint ~/vla/models/0704_dvt2_all/50000   --obs-glob "tmp/0704_all_calib_obs/*.npz"   --num-views 3   --chunk-size 50   --prompt-mode openpi_masked_fixed200   --fixed-state-prompt-len 200   --policy-profile auto   --percentile 99.9   --max-samples 256   --clear-existing   --verbose

(.venv) diana@localhost:~/vla/FlashRT$ CUDA_VISIBLE_DEVICES=0 python examples/pi05_websocket_policy_server.py   --checkpoint ~/vla/models/0704_dvt2_all/50000   --framework jax   --hardware thor   --num-views 3   --chunk-size 50   --prompt-mode openpi_masked_fixed200   --fixed-state-prompt-len 200   --policy-profile auto   --robot-type dvt2   --host 0.0.0.0   --port 8001   --log-infer-ms

peng.song@ubuntu-22-peng-song:/home/peng.song/vla/openpi$ CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py   --env H10W_DUAL3   --port 8000





python examples/compare_openpi_flashrt_outputs.py   --openpi-host 127.0.0.1   --openpi-port 8000   --flashrt-host 10.8.26.61   --flashrt-port 8001   --obs-glob "tmp/robot_obs_record/*.npz"   --require-action-shape 50,16   --require-stage-match   --summary-skip 1   --no-early-fail   --save tmp/openpi_vs_flashrt_thor_0629_compare_full.npz --steps 19


python examples/compare_openpi_flashrt_outputs.py   --openpi-host 127.0.0.1   --openpi-port 8000   --flashrt-host 10.8.26.61   --flashrt-port 8001   --obs-glob "tmp/0603_dvt2_sofa_episode_000000_obs_npz/*.npz"   --require-action-shape 50,16   --require-stage-match   --summary-skip 1   --no-early-fail   --save tmp/openpi_vs_flashrt_thor_0629_compare_full.npz --steps 157

