# temp
python examples/pi05_websocket_policy_server.py   --checkpoint ~/vla/models/exp_0303_posneg/29999   --framework jax   --hardware thor   --num-views 3   --chunk-size 50   --autotune 5   --host 0.0.0.0   --port 8001

uv run examples/simple_client/main.py   --env H10W_0303   --host 10.3.43.38   --port 8001