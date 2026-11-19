#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run --no-sync vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image":1,"video":0}' "$@"
