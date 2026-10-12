#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec uv run --no-sync vllm serve Qwen/Qwen3-0.6B \
    --gpu-memory-utilization 0.75 \
    --max-model-len 4096 "$@"
