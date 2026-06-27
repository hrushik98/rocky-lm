#!/usr/bin/env bash
# MLX only - fuse LoRA adapters into base weights.
# GPU path: fuse happens automatically at the end of train.py --device gpu
set -euo pipefail

set -a; source .env; set +a

MODEL="google/gemma-2-2b-it"
ADAPTER_DIR="./adapters"
SAVE_PATH="./rocky-gemma-2b"

echo "=== Fuse MLX LoRA Weights ==="
echo "Base:    $MODEL"
echo "Adapter: $ADAPTER_DIR"
echo "Output:  $SAVE_PATH"
echo ""

uv run python -m mlx_lm fuse \
    --model "$MODEL" \
    --adapter-path "$ADAPTER_DIR" \
    --save-path "$SAVE_PATH"

echo ""
echo "Fused model saved to $SAVE_PATH"
