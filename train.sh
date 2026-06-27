#!/usr/bin/env bash
set -euo pipefail

# Load .env so HF_TOKEN is available for gated model downloads
set -a; source .env; set +a

uv run python src/train.py "$@"
