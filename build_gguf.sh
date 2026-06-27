#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="./llama.cpp"
MODEL_DIR="./rocky-gemma-2b"
F16_GGUF="./rocky-f16.gguf"
Q4_GGUF="./rocky-q4_k_m.gguf"

echo "=== Phase 5: Build GGUF ==="

# Clone llama.cpp if not already present
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

echo "Installing llama.cpp Python requirements..."
uv pip install --index-strategy unsafe-best-match -r requirements.txt -q

echo "Compiling llama.cpp binaries..."
cmake -B build
cmake --build build --config Release -j"$(sysctl -n hw.logicalcpu)"

echo ""
echo "Converting to FP16 GGUF..."
../.venv/bin/python convert_hf_to_gguf.py "../$MODEL_DIR" \
    --outtype f16 \
    --outfile "../$F16_GGUF"

echo "Quantizing to Q4_K_M..."
./build/bin/llama-quantize "../$F16_GGUF" "../$Q4_GGUF" Q4_K_M

cd ..

echo ""
echo "Done."
echo "  FP16:   $F16_GGUF"
echo "  Q4_K_M: $Q4_GGUF  (copy this to your phone)"
