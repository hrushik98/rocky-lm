import os
from pathlib import Path
from mediapipe.tasks.python.genai import converter

# Paths
INPUT_CKPT = "/Users/hrushik/rocky-lm/rocky-gemma-2b"
VOCAB_MODEL_FILE = "/Users/hrushik/rocky-lm/rocky-gemma-2b/tokenizer.model"
OUTPUT_DIR = "/Users/hrushik/rocky-lm/rocky-task-output"
OUTPUT_FILE = "/Users/hrushik/rocky-lm/rocky.task"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("Starting MediaPipe conversion to .task format...")
print(f"Input checkpoint: {INPUT_CKPT}")
print(f"Vocab model file: {VOCAB_MODEL_FILE}")
print(f"Output file:      {OUTPUT_FILE}")

config = converter.ConversionConfig(
    input_ckpt=INPUT_CKPT,
    ckpt_format="safetensors",
    model_type="GEMMA2_2B",
    backend="cpu",
    output_dir=OUTPUT_DIR,
    vocab_model_file=VOCAB_MODEL_FILE,
    output_tflite_file=OUTPUT_FILE,
    is_quantized=True,
    attention_quant_bits=4,
    feedforward_quant_bits=4,
    embedding_quant_bits=4,
)

try:
    converter.convert_checkpoint(config)
    print("\nConversion successfully completed!")
    print(f"Model saved to: {OUTPUT_FILE}")
except Exception as e:
    print(f"\nConversion failed: {e}")
    import traceback
    traceback.print_exc()
