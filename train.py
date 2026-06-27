"""Unified Rocky training script.

Usage:
    uv run python train.py --device mlx   # Apple Silicon via mlx-lm
    uv run python train.py --device gpu   # NVIDIA via Unsloth + TRL
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL = "google/gemma-2-2b-it"
DATA_DIR = "./rocky_data"

# MLX paths
MLX_ADAPTER_DIR = "./adapters"

# GPU paths  (fuse + GGUF export happen inline with Unsloth)
GPU_ADAPTER_DIR = "./adapters-gpu"
GPU_FUSED_DIR = "./rocky-gemma-2b"
GPU_GGUF_DIR = "./rocky-gguf"


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

def train_mlx() -> None:
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", DATA_DIR,
        "--iters", "600",
        "--batch-size", "2",
        "--learning-rate", "1e-5",
        "--steps-per-eval", "50",
        "--adapter-path", MLX_ADAPTER_DIR,
    ]
    print("Command:", " ".join(cmd))
    print()
    result = subprocess.run(cmd, env=os.environ.copy())
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# GPU backend (Unsloth)
# ---------------------------------------------------------------------------

def train_gpu() -> None:
    # Defer imports so Mac users never hit a CUDA import error
    try:
        import torch
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError as exc:
        print(f"Missing GPU dependency: {exc}")
        print()
        print("On your NVIDIA machine, install with:")
        print("  uv add torch trl transformers datasets")
        print('  uv add "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        sys.exit(1)

    # ---- Model ----
    print(f"Loading {MODEL} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=2048,
        dtype=None,       # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ---- Dataset ----
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_DIR}/train.jsonl",
            "validation": f"{DATA_DIR}/valid.jsonl",
        },
    )

    def apply_chat_template(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in examples["messages"]
            ]
        }

    dataset = dataset.map(apply_chat_template, batched=True)

    # ---- Trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=600,
            learning_rate=1e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            output_dir=GPU_ADAPTER_DIR,
            load_best_model_at_end=True,
        ),
    )

    print("Starting Unsloth LoRA training...")
    trainer.train()

    # ---- Save adapter ----
    print(f"\nSaving adapter -> {GPU_ADAPTER_DIR}")
    model.save_pretrained(GPU_ADAPTER_DIR)
    tokenizer.save_pretrained(GPU_ADAPTER_DIR)

    # ---- Fuse weights ----
    print(f"Fusing weights  -> {GPU_FUSED_DIR}")
    model.save_pretrained_merged(GPU_FUSED_DIR, tokenizer, save_method="merged_16bit")

    # ---- GGUF export (skips the llama.cpp step entirely) ----
    print(f"Exporting GGUF  -> {GPU_GGUF_DIR}  [Q4_K_M]")
    model.save_pretrained_gguf(GPU_GGUF_DIR, tokenizer, quantization_method="q4_k_m")

    print("\nGPU training complete.")
    print(f"  Adapter:    {GPU_ADAPTER_DIR}/")
    print(f"  Fused HF:   {GPU_FUSED_DIR}/     <- upload this to HF")
    print(f"  GGUF:       {GPU_GGUF_DIR}/*.gguf <- copy this to your phone")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Rocky LoRA adapter")
    parser.add_argument(
        "--device",
        choices=["mlx", "gpu"],
        required=True,
        help="mlx = Apple Silicon (mlx-lm), gpu = NVIDIA (Unsloth)",
    )
    args = parser.parse_args()

    print(f"=== Rocky LoRA Training  [device: {args.device}] ===")
    print(f"Model : {MODEL}")
    print(f"Data  : {DATA_DIR}")
    print()

    if args.device == "mlx":
        train_mlx()
    else:
        train_gpu()


if __name__ == "__main__":
    main()
