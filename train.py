"""Unified Rocky training script.

Usage:
    uv run python train.py --device mlx   # Apple Silicon via mlx-lm
    uv run python train.py --device gpu   # NVIDIA via Unsloth + TRL
"""

import argparse
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = "google/gemma-2-2b-it"
DATA_DIR = "./rocky_data"

MLX_ADAPTER_DIR = "./adapters"

GPU_ADAPTER_DIR = "./adapters-gpu"
GPU_FUSED_DIR = "./rocky-gemma-2b"
GPU_GGUF_DIR = "./rocky-gguf"

TOTAL_STEPS = 600


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

def train_mlx() -> None:
    # mlx_lm already prints: Iter N: Train loss X.XXX, It/sec X.XX, Tokens/sec XXX
    # We wrap it with wall-clock timing and a final summary.
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", DATA_DIR,
        "--iters", str(TOTAL_STEPS),
        "--batch-size", "2",
        "--learning-rate", "1e-5",
        "--steps-per-eval", "50",
        "--adapter-path", MLX_ADAPTER_DIR,
    ]
    print("Command:", " ".join(cmd))
    print()

    start = time.perf_counter()
    result = subprocess.run(cmd, env=os.environ.copy())
    elapsed = time.perf_counter() - start

    print()
    print("=" * 50)
    print(f"MLX training finished in {_fmt_time(elapsed)}")
    print(f"Adapters saved to: {MLX_ADAPTER_DIR}/")
    print("=" * 50)

    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# GPU backend (Unsloth)
# ---------------------------------------------------------------------------

def train_gpu() -> None:
    try:
        import torch
        from datasets import load_dataset
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError as exc:
        print(f"Missing GPU dependency: {exc}")
        print()
        print("On your NVIDIA machine, install with:")
        print("  uv add torch trl transformers datasets")
        print('  uv add "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
        sys.exit(1)

    # ---- Metrics callback ----
    class RockyMetricsCallback(TrainerCallback):
        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
            self._t0 = time.perf_counter()
            self._last_step_t = self._t0
            print(f"{'Step':>6}  {'Train loss':>10}  {'Eval loss':>9}  {'LR':>8}  {'Tok/s':>7}  {'Elapsed':>8}  {'ETA':>8}")
            print("-" * 72)

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
            now = time.perf_counter()
            elapsed = now - self._t0
            step = state.global_step
            remaining = state.max_steps - step
            rate = step / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0

            tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 and tokens_seen else 0

            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            eval_loss = logs.get("eval_loss")

            loss_str = f"{loss:.4f}" if loss is not None else "     -"
            eval_str = f"{eval_loss:.4f}" if eval_loss is not None else "        -"
            lr_str = f"{lr:.2e}" if lr is not None else "       -"
            tok_str = f"{tok_per_sec:,.0f}" if tok_per_sec else "      -"

            print(
                f"{step:>6}  {loss_str:>10}  {eval_str:>9}  {lr_str:>8}"
                f"  {tok_str:>7}  {_fmt_time(elapsed):>8}  {_fmt_time(eta):>8}"
            )

        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kw):
            eval_loss = (metrics or {}).get("eval_loss")
            if eval_loss is not None:
                print(f"         eval  loss -> {eval_loss:.4f}")

        def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kw):
            total = time.perf_counter() - self._t0
            tokens_seen = getattr(state, "num_input_tokens_seen", 0) or 0
            print()
            print("=" * 50)
            print(f"GPU training finished in {_fmt_time(total)}")
            print(f"Total steps    : {state.global_step}")
            print(f"Tokens trained : {tokens_seen:,}")
            if total > 0 and tokens_seen:
                print(f"Avg tokens/sec : {tokens_seen / total:,.0f}")
            # GPU peak memory
            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                print(f"Peak GPU VRAM  : {peak_mb:,.0f} MB")
            print("=" * 50)

    # ---- Model ----
    load_start = time.perf_counter()
    print(f"Loading {MODEL} with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    print(f"Model loaded in {_fmt_time(time.perf_counter() - load_start)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
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
        callbacks=[RockyMetricsCallback()],
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=TOTAL_STEPS,
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
            report_to="none",       # silence W&B / tensorboard noise
            disable_tqdm=True,      # our callback replaces the progress bar
        ),
    )

    trainer.train()

    # ---- Save adapter ----
    print(f"\nSaving adapter  -> {GPU_ADAPTER_DIR}/")
    model.save_pretrained(GPU_ADAPTER_DIR)
    tokenizer.save_pretrained(GPU_ADAPTER_DIR)

    # ---- Fuse weights ----
    print(f"Fusing weights  -> {GPU_FUSED_DIR}/")
    model.save_pretrained_merged(GPU_FUSED_DIR, tokenizer, save_method="merged_16bit")

    # ---- GGUF export ----
    print(f"Exporting GGUF  -> {GPU_GGUF_DIR}/  [Q4_K_M]")
    model.save_pretrained_gguf(GPU_GGUF_DIR, tokenizer, quantization_method="q4_k_m")

    print(f"\n  Adapter  : {GPU_ADAPTER_DIR}/")
    print(f"  Fused HF : {GPU_FUSED_DIR}/  <- upload to HF")
    print(f"  GGUF     : {GPU_GGUF_DIR}/*.gguf  <- copy to phone")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Rocky LoRA adapter")
    parser.add_argument(
        "--device",
        choices=["mlx", "gpu"],
        required=True,
        help="mlx = Apple Silicon (mlx-lm)  |  gpu = NVIDIA (Unsloth)",
    )
    args = parser.parse_args()

    print(f"=== Rocky LoRA Training  [device: {args.device}] ===")
    print(f"Model : {MODEL}")
    print(f"Data  : {DATA_DIR}")
    print(f"Steps : {TOTAL_STEPS}")
    print()

    if args.device == "mlx":
        train_mlx()
    else:
        train_gpu()


if __name__ == "__main__":
    main()
