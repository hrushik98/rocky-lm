"""Rocky inference script with timing and token metrics.

Usage:
    # Single prompt
    uv run python inference.py --device mlx --model ./rocky-gemma-2b --prompt "How do I fix a merge conflict?"
    uv run python inference.py --device gpu --model ./rocky-gemma-2b --prompt "How do I fix a merge conflict?"

    # Interactive REPL (omit --prompt)
    uv run python inference.py --device mlx --model ./rocky-gemma-2b
"""

import argparse
import sys
import time

ROCKY_SYSTEM = "Speak like Rocky."


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.2f}s"


def _print_metrics(
    prompt_tokens: int,
    output_tokens: int,
    load_time: float,
    gen_time: float,
    peak_memory_mb: float | None = None,
) -> None:
    tok_per_sec = output_tokens / gen_time if gen_time > 0 else 0
    print()
    print("---- metrics ----")
    print(f"  Model load time  : {_fmt_time(load_time)}")
    print(f"  Prompt tokens    : {prompt_tokens}")
    print(f"  Output tokens    : {output_tokens}")
    print(f"  Generation time  : {_fmt_time(gen_time)}")
    print(f"  Tokens / sec     : {tok_per_sec:.1f}")
    if peak_memory_mb is not None:
        print(f"  Peak memory      : {peak_memory_mb:,.0f} MB")
    print("-----------------")


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------

def run_mlx(model_path: str, prompt: str | None) -> None:
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("mlx-lm not installed. Run: uv add mlx-lm")
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    load_start = time.perf_counter()
    model, tokenizer = load(model_path)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {_fmt_time(load_time)}")
    print()

    def infer(user_input: str) -> None:
        messages = [
            {"role": "user", "content": f"{ROCKY_SYSTEM}\n\n{user_input}"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(formatted)

        gen_start = time.perf_counter()
        response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=False)
        gen_time = time.perf_counter() - gen_start

        output_ids = tokenizer.encode(response)

        try:
            import mlx.core as mx
            peak_mb = mx.metal.get_peak_memory() / 1e6
        except Exception:
            peak_mb = None

        print(f"Rocky: {response}")
        _print_metrics(
            prompt_tokens=len(prompt_ids),
            output_tokens=len(output_ids),
            load_time=load_time,
            gen_time=gen_time,
            peak_memory_mb=peak_mb,
        )

    if prompt:
        infer(prompt)
    else:
        print("Interactive mode (Ctrl+C to quit)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not user_input:
                continue
            infer(user_input)


# ---------------------------------------------------------------------------
# GPU backend (transformers / Unsloth)
# ---------------------------------------------------------------------------

def run_gpu(model_path: str, prompt: str | None) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    except ImportError:
        print("Missing GPU dependencies. Run: uv add torch transformers")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading model from {model_path}  [device: {device}] ...")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {_fmt_time(load_time)}")
    print()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def infer(user_input: str) -> None:
        messages = [
            {"role": "user", "content": f"{ROCKY_SYSTEM}\n\n{user_input}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        prompt_tokens = inputs.shape[-1]

        print("Rocky: ", end="", flush=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
            )
        gen_time = time.perf_counter() - gen_start

        output_tokens = output.shape[-1] - prompt_tokens

        peak_mb = None
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

        _print_metrics(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            load_time=load_time,
            gen_time=gen_time,
            peak_memory_mb=peak_mb,
        )

    if prompt:
        infer(prompt)
    else:
        print("Interactive mode (Ctrl+C to quit)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not user_input:
                continue
            infer(user_input)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rocky inference with timing metrics")
    parser.add_argument(
        "--device",
        choices=["mlx", "gpu"],
        required=True,
        help="mlx = Apple Silicon  |  gpu = NVIDIA",
    )
    parser.add_argument(
        "--model",
        default="./rocky-gemma-2b",
        help="Path to fused model directory (default: ./rocky-gemma-2b)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    print(f"=== Rocky Inference  [device: {args.device}] ===")
    print(f"Model : {args.model}")
    print()

    if args.device == "mlx":
        run_mlx(args.model, args.prompt)
    else:
        run_gpu(args.model, args.prompt)


if __name__ == "__main__":
    main()
