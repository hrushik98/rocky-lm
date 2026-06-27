"""Generate Rocky synthetic training data via DeepSeek API."""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

SYSTEM_PROMPT = (
    "You are an advanced synthetic data generator. "
    "Your job is to generate a diverse dataset of software engineering, "
    "programming, and server management conversations between a developer "
    '("User") and a specialized assistant ("You") named Rocky.\n\n'
    "The assistant must strictly follow these structural speech patterns "
    "in every single response:\n"
    '- End questions with ", question?" - never invert syntax.\n'
    '- Negate with "no": "you no die", "ship no move".\n'
    '- Drop articles and "is/are": "textbook formatting bad", "plan good".\n'
    '- Repeat for intensity: "fast fast fast", "many many many", "bad bad bad".\n'
    '- Short sentences only. Never use the words "because", "which", or "that".\n'
    '- State emotion as a factual preface: "Sad,", "Happy happy.", "Failure,", "Excitement,".\n'
    "- Compound complex or abstract ideas with hyphens: "
    '"deployment-nervousness", "database-confusion".\n'
    "- End statements with a comma when casual, period when final.\n"
    "- CRITICAL: Technical terms, code blocks, inline code, syntax, URLs, "
    "CLI commands, stack traces, and error messages must remain 100% accurate, "
    "unaltered, and intact.\n\n"
    "Instructions:\n"
    "Generate 100 distinct chat interactions covering topics like Git errors, "
    "Docker configurations, React hydration bugs, Python optimizations, "
    "FastAPIs, and SQL queries. "
    "Format the output strictly as a JSONL (JSON Lines) file where each line "
    "is a standalone JSON object structured like this:\n"
    '{"messages": [{"role": "system", "content": "Speak like Rocky."}, '
    '{"role": "user", "content": "User question here"}, '
    '{"role": "assistant", "content": "Rocky response here"}]}'
)

BATCH_PROMPT = (
    "Generate exactly {n} distinct chat interactions as described. "
    "Output only raw JSONL - one JSON object per line, no markdown, "
    "no code fences, no explanation. Start immediately with the first {{...}}."
)

BATCH_SIZE = 25
TOTAL = 100
TRAIN_SPLIT = 0.9

OUT_DIR = Path("rocky_data")
OUT_DIR.mkdir(exist_ok=True)


def fetch_batch(n: int, batch_num: int) -> list[dict]:
    print(f"  Requesting batch {batch_num} ({n} examples)...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BATCH_PROMPT.format(n=n)},
        ],
        temperature=1.0,
        max_tokens=8192,
    )
    raw = response.choices[0].message.content.strip()

    records = []
    for line_num, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
        except json.JSONDecodeError as e:
            print(f"    Skipping malformed line {line_num}: {e}")
    print(f"    Got {len(records)} valid records.")
    return records


def main() -> None:
    all_records: list[dict] = []
    num_batches = TOTAL // BATCH_SIZE

    print(f"Generating {TOTAL} examples in {num_batches} batches of {BATCH_SIZE}...")
    for i in range(num_batches):
        batch = fetch_batch(BATCH_SIZE, i + 1)
        all_records.extend(batch)

    # Deduplicate by user content
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in all_records:
        key = json.dumps(rec.get("messages", [{}])[1])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    print(f"\n{len(unique)} unique records after deduplication.")

    random.shuffle(unique)
    split = int(len(unique) * TRAIN_SPLIT)
    train, valid = unique[:split], unique[split:]

    train_path = OUT_DIR / "train.jsonl"
    valid_path = OUT_DIR / "valid.jsonl"

    train_path.write_text("\n".join(json.dumps(r) for r in train) + "\n")
    valid_path.write_text("\n".join(json.dumps(r) for r in valid) + "\n")

    print(f"Saved {len(train)} train -> {train_path}")
    print(f"Saved {len(valid)} valid -> {valid_path}")
    print("\nData generation complete.")


if __name__ == "__main__":
    main()
