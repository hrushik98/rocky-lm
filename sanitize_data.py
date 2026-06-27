"""Sanitize datasets for Gemma 2 format compliance.
1. Map 'agent' role to 'assistant'.
2. Remove any records that do not strictly alternate user/assistant.
"""

import json
from pathlib import Path

DATA_DIR = Path("rocky_data")

for filename in ["train.jsonl", "valid.jsonl"]:
    path = DATA_DIR / filename
    if not path.exists():
        continue

    lines = path.read_text().splitlines()
    cleaned = []
    
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping unparseable line {idx+1} in {filename}")
            continue

        messages = record.get("messages", [])
        
        # 1. Map role 'agent' to 'assistant'
        for msg in messages:
            if msg.get("role") == "agent":
                msg["role"] = "assistant"
        
        # 2. Validate alternation
        valid = True
        if not messages:
            valid = False
        else:
            for i, msg in enumerate(messages):
                expected = "user" if i % 2 == 0 else "assistant"
                if msg.get("role") != expected:
                    valid = False
                    break
        
        if valid:
            cleaned.append(json.dumps(record))
        else:
            print(f"Removing invalid record {idx+1} in {filename}: {[m.get('role') for m in messages]}")

    path.write_text("\n".join(cleaned) + "\n")
    print(f"Saved {len(cleaned)} cleaned records to {path}")
