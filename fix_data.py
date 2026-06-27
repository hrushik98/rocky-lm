"""One-shot: merge system role into first user turn for Gemma 2 compatibility."""

import json
from pathlib import Path

DATA_DIR = Path("rocky_data")


def fix_messages(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    if messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "user":
        rest[0] = {
            "role": "user",
            "content": f"{system_content}\n\n{rest[0]['content']}",
        }
        return rest
    return rest


for path in [DATA_DIR / "train.jsonl", DATA_DIR / "valid.jsonl"]:
    lines = path.read_text().strip().splitlines()
    fixed = []
    for line in lines:
        record = json.loads(line)
        record["messages"] = fix_messages(record["messages"])
        fixed.append(json.dumps(record))
    path.write_text("\n".join(fixed) + "\n")
    print(f"Fixed {len(fixed)} records in {path}")
