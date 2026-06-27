"""Upload fused model to Hugging Face Hub."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "Phani1479432"
REPO_NAME = "rocky-gemma-2b"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"
MODEL_DIR = Path("./rocky-gemma-2b")


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Fused model not found at {MODEL_DIR}. Run fuse.sh first."
        )

    api = HfApi(token=HF_TOKEN)

    print(f"Creating repo {REPO_ID} if it doesn't exist...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print(f"Uploading {MODEL_DIR} -> {REPO_ID} ...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"\nModel live at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
