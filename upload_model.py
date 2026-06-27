"""Upload fused model and associated files to Hugging Face Hub."""

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
ADAPTERS_DIR = Path("./adapters")


def main() -> None:
    api = HfApi(token=HF_TOKEN)

    print(f"Creating repo {REPO_ID} if it doesn't exist...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    if MODEL_DIR.exists():
        print(f"Uploading fused model folder: {MODEL_DIR} -> {REPO_ID} ...")
        api.upload_folder(
            folder_path=str(MODEL_DIR),
            repo_id=REPO_ID,
            repo_type="model",
        )
    else:
        print(f"Warning: Fused model directory {MODEL_DIR} not found. Skipping.")

    if ADAPTERS_DIR.exists():
        print(f"Uploading adapters folder: {ADAPTERS_DIR} -> {REPO_ID}/adapters ...")
        api.upload_folder(
            folder_path=str(ADAPTERS_DIR),
            path_in_repo="adapters",
            repo_id=REPO_ID,
            repo_type="model",
        )
    else:
        print(f"Warning: Adapters directory {ADAPTERS_DIR} not found. Skipping.")

    # Upload other model files if they exist in the root
    extra_files = ["rocky-q4_k_m.gguf", "rocky-f16.gguf", "rocky.task", "rocky.litertlm"]
    for file_name in extra_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"Uploading file: {file_path} -> {REPO_ID}/{file_name} ...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=REPO_ID,
                repo_type="model",
            )

    print(f"\nModel live at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()

