# Rocky LM

Rocky LM is a custom conversational assistant fine-tuned from the base model **`google/gemma-2-2b-it`** (Gemma 2 2B Instruct) to speak and act like Rocky Balboa.

---

## 🚀 Quick Start: Run with Ollama

You can download and run the Rocky model instantly on a local Ollama instance with a single command (no repository cloning or local setup required):

```bash
curl -sSL https://huggingface.co/Phani1479432/rocky-gemma-2b/raw/main/Modelfile | ollama create rocky -f - && ollama run rocky
```

### What this does:
1. **Downloads the Modelfile**: Fetches the custom template and parameter configurations directly from Hugging Face.
2. **Creates the Model**: Registers the local model name `rocky` inside Ollama.
3. **Downloads the Quantized Weights**: Automatically pulls the quantized GGUF weights (`rocky-q4_k_m.gguf`) from Hugging Face.
4. **Starts the Chat**: Launches an interactive shell session.

---

## 🛠️ Project Structure & Workflow

The repository is organized as a step-by-step model development pipeline:

1. **Dataset Sanitization & Fixing**:
    * `sanitize_data.py`: Prepares conversation history to match Gemma 2 formats by validating turn alternations.
    * `fix_data.py`: Helper script to merge system instructions into user turns for format compliance.
2. **Training**:
    * `train.py` / `train.sh`: Runs LoRA fine-tuning using MLX on Apple Silicon.
    * Generates adapters stored in the `adapters/` folder.
3. **Model Fusion**:
    * `fuse.sh`: Fuses the LoRA adapters into the base `google/gemma-2-2b-it` weights, saving a standalone Hugging Face model under `rocky-gemma-2b/`.
4. **GGUF Compilation**:
    * `build_gguf.sh`: Clones and compiles `llama.cpp` to convert the fused weights into FP16 and quantized `Q4_K_M` GGUF formats.
5. **MediaPipe / LiteRT Conversion**:
    * `convert_to_task.py`: Converts the safetensors model to MediaPipe `.task` format for on-device deployment.
6. **Hugging Face Deployment**:
    * `upload_model.py`: Uploads the fused model, GGUFs, adapters, and `.task` files to the Hugging Face Hub under `Phani1479432/rocky-gemma-2b`.

---

## 💻 Local Developer Usage

### 1. Fine-Tuning
To start the training process locally on Apple Silicon:
```bash
./train.sh
```

### 2. Fuse Weights
To fuse the trained adapters back into the base Gemma model weights:
```bash
./fuse.sh
```

### 3. Create GGUF
To convert the model to GGUF formats for llama.cpp/Ollama:
```bash
./build_gguf.sh
```

### 4. Create local Ollama model (using local weights)
If you have `rocky-q4_k_m.gguf` locally in the root:
```bash
ollama create rocky -f Modelfile
ollama run rocky
```
