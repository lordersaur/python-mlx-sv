import json
import os
import random
import subprocess
import sys

try:
    from mlx_lm.utils import load_tokenizer
except ImportError:
    print("Error: mlx-lm not found. Please run: pip install mlx-lm")
    exit(1)

# Configuration
model_id = "unsloth/gemma-4-E4B-it-MLX-8bit"
input_file = "gemma4_symbol_search_read_sft_100.jsonl"
train_file = "train.jsonl"
valid_file = "valid.jsonl"
valid_ratio = 0.1  # 10% held out for validation


def convert_to_mlx_format():
    print(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = load_tokenizer(model_id)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)

    print(f"Converting {input_file} to MLX-friendly format...")
    converted = []
    with open(input_file, "r") as f_in:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                full_text = tokenizer.apply_chat_template(
                    data["conversations"], tokenize=False, add_generation_prompt=False
                )
                converted.append(json.dumps({"text": full_text}))
            except Exception as e:
                print(f"Skipping malformed line: {e}")

    random.shuffle(converted)
    split = max(1, int(len(converted) * valid_ratio))
    valid_examples = converted[:split]
    train_examples = converted[split:]

    with open(train_file, "w") as f:
        f.write("\n".join(train_examples) + "\n")
    with open(valid_file, "w") as f:
        f.write("\n".join(valid_examples) + "\n")

    print(f"Created {train_file} ({len(train_examples)} examples) and {valid_file} ({len(valid_examples)} examples)")


def run_lora_training():
    print("Starting MLX LoRA training...")
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",        # fixed: mlx_lm.lora not mlx_lm lora
        "--model",
        model_id,
        "--train",
        "--data",
        "./",
        "--iters",
        "600",                 # ~6 passes over 100 examples
        "--batch-size",
        "1",
        "--learning-rate",
        "1e-5",
        "--steps-per-report",
        "10",
        "--max-seq-length",
        "2048",                # raised from 1024 — tool-call turns are long
        "--num-layers",
        "8",
        "--grad-checkpoint",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Place it in {os.getcwd()}")
    else:
        convert_to_mlx_format()
        run_lora_training()
