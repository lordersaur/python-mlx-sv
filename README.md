# python-llama-sv

Local llama.cpp inference server for Linux + NVIDIA GPU. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

The server now defaults to a Qwen-style chat/template flow for:
`Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF` at `Q6_K` with a `32k` context target.

**Default model:** Qwen3.5-9B-DeepSeek-V4-Flash `Q6_K` GGUF  
**Requires:** Linux, NVIDIA GPU (tested on RTX 3070 12 GB), Python 3.11+

## Setup

**1. Install llama-cpp-python with CUDA support**

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**2. Install remaining dependencies**

```bash
pip install fastapi uvicorn transformers
```

**3. Download the GGUF model**

Download `Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF` and place the `Q6_K` file somewhere accessible, for example:
`~/Downloads/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf`

You also need the HuggingFace tokenizer for chat template rendering:

```bash
# Pre-cache the tokenizer (one-time, requires internet)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B')"
```

## Running the server

```bash
GGUF_MODEL_PATH=~/Downloads/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf \
HF_MODEL_ID=Qwen/Qwen3.5-9B \
N_CTX=32768 \
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://127.0.0.1:8000`.

`main.py` is a thin entry point. The FastAPI app is created in `llamasv/server.py`, and the model/tokenizer are loaded during FastAPI startup via the app lifespan hook rather than at Python import time.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GGUF_MODEL_PATH` | `~/Downloads/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf` | Path to the local GGUF file |
| `HF_MODEL_ID` | `Qwen/Qwen3.5-9B` | HuggingFace model ID for the tokenizer/chat template |
| `N_GPU_LAYERS` | `-1` | GPU layers to offload (`-1` = all) |
| `N_CTX` | `32768` | Context window size in tokens |
| `PUBLIC_MODEL_NAME` | *(GGUF filename)* | Model name reported by `/v1/models` |
| `LLAMA_VERBOSE` | `0` | Enables verbose llama.cpp logging when set to a truthy value |
| `RAW_OUTPUT_LOG_PATH` | `./llamasv-raw-outputs.txt` | Path used to store raw model outputs for offline inspection |

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat completions (streaming and non-streaming)
- `GET /v1/models` — Lists the loaded model

## Notes

- Only one inference request runs at a time (serialized via a threading lock).
- The runtime infers prompt formatting from `HF_MODEL_ID`; `Qwen/...` uses the tokenizer's Qwen chat template, while `gemma` IDs still keep the older Gemma path.
- Thinking mode is enabled by default. Pass `enable_thinking: false` in `extra_body` to disable.
- Tool calls use the tokenizer's native chat template and are parsed back into OpenAI-compatible `tool_calls` objects. Qwen XML-style `<tool_call>` blocks and the older Gemma native format are both supported.
- Thought-channel output is hidden from API responses by default. Pass `expose_thoughts: true` in `extra_body` to expose it for debugging.
- Prompt assembly uses the Hugging Face tokenizer chat template, consolidates system instructions into a single system turn, and feeds prior tool results back in the format expected by the selected model family.
- Raw model outputs are logged to `llamasv-raw-outputs.txt` by default, with per-session files under `llamasv-raw-outputs.sessions/` when `extra_body.session_id` is provided.
- The repo is organized as a small package:
  - `llamasv/runtime.py` for settings, runtime creation, and logging
  - `llamasv/prompting.py` for prompt assembly and truncation
  - `llamasv/messages.py` for message normalization and reasoning stripping
  - `llamasv/tooling.py` for Qwen and Gemma tool-call parsing
  - `llamasv/server.py` for the FastAPI app and routes
- For a 12 GB GPU, `Q6_K` plus `N_CTX=32768` is the intended starting point, but usable context still depends on the exact llama.cpp build and how much VRAM the desktop environment is already using.
