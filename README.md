# python-llama-sv

Local llama.cpp inference server for Linux + NVIDIA GPU. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

Implements Gemma 4 chat, thinking, and tool-calling prompt handling in line with Google's Prompt Formatting for Gemma 4 guidance:
https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

**Default model:** Gemma 4 E4B IT UD-Q8_K_XL GGUF  
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

Place your GGUF file somewhere accessible, e.g. `~/models/gemma-4-e4b-it-8bit.gguf`.

You also need the HuggingFace tokenizer for chat template rendering:

```bash
# Pre-cache the tokenizer (one-time, requires internet)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-4-e4b-it')"
```

## Running the server

```bash
GGUF_MODEL_PATH=~/models/gemma-4-e4b-it-8bit.gguf \
HF_MODEL_ID=google/gemma-4-e4b-it \
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server starts at `http://127.0.0.1:8000`.

`main.py` is a thin entry point. The FastAPI app is created in `llamasv/server.py`, and the model/tokenizer are loaded during FastAPI startup via the app lifespan hook rather than at Python import time.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `GGUF_MODEL_PATH` | `/home/gs/Downloads/gemma-4-E4B-it-UD-Q8_K_XL.gguf` | Path to the local GGUF file |
| `HF_MODEL_ID` | `google/gemma-4-e4b-it` | HuggingFace model ID for the tokenizer/chat template |
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
- Thinking mode (`<|think|>` blocks) is enabled by default. Pass `enable_thinking: false` in `extra_body` to disable.
- Tool calls use Gemma 4's native format and are parsed back into OpenAI-compatible `tool_calls` objects.
- Thought-channel output is hidden from API responses by default. Pass `expose_thoughts: true` in `extra_body` to expose it for debugging.
- Stop sequences include Gemma 4 end-of-turn markers and `<|tool_response>`.
- Prompt assembly uses the Hugging Face Gemma 4 chat template, consolidates system instructions into a single system turn, and feeds prior tool results back as Gemma-style `tool_responses`.
- Raw model outputs are logged to `llamasv-raw-outputs.txt` by default, with per-session files under `llamasv-raw-outputs.sessions/` when `extra_body.session_id` is provided.
- The repo is organized as a small package:
  - `llamasv/runtime.py` for settings, runtime creation, and logging
  - `llamasv/prompting.py` for prompt assembly and truncation
  - `llamasv/messages.py` for message normalization and thought stripping
  - `llamasv/tooling.py` for Gemma native tool-call parsing
  - `llamasv/server.py` for the FastAPI app and routes
- The RTX 3070 (12 GB) was previously used successfully for Gemma 4 E4B at 8-bit; usable `N_CTX` depends on your exact quantization and available VRAM.

## Gemma 4 Compliance

This server is intended to follow the text-and-tools portions of Google's Gemma 4 prompt-formatting spec:

- Uses the tokenizer's Gemma 4 chat template instead of manually assembling `<|turn|>` text.
- Enables thinking mode at the conversation level through a single system turn.
- Strips prior thought blocks before replaying assistant history.
- Parses Gemma native `<|tool_call>...<tool_call|>` output into OpenAI-compatible `tool_calls`.
- Preserves prior tool results in Gemma-compatible `tool_responses` form when rebuilding conversation history.
- Adds `<|tool_response>` to generation stop sequences as recommended for function calling.

This repo is aligned for Gemma 4 text chat and tool use. It does not currently implement multimodal placeholders such as image or audio turns.
