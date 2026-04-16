# python-mlx-sv

Local MLX inference server for Apple Silicon Macs. Exposes an OpenAI-compatible `/v1/chat/completions` endpoint used by [mlx-acp-agent](https://github.com/lordersaur/mlx-acp-agent).

**Default model:** `mlx-community/Qwen3-14B-4bit`  
**Requires:** Apple Silicon Mac (M1 or later)

## Setup

**1. Create a Python virtual environment**
```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
```

**2. Install dependencies**
```bash
pip install mlx-lm fastapi uvicorn
```

## Running the server

```bash
source ~/mlx-env/bin/activate
cd ~/python-mlx-sv && uvicorn main:app
```

The server starts at `http://127.0.0.1:8000`.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `MLX_MODEL` | `mlx-community/Qwen3-14B-4bit` | HuggingFace model ID to load |

The model is downloaded automatically on first run via the HuggingFace Hub.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat completions (streaming and non-streaming)
- `GET /v1/models` — Lists the loaded model

## Notes

- Only one inference request runs at a time (serialized via a threading lock to prevent Metal GPU crashes).
- Thinking mode (`<think>` blocks) is enabled by default for Qwen3. Send `/no_think` in the system message to disable it for faster responses.
