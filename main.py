import json
import os
import re
import threading
import time
import uuid
import warnings
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

app = FastAPI()

DEFAULT_MODEL_NAME = "mlx-community/Qwen3-14B-4bit"
STALE_MODEL_ALIASES = {"mlx", "mlx-community"}


def resolve_model_name() -> str:
    configured = (os.environ.get("MLX_MODEL") or "").strip()
    if not configured:
        return DEFAULT_MODEL_NAME
    if configured in STALE_MODEL_ALIASES:
        warnings.warn(
            f"MLX_MODEL={configured} is stale; using {DEFAULT_MODEL_NAME} instead.",
            stacklevel=2,
        )
        return DEFAULT_MODEL_NAME
    return configured


MODEL_NAME = resolve_model_name()
PUBLIC_MODEL_NAME = MODEL_NAME

model, tokenizer = load(MODEL_NAME)
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def message_to_dict(m: ChatMessage) -> dict:
    """Convert a ChatMessage to the dict format expected by apply_chat_template."""
    if m.role == "assistant" and m.tool_calls:
        cooked: list[Any] = []
        for tc in m.tool_calls:
            tc_copy = dict(tc)
            if isinstance(tc_copy.get("function"), dict):
                fn = dict(tc_copy["function"])
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except (json.JSONDecodeError, ValueError):
                        pass
                tc_copy["function"] = fn
            cooked.append(tc_copy)
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": normalize_content(m.content) or "",
            "tool_calls": cooked,
        }
    elif m.role == "tool":
        # Sanitize any tool-call XML embedded in fetched content (e.g. docs pages
        # that contain <function=...> or <tool_call> code examples) so the parser
        # doesn't fire on them when this result re-enters the context.
        raw = normalize_content(m.content)
        safe = re.sub(r"<(/?tool_call|function=\w+|/function|parameter=\w+|/parameter)>", r"[\1]", raw)
        msg = {
            "role": "user",
            "content": f"<tool_response>\n{safe}\n</tool_response>",
        }
    else:
        msg = {"role": m.role, "content": normalize_content(m.content)}
    return msg


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def _normalize_qwen_args(raw: str) -> dict | None:
    trimmed = raw.strip()
    if trimmed.startswith("{{") and trimmed.endswith("}}"):
        trimmed = trimmed[1:-1]

    normalized = trimmed.replace('<|"|>', '"')
    normalized = re.sub(r"([{,]\s*)(\w+)\s*:", r'\1"\2":', normalized)
    try:
        return json.loads(normalized)
    except (json.JSONDecodeError, ValueError):
        pass

    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        return None

    inner = trimmed[1:-1].strip()
    args: dict[str, Any] = {}
    i = 0
    n = len(inner)

    while i < n:
        while i < n and inner[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and (inner[i].isalnum() or inner[i] == "_"):
            i += 1
        key = inner[key_start:i].strip()
        if not key:
            return None

        while i < n and inner[i].isspace():
            i += 1
        if i >= n or inner[i] != ":":
            return None
        i += 1
        while i < n and inner[i].isspace():
            i += 1
        if i >= n:
            return None

        if inner.startswith('<|"|>', i):
            i += len('<|"|>')
            end = inner.find('<|"|>', i)
            if end == -1:
                return None
            value: Any = inner[i:end]
            i = end + len('<|"|>')
        elif inner[i] == '"':
            i += 1
            start = i
            while i < n and inner[i] != '"':
                i += 1
            if i >= n:
                return None
            value = inner[start:i]
            i += 1
        else:
            start = i
            while i < n and inner[i] not in ",}":
                i += 1
            token = inner[start:i].strip()
            if token in {"true", "false"}:
                value = token == "true"
            elif token == "null":
                value = None
            else:
                try:
                    value = int(token)
                except ValueError:
                    try:
                        value = float(token)
                    except ValueError:
                        value = token

        args[key] = value

        while i < n and inner[i].isspace():
            i += 1
        if i < n and inner[i] == ",":
            i += 1

    return args or None


def extract_tool_calls(text: str) -> list[dict] | None:
    calls: list[dict] = []
    matched_path: str | None = None
    decoder = json.JSONDecoder()
    normalized = text.replace('<|"|>', '"')

    for m in re.finditer(r"<tool_call>\s*(\{)", normalized, re.DOTALL):
        start = m.start(1)
        try:
            data, _ = decoder.raw_decode(normalized, start)
            if isinstance(data, dict) and "name" in data:
                matched_path = "standard_xml_json"
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {})),
                        },
                    }
                )
        except (json.JSONDecodeError, ValueError):
            continue

    if calls:
        print(f"[mlxsv] extract_tool_calls matched={matched_path} count={len(calls)}")
        return calls

    for m in re.finditer(
        r"(?s)<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>",
        normalized,
    ):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}

        for p in re.finditer(
            r"(?s)<parameter=([A-Za-z_]\w*)>\s*(.*?)\s*</parameter>",
            body,
        ):
            key = p.group(1)
            raw_value = p.group(2).strip()
            if raw_value.startswith("{") and raw_value.endswith("}"):
                parsed = _normalize_qwen_args(raw_value)
                args[key] = parsed if parsed is not None else raw_value
            elif raw_value in {"true", "false"}:
                args[key] = raw_value == "true"
            elif raw_value == "null":
                args[key] = None
            else:
                try:
                    args[key] = int(raw_value)
                except ValueError:
                    try:
                        args[key] = float(raw_value)
                    except ValueError:
                        args[key] = raw_value

        matched_path = "xml_function_parameters"
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args),
                },
            }
        )

    if calls:
        print(f"[mlxsv] extract_tool_calls matched={matched_path} count={len(calls)}")
        return calls

    for m in re.finditer(r"<\|tool_call>call:\s*(\{)", normalized, re.DOTALL):
        start = m.start(1)
        try:
            data, _ = decoder.raw_decode(normalized, start)
            if isinstance(data, dict) and "name" in data:
                matched_path = "qwen_hybrid_json"
                calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {})),
                        },
                    }
                )
        except (json.JSONDecodeError, ValueError):
            continue

    if calls:
        print(f"[mlxsv] extract_tool_calls matched={matched_path} count={len(calls)}")
        return calls

    for m in re.finditer(
        r"<\|tool_call>call:(\w+)\s*(\{\{.*?\}\}|\{.*?\})(?:<tool_call\|>)?",
        text,
        re.DOTALL,
    ):
        name = m.group(1)
        args = _normalize_qwen_args(m.group(2))
        if args is not None:
            matched_path = "qwen_compact_native"
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
            )

    if calls:
        print(f"[mlxsv] extract_tool_calls matched={matched_path} count={len(calls)}")
        return calls

    print(f"[mlxsv] extract_tool_calls matched=none sample={normalized[:300]!r}")
    return None


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def clean_output(text: str) -> str:
    """Minimal cleaning: remove control tokens but preserve <thinking> blocks."""
    for token in ["<|im_end|>", "<|im_start|>"]:
        text = text.replace(token, "")
    return text.strip()


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 60_000
_MAX_ANCHOR_CHARS = 45_000


def _msg_chars(msg: dict) -> int:
    total = len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        total += len(str(tc))
    return total


def _is_tool_response(msg: dict) -> bool:
    content = msg.get("content") or ""
    return isinstance(content, str) and content.startswith("<tool_response>")


def _trim_anchor(anchor: list[dict]) -> list[dict]:
    """Keep the user message + as many recent tool pairs as fit in _MAX_ANCHOR_CHARS.

    Tool results are user-role messages wrapped in <tool_response>. They always
    appear after an assistant tool_calls message. We trim from the oldest pair
    forward so the model always sees the most recent results.
    """
    if len(anchor) <= 1:
        return anchor
    user_msg = anchor[:1]
    tool_msgs = anchor[1:]
    budget = _MAX_ANCHOR_CHARS - _msg_chars(user_msg[0])
    kept: list[dict] = []
    for msg in reversed(tool_msgs):
        cost = _msg_chars(msg)
        if cost > budget:
            break
        kept.insert(0, msg)
        budget -= cost
    # Don't start mid-pair with an orphaned tool_response.
    while kept and _is_tool_response(kept[0]):
        kept.pop(0)
    return user_msg + kept


def truncate_messages(messages: list[dict]) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= 2:
        return system + non_system

    # Anchor on the LAST user message (current task), not the first.
    # Find the most recent user message to use as anchor.
    last_user_idx = None
    for i in range(len(non_system) - 1, -1, -1):
        if non_system[i].get("role") == "user" and not _is_tool_response(non_system[i]):
            last_user_idx = i
            break

    # If no plain user message found, fall back to the last message.
    if last_user_idx is None or last_user_idx == len(non_system) - 1:
        anchor = [non_system[-1]]
        middle = non_system[:-1]
    else:
        anchor = non_system[last_user_idx:]
        middle = non_system[:last_user_idx]

    # Trim the anchor itself so it never balloons past _MAX_ANCHOR_CHARS.
    anchor = _trim_anchor(anchor)

    budget = _MAX_CONTEXT_CHARS - sum(_msg_chars(m) for m in system + anchor)

    kept: list[dict] = []
    for msg in reversed(middle):
        cost = _msg_chars(msg)
        if cost > budget:
            break
        kept.insert(0, msg)
        budget -= cost

    while kept and _is_tool_response(kept[0]):
        kept.pop(0)

    dropped = len(middle) - len(kept)
    if dropped:
        print(
            f"[mlxsv] context_truncated dropped_turns={dropped} kept_turns={len(kept)}"
        )

    return system + kept + anchor


def extract_reasoning(text: str) -> str | None:
    """Preserve everything before the first tool call marker."""
    markers = ["<tool_call>", "<|tool_call>"]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    reasoning = text[:cut].strip()
    return reasoning if reasoning else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": PUBLIC_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    messages = [message_to_dict(m) for m in req.messages]
    messages = truncate_messages(messages)
    print(
        f"[mlxsv] chat_completions tools_in_request={len(req.tools or [])} messages={len(messages)}"
    )

    no_think = any(
        "/no_think" in (msg.get("content") or "")
        for msg in messages
        if msg.get("role") == "system"
    )

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": not no_think,
    }
    if req.tools:
        template_kwargs["tools"] = req.tools

    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception:
        template_kwargs.pop("tools", None)
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if req.stream:
        def event_stream():
            accumulated = ""
            with _inference_lock:
                try:
                    from mlx_lm import stream_generate as _stream_gen
                    for resp in _stream_gen(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=req.max_tokens or 2500,
                        sampler=make_sampler(req.temperature or 0.3),
                    ):
                        token = resp.text
                        accumulated += token
                        delta_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": PUBLIC_MODEL_NAME,
                            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"
                except Exception:
                    # Fallback: generate all at once, send as single chunk
                    raw = generate(
                        model, tokenizer,
                        prompt=prompt,
                        max_tokens=req.max_tokens or 2500,
                        sampler=make_sampler(req.temperature or 0.3),
                    )
                    accumulated = raw
                    delta_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": PUBLIC_MODEL_NAME,
                        "choices": [{"index": 0, "delta": {"content": accumulated}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(delta_chunk)}\n\n"

            # After stream ends, parse tool calls from full accumulated text.
            text = clean_output(accumulated)
            print(f"[mlxsv] raw_output={text[:800]!r}")
            tool_calls = extract_tool_calls(text) if req.tools else None
            finish_reason = "tool_calls" if tool_calls else "stop"
            print(f"[mlxsv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}")

            final_delta: dict[str, Any] = {}
            if tool_calls:
                final_delta["tool_calls"] = tool_calls
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": PUBLIC_MODEL_NAME,
                "choices": [{"index": 0, "delta": final_delta, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path.
    with _inference_lock:
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=req.max_tokens or 2500,
            sampler=make_sampler(req.temperature or 0.3),
        )

    text = clean_output(raw)
    print(f"[mlxsv] raw_output={text[:800]!r}")
    tool_calls = extract_tool_calls(text) if req.tools else None
    finish_reason = "tool_calls" if tool_calls else "stop"
    reasoning = extract_reasoning(text) if tool_calls else text

    response_message: dict[str, Any] = {"role": "assistant", "content": reasoning}
    if tool_calls:
        response_message["tool_calls"] = tool_calls

    print(
        f"[mlxsv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}"
    )
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": PUBLIC_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
    )
