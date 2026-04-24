import inspect
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

try:
    import optiq  # registers OptiQ model types with mlx_lm
except Exception:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

app = FastAPI()

DEFAULT_MODEL_NAME = "Jackrong/MLX-Qwopus3.5-9B-v3-6bit"


def resolve_model_name() -> str:
    configured = (os.environ.get("MLX_MODEL") or "").strip()
    if not configured:
        return DEFAULT_MODEL_NAME
    return configured


MODEL_NAME = resolve_model_name()
PUBLIC_MODEL_NAME = MODEL_NAME

model, tokenizer = load(MODEL_NAME)
_inference_lock = threading.Lock()


def _make_turbo_cache():
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


try:
    _MAKE_SAMPLER_PARAMS = set(inspect.signature(make_sampler).parameters)
except (TypeError, ValueError):
    _MAKE_SAMPLER_PARAMS = set()


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
    max_tokens: Optional[int] = 32768
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    presence_penalty: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    extra_body: Optional[dict[str, Any]] = None


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


def _tool_response_payload(raw: str) -> Any:
    try:
        val = json.loads(raw)
        return val if val is not None else "null"
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw


def _tool_call_name_by_id(tool_calls: Any) -> dict[str, str]:
    names: dict[str, str] = {}
    if not isinstance(tool_calls, list):
        return names
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        call_id = tc.get("id")
        function = tc.get("function")
        if isinstance(call_id, str) and isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names[call_id] = name
    return names


def strip_thought_blocks(text: str) -> str:
    """Strip thinking blocks from assistant history. Handles Qwen and Gemma formats."""
    # Balanced pairs
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    text = re.sub(r"(?s)<\|think\|>.*?<\|/think\|>", "", text)
    text = re.sub(r"(?s)<\|channel>thought(?:[ \t]*\r?\n|[ \t]+).*?<channel\|>", "", text)
    # Prefilled close (generation started inside a thought already open)
    text = re.sub(r"(?s)^.*?</think>", "", text)
    text = re.sub(r"(?s)^.*?<\|/think\|>", "", text)
    text = re.sub(r"(?s)^.*?<channel\|>", "", text)
    # Orphan open (unclosed block)
    text = re.sub(r"(?s)<think>.*$", "", text)
    text = re.sub(r"(?s)<\|think\|>.*$", "", text)
    text = re.sub(r"(?s)<\|channel>thought.*$", "", text)
    # Strip any residual stray tokens
    for token in ("<think>", "</think>", "<|think|>", "<|/think|>", "<|channel>", "<channel|>"):
        text = text.replace(token, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def message_to_dict(m: ChatMessage, tool_names: dict[str, str] | None = None) -> dict:
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
        raw = normalize_content(m.content)
        name = m.name or (tool_names or {}).get(m.tool_call_id or "") or None
        msg = {
            "role": "tool",
            "content": raw,
            "tool_call_id": m.tool_call_id or "0",
        }
        if name:
            msg["name"] = name
    elif m.role == "assistant":
        msg = {
            "role": "assistant",
            "content": strip_thought_blocks(normalize_content(m.content)),
        }
    else:
        msg = {"role": m.role, "content": normalize_content(m.content)}
    return msg


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict]:
    """Convert messages while preserving tool-call id -> function name context."""
    out: list[dict] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        converted = message_to_dict(message, tool_names)
        out.append(converted)
        if message.role == "assistant" and message.tool_calls:
            tool_names.update(_tool_call_name_by_id(converted.get("tool_calls")))
    return out


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


def _scan_qwen_calls(text: str) -> list[tuple[str, dict]]:
    """Parse Qwen3 native tool calls from <tool_call>...</tool_call> blocks."""
    results: list[tuple[str, dict]] = []
    for m in re.finditer(r"(?s)<tool_call>(.*?)</tool_call>", text):
        body = m.group(1).strip()
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        results.append((name, arguments))
    return results


def extract_tool_calls(text: str, tools: Any = None) -> list[dict] | None:
    calls: list[dict] = []

    for name, arguments in _scan_qwen_calls(text):
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        )

    if calls:
        print(f"[mlxsv] extract_tool_calls matched=qwen_native count={len(calls)}")
        return calls

    print(f"[mlxsv] extract_tool_calls matched=none sample={text[:300]!r}")
    return None


_MAX_TOOL_CALLS = 2


def _dedup_cap_calls(calls: list[dict]) -> list[dict]:
    """Deduplicate by (name, arguments) and cap at _MAX_TOOL_CALLS."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for c in calls:
        fn = c.get("function", {})
        key = (fn.get("name", ""), fn.get("arguments", ""))
        if key not in seen:
            seen.add(key)
            out.append(c)
            if len(out) >= _MAX_TOOL_CALLS:
                break
    return out


# ---------------------------------------------------------------------------
# Output cleaning
# ---------------------------------------------------------------------------


def clean_output(text: str) -> str:
    return text.strip()


def _tool_streamable_thought_delta(text: str, emitted: int) -> tuple[str, int]:
    """Return only thought-block text safe to stream before tool parsing.

    Tool calls are parsed after generation, so visible answer text must stay
    buffered. Explicit thinking blocks are safe to pass through because the ACP
    client routes them to the thought stream.
    """
    stripped = text.lstrip()
    if stripped.startswith(("<tool_call>", "<|tool_call>", "<|tool_call|>", "<tool_call|>", "call:")):
        return "", emitted

    open_tags = ["<think>", "<|think|>", "<|channel>thought"]
    close_tags = ["</think>", "<|/think|>", "<channel|>"]

    close = _earliest_tag(text, close_tags)
    if close is not None:
        close_idx, close_tag = close
        safe_end = close_idx + len(close_tag)
    else:
        opened = _earliest_tag(text, open_tags)
        if opened is None:
            return "", emitted
        open_idx, _ = opened
        if emitted < open_idx:
            emitted = open_idx
        hold = max(len(tag) for tag in close_tags) - 1
        safe_end = max(0, len(text) - hold)

    if safe_end <= emitted:
        return "", emitted
    return text[emitted:safe_end], safe_end


def _earliest_tag(text: str, tags: list[str]) -> tuple[int, str] | None:
    found = [(idx, tag) for tag in tags if (idx := text.find(tag)) >= 0]
    return min(found, key=lambda item: item[0]) if found else None


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 160_000
_MAX_ANCHOR_CHARS = 120_000


def _msg_chars(msg: dict) -> int:
    total = len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        total += len(str(tc))
    for tr in msg.get("tool_responses") or []:
        total += len(str(tr))
    return total


def _is_tool_response(msg: dict) -> bool:
    return msg.get("role") == "tool"


def _trim_anchor(anchor: list[dict]) -> list[dict]:
    """Keep the user message + as many recent tool pairs as fit in _MAX_ANCHOR_CHARS."""
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
    # Don't start mid-pair with an orphaned tool response.
    while kept and _is_tool_response(kept[0]):
        kept.pop(0)
    return user_msg + kept


def truncate_messages(messages: list[dict]) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= 2:
        return system + non_system

    # Anchor on the LAST user message (current task), not the first.
    last_user_idx = None
    for i in range(len(non_system) - 1, -1, -1):
        if non_system[i].get("role") == "user" and not _is_tool_response(non_system[i]):
            last_user_idx = i
            break

    if last_user_idx is None or last_user_idx == len(non_system) - 1:
        anchor = [non_system[-1]]
        middle = non_system[:-1]
    else:
        anchor = non_system[last_user_idx:]
        middle = non_system[:last_user_idx]

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


def _extract_thought(text: str) -> str | None:
    """Return the raw thinking block from model output, or None if absent."""
    for pattern in [
        r"(?s)<think>(.*?)</think>",
        r"(?s)<\|channel>thought\s*(.*?)<channel\|>",
        r"(?s)<\|think\|>(.*?)<\|/think\|>",
    ]:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


def extract_reasoning(text: str) -> str | None:
    """Extract reasoning from model output (thought block or pre-tool text)."""
    thought = _extract_thought(text)
    if thought:
        return thought
    # Fall back: preserve text before the first tool call block
    markers = ["<tool_call>", "<|tool_call>", "<|tool_call|>"]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    reasoning = text[:cut].strip()
    # Strip any residual think tags from the pre-tool text
    reasoning = strip_thought_blocks(reasoning)
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


def _extra_body(req: ChatCompletionRequest) -> dict[str, Any]:
    return req.extra_body if isinstance(req.extra_body, dict) else {}


def _chat_template_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    extra = _extra_body(req)
    value = extra.get("chat_template_kwargs")
    return value if isinstance(value, dict) else {}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _requested_enable_thinking(
    req: ChatCompletionRequest, messages: list[dict[str, Any]]
) -> bool:
    """Prefer explicit API config; default to thinking enabled."""
    template_kwargs = _chat_template_kwargs(req)
    explicit = _coerce_bool(template_kwargs.get("enable_thinking"))
    if explicit is not None:
        return explicit

    explicit = _coerce_bool(_extra_body(req).get("enable_thinking"))
    if explicit is not None:
        return explicit

    return True


def _consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Consolidate multiple system turns into one for models that expect a single system turn."""
    system_parts = [
        normalize_content(msg.get("content")).strip()
        for msg in messages
        if msg.get("role") == "system" and normalize_content(msg.get("content")).strip()
    ]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    if not system_parts:
        return non_system
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def _extra_number(req: ChatCompletionRequest, key: str) -> Any:
    return _extra_body(req).get(key)


def _log_first_message(messages: list[dict[str, Any]]) -> None:
    if not messages:
        print("[mlxsv] first_model_message role=<none> chars=0 preview=", flush=True)
        return

    first = messages[0]
    content = normalize_content(first.get("content"))
    preview = content[:300].replace("\n", "\\n")
    print(
        f"[mlxsv] first_model_message role={first.get('role', '<missing>')} "
        f"chars={len(content)} preview={preview}",
        flush=True,
    )


def _make_sampler(req: ChatCompletionRequest):
    # Qwen recommended sampling:
    #   thinking mode:     temperature=0.6, top_p=0.95, top_k=20, min_p=0
    #   non-thinking mode: temperature=0.7, top_p=0.8,  top_k=20, min_p=0
    enable_thinking = _coerce_bool(_extra_body(req).get("enable_thinking"))
    thinking = enable_thinking is not False  # default to thinking if unset

    # Use mode-appropriate defaults; respect an explicit client value only if it
    # differs from the thinking-mode default (0.6), meaning it was intentionally set.
    default_temp = 0.6 if thinking else 0.7
    requested_temp = req.temperature
    if not thinking and requested_temp is not None and abs(requested_temp - 0.6) < 1e-6:
        # Client sent the thinking-mode default — override to non-thinking default.
        temperature = 0.7
    else:
        temperature = requested_temp if requested_temp is not None else default_temp

    top_k = _extra_number(req, "top_k")
    kwargs: dict[str, Any] = {}

    candidates = {
        "top_p": req.top_p if req.top_p is not None else (0.95 if thinking else 0.8),
        "min_p": _extra_number(req, "min_p") if _extra_number(req, "min_p") is not None else 0.0,
        "top_k": top_k if top_k is not None else 20,
        "repetition_penalty": _extra_number(req, "repetition_penalty"),
        "presence_penalty": req.presence_penalty,
    }
    for key, value in candidates.items():
        if value is not None and key in _MAKE_SAMPLER_PARAMS:
            kwargs[key] = value

    if "temp" in _MAKE_SAMPLER_PARAMS:
        return make_sampler(temp=temperature, **kwargs)
    if "temperature" in _MAKE_SAMPLER_PARAMS:
        return make_sampler(temperature=temperature, **kwargs)
    return make_sampler(temperature, **kwargs)


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request):
    messages = messages_to_dicts(req.messages)
    messages = truncate_messages(messages)
    messages = _consolidate_system_messages(messages)

    requested_enable_thinking = _requested_enable_thinking(req, messages)

    print(
        f"[mlxsv] chat_completions tools_in_request={len(req.tools or [])} "
        f"messages={len(messages)} enable_thinking={requested_enable_thinking}",
        flush=True,
    )
    # _log_first_message(messages)

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    for key, value in _chat_template_kwargs(req).items():
        if key not in {"tools"}:
            template_kwargs[key] = value
    template_kwargs["enable_thinking"] = requested_enable_thinking
    if req.tools:
        template_kwargs["tools"] = req.tools

    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as e:
        # Drop enable_thinking first — keep tools so the model sees the schemas.
        dropped = template_kwargs.pop("enable_thinking", None)
        if dropped is not None:
            print(f"[mlxsv] template_warning dropped=enable_thinking reason={e!r}")
        try:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as e2:
            template_kwargs.pop("tools", None)
            print(f"[mlxsv] template_warning dropped=tools reason={e2!r}")
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if req.stream:

        async def event_stream():
            accumulated = ""
            tool_thought_emitted = 0
            with _inference_lock:
                try:
                    from mlx_lm import stream_generate as _stream_gen

                    for resp in _stream_gen(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=req.max_tokens or 32768,
                        sampler=_make_sampler(req),
                        prompt_cache=_make_turbo_cache(),
                    ):
                        if await request.is_disconnected():
                            print("[mlxsv] stream_cancelled client_disconnected=True")
                            return
                        token = resp.text
                        accumulated += token
                        if req.tools:
                            chunk, tool_thought_emitted = (
                                _tool_streamable_thought_delta(
                                    accumulated, tool_thought_emitted
                                )
                            )
                            if chunk:
                                delta_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": PUBLIC_MODEL_NAME,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": chunk},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(delta_chunk)}\n\n"
                            continue
                        delta_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": PUBLIC_MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"
                except Exception:
                    if await request.is_disconnected():
                        print("[mlxsv] stream_cancelled before_fallback=True")
                        return
                    # Fallback: generate all at once, send as single chunk
                    raw = generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=req.max_tokens or 32768,
                        sampler=_make_sampler(req),
                        prompt_cache=_make_turbo_cache(),
                    )
                    accumulated = raw
                    if not req.tools:
                        delta_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": PUBLIC_MODEL_NAME,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": accumulated},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"

            if await request.is_disconnected():
                print("[mlxsv] stream_cancelled after_generation=True")
                return

            # After stream ends, parse tool calls from full accumulated text.
            text = clean_output(accumulated)
            print(f"[mlxsv] raw_output={text[:2000]!r}")
            raw_calls = extract_tool_calls(text, req.tools) if req.tools else None
            tool_calls = _dedup_cap_calls(raw_calls) if raw_calls else None
            finish_reason = "tool_calls" if tool_calls else "stop"
            print(
                f"[mlxsv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}"
            )

            final_delta: dict[str, Any] = {}
            if tool_calls:
                final_delta["tool_calls"] = tool_calls
            elif req.tools and text:
                final_delta["content"] = text
            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": PUBLIC_MODEL_NAME,
                "choices": [
                    {"index": 0, "delta": final_delta, "finish_reason": finish_reason}
                ],
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
            max_tokens=req.max_tokens or 32768,
            sampler=_make_sampler(req),
            prompt_cache=_make_turbo_cache(),
        )

    text = clean_output(raw)
    print(f"[mlxsv] raw_output={text[:2000]!r}")
    raw_calls = extract_tool_calls(text, req.tools) if req.tools else None
    tool_calls = _dedup_cap_calls(raw_calls) if raw_calls else None
    finish_reason = "tool_calls" if tool_calls else "stop"
    reasoning = extract_reasoning(text) if tool_calls else strip_thought_blocks(text)

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
