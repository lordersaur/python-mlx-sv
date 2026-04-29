import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from llama_cpp import Llama
from transformers import AutoTokenizer
from pydantic import BaseModel

app = FastAPI()

DEFAULT_MODEL_REPO = "unsloth/gemma-4-E4B-it-GGUF"
DEFAULT_MODEL_FILE = "gemma-4-E4B-it-Q8_0.gguf"
_TOKENIZER_NAME = "unsloth/gemma-4-E4B-it"

MODEL_REPO = (os.environ.get("MODEL_REPO") or "").strip() or DEFAULT_MODEL_REPO
MODEL_FILE = (os.environ.get("MODEL_FILE") or "").strip() or DEFAULT_MODEL_FILE
PUBLIC_MODEL_NAME = f"{MODEL_REPO}/{MODEL_FILE}"

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False,
)
tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

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


def strip_gemma_thoughts(text: str) -> str:
    """Strip Gemma thinking blocks from assistant history before replay."""
    text = re.sub(r"(?s)<\|channel>thought\s*.*?<channel\|>", "", text)
    text = re.sub(r"(?s)<\|think\|>.*?<\|/think\|>", "", text)
    text = re.sub(r"(?s)<\|channel>thought\s*.*$", "", text)
    text = re.sub(r"(?s)<\|think\|>.*$", "", text)
    for token in ("<|channel>", "<channel|>", "<|think|>", "<|/think|>"):
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
        return {
            "role": "assistant",
            "content": normalize_content(m.content) or "",
            "tool_calls": cooked,
        }
    elif m.role == "tool":
        raw = normalize_content(m.content)
        name = (
            m.name
            or (tool_names or {}).get(m.tool_call_id or "")
            or (m.tool_call_id or "tool")
        )
        return {
            "role": "tool",
            "content": "",
            "tool_call_id": m.tool_call_id or "0",
            "tool_responses": [
                {
                    "name": name,
                    "response": _tool_response_payload(raw),
                }
            ],
        }
    elif m.role == "assistant":
        return {
            "role": "assistant",
            "content": strip_gemma_thoughts(normalize_content(m.content)),
        }
    else:
        return {"role": m.role, "content": normalize_content(m.content)}


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict]:
    """Convert messages using Gemma 4's native tool format.

    Tool messages are attached as tool_responses on the preceding assistant
    message. Consecutive tool results are merged into a single assistant turn
    to satisfy Gemma 4's strict user/assistant alternation.
    """
    out: list[dict] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        converted = message_to_dict(message, tool_names)
        if converted.get("role") == "tool":
            _attach_gemma_tool_response(out, converted)
        else:
            out.append(converted)
        if message.role == "assistant" and message.tool_calls:
            tool_names.update(_tool_call_name_by_id(converted.get("tool_calls")))
    return out


def _attach_gemma_tool_response(out: list[dict], tool_message: dict) -> None:
    """Attach a tool result to the preceding assistant message with tool_calls."""
    responses = tool_message.get("tool_responses") or []
    if not responses:
        return
    tool_call_id = tool_message.get("tool_call_id")
    for msg in reversed(out):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        if tool_call_id and not _assistant_has_tool_call_id(msg, tool_call_id):
            continue
        msg.setdefault("tool_responses", []).extend(responses)
        return
    out.append(tool_message)


def _assistant_has_tool_call_id(msg: dict, tool_call_id: str) -> bool:
    for call in msg.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id") == tool_call_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Tool call parsing — Gemma 4 native format
# ---------------------------------------------------------------------------


def _unescape_value(s: str, *, decode_control_escapes: bool) -> str:
    result: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if decode_control_escapes and nxt == "n":
                result.append("\n"); i += 2; continue
            if decode_control_escapes and nxt == "t":
                result.append("\t"); i += 2; continue
            if decode_control_escapes and nxt == "r":
                result.append("\r"); i += 2; continue
            if nxt == "\\":
                result.append("\\"); i += 2; continue
            if nxt == '"':
                result.append('"'); i += 2; continue
            if nxt == "'":
                result.append("'"); i += 2; continue
        result.append(c)
        i += 1
    return "".join(result)


def _normalize_gemma_args(raw: str) -> dict | None:
    trimmed = raw.strip()
    if trimmed.startswith("{{") and trimmed.endswith("}}"):
        trimmed = trimmed[1:-1]

    if '<|"|>' not in trimmed:
        normalized = re.sub(r"([{,]\s*)(\w+)\s*:", r'\1"\2":', trimmed)
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
            value: Any = _unescape_value(inner[i:end], decode_control_escapes=False)
            i = end + len('<|"|>')
        elif inner[i] == '"':
            i += 1
            start = i
            while i < n:
                if inner[i] == "\\" and i + 1 < n:
                    i += 2; continue
                if inner[i] == '"':
                    break
                i += 1
            if i >= n:
                return None
            value = _unescape_value(inner[start:i], decode_control_escapes=True)
            i += 1
        else:
            start = i
            depth = 0
            while i < n:
                ch = inner[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == "," and depth == 0:
                    break
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


_GEMMA_CALL_PREFIXES = ("<|tool_call>call:", "<|tool_call|>call:")


def _find_next_gemma_call_prefix(text: str, pos: int) -> tuple[int, str]:
    matches = [
        (idx, prefix)
        for prefix in _GEMMA_CALL_PREFIXES
        if (idx := text.find(prefix, pos)) >= 0
    ]
    return min(matches, key=lambda item: item[0]) if matches else (-1, "")


def _parse_gemma_call_name_and_body_start(
    text: str, start: int, prefix: str
) -> tuple[str, int] | None:
    j = start + len(prefix)
    name_start = j
    while j < len(text) and (text[j].isalnum() or text[j] == "_"):
        j += 1
    if j == name_start:
        return None
    name = text[name_start:j]
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text) or text[j] != "{":
        return None
    return name, j


def _scan_gemma_call_body(text: str, body_start: int) -> tuple[str, int] | None:
    depth = 0
    k = body_start
    while k < len(text):
        if text.startswith('<|"|>', k):
            k += len('<|"|>')
            end = text.find('<|"|>', k)
            if end < 0:
                return None
            k = end + len('<|"|>')
            continue
        c = text[k]
        if c == "{":
            depth += 1; k += 1
        elif c == "}":
            depth -= 1; k += 1
            if depth == 0:
                return text[body_start:k], k
        else:
            k += 1
    return None


def _scan_gemma_calls(text: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    pos = 0
    close_tag = "<tool_call|>"
    while True:
        i, prefix = _find_next_gemma_call_prefix(text, pos)
        if i < 0:
            break
        parsed = _parse_gemma_call_name_and_body_start(text, i, prefix)
        if parsed is None:
            pos = i + len(prefix); continue
        name, body_start = parsed
        scanned = _scan_gemma_call_body(text, body_start)
        if scanned is None:
            pos = body_start + 1; continue
        body, body_end = scanned
        tail = text[body_end:].lstrip()
        if not tail.startswith(close_tag):
            pos = body_end; continue
        pos = body_end + (len(text[body_end:]) - len(tail)) + len(close_tag)
        results.append((name, body))
    return results


# Gemma 4 behavioral cap: model reliably emits at most 3 parallel tool calls.
_MAX_TOOL_CALLS = 3


def extract_tool_calls(text: str) -> list[dict] | None:
    calls: list[dict] = []
    for name, body in _scan_gemma_calls(text):
        args = _normalize_gemma_args(body)
        if args is not None:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )

    if calls:
        # Deduplicate and cap.
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for c in calls:
            fn = c.get("function", {})
            key = (fn.get("name", ""), fn.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(c)
                if len(deduped) >= _MAX_TOOL_CALLS:
                    break
        print(f"[vllmsv] extract_tool_calls matched=gemma_native count={len(deduped)}")
        return deduped

    print(f"[vllmsv] extract_tool_calls matched=none sample={text[:300]!r}")
    return None


# ---------------------------------------------------------------------------
# Thinking block streaming helper
# ---------------------------------------------------------------------------


def _earliest_tag(text: str, tags: list[str]) -> tuple[int, str] | None:
    found = [(idx, tag) for tag in tags if (idx := text.find(tag)) >= 0]
    return min(found, key=lambda item: item[0]) if found else None


def _tool_streamable_thought_delta(text: str, emitted: int) -> tuple[str, int]:
    """Return the slice of thinking-block text safe to stream before tool parsing."""
    stripped = text.lstrip()
    if stripped.startswith(("<|tool_call>", "<|tool_call|>", "<tool_call|>", "call:")):
        return "", emitted

    open_tags = ["<|think|>", "<|channel>thought"]
    close_tags = ["<|/think|>", "<channel|>"]

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


def extract_reasoning(text: str) -> str | None:
    markers = ["<|tool_call>", "<|tool_call|>"]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    reasoning = text[:cut].strip()
    return reasoning if reasoning else None


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
    while kept and _is_tool_response(kept[0]):
        kept.pop(0)
    return user_msg + kept


def truncate_messages(messages: list[dict]) -> list[dict]:
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) <= 2:
        return system + non_system

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
        print(f"[vllmsv] context_truncated dropped_turns={dropped} kept_turns={len(kept)}")

    return system + kept + anchor


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def _extra_body(req: ChatCompletionRequest) -> dict[str, Any]:
    return req.extra_body if isinstance(req.extra_body, dict) else {}


def _chat_template_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    value = _extra_body(req).get("chat_template_kwargs")
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


def _requested_enable_thinking(req: ChatCompletionRequest) -> bool:
    template_kwargs = _chat_template_kwargs(req)
    explicit = _coerce_bool(template_kwargs.get("enable_thinking"))
    if explicit is not None:
        return explicit
    explicit = _coerce_bool(_extra_body(req).get("enable_thinking"))
    if explicit is not None:
        return explicit
    return True


def _consolidate_system_messages(messages: list[dict]) -> list[dict]:
    """Gemma 4 expects a single system turn."""
    system_parts = [
        normalize_content(msg.get("content")).strip()
        for msg in messages
        if msg.get("role") == "system" and normalize_content(msg.get("content")).strip()
    ]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    if not system_parts:
        return non_system
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def _apply_thinking_marker(messages: list[dict], enable_thinking: bool) -> list[dict]:
    out = [dict(msg) for msg in messages]
    system_index = next(
        (idx for idx, msg in enumerate(out) if msg.get("role") == "system"), None
    )
    if system_index is None:
        if enable_thinking:
            out.insert(0, {"role": "system", "content": "<|think|>"})
        return out
    content = normalize_content(out[system_index].get("content"))
    content = content.replace("<|think|>", "").strip()
    if enable_thinking:
        content = f"<|think|>\n{content}" if content else "<|think|>"
    out[system_index]["content"] = content
    return out


def _make_gen_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    extra = _extra_body(req)
    kwargs: dict[str, Any] = {
        "max_tokens": req.max_tokens or 32768,
        "temperature": req.temperature if req.temperature is not None else 1.0,
        "top_p": req.top_p if req.top_p is not None else 0.95,
        "stop": ["<end_of_turn>", "<|end_of_turn|>", "<turn|>"],
    }
    if extra.get("top_k") is not None:
        kwargs["top_k"] = extra["top_k"]
    if extra.get("min_p") is not None:
        kwargs["min_p"] = extra["min_p"]
    if extra.get("repeat_penalty") is not None:
        kwargs["repeat_penalty"] = extra["repeat_penalty"]
    if req.presence_penalty is not None:
        kwargs["presence_penalty"] = req.presence_penalty
    return kwargs


def _build_prompt(req: ChatCompletionRequest) -> tuple[str, bool]:
    """Return (prompt_string, enable_thinking)."""
    messages = messages_to_dicts(req.messages)
    messages = truncate_messages(messages)
    messages = _consolidate_system_messages(messages)

    enable_thinking = _requested_enable_thinking(req)
    messages = _apply_thinking_marker(messages, enable_thinking)

    roles = "→".join(m["role"] for m in messages)
    print(
        f"[vllmsv] chat_completions tools_in_request={len(req.tools or [])} "
        f"messages={len(messages)} enable_thinking={enable_thinking} roles={roles}",
        flush=True,
    )

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    for key, value in _chat_template_kwargs(req).items():
        if key not in {"tools"}:
            template_kwargs[key] = value
    if req.tools:
        template_kwargs["tools"] = req.tools

    try:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as e:
        dropped = template_kwargs.pop("enable_thinking", None)
        if dropped is not None:
            print(f"[vllmsv] template_warning dropped=enable_thinking reason={e!r}")
        try:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as e2:
            template_kwargs.pop("tools", None)
            print(f"[vllmsv] template_warning dropped=tools reason={e2!r}")
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)

    return prompt, enable_thinking


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
async def chat_completions(req: ChatCompletionRequest, request: Request):
    prompt, enable_thinking = _build_prompt(req)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    gen_kwargs = _make_gen_kwargs(req)

    if req.stream:

        async def event_stream():
            accumulated = ""
            tool_thought_emitted = 0
            token_queue: queue.Queue = queue.Queue()

            def producer():
                try:
                    with _inference_lock:
                        for chunk in llm(prompt, stream=True, **gen_kwargs):
                            token_queue.put(chunk["choices"][0]["text"])
                except Exception as e:
                    print(f"[vllmsv] stream_error={e!r}")
                finally:
                    token_queue.put(None)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                if await request.is_disconnected():
                    print("[vllmsv] stream_cancelled client_disconnected=True")
                    return
                token = await loop.run_in_executor(None, token_queue.get)
                if token is None:
                    break
                accumulated += token
                if req.tools:
                    chunk, tool_thought_emitted = _tool_streamable_thought_delta(
                        accumulated, tool_thought_emitted
                    )
                    if chunk:
                        yield f"data: {json.dumps(_chunk(completion_id, chunk))}\n\n"
                    continue
                if token:
                    yield f"data: {json.dumps(_chunk(completion_id, token))}\n\n"

            if await request.is_disconnected():
                print("[vllmsv] stream_cancelled after_generation=True")
                return

            text = accumulated.strip()
            print(f"[vllmsv] raw_output={text[:2000]!r}")
            tool_calls = extract_tool_calls(text) if req.tools else None
            finish_reason = "tool_calls" if tool_calls else "stop"
            print(f"[vllmsv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}")

            final_delta: dict[str, Any] = {}
            if tool_calls:
                final_delta["tool_calls"] = tool_calls
            elif req.tools and text:
                final_delta["content"] = text
            yield f"data: {json.dumps(_final_chunk(completion_id, final_delta, finish_reason))}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path.
    with _inference_lock:
        output = llm(prompt, **gen_kwargs)

    raw = output["choices"][0]["text"]
    text = raw.strip()
    print(f"[vllmsv] raw_output={text[:2000]!r}")
    tool_calls = extract_tool_calls(text) if req.tools else None
    finish_reason = "tool_calls" if tool_calls else "stop"
    reasoning = extract_reasoning(text) if tool_calls else text

    print(f"[vllmsv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}")
    response_message: dict[str, Any] = {"role": "assistant", "content": reasoning}
    if tool_calls:
        response_message["tool_calls"] = tool_calls

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


def _chunk(completion_id: str, content: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": PUBLIC_MODEL_NAME,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def _final_chunk(completion_id: str, delta: dict, finish_reason: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": PUBLIC_MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
