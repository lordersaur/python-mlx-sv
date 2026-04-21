import json
import os
import re
import threading
import time
import uuid
import warnings
import inspect
from typing import Any, Optional

import optiq  # registers qwen3_5_text model type with mlx_lm
from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel

app = FastAPI()

DEFAULT_MODEL_NAME = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
STALE_MODEL_ALIASES = {
    "mlx",
    "mlx-community",
    "mlx-community/Qwen3-14B-4bit",
    "PewterZz/OmniCoder-9B-abliterated-MLX-4bit",
}


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
# Qwen3.5 tokenizer update moved EOS to <|endoftext|> (248044) but the chat
# template still uses <|im_end|> (248046) as the turn terminator. Add it so
# the generator stops there instead of bleeding into hallucinated next turns.
_im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
if _im_end_id and _im_end_id not in tokenizer.eos_token_ids:
    tokenizer.eos_token_ids.add(_im_end_id)
patch_attention()
_text_cfg = getattr(model.args, 'text_config', {})
_head_dim = (_text_cfg.get('head_dim') if isinstance(_text_cfg, dict) else getattr(_text_cfg, 'head_dim', None)) or 128
_inference_lock = threading.Lock()


def _make_turbo_cache():
    """Per-request cache: TurboQuantKVCache for full-attention layers, standard for linear-attention."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    caches = make_prompt_cache(model)
    for i, c in enumerate(caches):
        if isinstance(c, KVCache):
            caches[i] = TurboQuantKVCache(head_dim=_head_dim, bits=4)
    return caches


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
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
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


def _unescape_value(s: str, *, decode_control_escapes: bool) -> str:
    """Decode escapes in a parsed tool argument value.

    Qwen compact tool-call arguments may use <|"|>...<|"|> delimited strings.
    Preserve source escapes for delimited strings so patch_file_tool can match
    file contents such as `"one\\ntwo"` or `.join("\\n")`.
    """
    result: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if decode_control_escapes and nxt == "n":
                result.append("\n")
                i += 2
                continue
            if decode_control_escapes and nxt == "t":
                result.append("\t")
                i += 2
                continue
            if decode_control_escapes and nxt == "r":
                result.append("\r")
                i += 2
                continue
            if nxt == "\\":
                result.append("\\")
                i += 2
                continue
            if nxt == '"':
                result.append('"')
                i += 2
                continue
            if nxt == "'":
                result.append("'")
                i += 2
                continue
        result.append(c)
        i += 1
    return "".join(result)


def _normalize_qwen_args(raw: str) -> dict | None:
    trimmed = raw.strip()
    if trimmed.startswith("{{") and trimmed.endswith("}}"):
        trimmed = trimmed[1:-1]

    # Only attempt the fast json.loads path when there are NO <|"|> delimiters,
    # since replacing them globally can collide with literal " inside values.
    if "<|\"|>" not in trimmed:
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
                    i += 2
                    continue
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


_QWEN_COMPACT_CALL_PREFIX = "<|tool_call>call:"


def _parse_qwen_compact_call_name_and_body_start(
    text: str, start: int
) -> tuple[str, int] | None:
    j = start + len(_QWEN_COMPACT_CALL_PREFIX)
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


def _scan_qwen_compact_body(text: str, body_start: int) -> tuple[str, int] | None:
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
            depth += 1
            k += 1
        elif c == "}":
            depth -= 1
            k += 1
            if depth == 0:
                return text[body_start:k], k
        else:
            k += 1
    return None


def _scan_qwen_compact_calls(text: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    pos = 0
    while True:
        i = text.find(_QWEN_COMPACT_CALL_PREFIX, pos)
        if i < 0:
            break
        parsed = _parse_qwen_compact_call_name_and_body_start(text, i)
        if parsed is None:
            pos = i + len(_QWEN_COMPACT_CALL_PREFIX)
            continue
        name, body_start = parsed
        scanned = _scan_qwen_compact_body(text, body_start)
        if scanned is None:
            pos = body_start + 1
            continue
        body, pos = scanned
        results.append((name, body))
    return results


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

    for name, body in _scan_qwen_compact_calls(text):
        args = _normalize_qwen_args(body)
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


_MAX_TOOL_CALLS = 3


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
    """Remove control tokens and thinking blocks."""
    # Truncate at the first <|im_end|> — that is the model's own end-of-turn
    # marker. Everything after it is the hallucinated next conversation turn
    # (<|im_start|>user\n...<|im_start|>assistant\n... repeating tool calls).
    # Truncate before stripping so the marker still serves as a boundary.
    first_im_end = text.find("<|im_end|>")
    if first_im_end != -1:
        text = text[:first_im_end]
    for token in ["<turn|>", "<|turn>"]:
        text = text.replace(token, "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

_MAX_CONTEXT_CHARS = 100_000
_MAX_ANCHOR_CHARS = 75_000


def _msg_chars(msg: dict) -> int:
    total = len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        total += len(str(tc))
    return total


def _is_tool_response(msg: dict) -> bool:
    content = msg.get("content") or ""
    return isinstance(content, str) and content.startswith("<tool_response>")


def _batch_tool_responses(messages: list[dict]) -> list[dict]:
    """Merge consecutive tool-response user messages into one user message.

    The Qwen3.5 chat template expects all tool results for a turn in a single
    user message with multiple <tool_response> blocks, not separate messages.
    """
    out: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if _is_tool_response(msg):
            combined = msg["content"]
            i += 1
            while i < len(messages) and _is_tool_response(messages[i]):
                combined += "\n" + messages[i]["content"]
                i += 1
            out.append({"role": "user", "content": combined})
        else:
            out.append(msg)
            i += 1
    return out


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


def _extract_thought(text: str) -> str | None:
    """Return the raw thinking block from model output, or None if absent."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


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
    """Prefer explicit API config; keep /no_think only as a compatibility fallback."""
    template_kwargs = _chat_template_kwargs(req)
    explicit = _coerce_bool(template_kwargs.get("enable_thinking"))
    if explicit is not None:
        return explicit

    explicit = _coerce_bool(_extra_body(req).get("enable_thinking"))
    if explicit is not None:
        return explicit

    no_think = any(
        "/no_think" in (msg.get("content") or "")
        for msg in messages
        if msg.get("role") == "system"
    )
    return not no_think


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


def _make_qwen_sampler(req: ChatCompletionRequest):
    temperature = req.temperature if req.temperature is not None else 0.7
    kwargs: dict[str, Any] = {}

    candidates = {
        "top_p": req.top_p,
        "min_p": _extra_number(req, "min_p"),
        "top_k": _extra_number(req, "top_k"),
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
def chat_completions(req: ChatCompletionRequest):
    messages = [message_to_dict(m) for m in req.messages]
    messages = _batch_tool_responses(messages)
    messages = truncate_messages(messages)

    requested_enable_thinking = _requested_enable_thinking(req, messages)

    print(
        f"[mlxsv] chat_completions tools_in_request={len(req.tools or [])} "
        f"messages={len(messages)} enable_thinking={requested_enable_thinking}",
        flush=True,
    )
    _log_first_message(messages)

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
        if dropped is False:
            return JSONResponse(
                {
                    "error": {
                        "message": "Qwen3.5 non-thinking mode requires chat_template_kwargs.enable_thinking=False support in the tokenizer chat template.",
                        "type": "template_configuration_error",
                    }
                },
                status_code=500,
            )
        try:
            prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as e2:
            template_kwargs.pop("tools", None)
            print(f"[mlxsv] template_warning dropped=tools reason={e2!r}")
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
                        max_tokens=req.max_tokens or 32768,
                        sampler=_make_qwen_sampler(req),
                        prompt_cache=_make_turbo_cache(),
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
                        max_tokens=req.max_tokens or 32768,
                        sampler=_make_qwen_sampler(req),
                        prompt_cache=_make_turbo_cache(),
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
            print(f"[mlxsv] raw_output={text[:2000]!r}")
            raw_calls = extract_tool_calls(text) if req.tools else None
            tool_calls = _dedup_cap_calls(raw_calls) if raw_calls else None
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
            max_tokens=req.max_tokens or 32768,
            sampler=_make_qwen_sampler(req),
            prompt_cache=_make_turbo_cache(),
        )

    text = clean_output(raw)
    print(f"[mlxsv] raw_output={text[:2000]!r}")
    raw_calls = extract_tool_calls(text) if req.tools else None
    tool_calls = _dedup_cap_calls(raw_calls) if raw_calls else None
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
