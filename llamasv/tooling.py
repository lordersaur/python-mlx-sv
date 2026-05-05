import json
import re
import uuid
from typing import Any

QWEN_TOOL_BLOCK_RE = re.compile(
    r"(?s)<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</function>\s*</tool_call>"
)
QWEN_PARAMETER_RE = re.compile(
    r"(?s)<parameter=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</parameter>"
)


def _unescape_value(s: str, *, decode_control_escapes: bool) -> str:
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
    parsed = _parse_gemma_value(trimmed, 0)
    if parsed is None:
        return None
    value, i = parsed
    i = _skip_gemma_ws(trimmed, i)
    if i != len(trimmed) or not isinstance(value, dict):
        return None
    return value


def _coerce_tool_value(raw: str) -> Any:
    trimmed = raw.strip()
    if not trimmed:
        return ""
    if trimmed in {"true", "false"}:
        return trimmed == "true"
    if trimmed == "null":
        return None
    if re.fullmatch(r"-?\d+", trimmed):
        try:
            return int(trimmed)
        except ValueError:
            return trimmed
    if re.fullmatch(r"-?\d+\.\d+", trimmed):
        try:
            return float(trimmed)
        except ValueError:
            return trimmed
    if trimmed[0] in {'"', "{", "["}:
        try:
            return json.loads(trimmed)
        except (json.JSONDecodeError, TypeError, ValueError):
            return trimmed
    return trimmed


def _skip_gemma_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n,":
        i += 1
    return i


def _parse_gemma_value(text: str, i: int) -> tuple[Any, int] | None:
    i = _skip_gemma_ws(text, i)
    if i >= len(text):
        return None

    if text.startswith('<|"|>', i):
        return _parse_gemma_pipe_string(text, i)
    if text[i] == '"':
        return _parse_gemma_quoted_string(text, i)
    if text[i] == "{":
        return _parse_gemma_object(text, i)
    if text[i] == "[":
        return _parse_gemma_array(text, i)
    return _parse_gemma_scalar(text, i)


def _parse_gemma_pipe_string(text: str, i: int) -> tuple[str, int] | None:
    i += len('<|"|>')
    end = text.find('<|"|>', i)
    if end == -1:
        return None
    return _unescape_value(text[i:end], decode_control_escapes=False), end + len('<|"|>')


def _parse_gemma_quoted_string(text: str, i: int) -> tuple[str, int] | None:
    i += 1
    start = i
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == '"':
            return _unescape_value(text[start:i], decode_control_escapes=True), i + 1
        i += 1
    return None


def _parse_gemma_key(text: str, i: int) -> tuple[str, int] | None:
    i = _skip_gemma_ws(text, i)
    if i >= len(text):
        return None
    if text.startswith('<|"|>', i):
        parsed = _parse_gemma_pipe_string(text, i)
        if parsed is None or not isinstance(parsed[0], str):
            return None
        return parsed
    if text[i] == '"':
        parsed = _parse_gemma_quoted_string(text, i)
        if parsed is None or not isinstance(parsed[0], str):
            return None
        return parsed

    start = i
    while i < len(text) and (text[i].isalnum() or text[i] == "_"):
        i += 1
    if i == start:
        return None
    return text[start:i], i


def _parse_gemma_object(text: str, i: int) -> tuple[dict[str, Any], int] | None:
    if text[i] != "{":
        return None
    i += 1
    out: dict[str, Any] = {}

    while True:
        i = _skip_gemma_ws(text, i)
        if i >= len(text):
            return None
        if text[i] == "}":
            return out, i + 1
        if text.startswith('<|"|>', i):
            # Tolerate stray pipe-string delimiters before object close.
            parsed = _parse_gemma_pipe_string(text, i)
            if parsed is None:
                return None
            value, i = parsed
            if value == "":
                i = _skip_gemma_ws(text, i)
                if i < len(text) and text[i] == "}":
                    return out, i + 1
            return None

        parsed_key = _parse_gemma_key(text, i)
        if parsed_key is None:
            return None
        key, i = parsed_key
        i = _skip_gemma_ws(text, i)
        if i >= len(text) or text[i] != ":":
            return None
        i += 1

        parsed_value = _parse_gemma_value(text, i)
        if parsed_value is None:
            return None
        value, i = parsed_value
        out[key] = value

        i = _skip_gemma_ws(text, i)
        if i < len(text) and text[i] == "}":
            return out, i + 1


def _parse_gemma_array(text: str, i: int) -> tuple[list[Any], int] | None:
    if text[i] != "[":
        return None
    i += 1
    out: list[Any] = []

    while True:
        i = _skip_gemma_ws(text, i)
        if i >= len(text):
            return None
        if text[i] == "]":
            return out, i + 1

        parsed = _parse_gemma_value(text, i)
        if parsed is None:
            return None
        value, i = parsed
        out.append(value)

        i = _skip_gemma_ws(text, i)
        if i < len(text) and text[i] == "]":
            return out, i + 1


def _parse_gemma_scalar(text: str, i: int) -> tuple[Any, int] | None:
    start = i
    while i < len(text) and text[i] not in ",]}":
        i += 1
    token = text[start:i].strip()
    if not token:
        return None
    if token in {"true", "false"}:
        return token == "true", i
    if token == "null":
        return None, i
    try:
        return int(token), i
    except ValueError:
        pass
    try:
        return float(token), i
    except ValueError:
        pass
    return token, i


GEMMA_CALL_PREFIXES = ("<|tool_call>call:", "<|tool_call|>call:")
MAX_TOOL_CALLS = 3


def _find_next_gemma_call_prefix(text: str, pos: int) -> tuple[int, str]:
    matches = [
        (idx, prefix)
        for prefix in GEMMA_CALL_PREFIXES
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
            pos = i + len(prefix)
            continue
        name, body_start = parsed
        scanned = _scan_gemma_call_body(text, body_start)
        if scanned is None:
            pos = body_start + 1
            continue
        body, body_end = scanned
        tail = text[body_end:].lstrip()
        if not tail.startswith(close_tag):
            pos = body_end
            continue
        pos = body_end + (len(text[body_end:]) - len(tail)) + len(close_tag)
        results.append((name, body))
    return results


def _scan_qwen_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    results: list[tuple[str, dict[str, Any]]] = []
    for match in QWEN_TOOL_BLOCK_RE.finditer(text):
        name = match.group(1)
        body = match.group(2).strip()
        args: dict[str, Any] = {}
        if body.startswith("{") and body.endswith("}"):
            parsed = _normalize_gemma_args(body)
            if isinstance(parsed, dict):
                args = parsed
        else:
            for param in QWEN_PARAMETER_RE.finditer(body):
                args[param.group(1)] = _coerce_tool_value(param.group(2))
        results.append((name, args))
    return results


_FALLBACK_CALL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\{")
_BARE_TOOL_LINE_RE = re.compile(r"(?m)^(?!\[)([a-z_][a-z0-9_]*)\s*$")


def has_tool_call_cue(text: str) -> bool:
    tail = text[-1024:]
    if any(
        marker in tail
        for marker in ("<|tool_call", "<tool_call|>", "<tool_call>", "<function=")
    ):
        return True
    if _FALLBACK_CALL_RE.search(tail):
        return True
    if _BARE_TOOL_LINE_RE.search(tail):
        return True
    return False


def _scan_fallback_calls(text: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    pos = 0
    close_tag = "<tool_call|>"
    while True:
        match = _FALLBACK_CALL_RE.search(text, pos)
        if match is None:
            break
        name = match.group(1)
        body_start = text.find("{", match.start(1))
        if body_start < 0:
            break
        scanned = _scan_gemma_call_body(text, body_start)
        if scanned is None:
            end = len(text)
            body = text[body_start:end].strip()
            if body:
                results.append((name, body))
            pos = end
            continue
        body, body_end = scanned
        tail = text[body_end:].lstrip()
        if tail.startswith(close_tag):
            pos = body_end + (len(text[body_end:]) - len(tail)) + len(close_tag)
        else:
            pos = body_end
        results.append((name, body))
    return results


def _scan_bare_tool_lines(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    if not stripped or "{" in stripped or "<tool_call" in stripped:
        return []

    matches = list(_BARE_TOOL_LINE_RE.finditer(stripped))
    if not matches:
        return []

    last = matches[-1].group(1)
    return [(last, "{}")]


def extract_tool_calls(text: str) -> list[dict] | None:
    calls: list[dict] = []
    scanned_mode = "none"
    scanned_qwen_calls = _scan_qwen_calls(text)
    if scanned_qwen_calls:
        scanned_mode = "qwen_xml"
        for name, args in scanned_qwen_calls:
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            )

    scanned_calls = _scan_gemma_calls(text) if not calls else []
    if not scanned_calls:
        scanned_calls = _scan_fallback_calls(text)
        if scanned_calls:
            scanned_mode = "fallback"
    if not scanned_calls:
        scanned_calls = _scan_bare_tool_lines(text)
        if scanned_calls:
            scanned_mode = "bare_line"

    if scanned_calls and not calls:
        for name, body in scanned_calls:
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
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for call in calls:
            fn = call.get("function", {})
            key = (fn.get("name", ""), fn.get("arguments", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(call)
                if len(deduped) >= MAX_TOOL_CALLS:
                    break
        if scanned_mode == "none":
            if _scan_gemma_calls(text):
                scanned_mode = "gemma_native"
            elif _scan_fallback_calls(text):
                scanned_mode = "fallback"
            else:
                scanned_mode = "bare_line"
        print(
            f"[llamasv] extract_tool_calls matched={scanned_mode} count={len(deduped)}"
        )
        return deduped

    print(f"[llamasv] extract_tool_calls matched=none sample={text[:300]!r}")
    return None


def _earliest_tag(text: str, tags: list[str]) -> tuple[int, str] | None:
    found = [(idx, tag) for tag in tags if (idx := text.find(tag)) >= 0]
    return min(found, key=lambda item: item[0]) if found else None


def tool_streamable_thought_delta(text: str, emitted: int) -> tuple[str, int]:
    stripped = text.lstrip()
    if stripped.startswith(
        (
            "<|tool_call>",
            "<|tool_call|>",
            "<tool_call|>",
            "<tool_call>",
            "<function=",
            "call:",
        )
    ):
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


def extract_reasoning(text: str) -> str | None:
    markers = ["<tool_call>", "<|tool_call>", "<|tool_call|>"]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    reasoning = text[:cut].strip()
    return reasoning if reasoning else None
