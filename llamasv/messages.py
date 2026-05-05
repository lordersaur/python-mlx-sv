import json
import re
from typing import Any

from .schemas import ChatMessage


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


def strip_reasoning_blocks(text: str) -> str:
    text = re.sub(r"(?s)<\|channel>thought\s*.*?<channel\|>", "", text)
    text = re.sub(r"(?s)<\|think\|>.*?<\|/think\|>", "", text)
    text = re.sub(r"(?s)<think>\s*.*?</think>", "", text)
    text = re.sub(r"(?s)<\|channel>thought\s*.*$", "", text)
    text = re.sub(r"(?s)<\|think\|>.*$", "", text)
    text = re.sub(r"(?s)<think>\s*.*$", "", text)
    for token in (
        "<|channel>",
        "<channel|>",
        "<|think|>",
        "<|/think|>",
        "<think>",
        "</think>",
    ):
        text = text.replace(token, "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_tool_call_markup(text: str) -> str:
    text = re.sub(r"(?s)<tool_call>\s*.*?</tool_call>", "", text)
    text = re.sub(
        r"(?s)<\|tool_call\|?>call:.*?<tool_call\|>",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^\s*<\|tool_call\|?>call:.*$",
        "",
        text,
    )
    text = re.sub(
        r"(?s)^\s*\[[^\n]*uses?[^\n]*tool[^\n]*\]\s*$",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^\s*[A-Za-z_][A-Za-z0-9_]*\{.*?<tool_call\|>\s*$",
        "",
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def public_response_text(text: str, *, hide_tool_calls: bool = False) -> str:
    text = strip_reasoning_blocks(text)
    if hide_tool_calls:
        text = strip_tool_call_markup(text)
    return text


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


def _assistant_has_tool_call_id(msg: dict, tool_call_id: str) -> bool:
    for call in msg.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id") == tool_call_id:
            return True
    return False


def _attach_gemma_tool_response(out: list[dict], tool_message: dict) -> None:
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


def message_to_dict(
    m: ChatMessage,
    tool_names: dict[str, str] | None = None,
    *,
    model_family: str = "generic",
) -> dict:
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
    if m.role == "tool":
        raw = normalize_content(m.content)
        name = (
            m.name
            or (tool_names or {}).get(m.tool_call_id or "")
            or (m.tool_call_id or "tool")
        )
        if model_family != "gemma":
            return {
                "role": "tool",
                "content": raw,
                "tool_call_id": m.tool_call_id or "0",
                "name": name,
            }
        return {
            "role": "tool",
            "content": "",
            "tool_call_id": m.tool_call_id or "0",
            "tool_responses": [{"name": name, "response": _tool_response_payload(raw)}],
        }
    if m.role == "assistant":
        return {
            "role": "assistant",
            "content": strip_reasoning_blocks(normalize_content(m.content)),
        }
    return {"role": m.role, "content": normalize_content(m.content)}


def messages_to_dicts(
    messages: list[ChatMessage], *, model_family: str = "generic"
) -> list[dict]:
    out: list[dict] = []
    tool_names: dict[str, str] = {}
    for message in messages:
        converted = message_to_dict(
            message,
            tool_names,
            model_family=model_family,
        )
        if converted.get("role") == "tool" and model_family == "gemma":
            _attach_gemma_tool_response(out, converted)
        else:
            out.append(converted)
        if message.role == "assistant" and message.tool_calls:
            tool_names.update(_tool_call_name_by_id(converted.get("tool_calls")))
    return out
