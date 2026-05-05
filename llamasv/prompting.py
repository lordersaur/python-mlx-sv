from typing import Any, Optional

from .messages import messages_to_dicts, normalize_content
from .runtime import AppRuntime, stop_tokens_for_model_family
from .schemas import ChatCompletionRequest

TITLE_GENERATOR_SYSTEM_MARKERS = (
    "you are a title generator",
    "you output only a thread title",
)
DEFAULT_TEMPERATURE_BY_FAMILY = {
    "qwen": 0.7,
    "gemma": 1.0,
    "generic": 1.0,
}


def extra_body(req: ChatCompletionRequest) -> dict[str, Any]:
    return req.extra_body if isinstance(req.extra_body, dict) else {}


def _chat_template_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    value = extra_body(req).get("chat_template_kwargs")
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


def is_title_generator_request(req: ChatCompletionRequest) -> bool:
    if not req.messages:
        return False
    first = req.messages[0]
    if first.role != "system":
        return False
    content = normalize_content(first.content).strip().lower()
    return all(marker in content for marker in TITLE_GENERATOR_SYSTEM_MARKERS)


def requested_enable_thinking(req: ChatCompletionRequest) -> bool:
    template_kwargs = _chat_template_kwargs(req)
    explicit = _coerce_bool(template_kwargs.get("enable_thinking"))
    if explicit is not None:
        return explicit
    explicit = _coerce_bool(extra_body(req).get("enable_thinking"))
    if explicit is not None:
        return explicit
    if is_title_generator_request(req):
        return False
    if req.tools:
        return False
    return True


def requested_expose_thoughts(req: ChatCompletionRequest) -> bool:
    explicit = _coerce_bool(extra_body(req).get("expose_thoughts"))
    return explicit is True


def _msg_chars(msg: dict) -> int:
    total = len(str(msg.get("content") or ""))
    for tc in msg.get("tool_calls") or []:
        total += len(str(tc))
    for tr in msg.get("tool_responses") or []:
        total += len(str(tr))
    return total


def _is_tool_response(msg: dict) -> bool:
    return msg.get("role") == "tool"


def _trim_anchor(anchor: list[dict], max_anchor_chars: int) -> list[dict]:
    if len(anchor) <= 1:
        return anchor
    user_msg = anchor[:1]
    tool_msgs = anchor[1:]
    budget = max_anchor_chars - _msg_chars(user_msg[0])
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


def truncate_messages(messages: list[dict], n_ctx: int) -> list[dict]:
    max_context_chars = (n_ctx - n_ctx // 4) * 4
    max_anchor_chars = max_context_chars * 3 // 4

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

    anchor = _trim_anchor(anchor, max_anchor_chars)
    budget = max_context_chars - sum(_msg_chars(m) for m in system + anchor)

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
            f"[llamasv] context_truncated dropped_turns={dropped} kept_turns={len(kept)}"
        )

    return system + kept + anchor


def _drop_oldest_non_system_message(messages: list[dict]) -> list[dict]:
    for i, msg in enumerate(messages):
        if msg.get("role") != "system":
            return messages[:i] + messages[i + 1 :]
    return messages


def _consolidate_system_messages(messages: list[dict]) -> list[dict]:
    system_parts = [
        normalize_content(msg.get("content")).strip()
        for msg in messages
        if msg.get("role") == "system" and normalize_content(msg.get("content")).strip()
    ]
    non_system = [msg for msg in messages if msg.get("role") != "system"]
    if not system_parts:
        return non_system
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


def _apply_gemma_thinking_marker(
    messages: list[dict], enable_thinking: bool
) -> list[dict]:
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


def make_gen_kwargs(req: ChatCompletionRequest, runtime: AppRuntime) -> dict[str, Any]:
    extra = extra_body(req)
    stop_tokens = list(stop_tokens_for_model_family(runtime.settings.model_family))
    if req.tools and runtime.settings.model_family == "gemma":
        stop_tokens.append("<tool_call|>")
    kwargs: dict[str, Any] = {
        "temperature": (
            req.temperature
            if req.temperature is not None
            else DEFAULT_TEMPERATURE_BY_FAMILY.get(runtime.settings.model_family, 1.0)
        ),
        "top_p": req.top_p if req.top_p is not None else 0.95,
        "top_k": int(extra.get("top_k", 64)),
        "stop": stop_tokens,
    }
    min_p = extra.get("min_p")
    if min_p is not None:
        kwargs["min_p"] = float(min_p)
    repeat_penalty = extra.get("repetition_penalty")
    if repeat_penalty is not None:
        kwargs["repeat_penalty"] = float(repeat_penalty)
    if req.presence_penalty is not None:
        kwargs["presence_penalty"] = float(req.presence_penalty)
    return kwargs


def _render_prompt(
    messages: list[dict], req: ChatCompletionRequest, runtime: AppRuntime
) -> tuple[str, bool]:
    enable_thinking = requested_enable_thinking(req)
    messages = _consolidate_system_messages(messages)
    if runtime.settings.model_family == "gemma":
        messages = _apply_gemma_thinking_marker(messages, enable_thinking)

    roles = "→".join(m["role"] for m in messages)
    print(
        f"[llamasv] chat_completions tools_in_request={len(req.tools or [])} "
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
        prompt = runtime.tokenizer.apply_chat_template(messages, **template_kwargs)
    except Exception as exc:
        dropped = template_kwargs.pop("enable_thinking", None)
        if dropped is not None:
            print(f"[llamasv] template_warning dropped=enable_thinking reason={exc!r}")
        try:
            prompt = runtime.tokenizer.apply_chat_template(messages, **template_kwargs)
        except Exception as exc2:
            template_kwargs.pop("tools", None)
            print(f"[llamasv] template_warning dropped=tools reason={exc2!r}")
            prompt = runtime.tokenizer.apply_chat_template(messages, **template_kwargs)

    if isinstance(prompt, str) and prompt.startswith("<bos>"):
        prompt = prompt[len("<bos>") :].lstrip()

    return prompt, enable_thinking


def _prompt_token_count(prompt: str, runtime: AppRuntime) -> int:
    encoded = runtime.tokenizer(
        prompt,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded.get("input_ids", [])
    return len(input_ids)


def _prompt_token_budget(req: ChatCompletionRequest, n_ctx: int) -> int:
    if is_title_generator_request(req):
        reserve = 64
    elif req.tools:
        reserve = 2048
    else:
        reserve = 1024
    return max(512, n_ctx - min(reserve, n_ctx // 2))


def build_prompt(req: ChatCompletionRequest, runtime: AppRuntime) -> tuple[str, bool]:
    original_messages = messages_to_dicts(
        req.messages,
        model_family=runtime.settings.model_family,
    )
    working_messages = truncate_messages(original_messages, runtime.settings.n_ctx)
    prompt_budget = _prompt_token_budget(req, runtime.settings.n_ctx)

    prompt, enable_thinking = _render_prompt(working_messages, req, runtime)
    prompt_tokens = _prompt_token_count(prompt, runtime)

    while prompt_tokens > prompt_budget and len(working_messages) > 2:
        prev_len = len(working_messages)
        working_messages = _drop_oldest_non_system_message(working_messages)
        if len(working_messages) == prev_len:
            break
        prompt, enable_thinking = _render_prompt(working_messages, req, runtime)
        prompt_tokens = _prompt_token_count(prompt, runtime)

    if prompt_tokens > prompt_budget:
        print(
            f"[llamasv] context_truncated_by_tokens prompt_tokens={prompt_tokens} "
            f"budget={prompt_budget} messages={len(working_messages)}",
            flush=True,
        )
    else:
        print(
            f"[llamasv] prompt_tokens={prompt_tokens} budget={prompt_budget} "
            f"messages={len(working_messages)}",
            flush=True,
        )

    return prompt, enable_thinking
