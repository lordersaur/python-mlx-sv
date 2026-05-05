import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .messages import public_response_text
from .prompting import (
    build_prompt,
    extra_body,
    is_title_generator_request,
    make_gen_kwargs,
    requested_expose_thoughts,
)
from .runtime import append_raw_output_log, create_runtime, get_runtime
from .schemas import ChatCompletionRequest
from .tooling import (
    extract_reasoning,
    extract_tool_calls,
    has_tool_call_cue,
    tool_streamable_thought_delta,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = create_runtime()
    try:
        yield
    finally:
        app.state.runtime = None


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get("/v1/models")
    def list_models(request: Request):
        runtime = get_runtime(request.app)
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.settings.public_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, request: Request):
        runtime = get_runtime(request.app)
        print(
            "[llamasv] request_summary "
            f"stream={bool(req.stream)} tools={len(req.tools or [])} "
            f"tool_choice={req.tool_choice!r} "
            f"messages={_summarize_messages(req.messages)} "
            f"extra_body_keys={sorted((req.extra_body or {}).keys())}",
            flush=True,
        )
        if is_title_generator_request(req):
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            title = _make_thread_title(req.messages)
            print(f"[llamasv] title_shortcut title={title!r}", flush=True)
            if req.stream:

                async def title_stream():
                    yield f"data: {json.dumps(_chunk(runtime, completion_id, '', role='assistant'))}\n\n"
                    yield f"data: {json.dumps(_chunk(runtime, completion_id, title))}\n\n"
                    yield (
                        "data: "
                        f"{json.dumps(_final_chunk(runtime, completion_id, {}, 'stop'))}\n\n"
                    )
                    yield "data: [DONE]\n\n"

                return StreamingResponse(title_stream(), media_type="text/event-stream")

            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": runtime.settings.public_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": title},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        prompt, _enable_thinking = build_prompt(req, runtime)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        gen_kwargs = make_gen_kwargs(req)
        session_id = str(extra_body(req).get("session_id") or "").strip() or None
        expose_thoughts = requested_expose_thoughts(req)
        max_tokens = req.max_tokens or 32768
        if req.tools and max_tokens == 32768:
            max_tokens = 2048

        if req.stream:

            async def event_stream():
                accumulated = ""
                visible_emitted = 0
                detected_tool_calls: list[dict] | None = None
                with runtime.inference_lock:
                    yield f"data: {json.dumps(_chunk(runtime, completion_id, "", role='assistant'))}\n\n"
                    for chunk in runtime.llm(
                        prompt,
                        max_tokens=max_tokens,
                        stream=True,
                        **gen_kwargs,
                    ):
                        if await request.is_disconnected():
                            print("[llamasv] stream_cancelled client_disconnected=True")
                            return
                        token = chunk["choices"][0]["text"]
                        accumulated += token
                        if req.tools:
                            if has_tool_call_cue(accumulated):
                                detected_tool_calls = extract_tool_calls(accumulated)
                                if detected_tool_calls:
                                    break
                        if expose_thoughts:
                            if req.tools:
                                delta, visible_emitted = tool_streamable_thought_delta(
                                    accumulated, visible_emitted
                                )
                                if delta:
                                    yield (
                                        f"data: {json.dumps(_chunk(runtime, completion_id, delta))}\n\n"
                                    )
                                continue
                            yield f"data: {json.dumps(_chunk(runtime, completion_id, token))}\n\n"
                            continue

                        if not req.tools:
                            visible_text = public_response_text(accumulated)
                            if len(visible_text) > visible_emitted:
                                delta = visible_text[visible_emitted:]
                                visible_emitted = len(visible_text)
                                if delta:
                                    yield f"data: {json.dumps(_chunk(runtime, completion_id, delta))}\n\n"

                if await request.is_disconnected():
                    print("[llamasv] stream_cancelled after_generation=True")
                    return

                text = accumulated.strip()
                print(f"[llamasv] raw_output={text[:2000]!r}")
                tool_calls = detected_tool_calls if req.tools else None
                if req.tools and tool_calls is None:
                    tool_calls = extract_tool_calls(text)
                finish_reason = "tool_calls" if tool_calls else "stop"
                append_raw_output_log(
                    runtime,
                    stream=True,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    raw_text=text,
                    session_id=session_id,
                    completion_id=completion_id,
                )
                print(
                    f"[llamasv] finish_reason={finish_reason} "
                    f"tool_call_count={len(tool_calls or [])}"
                )

                final_delta: dict[str, Any] = {}
                if tool_calls:
                    final_delta["tool_calls"] = tool_calls
                elif req.tools and text:
                    if expose_thoughts:
                        final_content = text
                        if final_content:
                            final_delta["content"] = final_content
                    else:
                        final_content = public_response_text(
                            text,
                            hide_tool_calls=bool(req.tools),
                        )
                        if len(final_content) > visible_emitted:
                            final_delta["content"] = final_content[visible_emitted:]
                yield (
                    "data: "
                    f"{json.dumps(_final_chunk(runtime, completion_id, final_delta, finish_reason))}\n\n"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        with runtime.inference_lock:
            output = runtime.llm(
                prompt,
                max_tokens=max_tokens,
                **gen_kwargs,
            )

        raw = output["choices"][0]["text"]
        text = raw.strip()
        print(f"[llamasv] raw_output={text[:2000]!r}")
        tool_calls = extract_tool_calls(text) if req.tools else None
        finish_reason = "tool_calls" if tool_calls else "stop"
        append_raw_output_log(
            runtime,
            stream=False,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_text=text,
            session_id=session_id,
            completion_id=completion_id,
        )

        if expose_thoughts:
            reasoning = extract_reasoning(text) if tool_calls else text
        else:
            reasoning = public_response_text(
                extract_reasoning(text) if tool_calls else text,
                hide_tool_calls=bool(req.tools),
            )

        print(
            f"[llamasv] finish_reason={finish_reason} tool_call_count={len(tool_calls or [])}"
        )
        response_message: dict[str, Any] = {"role": "assistant", "content": reasoning}
        if tool_calls:
            response_message["tool_calls"] = tool_calls

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": runtime.settings.public_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": response_message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )

    return app


def _chunk(runtime, completion_id: str, content: str, role: str | None = None) -> dict:
    delta: dict[str, Any] = {"content": content}
    if role is not None:
        delta["role"] = role
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": runtime.settings.public_model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _final_chunk(runtime, completion_id: str, delta: dict, finish_reason: str) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": runtime.settings.public_model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _summarize_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = json.dumps(content, ensure_ascii=True)
        elif content is None:
            text = ""
        else:
            text = str(content)
        text = " ".join(text.split())
        preview = text[:120]
        if len(text) > 120:
            preview += "..."
        parts.append(f"{idx}:{role}[{len(text)}]={preview!r}")
    return " | ".join(parts)


def _make_thread_title(messages: list[Any]) -> str:
    user_contents: list[str] = []
    for msg in messages:
        if getattr(msg, "role", None) != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            text = str(content)
        text = " ".join(text.split()).strip()
        if text:
            user_contents.append(text)
    if not user_contents:
        return "New conversation"
    seed = user_contents[-1]
    if seed.lower().startswith("generate a title for this conversation"):
        seed = user_contents[-2] if len(user_contents) >= 2 else "New conversation"
    words = seed.split()
    title = " ".join(words[:8]).strip(" -:,.")
    if not title:
        return "New conversation"
    return title[:80]


app = create_app()
