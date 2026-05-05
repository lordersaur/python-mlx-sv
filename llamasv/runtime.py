import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from llama_cpp import Llama
from transformers import AutoTokenizer

COMMON_STOP_TOKENS = ("<|im_end|>",)
FAMILY_STOP_TOKENS = {
    "gemma": ("<end_of_turn>", "<|end_of_turn|>", "<turn|>", "<|tool_response>"),
    "qwen": ("<tool_response>",),
    "generic": (),
}
DEFAULT_GGUF_PATH = os.path.expanduser(
    "~/Downloads/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf"
)
DEFAULT_HF_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_CTX = 32768


@dataclass(frozen=True)
class Settings:
    gguf_path: str
    hf_model_id: str
    model_family: str
    n_gpu_layers: int
    n_ctx: int
    llama_verbose: bool
    public_model_name: str
    raw_output_log_path: str


@dataclass
class AppRuntime:
    settings: Settings
    llm: Llama
    tokenizer: AutoTokenizer
    inference_lock: threading.Lock
    raw_output_log_lock: threading.Lock


def infer_model_family(hf_model_id: str) -> str:
    lowered = hf_model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return "generic"


def stop_tokens_for_model_family(model_family: str) -> tuple[str, ...]:
    return COMMON_STOP_TOKENS + FAMILY_STOP_TOKENS.get(model_family, ())


def load_settings() -> Settings:
    gguf_path = (os.environ.get("GGUF_MODEL_PATH") or "").strip() or DEFAULT_GGUF_PATH
    hf_model_id = (os.environ.get("HF_MODEL_ID") or "").strip() or DEFAULT_HF_MODEL
    model_family = infer_model_family(hf_model_id)
    n_gpu_layers = int(os.environ.get("N_GPU_LAYERS", str(DEFAULT_N_GPU_LAYERS)))
    n_ctx = int(os.environ.get("N_CTX", str(DEFAULT_N_CTX)))
    llama_verbose = (os.environ.get("LLAMA_VERBOSE") or "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    public_model_name = (
        os.environ.get("PUBLIC_MODEL_NAME") or ""
    ).strip() or os.path.basename(gguf_path)
    raw_output_log_path = (
        os.environ.get("RAW_OUTPUT_LOG_PATH") or ""
    ).strip() or os.path.join(os.getcwd(), "llamasv-raw-outputs.txt")
    return Settings(
        gguf_path=gguf_path,
        hf_model_id=hf_model_id,
        model_family=model_family,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        llama_verbose=llama_verbose,
        public_model_name=public_model_name,
        raw_output_log_path=raw_output_log_path,
    )


def _register_stop_tokens(
    tokenizer: AutoTokenizer, stop_tokens: tuple[str, ...]
) -> None:
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if eos_ids is None:
        eos_ids = []
    if not hasattr(eos_ids, "add") and not hasattr(eos_ids, "append"):
        eos_ids = list(eos_ids)

    for token in stop_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_ids:
            if hasattr(eos_ids, "add"):
                eos_ids.add(token_id)
            else:
                eos_ids.append(token_id)


def create_runtime() -> AppRuntime:
    settings = load_settings()

    print(f"[llamasv] Loading GGUF from: {settings.gguf_path}", flush=True)
    print(f"[llamasv] Loading tokenizer from: {settings.hf_model_id}", flush=True)
    print(
        f"[llamasv] n_gpu_layers={settings.n_gpu_layers} "
        f"n_ctx={settings.n_ctx} verbose={settings.llama_verbose} "
        f"model_family={settings.model_family}",
        flush=True,
    )
    print(
        f"[llamasv] stop_tokens={list(stop_tokens_for_model_family(settings.model_family))}",
        flush=True,
    )
    print(
        "[llamasv] Expect CUDA offload only if llama-cpp-python was built with "
        "GGML_CUDA=on; if GPU stays idle, check the startup logs for layer assignment.",
        flush=True,
    )

    llm = Llama(
        model_path=settings.gguf_path,
        n_gpu_layers=settings.n_gpu_layers,
        n_ctx=settings.n_ctx,
        verbose=settings.llama_verbose,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_id)
    _register_stop_tokens(
        tokenizer,
        stop_tokens_for_model_family(settings.model_family),
    )

    return AppRuntime(
        settings=settings,
        llm=llm,
        tokenizer=tokenizer,
        inference_lock=threading.Lock(),
        raw_output_log_lock=threading.Lock(),
    )


def raw_output_log_target(runtime: AppRuntime, session_id: str | None) -> Path:
    base_path = Path(runtime.settings.raw_output_log_path)
    if not session_id:
        return base_path
    safe_session_id = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id).strip("._-")
    if not safe_session_id:
        safe_session_id = "session"
    session_dir = base_path.parent / f"{base_path.stem}.sessions"
    return session_dir / f"{safe_session_id}.txt"


def append_raw_output_log(
    runtime: AppRuntime,
    *,
    stream: bool,
    finish_reason: str,
    tool_calls: list[dict] | None,
    raw_text: str,
    session_id: str | None = None,
    completion_id: str | None = None,
) -> None:
    try:
        target_path = raw_output_log_target(runtime, session_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        entry = (
            f"=== {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime())} "
            f"stream={stream} finish_reason={finish_reason} "
            f"tool_call_count={len(tool_calls or [])} "
            f"session_id={session_id or '-'} "
            f"completion_id={completion_id or '-'} ===\n"
            f"{raw_text}\n\n"
        )
        with runtime.raw_output_log_lock:
            with open(target_path, "a", encoding="utf-8") as handle:
                handle.write(entry)
    except Exception as exc:
        print(
            "[llamasv] raw_output_log_error "
            f"path={runtime.settings.raw_output_log_path!r} error={exc!r}",
            flush=True,
        )


def get_runtime(app: FastAPI) -> AppRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("App runtime is not initialized")
    return cast(AppRuntime, runtime)
