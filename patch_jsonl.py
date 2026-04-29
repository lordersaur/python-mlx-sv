"""
patch_jsonl.py — replaces fake tool responses in gemma4_symbol_search_read_sft_100.jsonl
with realistic output derived from the actual source files.

For every conversation:
  - search_code_tool response  → real ripgrep output (-n -C 3 --smart-case) + note
  - read_file_tool response     → real file content at the requested line range

Run from ~/python-mlx-sv/:
    python patch_jsonl.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

INPUT_FILE  = Path("gemma4_symbol_search_read_sft_100.jsonl")
OUTPUT_FILE = Path("gemma4_symbol_search_read_sft_100_patched.jsonl")

# Root of the Rust source tree (adjust if your layout differs)
REPO_ROOT = Path("../mlx-acp-agent").resolve()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_CALL_RE  = re.compile(r"<\|tool_call\|?>call:(\w+)\{(.*?)\}<tool_call\|?>", re.DOTALL)
TOOL_RESP_RE  = re.compile(r"<\|tool_response\|?>response:(\w+)\{<\|\"?\|?>(.*?)<\|\"?\|?>\}<tool_response\|?>", re.DOTALL)
PARAM_RE      = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL)


def parse_tool_call(content: str):
    """Return (tool_name, {param: value}) or None."""
    m = TOOL_CALL_RE.search(content)
    if not m:
        return None
    name   = m.group(1)
    params = dict(PARAM_RE.findall(m.group(2)))
    return name, params


def make_search_response(path: str, query: str) -> str | None:
    """Run ripgrep against the real file and return the output + factual note."""
    full_path = REPO_ROOT / path
    if not full_path.exists():
        return None

    result = subprocess.run(
        ["rg", "-n", "-C", "3", "--smart-case", "--max-count", "80", query, str(full_path)],
        capture_output=True, text=True
    )
    stdout = result.stdout
    if not stdout.strip():
        return None

    # Append the same factual note the Rust tool appends
    note = _build_note(stdout)
    return stdout.rstrip() + note


def _build_note(stdout: str) -> str:
    groups = stdout.split("\n--\n")
    last_group = groups[-1] if groups else stdout
    match_lines  = []
    all_lines    = []
    for line in last_group.splitlines():
        m = re.match(r"^(\d+)([:–-])", line)
        if m:
            n = int(m.group(1))
            all_lines.append(n)
            if m.group(2) == ":":
                match_lines.append(n)
    if not all_lines:
        return ""
    group_start = all_lines[0]
    last_line   = all_lines[-1]
    return f"\n[... last match group shows lines {group_start}–{last_line}; lines after {last_line} are not shown]"


def make_read_response(path: str, start_line: int, line_count: int) -> str | None:
    """Read the real file and return the requested line range in display format."""
    full_path = REPO_ROOT / path
    if not full_path.exists():
        return None
    try:
        lines = full_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    # 1-based; clamp to file length
    start  = max(1, start_line)
    end    = min(len(lines), start + line_count - 1)
    chunk  = lines[start - 1 : end]
    width  = len(str(end))
    formatted = "\n".join(f"{start + i:{width}} | {line}" for i, line in enumerate(chunk))
    total = len(lines)
    if end >= total:
        formatted += "\n[file complete]"
    else:
        formatted += f"\n[Continue at line {end + 1} — {total - end} lines remaining]"
    return formatted


def wrap_search_response(content: str) -> str:
    return f'<|tool_response>response:search_code_tool{{<|"|>{content}<|"|>}}<tool_response|>'


def wrap_read_response(content: str) -> str:
    return f'<|tool_response>response:read_file_tool{{<|"|>{content}<|"|>}}<tool_response|>'


# ---------------------------------------------------------------------------
# Main patch loop
# ---------------------------------------------------------------------------

def patch_conversation(turns: list[dict]) -> list[dict]:
    """
    Walk turns and replace fake tool responses with real content.
    We track what the previous tool call was so we know what to substitute.
    """
    patched   = []
    last_call = None   # (tool_name, params)

    for turn in turns:
        role    = turn["role"]
        content = turn["content"]

        if role == "assistant":
            # Is this turn a tool call?
            parsed = parse_tool_call(content)
            if parsed:
                last_call = parsed
                patched.append(turn)
                continue

            # Is this turn a tool response?
            resp_m = TOOL_RESP_RE.search(content)
            if resp_m and last_call:
                tool_name_resp = resp_m.group(1)
                name, params   = last_call

                if tool_name_resp == "search_code_tool" and name == "search_code_tool":
                    path  = params.get("path", "")
                    query = params.get("query", "")
                    real  = make_search_response(path, query)
                    if real:
                        patched.append({**turn, "content": wrap_search_response(real)})
                        continue

                elif tool_name_resp == "read_file_tool" and name == "read_file_tool":
                    path       = params.get("path", "")
                    try:
                        start  = int(params.get("start_line", "1"))
                        count  = int(params.get("line_count", "80"))
                    except ValueError:
                        start, count = 1, 80
                    real = make_read_response(path, start, count)
                    if real:
                        patched.append({**turn, "content": wrap_read_response(real)})
                        continue

        patched.append(turn)

    return patched


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found in {Path.cwd()}")
        sys.exit(1)

    # Verify rg is available
    if subprocess.run(["which", "rg"], capture_output=True).returncode != 0:
        print("Error: ripgrep (rg) not found. Install with: brew install ripgrep")
        sys.exit(1)

    ok = skipped = 0
    with INPUT_FILE.open() as fin, OUTPUT_FILE.open("w") as fout:
        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj   = json.loads(raw)
                turns = obj.get("conversations", [])
                obj["conversations"] = patch_conversation(turns)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                print(f"Line {lineno}: skipped — {e}")
                fout.write(raw + "\n")
                skipped += 1

    print(f"Done. Patched {ok} examples, skipped {skipped}.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
