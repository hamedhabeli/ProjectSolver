from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Protocol


class SupportsChat(Protocol):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a dict from model output that may include fences or prose."""

    candidates = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    fenced = _strip_outer_code_fence(stripped)
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                return parsed

        extracted = _extract_balanced_json_object(candidate)
        if extracted is None:
            continue
        try:
            parsed = json.loads(extracted)
        except JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JSONDecodeError("Could not parse a JSON object from model output", text, 0)


def llm_json_with_retry(
    llm: SupportsChat,
    system_prompt: str,
    user_prompt: str,
    *,
    max_json_retries: int = 2,
) -> dict[str, Any]:
    attempt_user_prompt = user_prompt
    last_error: str | None = None
    last_text: str | None = None

    for attempt in range(max_json_retries + 1):
        llm_text = llm.chat(system_prompt=system_prompt, user_prompt=attempt_user_prompt)
        last_text = llm_text
        try:
            return parse_json_object(llm_text)
        except JSONDecodeError as exc:
            last_error = str(exc)
            if attempt >= max_json_retries:
                preview = _summarize_text(llm_text)
                raise RuntimeError(
                    "LLM returned non-JSON output after retries. "
                    f"Parser error: {last_error}; preview={preview}"
                ) from exc

            attempt_user_prompt = (
                user_prompt
                + "\n\n"
                + "Your previous answer was invalid JSON and could not be parsed.\n"
                + f"Parser error: {last_error}\n"
                + "Return ONLY valid JSON, with no markdown, no code fences, no commentary."
            )

    preview = _summarize_text(last_text or "")
    raise RuntimeError(
        "Unreachable: JSON retry loop exhausted. "
        f"Last parser error: {last_error}; preview={preview}"
    )


def _strip_outer_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) < 2:
        return text

    if lines[-1].strip() == "```":
        inner = "\n".join(lines[1:-1]).strip()
    else:
        inner = "\n".join(lines[1:]).strip()

    return inner or text


def _extract_balanced_json_object(text: str) -> str | None:
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : idx + 1]

    return None


def _summarize_text(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
