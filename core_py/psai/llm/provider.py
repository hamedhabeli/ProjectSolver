from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenAICompatibleProvider(LLMProvider):
    """
    Minimal OpenAI-compatible chat.completions client using urllib.request.

    Works with providers that expose OpenAI-compatible endpoints.
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_s: int = 30
    max_retries: int = 2

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass api_key=...")

        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return _extract_openai_text(raw)
            except urllib.error.HTTPError as e:
                body = None
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    body = None
                last_err = RuntimeError(f"HTTPError {e.code}: {e.reason}; body={body}")
            except Exception as e:  # noqa: BLE001
                last_err = e

            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))

        assert last_err is not None
        raise last_err


@dataclass(frozen=True)
class GeminiModelInfo:
    name: str
    display_name: str
    description: str
    supported_generation_methods: list[str]


@dataclass(frozen=True)
class GeminiProvider(LLMProvider):
    """
    Gemini REST client using urllib.request only.
    """

    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    timeout_s: int = 30
    max_retries: int = 2
    temperature: float = 0.2
    top_p: float = 0.95
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or pass api_key=..."
            )

        if not str(self.model or "").strip():
            raise RuntimeError("Missing Gemini model name")

        url = (
            self.base_url.rstrip("/")
            + "/models/"
            + urllib.parse.quote(str(self.model), safe="")
            + ":generateContent?key="
            + urllib.parse.quote_plus(key)
        )
        payload: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
            },
        }
        return _request_json_text(url, payload, timeout_s=self.timeout_s, max_retries=self.max_retries)


def list_gemini_models(
    api_key: str,
    *,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    timeout_s: int = 30,
    max_retries: int = 2,
) -> list[GeminiModelInfo]:
    if not api_key:
        raise RuntimeError("Missing Gemini API key")

    url = base_url.rstrip("/") + "/models?key=" + urllib.parse.quote_plus(api_key)
    payload = _request_json(url, timeout_s=timeout_s, max_retries=max_retries)
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Unexpected Gemini models response shape")

    out: list[GeminiModelInfo] = []
    for item in models:
        if not isinstance(item, dict):
            continue

        methods = item.get("supportedGenerationMethods", [])
        if not isinstance(methods, list):
            methods = []

        methods_str = [str(x) for x in methods]
        if "generateContent" not in methods_str:
            continue

        out.append(
            GeminiModelInfo(
                name=str(item.get("name", "")),
                display_name=str(item.get("displayName", item.get("name", ""))),
                description=str(item.get("description", "")),
                supported_generation_methods=methods_str,
            )
        )

    out.sort(key=lambda m: (m.display_name.lower(), m.name.lower()))
    return out


def _request_json(
    url: str,
    *,
    timeout_s: int,
    max_retries: int,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers=headers,
                method="POST" if body is not None else "GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RuntimeError("Expected JSON object response")
            return parsed
        except urllib.error.HTTPError as e:
            body_text = None
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                body_text = None
            last_err = RuntimeError(f"HTTPError {e.code}: {e.reason}; body={body_text}")
        except Exception as e:  # noqa: BLE001
            last_err = e

        if attempt < max_retries:
            time.sleep(0.5 * (attempt + 1))

    assert last_err is not None
    raise last_err


def _request_json_text(
    url: str,
    payload: dict[str, object],
    *,
    timeout_s: int,
    max_retries: int,
) -> str:
    response = _request_json(url, timeout_s=timeout_s, max_retries=max_retries, payload=payload)
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini response missing candidates")

    first = candidates[0]
    if not isinstance(first, dict):
        raise RuntimeError("Invalid Gemini candidate")

    content = first.get("content", {})
    if not isinstance(content, dict):
        raise RuntimeError("Invalid Gemini candidate content")

    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise RuntimeError("Invalid Gemini candidate parts")

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)

    out = "".join(texts).strip()
    if not out:
        raise RuntimeError("Gemini response contained no text")
    return out


def _extract_openai_text(raw: str) -> str:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Expected JSON object response")

    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI-compatible response missing choices")

    choice0 = choices[0]
    if not isinstance(choice0, dict):
        raise RuntimeError("Invalid choice")

    message = choice0.get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("Invalid choice.message")

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        out = "".join(texts).strip()
        if out:
            return out

    raise RuntimeError("OpenAI-compatible response missing text")
