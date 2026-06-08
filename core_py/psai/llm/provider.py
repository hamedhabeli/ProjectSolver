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


@dataclass
class OpenAICompatibleProvider(LLMProvider):
    """
    Minimal OpenAI-compatible chat.completions client using urllib.request.

    Works with providers that expose OpenAI-compatible endpoints, e.g.:
      - OpenRouter
      - Groq
      - local OpenAI-compatible proxies
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


@dataclass
class GeminiProvider(LLMProvider):
    """
    Gemini REST client using urllib.request only.

    Uses:
      POST https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent?key=...
    Request body:
      {
        "systemInstruction": {"parts": [{"text": "..."}]},
        "contents": [{"role": "user", "parts": [{"text": "..."}]}]
      }
    Response parsing:
      candidates[0].content.parts[*].text
    """

    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    timeout_s: int = 30
    max_retries: int = 2
    temperature: float = 0.2
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        key = (
            self.api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        quoted_key = urllib.parse.quote_plus(key)
        url = (
            f"{self.base_url.rstrip('/')}/models/"
            f"{urllib.parse.quote(self.model, safe='')}:generateContent?key={quoted_key}"
        )

        payload: dict[str, object] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }

        if system_prompt.strip():
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}],
            }

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return _extract_gemini_text(raw)
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


def _extract_openai_text(raw_json: str) -> str:
    obj = json.loads(raw_json)
    choices = obj.get("choices")
    if not choices:
        raise RuntimeError(f"Unexpected response (no choices): {obj}")
    msg = choices[0].get("message", {})
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected response (no content): {obj}")
    return content


def _extract_gemini_text(raw_json: str) -> str:
    obj = json.loads(raw_json)

    if "error" in obj:
        raise RuntimeError(f"Gemini API error: {obj['error']}")

    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Unexpected Gemini response (no candidates): {obj}")

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise RuntimeError(f"Unexpected Gemini response shape: {obj}")

    content = candidate.get("content", {})
    if not isinstance(content, dict):
        raise RuntimeError(f"Unexpected Gemini response (no content): {obj}")

    parts = content.get("parts", [])
    if not isinstance(parts, list):
        raise RuntimeError(f"Unexpected Gemini response (no parts): {obj}")

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)

    if texts:
        return "".join(texts)

    raise RuntimeError(f"Unexpected Gemini response (no text parts): {obj}")
