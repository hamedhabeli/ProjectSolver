from __future__ import annotations

import json

import pytest

from psai.llm import provider as provider_module
from psai.llm.provider import GeminiProvider, list_gemini_models


class FakeResponse:
    def __init__(self, body: str, code: int = 200):
        self._body = body.encode("utf-8")
        self.code = code

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_list_gemini_models_filters_non_generatecontent(monkeypatch):
    payload = {
        "models": [
            {
                "name": "models/gemini-2.0-flash",
                "displayName": "Gemini Flash",
                "description": "fast",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "models/embedding-001",
                "displayName": "Embedding",
                "supportedGenerationMethods": ["embedContent"],
            },
        ]
    }

    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["url"] = req.full_url
        return FakeResponse(json.dumps(payload))

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", fake_urlopen)

    models = list_gemini_models("secret")
    assert captured["url"].endswith("/models?key=secret")
    assert len(models) == 1
    assert models[0].name == "models/gemini-2.0-flash"
    assert models[0].display_name == "Gemini Flash"
    assert "generateContent" in models[0].supported_generation_methods


def test_gemini_chat_sends_generation_config(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=30):
      captured["url"] = req.full_url
      captured["body"] = json.loads(req.data.decode("utf-8"))
      return FakeResponse(
          json.dumps(
              {
                  "candidates": [
                      {
                          "content": {
                              "parts": [
                                  {"text": "hello"},
                              ]
                          }
                      }
                  ]
              }
          )
      )

    monkeypatch.setattr(provider_module.urllib.request, "urlopen", fake_urlopen)

    provider = GeminiProvider(
        model="models/gemini-2.0-flash",
        api_key="secret",
        temperature=0.3,
        top_p=0.9,
    )
    text = provider.chat("system prompt", "user prompt")

    assert text == "hello"
    assert "/models/models%2Fgemini-2.0-flash:generateContent?key=secret" in captured["url"]
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == "system prompt"
    assert captured["body"]["contents"][0]["parts"][0]["text"] == "user prompt"
    assert captured["body"]["generationConfig"]["temperature"] == 0.3
    assert captured["body"]["generationConfig"]["topP"] == 0.9