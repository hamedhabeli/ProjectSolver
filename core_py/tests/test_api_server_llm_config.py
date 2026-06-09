from __future__ import annotations

import json
from pathlib import Path

import pytest

import psai.api_server as api_server
from psai.llm.provider import LLMProvider
from psai.state_store import StateStore


class DummyGeminiProvider(LLMProvider):
    last_init: dict[str, object] | None = None
    responses: list[str] = []

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
        timeout_s: int = 30,
        max_retries: int = 2,
        temperature: float = 0.2,
        top_p: float = 0.95,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        DummyGeminiProvider.last_init = {
            "api_key": api_key,
            "model": model,
            "timeout_s": timeout_s,
            "max_retries": max_retries,
            "temperature": temperature,
            "top_p": top_p,
            "base_url": base_url,
        }

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not DummyGeminiProvider.responses:
            raise RuntimeError("No more responses")
        return DummyGeminiProvider.responses.pop(0)


@pytest.fixture()
def store(tmp_path: Path):
    db_path = tmp_path / "test_api_server.sqlite3"
    s = StateStore(db_path=db_path)
    try:
        yield s
    finally:
        s.close()


def test_llm_config_round_trip_via_api(store: StateStore):
    server = api_server.ApiServer(store=store)

    saved = server._llm_set_config(
        {
            "config": {
                "provider": "gemini",
                "api_key": "persisted-key",
                "model": "models/gemini-2.0-flash",
                "temperature": 0.35,
                "top_p": 0.92,
                "timeout_s": 20,
                "max_retries": 1,
            }
        }
    )

    assert saved["api_key"] == "persisted-key"
    assert saved["model"] == "models/gemini-2.0-flash"

    loaded = server._llm_get_config({})
    assert loaded["api_key"] == "persisted-key"
    assert loaded["temperature"] == 0.35
    assert loaded["top_p"] == 0.92


def test_gemini_model_listing_route_uses_api_key(monkeypatch, store: StateStore):
    server = api_server.ApiServer(store=store)
    store.set_llm_config(
        {
            "provider": "gemini",
            "api_key": "stored-key",
            "model": "models/gemini-2.0-flash",
            "temperature": 0.2,
            "top_p": 0.95,
            "timeout_s": 30,
            "max_retries": 2,
        }
    )

    called = {}

    def fake_list_gemini_models(*, api_key: str, timeout_s: int, base_url: str = "https://generativelanguage.googleapis.com/v1beta", max_retries: int = 2):
        called["api_key"] = api_key
        called["timeout_s"] = timeout_s
        return [
            api_server.asdict(
                api_server.GeminiModelInfo(
                    name="models/gemini-2.0-flash",
                    display_name="Gemini Flash",
                    description="fast",
                    supported_generation_methods=["generateContent"],
                )
            )
        ]

    monkeypatch.setattr(api_server, "list_gemini_models", fake_list_gemini_models)

    res = server._llm_gemini_list_models({})
    assert called["api_key"] == "stored-key"
    assert res["models"][0]["name"] == "models/gemini-2.0-flash"


def test_workflow_uses_persisted_gemini_config(monkeypatch, store: StateStore):
    monkeypatch.setattr(api_server, "GeminiProvider", DummyGeminiProvider)

    server = api_server.ApiServer(store=store)
    store.set_llm_config(
        {
            "provider": "gemini",
            "api_key": "persisted-key",
            "model": "models/gemini-2.0-flash",
            "temperature": 0.45,
            "top_p": 0.91,
            "timeout_s": 30,
            "max_retries": 2,
        }
    )

    repo = store.repo_create("cfg-repo", initial_payload={})
    root = store.get_node(repo.repo_id, repo.head)
    ws = store.commit(
        repo.repo_id,
        parent_id=root.node_id,
        message="workspace",
        payload={
            "user_problem_text": "پ و نقیض پ",
            "context": {},
            "timeout_ms": 2000,
        },
    )

    formalize_response = json.dumps(
        {
            "items": [
                {
                    "item_id": "A1",
                    "fa_text": "پ",
                    "formal_smt2": "(declare-const p Bool) (assert p)",
                },
                {
                    "item_id": "A2",
                    "fa_text": "نقیض پ",
                    "formal_smt2": "(declare-const p Bool) (assert (not p))",
                },
            ]
        },
        ensure_ascii=False,
    )
    explain_response = json.dumps(
        {
            "nl_summary": "دو گزاره دربارهٔ p متناقض هستند.",
            "choices": [
                {
                    "choice_id": "C1",
                    "action": "retract",
                    "target_item_id": "A2",
                    "nl": "گزارهٔ دوم را حذف کنید.",
                }
            ],
        },
        ensure_ascii=False,
    )
    DummyGeminiProvider.responses = [formalize_response, explain_response]

    result = server._workflow_run({"repo_id": repo.repo_id, "workspace_id": ws.node_id})
    assert result["status"] == "ok"
    assert DummyGeminiProvider.last_init is not None
    assert DummyGeminiProvider.last_init["api_key"] == "persisted-key"
    assert DummyGeminiProvider.last_init["model"] == "models/gemini-2.0-flash"
    assert DummyGeminiProvider.last_init["temperature"] == 0.45
    assert DummyGeminiProvider.last_init["top_p"] == 0.91