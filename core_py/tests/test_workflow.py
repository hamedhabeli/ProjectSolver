from __future__ import annotations

import json
from pathlib import Path

import pytest

from psai.logic.orchestrator import run_workflow
from psai.llm.provider import LLMProvider
from psai.state_store import StateStore


class MockLLM(LLMProvider):
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise RuntimeError("No more mock responses available")
        return self._responses.pop(0)


@pytest.fixture()
def store(tmp_path: Path):
    db_path = tmp_path / "test_workflow.sqlite3"
    s = StateStore(db_path=db_path)
    try:
        yield s
    finally:
        s.close()


def test_run_workflow_formalize_check_explain_chain(store: StateStore):
    repo = store.repo_create("workflow-repo", initial_payload={})
    root = store.get_node(repo.repo_id, repo.head)

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
                    "fa_text": "نفی پ",
                    "formal_smt2": "(declare-const p Bool) (assert (not p))",
                },
            ]
        },
        ensure_ascii=False,
    )

    explain_response = json.dumps(
        {
            "nl_summary": "دو گزاره دربارهٔ p با هم ناسازگار هستند.",
            "choices": [
                {
                    "choice_id": "C1",
                    "action": "retract",
                    "target_item_id": "A2",
                    "nl": "گزارهٔ دوم را حذف یا اصلاح کنید.",
                }
            ],
        },
        ensure_ascii=False,
    )

    llm = MockLLM([formalize_response, explain_response])

    payload = {
        "user_problem_text": "پ و همچنین نقیض پ",
        "context": {},
        "timeout_ms": 2000,
    }
    ws = store.commit(repo.repo_id, parent_id=root.node_id, message="workspace", payload=payload)

    result = run_workflow(repo.repo_id, workspace_id=ws.node_id, store=store, llm=llm)

    assert result["step"] == "workflow.run"
    assert result["status"] == "ok"

    assert result["formalize"]["status"] == "ok"
    assert len(result["formalize"]["valid_items"]) == 2

    assert result["check_consistency"]["sat"]["status"] == "unsat"

    assert result["explain_contradiction"] is not None
    assert result["explain_contradiction"]["status"] == "ok"
    assert result["explain_contradiction"]["explanation"]["nl_summary"] == "دو گزاره دربارهٔ p با هم ناسازگار هستند."

    assert len(llm.calls) == 2
