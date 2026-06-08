from __future__ import annotations

import json
from pathlib import Path

import pytest

from psai.logic.orchestrator import run_cycle
from psai.llm.provider import LLMProvider
from psai.state_store import StateStore


class MockLLM(LLMProvider):
    def __init__(self, response_text: str) -> None:
        self._resp = response_text

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self._resp


@pytest.fixture()
def store(tmp_path: Path):
    db_path = tmp_path / "test_orchestrator.sqlite3"
    s = StateStore(db_path=db_path)
    try:
        yield s
    finally:
        s.close()


def test_run_cycle_detects_contradiction_and_extracts_core_ids(store: StateStore):
    repo = store.repo_create("R-contradiction", initial_payload={})
    root = store.get_node(repo.repo_id, repo.head)

    payload = {
        "timeout_ms": 2000,
        "theory_smt2": '''
            (declare-const p Bool)
            (assert p)
            (assert (not p))
        ''',
        "named_assertions": {
            "A1": "(declare-const p Bool) (assert p)",
            "A2": "(declare-const p Bool) (assert (not p))",
            "A3": "(declare-const q Bool) (assert q)"
        },
        "goal_smt2": "p"
    }

    ws = store.commit(repo.repo_id, parent_id=root.node_id, message="ws", payload=payload)
    llm = MockLLM("{}")

    res = run_cycle(repo.repo_id, workspace_id=ws.node_id, step="check_consistency", store=store, llm=llm)
    assert res["sat"]["status"] == "unsat"
    core = res["unsat_core_item_ids"]
    assert "A1" in core and "A2" in core
    assert "A3" not in core


def test_run_cycle_abduce_filters_invalid_and_finds_proving_candidate(store: StateStore):
    repo = store.repo_create("R-abduce", initial_payload={})
    root = store.get_node(repo.repo_id, repo.head)

    payload = {
        "timeout_ms": 2000,
        "theory_smt2": '''
            (declare-const p Bool)
            (declare-const q Bool)
            (assert p)
        ''',
        "goal_smt2": "q",
    }
    ws = store.commit(repo.repo_id, parent_id=root.node_id, message="ws", payload=payload)

    llm_response = json.dumps(
        {
            "candidates": [
                {"candidate_id": "C_bad", "kind": "assumption", "nl": "خراب", "formal": "(assert (=> p q)"},
                {"candidate_id": "C_good", "kind": "assumption", "nl": "اگر p آنگاه q", "formal": "(assert (=> p q))"},
            ]
        },
        ensure_ascii=False,
    )
    llm = MockLLM(llm_response)

    res = run_cycle(repo.repo_id, workspace_id=ws.node_id, step="abduce", store=store, llm=llm)
    invalid_ids = {c["candidate_id"] for c in res["invalid_candidates"]}
    assert "C_bad" in invalid_ids
    assert res["proved_by"] is not None
    assert res["proved_by"]["candidate_id"] == "C_good"
