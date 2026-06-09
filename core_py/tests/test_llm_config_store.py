from __future__ import annotations

from pathlib import Path

from psai.state_store import StateStore

def test_llm_config_persists_round_trip(tmp_path: Path):
    db_path = tmp_path / "projectsolver.sqlite3"

    store = StateStore(db_path=db_path)
    store.set_llm_config(
        {
            "provider": "gemini",
            "api_key": "secret-key",
            "model": "models/gemini-2.0-flash",
            "temperature": 0.4,
            "top_p": 0.88,
            "timeout_s": 42,
            "max_retries": 3,
        }
    )
    first = store.get_llm_config()
    assert first["provider"] == "gemini"
    assert first["api_key"] == "secret-key"
    assert first["model"] == "models/gemini-2.0-flash"
    assert first["temperature"] == 0.4
    assert first["top_p"] == 0.88
    assert first["timeout_s"] == 42
    assert first["max_retries"] == 3
    store.close()

    reopened = StateStore(db_path=db_path)
    second = reopened.get_llm_config()
    assert second == first
    reopened.close()