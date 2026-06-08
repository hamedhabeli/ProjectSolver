from __future__ import annotations

from psai.solvers.mus import extract_mus


def test_extract_mus_returns_conflicting_ids():
    named = {
        "A1": "(declare-const p Bool) (assert p)",
        "A2": "(declare-const p Bool) (assert (not p))",
        "A3": "(declare-const q Bool) (assert q)",
    }
    core = extract_mus(named, timeout_ms=2000)
    assert "A1" in core
    assert "A2" in core
    assert "A3" not in core
