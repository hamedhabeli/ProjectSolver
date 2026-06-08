from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import z3


@dataclass(frozen=True)
class MusResult:
    status: str
    core_item_ids: List[str]
    reason_unknown: Optional[str] = None


def extract_mus(named_assertions: Dict[str, str], timeout_ms: int) -> List[str]:
    """
    Return a minimal-ish unsat core over the named assertion snippets.

    For the current repo tests, this must reliably return the conflicting IDs
    from a small set of SMT-LIB2 snippets, even when unsat_core tracking is not
    dependable enough for the exact input shape.
    """
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    items: List[Tuple[str, str]] = sorted(named_assertions.items(), key=lambda kv: kv[0])
    if not items:
        return []

    full_status = _subset_status(items, timeout_ms)
    if full_status != "unsat":
        return []

    core_ids = [item_id for item_id, _ in items]

    # Greedy deletion: keep removing any item whose removal preserves UNSAT.
    changed = True
    while changed:
        changed = False
        for item_id in list(core_ids):
            trial = [(iid, smt2) for iid, smt2 in items if iid in core_ids and iid != item_id]
            if not trial:
                continue

            trial_status = _subset_status(trial, timeout_ms)
            if trial_status == "unsat":
                core_ids.remove(item_id)
                changed = True
                break

    return sorted(core_ids)


def extract_mus_detailed(named_assertions: Dict[str, str], timeout_ms: int) -> Dict[str, object]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    items: List[Tuple[str, str]] = sorted(named_assertions.items(), key=lambda kv: kv[0])
    if not items:
        return asdict(MusResult(status="sat", core_item_ids=[]))

    try:
        status = _subset_status(items, timeout_ms)

        if status == "sat":
            return asdict(MusResult(status="sat", core_item_ids=[]))

        if status == "unsat":
            return asdict(MusResult(status="unsat", core_item_ids=extract_mus(named_assertions, timeout_ms)))

        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown="timeout/unknown"))
    except z3.Z3Exception as e:
        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown=str(e)))


def _subset_status(items: Sequence[Tuple[str, str]], timeout_ms: int) -> str:
    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    try:
        for _, smt2 in items:
            for expr in _parse_as_bool_list(smt2):
                s.add(expr)

        r = s.check()
        if r == z3.sat:
            return "sat"
        if r == z3.unsat:
            return "unsat"
        return "unknown"
    except z3.Z3Exception:
        return "unknown"


def _parse_as_bool_list(smt2: str) -> List[z3.BoolRef]:
    smt2 = smt2.strip()
    if not smt2:
        raise z3.Z3Exception("Empty SMT-LIB2 snippet")

    parsed = z3.parse_smt2_string(smt2)

    exprs: List[z3.BoolRef] = []
    try:
        for p in parsed:
            if isinstance(p, z3.BoolRef):
                exprs.append(p)
            else:
                raise z3.Z3Exception(f"Non-boolean expression parsed: {p}")
    except TypeError:
        if isinstance(parsed, z3.BoolRef):
            exprs.append(parsed)
        else:
            raise z3.Z3Exception("Parsed content is not boolean.")

    if not exprs:
        raise z3.Z3Exception("No boolean expressions found")

    return exprs
