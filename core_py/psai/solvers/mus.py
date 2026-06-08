from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import z3


@dataclass(frozen=True)
class MusResult:
    status: str
    core_item_ids: List[str]
    reason_unknown: Optional[str] = None


def extract_mus(named_assertions: Dict[str, str], timeout_ms: int) -> List[str]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    items: List[Tuple[str, str]] = sorted(named_assertions.items(), key=lambda kv: kv[0])
    if not items:
        return []

    try:
        status = _check_subset_status(items, timeout_ms)
        if status != "unsat":
            return []

        core_ids = [item_id for item_id, _ in items]

        changed = True
        while changed:
            changed = False
            for item_id in list(core_ids):
                trial_items = [(iid, smt2) for iid, smt2 in items if iid in core_ids and iid != item_id]
                if not trial_items:
                    continue

                trial_status = _check_subset_status(trial_items, timeout_ms)
                if trial_status == "unsat":
                    core_ids.remove(item_id)
                    changed = True
                    break

        return sorted(core_ids)

    except z3.Z3Exception:
        return []


def extract_mus_detailed(named_assertions: Dict[str, str], timeout_ms: int) -> Dict[str, object]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    items: List[Tuple[str, str]] = sorted(named_assertions.items(), key=lambda kv: kv[0])
    if not items:
        return asdict(MusResult(status="sat", core_item_ids=[]))

    try:
        status = _check_subset_status(items, timeout_ms)

        if status == "sat":
            return asdict(MusResult(status="sat", core_item_ids=[]))

        if status == "unsat":
            core_ids = extract_mus(named_assertions, timeout_ms)
            return asdict(MusResult(status="unsat", core_item_ids=core_ids))

        reason = None
        try:
            reason = _reason_unknown_for_subset(items, timeout_ms)
        except Exception:  # noqa: BLE001
            reason = None

        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown=reason))

    except z3.Z3Exception as e:
        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown=str(e)))


def _check_subset_status(items: Sequence[Tuple[str, str]], timeout_ms: int) -> str:
    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    for _, smt2 in items:
        for expr in _parse_as_bool_list(smt2):
            s.add(expr)

    r = s.check()
    if r == z3.sat:
        return "sat"
    if r == z3.unsat:
        return "unsat"
    return "unknown"


def _reason_unknown_for_subset(items: Sequence[Tuple[str, str]], timeout_ms: int) -> Optional[str]:
    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    for _, smt2 in items:
        for expr in _parse_as_bool_list(smt2):
            s.add(expr)

    r = s.check()
    if r in (z3.sat, z3.unsat):
        return None
    return s.reason_unknown()


def _parse_as_bool_list(smt2: str) -> List[z3.BoolRef]:
    smt2 = smt2.strip()
    if not smt2:
        raise z3.Z3Exception("Empty SMT-LIB2 snippet")

    parsed = z3.parse_smt2_string(smt2)

    if isinstance(parsed, list):
        out: List[z3.BoolRef] = []
        for p in parsed:
            if isinstance(p, z3.BoolRef):
                out.append(p)
            else:
                raise z3.Z3Exception(f"Non-boolean expression parsed: {p}")
        if not out:
            raise z3.Z3Exception("No boolean expressions found")
        return out

    if isinstance(parsed, z3.BoolRef):
        return [parsed]

    raise z3.Z3Exception("Parsed content is not boolean.")
