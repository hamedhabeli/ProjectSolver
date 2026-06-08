from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

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

    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    try:
        for item_id, smt2 in items:
            exprs = _parse_as_bool_list(smt2)
            tracked = z3.Bool(f"track::{item_id}")
            combined = z3.And(*exprs) if len(exprs) > 1 else exprs[0]
            s.assert_and_track(combined, tracked)

        r = s.check()
        if r == z3.unsat:
            core = s.unsat_core()
            ids: List[str] = []
            for b in core:
                name = b.decl().name() if hasattr(b, "decl") else str(b)
                if name.startswith("track::"):
                    ids.append(name.split("track::", 1)[1])
                else:
                    ids.append(name)
            return sorted(set(ids))
        return []
    except z3.Z3Exception:
        return []


def extract_mus_detailed(named_assertions: Dict[str, str], timeout_ms: int) -> Dict[str, object]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    items: List[Tuple[str, str]] = sorted(named_assertions.items(), key=lambda kv: kv[0])
    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    try:
        for item_id, smt2 in items:
            exprs = _parse_as_bool_list(smt2)
            tracked = z3.Bool(f"track::{item_id}")
            combined = z3.And(*exprs) if len(exprs) > 1 else exprs[0]
            s.assert_and_track(combined, tracked)

        r = s.check()
        if r == z3.unsat:
            core = s.unsat_core()
            ids: List[str] = []
            for b in core:
                name = b.decl().name()
                ids.append(name.split("track::", 1)[1] if name.startswith("track::") else name)
            ids = sorted(set(ids))
            return asdict(MusResult(status="unsat", core_item_ids=ids))
        if r == z3.sat:
            return asdict(MusResult(status="sat", core_item_ids=[]))
        reason = None
        try:
            reason = s.reason_unknown()
        except Exception:  # noqa: BLE001
            reason = None
        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown=reason))
    except z3.Z3Exception as e:
        return asdict(MusResult(status="unknown", core_item_ids=[], reason_unknown=str(e)))


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
