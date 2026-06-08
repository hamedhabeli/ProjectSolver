from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import z3


@dataclass(frozen=True)
class SatResult:
    status: str
    reason_unknown: Optional[str] = None


@dataclass(frozen=True)
class ProveResult:
    status: str
    sat_status_of_negation: str
    reason_unknown: Optional[str] = None


def check_sat(theory_smt2: str, timeout_ms: int) -> Dict[str, Any]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    try:
        s.add(_parse_as_assertions(theory_smt2))
        r = s.check()
        return asdict(_sat_result_from_z3(r, s))
    except z3.Z3Exception as e:
        return asdict(SatResult(status="unknown", reason_unknown=str(e)))


def prove_goal(theory_smt2: str, goal_smt2: str, timeout_ms: int) -> Dict[str, Any]:
    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative")

    s = z3.Solver()
    if timeout_ms > 0:
        s.set("timeout", timeout_ms)

    try:
        s.add(_parse_as_assertions(theory_smt2))

        goal_expr = _parse_goal_expr(goal_smt2)
        s.add(z3.Not(goal_expr))

        r = s.check()
        sat_res = _sat_result_from_z3(r, s)

        if sat_res.status == "unsat":
            return asdict(ProveResult(status="proved", sat_status_of_negation="unsat"))
        if sat_res.status == "sat":
            return asdict(ProveResult(status="disproved", sat_status_of_negation="sat"))
        return asdict(ProveResult(status="undetermined", sat_status_of_negation="unknown", reason_unknown=sat_res.reason_unknown))
    except z3.Z3Exception as e:
        return asdict(ProveResult(status="undetermined", sat_status_of_negation="unknown", reason_unknown=str(e)))


def _sat_result_from_z3(r: z3.CheckSatResult, s: z3.Solver) -> SatResult:
    if r == z3.sat:
        return SatResult(status="sat")
    if r == z3.unsat:
        return SatResult(status="unsat")
    reason = None
    try:
        reason = s.reason_unknown()
    except Exception:  # noqa: BLE001
        reason = None
    return SatResult(status="unknown", reason_unknown=reason)


def _parse_as_assertions(smt2: str) -> list[z3.BoolRef]:
    smt2 = smt2.strip()
    if not smt2:
        return []

    parsed = z3.parse_smt2_string(smt2)

    if isinstance(parsed, list):
        out: list[z3.BoolRef] = []
        for p in parsed:
            if isinstance(p, z3.BoolRef):
                out.append(p)
            else:
                raise z3.Z3Exception(f"Non-boolean assertion parsed: {p}")
        return out

    if isinstance(parsed, z3.BoolRef):
        return [parsed]

    raise z3.Z3Exception("Parsed content is not a boolean expression/list.")


def _parse_goal_expr(goal_smt2: str) -> z3.BoolRef:
    goal_smt2 = goal_smt2.strip()
    if not goal_smt2:
        raise z3.Z3Exception("Empty goal SMT-LIB2 input")

    parsed = z3.parse_smt2_string(goal_smt2)

    if isinstance(parsed, list):
        bools = []
        for p in parsed:
            if isinstance(p, z3.BoolRef):
                bools.append(p)
            else:
                raise z3.Z3Exception(f"Non-boolean goal parsed: {p}")
        if not bools:
            raise z3.Z3Exception("No boolean expressions found in goal")
        return z3.And(*bools) if len(bools) > 1 else bools[0]

    if isinstance(parsed, z3.BoolRef):
        return parsed

    raise z3.Z3Exception("Goal is not a boolean expression/list.")
