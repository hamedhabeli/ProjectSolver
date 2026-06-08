from __future__ import annotations

from psai.logic.syntax_gate import validate_smtlib2
from psai.solvers.z3_backend import check_sat, prove_goal


def test_syntax_gate_valid_formula():
    smt = '''
    (declare-const p Bool)
    (declare-const q Bool)
    (assert (=> p q))
    (assert p)
    '''
    res = validate_smtlib2(smt)
    assert res["valid"] is True


def test_syntax_gate_invalid_parenthesis():
    smt = '''
    (declare-const p Bool)
    (assert (and p)
    '''
    res = validate_smtlib2(smt)
    assert res["valid"] is False


def test_syntax_gate_type_mismatch():
    smt = '''
    (declare-const x Int)
    (assert (and x true))
    '''
    res = validate_smtlib2(smt)
    assert res["valid"] is False
    assert res["errors"][0]["code"] == "Z3_PARSE_ERROR"


def test_z3_backend_prove_modus_ponens():
    theory = '''
    (declare-const p Bool)
    (declare-const q Bool)
    (assert (=> p q))
    (assert p)
    '''
    res = prove_goal(theory, "q", timeout_ms=2000)
    assert res["status"] == "proved"


def test_z3_backend_detect_contradiction_unsat():
    theory = '''
    (declare-const p Bool)
    (assert p)
    (assert (not p))
    '''
    res = check_sat(theory, timeout_ms=2000)
    assert res["status"] == "unsat"
