from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import z3


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: List[ValidationError]
    ast_summary: Optional[Dict[str, Any]] = None


def validate_smtlib2(formula_str: str) -> Dict[str, Any]:
    formula_str = formula_str.strip()
    if not formula_str:
        res = ValidationResult(
            valid=False,
            errors=[ValidationError(code="EMPTY", message="Empty SMT-LIB2 input")],
            ast_summary=None,
        )
        return _to_dict(res)

    try:
        parsed = z3.parse_smt2_string(formula_str)
        ast_summary = _summarize_parsed(parsed)
        res = ValidationResult(valid=True, errors=[], ast_summary=ast_summary)
        return _to_dict(res)
    except z3.Z3Exception as e:
        err = ValidationError(code="Z3_PARSE_ERROR", message="SMT-LIB2 parse/type error", detail=str(e))
        res = ValidationResult(valid=False, errors=[err], ast_summary=None)
        return _to_dict(res)
    except Exception as e:  # noqa: BLE001
        err = ValidationError(code="VALIDATION_ERROR", message="Unexpected validation error", detail=str(e))
        res = ValidationResult(valid=False, errors=[err], ast_summary=None)
        return _to_dict(res)


def _summarize_parsed(parsed: Union[z3.AstRef, List[z3.AstRef]]) -> Dict[str, Any]:
    if isinstance(parsed, list):
        kinds = []
        for p in parsed:
            kinds.append(_expr_kind(p))
        return {"count": len(parsed), "kinds": kinds}
    return {"count": 1, "kinds": [_expr_kind(parsed)]}


def _expr_kind(expr: z3.AstRef) -> str:
    if isinstance(expr, z3.BoolRef):
        return "Bool"
    if isinstance(expr, z3.ArithRef):
        return "Arith"
    return type(expr).__name__


def _to_dict(res: ValidationResult) -> Dict[str, Any]:
    return asdict(res)
