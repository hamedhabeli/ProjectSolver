from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from psai.llm.prompts import ABDUCTION_PROMPT, FORMALIZATION_PROMPT, MUS_EXPLANATION_PROMPT
from psai.llm.provider import LLMProvider
from psai.logic.syntax_gate import validate_smtlib2
from psai.solvers.mus import extract_mus
from psai.solvers.z3_backend import check_sat, prove_goal
from psai.state_store import StateStore


def _get_state_payload(store: StateStore, repo_id: str, workspace_id: str) -> Dict[str, Any]:
    node = store.get_node(repo_id, workspace_id)
    if not isinstance(node.payload, dict):
        raise ValueError("State payload must be a JSON object (dict)")
    return node.payload


def _require(payload: Dict[str, Any], key: str, typ: type) -> Any:
    if key not in payload:
        raise KeyError(f"Missing state key: {key}")
    val = payload[key]
    if not isinstance(val, typ):
        raise TypeError(f"Invalid type for state key '{key}': expected {typ.__name__}")
    return val


def _llm_json_with_retry(
    llm: LLMProvider,
    system_prompt: str,
    user_prompt: str,
    *,
    max_json_retries: int = 2,
) -> Dict[str, Any]:
    attempt_user_prompt = user_prompt
    last_error: Optional[str] = None

    for attempt in range(max_json_retries + 1):
        llm_text = llm.chat(system_prompt=system_prompt, user_prompt=attempt_user_prompt)
        try:
            parsed = json.loads(llm_text)
            if not isinstance(parsed, dict):
                raise JSONDecodeError("Top-level JSON must be an object", llm_text, 0)
            return parsed
        except JSONDecodeError as e:
            last_error = str(e)
            if attempt >= max_json_retries:
                raise
            attempt_user_prompt = (
                user_prompt
                + "\n\n"
                + "Your previous answer was invalid JSON and could not be parsed.\n"
                + f"Parser error: {last_error}\n"
                + "Return ONLY valid JSON, with no markdown, no code fences, no commentary."
            )

    raise RuntimeError("Unreachable: JSON retry loop exhausted")


def _split_top_level_commands(smt2: str) -> List[str]:
    text = smt2.strip()
    if not text:
        return []

    commands: List[str] = []
    depth = 0
    start = None
    i = 0

    while i < len(text):
        ch = text[i]

        if ch == ";":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                depth = 0
            if depth == 0 and start is not None:
                commands.append(text[start : i + 1].strip())
                start = None
        i += 1

    if start is not None and start < len(text):
        tail = text[start:].strip()
        if tail:
            commands.append(tail)

    if not commands and text:
        commands = [text]

    return commands


def _build_combined_theory_and_named_assertions(
    items: List[Dict[str, Any]],
) -> tuple[str, Dict[str, str]]:
    decls: List[str] = []
    asserts: List[str] = []
    seen_decls: set[str] = set()
    named_assertions: Dict[str, str] = {}

    for item in items:
        item_id = item["item_id"]
        formal_smt2 = item["formal_smt2"]
        named_assertions[item_id] = formal_smt2

        for cmd in _split_top_level_commands(formal_smt2):
            stripped = cmd.strip()
            if not stripped:
                continue
            if stripped.startswith("(declare-"):
                if stripped not in seen_decls:
                    seen_decls.add(stripped)
                    decls.append(stripped)
            elif stripped.startswith("(assert"):
                asserts.append(stripped)
            else:
                asserts.append(f"(assert {stripped})")

    theory_smt2 = "\n".join(decls + asserts)
    return theory_smt2, named_assertions


def run_cycle(
    repo_id: str,
    workspace_id: str,
    step: str,
    store: StateStore,
    llm: LLMProvider,
) -> dict:
    state = _get_state_payload(store, repo_id, workspace_id)

    if step == "check_consistency":
        theory_smt2 = _require(state, "theory_smt2", str)
        sat = check_sat(theory_smt2, timeout_ms=int(state.get("timeout_ms", 2000)))

        result: Dict[str, Any] = {
            "step": "check_consistency",
            "sat": sat,
        }

        if sat.get("status") == "unsat":
            named = state.get("named_assertions", None)
            if isinstance(named, dict) and named:
                if not all(isinstance(k, str) and isinstance(v, str) for k, v in named.items()):
                    raise TypeError("named_assertions must be a dict[str,str]")
                core_ids = extract_mus(named_assertions=named, timeout_ms=int(state.get("timeout_ms", 2000)))
                result["unsat_core_item_ids"] = core_ids
            else:
                result["unsat_core_item_ids"] = []
                result["note"] = "named_assertions not provided; unsat core unavailable"

        return result

    if step == "prove_goals":
        theory_smt2 = _require(state, "theory_smt2", str)
        goal_smt2 = _require(state, "goal_smt2", str)
        timeout_ms = int(state.get("timeout_ms", 2000))

        pr = prove_goal(theory_smt2, goal_smt2, timeout_ms=timeout_ms)
        return {"step": "prove_goals", "prove": pr}

    if step == "formalize":
        user_problem_text = _require(state, "user_problem_text", str)
        context = state.get("context", {})
        if not isinstance(context, dict):
            raise TypeError("context must be an object if provided")

        user_prompt = json.dumps(
            {
                "user_problem_text": user_problem_text,
                "context": context,
                "items": state.get("items", []),
            },
            ensure_ascii=False,
        )

        obj = _llm_json_with_retry(llm, FORMALIZATION_PROMPT, user_prompt, max_json_retries=2)

        items = obj.get("items", [])
        if not isinstance(items, list):
            return {
                "step": "formalize",
                "status": "llm_invalid_shape",
                "error": {"message": "items must be a list"},
            }

        valid_items: List[Dict[str, Any]] = []
        invalid_items: List[Dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("item_id", "")
            fa_text = item.get("fa_text", "")
            formal_smt2 = item.get("formal_smt2", "")
            if not isinstance(item_id, str) or not isinstance(fa_text, str) or not isinstance(formal_smt2, str):
                continue

            validation = validate_smtlib2(formal_smt2)
            if validation.get("valid") is True:
                valid_items.append(
                    {
                        "item_id": item_id,
                        "fa_text": fa_text,
                        "formal_smt2": formal_smt2,
                        "syntax": validation,
                    }
                )
            else:
                invalid_items.append(
                    {
                        "item_id": item_id,
                        "fa_text": fa_text,
                        "formal_smt2": formal_smt2,
                        "syntax": validation,
                    }
                )

        return {
            "step": "formalize",
            "status": "ok",
            "items": items,
            "valid_items": valid_items,
            "invalid_items": invalid_items,
        }

    if step == "explain_contradiction":
        unsat_core_item_ids = _require(state, "unsat_core_item_ids", list)
        named_assertions = _require(state, "named_assertions", dict)

        if not all(isinstance(x, str) for x in unsat_core_item_ids):
            raise TypeError("unsat_core_item_ids must be a list[str]")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in named_assertions.items()):
            raise TypeError("named_assertions must be a dict[str,str]")

        core_items = [
            {
                "item_id": item_id,
                "formal_smt2": named_assertions.get(item_id, ""),
                "fa_text": state.get("named_assertions_fa", {}).get(item_id, "")
                if isinstance(state.get("named_assertions_fa", {}), dict)
                else "",
            }
            for item_id in unsat_core_item_ids
        ]

        user_prompt = json.dumps(
            {
                "unsat_core_item_ids": unsat_core_item_ids,
                "core_items": core_items,
                "named_assertions": named_assertions,
                "user_language": "fa",
            },
            ensure_ascii=False,
        )

        explanation = _llm_json_with_retry(
            llm,
            MUS_EXPLANATION_PROMPT,
            user_prompt,
            max_json_retries=2,
        )

        return {
            "step": "explain_contradiction",
            "status": "ok",
            "unsat_core_item_ids": unsat_core_item_ids,
            "explanation": explanation,
        }

    if step == "abduce":
        theory_smt2 = _require(state, "theory_smt2", str)
        goal_smt2 = _require(state, "goal_smt2", str)
        timeout_ms = int(state.get("timeout_ms", 2000))

        user_prompt = json.dumps(
            {
                "goal": {"formal": goal_smt2},
                "theory": {"smt2": theory_smt2},
                "constraints": {"max_candidates": 8},
            },
            ensure_ascii=False,
        )

        obj = _llm_json_with_retry(llm, ABDUCTION_PROMPT, user_prompt, max_json_retries=2)

        cands = obj.get("candidates", [])
        if not isinstance(cands, list):
            return {
                "step": "abduce",
                "status": "llm_invalid_shape",
                "error": {"message": "candidates must be a list"},
            }

        valid_candidates: List[Dict[str, Any]] = []
        invalid_candidates: List[Dict[str, Any]] = []

        for c in cands:
            if not isinstance(c, dict):
                continue
            cand_id = c.get("candidate_id", "")
            formal = c.get("formal", "")
            if not isinstance(cand_id, str) or not isinstance(formal, str):
                continue

            v = validate_smtlib2(formal)
            if v.get("valid") is True:
                valid_candidates.append(
                    {
                        "candidate_id": cand_id,
                        "kind": c.get("kind"),
                        "nl": c.get("nl"),
                        "formal": formal,
                    }
                )
            else:
                invalid_candidates.append(
                    {
                        "candidate_id": cand_id,
                        "formal": formal,
                        "errors": v.get("errors", []),
                    }
                )

        proved_by: Optional[Dict[str, Any]] = None
        checks: List[Dict[str, Any]] = []

        for c in valid_candidates:
            combined_theory = theory_smt2 + "\n" + c["formal"]
            pr = prove_goal(combined_theory, goal_smt2, timeout_ms=timeout_ms)
            checks.append({"candidate_id": c["candidate_id"], "prove": pr})
            if pr.get("status") == "proved":
                proved_by = {"candidate_id": c["candidate_id"], "formal": c["formal"]}
                break

        return {
            "step": "abduce",
            "status": "ok",
            "valid_candidates": valid_candidates,
            "invalid_candidates": invalid_candidates,
            "checks": checks,
            "proved_by": proved_by,
        }

    return {"step": step, "status": "unknown_step"}


def run_workflow(
    repo_id: str,
    workspace_id: str,
    store: StateStore,
    llm: LLMProvider,
) -> dict:
    state = _get_state_payload(store, repo_id, workspace_id)

    if "user_problem_text" not in state:
        return {
            "step": "workflow.run",
            "status": "skipped",
            "reason": "user_problem_text not present in state",
        }

    if state.get("formalized", False) is True:
        formalize_res = {
            "step": "formalize",
            "status": "skipped",
            "reason": "state already marked as formalized",
        }
        raw_items = state.get("formalized_items", [])
        if not isinstance(raw_items, list):
            raw_items = []
    else:
        formalize_res = run_cycle(
            repo_id=repo_id,
            workspace_id=workspace_id,
            step="formalize",
            store=store,
            llm=llm,
        )
        raw_items = formalize_res.get("valid_items", [])

    if not isinstance(raw_items, list):
        raw_items = []

    if not raw_items:
        return {
            "step": "workflow.run",
            "status": "ok",
            "formalize": formalize_res,
            "check_consistency": {
                "step": "check_consistency",
                "sat": {"status": "unknown", "reason_unknown": "No valid formalized items"},
            },
            "explain_contradiction": None,
            "final": formalize_res,
        }

    theory_smt2, named_assertions = _build_combined_theory_and_named_assertions(raw_items)

    check_res = check_sat(theory_smt2, timeout_ms=int(state.get("timeout_ms", 2000)))
    final_explain: Optional[Dict[str, Any]] = None

    if check_res.get("status") == "unsat":
        unsat_core_item_ids = extract_mus(named_assertions=named_assertions, timeout_ms=int(state.get("timeout_ms", 2000)))

        explain_state = {
            "unsat_core_item_ids": unsat_core_item_ids,
            "named_assertions": named_assertions,
            "named_assertions_fa": {item["item_id"]: item.get("fa_text", "") for item in raw_items},
        }

        user_prompt = json.dumps(
            {
                "unsat_core_item_ids": explain_state["unsat_core_item_ids"],
                "core_items": [
                    {
                        "item_id": item_id,
                        "formal_smt2": named_assertions.get(item_id, ""),
                        "fa_text": explain_state["named_assertions_fa"].get(item_id, ""),
                    }
                    for item_id in unsat_core_item_ids
                ],
                "named_assertions": named_assertions,
                "user_language": "fa",
            },
            ensure_ascii=False,
        )

        explanation = _llm_json_with_retry(
            llm,
            MUS_EXPLANATION_PROMPT,
            user_prompt,
            max_json_retries=2,
        )

        final_explain = {
            "step": "explain_contradiction",
            "status": "ok",
            "unsat_core_item_ids": unsat_core_item_ids,
            "explanation": explanation,
        }

    return {
        "step": "workflow.run",
        "status": "ok",
        "formalize": formalize_res,
        "check_consistency": {"step": "check_consistency", "sat": check_res},
        "explain_contradiction": final_explain,
        "final": final_explain if final_explain is not None else {"step": "check_consistency", "sat": check_res},
    }
