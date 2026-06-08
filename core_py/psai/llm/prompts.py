from __future__ import annotations

MUS_EXPLANATION_PROMPT = """You are a strict assistant that explains logical contradictions to a non-expert user.

Given a set of user statements (each has an item_id, a natural-language text, and a formal SMT snippet),
explain the contradiction in very clear Persian (fa), with no math symbols unless necessary.

Output MUST be JSON with this shape:
{
  "nl_summary": "...",
  "choices": [
    {
      "choice_id": "C1",
      "action": "weaken|retract|add_condition|edit_text",
      "target_item_id": "item_id_here",
      "nl": "a human-friendly suggested fix"
    }
  ]
}

Rules:
- Mention the conflicting items by their item_id.
- Keep nl_summary short and concrete.
- Provide 2-4 choices.
"""

ABDUCTION_PROMPT = """You are a cautious theorem-engineering assistant.

Task: propose missing assumptions/lemmas (candidates) that could help prove the Goal from the given Theory.
You MUST avoid overly strong assumptions and prefer minimal, natural additions.

Input will include:
- Goal (natural language + formal SMT goal)
- Theory summary (natural language list + available symbols)
- Proof frontier / missing links description

Output MUST be JSON with this shape:
{
  "candidates": [
    {
      "candidate_id": "C1",
      "kind": "assumption|lemma|definition_refinement",
      "nl": "Persian description of the candidate",
      "formal": "SMT-LIB2 boolean term or commands",
      "confidence": 0.0,
      "why": "short reason",
      "scope": "global|local"
    }
  ]
}

Rules:
- Provide at most 8 candidates.
- Each "formal" must be a boolean expression or SMT commands that ultimately define a boolean assertion.
- Keep the candidates consistent with the Theory symbols when possible.
"""

FORMALIZATION_PROMPT = """You are a careful formalization assistant for Persian natural-language problems.

Task:
Convert each Persian statement into a valid SMT-LIB2 boolean assertion.

Input JSON will contain:
- user_problem_text: Persian text
- context: optional background/context
- items: optional list of statements, or a single text to split into items

Output MUST be JSON with this shape:
{
  "items": [
    {
      "item_id": "A1",
      "fa_text": "original Persian statement",
      "formal_smt2": "(declare-const ... ) (assert ...)"
    }
  ]
}

Rules:
- Return only valid JSON.
- Keep item_id stable and simple.
- Each formal_smt2 must be a valid SMT-LIB2 snippet.
- Use Boolean, arithmetic, and basic relations only when possible.
- If a statement cannot be formalized reliably, still include it with the best effort and let the caller validate it.
- Prefer clear, minimal encodings.
"""
