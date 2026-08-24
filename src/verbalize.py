"""Stage 2 -- Rule verbalization.

Turns each structured rule into ONE verifiable natural-language claim sentence.
This is the thin adapter that lets a structured object be checked by an NLI
detector (which reasons over sentences, not JSON). The verbalization must be a
faithful, literal rendering of the fields -- adding nothing -- so the gate is
checking the *extraction*, not the wording.

Kept as a deterministic template on purpose (fast, auditable, no extra model
call). You could swap in an LLM verbalizer, but then the template's neutrality
guarantee is lost, so template-first is the safer default for the POC.
"""

from __future__ import annotations

from .schema import Coverage, Rule

_COND_PHRASING = {
    "diagnosis:diabetes_mellitus": "the patient has diabetes mellitus",
    "diagnosis:gestational_diabetes": "the patient has gestational diabetes",
    "insulin_treated:true": "the patient is insulin-treated",
    "prior_auth:approved": "prior authorization has been approved",
    "age_ge:65": "the patient is aged 65 or older",
}


def _phrase(token: str) -> str:
    return _COND_PHRASING.get(token, token.replace(":", " ").replace("_", " "))


def verbalize(rule: Rule) -> Rule:
    conds = [_phrase(c) for c in rule.conditions]
    cover = "is covered" if rule.coverage == Coverage.COVERED else "is not covered"

    if conds:
        cond_clause = " and ".join(conds)
        sentence = f"{rule.service} {cover} when {cond_clause}."
    else:
        sentence = f"{rule.service} {cover}."

    if rule.exclusions:
        excl = " and ".join(_phrase(e) for e in rule.exclusions)
        sentence = sentence[:-1] + f", excluding cases where {excl}."

    # Capitalize first letter for a clean claim sentence.
    rule.claim_sentence = sentence[0].upper() + sentence[1:]
    return rule


def verbalize_all(rules: list[Rule]) -> list[Rule]:
    return [verbalize(r) for r in rules]
