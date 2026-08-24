"""Demo entrypoint for the policy -> rule -> faithfulness-gate pipeline.

Runs fully offline with the bundled mock components. Usage:

    python run_demo.py
    python run_demo.py --policy data/sample_policy.txt --claims data/sample_claims.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import run
from src.schema import Verdict

BAR = "=" * 68


def _fmt_rule(r):
    return f"[{r.rule_id}] {r.claim_sentence}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="data/sample_policy.txt")
    ap.add_argument("--claims", default="data/sample_claims.json")
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    policy_text = Path(args.policy).read_text(encoding="utf-8")
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8"))

    result = run(policy_text, claims, pass_threshold=args.threshold)

    print(BAR)
    print("STAGE 1-2  Extracted & verbalized rules")
    print(BAR)
    for r in result.rules:
        print(" ", _fmt_rule(r))

    print("\n" + BAR)
    print("STAGE 3    Faithfulness gate")
    print(BAR)
    for g in result.gate_results:
        mark = "PASS " if g.passed else "BLOCK"
        print(f"  [{mark}] {g.rule_id}  conf={g.confidence:<4}  ({g.verdict.value})")
        print(f"          claim   : {g.claim_sentence}")
        print(f"          evidence: {g.evidence[:80]}")
    if result.blocked_rules:
        print("\n  --> Routed to HUMAN REVIEW:",
              ", ".join(g.rule_id for g in result.blocked_rules))

    print("\n" + BAR)
    print("STAGE 4    Adjudication of sample claims (gate-approved rules only)")
    print(BAR)
    for d in result.decisions:
        print(f"  {d.claim_id}: {d.decision.upper():<16} {d.rationale}")

    print("\n" + BAR)
    passed = len(result.passed_rules)
    total = len(result.rules)
    print(f"SUMMARY    {passed}/{total} rules passed the gate; "
          f"{total - passed} blocked for human review.")
    print(BAR)


if __name__ == "__main__":
    main()
