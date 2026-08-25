"""Final CMS + LLM extraction + layered faithfulness gate demo.

This entrypoint leaves the old offline demo untouched. It loads structured
rules extracted from a public CMS CGM policy excerpt, runs the new layered gate,
and prints the rule-level verdicts plus the human review queue.

Usage:
    python run_cms_llm_demo.py
    python run_cms_llm_demo.py --policy data/cms_cgm_policy_clean.txt --rules data/llm_extracted_rules.json
    python run_cms_llm_demo.py --adversarial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.execute import adjudicate
from src.evaluation import evaluate_pipeline, write_evaluation_outputs
from src.completeness import (
    HEALTHCARE_POLICY_METADATA,
    audit_completeness,
    default_criteria_inventory,
)
from src.faithfulness_gate import ClaimVerdict, LayeredFaithfulnessGate
from src.schema import Coverage, Rule


BAR = "=" * 78


FALLBACK_CMS_POLICY = """CONTINUOUS GLUCOSE MONITORS (CGMs)

To be eligible for coverage of a CGM and related supplies, the beneficiary must meet all of the following initial coverage criteria (1)-(5):

1. The beneficiary has diabetes mellitus.

2. The beneficiary's treating practitioner has concluded that the beneficiary or beneficiary's caregiver has sufficient training using the CGM prescribed, as evidenced by providing a prescription.

3. The CGM is prescribed in accordance with its FDA indications for use.

4. The beneficiary for whom a CGM is being prescribed, to improve glycemic control, meets at least one of the criteria below:
   a. The beneficiary is insulin-treated; or
   b. The beneficiary has a history of problematic hypoglycemia with documentation of at least one of the following:
      i. Recurrent level 2 hypoglycemic events with glucose less than 54 mg/dL that persist despite multiple attempts to adjust medications or modify the diabetes treatment plan; or
      ii. A history of one level 3 hypoglycemic event with glucose less than 54 mg/dL characterized by altered mental or physical state requiring third-party assistance.

5. Within six months prior to ordering the CGM, the treating practitioner has an in-person or Medicare-approved telehealth visit with the beneficiary to evaluate diabetes control and determine that criteria (1)-(4) above are met.
"""


FALLBACK_LLM_RULES = {
    "rules": [
        {
            "rule_id": "CGM_coverage_insulin_1",
            "service": "continuous_glucose_monitor",
            "coverage": "covered",
            "conditions": [
                "diagnosis:diabetes_mellitus",
                "training_documented:true",
                "fda_indication:true",
                "insulin_treated:true",
                "six_month_visit:true",
            ],
            "exclusions": [],
            "source_hint": "CMS CGM initial coverage criteria, insulin-treated pathway",
        },
        {
            "rule_id": "CGM_coverage_hypo_level2_2",
            "service": "continuous_glucose_monitor",
            "coverage": "covered",
            "conditions": [
                "diagnosis:diabetes_mellitus",
                "training_documented:true",
                "fda_indication:true",
                "hypoglycemia_level2_recurrent:true",
                "attempts_adjustment_multiple:true",
                "six_month_visit:true",
            ],
            "exclusions": [],
            "source_hint": "CMS CGM initial coverage criteria, recurrent level 2 hypoglycemia pathway",
        },
        {
            "rule_id": "CGM_coverage_hypo_level3_3",
            "service": "continuous_glucose_monitor",
            "coverage": "covered",
            "conditions": [
                "diagnosis:diabetes_mellitus",
                "training_documented:true",
                "fda_indication:true",
                "hypoglycemia_level3_event:true",
                "third_party_assistance:true",
                "six_month_visit:true",
            ],
            "exclusions": [],
            "source_hint": "CMS CGM initial coverage criteria, level 3 hypoglycemia pathway",
        },
        {
            "rule_id": "CGM_injected_age_error",
            "service": "continuous_glucose_monitor",
            "coverage": "covered",
            "conditions": ["age_ge:65"],
            "exclusions": [],
            "source_hint": "Injected unsupported rule for gate demonstration",
        },
    ]
}


FALLBACK_CLAIMS = [
    {
        "claim_id": "CMS-C1",
        "service": "continuous glucose monitor",
        "attributes": {
            "diagnosis": "diabetes_mellitus",
            "training_documented": "true",
            "fda_indication": "true",
            "insulin_treated": "true",
            "six_month_visit": "true",
            "age": 58,
        },
    },
    {
        "claim_id": "CMS-C2",
        "service": "continuous glucose monitor",
        "attributes": {
            "diagnosis": "diabetes_mellitus",
            "training_documented": "true",
            "fda_indication": "true",
            "hypoglycemia_level3_event": "true",
            "third_party_assistance": "true",
            "six_month_visit": "true",
            "age": 44,
        },
    },
    {
        "claim_id": "CMS-C3",
        "service": "continuous glucose monitor",
        "attributes": {
            "diagnosis": "diabetes_mellitus",
            "training_documented": "true",
            "fda_indication": "true",
            "insulin_treated": "true",
            "six_month_visit": "false",
            "age": 70,
        },
    },
]


def _read_text_or_fallback(path: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return FALLBACK_CMS_POLICY


def _read_json_or_fallback(path: str, fallback):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return fallback


def _to_rules(payload: dict) -> list[Rule]:
    out: list[Rule] = []
    for item in payload.get("rules", []):
        rule = Rule(
            rule_id=item["rule_id"],
            service=item["service"].replace("_", " "),
            coverage=Coverage(item["coverage"]),
            conditions=list(item.get("conditions", [])),
            exclusions=list(item.get("exclusions", [])),
            source_hint=item.get("source_hint", ""),
        )
        rule.claim_sentence = item.get("source_hint") or f"{rule.service} {rule.coverage.value}"
        out.append(rule)
    return out


def _print_gate_results(results):
    print(BAR)
    print("LAYERED FAITHFULNESS GATE")
    print(BAR)
    for result in results:
        print(f"[{result.verdict.value:<12}] {result.rule_id}  {result.reason}")
        for claim_result in result.claim_results:
            span = claim_result.source_span
            score = span.score if span else 0.0
            evidence = span.text[:130] if span else ""
            print(f"  - {claim_result.claim.claim_id:<28} {claim_result.verdict.value:<12} score={score}")
            print(f"    claim   : {claim_result.claim.text}")
            print(f"    evidence: {evidence}")
            if claim_result.consistency and claim_result.consistency.risk_flags:
                print(f"    risks   : {'; '.join(claim_result.consistency.risk_flags)}")
            if claim_result.adversarial:
                print(f"    judge   : {claim_result.adversarial.judge_reasoning}")
        print()


def _print_human_review(queue):
    print(BAR)
    print("HUMAN REVIEW QUEUE")
    print(BAR)
    if not queue:
        print("No rules or claims require human review.")
        return
    for item in queue:
        print(f"[{item.verdict.value:<12}] {item.rule_id} / {item.claim_id}")
        print(f"  reason : {item.reason}")
        print(f"  claim  : {item.claim_text}")
        print(f"  source : {item.evidence[:160]}")


def _print_completeness(audit):
    print(BAR)
    print("COMPLETENESS GATE")
    print(BAR)
    print(f"criteria inventory : {len(audit.criteria)} source criteria")
    print(f"covered criteria   : {', '.join(audit.covered_criteria) or '(none)'}")
    print(f"missing criteria   : {', '.join(audit.missing_criteria) or '(none)'}")
    print(f"verdict            : {audit.verdict}")
    if audit.covered_alternatives:
        for group, covered in audit.covered_alternatives.items():
            print(f"alternative {group:<6}: {', '.join(covered) or '(none)'}")
    if audit.unmapped_conditions:
        print("unmapped generated conditions:")
        for rule_id, tokens in audit.unmapped_conditions.items():
            print(f"  - {rule_id}: {', '.join(tokens)}")
    if audit.notes:
        print("notes:")
        for note in audit.notes:
            print(f"  - {note}")


def _claim_trace(claim_result) -> dict:
    span = claim_result.source_span
    return {
        "claim_id": claim_result.claim.claim_id,
        "condition_token": claim_result.claim.condition_token,
        "claim": claim_result.claim.text,
        "verdict": claim_result.verdict.value,
        "reason": claim_result.reason,
        "evidence": span.text if span else "",
        "evidence_score": span.score if span else 0.0,
        "consistency_risk_flags": (
            claim_result.consistency.risk_flags if claim_result.consistency else []
        ),
        "semantic_confidence": (
            claim_result.semantic.confidence if claim_result.semantic else None
        ),
        "adversarial_judge_reasoning": (
            claim_result.adversarial.judge_reasoning if claim_result.adversarial else ""
        ),
    }


def _write_outputs(policy_text: str, rules: list[Rule], gate_results, queue, completeness_audit, output_dir: str):
    """Write final workflow artifacts for the policy-to-rule pipeline."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_rule = {r.rule_id: r for r in rules}

    reliable_rules = []
    for result in gate_results:
        if not result.passed:
            continue
        rule = by_rule[result.rule_id]
        reliable_rules.append(
            {
                "rule_id": rule.rule_id,
                "service": rule.service,
                "coverage": rule.coverage.value,
                "conditions": rule.conditions,
                "exclusions": rule.exclusions,
                "source_hint": rule.source_hint,
                "gate_verdict": result.verdict.value,
                "evidence_trace": [_claim_trace(c) for c in result.claim_results],
            }
        )

    reliable_payload = {
        "policy_id": "CMS_LCD_CGM_EXCERPT",
        "source": "Public CMS CGM coverage policy excerpt",
        "healthcare_policy_metadata": HEALTHCARE_POLICY_METADATA,
        "policy_characters": len(policy_text),
        "completeness_verdict": completeness_audit.verdict,
        "rules": reliable_rules,
    }

    review_payload = {
        "policy_id": "CMS_LCD_CGM_EXCERPT",
        "reviewer_role": HEALTHCARE_POLICY_METADATA["reviewer_role"],
        "items": [
            {
                "rule_id": item.rule_id,
                "claim_id": item.claim_id,
                "verdict": item.verdict.value,
                "reason": item.reason,
                "claim": item.claim_text,
                "evidence": item.evidence,
                "recommended_action": item.recommended_action,
            }
            for item in queue
        ],
    }

    inventory_payload = {
        "policy_id": "CMS_LCD_CGM_EXCERPT",
        "healthcare_policy_metadata": HEALTHCARE_POLICY_METADATA,
        "criteria": [
            {
                "criterion_id": c.criterion_id,
                "text": c.text,
                "role": c.role,
                "healthcare_primitive": c.healthcare_primitive,
                "condition_tokens": c.condition_tokens,
                "required_in_all_rules": c.required_in_all_rules,
                "alternative_group": c.alternative_group,
                "clinical_concepts": c.clinical_concepts,
                "documentation_requirements": c.documentation_requirements,
                "reviewer_role": c.reviewer_role,
                "source_span": c.source_span,
            }
            for c in completeness_audit.criteria
        ],
    }

    completeness_payload = {
        "policy_id": "CMS_LCD_CGM_EXCERPT",
        "verdict": completeness_audit.verdict,
        "covered_criteria": completeness_audit.covered_criteria,
        "missing_criteria": completeness_audit.missing_criteria,
        "covered_alternatives": completeness_audit.covered_alternatives,
        "missing_alternative_groups": completeness_audit.missing_alternative_groups,
        "unmapped_conditions": completeness_audit.unmapped_conditions,
        "notes": completeness_audit.notes,
    }

    report_lines = [
        "# Gate Report",
        "",
        "- Policy: CMS LCD CGM excerpt",
        f"- Domain: {HEALTHCARE_POLICY_METADATA['domain']}",
        f"- Policy type: {HEALTHCARE_POLICY_METADATA['policy_type']}",
        f"- Service category: {HEALTHCARE_POLICY_METADATA['service_category']}",
        f"- Source criteria: {len(completeness_audit.criteria)}",
        f"- Completeness verdict: {completeness_audit.verdict}",
        f"- Candidate rules: {len(rules)}",
        f"- Reliable rules: {len(reliable_rules)}",
        f"- Human review items: {len(queue)}",
        "",
        "## Completeness",
        "",
        f"- Covered criteria: {', '.join(completeness_audit.covered_criteria) or '(none)'}",
        f"- Missing criteria: {', '.join(completeness_audit.missing_criteria) or '(none)'}",
        "",
        "## Rule Verdicts",
        "",
    ]
    for result in gate_results:
        report_lines.append(f"- `{result.rule_id}`: **{result.verdict.value}** - {result.reason}")
    if queue:
        report_lines.extend(["", "## Human Review Queue", ""])
        for item in queue:
            report_lines.append(f"- `{item.rule_id}/{item.claim_id}`: {item.reason}")

    (out / "reliable_rules.json").write_text(
        json.dumps(reliable_payload, indent=2),
        encoding="utf-8",
    )
    (out / "human_review_queue.json").write_text(
        json.dumps(review_payload, indent=2),
        encoding="utf-8",
    )
    (out / "criteria_inventory.json").write_text(
        json.dumps(inventory_payload, indent=2),
        encoding="utf-8",
    )
    (out / "completeness_audit.json").write_text(
        json.dumps(completeness_payload, indent=2),
        encoding="utf-8",
    )
    (out / "gate_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "criteria_inventory": out / "criteria_inventory.json",
        "completeness_audit": out / "completeness_audit.json",
        "reliable_rules": out / "reliable_rules.json",
        "human_review_queue": out / "human_review_queue.json",
        "gate_report": out / "gate_report.md",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="data/cms_cgm_policy_clean.txt")
    parser.add_argument("--rules", default="data/llm_extracted_rules.json")
    parser.add_argument("--claims", default="data/cms_sample_claims.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--adversarial", action="store_true", help="Run OpenAI prosecutor-judge adjudication when OPENAI_API_KEY is set.")
    parser.add_argument("--evaluate", action="store_true", help="Write lightweight evaluation metrics for completeness, faithfulness, auditability, and execution gating.")
    args = parser.parse_args()

    policy_text = _read_text_or_fallback(args.policy)
    rules_payload = _read_json_or_fallback(args.rules, FALLBACK_LLM_RULES)
    claims = _read_json_or_fallback(args.claims, FALLBACK_CLAIMS)
    rules = _to_rules(rules_payload)

    print(BAR)
    print("CMS POLICY + LLM-EXTRACTED RULES")
    print(BAR)
    print(f"policy characters: {len(policy_text)}")
    print(f"rules loaded     : {len(rules)}")
    print(f"adversarial layer: {'enabled' if args.adversarial else 'off for deterministic demo'}")

    criteria = default_criteria_inventory(policy_text)
    completeness_audit = audit_completeness(rules, criteria)
    _print_completeness(completeness_audit)

    gate = LayeredFaithfulnessGate(policy_text, run_adversarial=args.adversarial)
    gate_results = gate.run_rules(rules)
    _print_gate_results(gate_results)

    queue = gate.human_review_queue(gate_results)
    _print_human_review(queue)

    output_paths = _write_outputs(
        policy_text,
        rules,
        gate_results,
        queue,
        completeness_audit,
        args.output_dir,
    )

    passed_ids = {r.rule_id for r in gate_results if r.passed}
    passed_rules = [r for r in rules if r.rule_id in passed_ids]
    decisions = adjudicate(claims, passed_rules)

    print("\n" + BAR)
    print("SYNTHETIC CLAIM EXECUTION (PASS RULES ONLY)")
    print(BAR)
    for decision in decisions:
        print(f"{decision.claim_id}: {decision.decision.upper():<16} {decision.rationale}")

    if args.evaluate:
        eval_report = evaluate_pipeline(
            rules=rules,
            gate_results=gate_results,
            completeness_audit=completeness_audit,
            human_review_queue=queue,
            decisions=decisions,
        )
        eval_paths = write_evaluation_outputs(eval_report, args.output_dir)
        output_paths.update(eval_paths)

        print("\n" + BAR)
        print("EVALUATION")
        print(BAR)
        for section, values in eval_report.metrics.items():
            print(section.upper())
            for key, value in values.items():
                print(f"  {key}: {value}")

    print("\n" + BAR)
    print("PIPELINE OUTPUT FILES")
    print(BAR)
    for label, path in output_paths.items():
        print(f"{label:<20} {path}")

    print("\n" + BAR)
    print(f"SUMMARY: {len(passed_rules)}/{len(rules)} rules passed; {len(queue)} claim(s) routed to human review.")
    print(BAR)


if __name__ == "__main__":
    main()
