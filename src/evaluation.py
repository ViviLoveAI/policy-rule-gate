"""Evaluation layer for the CMS policy-to-rule POC.

The evaluation is intentionally small and auditable: it uses a public CMS
policy excerpt, a source-grounded criteria inventory, synthetic perturbation
rules, and synthetic claim cases. It measures the reliability properties this
POC is designed to demonstrate rather than claiming clinical generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .completeness import CompletenessAudit
from .faithfulness_gate.schema import ClaimVerdict, RuleGateResult
from .schema import ClaimDecision, Rule


@dataclass
class EvaluationReport:
    metrics: dict
    notes: list[str]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def evaluate_pipeline(
    rules: list[Rule],
    gate_results: list[RuleGateResult],
    completeness_audit: CompletenessAudit,
    human_review_queue: list,
    decisions: list[ClaimDecision],
) -> EvaluationReport:
    """Compute lightweight reliability metrics for the POC."""

    total_criteria = len(completeness_audit.criteria)
    covered_criteria = len(completeness_audit.covered_criteria)
    alternative_total = sum(
        len(v) for v in completeness_audit.covered_alternatives.values()
    ) + len(completeness_audit.missing_alternative_groups)
    alternative_covered = sum(
        len(v) for v in completeness_audit.covered_alternatives.values()
    )

    all_claim_results = [c for r in gate_results for c in r.claim_results]
    pass_rules = [r for r in gate_results if r.verdict == ClaimVerdict.PASS]
    blocked_rules = [r for r in gate_results if r.verdict != ClaimVerdict.PASS]
    passed_claims = [c for c in all_claim_results if c.verdict == ClaimVerdict.PASS]
    non_pass_claims = [c for c in all_claim_results if c.verdict != ClaimVerdict.PASS]

    injected_rules = [
        r for r in gate_results
        if "injected" in r.rule_id.lower() or "error" in r.rule_id.lower()
    ]
    detected_injected = [r for r in injected_rules if r.verdict != ClaimVerdict.PASS]

    reliable_with_trace = [
        r for r in pass_rules
        if r.claim_results and all(c.source_span and c.source_span.text for c in r.claim_results)
    ]
    review_with_reason_evidence = [
        item for item in human_review_queue
        if item.reason and item.evidence
    ]
    executed_decisions = [d for d in decisions if d.decision != "no_matching_rule"]

    metrics = {
        "completeness": {
            "criteria_coverage": _ratio(covered_criteria, total_criteria),
            "covered_criteria": covered_criteria,
            "total_criteria": total_criteria,
            "missing_criteria_count": len(completeness_audit.missing_criteria),
            "alternative_branch_coverage": _ratio(alternative_covered, alternative_total),
            "covered_alternative_branches": alternative_covered,
            "total_alternative_branches": alternative_total,
            "unmapped_condition_count": sum(
                len(v) for v in completeness_audit.unmapped_conditions.values()
            ),
            "verdict": completeness_audit.verdict,
        },
        "faithfulness": {
            "candidate_rules": len(rules),
            "passed_rules": len(pass_rules),
            "blocked_or_review_rules": len(blocked_rules),
            "rule_pass_rate": _ratio(len(pass_rules), len(gate_results)),
            "atomic_claims": len(all_claim_results),
            "passed_atomic_claims": len(passed_claims),
            "non_pass_atomic_claims": len(non_pass_claims),
            "claim_pass_rate": _ratio(len(passed_claims), len(all_claim_results)),
            "human_review_items": len(human_review_queue),
        },
        "perturbation": {
            "injected_unsupported_rules": len(injected_rules),
            "detected_injected_unsupported_rules": len(detected_injected),
            "injected_detection_rate": _ratio(len(detected_injected), len(injected_rules)),
        },
        "auditability": {
            "reliable_rules_with_evidence_trace": len(reliable_with_trace),
            "reliable_rules": len(pass_rules),
            "evidence_trace_coverage": _ratio(len(reliable_with_trace), len(pass_rules)),
            "human_review_items_with_reason_and_evidence": len(review_with_reason_evidence),
            "human_review_item_trace_coverage": _ratio(
                len(review_with_reason_evidence), len(human_review_queue)
            ),
        },
        "execution_gating": {
            "synthetic_claim_cases": len(decisions),
            "claims_with_gate_approved_rule_match": len(executed_decisions),
            "claims_without_gate_approved_rule_match": len(decisions) - len(executed_decisions),
        },
    }

    notes = [
        "Evaluation uses public CMS policy text and synthetic perturbation/claim cases; no PHI is used.",
        "Completeness metrics evaluate coverage against a source-grounded criteria inventory.",
        "Faithfulness metrics evaluate whether generated rule claims pass the layered gate before execution.",
        "Production evaluation should expand to multiple NCD/LCD policies and reviewer-labeled edge cases.",
    ]
    return EvaluationReport(metrics=metrics, notes=notes)


def write_evaluation_outputs(report: EvaluationReport, output_dir: str) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "evaluation_report.json"
    md_path = out / "evaluation_report.md"

    payload = {"metrics": report.metrics, "notes": report.notes}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Evaluation Report", ""]
    for section, values in report.metrics.items():
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        for key, value in values.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in report.notes:
        lines.append(f"- {note}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"evaluation_report": json_path, "evaluation_summary": md_path}
