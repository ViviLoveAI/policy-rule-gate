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

from .completeness import CompletenessAudit, audit_completeness
from .faithfulness_gate import LayeredFaithfulnessGate
from .faithfulness_gate.schema import ClaimVerdict, RuleGateResult
from .schema import ClaimDecision, Coverage, Rule


@dataclass
class EvaluationReport:
    metrics: dict
    notes: list[str]


@dataclass
class PerturbationResult:
    case_id: str
    perturbation_type: str
    verdict: str
    routed: bool
    hard_fail: bool
    reason: str


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _perturbation_rules() -> list[Rule]:
    """Rules intentionally corrupted for pressure-testing the gate."""

    return [
        Rule(
            rule_id="PERT_age_threshold",
            service="continuous glucose monitor",
            coverage=Coverage.COVERED,
            conditions=["age_ge:65"],
            source_hint="Perturbation: unsupported age threshold",
        ),
        Rule(
            rule_id="PERT_prior_authorization",
            service="continuous glucose monitor",
            coverage=Coverage.COVERED,
            conditions=["prior_auth:approved"],
            source_hint="Perturbation: unsupported prior authorization condition",
        ),
        Rule(
            rule_id="PERT_numeric_threshold",
            service="continuous glucose monitor",
            coverage=Coverage.COVERED,
            conditions=["hypoglycemia_level2_glucose_lt_70:true"],
            source_hint="Perturbation: glucose threshold changed from 54 to 70 mg/dL",
        ),
        Rule(
            rule_id="PERT_polarity_flip",
            service="continuous glucose monitor",
            coverage=Coverage.NOT_COVERED,
            conditions=[],
            source_hint="Perturbation: coverage polarity flipped",
        ),
    ]


def _omission_rules() -> list[Rule]:
    """A pathway that omits a required criterion from the criteria inventory."""

    return [
        Rule(
            rule_id="PERT_omitted_six_month_visit",
            service="continuous glucose monitor",
            coverage=Coverage.COVERED,
            conditions=[
                "diagnosis:diabetes_mellitus",
                "training_documented:true",
                "fda_indication:true",
                "insulin_treated:true",
            ],
            source_hint="Perturbation: required six-month visit omitted",
        )
    ]


def _gate_reason(result: RuleGateResult) -> str:
    reasons = [
        claim.reason
        for claim in result.claim_results
        if claim.verdict != ClaimVerdict.PASS and claim.reason
    ]
    return "; ".join(reasons) or result.reason


def _run_perturbation_suite(policy_text: str) -> list[PerturbationResult]:
    """Run a small synthetic stress test without calling external APIs."""

    gate = LayeredFaithfulnessGate(policy_text, run_adversarial=False)
    gate_results = gate.run_rules(_perturbation_rules())
    results = [
        PerturbationResult(
            case_id=result.rule_id,
            perturbation_type=result.rule_id.removeprefix("PERT_"),
            verdict=result.verdict.value,
            routed=result.verdict != ClaimVerdict.PASS,
            hard_fail=result.verdict == ClaimVerdict.FAIL,
            reason=_gate_reason(result),
        )
        for result in gate_results
    ]

    omission_audit = audit_completeness(_omission_rules())
    results.append(
        PerturbationResult(
            case_id="PERT_required_criterion_omission",
            perturbation_type="required_criterion_omission",
            verdict=omission_audit.verdict,
            routed=omission_audit.verdict != "PASS",
            hard_fail=False,
            reason="; ".join(omission_audit.notes),
        )
    )
    return results


def evaluate_pipeline(
    rules: list[Rule],
    gate_results: list[RuleGateResult],
    completeness_audit: CompletenessAudit,
    human_review_queue: list,
    decisions: list[ClaimDecision],
    policy_text: str | None = None,
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
    perturbations = _run_perturbation_suite(policy_text) if policy_text else []
    routed_perturbations = [p for p in perturbations if p.routed]
    hard_failed_perturbations = [p for p in perturbations if p.hard_fail]

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
            "stress_test_cases": len(perturbations),
            "stress_test_hard_failures": len(hard_failed_perturbations),
            "stress_test_routed_to_review_or_fail": len(routed_perturbations),
            "hard_fail_rate": _ratio(len(hard_failed_perturbations), len(perturbations)),
            "routed_to_review_or_fail_rate": _ratio(len(routed_perturbations), len(perturbations)),
            "test_cases": [
                {
                    "case_id": p.case_id,
                    "perturbation_type": p.perturbation_type,
                    "verdict": p.verdict,
                    "routed": p.routed,
                    "hard_fail": p.hard_fail,
                    "reason": p.reason,
                }
                for p in perturbations
            ],
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
        "Perturbation metrics distinguish hard FAIL decisions from REVIEW routing; both prevent unsafe rules from executing.",
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
