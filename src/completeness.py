"""Completeness gate for policy-to-rule extraction.

Faithfulness catches unsupported additions. Completeness catches silent
omissions by checking generated rules against a source-grounded criteria
inventory: a map of required criteria, alternative branches, thresholds, and
documentation requirements stated in the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Rule


@dataclass
class Criterion:
    criterion_id: str
    text: str
    role: str
    healthcare_primitive: str
    condition_tokens: list[str]
    required_in_all_rules: bool = False
    alternative_group: str | None = None
    clinical_concepts: list[str] = field(default_factory=list)
    documentation_requirements: list[str] = field(default_factory=list)
    reviewer_role: str = "payment_integrity_policy_analyst"
    source_span: str = ""


@dataclass
class CompletenessAudit:
    verdict: str
    criteria: list[Criterion]
    covered_criteria: list[str] = field(default_factory=list)
    missing_criteria: list[str] = field(default_factory=list)
    covered_alternatives: dict[str, list[str]] = field(default_factory=dict)
    missing_alternative_groups: list[str] = field(default_factory=list)
    unmapped_conditions: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


CMS_CGM_CRITERIA = [
    Criterion(
        criterion_id="C1",
        text="The beneficiary has diabetes mellitus.",
        role="required",
        healthcare_primitive="clinical_condition",
        condition_tokens=["diagnosis:diabetes_mellitus"],
        required_in_all_rules=True,
        clinical_concepts=["diabetes mellitus"],
        source_span="The beneficiary has diabetes mellitus.",
    ),
    Criterion(
        criterion_id="C2",
        text="The practitioner documents sufficient CGM training by prescription.",
        role="required",
        healthcare_primitive="documentation_requirement",
        condition_tokens=["training_documented:true"],
        required_in_all_rules=True,
        documentation_requirements=["prescription", "training documentation"],
        source_span="The beneficiary's treating practitioner has concluded ... sufficient training ... as evidenced by providing a prescription.",
    ),
    Criterion(
        criterion_id="C3",
        text="The CGM is prescribed according to FDA indications.",
        role="required",
        healthcare_primitive="device_requirement",
        condition_tokens=["fda_indication:true"],
        required_in_all_rules=True,
        clinical_concepts=["continuous glucose monitor"],
        source_span="The CGM is prescribed in accordance with its FDA indications for use.",
    ),
    Criterion(
        criterion_id="C4a",
        text="The beneficiary is insulin-treated.",
        role="alternative",
        healthcare_primitive="treatment_status",
        condition_tokens=["insulin_treated:true"],
        alternative_group="C4",
        clinical_concepts=["insulin treatment", "diabetes mellitus"],
        source_span="The beneficiary is insulin-treated.",
    ),
    Criterion(
        criterion_id="C4b_i",
        text="The beneficiary has recurrent level 2 hypoglycemic events despite multiple treatment adjustments.",
        role="alternative",
        healthcare_primitive="clinical_event_threshold",
        condition_tokens=["hypoglycemia_level2_recurrent:true", "attempts_adjustment_multiple:true"],
        alternative_group="C4",
        clinical_concepts=["level 2 hypoglycemia", "glucose <54 mg/dL"],
        documentation_requirements=["documented hypoglycemia history", "treatment adjustment attempts"],
        source_span="Recurrent level 2 hypoglycemic events ... that persist despite multiple attempts to adjust medications or modify the diabetes treatment plan.",
    ),
    Criterion(
        criterion_id="C4b_ii",
        text="The beneficiary has one level 3 hypoglycemic event requiring third-party assistance.",
        role="alternative",
        healthcare_primitive="clinical_event_threshold",
        condition_tokens=["hypoglycemia_level3_event:true", "third_party_assistance:true"],
        alternative_group="C4",
        clinical_concepts=["level 3 hypoglycemia", "glucose <54 mg/dL", "third-party assistance"],
        documentation_requirements=["documented hypoglycemia history"],
        source_span="A history of one level 3 hypoglycemic event ... requiring third-party assistance.",
    ),
    Criterion(
        criterion_id="C5",
        text="The practitioner had a qualifying visit within six months before ordering the CGM.",
        role="required",
        healthcare_primitive="time_window_documentation_requirement",
        condition_tokens=["six_month_visit:true"],
        required_in_all_rules=True,
        documentation_requirements=["in-person visit", "Medicare-approved telehealth visit", "diabetes control evaluation"],
        source_span="Within six months prior to ordering the CGM ... visit ... to evaluate diabetes control and determine that criteria (1)-(4) are met.",
    ),
]


HEALTHCARE_POLICY_METADATA = {
    "domain": "healthcare_payment_integrity",
    "policy_type": "LCD coverage criteria excerpt",
    "coverage_domain": "durable_medical_equipment",
    "service_category": "continuous_glucose_monitor",
    "source_authority": "CMS Medicare Coverage Database",
    "reviewer_role": "payment_integrity_policy_analyst",
    "clinical_concepts": [
        "diabetes mellitus",
        "continuous glucose monitor",
        "insulin treatment",
        "level 2 hypoglycemia",
        "level 3 hypoglycemia",
    ],
    "documentation_requirements": [
        "prescription",
        "sufficient training documentation",
        "FDA indications for use",
        "six-month practitioner visit",
        "hypoglycemia documentation",
    ],
    "coding_relevance": [
        "ICD-10 diagnosis list referenced in LCD-related Policy Article",
        "coverage criteria can support downstream coding and payment review",
    ],
}


def default_criteria_inventory(policy_text: str) -> list[Criterion]:
    """Return a source-grounded criteria inventory for the CMS CGM demo policy."""

    return CMS_CGM_CRITERIA


def _criterion_covered(rule: Rule, criterion: Criterion) -> bool:
    rule_conditions = set(rule.conditions)
    return set(criterion.condition_tokens).issubset(rule_conditions)


def audit_completeness(
    rules: list[Rule],
    criteria: list[Criterion] | None = None,
) -> CompletenessAudit:
    """Audit whether generated rules cover the source criteria inventory."""

    criteria = criteria or CMS_CGM_CRITERIA
    covered_criteria: set[str] = set()
    missing_criteria: list[str] = []
    notes: list[str] = []

    for criterion in criteria:
        if any(_criterion_covered(rule, criterion) for rule in rules):
            covered_criteria.add(criterion.criterion_id)
        else:
            missing_criteria.append(criterion.criterion_id)

        if criterion.required_in_all_rules:
            missing_from = [
                rule.rule_id for rule in rules if not _criterion_covered(rule, criterion)
            ]
            if missing_from:
                notes.append(
                    f"{criterion.criterion_id} is required but missing from rule(s): {', '.join(missing_from)}"
                )

    alternative_groups = sorted({c.alternative_group for c in criteria if c.alternative_group})
    covered_alternatives: dict[str, list[str]] = {}
    missing_alternative_groups: list[str] = []
    for group in alternative_groups:
        group_criteria = [c for c in criteria if c.alternative_group == group]
        covered = [
            c.criterion_id
            for c in group_criteria
            if any(_criterion_covered(rule, c) for rule in rules)
        ]
        covered_alternatives[group] = covered
        if not covered:
            missing_alternative_groups.append(group)
            notes.append(f"Alternative group {group} has no covered pathway.")

    known_tokens = {tok for criterion in criteria for tok in criterion.condition_tokens}
    unmapped_conditions = {
        rule.rule_id: [token for token in rule.conditions if token not in known_tokens]
        for rule in rules
    }
    unmapped_conditions = {
        rule_id: tokens for rule_id, tokens in unmapped_conditions.items() if tokens
    }
    for rule_id, tokens in unmapped_conditions.items():
        notes.append(f"{rule_id} contains condition(s) not mapped to inventory: {', '.join(tokens)}")

    verdict = "PASS"
    if missing_criteria or missing_alternative_groups:
        verdict = "REVIEW"
    if unmapped_conditions:
        verdict = "REVIEW"

    return CompletenessAudit(
        verdict=verdict,
        criteria=criteria,
        covered_criteria=sorted(covered_criteria),
        missing_criteria=missing_criteria,
        covered_alternatives=covered_alternatives,
        missing_alternative_groups=missing_alternative_groups,
        unmapped_conditions=unmapped_conditions,
        notes=notes,
    )
