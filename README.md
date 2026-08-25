# Faithfulness-Gated Healthcare Policy-to-Rule Pipeline

This repository is a proof-of-concept for **Content Management in Health Care**:
converting written healthcare coverage policy into auditable structured rules,
while preventing unsupported additions and silent omissions before any generated
rule can be used for claim adjudication.

The demo is framed around a payer **payment-integrity** workflow. It uses a
public CMS Continuous Glucose Monitor (CGM) coverage-policy excerpt, extracts
candidate coverage pathways, verifies them against the source policy, and writes
both a reliable rules file and a human-review queue.

## Why This Matters

Healthcare payment organizations routinely translate written policy, such as
CMS NCDs/LCDs, medical necessity policies, coding articles, and payer-provider
contracts, into operational rules. That translation is slow and high risk:

- An LLM may **invent** unsupported conditions, such as an age threshold.
- It may **omit** required criteria, such as documentation or time-window rules.
- It may mishandle **alternative pathways**, such as "at least one of" logic.
- Payment rules need an **audit trail** before they can be trusted.

This project treats LLM output as a proposal, not authority. A generated rule is
allowed to execute only after it passes completeness and faithfulness gates.

## Core Idea

```text
criteria inventory = source-grounded map of what the policy says
structured rules   = executable coverage pathways compiled from that map
```

The system first builds or uses a source-grounded **criteria inventory**: a map
of required criteria, alternative branches, thresholds, documentation
requirements, and time windows in the policy. It then checks whether LLM-created
structured rules cover that inventory and whether each generated rule condition
is faithful to the original policy text.

## Pipeline

```text
CMS policy text
  -> Source-grounded criteria inventory
  -> Schema-constrained LLM rule extraction
  -> Completeness audit
       checks whether source criteria and alternative branches are covered
  -> Layered faithfulness gate
       source-span retrieval
       evidence-scoped consistency checks
       semantic support checks
       optional adversarial prosecutor-judge review
  -> Outputs
       PASS rules -> reliable_rules.json
       REVIEW/FAIL/NEEDS_SOURCE -> human_review_queue.json
  -> Execution
       apply only PASS rules to synthetic claim cases
  -> Evaluation
       completeness, faithfulness, perturbation, auditability, execution gating
```

## Healthcare-Specific Design

The final CMS workflow includes healthcare policy metadata and domain-specific
policy primitives:

- **Domain**: healthcare payment integrity
- **Coverage domain**: durable medical equipment
- **Service category**: continuous glucose monitor
- **Policy type**: CMS LCD-style coverage criteria excerpt
- **Reviewer role**: payment-integrity policy analyst
- **Policy primitives**:
  - clinical condition
  - treatment status
  - device requirement
  - documentation requirement
  - clinical event threshold
  - time-window documentation requirement

The generated outputs preserve clinical concepts, documentation requirements,
source spans, condition tokens, gate verdicts, and reviewer-facing reasons.

## Repository Layout

```text
data/
  sample_policy.txt
  sample_claims.json

src/
  completeness.py              # criteria inventory + completeness audit
  evaluation.py                # lightweight evaluation metrics
  execute.py                   # synthetic claim execution using PASS rules only
  schema.py                    # shared rule and claim-decision schemas

  faithfulness_gate/
    atomic_claims.py           # structured rule -> atomic claim conversion
    retrieval.py               # source-span retrieval
    consistency.py             # evidence-scoped numeric/modal consistency checks
    semantic.py                # lightweight semantic support check
    adversarial.py             # optional prosecutor-judge adjudication
    aggregate.py               # claim-level -> rule-level verdict aggregation
    gate.py                    # layered gate orchestrator
    schema.py                  # PASS / REVIEW / FAIL / NEEDS_SOURCE schemas

run_demo.py                    # legacy offline baseline demo
run_cms_llm_demo.py            # final CMS workflow demo
```

## Quick Start

The final workflow has a deterministic fallback path, so it can run without API
keys:

```bash
python run_cms_llm_demo.py --evaluate
```

This uses the built-in CMS CGM policy excerpt, fallback extracted rules, a
synthetic unsupported rule for the main gate demonstration, a separate
perturbation stress test, and synthetic claim cases.

Expected high-level result:

```text
3/4 rules passed
1 unsupported rule routed to human review
outputs/reliable_rules.json
outputs/human_review_queue.json
outputs/evaluation_report.json
```

## Run With Adversarial Review

To enable the optional prosecutor-judge review layer, set `OPENAI_API_KEY` and
run:

```bash
export OPENAI_API_KEY="..."
python run_cms_llm_demo.py --adversarial --evaluate
```

The adversarial layer is conditional. It is triggered for high-risk or uncertain
atomic claims, not for every claim. This keeps the workflow efficient while
still escalating cases where deterministic and semantic checks are not enough.

## Run With LLM-Extracted Rules

If you have generated `data/llm_extracted_rules.json` from the CMS policy
excerpt, pass it explicitly:

```bash
python run_cms_llm_demo.py \
  --policy data/cms_cgm_policy_clean.txt \
  --rules data/llm_extracted_rules.json \
  --adversarial \
  --evaluate
```

The current proof of concept uses schema-constrained LLM extraction: the model
emits structured JSON rules with controlled condition tokens. Those rules still
must pass the completeness and faithfulness gates before execution.

## Outputs

The final workflow writes artifacts under `outputs/`:

```text
outputs/criteria_inventory.json
outputs/completeness_audit.json
outputs/reliable_rules.json
outputs/human_review_queue.json
outputs/gate_report.md
outputs/evaluation_report.json
outputs/evaluation_report.md
```

### `criteria_inventory.json`

A source-grounded map of policy criteria. Each item includes:

- criterion id
- source text / source span
- healthcare primitive
- condition token mapping
- required vs alternative role
- clinical concepts
- documentation requirements
- reviewer role

### `reliable_rules.json`

Only rules that pass the faithfulness gate are exported. Each reliable rule
includes its conditions, coverage action, source hint, gate verdict, and
claim-level evidence trace.

### `human_review_queue.json`

Rules or claims with `REVIEW`, `FAIL`, or `NEEDS_SOURCE` verdicts are routed to
a payment-integrity policy analyst with:

- rule id
- claim id
- verdict
- reason
- generated claim
- retrieved evidence
- recommended action

### `evaluation_report.json`

Reports lightweight reliability metrics:

- completeness coverage
- alternative branch coverage
- rule and claim pass rates
- perturbation stress-test routing and hard-fail rates
- false-positive rate on valid policy pathways
- evidence-trace coverage
- execution gating on synthetic claims

## Evaluation Dataset

This POC intentionally avoids patient-level data and PHI. Evaluation uses:

- a public CMS CGM coverage-policy excerpt
- a source-grounded criteria inventory
- synthetic unsupported-rule perturbations
- synthetic claim cases

This setup tests the reliability risks that matter for policy automation:

- missing source criteria
- unsupported additions
- numeric/time-window mismatches
- coverage polarity flips
- false positives on valid extracted rules
- unmapped generated conditions
- whether only PASS rules are allowed to execute

The perturbation suite is separate from the main extracted-rule run. It injects
small corrupted rules such as unsupported age or prior-authorization conditions,
a glucose-threshold mismatch, a covered/not-covered polarity flip, and an
omitted required criterion. The consistency layer checks for generic
source-evidence conflicts, such as a claim introducing a structured attribute
that the retrieved source span does not contain, instead of relying only on
one-off hard-coded error patterns.

The evaluation also runs valid positive-control pathways through the same gate
to estimate false positives: cases where a faithful rule would be incorrectly
blocked for review.

Production evaluation would expand to multiple CMS NCD/LCD policies, coding
articles, payer-specific medical necessity policies, and reviewer-labeled edge
cases.

## Example Evaluation Metrics

The fallback CMS demo produces metrics such as:

```text
criteria_coverage: 1.0
alternative_branch_coverage: 1.0
hard_fail_rate: 0.6
routed_to_review_or_fail_rate: 1.0
false_positive_rate: 0.0
evidence_trace_coverage: 1.0
```

These numbers are not intended as a broad benchmark. They demonstrate that the
workflow can measure completeness, faithfulness, auditability, and execution
control on a small auditable policy example. The perturbation metrics separate
confirmed FAIL decisions from REVIEW routing, because a healthcare policy system
can be useful even when some ambiguous or incomplete outputs are blocked for
human review rather than automatically labeled as hard failures.

## Legacy Offline Demo

The earlier minimal demo is still available:

```bash
python run_demo.py
```

It runs entirely with mock components and demonstrates the basic policy -> rule
-> gate -> execution path. The final CMS workflow is `run_cms_llm_demo.py`.

## Limitations

- The current criteria inventory is hand-built for the CMS CGM excerpt. A
  production system should generate and review inventories across many policies.
- The fallback semantic check is lightweight and deterministic. The optional
  adversarial layer provides stronger review when API access is available.
- Synthetic claim cases are used only to demonstrate execution gating; they are
  not real patient or claims data.
- The POC covers one policy area. Broader validation should include multiple
  CMS LCD/NCD policies, billing/coding articles, and domain-expert labels.

## Summary

This project demonstrates a reliable AI workflow for healthcare policy
automation:

```text
LLM extraction proposes rules.
Completeness gate checks for omissions.
Faithfulness gate checks for unsupported additions.
Human review handles uncertainty.
Only PASS rules are exported and executed.
```
