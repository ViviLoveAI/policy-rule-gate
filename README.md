# Healthcare Policy-to-Rule Pipeline with Completeness and Faithfulness Gates

A proof-of-concept for **Content Management in Health Care**: converting written
coverage policy into executable adjudication rules, with a source-grounded
criteria inventory, completeness audit, layered faithfulness gate, and human
review queue before any rule can execute.

> Payment-integrity organizations turn written policy (CMS NCDs/LCDs, medical
> necessity policies, payer-provider contracts) into the rules that decide what
> gets paid. That translation is manual, slow, and cannot be wrong. This POC
> treats LLM extraction as proposal generation, then verifies both completeness
> (did we miss source criteria?) and faithfulness (did we invent unsupported
> conditions?) before producing a reliable rules file.

## Healthcare-specific design

The final CMS demo targets a payer payment-integrity workflow for durable
medical equipment coverage criteria:

- **Policy source**: CMS CGM coverage criteria excerpt
- **Policy primitives**: clinical conditions, documentation requirements,
  device requirements, clinical event thresholds, time-window requirements
- **Reviewer role**: payment-integrity policy analyst
- **Outputs**: reliable structured rules, source-grounded criteria inventory,
  completeness audit, and human review queue

The core distinction is:

```
criteria inventory = source-grounded map of what the policy says
structured rules   = executable coverage pathways compiled from that map
```

## Pipeline

```
CMS policy text
  -> criteria inventory         source-grounded policy condition map
  -> LLM extraction             candidate structured rules
  -> completeness audit         checks that source criteria are covered
  -> layered faithfulness gate  source span + consistency + semantic/adversarial review
       |-- PASS                 reliable_rules.json
       |-- REVIEW/FAIL          human_review_queue.json
  -> execution                  apply only PASS rules to synthetic claims
```

## Run (offline, zero dependencies)

```bash
python run_demo.py
```

The demo runs entirely on bundled mock components (`MockLLMClient`,
`MockNLIDetector`, `LexicalRetriever`). It extracts four rules from the sample
CGM policy; three are grounded and pass the gate, while one (`R4`, an invented
"age ≥ 65" threshold the policy never states) is **blocked and routed to human
review** — demonstrating the gate catching a numeric soft-fabrication.

## Run the final CMS workflow

```bash
python run_cms_llm_demo.py
python run_cms_llm_demo.py --adversarial
python run_cms_llm_demo.py --policy data/cms_cgm_policy_clean.txt --rules data/llm_extracted_rules.json --adversarial
```

The final workflow writes:

```
outputs/criteria_inventory.json
outputs/completeness_audit.json
outputs/reliable_rules.json
outputs/human_review_queue.json
outputs/gate_report.md
```

## Going live

Every real integration is a clearly marked seam. Swap three components:

| Stage | Mock (default) | Real seam |
|-------|----------------|-----------|
| Extraction | `MockLLMClient` | `OpenAIClient` — set `OPENAI_API_KEY` (`src/llm.py`) |
| Retrieval  | `LexicalRetriever` | `BM25Retriever` — `pip install rank-bm25` (`src/retriever.py`) |
| Faithfulness | `MockNLIDetector` | `DebertaNLIDetector` — wire in the PHANTOM detector (`src/nli.py`) |

Then in `src/pipeline.py::build_pipeline`, pass `llm=OpenAIClient()` and
`detector=DebertaNLIDetector()`. No other stage changes.

## Known limitations (intentional POC scope)

- The mock NLI is a lexical heuristic keyed on numeric/threshold specifiers, not
  a real entailment model. It stands in for the DeBERTa detector via the same
  interface; it does **not** catch evaluative/inferential soft-fabrications.
- The execution engine matches on the first applicable rule (no conjunction of
  all applicable rules), so a claim can match a coverage rule without satisfying
  a separate prior-authorization rule.
- Single policy, single service line, hand-built sample claims.

## Layout

```
data/    sample_policy.txt, sample_claims.json
src/     schema, llm, retriever, nli, extract, verbalize, gate, execute, pipeline
run_demo.py
```
