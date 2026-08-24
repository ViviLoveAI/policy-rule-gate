# Policy-to-Rule Pipeline with a Faithfulness Gate

A proof-of-concept for **Content Management in Health Care**: converting written
coverage policy into executable adjudication rules, with an NLI-based
**faithfulness gate** that blocks any generated rule not grounded in the source
policy before it can reach production.

> Payment-integrity organizations turn written policy (CMS NCDs/LCDs, medical
> necessity policies, payer-provider contracts) into the rules that decide what
> gets paid. That translation is manual, slow, and cannot be wrong. This POC
> automates it with an LLM and gates the output on faithfulness: every rule is
> verified against the policy text, and unverified rules are routed to human
> review with an auditable evidence trail.

## Pipeline

```
policy text
  -> [1] extract      LLM -> structured rules                (src/extract.py)
  -> [2] verbalize    each rule -> one NL claim sentence      (src/verbalize.py)
  -> [3] gate         retrieve source + NLI judge -> pass/block (src/gate.py)
         |                                          \--blocked--> human review
  -> [4] execute      apply passed rules to sample claims      (src/execute.py)
  -> demo output      policy -> rules (pass/block) -> decisions
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
