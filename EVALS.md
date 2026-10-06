# Evaluation suite

Retrieval quality was already measured in this repo (hit@k, MRR on 25 programmatically
generated questions). This suite adds the other half: **is the generated answer
faithful, correct, cited, and does the system refuse when it should?** It is structured
the way a release gate should be: cheap deterministic checks first, an LLM judge for the
fuzzy questions, and a regression comparison against a saved baseline.

## What runs

| Stage | What it measures | How | Cost |
|---|---|---|---|
| Retrieval | hit@5, MRR, context recall@3 | deterministic, against the known source filing | no LLM |
| End-to-end RAG | citations valid, cites the right filing, number lock, key facts present | deterministic (`evals/checks.py`) | generator only |
| End-to-end RAG | faithfulness, answer relevancy, correctness | DeepEval metrics with an LLM judge | judge |
| Generator in isolation | faithfulness when handed the golden filing | DeepEval, golden context, so a low score is purely the generator's fault | judge |
| Refusal | declines 10 adversarial questions (near-duplicate company names, unknown companies, facts no filing contains, wrong year, off topic) | deterministic | generator only |

**Pipeline under test:** weighted hybrid retrieval (0.2 dense / 0.8 BM25, the best
config from the retrieval experiments), an **entity metadata pre-filter** (a question that
names a known company searches only that company's filings), parent-document expansion
(small chunks to match, whole filings to answer), and **Meta Llama 3.1 8B on Amazon Bedrock**
for generation at temperature 0.

**Judge:** **Amazon Nova Pro on Amazon Bedrock**, deliberately a different model family from
the generator to avoid self-preference bias. Temperature 0, structured JSON output validated
against DeepEval's schemas, exponential backoff with jitter for Bedrock throttling, and token
usage recorded per run. Backends are pluggable (`evals/llm.py`: Bedrock, Hugging Face, Claude
CLI) via `EVAL_GENERATOR` / `EVAL_JUDGE`.

**Tracking:** every run is logged to **MLflow** (params, metrics, per-question tables, verdict,
git commit): `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.

## Ground truth

- `goldens/rag_goldens.json`: the same 25 questions as the retrieval experiments. The
  expected answer is **extracted from the source filing with a regex**, never written by
  hand or by an LLM, so ground truth is exact.
- `goldens/refusal_goldens.json`: 10 adversarial questions the system must decline. For
  any data product, a confident answer about the wrong entity is the worst failure.

## Deterministic checks (zero LLM calls)

- **Citation check:** every `[doc]` cited must be a document the model was actually given.
- **Number lock:** every number in the answer must appear in its context. This catches
  invented or "computed" figures mechanically.
- **Key facts:** the ground-truth numbers must all appear in the answer.

## Regression gate (`evals/registry.py`)

| Kind | Metrics | Rule |
|---|---|---|
| **Gate** (blocks release) | refusal rate, number lock, valid citations, key facts | exact metrics, zero tolerance |
| **Guardrail** (human review) | judge average scores | tolerance above the measured judge noise floor |
| **Guardrail** | generation p95 latency | 30% relative tolerance |
| Info | pass rates, token counts, judge calls | tracked only |

Judge metrics are gated on the **average score**, not the pass rate: pass rate is
threshold-anchored and swings wildly when scores cluster near the line.

## Judge calibration and noise

- `python -m evals.calibrate export`, then hand-label the CSV, then
  `python -m evals.calibrate score`: agreement and Cohen's kappa between the judge and a
  human, and between the deterministic key-fact check and a human.
- `python -m evals.noise_floor`: re-judges the same answers and reports how far each judge
  metric moves on identical input. That number sets the guardrail tolerance.

## Run it

```bash
python goldens/build_goldens.py
python -m evals.run_suite --skip-judge     # deterministic stages only
python -m evals.run_suite --save-baseline  # full run, accept as baseline
python -m evals.run_suite                  # full run, compare, exit 1 on gate regression
pytest tests/test_evals_offline.py -q      # tests for the eval code itself
```

CI (`.github/workflows/evals.yml`): offline tests on every push; the full suite on demand,
failing the job if any gate regresses.

## Results (Oct 5, 2026: Llama 3.1 8B generator, Nova Pro judge, both on Bedrock)

| Metric | Baseline | + Entity filter | + Correctness rubric v2 |
|---|---|---|---|
| Context recall@3 | 80% | **100%** | 100% |
| MRR | 0.810 | **1.000** | 1.000 |
| Key facts correct (end to end) | 80% | **100%** | 100% |
| Key facts correct (generator alone, golden filing) | 100% | 100% | 100% |
| False refusals on answerable questions | 20% | **0%** | 0% |
| Refusal rate, 10 adversarial questions (gate) | 100% | 100% | 100% |
| Number lock (gate) | 100% | 100% | 100% |
| Valid citations (gate) | 72% | 88% | 88% |
| Correctness, judge avg / pass rate | 0.672 / 72% | 0.863 / 92% | 0.840 / **100%** |
| Faithfulness, judge avg | 0.750 | 0.900 | 0.880 |
| Answer relevancy, judge avg | 0.880 | 0.990 | 0.990 |
| Generation latency p95 | 645 ms | 417 ms | 517 ms |
| Judge errors (throttling) | 1 | 3 | **0** (backoff + 3 workers) |
| Regression verdict | baseline | REVIEW | PASS |

### What the per-question results showed

1. **The generator was never the problem.** Given the right filing it got 100% of key facts.
   Every end-to-end miss was retrieval handing it a lookalike company (e.g. *Silverton Capital
   Partners'* filing, 11%, for a question about *Silverton Financial Corp*, 21%), and in
   every such case the model **refused instead of reporting the wrong company's number**.
2. **The v1 correctness rubric was wrong.** It told the judge to penalize "any number that
   differs from the expected output", which it read as "any extra number": complete, correct
   merger answers that also stated the true deal value scored 0.3. v2 penalizes only missing
   or contradicted facts; merger answers now score 1.0 and the correctness pass rate is 100%.
   The rubric version is logged as an MLflow param so scores are never compared across rubrics.
3. **Faithfulness is undefined for refusals.** The same kind of refusal scored 1.0 in a smoke
   test and 0.0 in the baseline. `rag.faithfulness_answered` scores answered questions only.
4. **Judge noise is real and measured.** Between runs 2 and 3, 24/25 generator answers were
   byte-identical, yet faithfulness changed on 5 of them. Re-judging identical answers:

   | Judge metric | Mean moved | Per-case max | Pass/fail flips |
   |---|---|---|---|
   | Faithfulness | 0.040 | 1.0 | 3/25 |
   | Correctness (GEval v2) | 0.016 | 0.10 | 2/25 |
   | Answer relevancy | 0.010 | 0.25 | 0/25 |

   Guardrail tolerances are set per metric from these numbers (`evals/registry.py`).

### Next
- Merger answers drop the `[doc]` citation (valid citations 88%): prompt-format fix, already
  caught by the deterministic citation gate.
- Majority-vote three judge calls for faithfulness, the noisiest gated metric.
- Hand-label `results/calibration.csv` to report judge-human agreement (Cohen's kappa).

## Deployment
The evaluated pipeline is served as a FastAPI app on AWS Lambda (container image from ECR,
Function URL, Bedrock for generation, IAM execution role), with the same number-lock check
applied at runtime as a guardrail. A read-only snapshot of the MLflow store is deployed the
same way. See [deploy/README.md](deploy/README.md).
