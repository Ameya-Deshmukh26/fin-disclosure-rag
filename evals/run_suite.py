"""
Runs the full eval suite and compares it to the saved baseline.

Stages (cheapest first):
  1. retrieval   deterministic: hit@5, MRR, context recall          (no LLM)
  2. rag         end to end: deterministic checks + judge metrics    (generator + judge)
  3. generator   isolation: answer from the GOLDEN filing only, so a
                 low faithfulness score is purely the generator's fault
  4. refusal     adversarial set: must decline, never invent         (generator only)

    python -m evals.run_suite                  # full run, compare to baseline
    python -m evals.run_suite --skip-judge     # deterministic stages only (cheap)
    python -m evals.run_suite --save-baseline  # accept this run as the new baseline

Exit code 1 if any GATE regressed, so CI can block a release.
"""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.test_case import LLMTestCase  # noqa: E402

from evals import checks  # noqa: E402
from evals.config import GOLDENS_RAG, GOLDENS_REFUSAL, JUDGE_MODEL, GENERATOR_MODEL, MAX_WORKERS, RESULTS_DIR  # noqa: E402
from evals.metrics import RUBRIC_VERSION, answer_relevancy, correctness, faithfulness, measure  # noqa: E402
from evals.registry import compare  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(xs):
    xs = [x for x in xs if x is not None]
    return 100.0 * sum(bool(x) for x in xs) / len(xs) if xs else None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else None


def run(skip_judge=False):
    from evals.rag import RAG
    rag = RAG()
    goldens, refusals = load(GOLDENS_RAG), load(GOLDENS_REFUSAL)
    judge = None
    if not skip_judge:
        from evals.judge import Judge
        judge = Judge()

    # ---- 1. retrieval (deterministic) --------------------------------------------
    ret = []
    for g in goldens:
        srcs = rag.retrieve_sources(g["question"])
        rank = srcs.index(g["expected_source"]) + 1 if g["expected_source"] in srcs else None
        ret.append({"id": g["id"], "rank": rank, "in_context": rank is not None and rank <= 3})

    # ---- 2. end-to-end RAG ----------------------------------------------------------
    cases = []
    for g in goldens:
        out = rag.answer(g["question"])
        c = {"id": g["id"], "question": g["question"], "expected_output": g["expected_output"],
             "expected_source": g["expected_source"], **out}
        c.update(checks.citation_check(out["answer"], out["context_sources"], g["expected_source"]))
        c.update(checks.number_lock(out["answer"], out["contexts"]))
        c.update(checks.key_facts(out["answer"], g["expected_numbers"]))
        c["refused"] = checks.is_refusal(out["answer"])
        cases.append(c)

    if judge:
        def judge_case(c):
            tc = LLMTestCase(input=c["question"], actual_output=c["answer"],
                             expected_output=c["expected_output"], retrieval_context=c["contexts"])
            return measure(tc, {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                                "correctness": correctness}, judge)
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for c, j in zip(cases, ex.map(judge_case, cases)):
                c["judge"] = j

    # ---- 3. generator in isolation (golden context) ---------------------------------
    gen = []
    for g in goldens:
        ans = rag.generate_from(g["question"], [g["expected_source"]])
        row = {"id": g["id"], "answer": ans}
        row.update(checks.number_lock(ans, g["ideal_context"]))
        row.update(checks.key_facts(ans, g["expected_numbers"]))
        gen.append(row)
    if judge:
        def judge_gen(pair):
            row, g = pair
            tc = LLMTestCase(input=g["question"], actual_output=row["answer"],
                             retrieval_context=g["ideal_context"])
            return measure(tc, {"faithfulness": faithfulness}, judge)
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            for row, j in zip(gen, ex.map(judge_gen, list(zip(gen, goldens)))):
                row["judge"] = j

    # ---- 4. refusal (adversarial) ----------------------------------------------------
    refs = []
    for r in refusals:
        out = rag.answer(r["question"])
        refs.append({"id": r["id"], "kind": r["kind"], "question": r["question"],
                     "answer": out["answer"], "refused": checks.is_refusal(out["answer"]),
                     "context_sources": out["context_sources"]})

    # ---- summary -----------------------------------------------------------------------
    def judge_avg(rows, name):
        return avg([r.get("judge", {}).get(name, {}).get("score") for r in rows])

    def judge_pass(rows, name):
        return pct([r.get("judge", {}).get(name, {}).get("success") for r in rows if "judge" in r])

    summary = {
        "retrieval.hit_at_5": pct([r["rank"] is not None for r in ret]),
        "retrieval.mrr": avg([1 / r["rank"] if r["rank"] else 0.0 for r in ret]),
        "retrieval.context_recall_at_3": pct([r["in_context"] for r in ret]),
        "rag.citations_valid_rate": pct([c["citations_valid"] for c in cases]),
        "rag.cites_expected_rate": pct([c["cites_expected"] for c in cases]),
        "rag.number_lock_rate": pct([c["number_lock_ok"] for c in cases]),
        "rag.key_facts_rate": pct([c["key_facts_ok"] for c in cases]),
        "rag.false_refusal_rate": pct([c["refused"] for c in cases]),
        "generator.number_lock_rate": pct([r["number_lock_ok"] for r in gen]),
        "generator.key_facts_rate": pct([r["key_facts_ok"] for r in gen]),
        "safety.refusal_rate": pct([r["refused"] for r in refs]),
        "ops.latency.generate_p50_ms": statistics.median([c["latency_ms"]["generate"] for c in cases]),
        "ops.latency.generate_p95_ms": p95([c["latency_ms"]["generate"] for c in cases]),
        "ops.latency.retrieve_p50_ms": statistics.median([c["latency_ms"]["retrieve"] for c in cases]),
        "ops.generator_input_tokens": rag.gen_input_tokens,
        "ops.generator_output_tokens": rag.gen_output_tokens,
    }
    if judge:
        for name in ("faithfulness", "answer_relevancy", "correctness"):
            summary[f"rag.{name}.avg_score"] = judge_avg(cases, name)
            summary[f"rag.{name}.pass_rate"] = judge_pass(cases, name)
        # Faithfulness is undefined for a refusal (no claims to check): the same refusal
        # scored 1.0 in one run and 0.0 in another. Score it only on answered questions;
        # refusals are measured by the refusal checks and by correctness instead.
        answered = [c for c in cases if not c["refused"]]
        summary["rag.faithfulness_answered.avg_score"] = judge_avg(answered, "faithfulness")
        summary["generator.faithfulness.avg_score"] = judge_avg(gen, "faithfulness")
        summary["generator.faithfulness.pass_rate"] = judge_pass(gen, "faithfulness")
        summary.update({f"ops.{k}": v for k, v in judge.usage().items()})
        summary["ops.judge_errors"] = sum(
            1 for rows in (cases, gen) for r in rows for m in r.get("judge", {}).values() if "error" in m)

    meta = {"generator": GENERATOR_MODEL, "judge": JUDGE_MODEL if judge else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_rag": len(cases), "n_refusal": len(refs)}
    return {"meta": meta, "summary": summary,
            "cases": {"retrieval": ret, "rag": cases, "generator": gen, "refusal": refs}}


def print_report(result, verdict=None):
    s = result["summary"]
    print("\n" + "=" * 64)
    print(f"EVAL SUITE  generator={result['meta']['generator']}  judge={result['meta']['judge']}")
    print("=" * 64)
    for k, v in s.items():
        print(f"  {k:<38} {v:>10.3f}" if isinstance(v, float) else f"  {k:<38} {v!s:>10}")
    misses = [r for r in result["cases"]["refusal"] if not r["refused"]]
    if misses:
        print("\n  Adversarial questions answered instead of refused:")
        for r in misses:
            print(f"    [{r['kind']}] {r['question']}\n        -> {r['answer'][:140]!r}")
    if verdict:
        print("\n  REGRESSION CHECK vs baseline:", verdict["verdict"])
        for row in verdict["rows"]:
            if row["status"] not in ("FLAT", "info"):
                print(f"    {row['status']:<9}{row['metric']:<38}{row['baseline']:.3f} -> {row['current']:.3f}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true", help="skip logging this run to MLflow")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    result = run(skip_judge=args.skip_judge)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(RESULTS_DIR, f"run_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    base_path = os.path.join(RESULTS_DIR, "baseline.json")
    verdict = None
    if args.save_baseline:
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(result["summary"], f, indent=2)
        print(f"Saved baseline -> {base_path}")
    elif os.path.exists(base_path):
        verdict = compare(load(base_path), result["summary"])
    print_report(result, verdict)

    if not args.no_mlflow:
        from evals import config
        from evals.tracking import log_eval_run
        run_id = log_eval_run(
            result,
            params={"generator": config.GENERATOR_MODEL,
                    "judge": config.JUDGE_MODEL if not args.skip_judge else "none",
                    "w_dense": config.W_DENSE, "k_retrieve": config.K_RETRIEVE,
                    "n_context_docs": config.N_CONTEXT_DOCS, "threshold": config.THRESHOLD,
                    "entity_filter": config.ENTITY_FILTER,
                    "correctness_rubric": RUBRIC_VERSION,
                    "embed_model": config.EMBED_MODEL, "baseline_saved": args.save_baseline},
            verdict=verdict, run_name=args.run_name,
            stage="deterministic" if args.skip_judge else "full")
        print(f"MLflow run logged: {run_id}  (python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001)")
    if verdict and verdict["verdict"] == "BLOCK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
