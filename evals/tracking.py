"""
MLflow tracking for the eval suite: every run is one MLflow run, so the dashboard can
compare configurations side by side (retrieval weights, generator, judge, thresholds).

What gets logged per run:
  params    the configuration under test (generator, judge, w_dense, k, context docs)
  metrics   every number in the run summary: retrieval, deterministic checks, judge
            averages, refusal rate, latency, token usage
  tags      regression verdict vs baseline (PASS / REVIEW / BLOCK), stage, git commit
  tables    per-question results (question, answer, scores, failures), browsable in the
            MLflow UI so a bad average can be traced to the exact question behind it
  artifact  the raw results JSON

Local store: sqlite:///mlflow.db in the repo root, artifacts in ./mlartifacts.
    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
"""
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
ARTIFACT_ROOT = (ROOT / "mlartifacts").as_uri()
EXPERIMENT = "fin-disclosure-rag-evals"


def _mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _safe_key(k: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-/ " else "_" for ch in k)


def log_eval_run(result: dict, params: dict, verdict: dict | None = None,
                 run_name: str | None = None, stage: str = "full") -> str:
    mlflow = _mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({_safe_key(k): float(v) for k, v in result["summary"].items()
                            if isinstance(v, (int, float))})
        mlflow.set_tags({"stage": stage, "git_commit": _git_commit(),
                         "verdict": verdict["verdict"] if verdict else "no_baseline"})

        cases = result.get("cases", {})
        if cases.get("rag"):
            rows = []
            for c in cases["rag"]:
                j = c.get("judge", {})
                rows.append({
                    "id": c["id"], "question": c["question"], "answer": c["answer"],
                    "expected": c["expected_output"],
                    "cites_expected": c.get("cites_expected"),
                    "number_lock_ok": c.get("number_lock_ok"),
                    "unsupported_numbers": ", ".join(c.get("unsupported_numbers", [])),
                    "key_facts_ok": c.get("key_facts_ok"),
                    "faithfulness": j.get("faithfulness", {}).get("score"),
                    "correctness": j.get("correctness", {}).get("score"),
                    "answer_relevancy": j.get("answer_relevancy", {}).get("score"),
                    "correctness_reason": (j.get("correctness", {}).get("reason") or "")[:500],
                    "generate_ms": round(c["latency_ms"]["generate"], 1),
                })
            mlflow.log_table({k: [r[k] for r in rows] for k in rows[0]}, artifact_file="rag_cases.json")
        if cases.get("refusal"):
            refs = cases["refusal"]
            mlflow.log_table({"id": [r["id"] for r in refs], "kind": [r["kind"] for r in refs],
                              "question": [r["question"] for r in refs],
                              "refused": [r["refused"] for r in refs],
                              "answer": [r["answer"] for r in refs]},
                             artifact_file="refusal_cases.json")
        if cases.get("retrieval"):
            ret = cases["retrieval"]
            mlflow.log_table({"id": [r["id"] for r in ret], "rank": [r["rank"] for r in ret],
                              "in_context": [r["in_context"] for r in ret]},
                             artifact_file="retrieval_cases.json")
        if verdict:
            mlflow.log_dict(verdict, "regression_verdict.json")
        mlflow.log_dict({"meta": result.get("meta", {}), "summary": result["summary"]}, "summary.json")
        return run.info.run_id
