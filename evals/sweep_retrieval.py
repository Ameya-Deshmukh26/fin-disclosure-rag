"""
Retrieval sweep, logged to MLflow: no LLM calls, so it runs anywhere for free.

Sweeps the dense/BM25 weight and the number of chunks retrieved per retriever, and
logs one MLflow run per configuration. In the dashboard, plot MRR against w_dense to
see the result this repo is built around: equal-weight fusion loses to BM25 alone,
and a BM25-leaning weight wins.

    python -m evals.sweep_retrieval
"""
import json
import statistics

from evals.config import EMBED_MODEL, GOLDENS_RAG, N_CONTEXT_DOCS
from evals.rag import RAG
from evals.tracking import log_eval_run

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
KS = [5, 10]


def main():
    rag = RAG()
    with open(GOLDENS_RAG, encoding="utf-8") as f:
        goldens = json.load(f)
    print(f"{'k':>3}{'w_dense':>9}{'hit@k':>8}{'MRR':>8}{'ctx@3':>8}")
    for k in KS:
        for w in WEIGHTS:
            ret = []
            for g in goldens:
                srcs = rag.retrieve_sources(g["question"], k=k, w_dense=w)
                rank = srcs.index(g["expected_source"]) + 1 if g["expected_source"] in srcs else None
                ret.append({"id": g["id"], "rank": rank, "in_context": rank is not None and rank <= N_CONTEXT_DOCS})
            summary = {
                "retrieval.hit_at_k": 100.0 * sum(r["rank"] is not None for r in ret) / len(ret),
                "retrieval.mrr": statistics.mean(1 / r["rank"] if r["rank"] else 0.0 for r in ret),
                "retrieval.context_recall_at_3": 100.0 * sum(r["in_context"] for r in ret) / len(ret),
            }
            log_eval_run({"summary": summary, "cases": {"retrieval": ret}},
                         params={"w_dense": w, "w_sparse": round(1 - w, 2), "k_retrieve": k,
                                 "embed_model": EMBED_MODEL, "n_questions": len(goldens)},
                         run_name=f"retrieval k={k} w_dense={w}", stage="retrieval_sweep")
            print(f"{k:>3}{w:>9.1f}{summary['retrieval.hit_at_k']:>7.0f}%"
                  f"{summary['retrieval.mrr']:>8.3f}{summary['retrieval.context_recall_at_3']:>7.0f}%")


if __name__ == "__main__":
    main()
