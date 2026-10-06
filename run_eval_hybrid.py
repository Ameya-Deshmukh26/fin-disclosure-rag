"""Compares dense-only vs BM25-only vs hybrid (RRF) on the same eval set."""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from eval_gen import build_eval
from hybrid import HybridRetriever

def score(retriever, model, evalset, mode, k=5):
    hits, rrs = 0, []
    for item in evalset:
        qv = model.encode(item["question"])
        if mode == "dense":
            idx = retriever.dense_search(qv, k)
        elif mode == "sparse":
            idx = retriever.sparse_search(item["question"], k)
        else:
            idx = retriever.hybrid_search(item["question"], qv, k=k)
        srcs = [retriever.chunks[i]["source"] for i in idx]
        if item["expected_source"] in srcs:
            hits += 1
            rrs.append(1.0 / (srcs.index(item["expected_source"]) + 1))
        else:
            rrs.append(0.0)
    return hits / len(evalset), sum(rrs) / len(rrs)

def main():
    model = SentenceTransformer("all-mpnet-base-v2")
    with open("index_big.pkl", "rb") as f:
        d = pickle.load(f)
    retriever = HybridRetriever(d["chunks"], d["embeddings"])
    evalset = build_eval(n=25)

    print(f"Corpus: {len(d['chunks'])} chunks | Eval: {len(evalset)} questions | k=5\n")
    print(f"{'METHOD':<28}{'HIT@5':>10}{'MRR':>10}")
    print("-" * 48)
    for mode, label in [("dense","Dense only (embeddings)"),
                        ("sparse","Sparse only (BM25)"),
                        ("hybrid","Hybrid (RRF fusion)")]:
        hr, mrr = score(retriever, model, evalset, mode)
        print(f"{label:<28}{hr:>9.1%}{mrr:>10.3f}")

if __name__ == "__main__":
    main()
