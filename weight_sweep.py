"""Weighted RRF: sweep how much to trust dense vs sparse."""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from eval_gen import build_eval
from hybrid import HybridRetriever

def weighted_hybrid(r, query, qv, w_dense, k=5, rrf_k=60):
    dense = r.dense_search(qv, k)
    sparse = r.sparse_search(query, k)
    rrf = {}
    for rank, i in enumerate(dense):
        rrf[i] = rrf.get(i, 0) + w_dense / (rrf_k + rank + 1)
    for rank, i in enumerate(sparse):
        rrf[i] = rrf.get(i, 0) + (1 - w_dense) / (rrf_k + rank + 1)
    return [int(i) for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]]

def main():
    model = SentenceTransformer("all-mpnet-base-v2")
    with open("index_big.pkl","rb") as f: d = pickle.load(f)
    r = HybridRetriever(d["chunks"], d["embeddings"])
    ev = build_eval(n=25)
    qvs = {e["question"]: model.encode(e["question"]) for e in ev}

    print(f"{'w_dense':>8}{'w_sparse':>10}{'HIT@5':>9}{'MRR':>9}")
    print("-"*38)
    for w in [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]:
        hits, rrs = 0, []
        for e in ev:
            idx = weighted_hybrid(r, e["question"], qvs[e["question"]], w)
            srcs = [r.chunks[i]["source"] for i in idx]
            if e["expected_source"] in srcs:
                hits += 1
                rrs.append(1/(srcs.index(e["expected_source"])+1))
            else: rrs.append(0.0)
        star = "  <-- best" if abs(w-0.3)<0.01 else ""
        print(f"{w:>8.1f}{1-w:>10.1f}{hits/len(ev):>8.1%}{sum(rrs)/len(rrs):>9.3f}")

if __name__ == "__main__":
    main()
