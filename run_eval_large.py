"""
Compares retrieval-only vs retrieval+reranking on the harder, ambiguous eval set.
Retrieval stage: fetch top 10 candidates (wide net, fast bi-encoder).
Rerank stage: cross-encoder re-scores those 10, keep top 3.
"""
import pickle
from sentence_transformers import SentenceTransformer
from eval_set_large import EVAL_QUESTIONS_LARGE
from retrieval import top_k_similar
from rerank import rerank

def evaluate(index_path: str, model: SentenceTransformer, use_rerank: bool, k_retrieve=10, k_final=3):
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    chunks, doc_vectors = data["chunks"], data["embeddings"]

    hits, reciprocal_ranks, details = 0, [], []
    for item in EVAL_QUESTIONS_LARGE:
        query_vec = model.encode(item["question"])
        results = top_k_similar(query_vec, doc_vectors, k=k_retrieve)
        candidates = [chunks[idx] for idx, _ in results]

        if use_rerank:
            final = rerank(item["question"], candidates, top_n=k_final)
        else:
            final = candidates[:k_final]

        retrieved_sources = [c["source"] for c in final]
        hit = item["expected_source"] in retrieved_sources
        hits += hit
        rank = retrieved_sources.index(item["expected_source"]) + 1 if hit else None
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)
        details.append({"q": item["question"][:50], "expected": item["expected_source"], "got": retrieved_sources, "rank": rank})

    return {"hit_rate": hits / len(EVAL_QUESTIONS_LARGE), "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks), "details": details}

def main():
    model = SentenceTransformer("all-mpnet-base-v2")

    print("\n=== Retrieval ONLY (no reranking) ===")
    r1 = evaluate("index_large.pkl", model, use_rerank=False)
    print(f"hit_rate: {r1['hit_rate']:.2%}   MRR: {r1['mrr']:.3f}")
    for d in r1["details"]:
        rank_str = f"rank {d['rank']}" if d["rank"] else "MISS"
        print(f"  [{rank_str}] {d['q']}... -> {d['got']}")

    print("\n=== Retrieval + Cross-Encoder RERANKING ===")
    r2 = evaluate("index_large.pkl", model, use_rerank=True)
    print(f"hit_rate: {r2['hit_rate']:.2%}   MRR: {r2['mrr']:.3f}")
    for d in r2["details"]:
        rank_str = f"rank {d['rank']}" if d["rank"] else "MISS"
        print(f"  [{rank_str}] {d['q']}... -> {d['got']}")

if __name__ == "__main__":
    main()
