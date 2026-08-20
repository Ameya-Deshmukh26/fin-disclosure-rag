"""
Evaluation harness: measures hit_rate@k AND MRR for a given index.

hit_rate@k = did the correct document appear ANYWHERE in the top-k? (binary, per question)
MRR (Mean Reciprocal Rank) = average of 1/rank_of_first_correct_hit across all questions.
  - correct result at rank 1 -> contributes 1.0
  - correct result at rank 2 -> contributes 0.5
  - correct result at rank 3 -> contributes 0.33
  - not found in top-k       -> contributes 0

MRR is more sensitive than hit_rate: two systems can tie on hit_rate@k (both "found it
somewhere in the top 3") while one consistently surfaces the right answer FIRST and the
other buries it at rank 2 or 3. That distinction matters a lot for RAG - the model's
generation step usually pays most attention to whatever is closest to the top.
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from eval_set import EVAL_QUESTIONS
from retrieval import top_k_similar

def evaluate_index(index_path: str, model: SentenceTransformer, k: int = 3) -> dict:
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    doc_vectors = data["embeddings"]

    hits = 0
    reciprocal_ranks = []
    details = []
    for item in EVAL_QUESTIONS:
        query_vec = model.encode(item["question"])
        results = top_k_similar(query_vec, doc_vectors, k=k)
        retrieved_sources = [chunks[idx]["source"] for idx, _ in results]

        hit = item["expected_source"] in retrieved_sources
        hits += hit

        if item["expected_source"] in retrieved_sources:
            rank = retrieved_sources.index(item["expected_source"]) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        details.append({
            "question": item["question"], "expected": item["expected_source"],
            "retrieved": retrieved_sources, "hit": hit, "rank": rank,
        })

    return {
        "hit_rate": hits / len(EVAL_QUESTIONS),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "details": details,
    }

def main():
    model = SentenceTransformer("all-mpnet-base-v2")

    for label, path in [("v1: fixed-size", "index_v1_fixed.pkl"), ("v2: recursive", "index_v2_recursive.pkl")]:
        result = evaluate_index(path, model, k=3)
        print(f"\n=== {label} ===")
        print(f"hit_rate@3: {result['hit_rate']:.2%}   MRR: {result['mrr']:.3f}")
        for d in result["details"]:
            rank_str = f"rank {d['rank']}" if d["rank"] else "NOT FOUND"
            print(f"  [{rank_str}] {d['question'][:45]}...")

if __name__ == "__main__":
    main()
