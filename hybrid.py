"""
Hybrid retrieval: dense (embeddings) + sparse (BM25), fused with Reciprocal Rank Fusion.

WHY: dense and sparse fail in opposite ways.
  dense  -> great at semantics ("cost synergies" ~ "merger savings")
            weak at exact tokens (company names, tickers, IDs, numbers)
  BM25   -> great at exact term overlap, weights rare terms higher
            no notion of meaning ("car" and "automobile" are unrelated to it)

RRF fuses by RANK, not score, so we never have to normalize a cosine similarity
against a BM25 score (which live on totally different scales).
    rrf_score(doc) = sum over lists of 1 / (k + rank_in_that_list)
k=60 is the standard constant from the original paper; it damps the influence of
very top ranks so a single list can't dominate.
"""
import numpy as np
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    """Simple whitespace+lowercase tokenizer for BM25 (NOT the model tokenizer)."""
    return text.lower().replace(",", " ").replace(".", " ").replace("$", " ").split()

class HybridRetriever:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def dense_search(self, query_vec, k):
        scores = self.embeddings @ query_vec / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec))
        idx = np.argsort(scores)[::-1][:k]
        return list(idx)

    def sparse_search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return list(idx)

    def hybrid_search(self, query, query_vec, k=20, rrf_k=60):
        dense_ranked = self.dense_search(query_vec, k)
        sparse_ranked = self.sparse_search(query, k)

        rrf = {}
        for rank, idx in enumerate(dense_ranked):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_ranked):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (rrf_k + rank + 1)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]
        return [int(i) for i, _ in fused]
