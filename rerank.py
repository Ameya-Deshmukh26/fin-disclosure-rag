"""
Cross-encoder reranking: takes (query, document) PAIRS together and scores
relevance directly - more accurate than comparing precomputed embeddings,
but must run at query time since it can't be precomputed per-document.

Used as a SECOND stage after fast bi-encoder retrieval narrows the field down.
"""
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    """
    candidates: list of {'text': ..., 'source': ...} dicts from initial retrieval
    Returns: top_n candidates re-sorted by cross-encoder relevance score
    """
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, score in scored[:top_n]]
