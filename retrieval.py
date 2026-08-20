import numpy as np

def top_k_similar(query_vector: np.ndarray, doc_vectors: np.ndarray, k: int) -> list:
    scores = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    )
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    top_k_scores = scores[top_k_indices]
    return list(zip(top_k_indices, top_k_scores))
