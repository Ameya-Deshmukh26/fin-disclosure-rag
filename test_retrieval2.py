import numpy as np

def top_k_similar(query_vector: np.ndarray, doc_vectors: np.ndarray, k: int) -> list:
    scores = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
    )
    top_k_indices = np.argpartition(scores, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    top_k_scores = scores[top_k_indices]
    return list(zip(top_k_indices, top_k_scores))

np.random.seed(0)
docs = np.random.rand(6, 4)
query = np.random.rand(4)
result = top_k_similar(query, docs, k=3)
print("Sorted descending?", [round(s,3) for _, s in result])
