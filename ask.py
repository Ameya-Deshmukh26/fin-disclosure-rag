"""
Full RAG pipeline end to end: question -> retrieve -> build prompt -> generate -> cited answer.
"""
import pickle
from sentence_transformers import SentenceTransformer
from retrieval import top_k_similar
from generate import build_prompt, call_llm

def ask(question: str, index_path: str = "index_v2_recursive.pkl", k: int = 3) -> str:
    model = SentenceTransformer("all-mpnet-base-v2")

    with open(index_path, "rb") as f:
        data = pickle.load(f)
    chunks, doc_vectors = data["chunks"], data["embeddings"]

    query_vec = model.encode(question)
    results = top_k_similar(query_vec, doc_vectors, k=k)
    retrieved_chunks = [chunks[idx] for idx, score in results]

    print("Retrieved chunks:")
    for (idx, score), chunk in zip(results, retrieved_chunks):
        print(f"  score={score:.3f} source={chunk['source']}")

    messages = build_prompt(question, retrieved_chunks)
    answer = call_llm(messages)
    return answer

if __name__ == "__main__":
    q = "What company is Meridian acquiring and what are the expected synergies?"
    print(f"\nQuestion: {q}\n")
    answer = ask(q)
    print(f"\nAnswer:\n{answer}")
