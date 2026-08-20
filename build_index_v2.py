"""
v2: same pipeline as build_index.py, but chunking strategy is RecursiveCharacterTextSplitter
instead of hand-written fixed-size chunk_text(). Same embedding model, same chunk_size/overlap
numbers, so chunking strategy is the ONLY variable that changed. That's what makes this a fair
comparison against v1.
"""
import os
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
OVERLAP = 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def load_documents(data_dir: str = "data") -> list[dict]:
    all_chunks = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": os.path.basename(filepath)})
    return all_chunks

def main():
    model = SentenceTransformer("all-mpnet-base-v2")
    chunks = load_documents()
    print(f"[v2 recursive] Loaded {len(chunks)} chunks from data/")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    print(f"[v2 recursive] Embedding shape: {embeddings.shape}")

    with open("index_v2_recursive.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": np.array(embeddings)}, f)
    print("Saved index_v2_recursive.pkl")

if __name__ == "__main__":
    main()
