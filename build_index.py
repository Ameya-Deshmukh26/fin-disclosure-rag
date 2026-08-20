"""
Build the RAG index: load docs -> chunk -> embed -> store in memory.
Embedding model: all-mpnet-base-v2 (sentence-transformers) - 768-dim, runs locally, no API key.
"""
import os
import glob
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import chunk_text

CHUNK_SIZE = 500   # characters
OVERLAP = 50       # characters

def load_documents(data_dir: str = "data") -> list[dict]:
    """Load each .txt file, tagging chunks with their source filename."""
    all_chunks = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": os.path.basename(filepath)})
    return all_chunks

def main():
    print("Loading model all-mpnet-base-v2 (first run downloads ~420MB, be patient)...")
    model = SentenceTransformer("all-mpnet-base-v2")

    chunks = load_documents()
    print(f"Loaded {len(chunks)} chunks from data/")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")  # should be (n_chunks, 768)

    # Save everything so we don't re-embed every time
    with open("index.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": np.array(embeddings)}, f)
    print("Saved index.pkl")

if __name__ == "__main__":
    main()
