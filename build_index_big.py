"""
Index-time build at scale, instrumented.

KEY PRODUCTION POINT: batch_size on model.encode().
Embedding one-at-a-time wastes the GPU/CPU - the model can process many texts in
parallel in a single forward pass. Batching is the single biggest index-time speedup.
"""
import glob, os, pickle, time
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunks(data_dir, chunk_size=400, overlap=40):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        for c in splitter.split_text(text):
            chunks.append({"text": c, "source": os.path.basename(fp)})
    return chunks

def main():
    model = SentenceTransformer("all-mpnet-base-v2")

    t0 = time.perf_counter()
    chunks = load_chunks("data_big")
    t_chunk = time.perf_counter() - t0
    texts = [c["text"] for c in chunks]
    print(f"CHUNKING:  {len(chunks)} chunks from 600 docs in {t_chunk:.2f}s")

    # --- batch size comparison on a SUBSET, to show the batching effect ---
    subset = texts[:200]
    print("\nBATCH SIZE EFFECT (200 chunks):")
    for bs in [1, 8, 32, 128]:
        t0 = time.perf_counter()
        model.encode(subset, batch_size=bs, show_progress_bar=False)
        dt = time.perf_counter() - t0
        print(f"  batch_size={bs:<4} {dt:6.2f}s   ({len(subset)/dt:6.1f} chunks/sec)")

    # --- full index build with a sane batch size ---
    print(f"\nFULL INDEX BUILD ({len(texts)} chunks, batch_size=64):")
    t0 = time.perf_counter()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    t_embed = time.perf_counter() - t0
    print(f"  embed: {t_embed:.2f}s  ({len(texts)/t_embed:.1f} chunks/sec)")
    print(f"  shape: {embeddings.shape}")
    print(f"  memory: {embeddings.nbytes/1024/1024:.2f} MB as float32")

    embeddings = np.array(embeddings, dtype=np.float32)
    with open("index_big.pkl","wb") as f:
        pickle.dump({"chunks":chunks,"embeddings":embeddings}, f)
    print(f"  saved index_big.pkl")

if __name__ == "__main__":
    main()
