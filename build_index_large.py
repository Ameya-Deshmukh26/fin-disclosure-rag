import glob, os, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build(model_name="all-mpnet-base-v2", data_dir="data_large", out_path="index_large.pkl"):
    model = SentenceTransformer(model_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = []
    for filepath in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        for c in splitter.split_text(text):
            chunks.append({"text": c, "source": os.path.basename(filepath)})
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=False)
    with open(out_path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": np.array(embeddings)}, f)
    print(f"[{model_name}] {len(chunks)} chunks -> {out_path}, shape {embeddings.shape}")

if __name__ == "__main__":
    build()
