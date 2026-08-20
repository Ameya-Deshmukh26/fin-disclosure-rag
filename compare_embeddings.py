"""
Compares embedding models on the SAME corpus + SAME eval set.
Everything else held constant, so the model is the only variable.

Measures the four things that actually matter in production:
  quality (hit_rate, MRR) / index build time / query latency / storage size
"""
import glob, os, time, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval import top_k_similar
from eval_gen import build_eval

MODELS = [
    ("all-mpnet-base-v2",          768),
    ("all-MiniLM-L6-v2",           384),
    ("BAAI/bge-small-en-v1.5",     384),
]

def load_chunks(data_dir="data_big"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        text = open(fp, encoding="utf-8").read()
        for c in splitter.split_text(text):
            chunks.append({"text": c, "source": os.path.basename(fp)})
    return chunks

def evaluate(model, chunks, vecs, evalset, k=5):
    hits, rrs, latencies = 0, [], []
    for item in evalset:
        t0 = time.perf_counter()
        qv = model.encode(item["question"])
        latencies.append((time.perf_counter()-t0)*1000)
        results = top_k_similar(qv, vecs, k=k)
        srcs = [chunks[i]["source"] for i,_ in results]
        if item["expected_source"] in srcs:
            hits += 1
            rrs.append(1.0/(srcs.index(item["expected_source"])+1))
        else:
            rrs.append(0.0)
    return {
        "hit_rate": hits/len(evalset),
        "mrr": sum(rrs)/len(rrs),
        "query_ms": sum(latencies)/len(latencies),
    }

def main():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    evalset = build_eval(n=25)
    print(f"Corpus: {len(chunks)} chunks | Eval: {len(evalset)} questions | k=5\n")

    rows = []
    for name, dim in MODELS:
        print(f"--- {name} ---")
        model = SentenceTransformer(name)

        t0 = time.perf_counter()
        vecs = np.array(model.encode(texts, batch_size=64, show_progress_bar=False), dtype=np.float32)
        build_s = time.perf_counter()-t0

        res = evaluate(model, chunks, vecs, evalset)
        size_mb = vecs.nbytes/1024/1024
        rows.append((name, vecs.shape[1], res["hit_rate"], res["mrr"], build_s, res["query_ms"], size_mb))
        print(f"  dim={vecs.shape[1]} hit_rate={res['hit_rate']:.2%} MRR={res['mrr']:.3f} "
              f"build={build_s:.1f}s query={res['query_ms']:.1f}ms size={size_mb:.2f}MB\n")

    print("="*104)
    print(f"{'MODEL':<28}{'DIM':>5}{'HIT@5':>9}{'MRR':>8}{'BUILD s':>10}{'QUERY ms':>10}{'SIZE MB':>10}")
    print("="*104)
    for name, dim, hr, mrr, bs, qms, sz in rows:
        print(f"{name:<28}{dim:>5}{hr:>8.1%}{mrr:>8.3f}{bs:>10.1f}{qms:>10.1f}{sz:>10.2f}")
    print("="*104)

if __name__ == "__main__":
    main()
