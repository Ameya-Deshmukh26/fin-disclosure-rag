"""
Per-stage latency benchmark for the full RAG pipeline.

Times each stage separately so you can see WHERE the time actually goes -
which is the only way to know what to optimize. Spoiler: it's almost never
the vector math.

Configurations compared:
  A) retrieval only (brute-force numpy cosine)
  B) retrieval + cross-encoder rerank
  C) full pipeline with generation
  D) full pipeline, cache HIT
"""
import pickle, time, statistics
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieval import top_k_similar
from rerank import rerank, get_reranker
from generate import build_prompt, call_llm
from cache import DiskCache

QUERIES = [
    "What are Bluewater Regional Bank's estimated merger cost synergies?",
    "What percentage of total assets is Ashcroft Capital Partners' CRE portfolio?",
    "What was Northgate Holdings' quarterly net revenue?",
    "What settlement did Pinecrest disclose regarding regulatory inquiries?",
    "What is Harborview's target total CEO compensation?",
]

def timeit(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000  # ms

def main():
    print("Loading models (not counted in per-query latency)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    get_reranker()  # warm up cross-encoder

    with open("index_big.pkl","rb") as f:
        data = pickle.load(f)
    chunks, vecs = data["chunks"], data["embeddings"]
    print(f"Index: {len(chunks)} chunks, {vecs.shape[1]}-dim, {vecs.nbytes/1024/1024:.1f} MB\n")

    # warm-up pass so first-call overhead doesn't skew results
    model.encode(QUERIES[0]); top_k_similar(model.encode(QUERIES[0]), vecs, 10)

    stage_times = {"embed_query":[], "retrieve":[], "rerank":[], "generate":[]}

    print("=" * 74)
    print(f"{'STAGE':<22}{'mean ms':>10}{'min':>10}{'max':>10}   {'% of pipeline':>14}")
    print("=" * 74)

    for q in QUERIES:
        qv, t_embed = timeit(model.encode, q)
        stage_times["embed_query"].append(t_embed)

        results, t_ret = timeit(top_k_similar, qv, vecs, 20)
        stage_times["retrieve"].append(t_ret)

        candidates = [chunks[i] for i,_ in results]
        final, t_rr = timeit(rerank, q, candidates, 3)
        stage_times["rerank"].append(t_rr)

        msgs = build_prompt(q, final)
        try:
            _, t_gen = timeit(call_llm, msgs)
        except Exception as e:
            t_gen = float("nan")
        stage_times["generate"].append(t_gen)

    means = {k: statistics.mean(v) for k,v in stage_times.items()}
    total = sum(v for v in means.values() if not np.isnan(v))

    for k, v in stage_times.items():
        clean = [x for x in v if not np.isnan(x)]
        if not clean:
            print(f"{k:<22}{'FAILED':>10}")
            continue
        m = statistics.mean(clean)
        pct = 100*m/total if total else 0
        print(f"{k:<22}{m:>10.1f}{min(clean):>10.1f}{max(clean):>10.1f}   {pct:>13.1f}%")

    print("=" * 74)
    print(f"{'TOTAL':<22}{total:>10.1f} ms")

    print("\n--- CONFIG COMPARISON (mean ms per query) ---")
    a = means['embed_query'] + means['retrieve']
    b = a + means['rerank']
    c = b + (0 if np.isnan(means['generate']) else means['generate'])
    print(f"A) retrieve only          : {a:8.1f} ms")
    print(f"B) + rerank               : {b:8.1f} ms   (+{means['rerank']:.1f} ms for the accuracy gain)")
    print(f"C) + generation           : {c:8.1f} ms")

    # cache demo
    cache = DiskCache("query_cache.json")
    q = QUERIES[0]
    t0 = time.perf_counter(); cache.get(q); t_miss = (time.perf_counter()-t0)*1000
    cache.set(q, "cached answer")
    t0 = time.perf_counter(); cache.get(q); t_hit = (time.perf_counter()-t0)*1000
    print(f"D) cache HIT              : {t_hit:8.3f} ms   (vs {c:.1f} ms cold = {c/max(t_hit,0.001):.0f}x faster)")

if __name__ == "__main__":
    main()
