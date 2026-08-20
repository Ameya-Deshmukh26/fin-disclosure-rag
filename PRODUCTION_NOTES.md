# Production RAG: Scale, Cost, Latency

Interview reference. Every section is "what breaks at scale, and what you do about it."

---

## 1. Index time: embedding a large corpus

**What breaks:** naive `model.encode(all_texts)` on millions of chunks - single-threaded,
no batching, everything in memory, no resume on failure.

**What you do:**

| Problem | Fix |
|---|---|
| Slow per-item embedding | **Batching.** `batch_size=64..256`. One forward pass over many texts. Biggest single win. |
| Memory blowup | **Stream in shards.** Process 10k chunks, write to store, free memory, repeat. Never hold the full matrix. |
| Job dies at 80% | **Checkpointing.** Persist progress per shard so a restart resumes, not restarts. |
| Full re-embed on every doc change | **Content-hash the chunk.** Only embed chunks whose hash changed. Incremental indexing. |
| Cost of API embeddings | Local model (what we used) removes per-token cost entirely. Or a cheaper/smaller API model. |
| CPU-bound | GPU gives 10-50x on embedding. Or parallelize across workers. |

**Storage math (know this cold):**
`n_chunks x n_dims x 4 bytes` (float32).
- 1M chunks x 768 dims x 4B = **~3 GB**
- 1M chunks x 1536 dims x 4B = **~6 GB**
This is why dimension choice is a cost decision, not just a quality one. Halving dims
halves storage and roughly halves search time.

**Quantization:** store as int8 instead of float32 -> 4x smaller, small recall loss.
Binary quantization -> 32x smaller, bigger loss, used with a rescoring pass.

---

## 2. Query time: search at scale

**What breaks:** our `top_k_similar` is **brute force** - it computes similarity against
EVERY vector. O(n x d) per query. Fine at 1k chunks, dead at 10M.

**The fix: Approximate Nearest Neighbor (ANN).**

| Index | How it works | Tradeoff |
|---|---|---|
| **Flat / brute force** | Compare against all vectors | Exact, O(n). Fine < ~100k vectors |
| **HNSW** (Chroma, Qdrant, Weaviate default) | Multi-layer proximity graph, greedy descent | ~O(log n), high recall, high memory |
| **IVF** (FAISS) | Cluster vectors, search only nearest clusters | Fast, tune `nprobe` for recall/speed |
| **IVF-PQ** | IVF + product quantization compression | Huge corpora, lowest memory, lossy |

**HNSW knobs to name:** `M` (graph connectivity), `ef_construction` (build quality),
`ef_search` (search breadth -> recall/latency dial at query time).

**Note the connection to what we built:** the k-means centroid routing on the Boehringer
system is conceptually IVF - partition the space, search only the relevant partition.
Same idea, hand-rolled.

**Sharding:** beyond one machine, partition by tenant/domain/date and fan out queries.
Also gives you access-control isolation for free.

---

## 3. Caching (three layers, different hit profiles)

| Layer | Key | Hits when | Saves |
|---|---|---|---|
| **Exact query cache** | normalized question string | identical repeat question | the ENTIRE pipeline |
| **Semantic cache** | query embedding + similarity threshold | paraphrase of a prior question | everything after embedding |
| **Embedding cache** | text hash | same text re-embedded | one embed call |
| **Prompt/KV cache** | provider-side, stable prompt prefix | same system prompt + context prefix | input token cost (~90% cheaper on reads) |

**Semantic cache danger:** threshold too loose -> you serve the answer to a *different*
question. Needs the same eval rigor as retrieval. Also needs invalidation when the
underlying documents change, or you serve confidently stale answers.

**Cache invalidation is the hard part:** if doc_014 is amended, every cached answer
derived from it is now wrong. Options: TTL, or tag cache entries with source doc IDs
and purge on document update.

---

## 4. Where the latency actually is

Measured on this project (see benchmark.py output). The ranking is the point:

**generation >> reranking > embedding > vector search**

The LLM call dominates by an order of magnitude. Consequences:
- Optimizing the vector math is almost always the wrong place to look first.
- **Streaming** the response transforms *perceived* latency without changing total time.
- **Model routing:** small/cheap model for easy queries, large model only when needed.
- Cutting retrieved context cuts input tokens -> real latency and cost win.
- Reranking cost is real but bounded (you rerank 20 candidates, not the corpus).

---

## 5. Cost levers, ordered by payoff

1. **Cache** (exact then semantic) - avoid the call entirely
2. **Model routing** - don't send trivial queries to the expensive model
3. **Trim context** - fewer retrieved chunks = fewer input tokens; rerank lets you send
   3 good chunks instead of 10 mediocre ones
4. **Prompt caching** - stable prefix billed at a fraction on reads
5. **Local embeddings** - removes embedding API cost line entirely
6. **Smaller dimensions** - storage + search cost
7. **Batch offline work** - batch APIs are ~50% cheaper for non-realtime jobs

---

## 6. Failure modes and guardrails

| Failure | Guardrail |
|---|---|
| Retrieval returns nothing relevant | Score threshold -> refuse to answer rather than generate from noise |
| Model ignores context, answers from memory | System prompt constraint + faithfulness check post-hoc |
| Hallucinated citation | Validate cited source IDs against actually-retrieved chunk IDs programmatically |
| Stale answers after doc update | Re-index pipeline + cache invalidation by source ID |
| Silent quality regression after a prompt change | Golden eval set in CI - block deploy on MRR/faithfulness drop |
| Cost spike | Per-user rate limits, token budget caps, alerting on spend |
| Retrieval drift over time | Monitor score distributions; alert when mean top-1 score declines |

---

## 7. Monitoring in production (what LangSmith gives you)

- **Per-request trace:** which chunks were retrieved, at what scores, what prompt was
  sent, what came back, latency per stage
- **Token + cost per request**, aggregated
- **Error and refusal rates**
- **Regression detection:** run the golden eval set on every deploy, compare to baseline

The critical distinction: **offline eval** (before deploy, on a fixed golden set) vs
**online monitoring** (in production, on real traffic, where you can't compute
correctness without labels - so you watch proxies: score distributions, refusal rate,
user feedback signals, latency, cost).
