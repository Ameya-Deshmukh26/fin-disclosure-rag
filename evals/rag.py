"""
The RAG pipeline under evaluation: the best configuration measured in this repo.

  1. Weighted hybrid retrieval: dense (mpnet) + BM25, fused with weighted RRF
     (w_dense = 0.2), the configuration that reached MRR 0.810.
  2. Parent-document expansion: chunks are small (3 per filing, the first is often
     just the header), so we map retrieved chunks back to their source filings and
     hand the generator whole documents. Small chunks for matching, full documents
     for answering.
  3. Generation with forced citations ([doc_xxxx.txt]) and an explicit refusal phrase.

CONTEXT_MODE="chunks" replaces step 2 for long documents: the model gets only the top
matched chunks, each with a [company | form | date] header (evals/build_chunk_index.py).
"""
import os
import pickle
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from generate import build_prompt
from hybrid import HybridRetriever, tokenize
from evals.config import (CHUNK_SIZE, CONTEXT_MODE, DATA_DIR, EMBED_MODEL, ENTITY_FILTER,
                          GENERATOR_MODEL, INDEX_PATH, K_RETRIEVE, N_CONTEXT_CHUNKS,
                          N_CONTEXT_DOCS, W_DENSE, chunk_index_path)
from evals.llm import complete


def read_doc(source: str) -> str:
    with open(os.path.join(DATA_DIR, source), encoding="utf-8") as f:
        return f.read()


class RAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        index = chunk_index_path(CHUNK_SIZE) if CONTEXT_MODE == "chunks" else INDEX_PATH
        with open(index, "rb") as f:
            data = pickle.load(f)
        self.retriever = HybridRetriever(data["chunks"], data["embeddings"])
        self.gen_input_tokens = 0
        self.gen_output_tokens = 0
        # Metadata: which company filed each document (the "Registrant" line).
        self.registrant = {}
        for c in self.retriever.chunks:
            s = c["source"]
            if s not in self.registrant:
                m = re.search(r"Registrant: (.+)", read_doc(s))
                self.registrant[s] = m.group(1).strip() if m else None
        self.entities = sorted({v for v in self.registrant.values() if v}, key=len, reverse=True)

    def entity_in(self, question: str) -> str | None:
        """Exact company named in the question, if it is one we have filings for."""
        q = question.lower()
        return next((e for e in self.entities if e.lower() in q), None)

    def _search(self, question, qv, k, idxs=None):
        """Dense + BM25 top-k, optionally restricted to a subset of chunk indices."""
        if idxs is None:
            return self.retriever.dense_search(qv, k), self.retriever.sparse_search(question, k)
        emb = self.retriever.embeddings[idxs]
        dense_scores = emb @ qv / (np.linalg.norm(emb, axis=1) * np.linalg.norm(qv))
        dense = [idxs[i] for i in np.argsort(dense_scores)[::-1][:k]]
        bm25 = self.retriever.bm25.get_scores(tokenize(question))
        sparse = sorted(idxs, key=lambda i: bm25[i], reverse=True)[:k]
        return dense, sparse

    def retrieve_sources(self, question: str, k: int = K_RETRIEVE,
                         w_dense: float = W_DENSE, rrf_k: int = 60) -> list[str]:
        """Ranked, de-duplicated source filenames from weighted RRF."""
        sources = []
        for i in self.ranked_chunks(question, k, w_dense, rrf_k):
            s = self.retriever.chunks[i]["source"]
            if s not in sources:
                sources.append(s)
        return sources

    def ranked_chunks(self, question: str, k: int = K_RETRIEVE,
                      w_dense: float = W_DENSE, rrf_k: int = 60) -> list[int]:
        """Chunk indices ranked by weighted RRF over dense + BM25.

        With ENTITY_FILTER on, a question that names a known company is searched only
        within that company's filings (a metadata pre-filter, like WHERE ein = ... before
        the vector search). A question naming an unknown company is NOT filtered, so the
        refusal behavior is still tested on the full corpus.
        """
        qv = self.embedder.encode(question)
        entity = self.entity_in(question) if ENTITY_FILTER else None
        idxs = ([i for i, c in enumerate(self.retriever.chunks)
                 if self.registrant[c["source"]] == entity] if entity else None)
        dense, sparse = self._search(question, qv, k, idxs)
        fused = {}
        for rank, i in enumerate(dense):
            fused[i] = fused.get(i, 0) + w_dense / (rrf_k + rank + 1)
        for rank, i in enumerate(sparse):
            fused[i] = fused.get(i, 0) + (1 - w_dense) / (rrf_k + rank + 1)
        return [int(i) for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    def generate_from(self, question: str, sources: list[str]) -> str:
        """Answer from whole filings."""
        return self.generate_ctx(question, [{"text": read_doc(s), "source": s} for s in sources])

    def generate_ctx(self, question: str, ctx: list[dict]) -> str:
        """Answer from given {text, source} passages (whole filings or chunks)."""
        # temperature 0: an eval run should measure the pipeline, not sampling noise
        text, usage = complete(GENERATOR_MODEL, build_prompt(question, ctx),
                               max_tokens=300, temperature=0.0)
        self.gen_input_tokens += usage.get("input_tokens", 0)
        self.gen_output_tokens += usage.get("output_tokens", 0)
        self.last_usage = usage          # per-call token usage, read by the online tracer
        return text

    def answer(self, question: str) -> dict:
        t0 = time.perf_counter()
        if CONTEXT_MODE == "chunks":
            ranked = self.ranked_chunks(question)
            picked = [self.retriever.chunks[i] for i in ranked[:N_CONTEXT_CHUNKS]]
            ctx = [{"text": c["text"], "source": c["source"]} for c in picked]
            sources = list(dict.fromkeys(self.retriever.chunks[i]["source"] for i in ranked))
            context_sources = list(dict.fromkeys(c["source"] for c in picked))
        else:
            sources = self.retrieve_sources(question)
            context_sources = sources[:N_CONTEXT_DOCS]
            ctx = [{"text": read_doc(s), "source": s} for s in context_sources]
        t1 = time.perf_counter()
        answer = self.generate_ctx(question, ctx)
        t2 = time.perf_counter()
        return {
            "answer": answer,
            "retrieved_sources": sources,
            "context_sources": context_sources,
            "contexts": [c["text"] for c in ctx],
            "input_tokens": self.last_usage.get("input_tokens", 0),
            "latency_ms": {"retrieve": (t1 - t0) * 1000, "generate": (t2 - t1) * 1000},
        }
