"""
LangSmith tracing on the HAND-ROLLED pipeline.

Two ways LangSmith gets data:
  1. AUTOMATIC - any LangChain/LCEL chain is traced with zero code changes,
     just env vars. (see ask_langchain.py)
  2. @traceable DECORATOR - for your own non-LangChain functions. This is how
     you trace a custom pipeline. Each decorated function becomes a span in the
     trace tree, with inputs/outputs/latency captured automatically.

What you get in the UI: a waterfall per request showing each stage's latency,
the exact inputs/outputs at every step (so you can see WHICH chunks were
retrieved for a bad answer), token counts, and errors. This is the difference
between "the answer was wrong" and "the answer was wrong because retrieval
returned the compensation doc instead of the risk doc."

SETUP: get a free key at smith.langchain.com -> Settings -> API Keys,
then add to hugging.env:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=lsv2_...
  LANGCHAIN_PROJECT=fin-disclosure-rag
"""
import os, pickle
from dotenv import load_dotenv
load_dotenv("hugging.env")

from langsmith import traceable
from sentence_transformers import SentenceTransformer
from retrieval import top_k_similar
from rerank import rerank
from generate import build_prompt, call_llm

_model = None
_index = None

def _lazy_load():
    global _model, _index
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    if _index is None:
        with open("index_big.pkl","rb") as f:
            _index = pickle.load(f)
    return _model, _index

@traceable(name="embed_query")
def embed_query(question: str):
    model, _ = _lazy_load()
    return model.encode(question).tolist()

@traceable(name="retrieve")
def retrieve(query_vec, k: int = 20):
    _, index = _lazy_load()
    import numpy as np
    results = top_k_similar(np.array(query_vec), index["embeddings"], k=k)
    return [{"text": index["chunks"][i]["text"],
             "source": index["chunks"][i]["source"],
             "score": float(s)} for i, s in results]

@traceable(name="rerank_stage")
def rerank_stage(question: str, candidates: list, top_n: int = 3):
    return rerank(question, candidates, top_n=top_n)

@traceable(name="generate_answer")
def generate_answer(question: str, final_chunks: list):
    msgs = build_prompt(question, final_chunks)
    return call_llm(msgs)

@traceable(name="rag_pipeline")
def rag_pipeline(question: str, k_retrieve: int = 20, k_final: int = 3):
    """Top-level span - everything below nests under this in the LangSmith trace tree."""
    qv = embed_query(question)
    candidates = retrieve(qv, k=k_retrieve)
    final = rerank_stage(question, candidates, top_n=k_final)
    answer = generate_answer(question, final)
    return {"answer": answer, "sources": [c["source"] for c in final]}

if __name__ == "__main__":
    traced = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    key = os.environ.get("LANGCHAIN_API_KEY")
    print(f"LangSmith tracing: {'ON' if traced and key else 'OFF (set LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY in hugging.env)'}")

    q = "What are Bluewater Regional Bank's estimated merger cost synergies?"
    out = rag_pipeline(q)
    print(f"\nQ: {q}")
    print(f"Sources: {out['sources']}")
    print(f"A: {out['answer']}")
