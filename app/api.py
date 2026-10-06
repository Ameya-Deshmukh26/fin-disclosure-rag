"""
Serving layer: the exact pipeline the eval suite scores, behind a FastAPI app.

  POST /ask   {"question": "..."} -> answer, cited sources, runtime checks, latency
  GET  /health

Runtime guardrails (the same deterministic checks the eval gate uses, applied per request):
  - number lock: if the answer contains a number that is not in the filings it was given,
    the answer is withheld and the user gets a refusal with the reason. A confident wrong
    number never reaches the user.
  - citation check: cited documents must be ones the model was actually given.
  - input limits: question length is capped.

Deployed on AWS Lambda (container image + Lambda Web Adapter + Function URL); generation on
Amazon Bedrock. Run locally with:  uvicorn app.api:app --port 8080
"""
import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from evals import checks
from evals.config import GENERATOR_MODEL
from evals.rag import RAG

MAX_QUESTION_CHARS = 300
REFUSAL = "I don't have enough information to answer that."

app = FastAPI(title="fin-disclosure-rag", version="1.0")


@lru_cache(maxsize=1)
def rag() -> RAG:
    return RAG()   # loads embedder + index once per container


_cache: dict[str, dict] = {}   # exact-match query cache (per container)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


@app.get("/health")
def health():
    return {"status": "ok", "generator": GENERATOR_MODEL, "entity_filter": True}


@app.post("/ask")
async def ask(q: Question):
    key = " ".join(q.question.lower().split())
    t0 = time.perf_counter()
    if key in _cache:
        hit = _cache[key]
        return {**hit, "cache": "hit",
                "latency_ms": {"total": round((time.perf_counter() - t0) * 1000, 2)}}
    try:
        # retrieval is CPU-bound and generation is a blocking SDK call: keep the event loop free
        out = await run_in_threadpool(rag().answer, q.question)
    except Exception as e:  # provider errors surface as 503, never as a made-up answer
        raise HTTPException(status_code=503, detail=f"generation unavailable: {type(e).__name__}")

    lock = checks.number_lock(out["answer"], out["contexts"])
    cites = checks.citation_check(out["answer"], out["context_sources"])
    refused = checks.is_refusal(out["answer"])
    answer, guardrail = out["answer"].strip(), None
    if not lock["number_lock_ok"]:
        answer = REFUSAL
        guardrail = f"withheld: numbers not found in the source filings {lock['unsupported_numbers']}"

    body = {
        "question": q.question,
        "answer": answer,
        "sources": out["context_sources"],
        "checks": {"number_lock_ok": lock["number_lock_ok"],
                   "citations_valid": cites["citations_valid"],
                   "refused": refused or guardrail is not None},
        "guardrail": guardrail,
        "latency_ms": {**{k: round(v) for k, v in out["latency_ms"].items()},
                       "total": round((time.perf_counter() - t0) * 1000)},
    }
    if guardrail is None:
        _cache[key] = body
    return {**body, "cache": "miss"}
