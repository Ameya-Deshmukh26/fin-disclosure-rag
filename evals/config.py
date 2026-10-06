"""
Single place for every knob the eval suite depends on.

The pipeline under test is the best configuration measured in this project:
weighted hybrid retrieval (0.2 dense / 0.8 BM25, MRR 0.810 on the 25-question set),
then parent-document expansion, then generation.

The judge is a DIFFERENT model family from the generator on purpose:
a Llama judge grading Llama answers is the textbook self-preference bias.
"""

import os

# "provider:model" specs (see evals/llm.py). Override with EVAL_GENERATOR / EVAL_JUDGE.
# Default: both on Amazon Bedrock. Generator = Meta Llama 3.1 8B; judge = Amazon Nova Pro,
# a different model family from the generator, so the judge isn't grading its own kind.
GENERATOR_MODEL = os.environ.get("EVAL_GENERATOR", "bedrock:us.meta.llama3-1-8b-instruct-v1:0")
# USD per 1M tokens for the generator, Bedrock on-demand (Llama 3.1 8B Instruct). Used to turn
# traced token counts into cost; update it if the model or the AWS price list changes.
GENERATOR_PRICE_PER_M = {"input": 0.22, "output": 0.22}
JUDGE_MODEL = os.environ.get("EVAL_JUDGE", "bedrock:us.amazon.nova-pro-v1:0")

EMBED_MODEL = "all-mpnet-base-v2"
INDEX_PATH = "index_big.pkl"
DATA_DIR = "data_big"

W_DENSE = 0.2          # weight on dense ranks in weighted RRF (rest goes to BM25)
K_RETRIEVE = 5         # chunks retrieved per retriever before fusion
N_CONTEXT_DOCS = 3     # parent documents handed to the generator
# What the generator is given:
#   "documents"  the top N_CONTEXT_DOCS whole filings (parent-document expansion; fine only
#                because these filings are about 500 characters)
#   "chunks"     only the top N_CONTEXT_CHUNKS matched chunks, each with a metadata header
#                (evals/build_chunk_index.py), so tokens don't grow with document length
CONTEXT_MODE = os.environ.get("EVAL_CONTEXT_MODE", "documents")
CHUNK_SIZE = int(os.environ.get("EVAL_CHUNK_SIZE", "200"))
N_CONTEXT_CHUNKS = int(os.environ.get("EVAL_CONTEXT_CHUNKS", "4"))


def chunk_index_path(size: int) -> str:
    return f"index_chunks_{size}.pkl"
# Metadata pre-filter: restrict search to the filings of the company named in the question.
ENTITY_FILTER = os.environ.get("EVAL_ENTITY_FILTER", "1") == "1"   # on by default; set 0 to reproduce the baseline

THRESHOLD = 0.7        # pass/fail line for judge metrics (0-1 scale)
MAX_WORKERS = 3        # parallel test cases; Bedrock throttled bursts at 4

GOLDENS_RAG = "goldens/rag_goldens.json"
GOLDENS_REFUSAL = "goldens/refusal_goldens.json"
RESULTS_DIR = "results"
