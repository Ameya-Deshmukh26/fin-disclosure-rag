"""
Chunk-level index for long documents: small chunks, each carrying its own identity.

The default pipeline sends whole filings to the model (parent-document expansion). That
only works because these filings are about 500 characters. A real 10-K is hundreds of
pages, so the model must get only the matched chunks, and a small chunk has a problem:
the sentence with the number ("a settlement of $28 million") often does not contain the
company name, which lives in the header. A right number for the wrong company is the
worst error in this domain.

Fix: a contextual header on every chunk, built from the filing's metadata:

    [Harborview Financial Corp | FORM 8-K EXCERPT - EXECUTIVE COMPENSATION | 2026-10-12]
    Target total compensation is $5 million, with maximum payout of $24 million.

The header is embedded and BM25-indexed with the text, and the model sees it too, so every
chunk says whose number it is. Tokens per question then depend on chunk size and count,
not on document length.

    python -m evals.build_chunk_index            # 200-char chunks -> index_chunks_200.pkl
    python -m evals.build_chunk_index --size 300
"""
import argparse
import os
import pickle
import re
import time

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evals.config import DATA_DIR, EMBED_MODEL, chunk_index_path


def split_filing(text: str, splitter) -> tuple[dict, list[str]]:
    """(metadata from the header lines, header-less body chunks)."""
    lines = text.splitlines()
    meta = {"form": lines[0].strip(),
            "company": re.search(r"Registrant: (.+)", text).group(1).strip(),
            "date": (re.search(r"Filing (?:Date|Period): (.+)", text) or [None, ""])[1].strip()}
    # Filings are hard-wrapped mid-sentence. Splitting on those line breaks cut "net revenue
    # between" from "$540 million and $580 million", so the chunk that matched the question
    # had no numbers and the chunk with the numbers matched nothing (answer reached the model
    # in 92% of golden questions). Unwrap first, then split at sentence ends.
    body = " ".join(l.strip() for l in lines[1:]
                    if l.strip() and not l.startswith(("Registrant:", "Filing Date:", "Filing Period:")))
    return meta, splitter.split_text(body)


def build(size: int, overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=[". ", "; ", ", ", " ", ""],
                                              keep_separator="end")
    chunks = []
    for name in sorted(os.listdir(DATA_DIR)):
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            meta, parts = split_filing(f.read(), splitter)
        header = f"[{meta['company']} | {meta['form']} | {meta['date']}]"
        for i, part in enumerate(parts):
            chunks.append({"text": f"{header}\n{part}", "source": name, "chunk": i,
                           "company": meta["company"]})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=0)   # sentence boundaries make overlap unneeded
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer   # heavy; only needed to embed
    t0 = time.perf_counter()
    chunks = build(args.size, args.overlap)
    emb = SentenceTransformer(EMBED_MODEL).encode([c["text"] for c in chunks], batch_size=64,
                                                  show_progress_bar=False)
    with open(chunk_index_path(args.size), "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": np.asarray(emb, dtype=np.float32),
                     "chunk_size": args.size, "overlap": args.overlap}, f)
    per_doc = len(chunks) / len({c["source"] for c in chunks})
    print(f"{len(chunks)} chunks ({per_doc:.1f} per filing, size {args.size}, overlap {args.overlap}) "
          f"-> {chunk_index_path(args.size)} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
