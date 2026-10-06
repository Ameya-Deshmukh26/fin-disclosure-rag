"""
Deterministic checks: zero LLM calls.

The judge is good at fuzzy questions ("is this faithful?"). It is the wrong tool for
questions code can answer exactly. These run first, cost nothing, never hallucinate,
and are what the release gate leans on hardest:

  citation_check  every cited [doc] must be a document we actually gave the model
  number_lock     every number in the answer must appear in the context it was given
                  (catches invented or computed figures, e.g. 157 -> 1,570)
  key_facts       the ground-truth numbers must all appear in the answer
  is_refusal      did the model decline instead of answering?
"""
import re

CITE = re.compile(r"\[(doc_[\w\-]+\.txt)\]")
NUM = re.compile(r"\d+(?:\.\d+)?")
REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "no information",
    "not mentioned",
    "not provided",
    "does not contain",
    "doesn't contain",
)


def citations(answer: str) -> set[str]:
    return set(CITE.findall(answer))


def _strip_citations(text: str) -> str:
    return CITE.sub(" ", text)


def numbers(text: str) -> set[str]:
    out = set()
    for n in NUM.findall(_strip_citations(text).replace(",", "")):
        out.add(n.rstrip("0").rstrip(".") if "." in n else n)
    return out


def citation_check(answer: str, context_sources: list[str], expected_source: str | None = None) -> dict:
    cited = citations(answer)
    return {
        "cited": sorted(cited),
        "has_citation": bool(cited),
        "citations_valid": bool(cited) and cited <= set(context_sources),
        "cites_expected": (expected_source in cited) if expected_source else None,
    }


def number_lock(answer: str, contexts: list[str]) -> dict:
    allowed = set()
    for c in contexts:
        allowed |= numbers(c)
    unsupported = sorted(numbers(answer) - allowed)
    return {"number_lock_ok": not unsupported, "unsupported_numbers": unsupported}


def key_facts(answer: str, expected_numbers: list[str]) -> dict:
    found = numbers(answer)
    missing = [n for n in expected_numbers if n not in found]
    return {"key_facts_ok": not missing, "missing_facts": missing}


def is_refusal(answer: str) -> bool:
    a = answer.lower().replace("’", "'")
    return any(m in a for m in REFUSAL_MARKERS)
