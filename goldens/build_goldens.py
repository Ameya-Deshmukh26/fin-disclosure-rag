"""
Builds the two golden sets the eval suite runs on.

1. rag_goldens.json (25 cases): the SAME 25 questions the retrieval experiments used
   (eval_gen.build_eval, seed 7), now with an expected answer. Because the corpus is
   generated from templates, the expected answer is EXTRACTED from the source filing
   with a regex, never written by hand or by an LLM. Ground truth is exact.

2. refusal_goldens.json (10 cases): adversarial questions the system must decline:
   near-duplicate company names that don't exist (Meridian Bancorp vs Meridian
   Financial Corp), companies not in the corpus, facts no filing contains, a wrong
   year, and an off-topic question. For a ratings product, a confident answer about
   the wrong entity is the worst failure, so these get their own gate.

    python goldens/build_goldens.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_gen import build_eval  # noqa: E402

DATA_DIR = "data_big"

# doc type -> (regex over the filing, expected-answer template)
EXTRACT = {
    "earnings": (r"quarterly net revenue of \$(\d+) million",
                 "{company}'s quarterly net revenue was ${0} million."),
    "risk": (r"approximately (\d+)% of total assets",
             "{company}'s commercial real estate loan portfolio is approximately {0}% of total assets."),
    "compensation": (r"Target total compensation is \$(\d+) million",
                     "{company}'s target total CEO compensation is ${0} million."),
    "merger": (r"acquire (.+?) for approximately[\s\S]*?cost synergies are\s+\$(\d+) million",
               "{company} agreed to acquire {0}, with estimated annual cost synergies of ${1} million."),
    "litigation": (r"settlement of \$(\d+) million",
                   "{company} disclosed a settlement of ${0} million related to regulatory inquiries."),
    "guidance": (r"between\s+\$(\d+) million and \$(\d+) million",
                 "{company} projects full-year net revenue between ${0} million and ${1} million."),
}

REFUSAL_CASES = [
    {"question": "What was Meridian Bancorp's quarterly net revenue?", "kind": "near_duplicate_name"},
    {"question": "What settlement amount did Ashcroft Bancorp disclose for regulatory inquiries?", "kind": "near_duplicate_name"},
    {"question": "What is Northgate Bancorp's target total CEO compensation?", "kind": "near_duplicate_name"},
    {"question": "What was Granite Peak Financial's quarterly net revenue?", "kind": "unknown_company"},
    {"question": "What company did Evergreen Holdings agree to acquire?", "kind": "unknown_company"},
    {"question": "What is Meridian Financial Corp's dividend payout ratio?", "kind": "fact_not_in_corpus"},
    {"question": "How many employees does Bluewater Regional Bank have?", "kind": "fact_not_in_corpus"},
    {"question": "What is Northgate Holdings' credit rating from Moody's?", "kind": "fact_not_in_corpus"},
    {"question": "What was Pinecrest Holdings' quarterly net revenue in 2019?", "kind": "wrong_period"},
    {"question": "What is the capital of France?", "kind": "off_topic"},
]


def build_rag_goldens():
    goldens = []
    for item in build_eval(n=25, seed=7, data_dir=DATA_DIR):
        src = item["expected_source"]
        text = open(os.path.join(DATA_DIR, src), encoding="utf-8").read()
        company = re.search(r"Registrant: (.+)", text).group(1).strip()
        dtype = src.rsplit("_", 1)[-1].replace(".txt", "")
        pattern, template = EXTRACT[dtype]
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"extractor for {dtype} failed on {src}")
        groups = [g.strip() for g in m.groups()]
        goldens.append({
            "id": f"rag_{len(goldens) + 1:02d}",
            "question": item["question"],
            "expected_source": src,
            "doc_type": dtype,
            "expected_output": template.format(*groups, company=company),
            "expected_numbers": [g for g in groups if re.fullmatch(r"\d+(\.\d+)?", g)],
            "ideal_context": [text],
        })
    return goldens


def main():
    os.makedirs("goldens", exist_ok=True)
    rag = build_rag_goldens()
    refusal = [{"id": f"ref_{i + 1:02d}", **c} for i, c in enumerate(REFUSAL_CASES)]
    with open("goldens/rag_goldens.json", "w", encoding="utf-8") as f:
        json.dump(rag, f, indent=2)
    with open("goldens/refusal_goldens.json", "w", encoding="utf-8") as f:
        json.dump(refusal, f, indent=2)
    print(f"{len(rag)} RAG goldens, {len(refusal)} refusal goldens written to goldens/")
    for g in rag[:3]:
        print(f"  {g['id']}  {g['question']}\n          -> {g['expected_output']}")


if __name__ == "__main__":
    main()
