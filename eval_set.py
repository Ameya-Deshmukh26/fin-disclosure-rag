"""
Retrieval-only eval set: for each question, we KNOW which source document contains
the answer. This lets us measure hit_rate@k without needing an LLM judge -
we just check whether a chunk from the correct document appears in the top-k results.
"""

EVAL_QUESTIONS = [
    {"question": "What was Meridian's Q4 net revenue and how much did it grow?", "expected_source": "doc1.txt"},
    {"question": "What percentage of total assets does the CRE loan portfolio represent?", "expected_source": "doc2.txt"},
    {"question": "How is CEO variable compensation tied to risk-adjusted returns?", "expected_source": "doc3.txt"},
    {"question": "What company is Meridian acquiring and for how much?", "expected_source": "doc4.txt"},
    {"question": "What are the estimated annual cost synergies from the merger?", "expected_source": "doc4.txt"},
    {"question": "What was the increase in the allowance for credit losses?", "expected_source": "doc2.txt"},
]
