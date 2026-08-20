"""
Harder eval set: questions that require distinguishing between structurally
near-identical documents across 4 companies (same filing types, different
company names/numbers). Tests whether retrieval confuses similar-looking docs.
"""
EVAL_QUESTIONS_LARGE = [
    {"question": "What are Bluewater Regional Bank's estimated merger cost synergies?", "expected_source": "doc_014_bluewater_merger.txt"},
    {"question": "What percentage of Ashcroft Capital Partners' total assets is the CRE loan portfolio?", "expected_source": "doc_007_ashcroft_risk.txt"},
    {"question": "What was Northgate Holdings' quarterly net revenue?", "expected_source": "doc_016_northgate_earnings.txt"},
    {"question": "What percentage of Meridian's CEO variable compensation is tied to risk-adjusted return on capital?", "expected_source": "doc_003_meridian_compensation.txt"},
    {"question": "What was the settlement amount disclosed by Ashcroft Capital Partners regarding regulatory inquiries?", "expected_source": "doc_010_ashcroft_litigation.txt"},
    {"question": "What company did Northgate Holdings agree to acquire?", "expected_source": "doc_019_northgate_merger.txt"},
    {"question": "What was Bluewater Regional Bank's allowance for credit losses increase?", "expected_source": "doc_012_bluewater_risk.txt"},
    {"question": "What is Meridian Financial Corp's target total compensation for the CEO?", "expected_source": "doc_003_meridian_compensation.txt"},
]
