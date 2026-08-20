"""
Generates a larger, more diverse synthetic financial disclosure corpus:
4 companies x 5 filing types = 20 documents, each with distinct facts
so retrieval has real discrimination to do (not trivially easy like 4 docs).
"""
import os

COMPANIES = ["Meridian Financial Corp", "Ashcroft Capital Partners", "Bluewater Regional Bank", "Northgate Holdings"]

TEMPLATES = {
    "earnings": """FORM 8-K EXCERPT - MATERIAL EVENT DISCLOSURE
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 2.02 Results of Operations and Financial Condition.
{company} reported quarterly net revenue of ${revenue} million, a {growth}% change year-over-year.
Operating margin was {margin}%. The company recorded credit loss provisions of ${provision} million.
Management guidance projects revenue growth of {guidance}% for the following fiscal year.
""",
    "risk": """FORM 10-Q EXCERPT - RISK FACTORS
Registrant: {company}
Filing Period: Q{quarter} 2026

Item 1A. Risk Factors.
Our commercial real estate loan portfolio represents approximately {cre_pct}% of total assets.
Non-performing CRE loans stand at {npl_pct}% of the CRE portfolio.
We increased our allowance for credit losses by ${allowance} million in response to these trends.
""",
    "compensation": """FORM 8-K EXCERPT - EXECUTIVE COMPENSATION
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 5.02 Compensation Arrangements.
The Compensation Committee approved {comp_pct}% of CEO variable compensation tied to
risk-adjusted return on capital. Base salary remains ${salary} million.
Target total compensation is ${target_comp} million.
""",
    "merger": """FORM 8-K EXCERPT - MERGER ANNOUNCEMENT
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 1.01 Entry into a Material Definitive Agreement.
{company} entered into a definitive agreement to acquire {target} for approximately
${deal_value} million in an all-stock transaction, expected to close in Q{close_q} 2026.
Estimated annual cost synergies are ${synergies} million.
""",
    "litigation": """FORM 8-K EXCERPT - LEGAL PROCEEDINGS
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 8.01 Other Events.
{company} disclosed a settlement of ${settlement} million related to regulatory
inquiries regarding {topic}. The company admitted no wrongdoing. Settlement is
expected to be paid in full within {payment_months} months.
"""
}

TARGETS = ["Coastal Trust Bank", "Vantage Credit Union", "Pinehill Savings", "Redwood Mutual"]
TOPICS = ["loan servicing practices", "disclosure timing", "data privacy compliance", "fee structures"]

def generate():
    os.makedirs("data_large", exist_ok=True)
    doc_id = 0
    manifest = []
    for i, company in enumerate(COMPANIES):
        for j, (dtype, template) in enumerate(TEMPLATES.items()):
            doc_id += 1
            fname = f"doc_{doc_id:03d}_{company.split()[0].lower()}_{dtype}.txt"
            fill = {
                "company": company, "month": (doc_id % 12) + 1, "day": (doc_id * 3 % 28) + 1,
                "revenue": 200 + doc_id * 17, "growth": (doc_id % 15) + 3, "margin": 18 + (doc_id % 8),
                "provision": 10 + doc_id * 2, "guidance": 5 + (doc_id % 6),
                "quarter": (doc_id % 4) + 1, "cre_pct": 12 + (doc_id % 10), "npl_pct": 1.0 + (doc_id % 5) * 0.3,
                "allowance": 8 + doc_id, "comp_pct": 40 + (doc_id % 5) * 10, "salary": 0.9 + (doc_id % 5) * 0.1,
                "target_comp": 6 + (doc_id % 6), "target": TARGETS[doc_id % len(TARGETS)],
                "deal_value": 200 + doc_id * 23, "close_q": (doc_id % 4) + 1, "synergies": 10 + doc_id * 2,
                "settlement": 5 + (doc_id % 10) * 2, "topic": TOPICS[doc_id % len(TOPICS)],
                "payment_months": 6 + (doc_id % 4) * 3,
            }
            text = template.format(**fill)
            with open(f"data_large/{fname}", "w", encoding="utf-8") as f:
                f.write(text)
            manifest.append({"file": fname, "company": company, "type": dtype, **fill})
    print(f"Generated {doc_id} documents in data_large/")
    return manifest

if __name__ == "__main__":
    generate()
