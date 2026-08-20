"""
Scales the corpus to ~600 documents so latency/scale effects are measurable.
Brute-force vs ANN, and caching, only show real differences at scale -
with 24 chunks everything is instant and the comparison is meaningless.
"""
import os, random

random.seed(42)

FIRST = ["Meridian","Ashcroft","Bluewater","Northgate","Pinecrest","Harborview","Stonebridge","Fairmont",
         "Calderwood","Westmark","Ridgeline","Silverton","Oakhaven","Brightwater","Kingsley","Thornfield",
         "Lakemont","Ironwood","Crestline","Summitvale"]
SUFFIX = ["Financial Corp","Capital Partners","Regional Bank","Holdings","Trust Company","Bancorp"]

TEMPLATES = {
"earnings":"""FORM 8-K EXCERPT - RESULTS OF OPERATIONS
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 2.02 Results of Operations and Financial Condition.
{company} reported quarterly net revenue of ${revenue} million, a {growth}% change year-over-year.
Operating margin was {margin}%, compared to {margin_prior}% in the prior year period.
The company recorded credit loss provisions of ${provision} million this quarter.
Diluted earnings per share were ${eps}. Management guidance projects revenue growth
of {guidance}% for the following fiscal year, contingent on macroeconomic conditions.
""",
"risk":"""FORM 10-Q EXCERPT - RISK FACTORS
Registrant: {company}
Filing Period: Q{quarter} 2026

Item 1A. Risk Factors.
Our commercial real estate loan portfolio represents approximately {cre_pct}% of total assets.
Non-performing CRE loans stand at {npl_pct}% of the CRE portfolio, compared to {npl_prior}%
in the prior quarter. We increased our allowance for credit losses by ${allowance} million.
Concentration risk in {region} markets represents a material exposure. Management believes
current reserves are adequate but continued deterioration could require further provisions.
""",
"compensation":"""FORM 8-K EXCERPT - EXECUTIVE COMPENSATION
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 5.02 Compensation Arrangements.
The Compensation Committee approved a structure tying {comp_pct}% of CEO variable
compensation to risk-adjusted return on capital. Base salary remains ${salary} million.
Target total compensation is ${target_comp} million, with maximum payout of ${max_comp} million.
The change responds to shareholder feedback on compensation alignment with credit risk outcomes.
""",
"merger":"""FORM 8-K EXCERPT - MATERIAL DEFINITIVE AGREEMENT
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 1.01 Entry into a Material Definitive Agreement.
{company} entered a definitive agreement to acquire {target} for approximately
${deal_value} million in an all-stock transaction, expected to close in Q{close_q} 2026
subject to regulatory approval. The combined entity would hold approximately
${combined_assets} billion in total assets. Estimated annual cost synergies are
${synergies} million, with the transaction accretive to EPS within {accretive} months.
""",
"litigation":"""FORM 8-K EXCERPT - LEGAL PROCEEDINGS
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 8.01 Other Events.
{company} disclosed a settlement of ${settlement} million related to regulatory
inquiries regarding {topic}. The company admitted no wrongdoing as part of the
resolution. The settlement is expected to be paid within {payment_months} months.
An additional ${reserve} million reserve was established for related matters.
""",
"guidance":"""FORM 8-K EXCERPT - FORWARD GUIDANCE REVISION
Registrant: {company}
Filing Date: 2026-{month:02d}-{day:02d}

Item 2.02 Results of Operations and Financial Condition.
{company} revised full-year guidance, now projecting net revenue between
${low} million and ${high} million, versus prior guidance of ${prior_low} million
to ${prior_high} million. The revision reflects {reason}. Expected full-year
operating margin is {fy_margin}%. Capital expenditure guidance is ${capex} million.
""",
}

TARGETS = ["Coastal Trust Bank","Vantage Credit Union","Pinehill Savings","Redwood Mutual",
           "Granite State Bank","Bayfront Financial","Cedar Valley Trust","Highland Federal"]
TOPICS = ["loan servicing practices","disclosure timing","data privacy compliance","fee structures",
          "anti-money-laundering controls","overdraft fee assessment","mortgage underwriting standards"]
REGIONS = ["Northeast","Mid-Atlantic","Southeast","Midwest","Pacific Northwest","Southwest"]
REASONS = ["softer loan demand","improved net interest margin","elevated credit costs",
           "stronger fee income","integration expenses from recent acquisitions"]

def generate(out_dir="data_big"):
    os.makedirs(out_dir, exist_ok=True)
    doc_id = 0
    for first in FIRST:
        for suf in SUFFIX[:5]:
            company = f"{first} {suf}"
            for dtype, template in TEMPLATES.items():
                doc_id += 1
                d = doc_id
                fill = {
                 "company":company,"month":(d%12)+1,"day":(d*3%28)+1,
                 "revenue":150+d*7,"growth":(d%19)-4,"margin":16+(d%12),"margin_prior":15+(d%11),
                 "provision":8+d%40,"eps":round(0.8+(d%30)*0.11,2),"guidance":3+(d%9),
                 "quarter":(d%4)+1,"cre_pct":9+(d%16),"npl_pct":round(0.8+(d%9)*0.25,2),
                 "npl_prior":round(0.6+(d%7)*0.22,2),"allowance":6+d%35,"region":REGIONS[d%len(REGIONS)],
                 "comp_pct":35+(d%7)*5,"salary":round(0.85+(d%8)*0.09,2),"target_comp":5+(d%9),
                 "max_comp":11+(d%14),"target":TARGETS[d%len(TARGETS)],"deal_value":160+d*11,
                 "close_q":(d%4)+1,"combined_assets":round(6+(d%22)*0.7,1),"synergies":8+d%28,
                 "accretive":12+(d%4)*6,"settlement":4+(d%17)*2,"topic":TOPICS[d%len(TOPICS)],
                 "payment_months":6+(d%4)*3,"reserve":2+(d%9),
                 "low":180+d*6,"high":220+d*6,"prior_low":175+d*6,"prior_high":230+d*6,
                 "reason":REASONS[d%len(REASONS)],"fy_margin":15+(d%11),"capex":20+d%50,
                }
                fname = f"doc_{doc_id:04d}_{first.lower()}_{dtype}.txt"
                with open(os.path.join(out_dir,fname),"w",encoding="utf-8") as f:
                    f.write(template.format(**fill))
    print(f"Generated {doc_id} documents in {out_dir}/")

if __name__ == "__main__":
    generate()
