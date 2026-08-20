"""
Generates eval questions programmatically from the corpus so ground truth is EXACT,
not hand-guessed. Full company name + document type uniquely identifies one file.
"""
import glob, os, re, random

QUESTION_BY_TYPE = {
    "earnings":     "What was {company}'s quarterly net revenue?",
    "risk":         "What percentage of {company}'s total assets is the commercial real estate loan portfolio?",
    "compensation": "What is {company}'s target total CEO compensation?",
    "merger":       "What company did {company} agree to acquire and what are the cost synergies?",
    "litigation":   "What settlement amount did {company} disclose for regulatory inquiries?",
    "guidance":     "What is {company}'s revised full-year revenue guidance range?",
}

def build_eval(n=25, seed=7, data_dir="data_big"):
    random.seed(seed)
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    picked = random.sample(files, n)
    evalset = []
    for fp in picked:
        text = open(fp, encoding="utf-8").read()
        company = re.search(r"Registrant: (.+)", text).group(1).strip()
        dtype = os.path.basename(fp).rsplit("_", 1)[-1].replace(".txt", "")
        q = QUESTION_BY_TYPE[dtype].format(company=company)
        evalset.append({"question": q, "expected_source": os.path.basename(fp)})
    return evalset

if __name__ == "__main__":
    ev = build_eval()
    print(f"{len(ev)} questions generated. Sample:")
    for e in ev[:5]:
        print(f"  {e['expected_source']}")
        print(f"    {e['question']}")
