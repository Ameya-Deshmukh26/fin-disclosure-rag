"""
Query router: picks the cheapest safe path for a question BEFORE any search or model call.

Online monitoring showed every question paid for a model call (about 500 tokens), even
"Who won the Super Bowl?" and attack prompts the model ended up refusing anyway. The
router is plain rules, so routing itself costs 0 tokens and well under a millisecond
(a small LLM classifier would add a paid call to every request just to decide the path).

  route       when                                          what happens            tokens
  injection   known attack phrasing                         refuse                  0
  company     names a company we have filings for           filtered search + LLM   ~500
  aggregate   "which company has the largest settlement"    lookup in a facts table 0
  smalltalk   greeting, "who are you", "thanks"             fixed reply             0
  off_topic   no company named and no finance words         refuse                  0
  search      finance question, company not recognized      full search + LLM       ~500

Order matters: a question that names a company always gets the full pipeline, so the
router only saves tokens on questions that never needed the model. "search" keeps unknown
companies (fakes, typos) on the model path, where refusing them is already tested.

The aggregate route answers from a facts table extracted from the filings with the same
regexes that build the golden answers, so its numbers are exact and every answer cites its
filing. Search can only hand the model 3 filings; a question about all 100 companies needs
every row, which is a database lookup, not retrieval.

    python -m evals.router      # route every known question and estimate the token savings
"""
import importlib.util
import json
import os
import pathlib
import re
from dataclasses import dataclass, field

from evals import checks
from evals.config import DATA_DIR, GOLDENS_RAG, GOLDENS_REFUSAL

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build_goldens", ROOT / "goldens" / "build_goldens.py")
_bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bg)
EXTRACT = _bg.EXTRACT

REFUSAL = "I don't have enough information to answer that."
OFF_TOPIC_REPLY = REFUSAL + " I can only answer questions about the company filings I have."
SMALLTALK_REPLY = ("Hi! I answer questions about the company filings I have: quarterly revenue, "
                   "CEO pay, real estate risk, mergers, regulatory settlements and revenue guidance. "
                   "For example: What was Harborview Financial Corp's quarterly net revenue?")

SMALLTALK = re.compile(
    r"^\W*(hi|hello|hey|yo|good (morning|afternoon|evening)|thanks|thank you)\b"
    r"|\bhow are you\b|\b(your|ur) name\b|\bwho are you\b|\bwhat (can|do) you do\b|\bwhat are you\b",
    re.I)
FINANCE = re.compile(
    r"\b(revenue|earnings|income|profit|loss|quarter\w*|fiscal|guidance|forecast|outlook|"
    r"settle\w*|lawsuit|litigation|regulat\w*|penalt\w*|compensation|pay|paid|salary|ceo|cfo|"
    r"executive|merger|acqui\w*|buy\w*|synerg\w*|deal|risk\w*|exposure|loans?|real estate|cre|"
    r"assets?|dividend\w*|stock|shares?|filing\w*|8-k|10-k|sec|bank\w*|bancorp|holdings|corp\w*|"
    r"compan(y|ies)|trust|capital partners|branch\w*|employees|credit|rating|financ\w*)\b",
    re.I)
SUPERLATIVE = re.compile(r"\b(largest|biggest|highest|most|top|greatest|smallest|lowest|least)\b", re.I)
LOW = re.compile(r"\b(smallest|lowest|least)\b", re.I)
WHICH = re.compile(r"\b(which|what|who)\b.{0,30}\b(company|companies|bank|banks|firm|registrant)\b"
                   r"|\bwho (has|had|paid|disclosed)\b", re.I)
# (words in the question, filing type, EXTRACT group holding the number, label)
METRICS = [
    (r"settle|penalt|regulat", "litigation", 0, "regulatory settlement"),
    (r"synerg", "merger", 1, "estimated annual cost synergies"),
    (r"compensation|\bpay\b|\bpaid\b|\bceo\b", "compensation", 0, "target total CEO compensation"),
    (r"real estate|\bcre\b|exposure", "risk", 0, "commercial real estate exposure"),
    (r"revenue|earnings|sales", "earnings", 0, "quarterly net revenue"),
]
GUIDANCE = re.compile(r"\b(guidance|forecast|outlook|range)\b|\bexpect\w*\b.{0,20}\brevenue\b", re.I)


def build_facts(data_dir: str = DATA_DIR) -> tuple[list[str], list[dict]]:
    """(every registrant, one row per filing whose key figure the golden regexes can read)."""
    companies, facts = set(), []
    for name in sorted(os.listdir(data_dir)):
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            text = f.read()
        reg = re.search(r"Registrant: (.+)", text)
        if not reg:
            continue
        company = reg.group(1).strip()
        companies.add(company)
        dtype = name.rsplit("_", 1)[-1].removesuffix(".txt")
        m = re.search(EXTRACT[dtype][0], text) if dtype in EXTRACT else None
        if m:
            facts.append({"company": company, "type": dtype, "values": m.groups(), "source": name})
    return sorted(companies), facts


@dataclass
class Route:
    name: str
    answer: str | None = None          # set when the router answers by itself: no model call
    sources: list[str] = field(default_factory=list)
    company: str | None = None

    @property
    def needs_model(self) -> bool:
        return self.answer is None


class Router:
    def __init__(self, companies: list[str], facts: list[dict]):
        self.companies = sorted(companies, key=len, reverse=True)   # longest name wins
        self.facts = facts

    @classmethod
    def load(cls, data_dir: str = DATA_DIR) -> "Router":
        return cls(*build_facts(data_dir))

    def company_in(self, question: str) -> str | None:
        q = question.lower()
        return next((c for c in self.companies if c.lower() in q), None)

    def _metric(self, question: str):
        if GUIDANCE.search(question):
            return None   # a range, not one number: leave it to the model path
        return next((m for m in METRICS if re.search(m[0], question, re.I)), None)

    def aggregate(self, question: str, metric) -> Route:
        _, dtype, group, label = metric
        rows = [r for r in self.facts if r["type"] == dtype]
        low = bool(LOW.search(question))
        best = (min if low else max)(float(r["values"][group]) for r in rows)
        winners = [r for r in rows if float(r["values"][group]) == best]
        value = winners[0]["values"][group]
        amount = f"{value}%" if dtype == "risk" else f"${value} million"
        names = [f"{r['company']} [{r['source']}]" for r in winners]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        word = "smallest" if low else "largest"
        # no counts in the text: every number in an answer must appear in a cited filing
        answer = f"The {word} {label} in any {dtype} filing is {amount}: {who}."
        return Route("aggregate", answer, [r["source"] for r in winners])

    def route(self, question: str) -> Route:
        if checks.is_injection(question):
            return Route("injection", REFUSAL)
        company = self.company_in(question)
        if company:
            return Route("company", company=company)
        if SUPERLATIVE.search(question) and WHICH.search(question):
            metric = self._metric(question)
            if metric:
                return self.aggregate(question, metric)
        if not FINANCE.search(question):
            if SMALLTALK.search(question):
                return Route("smalltalk", SMALLTALK_REPLY)
            return Route("off_topic", OFF_TOPIC_REPLY)
        return Route("search")


SMALLTALK_SAMPLES = [
    "hey how are you, give me your name?",
    "Hi! What can you do?",
    "Thanks, that helped.",
    "Good morning",
]


def main():
    router = Router.load()
    with open(GOLDENS_RAG, encoding="utf-8") as f:
        goldens = [g["question"] for g in json.load(f)]
    with open(GOLDENS_REFUSAL, encoding="utf-8") as f:
        refusals = [g["question"] for g in json.load(f)]
    traffic = []
    for path in ("results/online_sim.json", "results/online_attacks.json"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                traffic += json.load(f)["rows"]

    # Safety first: every golden question must still reach the model, and no trick question
    # may get a confident answer from the router.
    lost = [q for q in goldens if router.route(q).name != "company"]
    unsafe = [q for q in refusals if (r := router.route(q)).answer and not checks.is_refusal(r.answer)]
    print(f"golden questions kept on the model path: {len(goldens) - len(lost)}/{len(goldens)}")
    print(f"trick questions answered by the router:  {len(unsafe)} (must be 0)")
    for q in lost + unsafe:
        print("   !!", q)

    print(f"\n{'route':<10} {'tokens before':>13}  question")
    seen, before, after = set(), 0, 0
    for r in traffic:
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        route = router.route(r["question"])
        before += r["total_tokens"]
        after += 0 if not route.needs_model else r["total_tokens"]
        mark = "" if route.needs_model else "  (saved)"
        print(f"{route.name:<10} {r['total_tokens']:>13}  {r['question'][:58]}{mark}")
        if route.name == "aggregate":
            print(f"{'':<26}-> {route.answer}")
    n = len(seen)
    print(f"\nrecorded traffic: {n} questions, avg tokens {before / n:.0f} before, {after / n:.0f} "
          f"with the router ({100 * (before - after) / before:.0f}% fewer)")
    print("\nsmalltalk samples:")
    for q in SMALLTALK_SAMPLES:
        print(f"  {router.route(q).name:<10} {q}")


if __name__ == "__main__":
    main()
