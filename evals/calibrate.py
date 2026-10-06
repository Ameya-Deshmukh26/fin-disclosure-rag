"""
Judge calibration: don't trust a judge you haven't measured.

  python -m evals.calibrate export   # writes results/calibration.csv from the latest run
  (open the CSV, fill human_correct and human_faithful with 1 or 0 for each row)
  python -m evals.calibrate score    # agreement + Cohen's kappa: judge vs you,
                                     # and deterministic check vs you

Cohen's kappa corrects raw agreement for chance: if 90% of answers are correct, a judge
that always says "pass" agrees 90% of the time while knowing nothing. Kappa near 0
means chance-level; above ~0.6 is substantial agreement.
"""
import csv
import json
import os
import sys

from evals.config import RESULTS_DIR

CSV_PATH = os.path.join(RESULTS_DIR, "calibration.csv")
FIELDS = ["id", "question", "expected_output", "answer",
          "judge_correctness_score", "judge_correctness_pass",
          "judge_faithfulness_score", "judge_faithfulness_pass",
          "deterministic_key_facts_ok", "human_correct", "human_faithful", "notes"]


def export():
    with open(os.path.join(RESULTS_DIR, "latest.json"), encoding="utf-8") as f:
        run = json.load(f)
    rows = []
    for c in run["cases"]["rag"]:
        j = c.get("judge", {})
        rows.append({
            "id": c["id"], "question": c["question"], "expected_output": c["expected_output"],
            "answer": c["answer"],
            "judge_correctness_score": j.get("correctness", {}).get("score"),
            "judge_correctness_pass": int(bool(j.get("correctness", {}).get("success"))),
            "judge_faithfulness_score": j.get("faithfulness", {}).get("score"),
            "judge_faithfulness_pass": int(bool(j.get("faithfulness", {}).get("success"))),
            "deterministic_key_facts_ok": int(bool(c["key_facts_ok"])),
            "human_correct": "", "human_faithful": "", "notes": "",
        })
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH}. Label human_correct / human_faithful with 1 or 0.")


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _pairs(rows, human_col, other_col):
    pairs = [(int(r[human_col]), int(r[other_col])) for r in rows
             if r[human_col].strip() in ("0", "1") and str(r[other_col]).strip() in ("0", "1")]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def score():
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    report = {}
    for label, human, other in [("judge_correctness", "human_correct", "judge_correctness_pass"),
                                ("deterministic_key_facts", "human_correct", "deterministic_key_facts_ok"),
                                ("judge_faithfulness", "human_faithful", "judge_faithfulness_pass")]:
        h, o = _pairs(rows, human, other)
        if not h:
            continue
        agree = sum(x == y for x, y in zip(h, o)) / len(h)
        report[label] = {"n": len(h), "agreement": round(agree, 3), "kappa": round(kappa(h, o), 3),
                         "judge_says_pass_human_says_fail": sum(1 for x, y in zip(h, o) if x == 0 and y == 1),
                         "judge_says_fail_human_says_pass": sum(1 for x, y in zip(h, o) if x == 1 and y == 0)}
    with open(os.path.join(RESULTS_DIR, "calibration_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        print(f"{k:<26} n={v['n']:<3} agreement={v['agreement']:.0%}  kappa={v['kappa']:.2f}  "
              f"false-pass={v['judge_says_pass_human_says_fail']}  false-fail={v['judge_says_fail_human_says_pass']}")


if __name__ == "__main__":
    {"export": export, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
