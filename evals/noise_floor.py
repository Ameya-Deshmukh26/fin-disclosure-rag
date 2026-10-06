"""
Measures how much the judge disagrees with ITSELF.

Re-judges the exact same answers from the latest run (no regeneration), then reports
how far each judge metric moved. That movement is the noise floor: a regression
tolerance below it would flag noise as a regression; far above it would hide real ones.

    python -m evals.noise_floor
"""
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.config import MAX_WORKERS, RESULTS_DIR  # noqa: E402
from evals.judge import Judge  # noqa: E402
from evals.metrics import answer_relevancy, correctness, faithfulness, measure  # noqa: E402

FACTORIES = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy, "correctness": correctness}


def main():
    with open(os.path.join(RESULTS_DIR, "latest.json"), encoding="utf-8") as f:
        run = json.load(f)
    cases = [c for c in run["cases"]["rag"] if "judge" in c]
    judge = Judge()

    def rejudge(c):
        tc = LLMTestCase(input=c["question"], actual_output=c["answer"],
                         expected_output=c["expected_output"], retrieval_context=c["contexts"])
        return measure(tc, FACTORIES, judge)

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        second = list(ex.map(rejudge, cases))

    report = {}
    for name in FACTORIES:
        a = [c["judge"][name]["score"] for c in cases]
        b = [s[name]["score"] for s in second]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        deltas = [abs(x - y) for x, y in pairs]
        flips = sum(1 for c, s in zip(cases, second)
                    if c["judge"][name]["success"] != s[name]["success"])
        report[name] = {
            "n": len(pairs),
            "avg_score_run1": round(statistics.mean(x for x, _ in pairs), 3),
            "avg_score_run2": round(statistics.mean(y for _, y in pairs), 3),
            "avg_delta_of_means": round(abs(statistics.mean(x for x, _ in pairs) - statistics.mean(y for _, y in pairs)), 3),
            "mean_case_delta": round(statistics.mean(deltas), 3),
            "max_case_delta": round(max(deltas), 3),
            "pass_fail_flips": flips,
        }
    with open(os.path.join(RESULTS_DIR, "noise_floor.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for k, v in report.items():
        print(f"{k:<18} mean {v['avg_score_run1']:.3f} vs {v['avg_score_run2']:.3f}  "
              f"(delta {v['avg_delta_of_means']:.3f})  per-case mean {v['mean_case_delta']:.3f} "
              f"max {v['max_case_delta']:.3f}  flips {v['pass_fail_flips']}/{v['n']}")


if __name__ == "__main__":
    main()
