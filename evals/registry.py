"""
How a change in each metric is judged when comparing a run to the baseline.

Three questions per metric:
  DIRECTION  higher-is-better (faithfulness) or lower-is-better (latency)?
  KIND       gate      -> any regression beyond tolerance BLOCKS the release
             guardrail -> regression is flagged for a human (REVIEW), not blocked
             info      -> tracked, never affects the verdict
  TOLERANCE  how big a move is real versus run-to-run noise?

What is gated, and why:
  - Deterministic metrics (refusal rate, number lock, valid citations, key facts) are
    exact and cheap, so they are hard GATES with zero tolerance. For a ratings product,
    answering about a company that does not exist, or printing a number that is not in
    the source, is the failure that cannot ship.
  - Judge metrics wobble run to run, so they are GUARDRAILS on the average score with
    a tolerance above the measured noise floor (see `python -m evals.noise_floor`).
  - Latency is a guardrail with a relative tolerance; provider latency is noisy.
"""

GATE_EXACT = {"direction": "higher", "kind": "gate", "tol": 0.0, "rel_tol": 0.0}
JUDGE_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.05, "rel_tol": 0.0}
RETRIEVAL_GUARD = {"direction": "higher", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.0}
LATENCY_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.30}
ERROR_GUARD = {"direction": "lower", "kind": "guardrail", "tol": 0.0, "rel_tol": 0.0}
INFO = {"direction": "higher", "kind": "info", "tol": 0.0, "rel_tol": 0.0}

GATED = {
    "safety.refusal_rate",
    "rag.number_lock_rate",
    "rag.citations_valid_rate",
    "rag.key_facts_rate",
}


# Per-metric tolerances from the MEASURED noise floor (evals/noise_floor.py, Oct 5 2026,
# Nova Pro judge re-scoring identical answers): the faithfulness mean moved 0.040 (and 0.052
# between two runs with 24/25 identical answers), correctness 0.016, relevancy 0.010.
# Tolerance sits about 1.5x above the observed movement, so noise reads as FLAT.
JUDGE_TOLERANCE = {"faithfulness": 0.08, "correctness": 0.03, "answer_relevancy": 0.02}


def rule_for(metric_id: str) -> dict:
    if metric_id in GATED:
        return GATE_EXACT
    if metric_id.endswith(".avg_score"):
        for name, tol in JUDGE_TOLERANCE.items():
            if f".{name}" in metric_id:
                return {**JUDGE_GUARD, "tol": tol}
        return JUDGE_GUARD
    if metric_id.startswith("retrieval."):
        return RETRIEVAL_GUARD
    if metric_id.startswith("ops.latency.") and metric_id.endswith("p95_ms"):
        return LATENCY_GUARD
    if metric_id == "ops.judge_errors":
        # more judge errors means fewer cases actually scored: the averages are less trustworthy
        return ERROR_GUARD
    return INFO


def compare(baseline: dict, current: dict) -> dict:
    """Return per-metric verdicts and an overall verdict: PASS, REVIEW or BLOCK."""
    rows, overall = [], "PASS"
    for mid, base in sorted(baseline.items()):
        cur = current.get(mid)
        if not isinstance(base, (int, float)) or not isinstance(cur, (int, float)):
            continue
        rule = rule_for(mid)
        if rule["kind"] == "info":   # tracked only: never judged better or worse
            rows.append({"metric": mid, "baseline": base, "current": cur, "kind": "info", "status": "info"})
            continue
        worse = (base - cur) if rule["direction"] == "higher" else (cur - base)
        allowed = max(rule["tol"], rule["rel_tol"] * abs(base))
        if worse > allowed + 1e-9:
            status = {"gate": "BLOCK", "guardrail": "REVIEW"}.get(rule["kind"], "info")
        elif worse < -allowed - 1e-9:
            status = "IMPROVED"
        else:
            status = "FLAT"
        if status == "BLOCK":
            overall = "BLOCK"
        elif status == "REVIEW" and overall == "PASS":
            overall = "REVIEW"
        rows.append({"metric": mid, "baseline": base, "current": cur,
                     "kind": rule["kind"], "status": status})
    return {"verdict": overall, "rows": rows}
