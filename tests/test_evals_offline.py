"""
Tests for the eval code itself. No network, no models, no keys: safe for every CI push.
An eval harness with a bug in its checks is worse than no harness, because it reports
green while the product is broken.
"""
import pytest

from evals import checks
from evals.registry import compare, rule_for


def test_citations_extracted():
    assert checks.citations("Revenue was $157 million [doc_0001_meridian_earnings.txt].") == {
        "doc_0001_meridian_earnings.txt"}


def test_citation_outside_context_is_invalid():
    r = checks.citation_check("... [doc_0009_x_risk.txt]", ["doc_0001_a_earnings.txt"])
    assert r["has_citation"] and not r["citations_valid"]


def test_no_citation_is_invalid():
    assert not checks.citation_check("Revenue was $157 million.", ["doc_0001_a.txt"])["citations_valid"]


def test_number_lock_catches_inflated_figure():
    ctx = ["reported quarterly net revenue of $157 million, a -3% change"]
    assert checks.number_lock("Revenue was $157 million [doc_0001_a.txt].", ctx)["number_lock_ok"]
    bad = checks.number_lock("Revenue was $1,570 million.", ctx)
    assert not bad["number_lock_ok"] and bad["unsupported_numbers"] == ["1570"]


def test_number_lock_ignores_digits_inside_citations():
    ctx = ["net revenue of $157 million"]
    assert checks.number_lock("$157 million [doc_0001_meridian_earnings.txt]", ctx)["number_lock_ok"]


def test_key_facts():
    assert checks.key_facts("between $186 million and $226 million", ["186", "226"])["key_facts_ok"]
    assert checks.key_facts("between $186 million", ["186", "226"])["missing_facts"] == ["226"]


@pytest.mark.parametrize("text", [
    "I don't have enough information to answer that.",
    "I don’t have enough information to answer that.",
    "The context does not contain information about Meridian Bancorp.",
])
def test_refusal_detected(text):
    assert checks.is_refusal(text)


def test_answer_is_not_refusal():
    assert not checks.is_refusal("Quarterly net revenue was $157 million [doc_0001_a.txt].")


def test_rules():
    assert rule_for("safety.refusal_rate")["kind"] == "gate"
    assert rule_for("rag.faithfulness.avg_score")["kind"] == "guardrail"
    assert rule_for("ops.judge_calls")["kind"] == "info"


def test_gate_regression_blocks():
    v = compare({"safety.refusal_rate": 90.0}, {"safety.refusal_rate": 80.0})
    assert v["verdict"] == "BLOCK"


def test_judge_noise_within_tolerance_is_flat():
    v = compare({"rag.faithfulness.avg_score": 0.90}, {"rag.faithfulness.avg_score": 0.87})
    assert v["verdict"] == "PASS" and v["rows"][0]["status"] == "FLAT"


def test_judge_drop_beyond_tolerance_needs_review():
    v = compare({"rag.correctness.avg_score": 0.90}, {"rag.correctness.avg_score": 0.80})
    assert v["verdict"] == "REVIEW"


def test_latency_is_lower_is_better():
    v = compare({"ops.latency.generate_p95_ms": 1000.0}, {"ops.latency.generate_p95_ms": 1500.0})
    assert v["rows"][0]["status"] == "REVIEW"


def test_extract_json_handles_fences_and_prose():
    pytest.importorskip("deepeval")
    from evals.judge import extract_json
    assert extract_json('Sure!\n```json\n{"verdict": "yes", "reason": "a {b} c"}\n```') == {
        "verdict": "yes", "reason": "a {b} c"}


def test_info_metrics_are_never_judged():
    v = compare({"ops.judge_calls": 290.0}, {"ops.judge_calls": 300.0})
    assert v["rows"][0]["status"] == "info" and v["verdict"] == "PASS"


def test_more_judge_errors_is_flagged():
    v = compare({"ops.judge_errors": 1.0}, {"ops.judge_errors": 3.0})
    assert v["rows"][0]["status"] == "REVIEW"


def test_faithfulness_tolerance_comes_from_measured_noise():
    # 0.05 drop on faithfulness is inside its measured noise floor -> FLAT, not REVIEW
    assert compare({"rag.faithfulness.avg_score": 0.90}, {"rag.faithfulness.avg_score": 0.85})["verdict"] == "PASS"
    # the same drop on correctness (a stable metric) needs review
    assert compare({"rag.correctness.avg_score": 0.90}, {"rag.correctness.avg_score": 0.85})["verdict"] == "REVIEW"
