"""
The judge metrics, built fresh per test case (DeepEval metric objects hold state, so
sharing one across threads would mix up scores and reasons).

  Faithfulness     claims in the answer that are supported by the retrieved context
                   (DeepEval: extract claims, then verdict each against the context)
  AnswerRelevancy  statements in the answer that actually address the question
  Correctness      GEval with explicit steps, comparing against the regex-extracted
                   ground truth. Steps instead of a one-line criterion make the judge
                   more consistent run to run.

Faithfulness and correctness are different questions: an answer can be perfectly
faithful to the wrong filing. That is why both exist.
"""
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import SingleTurnParams

from evals.config import THRESHOLD


def faithfulness(judge):
    return FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False)


def answer_relevancy(judge):
    return AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False)


# v1 said "heavily penalize any number in the actual output that differs from the expected
# output". The baseline showed the judge reading that as "penalize any EXTRA number": two
# complete, correct merger answers that also stated the (true, in-filing) deal value scored
# 0.3. The rubric was measuring "matches the reference sentence", not "is correct".
# v2 penalizes only missing or contradicted facts; whether extra facts are grounded is
# faithfulness's job, not correctness's.
RUBRIC_VERSION = "v2"


def correctness(judge):
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Identify the facts in the expected output: company names, counterparties and numbers.",
            "Check whether the actual output states each of those facts with the same value.",
            "Penalize a missing expected fact, or a stated value that contradicts an expected fact.",
            "Penalize answers about a different company than the one asked about.",
            "Do NOT penalize additional facts that are not in the expected output; whether extra "
            "facts are supported by the source is measured separately by faithfulness.",
            "Do not penalize wording, citations, or style.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                           SingleTurnParams.EXPECTED_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        async_mode=False,
    )


def measure(test_case, factories, judge) -> dict:
    """Run each metric on one test case; a judge failure is recorded, not swallowed."""
    out = {}
    for name, factory in factories.items():
        m = factory(judge)
        try:
            m.measure(test_case)
            out[name] = {"score": m.score, "success": bool(m.success), "reason": m.reason}
        except Exception as e:
            out[name] = {"score": None, "success": False, "reason": None, "error": str(e)[:300]}
    return out
