"""
Online evaluation: MLflow's built-in LLM scorers run on the traces from online_sim.

None of these need an answer key, which is what makes them usable on live traffic:
  Safety                 is the response harmful or toxic?
  RetrievalGroundedness  is the response supported by the retrieved filings? (faithfulness)
  RelevanceToQuery       does the response address what was asked?

The judge is Amazon Nova Pro on Bedrock (through LiteLLM), the same judge as the offline
suite, and a different model family from the Llama generator.

    python -m evals.online_scorers            # scores the latest online_sim run
    python -m evals.online_scorers --limit 2  # quick smoke test
"""
import argparse
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, Safety  # noqa: E402

from evals.tracking import _mlflow  # noqa: E402

os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")   # new-account Bedrock quotas throttle bursts
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_RETRIES", "8")   # back off and retry rate-limited judge calls
JUDGE = os.environ.get("ONLINE_JUDGE", "bedrock:/us.amazon.nova-pro-v1:0")


def _aws_env_from_profile():
    """MLflow's Bedrock judge reads AWS keys from env vars, not from ~/.aws like boto3.
    Copy the active boto3 credentials into this process's environment (never printed
    or written anywhere), so the judge uses the same identity as the rest of the suite."""
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return
    import boto3
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    os.environ.setdefault("AWS_REGION", "us-east-1")
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")


def _route_bedrock_judges_through_litellm():
    """MLflow 3.16 sends 'bedrock:/' judge calls to its native gateway adapter, whose
    AmazonBedrockProvider does not implement get_endpoint_url yet. LiteLLM supports
    Bedrock fully, so skip the gateway adapter for Bedrock and let MLflow fall through
    to its LiteLLM adapter (next in its own adapter priority list)."""
    from mlflow.genai.judges.adapters import gateway_adapter
    original = gateway_adapter.GatewayAdapter.is_applicable.__func__

    @classmethod
    def is_applicable(cls, model_uri, prompt):
        if str(model_uri).startswith("bedrock:/"):
            return False
        return original(cls, model_uri=model_uri, prompt=prompt)

    gateway_adapter.GatewayAdapter.is_applicable = is_applicable


def main():
    _aws_env_from_profile()
    _route_bedrock_judges_through_litellm()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    _mlflow()
    runs = mlflow.search_runs(filter_string="tags.stage = 'online_monitoring'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    run_id = runs.iloc[0]["run_id"]
    traces = mlflow.search_traces(run_id=run_id, max_results=args.limit or 1000)
    print(f"scoring {len(traces)} traces from online run {run_id} with judge {JUDGE}")
    result = mlflow.genai.evaluate(
        data=traces,
        scorers=[Safety(model=JUDGE), RetrievalGroundedness(model=JUDGE), RelevanceToQuery(model=JUDGE)],
    )
    for k, v in sorted(result.metrics.items()):
        print(f"  {k:<45} {v}")


if __name__ == "__main__":
    main()
