# Read-only MLflow dashboard snapshot on AWS Lambda (Lambda Web Adapter + Function URL).
# The store is copied to /tmp at start because Lambda's image filesystem is read-only and
# SQLite needs a writable directory. Any write made through the UI lives only in that
# container's /tmp and disappears on the next cold start, so the snapshot cannot be altered.
FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir mlflow
COPY mlflow_snapshot /opt/mlflow

CMD ["sh", "-c", "rm -rf /tmp/mlflow && cp -r /opt/mlflow /tmp/mlflow && exec mlflow server --backend-store-uri sqlite:////tmp/mlflow/mlflow.db --default-artifact-root file:///tmp/mlflow/mlartifacts --host 0.0.0.0 --port 8080 --workers 1 --allowed-hosts '*' --cors-allowed-origins '*'"]
