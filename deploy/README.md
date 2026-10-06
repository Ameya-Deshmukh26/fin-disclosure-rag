# Deployment (AWS)

```
                 ┌──────────────────────── AWS (us-east-1) ─────────────────────────┐
  user ──HTTPS──►│ Lambda Function URL ─► Lambda: fin-rag-api (container from ECR)  │
                 │                         FastAPI: hybrid retrieval + entity filter │
                 │                         runtime guardrails (number lock)          │
                 │                              │ IAM role: fin-rag-lambda-role       │
                 │                              ▼                                    │
                 │                         Amazon Bedrock: Llama 3.1 8B (generation) │
                 │                                                                   │
  user ──HTTPS──►│ Lambda Function URL ─► Lambda: fin-rag-mlflow (read-only MLflow)  │
                 │                                                                   │
                 │ CloudWatch Logs ◄── both functions                                │
                 └───────────────────────────────────────────────────────────────────┘
  GitHub Actions: offline eval tests on every push; full Bedrock eval suite on demand,
  failing the job if a safety gate regresses.
```

| Piece | Service | Why |
|---|---|---|
| Container registry | **Amazon ECR** | Lambda runs images from ECR; scan-on-push enabled |
| Compute | **AWS Lambda** (container image, 3 GB memory) | Pay per request, scales to zero, no servers to patch |
| Web server in Lambda | **Lambda Web Adapter** | Runs the normal FastAPI / MLflow server unchanged |
| Public endpoint | **Lambda Function URL** | HTTPS endpoint without API Gateway |
| Models | **Amazon Bedrock** (Converse API) | Generator and judge inside the AWS boundary, IAM auth, no API keys |
| Permissions | **IAM role** `fin-rag-lambda-role` | Temporary credentials; least privilege (Bedrock invoke + logs) |
| Logs | **CloudWatch Logs** | Per-request logs and latency |
| Cost guard | **Reserved concurrency = 2** | Caps parallel copies, which bounds Bedrock spend on a public URL |

## Known tradeoffs (and what I'd do at real scale)
- **Cold start ~20 s** (PyTorch + the embedding model load). Fixes: provisioned concurrency, or
  switch query embeddings to Bedrock Titan so the image drops PyTorch entirely.
- **Index baked into the image.** Fine for a fixed corpus; with changing data, move vectors to
  Postgres + pgvector (Aurora/RDS) and keep the function stateless.
- **Per-container query cache.** Real deployments use a shared cache (ElastiCache/Redis).
- **MLflow is a read-only snapshot.** A live team setup would run the MLflow server with a
  Postgres backend and S3 artifacts, behind authentication.

## Commands
```bash
docker build --platform linux/amd64 -f deploy/api.Dockerfile -t fin-rag-api .
python deploy/make_mlflow_snapshot.py
docker build --platform linux/amd64 -f deploy/mlflow.Dockerfile -t fin-rag-mlflow .
# push both to ECR, then:
bash deploy/deploy_lambda.sh
```
