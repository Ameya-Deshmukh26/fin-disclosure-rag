#!/usr/bin/env bash
# Deploy the RAG API and the MLflow dashboard snapshot to AWS Lambda.
#
#   build  -> docker images (deploy/api.Dockerfile, deploy/mlflow.Dockerfile)
#   push   -> Amazon ECR (fin-rag-api, fin-rag-mlflow)
#   run    -> AWS Lambda container functions, each with a public Function URL
#
# Why Lambda: pay per request and scale to zero (a demo costs cents when idle), no servers
# to patch, and the Lambda Web Adapter runs an ordinary FastAPI/MLflow server unchanged.
# Why a role, not keys: the function assumes fin-rag-lambda-role (Bedrock invoke + logs),
# so there are no long-lived credentials in the function configuration.
# Cost guard: reserved concurrency caps how many copies can run at once, which bounds
# Bedrock spend if the public URL is hammered.
#
# Usage:  bash deploy/deploy_lambda.sh            (create or update both functions)
set -euo pipefail
shopt -s inherit_errexit   # make failures inside $(...) stop the script too

REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REG="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
ROLE_ARN="${ROLE_ARN:-arn:aws:iam::$ACCOUNT:role/fin-rag-lambda-role}"

deploy() {  # name image memory timeout concurrency
  local name=$1 image=$2 mem=$3 timeout=$4 conc=$5
  if aws lambda get-function --function-name "$name" --region $REGION >/dev/null 2>&1; then
    aws lambda update-function-code --function-name "$name" --image-uri "$REG/$image:latest" \
      --region $REGION >/dev/null
    echo "updated $name"
  else
    aws lambda create-function --function-name "$name" --package-type Image \
      --code ImageUri="$REG/$image:latest" --role "$ROLE_ARN" \
      --memory-size "$mem" --timeout "$timeout" --architectures x86_64 \
      --region $REGION >/dev/null
    aws lambda wait function-active-v2 --function-name "$name" --region $REGION
    aws lambda create-function-url-config --function-name "$name" --auth-type NONE \
      --cors 'AllowOrigins=["*"],AllowMethods=["GET","POST"],AllowHeaders=["content-type"]' \
      --region $REGION >/dev/null
    # public Function URL: allow anonymous invocation through the URL only
    aws lambda add-permission --function-name "$name" --statement-id public-url \
      --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE \
      --region $REGION >/dev/null
    aws lambda add-permission --function-name "$name" --statement-id public-url-invoke \
      --action lambda:InvokeFunction --principal "*" --invoked-via-function-url \
      --region $REGION >/dev/null 2>&1 || true
    aws lambda put-function-concurrency --function-name "$name" \
      --reserved-concurrent-executions "$conc" --region $REGION >/dev/null || true
    echo "created $name"
  fi
  aws lambda get-function-url-config --function-name "$name" --region $REGION \
    --query FunctionUrl --output text
}

echo "API:    $(deploy fin-rag-api fin-rag-api 3008 120 2 | tail -1)"
echo "MLflow: $(deploy fin-rag-mlflow fin-rag-mlflow 2048 60 2 | tail -1)"
