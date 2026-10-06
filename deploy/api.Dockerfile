# RAG API on AWS Lambda: FastAPI behind the Lambda Web Adapter, exposed by a Function URL.
FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    EVAL_ENTITY_FILTER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /var/task
# CPU-only torch keeps the image small enough to cold-start reasonably
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir sentence-transformers rank-bm25 numpy boto3 python-dotenv \
        fastapi "uvicorn[standard]" pydantic huggingface_hub
# bake the embedding model into the image (Lambda's filesystem is read-only at runtime)
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

COPY generate.py hybrid.py retrieval.py ./
COPY evals ./evals
COPY app ./app
COPY data_big ./data_big
COPY index_big.pkl ./

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
