FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    VALUEWHOLESALE_WARMUP_ON_STARTUP=true \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev
RUN uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('redis/langcache-embed-v3-small')"

COPY valuewholesale_agent ./valuewholesale_agent
COPY scripts ./scripts

EXPOSE 8080
CMD ["sh", "-c", "uv run --no-sync uvicorn valuewholesale_agent.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
