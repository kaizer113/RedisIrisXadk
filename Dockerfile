FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY valueharbor_agent ./valueharbor_agent
COPY scripts ./scripts

EXPOSE 8080
CMD ["sh", "-c", "uv run --no-sync uvicorn valueharbor_agent.api:app --host 0.0.0.0 --port ${PORT}"]
