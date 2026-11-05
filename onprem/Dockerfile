FROM python:3.13-slim@sha256:ae14d4ce8771606477a81446e7ef429edd6762305fc50c84507c1041fa8285d1

LABEL owner="lionel_giavelli" skip_deletion="yes"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 UV_CACHE_DIR=/tmp/uv-cache
WORKDIR /app
RUN pip install --no-cache-dir uv==0.11.7
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY valuewholesale_onprem ./valuewholesale_onprem
RUN useradd --create-home --uid 10001 demo
USER demo
EXPOSE 8080
CMD ["sh", "-c", "uv run --frozen --no-sync uvicorn valuewholesale_onprem.api:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-2}"]
