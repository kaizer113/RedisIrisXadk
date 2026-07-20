#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT before running this script}"
REGION="${VALUEWHOLESALE_DEPLOY_REGION:?Set VALUEWHOLESALE_DEPLOY_REGION before running this script}"
MEMORY_REGION="${GOOGLE_MEMORY_LOCATION:-$REGION}"
SERVICE="valuewholesale-shopping-agent"
REPOSITORY="valuewholesale"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE:latest"
LABELS="app=valuewholesale,environment=demo"
ACCESS_FLAGS=(--no-invoker-iam-check)
if [[ "${PUBLIC_ACCESS:-true}" == "false" ]]; then
  ACCESS_FLAGS=()
fi

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required"; exit 1; }
gcloud config set project "$PROJECT_ID" >/dev/null

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com

if ! gcloud artifacts repositories describe "$REPOSITORY" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" --repository-format docker --location "$REGION" --labels "$LABELS"
fi

gcloud builds submit --tag "$IMAGE" .

RUNTIME_ENV_FILE="$(mktemp /tmp/valuewholesale-cloud-run-env.XXXXXX.json)"
trap 'rm -f "$RUNTIME_ENV_FILE"' EXIT
uv run python - "$RUNTIME_ENV_FILE" <<'PY'
import json
import os
import sys

names = [
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_MEMORY_LOCATION",
    "GOOGLE_MODEL",
    "GOOGLE_MODELS",
    "GOOGLE_AGENT_ENGINE_ID",
    "VALUEWHOLESALE_VECTOR_SEARCH_ENABLED",
    "VALUEWHOLESALE_EMBEDDING_MODEL",
    "VALUEWHOLESALE_EMBEDDING_DEVICE",
    "VALUEWHOLESALE_EMBEDDING_CACHE_TTL_SECONDS",
    "VALUEWHOLESALE_SEMANTIC_ROUTER_THRESHOLD",
    "VALUEWHOLESALE_SEMANTIC_ROUTER_INDEX",
    "VALUEWHOLESALE_DEMO_MEMBER_ID",
    "VALUEWHOLESALE_DEMO_SESSION_ID",
    "REDIS_URL",
    "CTX_MCP_URL",
    "MCP_AGENT_KEY",
    "LANGCACHE_HOST",
    "LANGCACHE_CACHE_ID",
    "LANGCACHE_API_KEY",
    "LANGCACHE_SIMILARITY_THRESHOLD",
    "AGENT_MEMORY_BASE_URL",
    "AGENT_MEMORY_STORE_ID",
    "AGENT_MEMORY_API_KEY",
    "AGENT_MEMORY_NAMESPACE",
    "AGENT_MEMORY_SIMILARITY_THRESHOLD",
]
values = {name: os.environ[name] for name in names if os.environ.get(name)}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(values, stream)
PY

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --ingress all \
  "${ACCESS_FLAGS[@]}" \
  --labels "$LABELS" \
  --env-vars-file "$RUNTIME_ENV_FILE" \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 4 \
  --concurrency 40

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
