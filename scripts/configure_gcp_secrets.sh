#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-central-beach-194106}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-east4}"
SERVICE="valueharbor-shopping-agent"
LABELS="owner=lionel_giavelli,app=valueharbor,environment=demo"

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required"; exit 1; }

put_secret() {
  local secret_name="$1"
  local env_name="$2"
  local secret_value="${!env_name:-}"
  if [[ -z "$secret_value" ]]; then
    echo "Missing required environment variable: $env_name"
    exit 1
  fi
  if gcloud secrets describe "$secret_name" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud secrets update "$secret_name" --project "$PROJECT_ID" --update-labels "$LABELS" >/dev/null
  else
    gcloud secrets create "$secret_name" --project "$PROJECT_ID" --replication-policy automatic --labels "$LABELS" >/dev/null
  fi
  printf '%s' "$secret_value" | gcloud secrets versions add "$secret_name" --project "$PROJECT_ID" --data-file=- >/dev/null
}

put_secret valueharbor-redis-url REDIS_URL
put_secret valueharbor-mcp-agent-key MCP_AGENT_KEY
put_secret valueharbor-langcache-api-key LANGCACHE_API_KEY
put_secret valueharbor-agent-memory-api-key AGENT_MEMORY_API_KEY

for env_name in LANGCACHE_HOST LANGCACHE_CACHE_ID AGENT_MEMORY_BASE_URL AGENT_MEMORY_STORE_ID; do
  if [[ -z "${!env_name:-}" ]]; then
    echo "Missing required environment variable: $env_name"
    exit 1
  fi
done

gcloud run services update "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --update-env-vars "LANGCACHE_HOST=$LANGCACHE_HOST,LANGCACHE_CACHE_ID=$LANGCACHE_CACHE_ID,AGENT_MEMORY_BASE_URL=$AGENT_MEMORY_BASE_URL,AGENT_MEMORY_STORE_ID=$AGENT_MEMORY_STORE_ID" \
  --update-secrets "REDIS_URL=valueharbor-redis-url:latest,MCP_AGENT_KEY=valueharbor-mcp-agent-key:latest,LANGCACHE_API_KEY=valueharbor-langcache-api-key:latest,AGENT_MEMORY_API_KEY=valueharbor-agent-memory-api-key:latest"
