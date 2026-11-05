#!/usr/bin/env bash
set -euo pipefail
ONPREM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONPREM_ENV="${1:-$ONPREM_DIR/../.env.onprem-ga}"
if [[ ! -f "$ONPREM_ENV" ]]; then
  echo "Create $ONPREM_ENV from .env.onprem-ga.example with the provisioned GA scoped keys."
  exit 1
fi
ONPREM_ENV="$(cd "$(dirname "$ONPREM_ENV")" && pwd)/$(basename "$ONPREM_ENV")"
exec uv run --directory "$ONPREM_DIR" --frozen --env-file "$ONPREM_ENV" \
  uvicorn valuewholesale_onprem.api:app --host 0.0.0.0 --port 8083 --workers 2
