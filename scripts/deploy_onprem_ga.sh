#!/usr/bin/env bash
# Deploy only the existing Value Wholesale port-8083 container.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:?Usage: deploy_onprem_ga.sh IMAGE_AT_SHA256 ENV_FILE COMBINED_CA_FILE [deploy|canary]}"
ENV_FILE="${2:?Provide the private Docker-compatible GA environment file}"
CA_FILE="${3:?Provide the combined public and demo CA bundle}"
MODE="${4:-deploy}"
if [[ "$MODE" != deploy && "$MODE" != canary ]]; then
  echo 'Mode must be deploy or canary.' >&2
  exit 1
fi
if [[ ! "$IMAGE" =~ ^us-east4-docker.pkg.dev/central-beach-194106/valuewholesale/valuewholesale-onprem-ga@sha256:[0-9a-f]{64}$ ]]; then
  echo 'Use the dedicated on-prem GA image with its immutable sha256 digest.' >&2
  exit 1
fi
test -s "$ENV_FILE"
test -s "$CA_FILE"
gcloud auth print-access-token >/dev/null

# Parse exactly as Docker does: retain values literally, without shell expansion.
uv run --directory "$ROOT_DIR/onprem" --frozen --no-sync python - "$ENV_FILE" "$CA_FILE" <<'PY'
import os
from pathlib import Path
import ssl
import sys
from valuewholesale_onprem.config import Settings

values = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.lstrip().startswith('#'):
        continue
    key, separator, value = line.partition('=')
    if not separator or key.strip() != key or not key:
        raise SystemExit('Invalid Docker environment file line')
    if value[:1] in ('"', "'") or value[-1:] in ('"', "'"):
        raise SystemExit('Docker environment values must not have surrounding quotes')
    if key in values:
        raise SystemExit('Duplicate Docker environment key')
    values[key] = value
os.environ.clear()
os.environ.update(values)
settings = Settings()
if not all(settings.configured.values()):
    raise SystemExit('All GA services must be configured before cutover')
if len(settings.onprem_session_secret.get_secret_value()) < 32:
    raise SystemExit('The shared session secret must have at least 32 characters')
if values.get('SSL_CERT_FILE') != '/etc/iris-ca/ca.crt':
    raise SystemExit('Set SSL_CERT_FILE=/etc/iris-ca/ca.crt for the mounted CA bundle')
if values.get('GOOGLE_CLOUD_PROJECT') != 'central-beach-194106':
    raise SystemExit('Unexpected Google Cloud project')
ssl.create_default_context(cafile=sys.argv[2])
# Validate the native Mesh builder's real settings before touching the VM.
from redis_fastapi.streams import check_stream_settings
try:
    check_stream_settings()
except Exception as exc:
    raise SystemExit('Native Mesh settings validation failed (' + type(exc).__name__ + ')') from None
print('Runtime configuration and CA bundle validated; no credentials displayed')
PY

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
UPLOAD_PREFIX="onprem-ga-${STAMP}"
GCLOUD_ARGS=(--project central-beach-194106 --zone us-east4-c --quiet)
gcloud compute scp "$ENV_FILE" "valuewholesale-demo:~/${UPLOAD_PREFIX}.env" "${GCLOUD_ARGS[@]}"
gcloud compute scp "$CA_FILE" "valuewholesale-demo:~/${UPLOAD_PREFIX}.crt" "${GCLOUD_ARGS[@]}"
REMOTE_FILE="$(mktemp /tmp/onprem-ga-deploy.XXXXXX)"
trap 'rm -f "$REMOTE_FILE"' EXIT
chmod 600 "$REMOTE_FILE"
cat > "$REMOTE_FILE" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
IMAGE="$1"
STAMP="$2"
MODE="$3"
UPLOAD_PREFIX="onprem-ga-${STAMP}"
NAME=valuewholesale-onprem-agent
CANARY=valuewholesale-onprem-ga-canary
PREVIOUS="${NAME}-before-${STAMP}"
BACKUP="/var/backups/valuewholesale-onprem-ga/${STAMP}"
ENV_PATH="/etc/${NAME}.env"
CA_PATH="/etc/${NAME}-ca.crt"
USER_HOME="$(getent passwd "${SUDO_USER:?Run through sudo}" | cut -d: -f6)"
test -n "$USER_HOME"
CANARY_OWNED=false
cleanup_canary() {
  if [[ "$CANARY_OWNED" == true ]]; then
    docker rm -f "$CANARY" >/dev/null 2>&1 || true
    CANARY_OWNED=false
  fi
}
cleanup() {
  cleanup_canary
  rm -f "${USER_HOME}/${UPLOAD_PREFIX}.env" "${USER_HOME}/${UPLOAD_PREFIX}.crt"
}
trap cleanup EXIT

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo 'Expected existing port-8083 container is missing; refusing cutover.' >&2
  exit 1
fi
docker inspect "$NAME" --format '{{json .HostConfig.PortBindings}}' | python3 -c '
import json,sys
p=json.load(sys.stdin)
assert any(x.get("HostPort")=="8083" for v in p.values() for x in (v or [])), "Unexpected target port"
'
if docker inspect "$CANARY" >/dev/null 2>&1; then
  echo 'A previous canary exists; inspect it before retrying.' >&2
  exit 1
fi
if ss -lntH | awk '{print $4}' | grep -q ':18083$'; then
  echo 'Local canary port 18083 is occupied; refusing to alter it.' >&2
  exit 1
fi
install -d -m 700 "$BACKUP"
docker inspect "$NAME" > "$BACKUP/container-inspect.json"
docker ps --format '{{.Names}} {{.ID}} {{.Image}}' > "$BACKUP/containers-before.txt"
if test -f "$ENV_PATH"; then cp -p "$ENV_PATH" "$BACKUP/runtime.env"; fi
if test -f "$CA_PATH"; then cp -p "$CA_PATH" "$BACKUP/ca.crt"; fi
install -m 600 "${USER_HOME}/${UPLOAD_PREFIX}.env" "$BACKUP/new-runtime.env"
install -m 644 "${USER_HOME}/${UPLOAD_PREFIX}.crt" "$BACKUP/new-ca.crt"
rm -f "${USER_HOME}/${UPLOAD_PREFIX}.env" "${USER_HOME}/${UPLOAD_PREFIX}.crt"
chmod 600 "$BACKUP/container-inspect.json" "$BACKUP/containers-before.txt"

token=$(curl -fsS -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
printf '%s' "$token" | docker login -u oauth2accesstoken --password-stdin https://us-east4-docker.pkg.dev >/dev/null
unset token
docker pull "$IMAGE"
wait_ready() {
  local port="$1"
  for _ in $(seq 1 36); do
    if curl -fsS --max-time 12 "http://127.0.0.1:${port}/api/health" \
        | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ready") is True else 1)' 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}
run_new() {
  local name="$1" port="$2" env_path="$3" ca_path="$4"
  docker run -d --name "$name" --restart unless-stopped \
    --label owner=lionel_giavelli --label skip_deletion=yes \
    --env-file "$env_path" -e PORT=8080 -e WEB_CONCURRENCY=2 \
    --mount "type=bind,source=${ca_path},target=/etc/iris-ca/ca.crt,readonly" \
    -p "${port}:8080" "$IMAGE" >/dev/null
}
CANARY_OWNED=true
run_new "$CANARY" 127.0.0.1:18083 "$BACKUP/new-runtime.env" "$BACKUP/new-ca.crt"
if ! wait_ready 18083; then
  echo "GA canary did not become ready. Existing port-8083 container is unchanged. Backup: ${BACKUP}" >&2
  exit 1
fi
if [[ "$MODE" == canary ]]; then
  CANARY_OWNED=false
  echo "GA canary is ready at VM localhost:18083. Port 8083 is unchanged. Private backup: ${BACKUP}"
  echo "Run functional smoke checks, then remove only ${CANARY} before the final deploy."
  exit 0
fi
cleanup_canary

docker stop "$NAME" >/dev/null
if ! docker rename "$NAME" "$PREVIOUS"; then
  docker start "$NAME" >/dev/null
  exit 1
fi
rollback() {
  trap - ERR
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker rename "$PREVIOUS" "$NAME"
  docker start "$NAME" >/dev/null
  if test -f "$BACKUP/runtime.env"; then cp -p "$BACKUP/runtime.env" "$ENV_PATH"; else rm -f "$ENV_PATH"; fi
  if test -f "$BACKUP/ca.crt"; then cp -p "$BACKUP/ca.crt" "$CA_PATH"; else rm -f "$CA_PATH"; fi
  echo "Restored the previous port-8083 container. Backup: ${BACKUP}" >&2
}
trap rollback ERR
install -m 600 "$BACKUP/new-runtime.env" "$ENV_PATH"
install -m 644 "$BACKUP/new-ca.crt" "$CA_PATH"
if ! run_new "$NAME" 8083 "$ENV_PATH" "$CA_PATH" || ! wait_ready 8083; then
  rollback
  exit 1
fi
trap - ERR
docker ps --format '{{.Names}} {{.ID}} {{.Image}}' > "$BACKUP/containers-after.txt"
printf '%s\n' "$IMAGE" > "$BACKUP/deployed-image.txt"
echo "GA port-8083 health is ready. Previous stopped container: ${PREVIOUS}. Private backup: ${BACKUP}"
SH
gcloud compute scp "$REMOTE_FILE" "valuewholesale-demo:~/${UPLOAD_PREFIX}.sh" "${GCLOUD_ARGS[@]}"
gcloud compute ssh valuewholesale-demo "${GCLOUD_ARGS[@]}" \
  --command "sudo bash ~/${UPLOAD_PREFIX}.sh '$IMAGE' '$STAMP' '$MODE'; result=\$?; rm -f ~/${UPLOAD_PREFIX}.sh; exit \$result"
if [[ "$MODE" == canary ]]; then
  exit 0
fi
PUBLIC_IP="$(gcloud compute instances describe valuewholesale-demo "${GCLOUD_ARGS[@]}" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
curl -fsS --max-time 20 "http://${PUBLIC_IP}:8083/api/health" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"ready":d.get("ready"),"services":d.get("services")})); sys.exit(0 if d.get("ready") is True else 1)'
echo "Deployed only http://${PUBLIC_IP}:8083"
