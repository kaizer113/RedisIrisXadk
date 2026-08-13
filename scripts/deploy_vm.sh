#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_ENV_FILE="${VALUEWHOLESALE_VM_ENV_FILE:-.env}"
if [[ ! -f "$SOURCE_ENV_FILE" ]]; then
  echo "$SOURCE_ENV_FILE is required. Copy .env.example and configure it first."
  exit 1
fi

set -a
# shellcheck disable=SC1090,SC1091
source "$SOURCE_ENV_FILE"
set +a

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT before running this script}"
REGION="${VALUEWHOLESALE_DEPLOY_REGION:?Set VALUEWHOLESALE_DEPLOY_REGION before running this script}"
ZONE="${VALUEWHOLESALE_VM_ZONE:?Set VALUEWHOLESALE_VM_ZONE before running this script}"
VM_NAME="${VALUEWHOLESALE_VM_NAME:-valuewholesale-demo}"
EXPERIENCE="${EXPERIENCE_ID:-valuewholesale}"
CONTAINER_NAME="${VALUEWHOLESALE_VM_CONTAINER_NAME:-${EXPERIENCE}-agent}"
HOST_PORT="${VALUEWHOLESALE_VM_HOST_PORT:-80}"
MACHINE_TYPE="e2-standard-4"
NETWORK="${VALUEWHOLESALE_VM_NETWORK:-default}"
SUBNETWORK="${VALUEWHOLESALE_VM_SUBNETWORK:-}"
NETWORK_TAG="valuewholesale-web"
if [[ "$HOST_PORT" == "80" && "$EXPERIENCE" == "valuewholesale" ]]; then
  FIREWALL_RULE="${VALUEWHOLESALE_VM_FIREWALL_RULE:-valuewholesale-allow-http}"
else
  FIREWALL_RULE="${VALUEWHOLESALE_VM_FIREWALL_RULE:-${EXPERIENCE}-allow-${HOST_PORT}}"
fi
if [[ "$NETWORK" != "default" && -z "${VALUEWHOLESALE_VM_FIREWALL_RULE:-}" ]]; then
  FIREWALL_RULE="${FIREWALL_RULE}-${NETWORK}"
fi
REPOSITORY="valuewholesale"
SERVICE="valuewholesale-shopping-agent"
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$SERVICE:latest"
LABELS="app=${EXPERIENCE},environment=demo"

command -v gcloud >/dev/null 2>&1 || { echo "gcloud is required"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is required"; exit 1; }

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable compute.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com

if ! gcloud artifacts repositories describe "$REPOSITORY" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format docker \
    --location "$REGION" \
    --labels "$LABELS"
fi

if [[ "${VALUEWHOLESALE_SKIP_BUILD:-false}" != "true" ]]; then
  gcloud builds submit --tag "$IMAGE" .
fi

if gcloud compute firewall-rules describe "$FIREWALL_RULE" >/dev/null 2>&1; then
  firewall_network="$(gcloud compute firewall-rules describe "$FIREWALL_RULE" \
    --format='value(network.basename())')"
  if [[ "$firewall_network" != "$NETWORK" ]]; then
    echo "Existing firewall rule $FIREWALL_RULE uses $firewall_network; expected $NETWORK."
    echo "Set VALUEWHOLESALE_VM_FIREWALL_RULE to a rule on the target network."
    exit 1
  fi
else
  gcloud compute firewall-rules create "$FIREWALL_RULE" \
    --network "$NETWORK" \
    --direction INGRESS \
    --action ALLOW \
    --rules "tcp:$HOST_PORT" \
    --source-ranges 0.0.0.0/0 \
    --target-tags "$NETWORK_TAG" \
    --description "Public HTTP access for the $EXPERIENCE demo" \
    || gcloud compute firewall-rules describe "$FIREWALL_RULE" >/dev/null
fi

if gcloud compute instances describe "$VM_NAME" --zone "$ZONE" >/dev/null 2>&1; then
  current_network="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" \
    --format='value(networkInterfaces[0].network.basename())')"
  if [[ "$current_network" != "$NETWORK" ]]; then
    echo "Existing VM $VM_NAME uses network $current_network; expected $NETWORK."
    echo "Recreate the VM explicitly before deploying to a different network."
    exit 1
  fi
  current_type="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --format='value(machineType.basename())')"
  if [[ "$current_type" != "$MACHINE_TYPE" ]]; then
    echo "Existing VM $VM_NAME uses $current_type; expected $MACHINE_TYPE."
    echo "Choose another VALUEWHOLESALE_VM_NAME or resize the VM explicitly."
    exit 1
  fi
  status="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --format='value(status)')"
  if [[ "$status" != "RUNNING" ]]; then
    gcloud compute instances start "$VM_NAME" --zone "$ZONE"
  fi
else
  network_interface="network=$NETWORK,network-tier=PREMIUM,nic-type=GVNIC"
  if [[ -n "$SUBNETWORK" ]]; then
    network_interface="network=$NETWORK,subnet=$SUBNETWORK,network-tier=PREMIUM,nic-type=GVNIC"
  fi
  gcloud compute instances create "$VM_NAME" \
    --quiet \
    --zone "$ZONE" \
    --machine-type "$MACHINE_TYPE" \
    --network-interface "$network_interface" \
    --tags "$NETWORK_TAG" \
    --labels "$LABELS" \
    --image-family debian-12 \
    --image-project debian-cloud \
    --boot-disk-type pd-balanced \
    --boot-disk-size 30GB \
    --scopes cloud-platform \
    --maintenance-policy MIGRATE \
    --provisioning-model STANDARD \
    --shielded-secure-boot \
    --metadata-from-file "startup-script=$ROOT_DIR/scripts/vm_startup.sh"
fi

echo "Waiting for Docker installation and SSH..."
ready=false
for _ in $(seq 1 40); do
  if gcloud compute ssh "$VM_NAME" --zone "$ZONE" \
    --command 'command -v docker >/dev/null && sudo systemctl is-active --quiet docker' \
    --quiet >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 5
done
if [[ "$ready" != "true" ]]; then
  echo "VM did not become ready within the expected time."
  exit 1
fi

RUNTIME_ENV_FILE="$(mktemp "/tmp/${EXPERIENCE}-vm-env.XXXXXX")"
trap 'rm -f "$RUNTIME_ENV_FILE"' EXIT
chmod 600 "$RUNTIME_ENV_FILE"
while IFS= read -r line; do
  key="${line%%=*}"
  case "$key" in
    GOOGLE_*|VALUEWHOLESALE_*|EXPERIENCE_ID|DATASET_DIR|STATIC_DIR|REDIS_KEY_PREFIX|REDIS_INDEX_PREFIX|EMBEDDING_CACHE_NAME|ADK_*|CONTEXT_*|MEMORY_BANK_DISPLAY_NAME|REDIS_URL|CTX_MCP_URL|MCP_AGENT_KEY|LANGCACHE_*|AGENT_MEMORY_*|PORT|LOG_LEVEL)
      if [[ "$key" == "REDIS_URL" && -n "${VALUEWHOLESALE_VM_REDIS_HOST:-}" ]]; then
        redis_value="${line#REDIS_URL=}"
        redis_prefix="${redis_value%@*}"
        redis_host_and_port="${redis_value##*@}"
        redis_port="${redis_host_and_port##*:}"
        if [[ "$redis_prefix" == "$redis_value" || "$redis_port" == "$redis_host_and_port" ]]; then
          echo "REDIS_URL must include credentials, a hostname, and a port."
          exit 1
        fi
        line="REDIS_URL=${redis_prefix}@${VALUEWHOLESALE_VM_REDIS_HOST}:${redis_port}"
      fi
      printf '%s\n' "$line" >> "$RUNTIME_ENV_FILE"
      ;;
  esac
done < "$SOURCE_ENV_FILE"

gcloud compute scp "$RUNTIME_ENV_FILE" "$VM_NAME:~/${CONTAINER_NAME}.env" --zone "$ZONE" --quiet
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --quiet --command "
  set -e
  sudo install -o root -g root -m 600 ~/${CONTAINER_NAME}.env /etc/${CONTAINER_NAME}.env
  rm -f ~/${CONTAINER_NAME}.env
  token=\$(curl -fsS -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"access_token\"])')
  printf '%s' \"\$token\" | sudo docker login -u oauth2accesstoken --password-stdin https://$REGION-docker.pkg.dev
  sudo docker image prune -f >/dev/null
  sudo docker pull '$IMAGE'
  sudo docker rm -f '$CONTAINER_NAME' >/dev/null 2>&1 || true
  sudo docker run -d \
    --name '$CONTAINER_NAME' \
    --restart unless-stopped \
    --env-file '/etc/${CONTAINER_NAME}.env' \
    -e PORT=8080 \
    -e WEB_CONCURRENCY=2 \
    -p '$HOST_PORT:8080' \
    '$IMAGE'
  sudo docker image prune -f >/dev/null
"

PUBLIC_IP="$(gcloud compute instances describe "$VM_NAME" --zone "$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
PUBLIC_URL="http://$PUBLIC_IP"
if [[ "$HOST_PORT" != "80" ]]; then
  PUBLIC_URL="$PUBLIC_URL:$HOST_PORT"
fi

echo "Waiting for the public health endpoint..."
healthy=false
for _ in $(seq 1 30); do
  if curl -fsS "$PUBLIC_URL/api/health" >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 2
done
if [[ "$healthy" != "true" ]]; then
  echo "Container started, but the public health check did not pass."
  echo "Inspect it with: gcloud compute ssh $VM_NAME --zone $ZONE --command 'sudo docker logs $CONTAINER_NAME'"
  exit 1
fi

echo "$PUBLIC_URL"
