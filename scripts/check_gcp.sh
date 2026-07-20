#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT before running this script}"
REGION="${VALUEWHOLESALE_DEPLOY_REGION:?Set VALUEWHOLESALE_DEPLOY_REGION before running this script}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed. Install the Google Cloud CLI, then rerun make check-gcp."
  exit 1
fi

gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud projects describe "$PROJECT_ID" --format='value(projectId,name)'
gcloud config set project "$PROJECT_ID" >/dev/null
gcloud config set run/region "$REGION" >/dev/null
echo "GCP CLI is ready for project $PROJECT_ID; Cloud Run region is $REGION."
