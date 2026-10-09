#!/usr/bin/env bash
# One-shot manual deploy of Clipper to Google Cloud Run (same thing the GitHub Action does).
#
# Usage:
#   ./deploy.sh <PASSWORD> [GEMINI_API_KEY] [GCS_BUCKET]
#
# Requires: gcloud CLI logged in (`gcloud auth login`) and a project selected
# (`gcloud config set project PROJECT_ID`) or GCP_PROJECT_ID exported.
# Optional env: GCP_REGION (default europe-west1), SERVICE (default clipper), CLOUD_RUN_SA.
set -euo pipefail

PASSWORD="${1:-}"
GEMINI_API_KEY="${2:-}"
GCS_BUCKET="${3:-}"
REGION="${GCP_REGION:-europe-west1}"
SERVICE="${SERVICE:-clipper}"
PROJECT="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"

if [ -z "$PASSWORD" ]; then
  echo "usage: $0 <PASSWORD> [GEMINI_API_KEY] [GCS_BUCKET]" >&2
  exit 1
fi
if [ -z "$PROJECT" ]; then
  echo "No GCP project: run 'gcloud config set project PROJECT_ID' or export GCP_PROJECT_ID" >&2
  exit 1
fi

cd "$(dirname "$0")"

echo ">> project=$PROJECT region=$REGION service=$SERVICE"
echo ">> enabling APIs (Cloud Run, Cloud Build, Artifact Registry, Cloud Storage) ..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com storage.googleapis.com --project "$PROJECT"

EXTRA=()
if [ -n "${CLOUD_RUN_SA:-}" ]; then
  EXTRA+=(--service-account "$CLOUD_RUN_SA")
fi

# "^|^" switches the list delimiter from "," to "|" so values may contain commas.
ENV_VARS="^|^PASSWORD=${PASSWORD}|GEMINI_API_KEY=${GEMINI_API_KEY}|GCS_BUCKET=${GCS_BUCKET}|GRADIO_ANALYTICS_ENABLED=False"

echo ">> deploying (first build takes ~10-15 min) ..."
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --source . \
  --region "$REGION" \
  --platform managed \
  --cpu 2 \
  --memory 4Gi \
  --timeout 3600 \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 4 \
  --allow-unauthenticated \
  --session-affinity \
  --execution-environment gen2 \
  --set-env-vars "$ENV_VARS" \
  --quiet \
  "${EXTRA[@]}"

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')
echo
echo ">> deployed: $URL   (login: admin / <PASSWORD>)"
