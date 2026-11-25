#!/usr/bin/env bash
# One-click deploy to Google Cloud Run.
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. gcloud init
#   3. gcloud auth login
#   4. gcloud config set project YOUR_PROJECT_ID
#
# Usage:
#   ./deploy-cloudrun.sh
#
set -euo pipefail

PROJECT_ID="$(gcloud config get-value project)"
SERVICE_NAME="${SERVICE_NAME:-pneumonia-api}"
REGION="${REGION:-us-central1}"
FRONTEND_URL="${FRONTEND_URL:-https://careview-imaging.netlify.app}"

echo "==> Deploying ${SERVICE_NAME} to Cloud Run (project: ${PROJECT_ID}, region: ${REGION})"

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --region "$REGION" \
  --memory 2Gi \
  --cpu 1 \
  --allow-unauthenticated \
  --set-env-vars "PNEUMONIA_CORS_ORIGINS=${FRONTEND_URL}" \
  --set-env-vars "PNEUMONIA_MODEL_PATH=/app/models/pneumonia_classifier.keras" \
  --set-env-vars "PNEUMONIA_METADATA_PATH=/app/models/model_metadata.json" \
  --set-env-vars "TF_CPP_MIN_LOG_LEVEL=3" \
  --set-env-vars "TF_ENABLE_ONEDNN_OPTS=0"

echo "==> Done. Your backend URL is printed above."
