#!/usr/bin/env bash
# Render build script for the pneumonia-detection backend.
#
# Render checks out the repo at /opt/render/project/src and runs this script
# from the repo root. We install the runtime dependencies only.

set -euo pipefail

echo "==> Installing backend runtime dependencies"
pip install -r backend/requirements.txt

echo "==> Verifying model artifacts are present"
ls -lh models/pneumonia_classifier.keras models/model_metadata.json

echo "==> Build complete"
