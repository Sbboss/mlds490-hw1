#!/usr/bin/env bash
# Push code only (no results, no venv) to deepdish.
set -euo pipefail
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER to your NetID}"
rsync -az --delete \
  --exclude '.git' \
  --exclude 'results' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.DS_Store' \
  ./ \
  "${REMOTE_USER}@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/"
echo "synced up -> ~/mlds490_hw1"
