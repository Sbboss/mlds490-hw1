#!/usr/bin/env bash
# Push code only (no results, no venv) to deepdish.
set -euo pipefail
rsync -az --delete \
  --exclude '.git' \
  --exclude 'results' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.DS_Store' \
  "/Users/sbboss/NU/Q4/MLDS-490/HW#1/" \
  hia9393@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/
echo "synced up -> ~/mlds490_hw1"
