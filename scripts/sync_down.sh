#!/usr/bin/env bash
# Pull results (curves, small checkpoints, eval csv, figures). Skip files > 200MB.
set -euo pipefail
REMOTE_USER="${REMOTE_USER:?set REMOTE_USER to your NetID}"
mkdir -p results
rsync -az --max-size=200m \
  "${REMOTE_USER}@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/results/" \
  results/
echo "synced down -> results/"
