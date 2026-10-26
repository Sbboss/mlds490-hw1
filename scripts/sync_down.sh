#!/usr/bin/env bash
# Pull results (curves, small checkpoints, eval csv, figures). Skip files > 200MB.
set -euo pipefail
mkdir -p "/Users/sbboss/NU/Q4/MLDS-490/HW#1/results"
rsync -az --max-size=200m \
  hia9393@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/results/ \
  "/Users/sbboss/NU/Q4/MLDS-490/HW#1/results/"
echo "synced down -> results/"
