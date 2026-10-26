#!/usr/bin/env bash
# One CartPole evaluation at a time. Skips runs that already have eval_stats.json.
set -u
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
shopt -s nullglob
for f in results/cartpole/*/*/final.pt; do
  d=$(dirname "$f")
  if [[ -f "$d/eval_stats.json" ]]; then
    echo "SKIP $d"
    continue
  fi
  echo "START $d"
  if ! .venv/bin/python -u -m pg.evaluate --env cartpole --run "$d" --n 500 > "$d/eval_console.txt" 2>&1; then
    echo "FAIL $d"
  else
    echo "OK $d"
  fi
done
echo ALL_EVAL_DONE
