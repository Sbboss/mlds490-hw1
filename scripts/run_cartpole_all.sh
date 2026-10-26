#!/usr/bin/env bash
# Launch CartPole C0–C2 (5 seeds) plus C3–C5. CPU; at most 8 concurrent jobs.
set -euo pipefail
ROOT="${ROOT:-$HOME/mlds490_hw1}"
PY="${PY:-$HOME/venvs/mlds490/bin/python}"
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

run() {
  local out="$1"; shift
  mkdir -p "$out"
  nice -n 10 "$PY" -m pg.train_cartpole --out "$out" "$@" \
    > >(tee -a "$out/log.txt") 2>&1
}

# C0 Part 2 primary
for s in 0 1 2 3 4; do
  run "results/cartpole/C0_none/seed${s}" --seed "$s" --baseline none --episodes 2000 &
done
wait

# C1 Part 3 primary
for s in 0 1 2 3 4; do
  run "results/cartpole/C1_value/seed${s}" --seed "$s" --baseline value --episodes 2000 &
done
wait

# C2 Part 3 secondary
for s in 0 1 2 3 4; do
  run "results/cartpole/C2_ema/seed${s}" --seed "$s" --baseline ema --episodes 2000 &
done
wait

# C3 gamma^t ablation (3 seeds)
for s in 0 1 2; do
  run "results/cartpole/C3_gamma_weight/seed${s}" --seed "$s" --baseline none --gamma_weight on --episodes 2000 &
done
wait

# C4 lr 0.001
for s in 0 1 2; do
  run "results/cartpole/C4_lr1e3/seed${s}" --seed "$s" --baseline none --lr 0.001 --episodes 2000 &
done
wait

# C5 reward-sign reversal (T-C6)
for s in 0 1 2; do
  run "results/cartpole/C5_signflip/seed${s}" --seed "$s" --baseline none --reward_sign -1 --episodes 300 &
done
wait

echo "cartpole matrix done"
