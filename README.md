# MLDS 490 – Homework 1

Value / policy iteration on a shuttle-dispatch MDP (Part 1) and REINFORCE
on CartPole-v1 and ALE/Pong-v5, with and without a baseline (Parts 2–3).

## How to run

### Local (CPU; CartPole + Part 1 + validation)

```bash
cd "/Users/sbboss/NU/Q4/MLDS-490/HW#1"    # quote the path: it contains #
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch
pip install -r requirements.txt

# Part 1
cd shuttle && python run_shuttle.py && python validation.py && python -m pytest -q test_shuttle.py && cd ..

# Parts 2/3 validation (CPU; add --pong for T-P*)
python -m validation.run_validation
python -m validation.run_validation --pong

# CartPole training, one seed
python -m pg.train_cartpole --baseline none --seed 0 --out results/cartpole/C0_none/seed0
python -m pg.evaluate --env cartpole --run results/cartpole/C0_none/seed0
python -m pg.plot_results
```

The full CartPole matrix is `scripts/run_cartpole_all.sh` (or the equivalent
local loop in this README’s “Experiment matrix”).

### deepdish (one GPU; Pong)

```bash
# once
ssh hia9393@mlds-deepdish5.ads.northwestern.edu
# then, from the laptop:
bash scripts/sync_up.sh
ssh hia9393@mlds-deepdish5.ads.northwestern.edu 'bash ~/mlds490_hw1/scripts/remote_setup.sh'

# every Pong process must see exactly one card
export CUDA_VISIBLE_DEVICES=1
source ~/venvs/mlds490/bin/activate
cd ~/mlds490_hw1
python -m scripts.bench_pong          # T-P11; writes results/pong_bench.json
# smoke 200 episodes, then LR sweep, then full P0/P1 — see PLAN.md §8.2
nice -n 10 python -m pg.train_pong --baseline none --lr 1e-3 --workers 16 \
    --out results/pong/P0_none/seed0
python -m pg.evaluate --env pong --run results/pong/P0_none/seed0 --workers 16
```

`tmux` is not installed on deepdish; long jobs use `nohup setsid` and write
`log.txt`. Pull artifacts with `bash scripts/sync_down.sh`.

## Repository

Public GitHub: https://github.com/Sbboss/mlds490-hw1

`shuttle/` is Part 1; `pg/` is Parts 2–3; `validation/` is the edge-case suite;
`results/` holds the CSV/JSON curves and evals the report quotes; `figures/`
is what the report includes. Large Pong weight files (`final.pt`) are not in
the repo. Rebuild the PDF with `.venv/bin/python scripts/build_report.py`
(the system pandoc binary on the machine used here does not run).

## Experiment matrix (CartPole)

| id | config | seeds | episodes |
|---|---|---|---|
| C0 | no baseline, w=1, lr 0.01 | 0–4 | 2000 |
| C1 | value baseline | 0–4 | 2000 |
| C2 | EMA baseline | 0–4 | 2000 |
| C3 | `--gamma_weight on` | 0–2 | 2000 |
| C4 | lr 0.001 | 0–2 | 2000 |
| C5 | `--reward_sign -1` | 0–2 | 300 |

Headline Part 2/3 CartPole figures use **seed 0**. Multi-seed bands use all seeds.

## Expected runtimes (order of magnitude)

- Part 1 solvers + 38 checks: ~10 s
- T-R*/T-B*/T-C* (no Pong): ~2 s
- T-P* including 20-episode anchors: tens of minutes
- CartPole 2000 episodes, one seed, CPU: a few minutes
- Pong 25k episodes or 4 h wall-clock on deepdish, 16 CPU actors + 1 GPU

## Hardware used

- Local Mac for Part 1, CartPole, and tests
- `mlds-deepdish5`: 48 CPU, 93 GB RAM, one GPU (`CUDA_VISIBLE_DEVICES=0`,
  the RTX PRO 5000 Blackwell; see `DECISIONS.md`). Never more than one card,
  never more than 32 worker processes.
