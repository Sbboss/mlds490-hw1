# MLDS 490 Homework 1

Value and policy iteration on a shuttle-dispatch MDP (Part 1), and REINFORCE on CartPole-v1 and ALE/Pong-v5, with and without a baseline (Parts 2 and 3). The write-up is `report/report.pdf`, built from `report/report.md`.

## What is where

- `shuttle/`: Part 1. The MDP, the three solvers, the figures and the validation checks.
- `pg/`: Parts 2 and 3. Models, the REINFORCE loss and baselines, the CartPole and Pong trainers, evaluation and plotting.
- `validation/`: edge-case and correctness tests for the policy-gradient code. Output is in `validation_pg_results.md`.
- `scripts/`: run, sync and report-building scripts.
- `results/`: the training curves, evaluation CSVs and JSON files that the report quotes.
- `figures/`: the figures used in the report.

The large Pong weight files (`final.pt`) and the intermediate checkpoints are not in the repo.

## Setup

```bash
git clone https://github.com/Sbboss/mlds490-hw1 && cd mlds490-hw1
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install torch && pip install -r requirements.txt
```

## Part 1 (CPU, a few seconds)

```bash
cd shuttle
python run_shuttle.py && python validation.py && python -m pytest -q test_shuttle.py
cd ..
```

## Validation for Parts 2 and 3 (CPU)

```bash
python -m validation.run_validation
python -m validation.run_validation --pong      # needs ale-py, takes a few minutes
python -m validation.test_tb5                   # gradient-variance experiment, needs C0 seed 0
```

## CartPole (CPU, a few minutes per seed)

```bash
python -m pg.train_cartpole --baseline none --seed 0 --episodes 2000 \
    --out results/cartpole/C0_none/seed0
python -m pg.evaluate --env cartpole --run results/cartpole/C0_none/seed0
python -m pg.plot_results
```

`--baseline value` and `--baseline ema` give the Part 3 runs. `scripts/run_cartpole_all.sh` runs the whole matrix:

| id | config | seeds | episodes |
|---|---|---|---|
| C0 | no baseline, lr 0.01 | 0 to 4 | 2000 |
| C1 | value baseline | 0 to 4 | 2000 |
| C2 | EMA baseline | 0 to 4 | 2000 |
| C3 | `--gamma_weight on` | 0 to 2 | 2000 |
| C4 | lr 0.001 | 0 to 2 | 2000 |
| C5 | `--reward_sign -1` | 0 to 2 | 300 |
| C7 | 8 episodes per update, lr 0.01 | 0 to 2 | 2000 |
| C8 | lr 0.003 | 0 to 2 | 2000 |

## Pong (GPU, run on deepdish)

Pong needs exactly one GPU on deepdish. Torch numbers the cards differently from `nvidia-smi`, so check `nvidia-smi` and confirm that only the card you meant to use has memory in use. `scripts/remote_setup.sh` sets up the environment.

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
nice -n 10 python -m pg.train_pong --baseline none --lr 0.01 --workers 16 \
    --episodes_per_update 16 --max_episodes 25000 --max_hours 4 --seed 0 \
    --out results/pong/P0_none/seed0
python -m pg.evaluate --env pong --run results/pong/P0_none/seed0 --workers 16
```

Use `--baseline value` or `--baseline ema` for Part 3. The extended runs (lr 0.001, no early stop, 3-hour cap) and their exact commands are in the "How to run" section of the report. `tmux` isn't installed on that node, so long jobs run under `nohup setsid` and write `log.txt`. `bash scripts/sync_down.sh` pulls the results back.

## Rebuilding the report

```bash
python scripts/build_report.py
```

This converts `report/report.md` to HTML with Python-Markdown, inlines the figures and prints the PDF with headless Chrome. If Chrome is somewhere else on your machine, change the `CHROME` path at the top of the script.

## Rough runtimes

- Part 1 solvers and 38 checks: a few seconds.
- Policy-gradient validation without Pong: a few seconds. With `--pong`: several minutes.
- CartPole, 2000 episodes, one seed: a few minutes.
- Pong: 5,000 to 8,000 episodes in 15 to 20 minutes at lr 0.01 before the early stop. The 3-hour extended runs played about 40,000 episodes each.

## Hardware

Part 1 and the shuttle tests ran on the local Mac. The policy-gradient validation suite (including `--pong`) ran on the local Mac. The CartPole matrix ran on the local Mac on CPU. Pong ran on `mlds-deepdish5` (48 CPUs, 93 GB RAM) with one RTX PRO 5000 Blackwell GPU, 16 actor processes per job and at most 32 across jobs.
