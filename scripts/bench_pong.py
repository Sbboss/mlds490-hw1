"""T-P11: steps/s vs worker count. Choose the smallest n within 10% of the best."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from pg.models import MLPPolicy, state_dict_numpy
from pg.pong_utils import init_pong_worker, play_one_episode


def bench_one(n_workers: int, n_episodes: int, hidden=200) -> dict:
    policy = MLPPolicy(6400, hidden, 2)
    weights = state_dict_numpy(policy)
    ctx = mp.get_context("spawn")
    # warmup: one episode to start envs
    pool = ctx.Pool(n_workers, initializer=init_pong_worker, initargs=("mlp", hidden, False))
    tasks = [{"weights": weights, "seed": 10_000 + i, "greedy": False} for i in range(n_episodes)]
    t0 = time.time()
    trajs = pool.map(play_one_episode, tasks)
    dt = time.time() - t0
    pool.close()
    pool.join()
    steps = int(sum(t["length"] for t in trajs))
    return {
        "workers": n_workers,
        "episodes": n_episodes,
        "seconds": dt,
        "env_steps": steps,
        "steps_per_s": steps / dt,
        "episodes_per_s": n_episodes / dt,
        "mean_len": float(np.mean([t["length"] for t in trajs])),
        "mean_reward": float(np.mean([t["reward"] for t in trajs])),
    }


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)
    results = []
    # 8 episodes per setting so 1-worker is not endless; 32-worker still saturates
    for n in (1, 8, 16, 24, 32):
        print(f"bench workers={n}", flush=True)
        # more episodes at higher worker counts so the pool is busy
        n_ep = max(n, 8)
        rec = bench_one(n, n_ep)
        print(rec, flush=True)
        results.append(rec)
    best = max(r["steps_per_s"] for r in results)
    # smallest worker count within 10% of best
    within = [r for r in results if r["steps_per_s"] >= 0.9 * best]
    chosen = min(within, key=lambda r: r["workers"])
    budget = {
        "chosen_workers": chosen["workers"],
        "chosen_steps_per_s": chosen["steps_per_s"],
        "episodes_per_s": chosen["episodes_per_s"],
        "episodes_in_4h": chosen["episodes_per_s"] * 4 * 3600,
        "best_steps_per_s": best,
        "rows": results,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/pong_bench.json").write_text(json.dumps(budget, indent=2))
    print(json.dumps(budget, indent=2))


if __name__ == "__main__":
    main()
