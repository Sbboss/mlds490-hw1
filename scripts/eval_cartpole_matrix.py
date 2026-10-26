"""Evaluate every CartPole run that has final.pt and no eval_stats.json yet.

Spawn-safe: this file is a real module, unlike a stdin script.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import multiprocessing as mp


def evaluate_one(run: str) -> str:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    try:
        from pg.evaluate import evaluate_run
        evaluate_run(run, "cartpole", n=500, eval_seed=10_000, hidden=128)
        return f"OK {run}"
    except Exception:
        return f"FAIL {run}\n{traceback.format_exc()}"


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    runs = sorted(p.parent for p in (root / "results" / "cartpole").glob("*/*/final.pt"))
    todo = [str(r) for r in runs if not (r / "eval_stats.json").exists()]
    print(f"eval todo {len(todo)} / {len(runs)}", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        for msg in pool.imap_unordered(evaluate_one, todo):
            print(msg, flush=True)
    print("ALL_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
