"""Final-policy evaluation: 500 sampled + 500 greedy + last-20% checkpoint spread."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from pg.common import CSVLogger, ensure_dir, set_seed
from pg.models import make_policy, state_dict_numpy
from pg.pong_utils import init_pong_worker, play_one_episode
from pg.train_cartpole import classify_end, load_policy, make_cartpole, rollout


def _write_eval_csv(path, rows, fields):
    log = CSVLogger(path, fields, append=False)
    for r in rows:
        log.write(r)
    log.close()


def _stats(rewards, extra=None):
    r = np.asarray(rewards, dtype=np.float64)
    d = {
        "mean": float(r.mean()) if r.size else float("nan"),
        "std": float(r.std(ddof=1)) if r.size > 1 else 0.0,
        "min": float(r.min()) if r.size else float("nan"),
        "max": float(r.max()) if r.size else float("nan"),
        "n": int(r.size),
    }
    if extra:
        d.update(extra)
    return d


def eval_cartpole(ckpt_path, out_dir, n=500, seed=10_000, hidden=128, greedy=False):
    policy, _ = load_policy(ckpt_path, hidden=hidden)
    env = make_cartpole()
    rows, causes = [], []
    for i in range(n):
        roll = rollout(env, policy, seed=seed + i, greedy=greedy)
        cause = classify_end(roll["last_obs"], roll["terminated"], roll["truncated"], env)
        causes.append(cause)
        rows.append({
            "episode": i,
            "reward": roll["undiscounted"],
            "length": roll["length"],
            "terminated": int(roll["terminated"]),
            "truncated": int(roll["truncated"]),
            "cause": cause,
        })
    env.close()
    frac = {k: causes.count(k) / max(len(causes), 1)
            for k in ("x-limit", "angle", "time-limit", "unknown")}
    return rows, _stats([r["reward"] for r in rows], {"causes": frac})


def eval_pong(ckpt_path, out_dir, n=500, seed=10_000, workers=16, arch="mlp",
              hidden=200, greedy=False):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(arch, 6400, hidden, 2)
    policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
    weights = state_dict_numpy(policy)
    ctx = mp.get_context("spawn")
    n_workers = min(workers, n)
    pool = ctx.Pool(processes=n_workers, initializer=init_pong_worker,
                    initargs=(arch, hidden, False))
    tasks = [{"weights": weights, "seed": seed + i, "greedy": greedy} for i in range(n)]
    try:
        trajs = pool.map(play_one_episode, tasks)
    finally:
        pool.close()
        pool.join()
    rows = []
    for i, tr in enumerate(trajs):
        rows.append({
            "episode": i,
            "reward": tr["reward"],
            "length": tr["length"],
            "points_won": tr["points_won"],
            "points_lost": tr["points_lost"],
        })
    return rows, _stats([r["reward"] for r in rows])


def last20_checkpoints(run_dir: Path, total_hint: int | None = None):
    """Pick up to 5 numbered checkpoints from the last 20% of training."""
    ckpts = sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        fin = run_dir / "final.pt"
        return [fin] if fin.exists() else []
    nums = [int(p.stem.split("_")[1]) for p in ckpts]
    total = total_hint or nums[-1]
    cutoff = 0.8 * total
    late = [p for p, n in zip(ckpts, nums) if n >= cutoff]
    if not late:
        late = ckpts[-5:]
    if len(late) <= 5:
        return late
    idx = np.linspace(0, len(late) - 1, 5).round().astype(int)
    return [late[i] for i in idx]


def evaluate_run(run_dir, env, n=500, eval_seed=10_000, workers=16,
                 hidden=None, arch="mlp"):
    run_dir = Path(run_dir)
    out = ensure_dir(run_dir)
    final = run_dir / "final.pt"
    if not final.exists():
        final = run_dir / "latest.pt"
    if not final.exists():
        raise FileNotFoundError(f"no final.pt/latest.pt in {run_dir}")

    if env == "cartpole":
        hidden = hidden or 128
        rows_s, stats_s = eval_cartpole(final, out, n=n, seed=eval_seed, hidden=hidden, greedy=False)
        rows_g, stats_g = eval_cartpole(final, out, n=n, seed=eval_seed + 10_000, hidden=hidden, greedy=True)
        _write_eval_csv(out / "eval_sampled.csv", rows_s,
                        ["episode", "reward", "length", "terminated", "truncated", "cause"])
        _write_eval_csv(out / "eval_greedy.csv", rows_g,
                        ["episode", "reward", "length", "terminated", "truncated", "cause"])
        spread = {}
        for ck in last20_checkpoints(run_dir):
            _, st = eval_cartpole(ck, out, n=50, seed=eval_seed + 20_000, hidden=hidden, greedy=False)
            spread[ck.name] = st
    else:
        hidden = hidden or 200
        rows_s, stats_s = eval_pong(final, out, n=n, seed=eval_seed, workers=workers,
                                    arch=arch, hidden=hidden, greedy=False)
        rows_g, stats_g = eval_pong(final, out, n=n, seed=eval_seed + 10_000, workers=workers,
                                    arch=arch, hidden=hidden, greedy=True)
        _write_eval_csv(out / "eval_sampled.csv", rows_s,
                        ["episode", "reward", "length", "points_won", "points_lost"])
        _write_eval_csv(out / "eval_greedy.csv", rows_g,
                        ["episode", "reward", "length", "points_won", "points_lost"])
        spread = {}
        for ck in last20_checkpoints(run_dir):
            _, st = eval_pong(ck, out, n=50, seed=eval_seed + 20_000, workers=workers,
                              arch=arch, hidden=hidden, greedy=False)
            spread[ck.name] = st

    payload = {
        "sampled": stats_s,
        "greedy": stats_g,
        "n": n,
        "eval_seed": eval_seed,
        "ckpt": str(final),
    }
    (out / "eval_stats.json").write_text(json.dumps(payload, indent=2))
    (out / "checkpoint_spread.json").write_text(json.dumps(spread, indent=2))
    print(json.dumps({"sampled": stats_s, "greedy": stats_g, "spread": spread}, indent=2))
    return payload, spread


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--env", choices=["cartpole", "pong"], required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--arch", default="mlp")
    args = p.parse_args(argv)
    set_seed(args.seed)
    evaluate_run(args.run, args.env, n=args.n, eval_seed=args.seed,
                 workers=args.workers, hidden=args.hidden, arch=args.arch)


if __name__ == "__main__":
    main()
