"""Pong-v5 REINFORCE: CPU actor pool + single-GPU learner.

Gymnasium vector envs are intentionally not used: 1.x auto-reset can splice
the next episode onto the end of the current one and silently corrupt
returns. A spawn-context pool with one env per worker avoids that.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from pg.common import (
    CSVLogger,
    discount_returns,
    ensure_dir,
    gamma_weights,
    load_rng_state,
    moving_average,
    save_rng_state,
    set_seed,
    standardize_returns,
)
from pg.models import make_policy, make_value, state_dict_numpy
from pg.pong_utils import init_pong_worker, play_one_episode
from pg.reinforce import EMABaseline, ValueBaseline, clip_grad_, reinforce_loss


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _recompute_logp(policy, states, actions):
    """Learner-side logπ with the weights that generated the batch (on-policy)."""
    x = states.float()
    dist = Categorical(logits=policy(x))
    return dist.log_prob(actions), dist.entropy().mean(), dist.probs


def save_latest(path, policy, opt, vbase, ema, episodes_done, updates, extra=None):
    payload = {
        "policy": policy.state_dict(),
        "optimizer": opt.state_dict(),
        "episodes_done": episodes_done,
        "updates": updates,
        "ema": ema.state_dict() if ema is not None else None,
        "value": None if vbase is None else vbase.net.state_dict(),
        "value_opt": None if vbase is None else vbase.opt.state_dict(),
        "rng": save_rng_state(),
    }
    if extra:
        payload.update(extra)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(args):
    out = ensure_dir(args.out)
    set_seed(args.seed)
    device = _device()
    print(f"pong device={device} cuda={torch.cuda.is_available()} "
          f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)

    policy = make_policy(args.arch, 6400, args.hidden, 2).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    ema = EMABaseline() if args.baseline == "ema" else None
    vbase = None
    if args.baseline == "value":
        vbase = ValueBaseline(make_value(args.arch, 6400, args.hidden).to(device),
                              lr=1e-3, minibatch=4096)

    start_ep = 0
    updates = 0
    resume_path = Path(args.out) / "latest.pt"
    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        opt.load_state_dict(ckpt["optimizer"])
        start_ep = int(ckpt["episodes_done"])
        updates = int(ckpt.get("updates", 0))
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        if vbase is not None and ckpt.get("value") is not None:
            vbase.net.load_state_dict(ckpt["value"])
            vbase.opt.load_state_dict(ckpt["value_opt"])
        if ckpt.get("rng"):
            try:
                load_rng_state(ckpt["rng"])
            except Exception as exc:
                print(f"rng restore skipped: {exc}", flush=True)
        print(f"resumed from {resume_path} at episode {start_ep}", flush=True)

    ctx = mp.get_context("spawn")
    n_workers = min(args.workers, args.episodes_per_update)
    pool = ctx.Pool(
        processes=n_workers,
        initializer=init_pong_worker,
        initargs=(args.arch, args.hidden, False),
    )

    curve_fields = ["update", "episodes_done", "env_steps", "batch_mean_reward",
                    "batch_mean_len", "entropy", "frac_idx0", "grad_norm", "loss",
                    "ma100", "wall_s", "steps_per_s"]
    ep_fields = ["episode", "reward", "length", "points_won", "points_lost"]
    clog = CSVLogger(out / "curve.csv", curve_fields, append=args.resume and start_ep > 0)
    elog = CSVLogger(out / "episode_rewards.csv", ep_fields, append=args.resume and start_ep > 0)

    all_rewards = []
    if args.resume and (out / "episode_rewards.csv").exists():
        try:
            import pandas as pd
            prev = pd.read_csv(out / "episode_rewards.csv")
            all_rewards = prev["reward"].tolist()
        except Exception:
            pass

    t0 = time.time()
    env_steps = 0
    episodes_done = start_ep
    max_seconds = args.max_hours * 3600.0
    log_path = out / "log.txt"
    log_fh = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    try:
        while episodes_done < args.max_episodes:
            if time.time() - t0 > max_seconds:
                log(f"hit max_hours={args.max_hours}; stopping")
                break

            weights = state_dict_numpy(policy)
            tasks = []
            for i in range(args.episodes_per_update):
                ep_idx = episodes_done + i
                tasks.append({
                    "weights": weights,
                    "seed": args.seed * 1_000_003 + ep_idx,
                    "greedy": False,
                })
            t_batch = time.time()
            trajs = pool.map(play_one_episode, tasks)
            batch_s = time.time() - t_batch

            logps, Gs, bases, ws, states_t, G_t = [], [], [], [], [], []
            rewards_b, lens_b, n0, n_act = [], [], 0, 0
            for tr in trajs:
                elog.write({
                    "episode": episodes_done + 1 + len(rewards_b),
                    "reward": tr["reward"],
                    "length": tr["length"],
                    "points_won": tr["points_won"],
                    "points_lost": tr["points_lost"],
                })
                all_rewards.append(tr["reward"])
                rewards_b.append(tr["reward"])
                lens_b.append(tr["length"])
                env_steps += tr["length"]
                n0 += int((tr["actions"] == 0).sum())
                n_act += int(tr["actions"].size)

                st = torch.from_numpy(np.ascontiguousarray(tr["states"])).to(device)
                ac = torch.from_numpy(tr["actions"].astype(np.int64)).to(device)
                lp, _, _ = _recompute_logp(policy, st, ac)
                G = discount_returns(tr["rewards"], args.gamma, reset_on_nonzero=bool(args.pong_reset_at_point))
                if args.standardize:
                    G = standardize_returns(G)
                g_t = torch.as_tensor(G, dtype=torch.float32, device=device)
                logps.append(lp)
                Gs.append(g_t)
                states_t.append(st)
                G_t.append(g_t)
                if args.gamma_weight == "on":
                    ws.append(torch.as_tensor(gamma_weights(len(G), args.gamma),
                                              dtype=torch.float32, device=device))
                if args.baseline == "ema":
                    bases.append(torch.full_like(g_t, ema.value()))
                elif args.baseline == "value":
                    bases.append(vbase.values(st.float()))

            loss = reinforce_loss(
                logps, Gs,
                baseline_values=bases if args.baseline != "none" else None,
                weights=ws if args.gamma_weight == "on" else None,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = clip_grad_(policy.parameters(), args.grad_clip)
            opt.step()

            loss_value = float(loss.detach())
            if args.baseline == "value":
                cat_s = torch.cat(states_t, 0).float()
                cat_g = torch.cat(G_t, 0)
                vbase.update(cat_s, cat_g)
                del cat_s, cat_g
            if args.baseline == "ema":
                ema.update(float(torch.cat(G_t).mean().item()))

            # Entropy of this batch before the graphs are dropped.
            with torch.no_grad():
                ent_sum, ntok = 0.0, 0
                for st, ac in zip(states_t, [tr["actions"] for tr in trajs]):
                    ac_t = torch.as_tensor(ac.astype("int64"), device=device)
                    _, ent, _ = _recompute_logp(policy, st, ac_t)
                    ent_sum += float(ent) * st.shape[0]
                    ntok += int(st.shape[0])
                entropy = ent_sum / max(ntok, 1)

            # The value pass was retaining every frame's activation and filled
            # the 48 GB card (measured 41 GB). Drop the batch before the actors
            # come back with the next one.
            del logps, Gs, bases, ws, states_t, G_t, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

            episodes_done += len(trajs)
            updates += 1
            ma = moving_average(np.asarray(all_rewards, dtype=np.float64), 100)
            ma100 = float(ma[-1]) if ma.size and not np.isnan(ma[-1]) else float("nan")
            sps = env_steps / max(time.time() - t0, 1e-6)
            row = {
                "update": updates,
                "episodes_done": episodes_done,
                "env_steps": env_steps,
                "batch_mean_reward": float(np.mean(rewards_b)),
                "batch_mean_len": float(np.mean(lens_b)),
                "entropy": entropy,
                "frac_idx0": n0 / max(n_act, 1),
                "grad_norm": float(gn),
                "loss": loss_value,
                "ma100": ma100,
                "wall_s": time.time() - t0,
                "steps_per_s": sps,
            }
            clog.write(row)
            log(
                f"up={updates} ep={episodes_done} R={row['batch_mean_reward']:.2f} "
                f"len={row['batch_mean_len']:.0f} H={row['entropy']:.3f} "
                f"p0={row['frac_idx0']:.2f} ma100={ma100:.2f} "
                f"sps={sps:.0f} batch={batch_s:.1f}s wall={row['wall_s']:.0f}s"
            )

            save_latest(out / "latest.pt", policy, opt, vbase, ema, episodes_done, updates,
                        extra={"args": vars(args)})
            if episodes_done % 200 < args.episodes_per_update:
                save_latest(out / f"ckpt_{episodes_done}.pt", policy, opt, vbase, ema,
                            episodes_done, updates)

            # early-stop: essentially won, or flat for 3000 episodes
            if len(all_rewards) >= 100 and ma100 >= 15:
                log("early stop: MA100 >= +15")
                break
            if len(all_rewards) >= 3100:
                ma_now = ma100
                ma_old = float(moving_average(np.asarray(all_rewards, dtype=np.float64), 100)[-3001])
                if not np.isnan(ma_old) and abs(ma_now - ma_old) < 0.3:
                    log(f"early stop: MA100 flat ({ma_old:.2f} -> {ma_now:.2f}) over 3000 episodes")
                    break

            if not np.isfinite(row["loss"]) or not np.isfinite(row["entropy"]):
                log("NaN detected; stopping")
                break
    finally:
        pool.close()
        pool.join()
        clog.close()
        elog.close()
        log_fh.close()
        # keep a copy named final.pt for evaluate.py
        if (out / "latest.pt").exists():
            import shutil
            shutil.copy2(out / "latest.pt", out / "final.pt")
        (out / "run_meta.json").write_text(json.dumps({
            "episodes_done": episodes_done, "updates": updates,
            "seconds": time.time() - t0, "device": str(device),
            "args": vars(args),
        }, indent=2))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--episodes_per_update", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max_episodes", type=int, default=25000)
    p.add_argument("--max_hours", type=float, default=4.0)
    p.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--baseline", choices=["none", "ema", "value"], default="none")
    p.add_argument("--standardize", type=int, default=0)
    p.add_argument("--pong_reset_at_point", type=int, default=1)
    p.add_argument("--gamma_weight", choices=["off", "on"], default="off")
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=str, required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    # spawn is the default on macOS; set it here so `python -m` works on Linux too
    mp.set_start_method("spawn", force=True)
    main()
