"""All required figures. Missing inputs are skipped so we can plot incrementally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pg.common import AQUA, BLUE, INK, INK2, ORANGE, apply_plot_style, moving_average

FIG = Path("figures")
RES = Path("results")


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def _curve(path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def plot_training(csv_path, title, out_name, hline=None, ylim=None, reward_col="reward"):
    df = _curve(csv_path)
    if df is None:
        print("skip (missing)", csv_path)
        return
    r = df[reward_col].to_numpy(dtype=np.float64)
    ma = moving_average(r, 100)
    x = np.arange(1, len(r) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, r, color=BLUE, alpha=0.25, lw=0.8, label="episode reward")
    ax.plot(x, ma, color=BLUE, lw=2.2, label="100-episode mean")
    if hline is not None:
        ax.axhline(hline, color=INK2, ls="--", lw=1.0, label=str(hline))
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("training episode")
    ax.set_ylabel("undiscounted episode reward")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_name)


def plot_hist(csv_path, title, out_name, bins, xlim=None, greedy_csv=None):
    p = Path(csv_path)
    if not p.exists():
        print("skip (missing)", csv_path)
        return
    r = pd.read_csv(p)["reward"].to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(r, bins=bins, color=BLUE, edgecolor=INK, linewidth=0.4)
    ax.axvline(r.mean(), color=ORANGE, lw=1.6, label=f"mean={r.mean():.2f}")
    ax.set_xlabel("episode reward (final policy, sampled)")
    ax.set_ylabel("count")
    extra = ""
    if greedy_csv and Path(greedy_csv).exists():
        g = pd.read_csv(greedy_csv)["reward"].to_numpy()
        extra = f"; greedy mean={g.mean():.2f} ± {g.std(ddof=1):.2f}"
    ax.set_title(title)
    # Put the annotation on the empty side of the axis so it does not sit on the bar.
    on_right = False
    if xlim is not None:
        on_right = float(r.mean()) > 0.5 * (xlim[0] + xlim[1])
    ax.text(0.02 if on_right else 0.98, 0.94,
            f"n={len(r)}  mean={r.mean():.2f} ± {r.std(ddof=1):.2f}{extra}",
            transform=ax.transAxes, ha="left" if on_right else "right",
            va="top", color=INK2, fontsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    _save(fig, out_name)


def plot_multiseed(configs, title, out_name, hline=None, ylim=None, legend_loc=None,
                   legend_below=False):
    """configs: list of (label, color, list[csv paths])."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    any_ = False
    for label, color, paths in configs:
        series = []
        for p in paths:
            df = _curve(p)
            if df is None:
                continue
            series.append(moving_average(df["reward"].to_numpy(dtype=np.float64), 100))
        if not series:
            continue
        any_ = True
        M = max(len(s) for s in series)
        arr = np.full((len(series), M), np.nan)
        for i, s in enumerate(series):
            arr[i, :len(s)] = s
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1) if len(series) > 1 else np.zeros_like(mu)
        x = np.arange(1, M + 1)
        ax.plot(x, mu, color=color, lw=2.2, label=label)
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    if not any_:
        plt.close(fig)
        print("skip (no series)", out_name)
        return
    if hline is not None:
        ax.axhline(hline, color=INK2, ls="--", lw=1.0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("training episode")
    ax.set_ylabel("100-episode mean reward")
    ax.set_title(title)
    if legend_below:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
        FIG.mkdir(exist_ok=True)
        path = FIG / out_name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)
        return
    if legend_loc:
        ax.legend(loc=legend_loc)
    else:
        ax.legend()
    _save(fig, out_name)


def plot_tb5():
    p = Path("results/validation/tb5.json")
    if not p.exists():
        print("skip T-B5 figure")
        return
    d = json.loads(p.read_text())
    tags = [k for k in ("ep200", "ep1000", "final") if k in d]
    if not tags:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    x = np.arange(len(tags))
    r2 = [d[t]["R2"] for t in tags]
    axes[0].bar(x, r2, color=BLUE, width=0.6)
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_xticks(x, tags)
    axes[0].set_ylabel("value-net $R^2$ on held-out $G_t$")
    axes[0].set_title("T-B5: explained variance")
    w = 0.25
    axes[1].bar(x - w, [d[t]["ratio_const"] for t in tags], width=w, color=ORANGE, label="const / b=0")
    axes[1].bar(x + 0, [d[t]["ratio_V"] for t in tags], width=w, color=AQUA, label="V / b=0")
    axes[1].axhline(1.0, color=INK2, ls="--", lw=1)
    axes[1].set_xticks(x, tags)
    axes[1].set_ylabel("tr(Cov) ratio vs $b=0$")
    axes[1].set_title("T-B5: gradient-variance ratio")
    axes[1].legend()
    _save(fig, "fig_tb5_gradient_variance.png")


def main():
    apply_plot_style()
    FIG.mkdir(exist_ok=True)

    # Part 2 / 3 CartPole seed 0
    plot_training("results/cartpole/C0_none/seed0/curve.csv",
                  "CartPole-v1  REINFORCE  no baseline  lr=0.01  γ=0.95  seed=0",
                  "fig_cartpole_p2_curve.png", hline=500)
    plot_training("results/cartpole/C1_value/seed0/curve.csv",
                  "CartPole-v1  REINFORCE  value baseline  lr=0.01  γ=0.95  seed=0",
                  "fig_cartpole_p3_curve.png", hline=500)
    plot_training("results/cartpole/C2_ema/seed0/curve.csv",
                  "CartPole-v1  REINFORCE  EMA baseline  lr=0.01  γ=0.95  seed=0",
                  "fig_cartpole_ema_curve.png", hline=500)
    plot_hist("results/cartpole/C0_none/seed0/eval_sampled.csv",
              "CartPole Part 2: 500 sampled rollouts of the final policy (seed 0)",
              "fig_cartpole_p2_hist.png", bins=np.arange(0, 525, 25), xlim=(0, 510),
              greedy_csv="results/cartpole/C0_none/seed0/eval_greedy.csv")
    plot_hist("results/cartpole/C1_value/seed0/eval_sampled.csv",
              "CartPole Part 3: 500 sampled rollouts of the final policy (seed 0)",
              "fig_cartpole_p3_hist.png", bins=np.arange(0, 525, 25), xlim=(0, 510),
              greedy_csv="results/cartpole/C1_value/seed0/eval_greedy.csv")

    plot_multiseed([
        ("no baseline (Part 2)", BLUE,
         [f"results/cartpole/C0_none/seed{s}/curve.csv" for s in range(5)]),
        ("value baseline (Part 3)", ORANGE,
         [f"results/cartpole/C1_value/seed{s}/curve.csv" for s in range(5)]),
        ("EMA baseline", AQUA,
         [f"results/cartpole/C2_ema/seed{s}/curve.csv" for s in range(5)]),
    ], "CartPole: 5-seed mean ± std of the 100-episode moving average",
       "fig_cartpole_multiseed.png", hline=500)

    plot_multiseed([
        ("w_t = 1 (C0)", BLUE, [f"results/cartpole/C0_none/seed{s}/curve.csv" for s in range(3)]),
        (r"$w_t = \gamma^t$ (C3)", ORANGE,
         [f"results/cartpole/C3_gamma_weight/seed{s}/curve.csv" for s in range(3)]),
        ("lr = 0.001 (C4)", AQUA,
         [f"results/cartpole/C4_lr1e3/seed{s}/curve.csv" for s in range(3)]),
        ("8 episodes/update (C7)", INK,
         [f"results/cartpole/C7_batch8/seed{s}/curve.csv" for s in range(3)]),
    ], "CartPole ablations: γ^t weighting, learning rate, batch (3 seeds)",
       "fig_cartpole_ablations.png", hline=500)

    # Pong
    plot_training("results/pong/P0_none/seed0/episode_rewards.csv",
                  "Pong-v5  REINFORCE  no baseline  lr=0.01  γ=0.99  seed=0",
                  "fig_pong_p2_curve.png", ylim=(-21, 21))
    plot_training("results/pong/P1_value/seed0/episode_rewards.csv",
                  "Pong-v5  REINFORCE  value baseline  lr=0.01  γ=0.99  seed=0",
                  "fig_pong_p3_curve.png", ylim=(-21, 21))
    plot_hist("results/pong/P0_none/seed0/eval_sampled.csv",
              "Pong Part 2: 500 sampled rollouts of the final policy",
              "fig_pong_p2_hist.png", bins=np.arange(-21.5, 22.5, 1), xlim=(-22, 22),
              greedy_csv="results/pong/P0_none/seed0/eval_greedy.csv")
    plot_hist("results/pong/P1_value/seed0/eval_sampled.csv",
              "Pong Part 3: 500 sampled rollouts of the final policy",
              "fig_pong_p3_hist.png", bins=np.arange(-21.5, 22.5, 1), xlim=(-22, 22),
              greedy_csv="results/pong/P1_value/seed0/eval_greedy.csv")
    # Extended runs: lr 0.001, early-stop off, 3-hour cap. Not the primary results.
    plot_training("results/pong/ext_P0_lr1e-3/seed0/episode_rewards.csv",
                  "Pong extended  no baseline  lr=0.001  γ=0.99  seed=0",
                  "fig_pong_ext_p0_curve.png", ylim=(-21, 21))
    plot_training("results/pong/ext_P1_lr1e-3/seed0/episode_rewards.csv",
                  "Pong extended  value baseline  lr=0.001  γ=0.99  seed=0",
                  "fig_pong_ext_p1_curve.png", ylim=(-21, 21))
    plot_hist("results/pong/ext_P0_lr1e-3/seed0/eval_sampled.csv",
              "Pong extended, no baseline: 500 sampled rollouts (lr=0.001)",
              "fig_pong_ext_p0_hist.png", bins=np.arange(-21.5, 22.5, 1), xlim=(-22, 22),
              greedy_csv="results/pong/ext_P0_lr1e-3/seed0/eval_greedy.csv")
    plot_hist("results/pong/ext_P1_lr1e-3/seed0/eval_sampled.csv",
              "Pong extended, value baseline: 500 sampled rollouts (lr=0.001)",
              "fig_pong_ext_p1_hist.png", bins=np.arange(-21.5, 22.5, 1), xlim=(-22, 22),
              greedy_csv="results/pong/ext_P1_lr1e-3/seed0/eval_greedy.csv")

    # LR sweep + overlays
    sweep = []
    for lr, tag in [("1e-2", "1e-2"), ("1e-3", "1e-3"), ("3e-4", "3e-4"), ("1e-4", "1e-4")]:
        sweep.append((f"lr={tag}", None, [f"results/pong/sweep_lr{tag}/seed0/episode_rewards.csv"]))
    # color them
    colors = [BLUE, ORANGE, AQUA, INK]
    sweep = [(lab, colors[i], paths) for i, (lab, _, paths) in enumerate(sweep)]
    plot_multiseed(sweep, "Pong LR sweep (baseline none, 2000 episodes, training reward)",
                   "fig_pong_lr_sweep.png", ylim=(-21, 21))
    plot_multiseed([
        ("P0 lr 0.01, no baseline", BLUE, ["results/pong/P0_none/seed0/episode_rewards.csv"]),
        ("P1 lr 0.01, value", ORANGE, ["results/pong/P1_value/seed0/episode_rewards.csv"]),
        ("P2 lr 0.01, EMA", AQUA, ["results/pong/P2_ema/seed0/episode_rewards.csv"]),
        ("ext lr 0.001, no baseline", INK, ["results/pong/ext_P0_lr1e-3/seed0/episode_rewards.csv"]),
        ("ext lr 0.001, value", "#6b4c9a", ["results/pong/ext_P1_lr1e-3/seed0/episode_rewards.csv"]),
    ], "Pong: 100-episode mean, primary runs and the lr 0.001 extended runs",
       "fig_pong_full_overlay.png", ylim=(-21, 21), legend_below=True)

    plot_tb5()
    print("plot_results done")


if __name__ == "__main__":
    main()
