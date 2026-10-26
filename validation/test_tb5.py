"""T-B5 gradient-variance experiment (runs AFTER C0 seed 0 has checkpoints)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pg.common import discount_returns, set_seed
from pg.models import MLPPolicy, ValueNet
from pg.train_cartpole import make_cartpole, rollout
from validation.records import record


def _flat_grad(policy, roll, G, baseline):
    # Recompute logπ from stored (s,a) so we can form several estimators on
    # the same trajectories without reusing a freed autograd graph.
    x = torch.as_tensor(roll["states"], dtype=torch.float32)
    dist = policy.dist(x)
    lp = dist.log_prob(torch.as_tensor(roll["actions"], dtype=torch.int64))
    g = torch.as_tensor(G, dtype=torch.float32)
    adv = g if baseline is None else g - baseline.detach()
    policy.zero_grad(set_to_none=True)
    (-(lp * adv).sum()).backward()
    return np.concatenate([
        p.grad.detach().reshape(-1).cpu().numpy().copy() for p in policy.parameters()
    ])


def run_tb5(run_dir="results/cartpole/C0_none/seed0", hidden=128, n=1000):
    run_dir = Path(run_dir)
    tags = [("ep200", run_dir / "ckpt_200.pt"),
            ("ep1000", run_dir / "ckpt_1000.pt"),
            ("final", run_dir / "final.pt")]
    out = {}
    env = make_cartpole()
    for tag, path in tags:
        if not path.exists():
            record(f"T-B5_{tag}_missing", "prediction", False, f"{path} missing",
                   "checkpoint present", "T-B5 needs C0 seed 0 checkpoints.")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        policy = MLPPolicy(4, hidden, 2)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        # collect n fresh episodes with frozen policy
        rolls, Gs = [], []
        set_seed(12345)
        policy.eval()
        for i in range(n):
            roll = rollout(env, policy, seed=50_000 + i)
            # drop the per-step logp tensors (they hold graphs we do not need)
            roll = {k: v for k, v in roll.items() if k != "logps"}
            G = discount_returns(roll["rewards"], 0.95)
            rolls.append(roll)
            Gs.append(G)
        A, B = slice(0, n // 2), slice(n // 2, n)
        # constant baseline from A
        const = float(np.concatenate(Gs[A]).mean())
        # value net on A
        vnet = ValueNet(4, hidden)
        opt = torch.optim.Adam(vnet.parameters(), lr=1e-3)
        states_A = torch.as_tensor(np.concatenate([r["states"] for r in rolls[A]]), dtype=torch.float32)
        GA = torch.as_tensor(np.concatenate(Gs[A]), dtype=torch.float32)
        # Adam steps are ~lr in magnitude (T-R4), so 40 epochs never leave
        # the near-zero init when G is O(10). Train until the MSE plateaus.
        prev = None
        epochs_run = 0
        for epoch in range(4000):
            pred = vnet(states_A)
            loss = F.mse_loss(pred, GA)
            opt.zero_grad(); loss.backward(); opt.step()
            epochs_run = epoch + 1
            cur = float(loss.detach())
            if prev is not None and abs(prev - cur) < 1e-4 and epoch > 200:
                break
            prev = cur
        states_B = torch.as_tensor(np.concatenate([r["states"] for r in rolls[B]]), dtype=torch.float32)
        GB = torch.as_tensor(np.concatenate(Gs[B]), dtype=torch.float32)
        with torch.no_grad():
            VB = vnet(states_B)
        r2 = 1.0 - float(torch.var(GB - VB) / torch.var(GB))
        # per-episode grads on B
        g0, gc, gv = [], [], []
        for roll, G in zip(rolls[B], Gs[B]):
            st = torch.as_tensor(roll["states"], dtype=torch.float32)
            with torch.no_grad():
                v = vnet(st)
            g0.append(_flat_grad(policy, roll, G, None))
            gc.append(_flat_grad(policy, roll, G, torch.full((len(G),), const)))
            gv.append(_flat_grad(policy, roll, G, v))
        g0, gc, gv = map(np.stack, (g0, gc, gv))
        def tr(x):
            # trace of covariance of the per-episode gradient vectors
            return float(np.var(x, axis=0, ddof=1).sum())
        def cos(a, b):
            ma, mb = a.mean(0), b.mean(0)
            return float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))
        rec = {
            "R2": r2,
            "value_epochs": epochs_run,
            "train_mse": float(prev if prev is not None else 0.0),
            "tr_b0": tr(g0),
            "tr_const": tr(gc),
            "tr_V": tr(gv),
            "ratio_const": tr(gc) / tr(g0),
            "ratio_V": tr(gv) / tr(g0),
            "cos_0_const": cos(g0, gc),
            "cos_0_V": cos(g0, gv),
            "const": const,
        }
        out[tag] = rec
        record(f"T-B5_{tag}_R2", "prediction", r2 > 0.5,
               f"R2={r2:.3f} ratios const/V vs b0 = {rec['ratio_const']:.3f}/{rec['ratio_V']:.3f} "
               f"cos={rec['cos_0_const']:.3f}/{rec['cos_0_V']:.3f}",
               "R² > 0.5 at final (guess); ratios < 1; cosine ≈ 1",
               "Paired gradient-variance comparison on held-out episodes.")
    env.close()
    Path("results/validation").mkdir(parents=True, exist_ok=True)
    Path("results/validation/tb5.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from validation.records import clear, snapshot
    clear()
    run_tb5()
    print(json.dumps(snapshot(), indent=2))
