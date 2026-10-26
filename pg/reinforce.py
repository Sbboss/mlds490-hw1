"""REINFORCE loss and baselines.

Estimator (D1):
    L = -(1/B) Σ_episodes Σ_t  w_t · logπ(a_t|s_t) · (G_t − b_t)

b_t is action-independent (0, a constant EMA, or V_φ(s_t)), so the estimator
stays unbiased for the corresponding objective. Baseline values are detached
inside the loss so the value net cannot sneak a gradient through the policy
objective (T-B3).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F


TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]


def _as_episode_list(x) -> list:
    if x is None:
        return []
    if torch.is_tensor(x):
        return [x]
    return list(x)


def reinforce_loss(
    logp: TensorOrList,
    G: TensorOrList,
    baseline_values: Optional[TensorOrList] = None,
    weights: Optional[TensorOrList] = None,
) -> torch.Tensor:
    """Sum over timesteps of each episode, then average over the batch.

    Why sum-then-average-episodes (not mean over timesteps): a longer CartPole
    episode has more terms, so its gradient is larger — that is the signal
    T-C7 checks. Mean-over-timesteps would cancel most of that.
    """
    logps = _as_episode_list(logp)
    Gs = _as_episode_list(G)
    assert len(logps) == len(Gs) and len(logps) > 0
    B = len(logps)
    bs = _as_episode_list(baseline_values) if baseline_values is not None else [None] * B
    ws = _as_episode_list(weights) if weights is not None else [None] * B
    if len(bs) == 1 and B > 1:
        bs = bs * B
    if len(ws) == 1 and B > 1:
        ws = ws * B

    total = logps[0].new_zeros(())
    for i in range(B):
        lp = logps[i]
        g = Gs[i]
        if not torch.is_tensor(g):
            g = torch.as_tensor(g, dtype=lp.dtype, device=lp.device)
        else:
            g = g.to(device=lp.device, dtype=lp.dtype)
        assert lp.shape == g.shape, f"len(logp)={tuple(lp.shape)} != len(G)={tuple(g.shape)}"
        adv = g
        if bs[i] is not None:
            # Detach here even if the caller forgot: value params must not
            # receive policy-loss gradients (T-B3).
            b = bs[i]
            if not torch.is_tensor(b):
                b = torch.as_tensor(b, dtype=lp.dtype, device=lp.device)
            adv = adv - b.detach()
        if ws[i] is not None:
            w = ws[i]
            if not torch.is_tensor(w):
                w = torch.as_tensor(w, dtype=lp.dtype, device=lp.device)
            contrib = (w * lp * adv).sum()
        else:
            contrib = (lp * adv).sum()
        total = total + contrib

    loss = -total / B
    assert torch.isfinite(loss), f"non-finite REINFORCE loss: {loss}"
    return loss


class EMABaseline:
    """Constant baseline = EMA of *previous* batches' mean return (D5).

    The value used for the current batch is the EMA *before* incorporating
    that batch. Updating first would make b depend on the sampled actions
    of the current batch and bias the estimator (T-B4).
    First batch uses b=0 (no previous data).
    """

    def __init__(self, decay: float = 0.9):
        self.decay = float(decay)
        self.b = 0.0
        self.n_seen = 0

    def value(self) -> float:
        return float(self.b)

    def update(self, mean_G: float) -> None:
        mean_G = float(mean_G)
        if self.n_seen == 0:
            self.b = mean_G
        else:
            self.b = self.decay * self.b + (1.0 - self.decay) * mean_G
        self.n_seen += 1

    def state_dict(self) -> dict:
        return {"decay": self.decay, "b": self.b, "n_seen": self.n_seen}

    def load_state_dict(self, d: dict) -> None:
        self.decay = float(d["decay"])
        self.b = float(d["b"])
        self.n_seen = int(d["n_seen"])


class ValueBaseline:
    """Learned V_φ(s) trained with MSE to the same G_t the policy uses (D5, T-B6).

    Own Adam, lr=1e-3 for both games (stated in the report). One pass over the
    batch per policy update; Pong uses minibatches of 4096 because a 16-episode
    batch is ~50k states.
    """

    def __init__(self, net: torch.nn.Module, lr: float = 1e-3, minibatch: int = 4096):
        self.net = net
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.minibatch = int(minibatch)
        self.lr = float(lr)

    @torch.no_grad()
    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def values_with_grad(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    def update(self, states: torch.Tensor, G: torch.Tensor) -> float:
        """One pass over (states, G). Called *after* the policy step."""
        self.net.train()
        G = G.detach()
        n = int(states.shape[0])
        if n == 0:
            return 0.0
        mb = min(self.minibatch, n)
        # index_select + detach so a saved activation is a minibatch, not a
        # view that keeps the whole episode-cat (tens of thousands of Pong
        # frames) alive for every minibatch graph. That view was filling the
        # 48 GB card across updates.
        perm = torch.randperm(n, device=states.device)
        last = 0.0
        for start in range(0, n, mb):
            sl = perm[start:start + mb]
            xb = states.index_select(0, sl).detach()
            yb = G.index_select(0, sl).detach()
            self.opt.zero_grad(set_to_none=True)
            pred = self.net(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            self.opt.step()
            last = float(loss.detach())
            del pred, loss, xb, yb
        return last


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def clip_grad_(params, max_norm: float) -> float:
    if max_norm is None or max_norm <= 0:
        return grad_norm(params)
    return float(torch.nn.utils.clip_grad_norm_(list(params), max_norm))
