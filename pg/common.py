"""Shared helpers for REINFORCE: returns, seeds, logging, plot style.

Why these live here: every trainer, evaluator, and validation test must use
the same discounting and the same 100-episode moving average definition,
otherwise a plotting bug can masquerade as a learning result.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch


# Plot style copied from shuttle/run_shuttle.py so Part 1 and Parts 2/3 match.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def apply_plot_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def discount_returns(rewards, gamma: float, reset_on_nonzero: bool = False) -> np.ndarray:
    """Backward Monte-Carlo returns G_t = r_t + gamma * G_{t+1}.

    If reset_on_nonzero is True (Pong, D2): when r_t != 0 we zero the running
    return *before* adding r_t. That makes each point an independent subgame
    so a later +1 cannot leak credit into an earlier rally. Without the reset,
    G at the losing point includes later points (see T-R1).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = rewards.shape[0]
    out = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in range(T - 1, -1, -1):
        r = float(rewards[t])
        if reset_on_nonzero and r != 0.0:
            G = 0.0
        G = r + gamma * G
        out[t] = G
    return out


def gamma_weights(T: int, gamma: float) -> np.ndarray:
    """w_t = gamma^t, the exact discounted-objective gradient weight (D1)."""
    return gamma ** np.arange(T, dtype=np.float64)


def moving_average(x, w: int = 100) -> np.ndarray:
    """Trailing window mean; NaN for indices < w-1 (no partial windows, D7)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.size < w:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) for T-C4 bit-identity."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def episode_generator(seed: int) -> torch.Generator:
    """Dedicated RNG so action sampling does not consume the global torch RNG."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def sample_action(dist, generator=None, greedy: bool = False) -> torch.Tensor:
    """Sample an action; `Categorical.sample` has no `generator=` on some builds.

    torch.multinomial is the primitive Categorical uses internally, and it
    does accept a Generator, which is what T-C4's bit-identity needs.
    """
    if greedy:
        return torch.argmax(dist.logits, dim=-1)
    if generator is None:
        return dist.sample()
    return torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)


def standardize_returns(G: np.ndarray) -> np.ndarray:
    """(G - mean) / (std + 1e-8). Skip length-1 (std is 0 / undefined). T-R5."""
    G = np.asarray(G, dtype=np.float64)
    if G.size <= 1:
        return G.copy()
    std = float(G.std(ddof=0))  # unbiased=False
    return (G - G.mean()) / (std + 1e-8)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class CSVLogger:
    """Flush-every-write CSV so a killed job still has a usable curve."""

    def __init__(self, path, fieldnames, append: bool = False):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        mode = "a" if append and exists else "w"
        self._fh = open(self.path, mode, newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")
        if mode == "w":
            self._w.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _as_byte_tensor(t) -> torch.Tensor:
    """torch.save/load turns a ByteTensor into a plain uint8 Tensor, and
    set_rng_state rejects anything that is not the ByteTensor subclass."""
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    t = t.detach().cpu().contiguous()
    if not isinstance(t, torch.ByteTensor):
        t = t.type(torch.ByteTensor)
    return t


def load_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in state["cuda"]])
        except Exception:
            pass
