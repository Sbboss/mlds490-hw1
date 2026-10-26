"""Pong preprocessing, action map, and CPU episode worker.

The assignment's preprocess() mutates its input (the slices are views).
We always copy first so a later `env.step` / render is not looking at a
zeroed frame (T-P3).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from pg.common import sample_action

# Network index 0 → ALE action 2 (RIGHT), index 1 → action 3 (LEFT).
# The 6-action minimal set is {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE};
# we never emit the other four (T-P1).
ACTION_MAP = (2, 3)

# Right-paddle columns on the 80×80 preprocessed frame (verified in T-P2).
AGENT_COLS = (68, 74)
# Left (opponent) paddle; ball lives in the interior.
OPP_COLS = (8, 12)


def preprocess(image) -> np.ndarray:
    """Assignment preprocess, copy-safe. 210×160×3 uint8 → 80×80 float {0,1}."""
    image = np.asarray(image)
    image = image.copy()  # views from the slices would otherwise mutate the caller
    image = image[35:195]
    image = image[::2, ::2, 0]
    image[image == 144] = 0
    image[image == 109] = 0
    image[image != 0] = 1
    return np.reshape(image.astype(np.float32), (80, 80))


def frame_diff_state(curr_proc: np.ndarray, prev_proc: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """x_t = pre(frame_t) − pre(frame_{t-1}), stored as int8 in {-1,0,1}.

    At episode start we set prev = curr so the first difference is all zeros
    (the agent has no motion cue on the first action; documented, tested in T-P3).
    """
    if prev_proc is None:
        prev_proc = curr_proc
    diff = (curr_proc - prev_proc).ravel().astype(np.int8)
    return diff, curr_proc


def make_pong_env(deterministic: bool = False, render_mode=None):
    """ALE/Pong-v5 with default settings unless a test asks for sticky-action off."""
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    kwargs = {}
    if deterministic:
        kwargs["repeat_action_probability"] = 0.0
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make("ALE/Pong-v5", **kwargs)
    return env


# ---------------------------------------------------------------------------
# multiprocessing workers (spawn). Never touch CUDA.
# ---------------------------------------------------------------------------
_W = {}


def init_pong_worker(arch: str = "mlp", hidden: int = 200, deterministic: bool = False):
    """Pool initializer: one env per process, CPU-only, single thread."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    from pg.models import make_policy

    env = make_pong_env(deterministic=deterministic)
    policy = make_policy(arch, in_dim=6400, hidden=hidden, n_actions=2)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    _W["env"] = env
    _W["policy"] = policy
    _W["arch"] = arch
    _W["hidden"] = hidden


def play_one_episode(task: dict) -> dict:
    """Play one full episode with the given numpy weights. CPU, no CUDA.

    task = {weights: dict[str, ndarray], seed: int, greedy: bool}
    Returns int8 states, uint8 action *indices*, float32 rewards and logps.
    """
    from pg.models import load_state_dict_numpy

    policy = _W["policy"]
    env = _W["env"]
    load_state_dict_numpy(policy, task["weights"])
    seed = int(task["seed"])
    greedy = bool(task.get("greedy", False))
    gen = torch.Generator()
    gen.manual_seed(seed)

    obs, _ = env.reset(seed=seed)
    prev = None
    states, actions, rewards, logps = [], [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        proc = preprocess(obs)
        state, prev = frame_diff_state(proc, prev)
        x = torch.from_numpy(state.astype(np.float32))
        with torch.no_grad():
            dist = policy.dist(x)
            idx = sample_action(dist, generator=gen, greedy=greedy)
            logp = dist.log_prob(idx)
        env_a = ACTION_MAP[int(idx.item())]
        obs, r, terminated, truncated, _ = env.step(env_a)
        states.append(state)
        actions.append(int(idx.item()))
        rewards.append(float(r))
        logps.append(float(logp.item()))

    return {
        "states": np.stack(states, axis=0).astype(np.int8),
        "actions": np.asarray(actions, dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "logp": np.asarray(logps, dtype=np.float32),
        "seed": seed,
        "reward": float(np.sum(rewards)),
        "length": int(len(rewards)),
        "points_won": int(np.sum(np.asarray(rewards) > 0)),
        "points_lost": int(np.sum(np.asarray(rewards) < 0)),
    }


def paddle_centroid(frame80: np.ndarray, cols=AGENT_COLS) -> Optional[float]:
    region = frame80[:, cols[0]:cols[1]]
    rows = np.where(region.any(axis=1))[0]
    if rows.size == 0:
        return None
    return float(rows.mean())


def ball_position(frame80: np.ndarray) -> Optional[tuple[float, float]]:
    """Rightmost non-paddle blob: (row, col).

    Only the paddle columns are masked. Zeroing everything from column 64
    hides the ball on the approach, which is exactly when the paddle has to move.
    A lone pixel on the far right edge (the ball leaving the court) is ignored
    when another blob is still in play.
    """
    mask = np.array(frame80, copy=True)
    mask[:, 67:76] = 0
    mask[:, 5:13] = 0
    mask[:, 0:2] = 0
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return None
    cmax = int(cols.max())
    if cmax >= 77 and np.any(cols < 67):
        keep = cols < 67
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            return None
        cmax = int(cols.max())
    sel = cols >= cmax - 1
    return float(rows[sel].mean()), float(cols[sel].mean())


def _reflect_row(y: float, lo: float = 0.5, hi: float = 78.5) -> float:
    """Fold a predicted row back into the playfield (top and bottom walls)."""
    span = (hi - lo) * 2.0
    z = (y - lo) % span
    if z < 0:
        z += span
    if z > (hi - lo):
        z = span - z
    return lo + z


def scripted_action(frame80: np.ndarray, state: Optional[dict] = None) -> tuple[int, dict]:
    """Hand-written opponent for T-P9. Returns (env action in {2,3}, state).

    A bang-bang chase of the ball's current row loses every point: one
    frameskipped step moves the paddle ~11 px, so the paddle oscillates
    around the ball and never stays on it. This controller predicts the
    row where the ball will meet the agent paddle (reflecting off the
    walls) and only moves when the error exceeds 5 px. Inside that
    deadband it alternates actions, which cancels and holds position.
    """
    if state is None:
        state = {"prev": None, "vx": 0.0, "vy": 0.0, "flip": 0}
    pos = ball_position(frame80)
    pr = paddle_centroid(frame80)
    prev = state.get("prev")
    if pos is not None and prev is not None:
        vx = pos[1] - prev[1]
        vy = pos[0] - prev[0]
        # A one-frame dropout makes the blob jump; ignore that sample.
        if abs(vx) <= 6 and abs(vy) <= 8:
            state["vx"] = vx
            state["vy"] = vy
    vx = float(state["vx"])
    vy = float(state["vy"])
    target = 40.0 if pr is None else float(pr)
    if pos is not None and vx > 0.5 and pos[1] < 70:
        steps = max((69.0 - pos[1]) / vx, 0.0)
        target = _reflect_row(pos[0] + vy * steps)
    elif vx < -0.5:
        target = 40.0
    if pr is None:
        action = 2
    else:
        err = float(pr) - target  # positive: paddle is below the target
        if err > 5:
            action = 2  # row decreases (T-P2)
        elif err < -5:
            action = 3
        else:
            state["flip"] ^= 1
            action = 2 if state["flip"] else 3
    state["prev"] = pos
    return action, state


def ball_row(frame80: np.ndarray) -> Optional[float]:
    """Row of the ball, or None. Kept for callers that only need the row."""
    pos = ball_position(frame80)
    return None if pos is None else pos[0]
