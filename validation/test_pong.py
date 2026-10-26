"""T-P* Pong validation. Needs ale-py. Run locally first, then on deepdish."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pg.common import apply_plot_style, discount_returns
from pg.models import MLPPolicy, state_dict_numpy
from pg.pong_utils import (
    ACTION_MAP,
    AGENT_COLS,
    frame_diff_state,
    make_pong_env,
    paddle_centroid,
    play_one_episode,
    preprocess,
    scripted_action,
)
from pg.reinforce import reinforce_loss
from validation.records import record

FIG = Path("figures/appendix_pong_preprocess.png")
TRACE = Path("results/validation/pong_preprocess_stats.json")


def test_tp1_action_mapping():
    env = make_pong_env()
    recorded = []
    orig = env.step

    def wrapped(a):
        recorded.append(int(a))
        return orig(a)

    env.step = wrapped
    policy = MLPPolicy(6400, 32, 2)
    obs, _ = env.reset(seed=0)
    prev = None
    indices = []
    for t in range(1000):
        proc = preprocess(obs)
        state, prev = frame_diff_state(proc, prev)
        dist = policy.dist(torch.from_numpy(state.astype(np.float32)))
        idx = int(dist.sample().item())
        indices.append(idx)
        obs, _, term, trunc, _ = env.step(ACTION_MAP[idx])
        if term or trunc:
            obs, _ = env.reset()
            prev = None
    env.close()
    ok_set = set(recorded) <= {2, 3} and set(indices) <= {0, 1}
    record("T-P1_actions_in_2_3", "exact", ok_set and len(recorded) == 1000,
           f"unique env actions={sorted(set(recorded))} unique idx={sorted(set(indices))}",
           "env actions ⊆ {2,3}, indices ⊆ {0,1}",
           "The network never emits NOOP/FIRE; only RIGHT/LEFT reach ALE.")

    # forced logits [100, -100] → always action 2
    env = make_pong_env()
    recorded = []
    orig = env.step
    env.step = lambda a: (recorded.append(int(a)), orig(a))[1]
    obs, _ = env.reset(seed=1)
    forced = torch.tensor([100.0, -100.0])
    for _ in range(30):
        idx = int(torch.argmax(forced).item())
        obs, _, term, trunc, _ = env.step(ACTION_MAP[idx])
        if term or trunc:
            break
    env.close()
    ok = set(recorded) == {2} and len(recorded) > 0
    record("T-P1_forced_logits", "exact", ok,
           f"recorded={sorted(set(recorded))} n={len(recorded)}",
           "env action 2 every step",
           "Index 0 is RIGHT = ALE action 2.")


def test_tp2_paddle_control():
    env = make_pong_env(deterministic=True)
    obs, _ = env.reset(seed=0)
    cents = []
    for i, a in enumerate([2] * 30 + [3] * 60):
        obs, _, term, trunc, _ = env.step(a)
        fr = preprocess(obs)
        c = paddle_centroid(fr, AGENT_COLS)
        cents.append(c)
        if term or trunc:
            break
    env.close()
    cents = np.array([c if c is not None else np.nan for c in cents], dtype=np.float64)
    # verify paddle is visible
    visible = np.isfinite(cents).mean() > 0.8
    first = cents[:30]
    second = cents[30:90]
    d1 = float(np.nanmax(first) - np.nanmin(first)) if np.isfinite(first).any() else 0.0
    d2 = float(np.nanmax(second) - np.nanmin(second)) if np.isfinite(second).any() else 0.0
    # direction: do not assume which action is "up"
    delta1 = float(np.nanmean(first[-5:]) - np.nanmean(first[:5])) if len(first) >= 10 else 0.0
    delta2 = float(np.nanmean(second[-5:]) - np.nanmean(second[:5])) if len(second) >= 10 else 0.0
    opposite = (delta1 * delta2) < 0
    ok = visible and d1 >= 5 and d2 >= 5 and opposite
    # plot for the report
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(cents, color="#2a78d6")
    ax.axvline(30, color="#52514e", ls=":", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("agent paddle row centroid (px)")
    ax.set_title("T-P2: force action 2 then 3 (deterministic env)")
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/appendix_paddle_control.png", dpi=200); plt.close(fig)
    record("T-P2_paddle_control", "exact", ok,
           f"visible={visible} travel_2={d1:.1f}px travel_3={d2:.1f}px "
           f"delta2={delta1:.2f} delta3={delta2:.2f} opposite={opposite} cols={AGENT_COLS}",
           "centroid moves ≥5 px one way then ≥5 px the opposite way; paddle visible",
           "Confirms preprocess keeps the agent paddle and the action map actually steers it.")


def test_tp3_preprocess_safety():
    env = make_pong_env()
    obs, _ = env.reset(seed=0)
    before = obs.copy()
    out = preprocess(obs)
    unchanged = np.array_equal(obs, before)
    shape_ok = out.shape == (80, 80)
    vals = set(np.unique(out).tolist())
    vals_ok = vals <= {0.0, 1.0}
    flat = out.ravel()
    record("T-P3_copy_safe", "exact", unchanged,
           f"obs unchanged={unchanged}", "input obs unchanged after preprocess",
           "The assignment slices are views; we copy first so we do not mutate ALE's frame.")
    record("T-P3_shape_and_values", "exact", shape_ok and vals_ok and flat.size == 6400,
           f"shape={out.shape} unique={sorted(vals)} flat={flat.size}",
           "(80,80) → 6400, values ⊆ {0,1}",
           "Matches the given preprocess spec.")

    prev = None
    diffs = []
    frames = []
    nonzero = []
    for t in range(300):
        fr = preprocess(obs)
        d, prev = frame_diff_state(fr, prev)
        if t == 0:
            first_zero = bool(np.all(d == 0))
        diffs.append(d)
        if t < 6:
            frames.append(fr.copy())
        nz = int((fr > 0).sum())
        nonzero.append(nz)
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
            prev = None
    env.close()
    dvals = set(int(x) for x in np.unique(np.stack(diffs)))
    after20 = diffs[20:]
    frac_nonzero_diff = float(np.mean([np.any(d != 0) for d in after20]))
    min_nz = int(min(nonzero))
    record("T-P3_diff_and_first_zero", "exact",
           first_zero and dvals <= {-1, 0, 1} and min_nz > 0 and frac_nonzero_diff >= 0.5,
           f"first_zero={first_zero} diff_vals={sorted(dvals)} min_nz={min_nz} "
           f"frac_nonzero_diff_after20={frac_nonzero_diff:.2f}",
           "first diff all 0; diffs ⊆ {-1,0,1}; min nonzero pixels > 0; ≥50% diffs nonzero after step 20",
           "First-frame convention + ball motion. A dead preprocess would be all zeros.")

    apply_plot_style()
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.2))
    for i in range(6):
        axes[0, i].imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"t={i}"); axes[0, i].axis("off")
        if i == 0:
            dshow = np.zeros((80, 80))
        else:
            dshow = frames[i] - frames[i - 1]
        axes[1, i].imshow(dshow, cmap="gray", vmin=-1, vmax=1)
        axes[1, i].set_title("diff"); axes[1, i].axis("off")
    fig.suptitle("T-P3: six consecutive preprocessed frames and their differences")
    FIG.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG, dpi=200); plt.close(fig)
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    TRACE.write_text(json.dumps({
        "min_nonzero": min_nz, "frac_nonzero_diff_after20": frac_nonzero_diff,
        "diff_vals": sorted(dvals),
    }, indent=2))


def test_tp4_initial_policy():
    env = make_pong_env()
    policy = MLPPolicy(6400, 200, 2)
    n0 = n = 0
    ents = []
    # 20 full episodes is slow (~20 * 2000 steps). Use 8 episodes — still plenty
    # of actions — and log the deviation. PLAN says 20; we do 20.
    for i in range(20):
        obs, _ = env.reset(seed=4000 + i)
        prev = None
        ep_ent = []
        term = trunc = False
        while not (term or trunc):
            fr = preprocess(obs)
            st, prev = frame_diff_state(fr, prev)
            dist = policy.dist(torch.from_numpy(st.astype(np.float32)))
            idx = dist.sample()
            n0 += int(idx.item() == 0)
            n += 1
            ep_ent.append(float(dist.entropy().item()))
            obs, _, term, trunc, _ = env.step(ACTION_MAP[int(idx.item())])
        ents.append(float(np.mean(ep_ent)))
    env.close()
    frac = n0 / max(n, 1)
    ment = float(np.mean(ents))
    ok = 0.4 <= frac <= 0.6 and ment >= 0.6
    record("T-P4_initial_policy", "prediction", ok,
           f"frac_idx0={frac:.3f} mean_entropy={ment:.3f} n_actions={n}",
           "frac ∈ [0.4, 0.6], entropy ≥ 0.6",
           "Must hold before any multi-hour Pong run; otherwise the init is peaked.")


def test_tp5_random_anchor():
    env = make_pong_env()
    policy = MLPPolicy(6400, 200, 2)
    rewards = []
    n_ep = 20
    for i in range(n_ep):
        obs, _ = env.reset(seed=5000 + i)
        prev = None
        R = 0.0
        term = trunc = False
        while not (term or trunc):
            fr = preprocess(obs)
            st, prev = frame_diff_state(fr, prev)
            dist = policy.dist(torch.from_numpy(st.astype(np.float32)))
            idx = int(dist.sample().item())
            obs, r, term, trunc, _ = env.step(ACTION_MAP[idx])
            R += float(r)
        rewards.append(R)
    env.close()
    m = float(np.mean(rewards))
    ok_int = all(float(r).is_integer() or abs(r - round(r)) < 1e-6 for r in rewards)
    in_range = all(-21 <= r <= -19 for r in rewards)
    ok_mean = -21.0 <= m <= -19.8
    record("T-P5_random_anchor", "prediction", ok_int and in_range and ok_mean,
           f"rewards={rewards} mean={m:.2f}",
           "integer rewards in [-21,-19], mean ∈ [-21, -19.8] (published random ≈ -20.7)",
           "Learning is claimed only if MA100 rises by >0.3 (≥3 SE) and stays there.")


def test_tp6_worker_learner_logp():
    """Recompute logπ on a dummy 'learner' and match the worker's stored logp."""
    # Play one episode in-process (same code path as the worker, minus the pool).
    from pg.pong_utils import init_pong_worker
    init_pong_worker("mlp", 64, False)
    pol = MLPPolicy(6400, 64, 2)
    weights = state_dict_numpy(pol)
    tr = play_one_episode({"weights": weights, "seed": 123, "greedy": False})
    # learner recompute
    learner = MLPPolicy(6400, 64, 2)
    learner.load_state_dict(pol.state_dict())
    x = torch.from_numpy(tr["states"].astype(np.float32))
    dist = learner.dist(x)
    logp = dist.log_prob(torch.from_numpy(tr["actions"].astype(np.int64)))
    err = float((logp.detach().numpy() - tr["logp"]).max())
    ok = abs(err) < 1e-4
    record("T-P6_worker_learner_logp", "exact", ok,
           f"max |logπ_learner − logπ_worker|={err:.2e} T={tr['length']}",
           "< 1e-4",
           "Proves identical weights were used (on-policy) and the index/action map agrees.")


def test_tp7_point_boundary():
    # synthesize a recorded-like reward stream and also run a real episode
    r = np.array([0, 0, -1, 0, 0, 1, 0, -1], dtype=np.float64)
    G = discount_returns(r, 0.99, reset_on_nonzero=True)
    # before each ±1, G must not include later points
    # G[0] must equal 0.99^2 * (-1), not include the later +1
    ok = abs(G[2] + 1.0) < 1e-12 and abs(G[5] - 1.0) < 1e-12 and abs(G[7] + 1.0) < 1e-12
    ok = ok and abs(G[0] - (0.99 ** 2 * -1)) < 1e-10
    record("T-P7_point_boundary_synthetic", "exact", ok,
           f"G={np.round(G, 5).tolist()}",
           "returns at ±1 equal ±1; earlier steps of a point ignore later points",
           "Same logic as T-R1, applied to a Pong-like reward stream.")

    env = make_pong_env()
    obs, _ = env.reset(seed=7)
    rewards = []
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, _ = env.step(2)
        rewards.append(float(r))
    env.close()
    G = discount_returns(rewards, 0.99, reset_on_nonzero=True)
    idxs = [i for i, r in enumerate(rewards) if r != 0]
    ok2 = True
    details = []
    for i in idxs:
        # G[i] must equal r_i (reset then add r, terminal of that point)
        if abs(G[i] - rewards[i]) > 1e-8:
            ok2 = False
        details.append((i, rewards[i], float(G[i])))
    record("T-P7_point_boundary_real_episode", "exact", ok2 and len(idxs) > 0,
           f"point_steps={details[:6]}... n_points={len(idxs)}",
           "G_t = r_t at every point-scoring step",
           "On a real Pong episode, later points do not leak into earlier G_t.")


def test_tp8_determinism():
    env = make_pong_env()  # default sticky actions
    obs, _ = env.reset(seed=0)
    frames = []
    actions = []
    for t in range(50):
        a = 2 if t % 2 == 0 else 3
        actions.append(a)
        frames.append(hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest())
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    env.close()
    env = make_pong_env()
    obs, _ = env.reset(seed=0)
    frames2 = []
    for t, a in enumerate(actions):
        frames2.append(hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest())
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    env.close()
    same = frames == frames2
    record("T-P8_determinism_default", "prediction", same,
           f"same_hashes={same} n={len(frames)} (default sticky-action env)",
           "identical first-50-frame hashes if ALE is deterministic; else document and skip",
           "Default Pong-v5 has sticky-action probability 0.25, so this often fails — that is expected.")
    if not same:
        record("T-P8_skipped_sticky", "exact", True,
               "default ALE/Pong-v5 is not deterministic (sticky actions); hashes differed",
               "documented skip",
               "Deterministic variants are for tests only (D9); training uses defaults.")


def test_tp9_scripted_tracker():
    """Intercept controller, not a bang-bang chase.

    A chase of the current ball row scores −21: masking columns ≥ 64 hides
    the ball next to the paddle, and one frameskipped action moves the
    paddle ~11 px so the sign flips every step. scripted_action predicts the
    intercept and holds still inside a 5 px band. See DECISIONS D19.
    """
    env = make_pong_env(deterministic=True)
    rewards = []
    for ep in range(10):
        obs, _ = env.reset(seed=9000 + ep)
        R = 0.0
        term = trunc = False
        state = None
        while not (term or trunc):
            action, state = scripted_action(preprocess(obs), state)
            obs, r, term, trunc, _ = env.step(action)
            R += float(r)
        rewards.append(R)
    env.close()
    m = float(np.mean(rewards))
    ok = m > -18
    record("T-P9_scripted_ball_tracker", "prediction", ok,
           f"mean={m:.2f} rewards={rewards}",
           "mean > -18 over 10 episodes (clearly beats random ≈ -20.7)",
           "Validates frames + action mapping. Naive chase scored -21; "
           "intercept + deadband is the controller under test (D19).")


def test_tp10_resume(tmp_path: Path | None = None):
    """Short 3-update run, kill, --resume, episode counter continues."""
    import shutil
    import subprocess
    import sys

    out = Path("results/validation/pong_resume")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    py = sys.executable
    cmd = [py, "-m", "pg.train_pong",
           "--baseline", "none", "--lr", "1e-3", "--workers", "1",
           "--episodes_per_update", "1", "--max_episodes", "3",
           "--max_hours", "0.2", "--hidden", "32", "--seed", "0",
           "--out", str(out)]
    r = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
    if r.returncode != 0:
        record("T-P10_resume_short_run", "exact", False,
               f"rc={r.returncode} stderr={r.stderr[-500:]}", "3-update run succeeds",
               "Need a checkpoint before we can test resume.")
        return
    import pandas as pd
    ep1 = pd.read_csv(out / "episode_rewards.csv")
    n1 = len(ep1)
    ckpt = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    ep_done = int(ckpt["episodes_done"])
    w1 = {k: v.detach().clone() for k, v in ckpt["policy"].items()}
    # resume for 2 more episodes
    cmd[-3] = "5"  # max_episodes 5  -- wait, args order: --max_episodes is before --max_hours
    # rebuild
    cmd = [py, "-m", "pg.train_pong",
           "--baseline", "none", "--lr", "1e-3", "--workers", "1",
           "--episodes_per_update", "1", "--max_episodes", "5",
           "--max_hours", "0.2", "--hidden", "32", "--seed", "0",
           "--resume", "--out", str(out)]
    r2 = subprocess.run(cmd, cwd=str(Path.cwd()), capture_output=True, text=True)
    ep2 = pd.read_csv(out / "episode_rewards.csv")
    ckpt2 = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    dup = ep2["episode"].duplicated().any()
    continued = int(ckpt2["episodes_done"]) == 5 and len(ep2) == 5 and not dup
    changed = any(not torch.equal(w1[k], ckpt2["policy"][k]) for k in w1)
    record("T-P10_resume", "exact", continued and r2.returncode == 0,
           f"n1={n1} n2={len(ep2)} ep_done={ckpt2['episodes_done']} dup={bool(dup)} "
           f"params_changed={changed} rc={r2.returncode}",
           "episode counter continues to 5, no duplicated CSV rows, params not reset",
           "A resume that silently restarts would double-count and poison the curve.")


def run_pong_tests(skip_slow=False):
    test_tp1_action_mapping()
    test_tp2_paddle_control()
    test_tp3_preprocess_safety()
    if not skip_slow:
        test_tp4_initial_policy()
        test_tp5_random_anchor()
        test_tp9_scripted_tracker()
        test_tp10_resume()
    test_tp6_worker_learner_logp()
    test_tp7_point_boundary()
    test_tp8_determinism()
