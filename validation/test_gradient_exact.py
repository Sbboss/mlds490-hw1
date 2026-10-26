"""T-R2 (toy MDP exact gradient) and T-R3 (bandit closed form).

These are the strongest correctness checks: the sample REINFORCE mean must
land inside 4 SE of a quantity we can write down by enumerating every
trajectory. If the estimator is off by a γ^t vs 1 factor, T-R2 catches it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from validation.records import record

# 2-step MDP (PLAN §7.1 T-R2)
# s=0 --a0--> s=1+a0 --a1--> terminate
R0 = np.array([0.0, 1.0], dtype=np.float64)
R1 = {
    1: np.array([0.5, 0.0], dtype=np.float64),
    2: np.array([0.0, 2.0], dtype=np.float64),
}
GAMMA = 0.5
N = 200_000


def _softmax(theta):
    e = np.exp(theta - theta.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _exact_J_and_grad(theta_np: np.ndarray, gamma=GAMMA):
    """Enumerate 4 trajectories; autograd gives the true ∇J."""
    theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
    J = theta.new_zeros(())
    for a0 in (0, 1):
        p0 = Categorical(logits=theta[0]).probs[a0]
        r0 = float(R0[a0])
        s1 = 1 + a0
        for a1 in (0, 1):
            p1 = Categorical(logits=theta[s1]).probs[a1]
            r1 = float(R1[s1][a1])
            J = J + p0 * p1 * (r0 + gamma * r1)
    J.backward()
    return float(J.detach()), theta.grad.detach().numpy().copy()


def _exact_reinforce_expectation(theta_np, w_gamma: bool, gamma=GAMMA):
    """E[ Σ_t w_t ∇logπ(a_t|s_t) G_t ] by enumeration (6 numbers, 3×2)."""
    pi = _softmax(theta_np)
    acc = np.zeros_like(theta_np)
    for a0 in (0, 1):
        p0 = pi[0, a0]
        r0 = R0[a0]
        s1 = 1 + a0
        for a1 in (0, 1):
            p1 = pi[s1, a1]
            r1 = R1[s1][a1]
            G0 = r0 + gamma * r1
            G1 = r1
            w0 = 1.0
            w1 = gamma if w_gamma else 1.0
            # ∇_θ logπ(a|s) = e_a − π(s)
            g0 = np.zeros_like(theta_np)
            g0[0] = -pi[0]
            g0[0, a0] += 1.0
            g1 = np.zeros_like(theta_np)
            g1[s1] = -pi[s1]
            g1[s1, a1] += 1.0
            acc += p0 * p1 * (w0 * g0 * G0 + w1 * g1 * G1)
    return acc


def _sample_reinforce(theta_np, n, w_gamma: bool, rng, gamma=GAMMA):
    """Vectorized N trajectories; returns (mean, se) of the 3×2 estimator."""
    pi = _softmax(theta_np)
    a0 = rng.choice(2, size=n, p=pi[0])
    s1 = 1 + a0
    # sample a1 | s1
    u = rng.random(n)
    a1 = np.where(u < pi[s1, 0], 0, 1)
    r0 = R0[a0]
    r1 = np.where(s1 == 1, R1[1][a1], R1[2][a1])
    G0 = r0 + gamma * r1
    G1 = r1
    w0 = np.ones(n)
    w1 = np.full(n, gamma if w_gamma else 1.0)

    grads = np.zeros((n,) + theta_np.shape)
    # ∇logπ(a0|s=0) * w0 * G0
    onehot0 = np.zeros((n, 2))
    onehot0[np.arange(n), a0] = 1.0
    grads[:, 0, :] = (onehot0 - pi[0]) * (w0 * G0)[:, None]
    # ∇logπ(a1|s1) * w1 * G1
    onehot1 = np.zeros((n, 2))
    onehot1[np.arange(n), a1] = 1.0
    grads[np.arange(n), s1, :] += (onehot1 - pi[s1]) * (w1 * G1)[:, None]
    mean = grads.mean(axis=0)
    se = grads.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, se


def test_tr2():
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 0.5, size=(3, 2))
    J, exact_grad = _exact_J_and_grad(theta)
    # weighted estimator must match the true gradient
    mean_w, se_w = _sample_reinforce(theta, N, w_gamma=True, rng=np.random.default_rng(1))
    z_w = np.abs(mean_w - exact_grad) / np.maximum(se_w, 1e-12)
    ok_w = bool(np.all(z_w < 4.0))
    record("T-R2_weighted_matches_true_grad", "exact", ok_w,
           f"max |z|={z_w.max():.2f}; mean={np.round(mean_w,4).tolist()}; exact={np.round(exact_grad,4).tolist()}",
           "every component of the N=200k mean within 4 SE of ∇J",
           "w_t=γ^t is the exact gradient of J=E[r0+γ r1].")

    # unweighted: matches ITS own expectation, and differs from ∇J
    exp_u = _exact_reinforce_expectation(theta, w_gamma=False)
    mean_u, se_u = _sample_reinforce(theta, N, w_gamma=False, rng=np.random.default_rng(2))
    z_u = np.abs(mean_u - exp_u) / np.maximum(se_u, 1e-12)
    ok_u = bool(np.all(z_u < 4.0))
    record("T-R2_unweighted_matches_its_expectation", "exact", ok_u,
           f"max |z|={z_u.max():.2f}", "within 4 SE of enumerated E[unweighted]",
           "The w_t=1 estimator is unbiased for a different (state-distribution) objective.")

    z_diff = np.abs(exp_u - exact_grad) / np.maximum(se_u, 1e-12)
    ok_diff = bool(np.any(z_diff > 10.0))
    record("T-R2_unweighted_differs_from_true_grad", "exact", ok_diff,
           f"max |E[unweighted]−∇J| / SE = {z_diff.max():.1f}; "
           f"E_u={np.round(exp_u,4).tolist()} ∇J={np.round(exact_grad,4).tolist()}",
           "differs from true ∇J by >10 SE in at least one component",
           "With γ=0.5 the unweighted step-2 term is off by a factor of 2 (1/γ).")

    record("T-R2_J_value", "exact", True, f"J(θ)={J:.6f}", "enumerated J (logged)",
           "Reference value of the toy objective at the fixed θ (seed 0, std 0.5).")


def test_tr3_bandit():
    """Softmax bandit, θ=0, r=(1,0): E[∇] = (+0.25, −0.25)."""
    n = 200_000
    rng = np.random.default_rng(0)
    theta = np.zeros(2)
    pi = _softmax(theta[None, :])[0]  # (0.5, 0.5)
    r = np.array([1.0, 0.0])
    a = rng.choice(2, size=n, p=pi)
    # ∇logπ(a) * r_a
    grads = np.zeros((n, 2))
    onehot = np.zeros((n, 2))
    onehot[np.arange(n), a] = 1.0
    grads = (onehot - pi) * r[a][:, None]
    mean = grads.mean(axis=0)
    se = grads.std(axis=0, ddof=1) / np.sqrt(n)
    exp = np.array([0.25, -0.25])
    z = np.abs(mean - exp) / se
    ok = bool(np.all(z < 4.0))
    record("T-R3_bandit_closed_form", "exact", ok,
           f"mean={mean.tolist()} se={se.tolist()} z={z.tolist()}",
           "(+0.25, -0.25) within 4 SE, N=200k",
           "Closed-form softmax bandit; catches a missing (e_a−π) or a flipped sign.")


def run_gradient_tests():
    test_tr2()
    test_tr3_bandit()
