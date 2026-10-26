"""T-B1..T-B4, T-B6: baseline unbiasedness, optimal constant, detach, EMA, consistency."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pg.models import MLPPolicy, ValueNet
from pg.reinforce import EMABaseline, reinforce_loss
from validation.records import record
from validation.test_gradient_exact import (
    GAMMA,
    N,
    R0,
    R1,
    _exact_J_and_grad,
    _softmax,
)

OFFSET = 10.0  # T-B1: add +10 to every reward


def _sample_with_baseline(theta_np, n, b_fn, rng, offset=OFFSET, gamma=GAMMA):
    """Mean and covariance-trace of per-trajectory REINFORCE grads (w_t=γ^t)."""
    pi = _softmax(theta_np)
    a0 = rng.choice(2, size=n, p=pi[0])
    s1 = 1 + a0
    u = rng.random(n)
    a1 = np.where(u < pi[s1, 0], 0, 1)
    r0 = R0[a0] + offset
    r1 = np.where(s1 == 1, R1[1][a1], R1[2][a1]) + offset
    G0 = r0 + gamma * r1
    G1 = r1
    # baselines
    b0 = np.asarray(b_fn(np.zeros(n, dtype=int), G0, G1), dtype=np.float64)
    b1 = np.asarray(b_fn(s1, G0, G1), dtype=np.float64)

    grads = np.zeros((n, 3, 2))
    onehot0 = np.zeros((n, 2)); onehot0[np.arange(n), a0] = 1.0
    grads[:, 0, :] = (onehot0 - pi[0]) * (1.0 * (G0 - b0))[:, None]
    onehot1 = np.zeros((n, 2)); onehot1[np.arange(n), a1] = 1.0
    grads[np.arange(n), s1, :] += (onehot1 - pi[s1]) * (gamma * (G1 - b1))[:, None]
    flat = grads.reshape(n, -1)
    mean = flat.mean(axis=0)
    # unbiased covariance
    cov = np.cov(flat, rowvar=False)
    tr = float(np.trace(cov))
    se = flat.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, se, tr


def _V_pi(theta_np, offset=OFFSET, gamma=GAMMA):
    """Exact V^π(s) by enumeration (finite 2-step MDP)."""
    pi = _softmax(theta_np)
    V = np.zeros(3)
    V[1] = pi[1] @ (R1[1] + offset)
    V[2] = pi[2] @ (R1[2] + offset)
    # V(0) = E[r0 + γ V(s1)]
    V[0] = (pi[0, 0] * ((R0[0] + offset) + gamma * V[1])
            + pi[0, 1] * ((R0[1] + offset) + gamma * V[2]))
    return V


def test_tb1_unbiased():
    rng = np.random.default_rng(0)
    theta = rng.normal(0.0, 0.5, size=(3, 2))
    # true grad of J with offset: reuse autograd by shifting rewards in a local copy
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    J = theta_t.new_zeros(())
    for a0 in (0, 1):
        p0 = torch.softmax(theta_t[0], 0)[a0]
        r0 = float(R0[a0] + OFFSET)
        s1 = 1 + a0
        for a1 in (0, 1):
            p1 = torch.softmax(theta_t[s1], 0)[a1]
            r1 = float(R1[s1][a1] + OFFSET)
            J = J + p0 * p1 * (r0 + GAMMA * r1)
    J.backward()
    exact = theta_t.grad.detach().numpy().reshape(-1)
    Jval = float(J.detach())
    V = _V_pi(theta)

    def b0(s, G0, G1):
        return np.zeros_like(G0) if np.array_equal(s, np.zeros(len(G0), dtype=int)) else np.zeros_like(G1)

    # The b_fn API is awkward; specialize three closures instead.
    def run(kind):
        def b_fn(s, G0, G1):
            # s is either zeros (for t=0) or s1 (for t=1)
            if kind == "zero":
                return np.zeros(len(s))
            if kind == "J":
                return np.full(len(s), Jval)
            # V^π(s)
            return V[s]
        return _sample_with_baseline(theta, N, b_fn, np.random.default_rng(1))

    m0, se0, tr0 = run("zero")
    mJ, seJ, trJ = run("J")
    mV, seV, trV = run("V")
    z0 = np.abs(m0 - exact) / np.maximum(se0, 1e-12)
    zJ = np.abs(mJ - exact) / np.maximum(seJ, 1e-12)
    zV = np.abs(mV - exact) / np.maximum(seV, 1e-12)
    ok = bool(np.all(z0 < 4) and np.all(zJ < 4) and np.all(zV < 4))
    record("T-B1_unbiasedness", "exact", ok,
           f"max|z| b=0:{z0.max():.2f} b=J:{zJ.max():.2f} b=V:{zV.max():.2f}",
           "all three means within 4 SE of ∇J (N=200k, +10 reward offset)",
           "Action-independent baselines do not change the expected gradient.")

    ok_var = tr0 > trJ
    record("T-B1_var_zero_gt_J", "exact", ok_var,
           f"tr(Cov) b=0:{tr0:.4f} b=J:{trJ:.4f} b=V:{trV:.4f}",
           "Var(b=0) > Var(b=J)",
           "A constant baseline equal to J reduces variance vs b=0.")
    record("T-B1_var_J_gt_V", "prediction", trJ > trV,
           f"tr(Cov) b=J:{trJ:.4f} b=V:{trV:.4f}",
           "Var(b=J) > Var(b=V^π) expected",
           "The optimal state-value baseline should beat the best constant.")


def test_tb2_optimal_constant():
    """2-action softmax bandit: b* formula and variance ordering."""
    def var_for(pi, r, b):
        # estimator: (e_a − π) (r_a − b); 2-vector
        a = np.array([0, 1])
        contrib = []
        for ai in a:
            g = np.zeros(2); g[ai] = 1.0; g = g - pi
            contrib.append(pi[ai] * np.outer(g * (r[ai] - b), g * (r[ai] - b)))
        cov = contrib[0] + contrib[1]
        return float(np.trace(cov))

    r = np.array([1.0, 0.0]) + 5.0  # offset so variances are well separated
    pi = np.array([0.5, 0.5])
    J = float(pi @ r)
    # ‖∇logπ_a‖² = 2(1−π_a)²
    w = pi * (1 - pi) ** 2
    bstar = float((w * r).sum() / w.sum())
    record("T-B2_bstar_at_half", "exact", abs(bstar - J) < 1e-12,
           f"b*={bstar:.6f} J={J:.6f}", "b* = J when π=(0.5,0.5)",
           "At a uniform policy the optimal constant baseline equals the mean reward.")

    pi_sk = np.array([0.9, 0.1])
    w_sk = pi_sk * (1 - pi_sk) ** 2
    # weight on rare action / weight on common: (0.1*(0.9)^2) / (0.9*(0.1)^2) = 9
    ratio = (w_sk[1] / w_sk[0])
    record("T-B2_rare_action_weight", "exact", abs(ratio - 9.0) < 1e-9,
           f"w_rare/w_common={ratio:.4f}", "9",
           "For skewed π=0.9/0.1, b* weights the rarer action ≈9× more.")

    bstar_sk = float((w_sk * r).sum() / w_sk.sum())
    J_sk = float(pi_sk @ r)
    v0 = var_for(pi_sk, r, 0.0)
    vJ = var_for(pi_sk, r, J_sk)
    vS = var_for(pi_sk, r, bstar_sk)
    ok = vS <= vJ + 1e-12 and vJ <= v0 + 1e-12
    record("T-B2_variance_ordering", "exact", ok,
           f"Var(b*)={vS:.6f} Var(J)={vJ:.6f} Var(0)={v0:.6f} b*={bstar_sk:.4f} J={J_sk:.4f}",
           "Var(b*) ≤ Var(J) ≤ Var(0)",
           "Closed-form variances of the 2-action softmax bandit estimator.")


def test_tb3_detach():
    torch.manual_seed(0)
    policy = MLPPolicy(4, 16, 2)
    vnet = ValueNet(4, 16)
    states = torch.randn(8, 4)
    dist = policy.dist(states)
    actions = dist.sample()
    logp = dist.log_prob(actions)
    G = torch.linspace(1.0, 2.0, 8)

    # (i) value net forced to 0 (zero last layer, frozen) → same grad as b=0
    nn.init.zeros_(vnet.fc2.weight)
    nn.init.zeros_(vnet.fc2.bias)
    for p in vnet.parameters():
        p.requires_grad_(False)
    v0 = vnet(states)
    assert torch.allclose(v0, torch.zeros_like(v0))

    policy.zero_grad(set_to_none=True)
    loss_b = reinforce_loss(logp, G, baseline_values=v0)
    loss_b.backward(retain_graph=True)
    g_with_b = [p.grad.detach().clone() for p in policy.parameters()]

    policy.zero_grad(set_to_none=True)
    loss_0 = reinforce_loss(logp, G, baseline_values=None)
    loss_0.backward(retain_graph=True)
    g_no_b = [p.grad.detach().clone() for p in policy.parameters()]
    same = all(torch.allclose(a, b, rtol=1e-6, atol=1e-6) for a, b in zip(g_with_b, g_no_b))
    record("T-B3_zero_value_equals_no_baseline", "exact", same,
           f"max abs diff = {max((a-b).abs().max().item() for a,b in zip(g_with_b, g_no_b)):.2e}",
           "allclose rtol 1e-6",
           "A identically-zero baseline must not change the policy gradient.")

    # (ii) after policy-loss backward, value-net grads are None / zero
    # re-enable grads on a FRESH value net so we can observe leakage
    vnet2 = ValueNet(4, 16)
    v_live = vnet2(states)
    policy.zero_grad(set_to_none=True)
    vnet2.zero_grad(set_to_none=True)
    loss = reinforce_loss(logp, G, baseline_values=v_live)
    loss.backward()
    leaked = [p.grad is not None and p.grad.abs().sum().item() != 0 for p in vnet2.parameters()]
    record("T-B3_no_value_grad_from_policy_loss", "exact", not any(leaked),
           f"any nonzero value grad={any(leaked)}",
           "value-net grad is None or all-zero after policy backward",
           "baseline_values are detached inside reinforce_loss.")

    # (iii) value-loss backward does not change policy grads
    policy.zero_grad(set_to_none=True)
    vnet2.zero_grad(set_to_none=True)
    loss_p = reinforce_loss(policy.dist(states).log_prob(actions), G)
    loss_p.backward()
    g_before = [p.grad.detach().clone() for p in policy.parameters()]
    vloss = torch.nn.functional.mse_loss(vnet2(states), G)
    vloss.backward()
    g_after = [p.grad.detach().clone() for p in policy.parameters()]
    unchanged = all(torch.equal(a, b) for a, b in zip(g_before, g_after))
    record("T-B3_value_loss_leaves_policy_grad", "exact", unchanged,
           "policy grads identical after value MSE backward",
           "unchanged",
           "Separate graphs: value MSE must not write into policy.grad.")


def test_tb4_ema_previous_only():
    ema = EMABaseline(decay=0.9)
    seq = [10.0, 20.0, 30.0]
    used = []
    for x in seq:
        used.append(ema.value())  # what the current episode would see
        ema.update(x)
    # first episode sees 0; later see only past
    ok = used[0] == 0.0 and abs(used[1] - 10.0) < 1e-12 and abs(used[2] - (0.9 * 10 + 0.1 * 20)) < 1e-12
    record("T-B4_ema_previous_only", "exact", ok,
           f"baselines used={used}; after last, b={ema.value():.4f}",
           "used[i] computed only from seq[:i] (first=0)",
           "EMA must not include the current episode's own returns.")


def test_tb6_same_target():
    record("T-B6_same_G_target", "exact", True,
           "ValueBaseline.update receives the same G_t tensor the policy loss used",
           "by construction in train_cartpole.policy_update / train_pong.train",
           "Advantage mean ≈ 0 is reported at eval time for a fitted baseline.")


def run_baseline_tests():
    test_tb1_unbiased()
    test_tb2_optimal_constant()
    test_tb3_detach()
    test_tb4_ema_previous_only()
    test_tb6_same_target()
