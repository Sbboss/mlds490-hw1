"""T-R1, T-R4, T-R5: returns, Adam first step, NaN guard."""
from __future__ import annotations

import numpy as np
import torch

from pg.common import discount_returns, standardize_returns
from pg.reinforce import reinforce_loss
from validation.records import record


def test_tr1_returns():
    g = discount_returns([1, 1, 1], 0.5)
    ok = np.allclose(g, [1.75, 1.5, 1.0])
    record("T-R1_half", "exact", ok, np.round(g, 6).tolist(), "[1.75, 1.5, 1.0]",
           "Backward loop G=r+γG. A forward-order bug would give [1, 1.5, 1.75].")

    g = discount_returns([1, 1, 1], 0.95)
    exp = np.array([1 + 0.95 * 1.95, 1.95, 1.0])  # 2.8525, 1.95, 1.0
    ok = np.allclose(g, exp)
    record("T-R1_cartpole_gamma", "exact", ok, np.round(g, 6).tolist(), "[2.8525, 1.95, 1.0]",
           "CartPole γ=0.95 hand-computed return-to-go on [1,1,1].")

    r = [0, 0, -1, 0, 0, 1]
    g = discount_returns(r, 0.99, reset_on_nonzero=True)
    exp = np.array([-0.9801, -0.99, -1.0, 0.9801, 0.99, 1.0])
    ok = np.allclose(g, exp, atol=1e-10)
    record("T-R1_pong_reset", "exact", ok, np.round(g, 6).tolist(),
           "[-0.9801, -0.99, -1, 0.9801, 0.99, 1]",
           "Point-boundary reset: later +1 must not leak into the earlier −1.")

    g2 = discount_returns(r, 0.99, reset_on_nonzero=False)
    ok = abs(g2[2] - (-0.029701)) < 1e-6
    record("T-R1_pong_noleak_contrast", "exact", ok, f"G[2]={g2[2]:.6f}", "≈ -0.0297",
           "Without reset, the −1 point still contains later points (the leak D2 avoids).")


def test_tr4_adam_first_step():
    """Bias-corrected Adam: first step of a param with nonzero grad has |Δ| = lr."""
    lr = 0.01
    p = torch.nn.Parameter(torch.tensor([0.0, 1.0, -2.0]))
    opt = torch.optim.Adam([p], lr=lr)
    # loss = 3*p[0] + 0*p[1] + (-4)*p[2]  → grad = (3, 0, -4)
    loss = 3.0 * p[0] + 0.0 * p[1] - 4.0 * p[2]
    opt.zero_grad()
    loss.backward()
    before = p.detach().clone()
    opt.step()
    delta = (p.detach() - before).numpy()
    # nonzero-grad coords move by exactly ±lr; zero-grad coord stays put
    ok = (abs(abs(delta[0]) - lr) < 1e-6 and abs(delta[1]) < 1e-8
          and abs(abs(delta[2]) - lr) < 1e-6
          and np.sign(delta[0]) == -1 and np.sign(delta[2]) == 1)
    record("T-R4_adam_first_step", "exact", ok,
           f"delta={delta.tolist()}", f"±{lr} on nonzero grads, 0 on zero grad",
           "Explains why Adam hides gradient-scale differences between baseline arms.")


def test_tr5_nan_guard():
    def _run(rewards):
        logp = torch.zeros(len(rewards), requires_grad=True)
        G = torch.as_tensor(standardize_returns(np.asarray(rewards, dtype=np.float64)),
                            dtype=torch.float32)
        loss = reinforce_loss(logp, G)
        loss.backward()
        return float(loss.detach()), logp.grad

    l1, g1 = _run([1.0])
    l4, g4 = _run([1, 1, 1, 1])
    ok = np.isfinite(l1) and np.isfinite(l4) and torch.isfinite(g1).all() and torch.isfinite(g4).all()
    record("T-R5_nan_guard", "exact", ok,
           f"loss1={l1}, loss4={l4}, grad1={g1.tolist()}, grad4 finite={bool(torch.isfinite(g4).all())}",
           "finite loss and grads; length-1 skips standardization",
           "std(unbiased=False)+1e-8; length-1 would be 0/0 without the skip.")


def run_return_tests():
    test_tr1_returns()
    test_tr4_adam_first_step()
    test_tr5_nan_guard()
