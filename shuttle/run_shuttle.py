"""Solve the shuttle MDP three ways and produce the three required figures.

    python run_shuttle.py            # writes figures/*.png and results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shuttle_mdp import (ShuttleMDP, backward_induction, policy_iteration, threshold_of, value_iteration)

OUT = Path(__file__).parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated categorical slots 1 and 2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def style_axes(ax):
    ax.set_axisbelow(True)


def main():
    m = ShuttleMDP()
    T = 500
    V_bi, pol_bi = backward_induction(m, T)
    vi = value_iteration(m)
    pi = policy_iteration(m)
    s = np.arange(m.S)
    thr, _ = threshold_of(pi.policy)

    # ------------------------------------------------------------------ (a) enumeration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot(s, V_bi[0], color=BLUE)
    a1.set_title("(a) Optimal value function at time 0, T = 500")
    a1.set_xlabel("customers waiting, s"); a1.set_ylabel("expected discounted cost V$_0$(s)")
    a1.annotate(f"V$_0$(0) = {V_bi[0,0]:.1f}", (0, V_bi[0, 0]), (12, V_bi[0, 0] + 500), color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    a1.annotate(f"V$_0$(200) = {V_bi[0,200]:.0f}", (200, V_bi[0, 200]), (118, V_bi[0, 200] - 150), color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2))
    z = s <= 40
    a2.plot(s[z], V_bi[0][z], color=BLUE, marker="o", markersize=4)
    a2.axvline(thr, color=INK2, linestyle=":", linewidth=1.2)
    a2.text(thr + 0.6, V_bi[0, 0] + 8, f"policy switches to\ndispatch at s = {thr}", color=INK2, fontsize=9, va="bottom")
    a2.set_title("Zoom: s = 0 to 40")
    a2.set_xlabel("customers waiting, s"); a2.set_ylabel("V$_0$(s)")
    for ax in (a1, a2): style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig_a_enumeration_V0.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------------ (b) value iteration
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
    b1.plot(s, vi.V, color=BLUE, label="value iteration V*(s)")
    b1.plot(s, V_bi[0], color=ORANGE, linestyle="--", linewidth=1.6, label="enumeration V$_0$(s), T = 500")
    b1.set_title("(b) Optimal value function, infinite horizon")
    b1.set_xlabel("customers waiting, s"); b1.set_ylabel("expected discounted cost V*(s)")
    b1.legend(loc="upper left")
    gap = float(np.abs(V_bi[0] - vi.V).max())
    b1.text(0.98, 0.06, f"max |V$_0$ − V*| = {gap:.1e}", transform=b1.transAxes, ha="right", color=INK2, fontsize=9)
    n = np.arange(1, len(vi.residuals) + 1)
    b2.semilogy(n, vi.residuals, color=BLUE, label="residual ‖V$_{n+1}$ − V$_n$‖$_∞$")
    b2.semilogy(n, vi.residuals[0] * m.gamma ** (n - 1), color=ORANGE, linestyle="--", linewidth=1.6,
                label="γ$^{n-1}$ × first residual")
    b2.set_title(f"Convergence: {vi.iterations} iterations to ε = 1e-10")
    b2.set_xlabel("iteration n"); b2.set_ylabel("sup-norm residual"); b2.legend(loc="upper right")
    for ax in (b1, b2): style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig_b_value_iteration.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------------ (c) policy iteration
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    c1.step(s, pi.policy, where="post", color=BLUE)
    c1.set_yticks([0, 1]); c1.set_yticklabels(["hold (0)", "dispatch (1)"])
    c1.set_ylim(-0.1, 1.15)
    c1.axvline(thr, color=INK2, linestyle=":", linewidth=1.2)
    c1.text(thr + 3, 0.5, f"dispatch whenever s ≥ {thr}", color=INK, fontsize=10)
    c1.set_title("(c) Optimal policy from policy iteration")
    c1.set_xlabel("customers waiting, s"); c1.set_ylabel("optimal action")
    changes = [int((pi.policy_history[k + 1] != pi.policy_history[k]).sum()) for k in range(len(pi.policy_history) - 1)]
    changes = changes + [0]       # the last iteration confirms no state changes
    ks = np.arange(1, len(changes) + 1)
    c2.bar(ks, changes, color=BLUE, width=0.6)
    for k, v in zip(ks, changes):
        c2.text(k, v + 1.2, str(v), ha="center", color=INK, fontsize=9)
    c2.set_title(f"Policy iteration converges in {pi.iterations} iterations\n(start: always hold)")
    c2.set_xlabel("iteration"); c2.set_ylabel("states whose action changed")
    c2.set_xticks(ks); c2.set_ylim(0, max(changes) * 1.15)
    for ax in (c1, c2): style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG / "fig_c_policy_iteration.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------------ numbers for the report
    slopes = {int(k): float(vi.V[k + 1] - vi.V[k]) for k in (10, 50, 100, 199)}
    res = dict(
        params=dict(K=m.K, cf=m.cf, ch=m.ch, cap=m.cap, gamma=m.gamma, T=T),
        V0_at_0=float(V_bi[0, 0]), V0_at_15=float(V_bi[0, 15]), V0_at_200=float(V_bi[0, 200]),
        enumeration_vs_VI_max_gap=gap,
        VI_iterations=vi.iterations, PI_iterations=pi.iterations,
        VI_vs_PI_max_gap=float(np.abs(vi.V - pi.V).max()),
        policies_identical=bool(np.array_equal(vi.policy, pi.policy) and np.array_equal(pol_bi[0], pi.policy)),
        threshold=thr, finite_diff_slopes=slopes,
        pi_states_changed_per_iteration=changes,
    )
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
