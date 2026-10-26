"""Validation suite for the shuttle MDP (edge cases + quantitative correctness evidence).

Run:   python validation.py            -> prints a table, writes validation_results.md / .json
Pytest: pytest -q test_shuttle.py       -> asserts every 'exact' check

Each check returns a record:
    name      short id
    kind      'exact'      mathematical identity / hard assertion -> must pass
              'prediction' back-of-envelope or theory-derived expectation -> reported honestly
                           whether or not it holds; does not fail the suite
    passed    bool
    measured  what we measured
    expected  what we expected
    why       what the check validates (one line)
Independent implementations (plain loops, brute-force enumeration, Monte Carlo) are used
wherever possible so a bug in the matrix model cannot hide itself.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from shuttle_mdp import (DISPATCH, HOLD, ShuttleMDP, backward_induction, bellman_Q, evaluate_policy,
                         evaluate_policy_sweeps, greedy, policy_iteration, simulate_discounted_cost,
                         stationary_distribution, threshold_of, value_iteration)

RECORDS: list[dict] = []


def record(name, kind, passed, measured, expected, why):
    RECORDS.append(dict(name=name, kind=kind, passed=bool(passed), measured=str(measured),
                        expected=str(expected), why=why))


# ---------------------------------------------------------------------------------------------
# Model-level edge cases
# ---------------------------------------------------------------------------------------------
def check_transition_matrices():
    m = ShuttleMDP()
    row_err = float(np.abs(m.P.sum(axis=2) - 1).max())
    record("rows_sum_to_one", "exact", row_err < 1e-12 and m.P.min() >= 0,
           f"max |row sum - 1| = {row_err:.2e}", "< 1e-12, all entries >= 0",
           "No probability mass is lost or created, including at the cap.")
    P = m.P
    ok = (abs(P[HOLD, 199, 200] - 1) < 1e-15 and abs(P[HOLD, 198, 199] - .2) < 1e-15
          and abs(P[HOLD, 198, 200] - .8) < 1e-15 and abs(P[HOLD, 200, 200] - 1) < 1e-15)
    record("cap_clipping_mass", "exact", ok,
           f"P[hold][199,200]={P[HOLD,199,200]}, P[hold][198,199]={P[HOLD,198,199]}, P[hold][198,200]={P[HOLD,198,200]}",
           "1.0, 0.2, 0.8",
           "Arrivals that overflow the cap are clipped onto state 200 with the right probability.")
    ok = (np.allclose(P[DISPATCH, 0, 1:6], .2) and np.allclose(P[DISPATCH, 200, 186:191], .2)
          and abs(P[DISPATCH, 10, 1:6].sum() - 1) < 1e-15)
    record("dispatch_transitions", "exact", ok, "s=0 -> 1..5 (0.2 each); s=200 -> 186..190; s=10 -> 1..5",
           "uniform on leftover+{1..5}",
           "Dispatching with s<K (even s=0) ships everyone; s>K leaves s-K behind.")
    c = m.cost
    ok = (c[HOLD, 0] == 0 and c[DISPATCH, 0] == 100 and c[HOLD, 15] == 30 and c[DISPATCH, 15] == 100
          and c[DISPATCH, 20] == 110)
    record("cost_table", "exact", ok,
           f"c(hold,0)={c[HOLD,0]}, c(disp,0)={c[DISPATCH,0]}, c(hold,15)={c[HOLD,15]}, c(disp,15)={c[DISPATCH,15]}, c(disp,20)={c[DISPATCH,20]}",
           "0, 100, 30, 100, 110",
           "Costs follow the stated convention; dispatching an empty station still costs cf.")


# ---------------------------------------------------------------------------------------------
# (a) horizon / enumeration checks
# ---------------------------------------------------------------------------------------------
def check_horizon_hand_calcs():
    m = ShuttleMDP()
    V, _ = backward_induction(m, 1)
    hand = np.array([min(m.ch * s, m.cf + m.ch * max(s - m.K, 0)) for s in range(m.S)])
    err1 = float(np.abs(V[0] - hand).max())
    record("horizon_T1", "exact", err1 < 1e-12, f"max err = {err1:.2e}", "< 1e-12",
           "T=1: V0(s) = min(hold cost, dispatch cost). Catches an off-by-one in the backward loop.")
    # T=2 with plain Python loops (no matrices)
    V2, _ = backward_induction(m, 2)
    V1 = hand
    hand2 = np.zeros(m.S)
    for s in range(m.S):
        best = 1e18
        for a in (0, 1):
            left = s if a == 0 else max(s - m.K, 0)
            cost = m.cf * a + m.ch * left
            ev = sum(0.2 * V1[min(m.cap, left + A)] for A in range(1, 6))
            best = min(best, cost + m.gamma * ev)
        hand2[s] = best
    err2 = float(np.abs(V2[0] - hand2).max())
    record("horizon_T2", "exact", err2 < 1e-10, f"max err = {err2:.2e}", "< 1e-10",
           "T=2 recursion matches an independent loop implementation (no matrices).")
    Vt, _ = backward_induction(m, 500)
    record("horizon_array_shape", "exact", Vt.shape == (501, m.S) and np.all(Vt[500] == 0),
           f"V shape {Vt.shape}, V[T] all zero = {bool(np.all(Vt[500]==0))}", "(501, 201), V[500]=0",
           "Exactly T=500 backups: V[T]=0 terminal and V[0] is time zero.")


def check_bruteforce_enumeration():
    """True enumeration: with deterministic arrivals A=3 the optimal closed-loop value equals the
    best open-loop action sequence, so we can enumerate all 2^T sequences for T=12."""
    T = 12
    m = ShuttleMDP(arrivals=(3,), arrival_probs=(1.0,))
    V, _ = backward_induction(m, T)
    worst = 0.0
    for s0 in (0, 7, 14, 15, 16, 25, 40):
        best = 1e18
        for seq in itertools.product((0, 1), repeat=T):
            s, tot, d = s0, 0.0, 1.0
            for a in seq:
                left = s if a == 0 else max(s - 15, 0)
                tot += d * (100.0 * a + 2.0 * left)
                s = min(200, left + 3)
                d *= 0.95
            best = min(best, tot)
        worst = max(worst, abs(best - V[0, s0]))
    record("bruteforce_enumeration_T12", "exact", worst < 1e-9,
           f"max |enumerated optimum - backward induction| = {worst:.2e} over 7 start states x 4096 sequences",
           "< 1e-9",
           "Backward induction equals literal enumeration of all 2^12 action sequences (deterministic arrivals A=3).")


def check_horizon_vs_infinite(vi):
    m = ShuttleMDP()
    V, _ = backward_induction(m, 500)
    gap = float(np.abs(V[0] - vi.V).max())
    record("T500_vs_infinite_horizon", "exact", gap < 1e-6, f"max |V_500(.) - V_inf(.)| = {gap:.2e}",
           "< 1e-6 (theory: gamma^500 * ||V*|| ~ 0.95^500 * 3800 ~ 3e-8 worst case)",
           "Finite-horizon T=500 and infinite-horizon solutions agree, since gamma^500 ~ 7e-12.")


# ---------------------------------------------------------------------------------------------
# (b) / (c) solver agreement and convergence behaviour
# ---------------------------------------------------------------------------------------------
def check_vi_pi_agreement(vi, pi):
    m = ShuttleMDP()
    dv = float(np.abs(vi.V - pi.V).max())
    same = bool(np.array_equal(vi.policy, pi.policy))
    record("VI_equals_PI", "exact", dv < 1e-6 and same, f"max |V_VI - V_PI| = {dv:.2e}; policies identical = {same}",
           "< 1e-6 and identical", "Two independent algorithms reach the same fixed point.")
    finals = []
    for name, init in (("always_hold", np.zeros(m.S, int)), ("always_dispatch", np.ones(m.S, int)),
                       ("random", np.random.default_rng(0).integers(0, 2, m.S))):
        r = policy_iteration(m, init)
        finals.append((name, r.iterations, float(np.abs(r.V - pi.V).max()), bool(np.array_equal(r.policy, pi.policy))))
    ok = all(f[2] < 1e-8 and f[3] for f in finals)
    record("PI_start_independence", "exact", ok,
           "; ".join(f"{n}: {it} iters, dV={d:.1e}, same policy={p}" for n, it, d, p in finals),
           "same V and policy from all starts", "Policy iteration converges to the optimum from any initial policy.")
    mono = max(float((pi.V_history[k + 1] - pi.V_history[k]).max()) for k in range(len(pi.V_history) - 1))
    record("PI_monotone_improvement", "exact", mono <= 1e-9,
           f"max_s (V_{{k+1}} - V_k) over all iterations = {mono:.2e}; iterations = {pi.iterations}",
           "<= 1e-9 (costs never increase)", "Policy improvement theorem holds numerically; also catches sign errors in cost vs reward.")
    record("PI_few_iterations", "prediction", pi.iterations <= 10, f"{pi.iterations} iterations", "<= 10 (typical)",
           "Policy iteration is Newton-like and should need few iterations; many would suggest a bug.")
    pol = pi.policy
    sw = evaluate_policy_sweeps(m, pol, 10000)
    ex = evaluate_policy(m, pol)
    d = float(np.abs(sw - ex).max())
    record("PI_eval_solve_vs_sweeps", "exact", d < 1e-8, f"max |linear solve - 10000 sweeps| = {d:.2e}", "< 1e-8",
           "The linear-solve policy evaluation agrees with brute-force iterative evaluation.")


def check_vi_convergence(vi_ref, pi):
    m = ShuttleMDP()
    vi = value_iteration(m, V_star=pi.V)
    r0 = vi.residuals[0]
    g = m.gamma
    viol = max(e - g ** (k + 1) / (1 - g) * r0 for k, e in enumerate(vi.errors))
    record("VI_contraction_bound", "exact", viol <= 1e-9,
           f"max_n [ ||V_n - V*|| - gamma^n ||V_1-V_0|| /(1-gamma) ] = {viol:.2e} over {vi.iterations} iterations",
           "<= 0", "Value iteration error stays under the theoretical contraction bound at every iteration.")
    # the residual ratio should approach gamma (average cost > 0)
    res = np.array(vi.residuals)
    k = int(np.argmax(res < 1e-6))
    ratio = float(res[k + 1] / res[k])
    record("VI_residual_ratio_to_gamma", "prediction", abs(ratio - g) < 5e-3,
           f"residual ratio near residual 1e-6 = {ratio:.4f}", "~0.95",
           "Asymptotic VI rate equals gamma because the constant direction (average cost > 0) contracts exactly at gamma.")
    final_err = float(np.abs(vi.V - pi.V).max())
    bound = g / (1 - g) * vi.residuals[-1]
    record("VI_stopping_rule", "exact", final_err <= bound + 1e-12 and final_err < 1e-9,
           f"final error {final_err:.2e} <= gamma/(1-gamma) * last residual = {bound:.2e}", "error <= bound, error < 1e-9",
           "The stopping tolerance eps(1-gamma)/gamma delivers the promised accuracy.")
    record("VI_iteration_count", "prediction", 350 <= vi.iterations <= 700, f"{vi.iterations} iterations to eps=1e-10",
           "about 350-700 (worst case gamma^n * 470 < 5e-12 gives n ~ 630)", "Iteration count consistent with the contraction rate.")


# ---------------------------------------------------------------------------------------------
# Structure of the solution and invariances
# ---------------------------------------------------------------------------------------------
def check_structure(pi):
    m = ShuttleMDP()
    dV = np.diff(pi.V)
    slope_bound = m.ch / (1 - m.gamma)
    record("V_monotone_and_slope_bound", "exact", dV.min() >= -1e-9 and dV.max() <= slope_bound + 1e-9,
           f"min slope = {dV.min():.3f}, max slope = {dV.max():.3f}", f">= 0 and <= ch/(1-gamma) = {slope_bound:.1f}",
           "An extra waiting customer cannot help and costs at most ch every period forever.")
    thr, is_thr = threshold_of(pi.policy)
    record("threshold_structure", "exact", is_thr, f"threshold = {thr}, monotone threshold policy = {is_thr}",
           "dispatch iff s >= threshold", "Optimal policy is of control-limit type (no 'hold' above the first dispatch state).")
    record("threshold_value", "prediction", thr is not None and 13 <= thr <= 17, f"threshold = {thr}",
           "about 13-17 (queue ~ capacity K=15)", "Dispatching roughly when the shuttle can be filled.")
    D = np.diff(pi.V, 2)
    record("V_convexity_observation", "prediction", D.min() >= -1e-9, f"min second difference = {D.min():.3f}",
           ">= 0 (not guaranteed: min over actions of convex functions need not be convex)",
           "Observation only.")


def check_scaling_and_limits(pi):
    m = ShuttleMDP()
    big = ShuttleMDP(cf=1000.0, ch=20.0)
    r = policy_iteration(big)
    err = float(np.abs(r.V - 10 * pi.V).max() / np.abs(10 * pi.V).max())
    record("cost_scaling_x10", "exact", err < 1e-9 and np.array_equal(r.policy, pi.policy),
           f"relative max |V' - 10 V| = {err:.2e}, same policy = {bool(np.array_equal(r.policy, pi.policy))}",
           "< 1e-9, same policy", "Scaling both costs by 10 scales V by exactly 10 and leaves the policy unchanged (no hard-coded constants).")
    z = policy_iteration(ShuttleMDP(ch=0.0))
    record("limit_ch_zero", "exact", z.policy.sum() == 0 and np.abs(z.V).max() < 1e-12,
           f"dispatch states = {int(z.policy.sum())}, max|V| = {np.abs(z.V).max():.1e}", "never dispatch, V=0",
           "With no holding cost there is no reason to ever pay cf.")
    z = policy_iteration(ShuttleMDP(cf=0.0))
    ok = bool(np.all(z.policy[1:] == DISPATCH))
    record("limit_cf_zero", "exact", ok, f"dispatch at all s>=1: {ok} (s=0 is a tie -> hold)", "always dispatch for s>=1",
           "With free dispatch, always ship.")
    g0 = policy_iteration(ShuttleMDP(gamma=0.0))
    record("limit_gamma_zero", "exact", g0.policy.sum() == 0,
           f"dispatch states = {int(g0.policy.sum())}", "never dispatch (dispatch saves <= ch*K = 30 < cf = 100)",
           "A myopic agent never dispatches under these parameters.")
    thr = {}
    for gm in (0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99):
        r = policy_iteration(ShuttleMDP(gamma=gm))
        t, _ = threshold_of(r.policy)
        thr[gm] = t
    vals = [(10 ** 9 if t is None else t) for t in thr.values()]
    mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    first = next((gm for gm, t in thr.items() if t is not None), None)
    record("threshold_vs_gamma", "prediction", mono, f"threshold by gamma: {thr}",
           "nonincreasing in gamma; dispatch first appears around gamma ~ 0.7-0.8",
           f"More patient agents dispatch earlier; dispatch first used at gamma = {first}.")


def check_cap_invariance(pi):
    base = pi
    for cap in (300, 400):
        r = policy_iteration(ShuttleMDP(cap=cap))
        dv = float(np.abs(r.V[:201] - base.V).max())
        same = bool(np.array_equal(r.policy[:201], base.policy))
        record(f"cap_{cap}_vs_200", "exact", dv < 1e-8 and same,
               f"max |V_{cap} - V_200| on s<=200 = {dv:.2e}; policy identical = {same}", "< 1e-8, identical",
               "The optimal policy never lets the queue approach 200, so the cap is irrelevant.")
    r30 = policy_iteration(ShuttleMDP(cap=30))
    dv = float(np.abs(r30.V - base.V[:31]).max())
    record("cap_30_vs_200", "exact", dv < 1e-8 and np.array_equal(r30.policy, base.policy[:31]),
           f"max |V_30 - V_200| on s<=30 = {dv:.2e}", "< 1e-8", "Even a cap of 30 gives identical values on the shared states.")
    r10 = policy_iteration(ShuttleMDP(cap=10))
    record("cap_10_artifact", "prediction", r10.V[0] < base.V[0] and r10.policy.sum() == 0,
           f"V_10(0) = {r10.V[0]:.2f} vs V_200(0) = {base.V[0]:.2f}; dispatch states = {int(r10.policy.sum())}",
           "V lower and never dispatch: overflow customers vanish for free, so a tiny cap is an artifact",
           "Shows what the cap assumption does when it is binding.")


def check_timing_sensitivity(pi):
    alt = policy_iteration(ShuttleMDP(holding="next_state"))
    t_alt, _ = threshold_of(alt.policy)
    t_base, _ = threshold_of(pi.policy)
    record("cost_timing_sensitivity", "prediction", t_alt is not None and abs(t_alt - t_base) <= 3,
           f"threshold with holding charged on leftover = {t_base}; on post-arrival queue = {t_alt}; "
           f"V(0): {pi.V[0]:.2f} vs {alt.V[0]:.2f}", "threshold within +-3",
           "Charging ch on the post-arrival queue adds the constant ch*E[A] = 6 per period, i.e. exactly 120 to every V "
           "(until the cap binds), so the policy cannot change: the timing ambiguity is provably harmless here.")


# ---------------------------------------------------------------------------------------------
# Independent verification of the optimal policy's value
# ---------------------------------------------------------------------------------------------
def check_monte_carlo_and_theory(pi):
    m = ShuttleMDP()
    rng = np.random.default_rng(12345)
    n = 40000
    tot = simulate_discounted_cost(m, pi.policy, 0, n, 600, rng)
    mean, se = float(tot.mean()), float(tot.std(ddof=1) / np.sqrt(n))
    z = (mean - pi.V[0]) / se
    record("monte_carlo_V0", "exact", abs(z) < 3.5,
           f"MC mean = {mean:.2f} +- {se:.2f} (SE) vs V*(0) = {pi.V[0]:.2f}; z = {z:.2f}", "|z| < 3.5",
           "Simulating the optimal policy with random arrivals reproduces V*(0) (independent of the matrices).")
    # a deliberately bad policy must be worse by a large margin
    never = np.zeros(m.S, int)
    v_never = evaluate_policy(m, never)
    record("optimal_beats_alternatives", "exact", bool(np.all(pi.V <= v_never + 1e-9) and np.all(pi.V <= evaluate_policy(m, np.ones(m.S, int)) + 1e-9)),
           f"V*(0)={pi.V[0]:.1f}, never-dispatch V(0)={v_never[0]:.1f}, always-dispatch V(0)={evaluate_policy(m, np.ones(m.S, int))[0]:.1f}",
           "V* <= V_pi for every state", "The optimum is no worse than trivial policies.")
    # deterministic arrivals: forward simulation by plain loop vs DP
    md = ShuttleMDP(arrivals=(3,), arrival_probs=(1.0,))
    rd = policy_iteration(md)
    s, tot, d = 0, 0.0, 1.0
    for _ in range(3000):
        a = int(rd.policy[s]); left = s if a == 0 else max(s - 15, 0)
        tot += d * (100.0 * a + 2.0 * left); s = min(200, left + 3); d *= 0.95
    record("deterministic_arrivals_forward_sim", "exact", abs(tot - rd.V[0]) < 1e-8,
           f"forward simulation = {tot:.6f}, DP V*(0) = {rd.V[0]:.6f}", "equal to 1e-8",
           "With A=3 every period, plain-loop simulation of the DP policy matches DP exactly.")
    mu, resid = stationary_distribution(m, pi.policy)
    s_idx = np.arange(m.S)
    g = float(mu @ m.cost[pi.policy, s_idx])
    lhs = (1 - m.gamma) * float(mu @ pi.V)
    record("stationary_identity", "exact", abs(lhs - g) < 1e-6 and resid < 1e-9,
           f"(1-gamma) E_mu[V*] = {lhs:.6f}; average cost g = {g:.6f}; stationarity residual {resid:.1e}",
           "equal", "Exact identity linking discounted values and the long-run average cost under the optimal policy.")
    freq = float(mu @ pi.policy)
    thr, _ = threshold_of(pi.policy)
    exp = 3.0 / 15.0
    if thr is not None and thr >= 15:
        record("dispatch_frequency", "exact", abs(freq - exp) < 1e-6, f"dispatch frequency = {freq:.6f}", "3/15 = 0.2",
               "Flow balance: each dispatch serves 15, arrivals average 3 per period.")
    else:
        served = 3.0 / freq
        record("dispatch_frequency", "exact", freq > exp,
               f"threshold {thr} < K=15 so dispatches ship fewer than 15; frequency = {freq:.4f} (> 0.2), "
               f"mean customers served per dispatch = {served:.2f}",
               "frequency = 3 / (mean customers served per dispatch) > 0.2",
               "Flow balance with partial loads: frequency x mean load = mean arrival rate (3).")
    record("average_cost_prediction", "prediction", 30 <= g <= 40, f"average cost per period g = {g:.2f}", "about 30-40",
           "Back-of-envelope: dispatch about every 5 periods (20/period) plus holding (about 2 x queue).")


# ---------------------------------------------------------------------------------------------
def run_all():
    RECORDS.clear()
    m = ShuttleMDP()
    pi = policy_iteration(m)
    vi = value_iteration(m)
    check_transition_matrices()
    check_horizon_hand_calcs()
    check_bruteforce_enumeration()
    check_horizon_vs_infinite(vi)
    check_vi_pi_agreement(vi, pi)
    check_vi_convergence(vi, pi)
    check_structure(pi)
    check_scaling_and_limits(pi)
    check_cap_invariance(pi)
    check_timing_sensitivity(pi)
    check_monte_carlo_and_theory(pi)
    return list(RECORDS)


def write_report(records, out_dir: Path):
    (out_dir / "validation_results.json").write_text(json.dumps(records, indent=1))
    lines = ["| # | check | kind | result | measured | expected |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(records, 1):
        res = "PASS" if r["passed"] else ("FAIL" if r["kind"] == "exact" else "NOT HELD")
        lines.append(f"| {i} | {r['name']} | {r['kind']} | {res} | {r['measured']} | {r['expected']} |")
    (out_dir / "validation_results.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    recs = run_all()
    table = write_report(recs, Path(__file__).parent)
    print(table)
    failed = [r for r in recs if r["kind"] == "exact" and not r["passed"]]
    held = [r for r in recs if r["kind"] == "prediction" and not r["passed"]]
    print(f"\n{len(recs)} checks: {len(failed)} exact failures, {len(held)} predictions not held")
    sys.exit(1 if failed else 0)
