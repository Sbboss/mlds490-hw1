"""Shuttle dispatching MDP and its three solvers.

Model (all assumptions are stated here and repeated in the report)
-------------------------------------------------------------------
State  s in {0, ..., cap}       customers waiting at the station at the start of a period.
Action a in {HOLD=0, DISPATCH=1}.
Dispatch ships min(s, K) customers.  Customers left behind ("leftover"):
    HOLD     -> leftover = s
    DISPATCH -> leftover = max(s - K, 0)
Cost (we minimise cost; gamma-discounted):
    c(s, a) = cf * a + ch * leftover            (holding="leftover", the default)
i.e. ch is charged on customers still waiting AFTER the dispatch decision, before new
arrivals are counted.  An alternative convention (holding="next_state") charges ch on the
customers present after arrivals, used as a sensitivity check.
Next state: s' = min(cap, leftover + A),  A ~ Unif{1,...,5}.
Arrivals that would push the queue above `cap` are DISCARDED at no cost (assumption; the
cap is a parameter and the validation suite shows it does not matter for cap >= ~30).

Everything is float64.  Ties between actions are resolved in favour of HOLD unless
DISPATCH is better by more than TIE_TOL (so VI and PI return identical policies).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOLD, DISPATCH = 0, 1
TIE_TOL = 1e-9


class ShuttleMDP:
    def __init__(self, K=15, cf=100.0, ch=2.0, cap=200, arrivals=(1, 2, 3, 4, 5),
                 arrival_probs=None, gamma=0.95, holding="leftover"):
        assert holding in ("leftover", "next_state")
        self.K, self.cf, self.ch, self.cap, self.gamma = K, float(cf), float(ch), cap, float(gamma)
        self.holding = holding
        self.arrivals = np.asarray(arrivals, dtype=int)
        if arrival_probs is None:
            arrival_probs = np.full(len(self.arrivals), 1.0 / len(self.arrivals))
        self.arrival_probs = np.asarray(arrival_probs, dtype=np.float64)
        assert abs(self.arrival_probs.sum() - 1.0) < 1e-12

        S = cap + 1
        self.S = S
        s = np.arange(S)
        self.leftover = np.stack([s, np.maximum(s - K, 0)])          # (2, S)
        P = np.zeros((2, S, S))
        cost = np.zeros((2, S))
        for a in (HOLD, DISPATCH):
            for A, pa in zip(self.arrivals, self.arrival_probs):
                nxt = np.minimum(cap, self.leftover[a] + A)
                np.add.at(P[a], (s, nxt), pa)
                if holding == "next_state":
                    cost[a] += pa * self.ch * nxt
            if holding == "leftover":
                cost[a] = self.ch * self.leftover[a]
            cost[a] += self.cf * a
        self.P, self.cost = P, cost


# ----------------------------------------------------------------------------- helpers
def bellman_Q(mdp: ShuttleMDP, V: np.ndarray) -> np.ndarray:
    """Q[a, s] = c(s,a) + gamma * sum_s' P[a,s,s'] V[s']."""
    return mdp.cost + mdp.gamma * np.einsum("aij,j->ai", mdp.P, V)


def greedy(Q: np.ndarray) -> np.ndarray:
    """Canonical greedy policy: dispatch only if strictly better than hold by > TIE_TOL."""
    return (Q[DISPATCH] < Q[HOLD] - TIE_TOL).astype(int)


def threshold_of(policy: np.ndarray):
    """Return (threshold, is_threshold_policy).  threshold=None means never dispatch."""
    idx = np.flatnonzero(policy == DISPATCH)
    if len(idx) == 0:
        return None, True
    thr = int(idx[0])
    return thr, bool(np.all(policy[thr:] == DISPATCH))


# ----------------------------------------------------------------------------- (a) enumeration
def backward_induction(mdp: ShuttleMDP, T: int = 500):
    """Exhaustive finite-horizon recursion over all T decision epochs.

    V[t, s] is the optimal expected discounted cost-to-go from period t, with V[T] = 0.
    "Enumeration" over 2^T action sequences is infeasible for T = 500; backward induction
    evaluates every (t, s, a) triple exactly once, which is the exhaustive recursion.
    Returns V with shape (T+1, S) and policy with shape (T, S).
    """
    V = np.zeros((T + 1, mdp.S))
    policy = np.zeros((T, mdp.S), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = bellman_Q(mdp, V[t + 1])
        V[t] = Q.min(axis=0)
        policy[t] = greedy(Q)
    return V, policy


# ----------------------------------------------------------------------------- (b) value iteration
@dataclass
class VIResult:
    V: np.ndarray
    policy: np.ndarray
    iterations: int
    residuals: list = field(default_factory=list)   # ||V_{n+1}-V_n||_inf
    errors: list = field(default_factory=list)       # ||V_{n+1}-V*||_inf (only if V_star given)


def value_iteration(mdp: ShuttleMDP, eps: float = 1e-10, max_iter: int = 200000, V_star=None):
    """Value iteration from V0 = 0.

    Stop when the sup-norm residual < eps*(1-gamma)/gamma.  Then ||V - V*|| <= eps because
    the true error can be up to gamma/(1-gamma) times the residual.
    """
    g = mdp.gamma
    tol = eps * (1 - g) / g if g > 0 else eps
    V = np.zeros(mdp.S)
    residuals, errors = [], []
    for it in range(1, max_iter + 1):
        Vn = bellman_Q(mdp, V).min(axis=0)
        r = float(np.max(np.abs(Vn - V)))
        residuals.append(r)
        if V_star is not None:
            errors.append(float(np.max(np.abs(Vn - V_star))))
        V = Vn
        if r < tol:
            break
    policy = greedy(bellman_Q(mdp, V))
    return VIResult(V, policy, it, residuals, errors)


# ----------------------------------------------------------------------------- (c) policy iteration
def evaluate_policy(mdp: ShuttleMDP, policy: np.ndarray) -> np.ndarray:
    """Exact policy evaluation: solve (I - gamma P_pi) V = c_pi."""
    s = np.arange(mdp.S)
    P_pi = mdp.P[policy, s, :]
    c_pi = mdp.cost[policy, s]
    return np.linalg.solve(np.eye(mdp.S) - mdp.gamma * P_pi, c_pi)


def evaluate_policy_sweeps(mdp: ShuttleMDP, policy: np.ndarray, sweeps: int = 10000) -> np.ndarray:
    """Iterative policy evaluation (independent of the linear solve; used as a cross-check)."""
    s = np.arange(mdp.S)
    P_pi = mdp.P[policy, s, :]
    c_pi = mdp.cost[policy, s]
    V = np.zeros(mdp.S)
    for _ in range(sweeps):
        V = c_pi + mdp.gamma * P_pi @ V
    return V


@dataclass
class PIResult:
    V: np.ndarray
    policy: np.ndarray
    iterations: int
    V_history: list = field(default_factory=list)
    policy_history: list = field(default_factory=list)


def policy_iteration(mdp: ShuttleMDP, init_policy=None, max_iter: int = 500):
    """Howard policy iteration.  An action is switched only if strictly better by > TIE_TOL,
    which guarantees termination (no cycling among tied policies)."""
    s = np.arange(mdp.S)
    pol = np.zeros(mdp.S, dtype=int) if init_policy is None else np.asarray(init_policy, dtype=int).copy()
    V_hist, pol_hist = [], [pol.copy()]
    for it in range(1, max_iter + 1):
        V = evaluate_policy(mdp, pol)
        V_hist.append(V)
        Q = bellman_Q(mdp, V)
        best = np.argmin(Q, axis=0)
        improve = Q[best, s] < Q[pol, s] - TIE_TOL
        if not improve.any():
            break
        pol = np.where(improve, best, pol)
        pol_hist.append(pol.copy())
    return PIResult(V, greedy(bellman_Q(mdp, V)), it, V_hist, pol_hist)


# ----------------------------------------------------------------------------- diagnostics
def stationary_distribution(mdp: ShuttleMDP, policy: np.ndarray, s0: int = 0,
                            n_iter: int = 20000, tail: int = 5000):
    """Long-run state distribution of the chain started at s0 (Cesaro average of the last
    `tail` iterates).  Returns (mu, residual ||mu P - mu||_1)."""
    s = np.arange(mdp.S)
    P_pi = mdp.P[policy, s, :]
    mu = np.zeros(mdp.S)
    mu[s0] = 1.0
    acc = np.zeros(mdp.S)
    for k in range(n_iter):
        mu = mu @ P_pi
        if k >= n_iter - tail:
            acc += mu
    mu = acc / tail
    return mu, float(np.abs(mu @ P_pi - mu).sum())


def simulate_discounted_cost(mdp: ShuttleMDP, policy: np.ndarray, s0: int, n_episodes: int,
                             horizon: int, rng: np.random.Generator):
    """Monte Carlo discounted cost of a stationary policy (default cost convention only)."""
    assert mdp.holding == "leftover"
    s = np.full(n_episodes, s0)
    total = np.zeros(n_episodes)
    disc = 1.0
    for _ in range(horizon):
        a = policy[s]
        left = mdp.leftover[a, s]
        total += disc * (mdp.cf * a + mdp.ch * left)
        A = rng.choice(mdp.arrivals, size=n_episodes, p=mdp.arrival_probs)
        s = np.minimum(mdp.cap, left + A)
        disc *= mdp.gamma
    return total
