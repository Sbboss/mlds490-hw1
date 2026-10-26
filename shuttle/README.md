# Part 1: Shuttle dispatching MDP (value iteration / policy iteration / enumeration)

## Run
```bash
pip install numpy matplotlib scipy pytest
python run_shuttle.py          # solves (a) T=500 enumeration, (b) value iteration, (c) policy iteration;
                               # writes figures/*.png and results.json   (about 1 second)
python validation.py           # 38 validation checks -> validation_results.md / .json
pytest -q test_shuttle.py      # the 'exact' checks as pytest tests
```
No GPU or special hardware is needed.

## Model and assumptions (state these in the report)
- State s = customers waiting, 0..200. Action: hold or dispatch. A dispatch ships min(s, K=15).
- Cost = cf * [dispatched] + ch * (customers left waiting AFTER the decision, before new arrivals). cf=100, ch=2.
- Next state = min(200, leftover + A), A ~ Unif{1..5}. Arrivals above 200 are discarded at no cost.
- gamma = 0.95 for all three methods. Costs are minimised. Ties go to HOLD unless dispatch is better by more than 1e-9.
- float64 throughout. "Enumeration" = exhaustive backward induction over all T=500 decision epochs
  (literal enumeration of 2^500 action sequences is impossible; literal enumeration of 2^12 sequences is used as a test).

## What each validation check does
See `validation_results.md` for measured vs expected values. Checks marked `exact` are identities or hard
assertions (must pass). Checks marked `prediction` are back-of-envelope expectations that are reported
honestly even when they fail. Groups: model edge cases (transition rows, cap clipping, s<K / s>K dispatch);
horizon checks (T=1, T=2 by hand, brute-force enumeration for T=12 with A=3, T=500 vs infinite horizon);
solver agreement (VI vs PI vs three PI starting policies, PI monotone improvement, VI contraction bound);
structure and invariances (monotone V, slope bound, cost scaling, limiting cases ch=0 / cf=0 / gamma=0,
cap 200/300/400/30/10, cost-timing sensitivity); independent verification (Monte Carlo of V*(0),
forward simulation with deterministic arrivals, (1-gamma) E_mu[V*] = average cost).
