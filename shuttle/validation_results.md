| # | check | kind | result | measured | expected |
|---|---|---|---|---|---|
| 1 | rows_sum_to_one | exact | PASS | max |row sum - 1| = 0.00e+00 | < 1e-12, all entries >= 0 |
| 2 | cap_clipping_mass | exact | PASS | P[hold][199,200]=1.0, P[hold][198,199]=0.2, P[hold][198,200]=0.8 | 1.0, 0.2, 0.8 |
| 3 | dispatch_transitions | exact | PASS | s=0 -> 1..5 (0.2 each); s=200 -> 186..190; s=10 -> 1..5 | uniform on leftover+{1..5} |
| 4 | cost_table | exact | PASS | c(hold,0)=0.0, c(disp,0)=100.0, c(hold,15)=30.0, c(disp,15)=100.0, c(disp,20)=110.0 | 0, 100, 30, 100, 110 |
| 5 | horizon_T1 | exact | PASS | max err = 0.00e+00 | < 1e-12 |
| 6 | horizon_T2 | exact | PASS | max err = 1.14e-13 | < 1e-10 |
| 7 | horizon_array_shape | exact | PASS | V shape (501, 201), V[T] all zero = True | (501, 201), V[500]=0 |
| 8 | bruteforce_enumeration_T12 | exact | PASS | max |enumerated optimum - backward induction| = 1.14e-13 over 7 start states x 4096 sequences | < 1e-9 |
| 9 | T500_vs_infinite_horizon | exact | PASS | max |V_500(.) - V_inf(.)| = 4.86e-09 | < 1e-6 (theory: gamma^500 * ||V*|| ~ 0.95^500 * 3800 ~ 3e-8 worst case) |
| 10 | VI_equals_PI | exact | PASS | max |V_VI - V_PI| = 8.23e-11; policies identical = True | < 1e-6 and identical |
| 11 | PI_start_independence | exact | PASS | always_hold: 6 iters, dV=0.0e+00, same policy=True; always_dispatch: 5 iters, dV=0.0e+00, same policy=True; random: 4 iters, dV=0.0e+00, same policy=True | same V and policy from all starts |
| 12 | PI_monotone_improvement | exact | PASS | max_s (V_{k+1} - V_k) over all iterations = 0.00e+00; iterations = 6 | <= 1e-9 (costs never increase) |
| 13 | PI_few_iterations | prediction | PASS | 6 iterations | <= 10 (typical) |
| 14 | PI_eval_solve_vs_sweeps | exact | PASS | max |linear solve - 10000 sweeps| = 9.09e-13 | < 1e-8 |
| 15 | VI_contraction_bound | exact | PASS | max_n [ ||V_n - V*|| - gamma^n ||V_1-V_0|| /(1-gamma) ] = -8.79e-10 over 580 iterations | <= 0 |
| 16 | VI_residual_ratio_to_gamma | prediction | PASS | residual ratio near residual 1e-6 = 0.9500 | ~0.95 |
| 17 | VI_stopping_rule | exact | PASS | final error 8.23e-11 <= gamma/(1-gamma) * last residual = 9.50e-11 | error <= bound, error < 1e-9 |
| 18 | VI_iteration_count | prediction | PASS | 580 iterations to eps=1e-10 | about 350-700 (worst case gamma^n * 470 < 5e-12 gives n ~ 630) |
| 19 | V_monotone_and_slope_bound | exact | PASS | min slope = -0.000, max slope = 24.852 | >= 0 and <= ch/(1-gamma) = 40.0 |
| 20 | threshold_structure | exact | PASS | threshold = 13, monotone threshold policy = True | dispatch iff s >= threshold |
| 21 | threshold_value | prediction | PASS | threshold = 13 | about 13-17 (queue ~ capacity K=15) |
| 22 | V_convexity_observation | prediction | NOT HELD | min second difference = -3.778 | >= 0 (not guaranteed: min over actions of convex functions need not be convex) |
| 23 | cost_scaling_x10 | exact | PASS | relative max |V' - 10 V| = 2.86e-16, same policy = True | < 1e-9, same policy |
| 24 | limit_ch_zero | exact | PASS | dispatch states = 0, max|V| = 0.0e+00 | never dispatch, V=0 |
| 25 | limit_cf_zero | exact | PASS | dispatch at all s>=1: True (s=0 is a tie -> hold) | always dispatch for s>=1 |
| 26 | limit_gamma_zero | exact | PASS | dispatch states = 0 | never dispatch (dispatch saves <= ch*K = 30 < cf = 100) |
| 27 | threshold_vs_gamma | prediction | PASS | threshold by gamma: {0.0: None, 0.3: None, 0.5: None, 0.6: None, 0.7: None, 0.75: 15, 0.8: 14, 0.85: 14, 0.9: 13, 0.95: 13, 0.99: 13} | nonincreasing in gamma; dispatch first appears around gamma ~ 0.7-0.8 |
| 28 | cap_300_vs_200 | exact | PASS | max |V_300 - V_200| on s<=200 = 4.55e-13; policy identical = True | < 1e-8, identical |
| 29 | cap_400_vs_200 | exact | PASS | max |V_400 - V_200| on s<=200 = 4.55e-13; policy identical = True | < 1e-8, identical |
| 30 | cap_30_vs_200 | exact | PASS | max |V_30 - V_200| on s<=30 = 2.27e-13 | < 1e-8 |
| 31 | cap_10_artifact | prediction | PASS | V_10(0) = 356.04 vs V_200(0) = 576.67; dispatch states = 0 | V lower and never dispatch: overflow customers vanish for free, so a tiny cap is an artifact |
| 32 | cost_timing_sensitivity | prediction | PASS | threshold with holding charged on leftover = 13; on post-arrival queue = 13; V(0): 576.67 vs 696.67 | threshold within +-3 |
| 33 | monte_carlo_V0 | exact | PASS | MC mean = 576.56 +- 0.16 (SE) vs V*(0) = 576.67; z = -0.68 | |z| < 3.5 |
| 34 | optimal_beats_alternatives | exact | PASS | V*(0)=576.7, never-dispatch V(0)=2203.9, always-dispatch V(0)=2000.0 | V* <= V_pi for every state |
| 35 | deterministic_arrivals_forward_sim | exact | PASS | forward simulation = 569.753019, DP V*(0) = 569.753019 | equal to 1e-8 |
| 36 | stationary_identity | exact | PASS | (1-gamma) E_mu[V*] = 32.344188; average cost g = 32.344188; stationarity residual 6.1e-14 | equal |
| 37 | dispatch_frequency | exact | PASS | threshold 13 < K=15 so dispatches ship fewer than 15; frequency = 0.2133 (> 0.2), mean customers served per dispatch = 14.06 | frequency = 3 / (mean customers served per dispatch) > 0.2 |
| 38 | average_cost_prediction | prediction | PASS | average cost per period g = 32.34 | about 30-40 |
