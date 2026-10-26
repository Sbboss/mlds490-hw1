| # | check | kind | result | measured | expected |
|---|---|---|---|---|---|
| 1 | T-R1_half | exact | PASS | [1.75, 1.5, 1.0] | [1.75, 1.5, 1.0] |
| 2 | T-R1_cartpole_gamma | exact | PASS | [2.8525, 1.95, 1.0] | [2.8525, 1.95, 1.0] |
| 3 | T-R1_pong_reset | exact | PASS | [-0.9801, -0.99, -1.0, 0.9801, 0.99, 1.0] | [-0.9801, -0.99, -1, 0.9801, 0.99, 1] |
| 4 | T-R1_pong_noleak_contrast | exact | PASS | G[2]=-0.029701 | ≈ -0.0297 |
| 5 | T-R4_adam_first_step | exact | PASS | delta=[-0.010000000707805157, 0.0, 0.009999990463256836] | ±0.01 on nonzero grads, 0 on zero grad |
| 6 | T-R5_nan_guard | exact | PASS | loss1=-0.0, loss4=-0.0, grad1=[-1.0], grad4 finite=True | finite loss and grads; length-1 skips standardization |
| 7 | T-R2_weighted_matches_true_grad | exact | PASS | max /z/=1.13; mean=[[-0.3647, 0.3647], [0.0327, -0.0327], [-0.1108, 0.1108]]; exact=[[-0.3656, 0.3656], [0.0327, -0.0327], [-0.1113, 0.1113]] | every component of the N=200k mean within 4 SE of ∇J |
| 8 | T-R2_unweighted_matches_its_expectation | exact | PASS | max /z/=0.72 | within 4 SE of enumerated E[unweighted] |
| 9 | T-R2_unweighted_differs_from_true_grad | exact | PASS | max /E[unweighted]−∇J/ / SE = 147.0; E_u=[[-0.3656, 0.3656], [0.0653, -0.0653], [-0.2225, 0.2225]] ∇J=[[-0.3656, 0.3656], [0.0327, -0.0327], [-0.1113, 0.1113]] | differs from true ∇J by >10 SE in at least one component |
| 10 | T-R2_J_value | exact | PASS | J(θ)=0.828706 | enumerated J (logged) |
| 11 | T-R3_bandit_closed_form | exact | PASS | mean=[0.25087, -0.25087] se=[0.0005590150069434007, 0.0005590150069434007] z=[1.5563088453689187, 1.5563088453689187] | (+0.25, -0.25) within 4 SE, N=200k |
| 12 | T-B1_unbiasedness | exact | PASS | max/z/ b=0:0.98 b=J:0.92 b=V:0.92 | all three means within 4 SE of ∇J (N=200k, +10 reward offset) |
| 13 | T-B1_var_zero_gt_J | exact | PASS | tr(Cov) b=0:139.6030 b=J:3.5819 b=V:0.1122 | Var(b=0) > Var(b=J) |
| 14 | T-B1_var_J_gt_V | prediction | PASS | tr(Cov) b=J:3.5819 b=V:0.1122 | Var(b=J) > Var(b=V^π) expected |
| 15 | T-B2_bstar_at_half | exact | PASS | b*=5.500000 J=5.500000 | b* = J when π=(0.5,0.5) |
| 16 | T-B2_rare_action_weight | exact | PASS | w_rare/w_common=9.0000 | 9 |
| 17 | T-B2_variance_ordering | exact | PASS | Var(b*)=0.016200 Var(J)=0.131400 Var(0)=4.698000 b*=5.1000 J=5.9000 | Var(b*) ≤ Var(J) ≤ Var(0) |
| 18 | T-B3_zero_value_equals_no_baseline | exact | PASS | max abs diff = 0.00e+00 | allclose rtol 1e-6 |
| 19 | T-B3_no_value_grad_from_policy_loss | exact | PASS | any nonzero value grad=False | value-net grad is None or all-zero after policy backward |
| 20 | T-B3_value_loss_leaves_policy_grad | exact | PASS | policy grads identical after value MSE backward | unchanged |
| 21 | T-B4_ema_previous_only | exact | PASS | baselines used=[0.0, 10.0, 11.0]; after last, b=12.9000 | used[i] computed only from seq[:i] (first=0) |
| 22 | T-B6_same_G_target | exact | PASS | ValueBaseline.update receives the same G_t tensor the policy loss used | by construction in train_cartpole.policy_update / train_pong.train |
| 23 | T-C1_term_vs_trunc | exact | PASS | lengths=[500, 500, 500, 500, 500] flags=[(True, False, 'time-limit'), (True, False, 'time-limit'), (True, False, 'time-limit'), (True, False, 'time-limit'), (True, False, 'time-... | every episode length 500, truncated=True, terminated=False, cause=time-limit |
| 24 | T-C2_random_policy_anchor | prediction | PASS | mean=22.01 std=11.21 n=500 | mean ∈ [19, 26], std ∈ [8, 17] (published random ≈ 22 ± 12) |
| 25 | T-C3_initial_policy | prediction | PASS | frac_a0=0.520 mean_entropy=0.693 | frac ∈ [0.4, 0.6], entropy ≥ 0.6 nats (max 0.693) |
| 26 | T-C4_same_seed_same_traj | exact | PASS | same rewards=True actions=True states=True | identical obs/actions/rewards for the first episode |
| 27 | T-C4_10step_trace | exact | PASS | wrote results/validation/cartpole_10step_trace.json (10 steps) | table of (t, obs, probs, action, reward, G_t, logπ) |
| 28 | T-C4_three_updates_bit_identical | exact | PASS | two independent 3-update runs with seed=7 | bit-identical parameters on CPU |
| 29 | T-C4_omitting_zero_grad_changes_params | exact | PASS | dirty p=0.800000011920929 clean p=0.6000000238418579 | parameters differ when zero_grad is omitted |
| 30 | T-C5_bookkeeping | exact | PASS | len(logp)=len(R)=len(G)=17; buf cleared=True | equal lengths at every update; buffers empty after the step |
| 31 | T-C7_sum_loss_corr | prediction | PASS | corr(length, //g//_sum)=0.771 | r > 0.7 under the summed (per-episode) loss |
| 32 | T-C7_mean_loss_corr | prediction | PASS | corr(length, //g//_mean)=-0.262 | weak under mean-over-timesteps (typically /r/ < 0.7) |
| 33 | T-P1_actions_in_2_3 | exact | PASS | unique env actions=[2, 3] unique idx=[0, 1] | env actions ⊆ {2,3}, indices ⊆ {0,1} |
| 34 | T-P1_forced_logits | exact | PASS | recorded=[2] n=30 | env action 2 every step |
| 35 | T-P2_paddle_control | exact | PASS | visible=True travel_2=28.5px travel_3=77.5px delta2=-11.00 delta3=63.10 opposite=True cols=(68, 74) | centroid moves ≥5 px one way then ≥5 px the opposite way; paddle visible |
| 36 | T-P3_copy_safe | exact | PASS | obs unchanged=True | input obs unchanged after preprocess |
| 37 | T-P3_shape_and_values | exact | PASS | shape=(80, 80) unique=[0.0, 1.0] flat=6400 | (80,80) → 6400, values ⊆ {0,1} |
| 38 | T-P3_diff_and_first_zero | exact | PASS | first_zero=True diff_vals=[-1, 0, 1] min_nz=6 frac_nonzero_diff_after20=0.96 | first diff all 0; diffs ⊆ {-1,0,1}; min nonzero pixels > 0; ≥50% diffs nonzero after step 20 |
| 39 | T-P4_initial_policy | prediction | PASS | frac_idx0=0.502 mean_entropy=0.693 n_actions=20383 | frac ∈ [0.4, 0.6], entropy ≥ 0.6 |
| 40 | T-P5_random_anchor | prediction | NOT HELD | rewards=[-20.0, -21.0, -21.0, -19.0, -21.0, -21.0, -21.0, -21.0, -21.0, -20.0, -19.0, -20.0, -20.0, -21.0, -21.0, -18.0, -21.0, -20.0, -21.0, -21.0] mean=-20.40 | integer rewards in [-21,-19], mean ∈ [-21, -19.8] (published random ≈ -20.7) |
| 41 | T-P9_scripted_ball_tracker | prediction | PASS | mean=9.00 rewards=[9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0] | mean > -18 over 10 episodes (clearly beats random ≈ -20.7) |
| 42 | T-P10_resume | exact | PASS | n1=3 n2=5 ep_done=5 dup=False params_changed=True rc=0 | episode counter continues to 5, no duplicated CSV rows, params not reset |
| 43 | T-P6_worker_learner_logp | exact | PASS | max /logπ_learner − logπ_worker/=0.00e+00 T=945 | < 1e-4 |
| 44 | T-P7_point_boundary_synthetic | exact | PASS | G=[-0.9801, -0.99, -1.0, 0.9801, 0.99, 1.0, -0.99, -1.0] | returns at ±1 equal ±1; earlier steps of a point ignore later points |
| 45 | T-P7_point_boundary_real_episode | exact | PASS | point_steps=[(63, -1.0, -1.0), (98, -1.0, -1.0), (133, -1.0, -1.0), (168, -1.0, -1.0), (203, -1.0, -1.0), (238, -1.0, -1.0)]... n_points=21 | G_t = r_t at every point-scoring step |
| 46 | T-P8_determinism_default | prediction | PASS | same_hashes=True n=50 (default sticky-action env) | identical first-50-frame hashes if ALE is deterministic; else document and skip |
