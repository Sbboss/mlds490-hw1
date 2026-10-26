# Study guide — MLDS 490 HW1

This is the explanation of the code you would give a TA. Read it with `PLAN.md` (the spec) and `DECISIONS.md` (every place the implementation left the pre-registered plan).

## What each module does

**`shuttle/`** solves a finite-state MDP exactly. The state is the number of customers waiting, `s = 0, 1, …, 200`. Each period you either hold or dispatch. Dispatch costs `cf = 100` and empties the queue. Holding is free up front, but every customer still waiting after the decision costs `ch = 2`. Then `A ~ Uniform{1,…,5}` new customers arrive and the next state is `min(200, leftover + A)`. Customers above 200 disappear for free. The discount is `γ = 0.95`. Costs are minimized. A tie (within `1e-9`) is broken toward hold, so the policy dispatches only when dispatch is strictly better.

"Enumeration" in the write-up is backward induction over `T = 500` stages, not a literal `2^500` tree. A literal `2^12` enumeration with deterministic arrivals is one of the tests, and it matches the DP.

**`pg/common.py`** holds the pieces every trainer shares: seeding, the backward return `G_t = r_t + γ G_{t+1}`, a 100-episode moving average that is NaN until episode 100, and CSV logs that flush every row.

**`pg/models.py`** is a 2-action softmax policy. CartPole is `4 → 128 → 2`. Pong is `6400 → 200 → 2`. The last layer is initialized at `0.01 ×` the default weights and bias 0, so the network starts nearly uniform (entropy about `ln 2 = 0.693`). There is a small CNN with the same last-layer trick, used only if the Pong ladder reaches `--arch cnn`. The value network is the same trunk with one scalar output.

**`pg/reinforce.py`** is the estimator

```
L = −(1/B) Σ_episodes Σ_t  w_t · log π(a_t|s_t) · (G_t − b_t)
```

The sum is over time, then the average is over episodes. A longer episode therefore produces a larger gradient. That is deliberate: T-C7 shows the correlation between episode length and gradient norm is high for this sum and weak if you average over timesteps instead. `b_t` is detached inside the loss, so a value network cannot get a gradient from the policy objective.

**`pg/train_cartpole.py`** runs one (or a few) episodes, computes `G`, and takes one Adam step. It records why the episode ended: the cart left `|x| > 2.4`, the pole passed 12°, or the 500-step cap.

**`pg/pong_utils.py` and `pg/train_pong.py`** do not use Gymnasium vector environments. Those auto-reset in a way that can glue the next episode onto the current return. Instead, a spawn pool of CPU actors each owns one `ALE/Pong-v5`. A task is `(weights, seed)`. The actor plays one full episode and returns states, action *indices*, rewards, and its own log-probabilities. The learner recomputes `log π` on the GPU with the same weights (T-P6 checks they match) and takes one on-policy Adam step. The network's index 0 is ALE action 2 (RIGHT) and index 1 is action 3 (LEFT). The state is the difference of two preprocessed frames, stored as int8 in `{−1, 0, 1}`. The first difference is all zeros because there is no previous frame.

**`pg/evaluate.py`** rolls out the *final* checkpoint, not the best one, for 500 episodes, sampling from the stochastic policy. A second 500 uses the greedy action. Five checkpoints from the last 20% of training are rolled out for 50 episodes each, so a lucky final save cannot hide.

## Why the design looks like this

**Reward-to-go, and `w_t = 1` by default.** The gradient of the discounted objective `E[Σ_t γ^t r_t]` multiplies the step-`t` term by `γ^t`. At `γ = 0.95` that factor is about 0.006 by step 100, so the end of a long CartPole episode barely moves the parameters. The unweighted estimator (`w_t = 1`) is the usual "basic REINFORCE" and optimizes a slightly different, state-distribution objective. T-R2 shows the two gradients are not the same number: with `γ = 0.5` the unweighted step-2 term is off by a factor of two. CartPole config C3 turns the `γ^t` weight on so the report can show what that does to learning. Say this out loud if asked: the primary runs do **not** optimize exactly the discounted objective written in the assignment.

**Pong returns reset when a point is scored.** A point (`r = ±1`) ends a rally that is almost its own game. Without the reset, a later point leaks credit into earlier rallies. T-R1 computes both versions on `[0, 0, −1, 0, 0, 1]`. With the reset, `G` at the −1 step is −1. Without it, that same step is about −0.03 because the later +1 almost cancels it. The flag is `--pong_reset_at_point`, default on.

**No return standardization in the primary runs.** Subtracting the batch mean and dividing by the batch standard deviation is already a baseline. Using it in Part 2 would make the Part 2 vs Part 3 comparison meaningless. `--standardize 1` exists and is labeled if a Pong ladder rung needs it.

**Baselines (Part 3).** Both are action-independent, which is what keeps the policy gradient unbiased.

- Learned `b(s) = V_φ(s)`. A second network, its own Adam at `1e-3`, mean-squared error against the same `G_t` the policy uses. The advantage is `G_t − V_φ(s_t).detach()`. The value update happens *after* the policy step. On Pong the value pass is minibatches of 4096.
- Constant EMA. `b ← 0.9 b + 0.1 · mean(G)` using only *previous* batches. The batch in the update still sees the old `b`. The first batch sees `b = 0` because there is no history. If the EMA used the current episode's returns it would depend on the actions just sampled and the estimator would be biased. T-B4 checks the "previous only" rule on a recorded sequence.

**Adam hides gradient scale.** On the first update, every parameter with a nonzero gradient moves by exactly the learning rate (T-R4, bias-corrected Adam). A baseline that cuts the gradient variance by 5× does not take a 5× bigger step. The gain shows up in the gradient covariance (T-B5), not as a larger step. Across the five CartPole seeds it did **not** show up as a smaller standard deviation: the value baseline's five evaluation means are 305 ± 262, against 103 ± 118 with no baseline. Both have a seed that collapses. Do not tell a TA that the baseline made the runs more stable. Say that it raised the average and that the seed spread is still huge.

**CartPole `lr = 0.01` is the primary number even when it is a bad learning rate.** The assignment asks for 0.01. Config C0 is that run. It is unstable: some seeds collapse to episodes of length ~9. The plan said to keep that result and add labeled variants if the five-seed average MA100 ended under 100. It did (about 88). C7 uses 8 episodes per update and C8 uses `lr = 0.003`. Those are not the headline curves.

**Final policy, not best policy.** Evaluation seeds start at 10000 and are not the training seeds. Checkpoint spread is reported because the last save is a noisy draw. On C0 seed 0 the five late checkpoints do not agree (one 50-episode mean is above 250 and another is near 60).

## What the validation tests prove

| test | what it would catch |
|---|---|
| T-R1 | A forward return (the classic `[1, 1.5, 1.75]` bug) or a Pong return that leaks across points. |
| T-R2 | An implementation of REINFORCE that does not match the enumerated gradient on a 2-step MDP, including the `γ^t` vs `w_t = 1` distinction. N = 200,000. |
| T-R3 | The softmax-bandit closed form `(+0.25, −0.25)` at `θ = 0`, `r = (1, 0)`. |
| T-R4 | An optimizer that is not bias-corrected Adam. |
| T-R5 | A standardize-by-zero NaN on a one-step episode. |
| T-C1 | Treating the 500-step cap as a true terminal. With thresholds removed, every episode is length 500, `truncated`, cause `time-limit`. |
| T-C2 / T-P5 | A broken environment: an untrained CartPole should score about 22, an untrained Pong about −20.7. |
| T-C3 / T-P4 | A peaked initial policy. Fraction of action 0 must be near 1/2 and entropy near 0.69 before any long run. |
| T-C4 | A seed that does not replay, or a forgotten `zero_grad` (the test checks that omitting it *does* change the weights). |
| T-C6 | The sign of the reward. Negating `r` should make the agent fail fast (last-50 length under 15). |
| T-C7 | Averaging the loss over timesteps, which washes out the length signal. |
| T-P1 | Sending NOOP/FIRE, or swapping the index→action map. Forced logits `[100, −100]` must send action 2 every step. |
| T-P2 | A preprocess that erases the paddle. Forcing action 2 then 3 moves the right paddle by ≥ 5 px each way. |
| T-P3 | The assignment `preprocess` mutating the caller's frame. We copy first. Also checks the first difference is 0 and later differences are in `{−1,0,1}`. |
| T-P6 | Actors and the learner using different weights (silently off-policy). |
| T-P7 | Point-boundary returns on a real episode: `G_t = r_t` at every scoring step. |
| T-P9 | The whole frame pipeline. A hand-written intercept controller has to beat −18. A naive chase scored −21; see D19. |
| T-P10 | `--resume` restarting from episode 0 or duplicating CSV rows. |
| T-B1 | A baseline that changes the mean gradient. With +10 added to every reward, `b = 0`, `b = J`, and `b = V^π` agree within 4 standard errors, and the trace of the covariance drops. |
| T-B2 | The optimal constant baseline of a 2-action bandit. At `π = (0.5, 0.5)`, `b* = J`. At a skewed `π` the rarer action is weighted about 9×. |
| T-B3 | A value head that leaks into the policy gradient, or a value-loss backward that overwrites the policy gradient. |
| T-B5 | The practical claim "a baseline reduces variance." Same trajectories, three baselines, cosine of the mean gradients (should be ~1) and `tr(Cov)` ratios. `R² < 0` means the value fit is worse than predicting the mean; that is reported, not hidden. |

Prediction rows are allowed to fail. The ones that did not hold are written as NOT HELD with the measured number. Do not describe them as passes.

## Five questions a TA is likely to ask

**1. Why is the optimal shuttle policy "dispatch if and only if `s ≥ 13`"?**
Dispatch costs 100 and clears the queue. Holding costs `2s` now and leaves you with a larger state. Below 13 the immediate holding cost plus the discounted future is still cheaper than paying 100. At 13 and above it is not. `V` is flat on `s = 13, 14, 15` because all three dispatch, pay 100, and land on the same next-state distribution (arrivals 1…5). The slope `dV/ds` at 50, 100, and 199 is about 10.6, 16.9, and 24.9, all below the bound `ch/(1−γ) = 40`.

**2. What happens if the cap is 10 instead of 200?**
Overflow customers vanish for free, so the problem becomes artificially cheap (`V(0) ≈ 356` instead of `577`) and the optimal policy never dispatches. Caps 30, 200, 300, and 400 give the same value and the same policy on the states they share. Charging `ch` on the post-arrival queue adds about 120 to every `V` and cannot change the action, because the arrival is not under your control.

**3. Is your REINFORCE gradient the gradient of `E[Σ γ^t r_t]`?**
Only when `--gamma_weight on`. The primary runs use `w_t = 1`. T-R2 is the proof that those two estimators differ, and that the weighted one matches automatic differentiation of the true objective.

**4. How do you know the Pong actors are on-policy?**
Every episode in a batch is played with one frozen weight vector. The learner recomputes `log π(a|s)` and T-P6 checks it against the actor's log-probability to `< 1e-4`. The actor never touches CUDA (`CUDA_VISIBLE_DEVICES=""` inside the worker). Actions sent to ALE are only 2 and 3 (T-P1).

**5. Why did a baseline not simply "work better" by taking bigger steps?**
Adam normalizes the step (T-R4). The evidence for the baseline is the variance of the gradient on a frozen policy (T-B5), not the size of a single update. On CartPole the five-seed evaluation means are 103 ± 118 with no baseline and 305 ± 262 with the value baseline: the average went up and the spread did not shrink. On Pong, one seed, the final policies score −2.62 ± 7.24 with no baseline and +10.50 ± 4.75 with the value baseline (500 episodes each). Also say the failure modes out loud: CartPole at `lr = 0.01` with one episode per update collapses on some seeds; `γ^t` weighting (C3) removed the late-episode signal and did not learn; `γ = 0` was supposed to stay near the random policy and instead collapsed to length ~9; a Pong moving average has to rise by more than about 0.3 from −20.7 before that is distinguishable from noise.
