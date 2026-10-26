# MLDS 490 – HW#1 – Execution Plan for Cursor

Prepared Sun 2026-10-25 (Chicago time). **Deadline: Mon 2026-10-26, 11:59 pm CT (Canvas, PDF report + GitHub link).**
Student: Shiva (Northwestern MSAI). This plan was written by Claude after a multi-reviewer edge-case analysis. **The student will not come back to answer implementation questions, so this file contains everything you need. Make sensible decisions yourself, log them in `DECISIONS.md`, and keep going.**

---------------------------------------------------------------------------------------------------

## 0. Ground rules (read first)

1. **Honesty over polish.** Report what the code actually produced. Never invent numbers, never fill a results table with the "expected" values from this plan, never cherry-pick a seed without saying so. If Pong does not learn, say so and show the diagnostics.
2. **The professor allows AI, but grades the thinking.** 45 of 100 points are "edge cases + tests that give quantitative evidence" (beyond unit tests/format checks). The student must be able to explain everything, so:
   - comment *why* a line exists, not only what it does;
   - write `STUDY_GUIDE.md` (2–3 pages): what each module does, why each design decision was made, what each validation test proves, and the five questions a TA is most likely to ask with short answers.
3. **Tests gate long runs.** Do not launch any long training until the validation tests for that part pass (§7).
4. **Only stop and write `BLOCKERS.md` for**: (a) SSH needs a password/2-factor prompt you cannot answer (see §2.2), (b) GitHub credentials, (c) a CUDA/driver problem not fixed by the ladder in §2.4. For anything else decide, log, continue. If one part is blocked, continue with the other parts.
5. Always **quote paths**: the local folder name contains `#`.
6. Never write passwords, tokens or keys into any file or commit.
7. Long jobs (> 2 min) run inside `tmux` (or `nohup setsid`) on deepdish; poll the log files; never block the terminal on a multi-hour command.

---------------------------------------------------------------------------------------------------

## 1. Assignment requirements (copied from the PDF; the PDF may not be in the folder)

**Submit on Canvas a PDF report containing**: GitHub username + repository name + link; instructions for running the code; labeled/formatted plots for each part; responses to all discussion questions. Missing items = significant deductions. All code used must be in the GitHub repo.

**For EVERY part the report must also contain**:
- *Edge cases*: edge cases of the problem (not the generic ones), how each was handled, and how you ensured the handling is correct.
- *Tests validating correctness*: "Beyond asking AI, how do you know your algorithm works?" Change parameters/structure and check the output matches prediction; reduce stochasticity so the process is nearly deterministic; isolate the first 10 steps and look at states, values, rewards, policy at each step. Give **directions for running the validation/test experiments** and what each test does and how it validates. Provide **quantitative and convincing evidence**. Not unit tests / data-format checks.
- Final question to answer once: *"Given an additional 3 months, what other validation tests and experiments would you perform?"*

**Part 1 – Value and Policy Iteration (30 pts: 15 edge cases/tests, 15 plots).** Shuttle dispatch MDP: K=15, cf=100, ch=2 (per customer left waiting, incurred after dispatch), arrivals A_t ~ Unif{1..5}, at most 200 customers, γ=0.95. (a) Enumeration, T=500: plot optimal V at time 0 vs customers waiting. (b) Value iteration (T=∞): plot V* vs customers. (c) Policy iteration: plot optimal policy vs customers. State any assumption.

**Part 2 – Policy gradient (40 pts: 20 / 20).** PyTorch REINFORCE, Adam, lr 0.01 (another lr allowed if you mention it). Objective = discounted reward over an episode. (a) CartPole-v1, γ=0.95. (b) Pong-v5, γ=0.99, frames as input; network chooses only between actions 2 (RIGHT) and 3 (LEFT) of the 6 actions; the given `preprocess()` may be used. For both: plot episode reward vs training episodes with a 100-episode moving average overlaid; after training roll out **500 episodes**; histogram of the 500 rewards; report mean and std. Must use `CartPole-v1` and `Pong-v5`. On deepdish use a single GPU card.

**Part 3 – Baseline (20 pts: 10 / 10).** Repeat Part 2 with a constant or non-constant baseline of your choice; explain which and how it was implemented.

**Report formatting/readability: 10 pts.**

Given preprocess function:
```python
def preprocess(image):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array """
    image = image[35:195]            # crop
    image = image[::2, ::2, 0]       # downsample by factor of 2
    image[image == 144] = 0          # erase background (background type 1)
    image[image == 109] = 0          # erase background (background type 2)
    image[image != 0] = 1            # everything else (paddles, ball) just set to 1
    return np.reshape(image.astype(np.float32).ravel(), [80, 80])
```
**Warning:** the slices are views, so this mutates the caller's frame. Call `image = image.copy()` first (test T-P3).

---------------------------------------------------------------------------------------------------

## 2. Environment and access

### 2.1 Machines
- **Local (Mac)**: project root `"/Users/sbboss/NU/Q4/MLDS-490/HW#1/"`. Develop here. Use a local venv (`.venv`) for unit tests that need only CPU.
- **Remote (deepdish)**: `ssh hia9393@mlds-deepdish5.ads.northwestern.edu` — use it for GPU testing and all long training. Hardware reported by the student: 3 GPUs (index 0: RTX 5060 Ti 16 GB, index 1: RTX PRO 5000 Blackwell 48 GB, index 2: RTX 5060 Ti 16 GB), **48 CPU cores, 93 GB RAM, 95 GB swap, shared node** (hostname prompt `mlds-deepdish5`).
- **HARD RULE from the assignment: on deepdish use a single GPU card.** Set `CUDA_VISIBLE_DEVICES=<one index>` for every process, and verify with `nvidia-smi` that only that card is in use. All jobs go on that one card (the networks are tiny).
- Be a good citizen on the shared node: check `uptime`, `nvidia-smi`, `who` first; run jobs with `nice -n 10`; **never use more than 32 CPU worker processes in total across all your concurrent jobs**.

### 2.2 SSH
First test (non-interactive):
```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 hia9393@mlds-deepdish5.ads.northwestern.edu 'hostname; nproc; uptime'
```
If this asks for a password or a 2-factor prompt that you cannot answer, **STOP and ask the student to do this once in their own terminal**, then continue:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""        # skip if a key exists
ssh-copy-id hia9393@mlds-deepdish5.ads.northwestern.edu
```
Optionally suggest (do not edit the student's `~/.ssh/config` yourself) this block so one authenticated connection is reused:
```
Host deepdish
  HostName mlds-deepdish5.ads.northwestern.edu
  User hia9393
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
  ServerAliveInterval 30
```
Remote workspace: `~/mlds490_hw1` (no `#` in the remote path).

### 2.3 Sync scripts (create `scripts/sync_up.sh`, `scripts/sync_down.sh`, quote the local path)
```bash
# up: code only
rsync -az --delete --exclude '.git' --exclude 'results' --exclude '.venv' --exclude '__pycache__' \
  "/Users/sbboss/NU/Q4/MLDS-490/HW#1/" hia9393@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/
# down: results (curves, small checkpoints, eval csv, figures); skip huge files if any > 200MB
rsync -az hia9393@mlds-deepdish5.ads.northwestern.edu:~/mlds490_hw1/results/ "/Users/sbboss/NU/Q4/MLDS-490/HW#1/results/"
```

### 2.4 Remote setup (do once; record exact versions in `results/env_versions.txt`)
```bash
ssh hia9393@mlds-deepdish5.ads.northwestern.edu 'nvidia-smi; nproc; free -g; df -h ~; which tmux python3 conda git; python3 --version'
python3 -m venv ~/venvs/mlds490 && source ~/venvs/mlds490/bin/activate && pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128      # Blackwell GPUs need a CUDA >= 12.8 build
pip install -r requirements.txt                                            # numpy matplotlib pandas pytest tqdm "gymnasium[atari]" ale-py
python -c "import torch;print(torch.__version__,torch.cuda.is_available());x=torch.randn(512,512,device='cuda');print((x@x).sum().item())"
```
**CUDA ladder if the matmul fails with "no kernel image is available" / unsupported capability**: (1) newest stable index (`cu129`, `cu130` if it exists), (2) nightly (`--pre ... /whl/nightly/cu128`), (3) run the policy-gradient update on CPU (the networks are small; only Pong's 6400-input layers benefit from GPU) and document it in `DECISIONS.md`. Do not burn more than 30 minutes on this.
Pong needs `import ale_py; gymnasium.register_envs(ale_py)` then `gym.make("ALE/Pong-v5")`. With ale-py ≥ 0.9 the ROM is bundled (no AutoROM). Verify the env has **6 actions** and `obs.shape == (210,160,3)`.

---------------------------------------------------------------------------------------------------

## 3. Part 1 is DONE – integrate, do not rewrite

The folder `shuttle/` (from `shuttle_part1.zip`, delivered by Claude) contains `shuttle_mdp.py`, `run_shuttle.py`, `validation.py`, `test_shuttle.py`, `README.md`, `figures/*.png`, `validation_results.md/.json`, `results.json`. If it is not already in the project folder, ask the student to copy it there (it is the only item you cannot recreate). Re-run `python run_shuttle.py && python validation.py && pytest -q test_shuttle.py` locally and confirm you reproduce these numbers (tolerance 1e-6):

| quantity | value |
|---|---|
| V₀(0) at T=500 / V*(0) | 576.6702 |
| V*(15) | 676.6702 (= V*(0)+100) |
| V*(200) | 3821.586 |
| optimal policy | dispatch iff s ≥ 13 (identical for T=500 recursion, VI, PI) |
| VI iterations (ε=1e-10) / PI iterations (from always-hold) | 580 / 6 |
| max abs V₀(T=500) − V*(VI) | 4.9e-9 |
| Monte Carlo of optimal policy from s=0 | 576.56 ± 0.16 (SE) |
| average cost per period g / dispatch frequency | 32.34 / 0.2133 |
| cap=10 artifact | V(0)=356.04, never dispatch |
| validation | 38 checks: all 29 `exact` pass; 9 `prediction` checks, 1 not held (V is not convex; min second difference −3.78) |

Model assumptions to state in the report: state = customers waiting 0..200; cost = cf·[dispatch] + ch·(customers left waiting after the decision, before new arrivals); next state = min(200, leftover + A); arrivals above 200 are discarded for free; costs are minimised; ties → hold (dispatch only if better by > 1e-9); float64. "Enumeration" = exhaustive backward induction over all T=500 epochs (literal 2^500 enumeration is impossible; a literal 2^12 enumeration with deterministic arrivals is used as a test).

Facts worth putting in the Part 1 write-up (all measured): V is flat on s = 13..15 (dispatch costs 100 and empties the queue); cap 200/300/400/30 give identical V and policy; cap 10 makes the problem artificially cheap (overflow customers vanish for free) and flips the policy to never-dispatch; charging ch on the post-arrival queue adds a constant 120 to every V so the policy cannot change; slopes dV/ds at s = 50, 100, 199 are 10.6, 16.9, 24.9, below the bound ch/(1−γ)=40.

In the report, Part 1 needs: assumptions, the three figures, a table of the validation checks (copy `validation_results.md`, regenerated), how to run, and the edge-case discussion.

---------------------------------------------------------------------------------------------------

## 4. Design decisions (pre-registered; log any deviation in `DECISIONS.md`)

| # | Decision | Reason |
|---|---|---|
| D1 | Policy-gradient estimator: `loss = −(1/B) Σ_episodes Σ_t  w_t · logπ(a_t|s_t) · (G_t − b_t)`, with **reward-to-go** `G_t = Σ_{k≥t} γ^{k−t} r_k`. Default weight `w_t = 1` (`--gamma_weight off`), the standard "basic REINFORCE". Option `--gamma_weight on` uses `w_t = γ^t`, the exact gradient of the stated discounted objective. | The exact discounted gradient needs γ^t, which at γ=0.95 is 0.006 by step 100, nearly erasing the signal from late steps of long CartPole episodes. The test in T-R2 proves the difference numerically; a CartPole ablation (C3) shows its effect. State this plainly in the report; the unweighted estimator optimises a slightly different (state-distribution) objective. |
| D2 | CartPole γ=0.95. Pong γ=0.99 with **returns reset at point boundaries** (when r≠0 the running return is reset to 0 before accumulating backward). Flag `--pong_reset_at_point {1,0}`, default 1. | Each point is a nearly independent subgame; without the reset, later points leak credit into earlier ones. Disclose it as a modelling choice. |
| D3 | CartPole: Adam lr **0.01** (as specified), **1 episode per update**, 2000 episodes, MLP 4→128→2. Pong: batch of **16 episodes per update** (on-policy: all 16 use the same weights), lr chosen by the sweep in §8, MLP 6400→200→2 (Karpathy-style) with `--arch cnn` fallback. | Faithful basic REINFORCE where cheap; batching is needed for Pong's variance and throughput. State the deviation and the sweep evidence in the report. |
| D4 | **No return standardization** in the primary runs (`--standardize` exists only for clearly labelled variants). | Standardizing returns is already an implicit mean baseline; it would confound the Part 2 vs Part 3 comparison. |
| D5 | Part 3 baselines: **(primary) learned state-value baseline** `b(s_t)=V_φ(s_t)`, a separate network with its own optimizer, trained with MSE to the same `G_t`; advantage `G_t − V_φ(s_t).detach()`, no advantage standardization. **(secondary) constant baseline**: exponential moving average of the mean return from *previous* episodes only (`b ← 0.9 b + 0.1·mean(G)` updated after each update). | Both are action-independent (unbiased). The EMA must not use the current episode's own returns, otherwise it depends on the sampled actions. |
| D6 | Final evaluation = **500 episodes with the final (not best) policy, sampling actions from the policy** (the stochastic policy that was trained). Also report greedy (argmax) rollouts as a secondary row. Evaluation seeds differ from training seeds. | Decide the rule before seeing results; avoids the "final-policy lottery". Additionally evaluate 5 checkpoints from the last 20% of training (50 episodes each) and report the spread. |
| D7 | Plot conventions: x = training episode; thin light line = per-episode **undiscounted** reward; bold line = trailing 100-episode mean, **defined from episode 100 onward** (no partial windows; array has NaN before). CartPole additionally gets a dashed line at 500. Pong y-range [−21, 21]. | The discounted return at γ=0.95 is capped near 20, so only undiscounted reward is interpretable; also store discounted returns in the CSV. |
| D8 | Seeds: CartPole 5 seeds (0–4) per main config; Pong 1 seed (0) per config, a second seed only if time allows. | Compute budget. |
| D9 | Env settings: `CartPole-v1` and `ALE/Pong-v5` **with default settings** (frameskip 4, sticky-action probability 0.25, minimal 6-action set). Do not change them for training or evaluation. Deterministic variants (`repeat_action_probability=0`) are for tests only. | Assignment requires the specified versions. |
| D10 | CartPole returns are plain Monte-Carlo with no bootstrapping, even at the 500-step truncation. Log `terminated`/`truncated` and the termination cause (|x|>2.4, |θ|>12°, or time-limit) per episode. | Keeps the estimator unbiased for the stated (truncated-horizon) objective; report the known limitation that V(s) near the cap is slightly low. |

---------------------------------------------------------------------------------------------------

## 5. Repository layout

```
HW#1/
  PLAN.md  DECISIONS.md  BLOCKERS.md(if any)  TODO_USER.md  STUDY_GUIDE.md  README.md  requirements.txt  .gitignore
  scripts/   sync_up.sh sync_down.sh remote_setup.sh bench_pong.py run_cartpole_all.sh run_pong.sh
  shuttle/   (Part 1, already done)
  pg/
    common.py          # seeding, discount_returns, moving_average, plotting style, csv logging
    models.py          # MLPPolicy, CNNPolicy, ValueNet
    reinforce.py       # loss, baselines (None/EMA/Value), gradient helpers
    pong_utils.py      # preprocess (copy-safe), ACTION_MAP, frame-diff state builder, episode worker
    train_cartpole.py  # CLI
    train_pong.py      # CLI (worker pool + GPU learner, checkpoint/resume)
    evaluate.py        # 500-episode rollouts, greedy+sampled, checkpoint spread, termination causes
    plot_results.py    # all figures
  validation/
    test_returns.py test_gradient_exact.py test_baseline.py test_cartpole.py test_pong.py
    run_validation.py  # runs the experiments below, writes validation_pg_results.md/.json
  results/   (git-ignored except small csv/json/png)  
  figures/   report/   (report.md → report.pdf)
```
Reuse the plot style from `shuttle/run_shuttle.py` (rcParams block; colors blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`; light surface `#fcfcfb`; thin lines; top/right spines off; direct labels; legend whenever ≥ 2 series).

---------------------------------------------------------------------------------------------------

## 6. Implementation specification

### 6.1 `common.py`
- `discount_returns(rewards, gamma, reset_on_nonzero=False) -> np.ndarray(float64)`: **backward** loop `G = r + gamma*G`; if `reset_on_nonzero` and `r != 0`, set `G = 0` *before* adding `r`... i.e. `if r != 0: G = 0.0; G = r + gamma*G`. Output same length as input.
- `moving_average(x, w=100)`: trailing mean, NaN for indices < w−1.
- `set_seed(seed)`: python, numpy, torch (+cuda). Per-episode generators: `torch.Generator().manual_seed(seed)`.
- CSV logging helper (flush every write).

### 6.2 Models (`models.py`)
- `MLPPolicy(in_dim, hidden, n_actions=2)`: Linear→ReLU→Linear. **Last layer init**: weights × 0.01, bias 0, so the initial policy is ≈ uniform (entropy ≈ 0.69). Output = logits; use `torch.distributions.Categorical(logits=...)`.
- `CNNPolicy` (fallback only): input 1×80×80 frame-diff; Conv(1,16,k=8,s=4)→ReLU→Conv(16,32,k=4,s=2)→ReLU→flatten→Linear(·,256)→ReLU→Linear(256,2), same last-layer init.
- `ValueNet`: same trunk as the policy, 1 scalar output.

### 6.3 `reinforce.py`
- `reinforce_loss(logp, G, baseline_values=None, weights=None)`; `baseline_values` must be **detached** inside the function. Return a scalar summed over timesteps of each episode and averaged over episodes in the batch. Assert `len(logp)==len(G)`, assert `torch.isfinite(loss)`.
- `EMABaseline`, `ValueBaseline` (own `torch.optim.Adam`, lr 1e-3 for CartPole... use 1e-3 for both games, state it; Pong value net updates with minibatches of 4096 states, 1 pass over the batch per update). The value net is updated **after** the policy update, using the same batch.
- Gradient clipping is **off by default** (`--grad_clip 0`); an option exists for ablations.

### 6.4 CartPole (`train_cartpole.py`)
Flags: `--seed --episodes 2000 --lr 0.01 --gamma 0.95 --baseline {none,ema,value} --gamma_weight {off,on} --episodes_per_update 1 --reward_sign 1 --standardize 0 --out DIR`.
Per episode: sample actions with a per-episode generator, store logπ with grad (or recompute in batch), rewards, states; at the end compute G, loss, `zero_grad → backward → step`. Write `curve.csv` with columns `episode, reward, length, discounted_return, terminated, truncated, cause, entropy, grad_norm, loss`. Save checkpoints every 100 episodes (`ckpt_{ep}.pt`) and `final.pt`. Runs on CPU (fast); run seeds in parallel processes (≤ 8).

### 6.5 Pong (`train_pong.py`, `pong_utils.py`) – batched actors, GPU learner
**State**: `x_t = pre(frame_t) − pre(frame_{t−1})` (float → store as **int8** in {−1,0,1}), 6400-vector. At episode start set the previous frame equal to the first frame, so the first difference is all zeros (document; test it). `pre()` is the provided function applied to a **copy**.
**Actions**: `ACTION_MAP = (2, 3)`; network index 0 → env action 2 (RIGHT), index 1 → env action 3 (LEFT). Store the **index** for `log_prob`; send `ACTION_MAP[idx]` to `env.step`.
**Parallelism (do NOT use gymnasium vector envs**: their auto-reset semantics changed in 1.x and silently corrupt trajectories). Use a persistent `multiprocessing` pool with the **spawn** context: each worker creates its own `ALE/Pong-v5` env once, `torch.set_num_threads(1)`, runs on CPU, never touches CUDA (set `CUDA_VISIBLE_DEVICES=""` in the worker initializer). A task = `(weights_numpy, seed)`; the worker plays **one full episode** with the given weights and returns `(states int8 [T,6400], actions uint8 [T], rewards float32 [T], logp float32 [T])`. The learner submits `episodes_per_update` tasks (default 16 = number of episodes per batch, ≥ workers), collects them, moves the batch to the GPU, **recomputes logπ with the same weights**, computes returns (D2), baseline, loss, Adam step, then sends the new weights. Because all episodes in a batch use identical weights this is exactly on-policy REINFORCE. Seeds: `env.reset(seed=base+episode_idx)` only on the first reset of the env per episode; actions via a per-episode `torch.Generator`.
**Logging** (every update, flushed): `update, episodes_done, env_steps, batch_mean_reward, batch_mean_len, entropy, frac_idx0, grad_norm, loss, ma100, wall_s, steps_per_s`; per-episode `episode_rewards.csv` (`episode, reward, length, points_won, points_lost`). 
**Checkpoint/resume**: `latest.pt` every update (policy, optimizer, value net + optimizer, EMA, episode count, RNG states) and a numbered checkpoint every 200 episodes. `--resume` must continue seamlessly (test by killing and resuming a short run). Flags: `--workers 32 --episodes_per_update 16 --lr --max_episodes 25000 --max_hours 4 --arch mlp --hidden 200 --baseline {none,ema,value} --standardize 0 --pong_reset_at_point 1 --seed 0 --out DIR`.
**Memory**: 16 episodes × ~3000 steps × 6400 int8 ≈ 300 MB per batch; fine.
**Pong evaluation** (`evaluate.py`): reuse the worker pool with the final weights, 500 episodes, sampled and greedy; also last-20%-checkpoint spread.

### 6.6 Outputs per run
`results/<env>/<config>/seed<k>/{curve.csv, final.pt (or latest.pt), eval_sampled.csv, eval_greedy.csv, eval_stats.json, checkpoint_spread.json, log.txt}`; `eval_stats.json` has mean, std, min, max, n, and for CartPole the fraction of episodes ending in {x-limit, angle, time-limit}.

---------------------------------------------------------------------------------------------------

## 7. Validation suite (the 45 "edge cases + tests" points). Implement all; run; report **measured** values.

Each test result record: `name, kind (exact|prediction), passed, measured, expected, why` (same format as `shuttle/validation.py`). `exact` = identity/hard assertion (must pass); `prediction` = expectation reported honestly even if it fails. Write `validation_pg_results.md` and include its tables in the report, grouped by part.

### 7.1 Estimator and returns (Part 2, run first, CPU, seconds)
- **T-R1 returns, hand-computed (exact)**: `[1,1,1]`, γ=0.5 → `[1.75, 1.5, 1.0]`. CartPole `[1,1,1]`, γ=0.95 → `[2.8525, 1.95, 1.0]`. Pong with point reset: rewards `[0,0,-1,0,0,1]`, γ=0.99 → `[-0.9801, -0.99, -1, 0.9801, 0.99, 1]`; the same rewards **without** reset give `G[2] ≈ -0.0297` (shows the leak). A forward-order bug would give `[1, 1.5, 1.75]`.
- **T-R2 exact gradient on a toy MDP (exact; the strongest correctness evidence)**: tabular softmax policy θ ∈ R^{3×2} (fixed random, seed 0, std 0.5). 2-step MDP: state 0, action a0 ∈{0,1}, reward `R0=[0.0,1.0][a0]`, next state `1+a0`; at state `1+a0` action a1, reward `R1[1]=[0.5,0.0]`, `R1[2]=[0.0,2.0]`, then terminate. γ=0.5. Enumerate the 4 trajectories to get the exact `J(θ)=E[r0+γ r1]` and its **autograd gradient**. Draw N=200,000 sampled trajectories (vectorized), compute the REINFORCE estimator per trajectory `Σ_t w_t ∇logπ(a_t|s_t) G_t`. Assert: with `w_t=γ^t`, every component of the sample mean is within 4 standard errors of the exact gradient. With `w_t=1`, compute its exact expectation by enumeration; assert the sample mean matches *that* within 4 SE and that it **differs from the true gradient** by more than 10 SE in at least one component (tune the reward table if the difference is too small). Interpretation to write down: with γ=0.5 the unweighted step-2 term is off by a factor of 2 (1/γ).
- **T-R3 bandit closed form (exact)**: softmax bandit, θ=0, r=(1,0): expected REINFORCE gradient = (+0.25, −0.25); N=200k within 4 SE.
- **T-R4 Adam first step (exact)**: on the first update every parameter with non-zero gradient moves by exactly `lr` (= 0.01) in magnitude (bias-corrected Adam). Explains why Adam hides gradient-scale differences between baseline and no-baseline runs.
- **T-R5 NaN guard (exact)**: rewards `[1.0]` and `[1,1,1,1]` through the loss (with `--standardize 1`) give finite loss and finite gradients (use `std(unbiased=False)+1e-8`; skip standardization for length-1).

### 7.2 CartPole (Part 2 and 3)
- **T-C1 termination vs truncation (exact)**: env with `env.unwrapped.theta_threshold_radians = 1e9` and `x_threshold = 1e9` and a random policy → every episode has length exactly 500, `truncated=True`, `terminated=False`. Also check the stored cause equals "time-limit".
- **T-C2 random-policy anchor (prediction)**: untrained net, 500 episodes: mean ∈ [19, 26], std ∈ [8, 17] (published random-policy mean ≈ 22, std ≈ 12; SE ≈ 0.5).
- **T-C3 initial policy (prediction)**: over 20 episodes, fraction of action 0 ∈ [0.4, 0.6] and mean entropy ≥ 0.6 nats (max 0.693).
- **T-C4 determinism & first-10-step trace (exact)**: two runs, same seed → identical observations/actions/rewards for the first 10 steps; after 3 updates two fresh agents with the same seed have bit-identical parameters on CPU. Print the 10-step trace table (t, observation, probs, action, reward, G_t, logπ) and put it in the report. Omitting `zero_grad` must change the parameters (proves the test can fail).
- **T-C5 bookkeeping (exact)**: `len(logp)==len(rewards)==len(G)` asserted at every update; buffers cleared after each update.
- **T-C6 reward-sign reversal (prediction)**: multiply rewards by −1; after 300 episodes (3 seeds) the mean length of the last 50 episodes < 15 (the agent learns to fail fast; random is ≈ 22). Report the measured values either way.
- **T-C7 loss reduction / step size (prediction)**: over 20 fixed-weight episodes, correlation between episode length and gradient norm is high (r > 0.7) under the summed loss and weak under mean-over-timesteps; report both numbers and the lr 0.01 vs 0.001 comparison (C4 below).
- **T-C8 γ=0 (exact theory)**: with γ=0, `G_t = 1` for all t, so the expected gradient is zero. Run γ=0 for 3 seeds × 1000 episodes: no learning expected (mean length stays within the random-policy range). Optional sweep γ ∈ {0.5, 0.8, 0.99}: report episodes to reach MA100 ≥ 195; **do not assume** the ordering (a short horizon still gives some signal near failure).
- **T-C9 final-policy lottery (report)**: evaluate 5 checkpoints from the last 20% of training, 50 episodes each; if their means differ by > 100, say so.
- **T-C10 termination-cause statistics**: of the 500 evaluation episodes, fractions ending by x-limit / angle / time-limit; hypothesis (not guaranteed): γ=0.95's ≈ 20-step horizon leaves some failures by cart drift. Report the numbers and the bimodality of the histogram if present.

### 7.3 Pong (Part 2 and 3). Needs ale-py; run locally first, then on deepdish
- **T-P1 action mapping (exact)**: wrap `env.step` to record actions: over 1000 steps all recorded env actions ∈ {2,3} and stored indices ∈ {0,1}; a policy forced to logits `[100,−100]` sends env action 2 every step.
- **T-P2 paddle control (exact, deterministic env for tests only: `repeat_action_probability=0`)**: force action 2 for 30 steps then action 3 for 60 steps; the agent paddle's vertical centroid in the preprocessed frame (agent = right paddle; look in roughly columns 68–73 of the 80×80 frame, **verify by plotting**) moves ≥ 5 px one way, then ≥ 5 px the opposite way. Do not assume which action is "up". Also confirms `preprocess` keeps the paddle.
- **T-P3 preprocess safety (exact)**: after `preprocess(obs)` the input `obs` is unchanged (copy-safe); output shape is (80,80) → flatten (6400,); values ⊆ {0,1}; frame difference values ⊆ {−1,0,1}; the first difference after reset is all zeros; across 300 random-policy steps min nonzero-pixel count > 0 and the difference is non-zero on ≥ 50% of steps after step 20 (the ball moves). Save a PNG of 6 consecutive preprocessed frames and their differences for the report appendix and **look at it**.
- **T-P4 initial policy (prediction)**: over 20 episodes the fraction of index 0 ∈ [0.4, 0.6] and mean entropy ≥ 0.6 nats. If not, fix the init before spending hours.
- **T-P5 random anchor (prediction)**: untrained net, 20–50 episodes: integer rewards in [−21, −19], mean ≈ −20.7 (published random-agent score, from memory) — accept [−21, −19.8]; the 100-episode MA has SE ≈ 0.07–0.1, so claim learning only if the MA rises by more than ≈ 0.3 (≥ 3 SE) and stays there.
- **T-P6 worker/learner consistency (exact)**: for one batch, the learner's recomputed logπ(a|s) (GPU) matches the worker's logged logπ to < 1e-4 → proves identical weights were used (on-policy) and the index/action mapping is consistent.
- **T-P7 point-boundary returns (exact)**: on a recorded episode, returns at steps before each ±1 reward never include later points' rewards (check against T-R1 logic on real data).
- **T-P8 determinism (soft)**: same seed + same action sequence → identical first 50 frames (hash). If ALE is not deterministic in default mode, record that and skip.
- **T-P9 scripted ball-tracker (prediction)**: a hand-written policy (move the agent paddle toward the ball's row; ball = nonzero pixels outside the paddle columns) should clearly beat random (e.g. > −18 over 10 episodes). It validates the whole pipeline (frames, mapping). If it does not, investigate the frames before training. Not required to be good.
- **T-P10 resume (exact)**: kill a short run after 3 updates, `--resume`, and confirm the episode counter and parameters continue (no restart, no duplicated episodes in the CSV).
- **T-P11 throughput benchmark** (`scripts/bench_pong.py`): steps/s for workers ∈ {1, 8, 16, 24, 32} with the real policy forward; choose the smallest worker count within 10% of the best; compute episodes/s and the episode budget for 4 hours. Save to `results/pong_bench.json`.

### 7.4 Baseline (Part 3)
- **T-B1 unbiasedness (exact)**: on the toy MDP with +10 added to every reward, estimators with `b=0`, `b=J` (constant), and `b=V^π(s)` (exact state value by enumeration) have the same mean within 4 SE (N=200k). Variance (trace of covariance): `Var(b=0) > Var(b=J)` is expected (exact check); `Var(b=J) > Var(b=V^π)` expected (prediction).
- **T-B2 optimal constant baseline (exact, closed form)**: for the 2-action softmax bandit, `‖∇logπ_a‖² = 2(1−π_a)²`, so `b* = Σ_a π_a(1−π_a)² r_a / Σ_a π_a(1−π_a)²`; at π=(0.5,0.5), `b* = J`; for skewed π (e.g. 0.9/0.1) `b*` weights the rarer action more (≈ 9×). Verify numerically against sampled variance for b ∈ {0, J, b*}: ordering `Var(b*) ≤ Var(J) ≤ Var(0)` when rewards are offset.
- **T-B3 no leak / detach (exact)**: (i) with the value net's output forced to exactly 0 (zero last layer and frozen), the Part 3 gradient equals the Part 2 gradient for the same trajectories (`allclose`, rtol 1e-6); (ii) after the policy-loss `backward()`, every value-net parameter has `grad is None` (or all-zero); (iii) after the value-loss backward, policy grads are unchanged.
- **T-B4 EMA baseline independence (exact)**: the baseline used for episode i is computed only from episodes < i (assert by construction/test with a recorded sequence).
- **T-B5 gradient-variance experiment (the key quantitative evidence)**: freeze the policy at three CartPole checkpoints (episode 200, 1000, final) of config C0 seed 0. Collect 1000 fresh episodes; split into A (500) and B (500). Train a value net on A (many epochs) and compute the constant baseline from A. On B compute per-episode gradient vectors for `b=0`, `b=const`, `b=V_φ` (policy gradient only, same trajectories — a paired comparison). Report: mean-gradient cosine similarity between arms (expect ≈ 1 up to noise), `trace(Cov)` ratio vs `b=0` (expect < 1 for the baselines), and the value net's **explained variance on B**, `R² = 1 − Var(G − V)/Var(G)` (expect > 0.5 for CartPole at the final checkpoint — a guess; **R² < 0 means the baseline hurts**; report it honestly). Note that Adam normalizes gradient scale (T-R4), so variance reduction shows up as less noisy curves across seeds, not larger steps.
- **T-B6 consistency (exact)**: value-net target is the same `G_t` the policy uses; the advantage mean over a batch is reported (should be ≈ 0 for a good baseline).
- **T-B7 seed spread (report)**: CartPole 5 seeds × {none, ema, value}: report mean and std across seeds of the final MA100 and of the 500-episode mean; expectation (not guaranteed): lower across-seed std with the value baseline.

---------------------------------------------------------------------------------------------------

## 8. Experiment matrix and Pong bring-up protocol

### 8.1 CartPole (CPU; ≤ 8 parallel processes on deepdish, or locally if faster)
| id | config | seeds | episodes | purpose |
|---|---|---|---|---|
| C0 | baseline none, w=1, lr 0.01, 1 ep/update | 0–4 | 2000 | **Part 2 primary** |
| C1 | value baseline | 0–4 | 2000 | **Part 3 primary** |
| C2 | EMA baseline | 0–4 | 2000 | Part 3 secondary |
| C3 | none, `--gamma_weight on` | 0–2 | 2000 | γ^t ablation (D1) |
| C4 | none, lr 0.001 | 0–2 | 2000 | lr sensitivity (T-C7) |
| C5 | reward sign −1 | 0–2 | 300 | T-C6 |
| C6 | γ ∈ {0, 0.5, 0.8, 0.99} (optional) | 0–2 | 1000–1500 | T-C8 |
Fallback if C0 averages < 100 after 2000 episodes: add variants with 8 episodes/update or lr 3e-3, **keep and report the lr 0.01 result as the primary** and label the variants.
After training: `evaluate.py` for every run (500 episodes sampled + greedy + last-20% checkpoint spread). The headline Part 2/3 CartPole figures use **seed 0** (state this); multi-seed mean ± std figures use all seeds.

### 8.2 Pong bring-up protocol (a fixed ladder; follow it without asking)
1. **Benchmark (T-P11)** → choose `--workers` (≤ 32).
2. **Smoke test, 200 episodes**, lr 1e-3, 16 ep/update, baseline none: no NaN; entropy stays ≥ 0.3; `frac_idx0` stays in (0.15, 0.85); throughput logged. Fix before proceeding.
3. **LR sweep, 2000 episodes each, baseline none** (run 2 at a time, 16 workers each, or sequentially): lr ∈ {1e-2, 1e-3, 3e-4, 1e-4}. Selection metric: final MA100 of training reward, plus mean episode length and points won as secondary signals (early learning often shows up as longer rallies). Record all four curves; the lr=0.01 run is evidence for the report's justification of deviating from 0.01. Selection uses training reward (say so).
4. **If no setting leaves −21 (MA100 improvement < 1.0 after 2000 episodes)**, climb the ladder, one rung at a time, 2000 episodes each, and stop at the first that learns: (a) `--episodes_per_update 32`; (b) `--arch cnn`; (c) `--standardize 1` (labelled variant: it adds an implicit mean baseline); (d) hidden 400. Document every rung tried and the result.
5. **Full runs** with the selected setting, each capped at `min(25,000 episodes, 4 hours wall-clock)`: **(P0) baseline none = Part 2**, **(P1) value baseline = Part 3**. If throughput allows (total workers ≤ 32), also **(P2) EMA baseline**. If P0 needed `--standardize 1` to learn, then Part 2 reports both P0-raw and P0-std, and Part 3 baselines are compared against the matching arm. Stop a run early only if the 100-episode MA ≥ +15 (essentially won) or has been flat (< 0.3 change) for the last 3,000 episodes.
6. **Final evaluation**: 500 episodes sampled + greedy for every full run, last-20% checkpoint spread, histogram, mean/std.
Launch pattern on deepdish:
```bash
tmux new -s pong0
export CUDA_VISIBLE_DEVICES=1   # or whichever single card is idle (see nvidia-smi); never more than one card
source ~/venvs/mlds490/bin/activate && cd ~/mlds490_hw1
nice -n 10 python -m pg.train_pong --baseline none --lr <LR> --workers 16 --out results/pong/none/seed0 2>&1 | tee -a results/pong/none/seed0/log.txt
```
Poll `tail -n 5 .../log.txt` and `curve.csv` every ~10–15 minutes while doing other work.

---------------------------------------------------------------------------------------------------

## 9. Figures and tables (`plot_results.py`)
For each of {CartPole, Pong} × {Part 2 (no baseline), Part 3 (baseline)}:
1. **Training curve**: per-episode reward (thin, light) + MA100 (bold, from episode 100). Title includes env, method, lr, γ, seed.
2. **Rollout histogram**: 500 final-policy episodes (sampled). CartPole bins of width 25 up to 500 (note the 500 truncation spike); Pong integer bins −21…21. Annotate mean ± std and n=500; secondary greedy numbers in the caption or a small table.
Additional figures:
3. CartPole: 5-seed MA100 mean ± std band, Part 2 vs Part 3 (and EMA) overlaid; undiscounted reward on the y-axis, a dashed line at 500.
4. Pong: MA100 of all full runs overlaid (P0, P1, P2 and variants), plus the lr-sweep figure.
5. Gradient-variance figure/table from T-B5; baseline R².
6. First-10-step trace tables (CartPole, Pong) and the preprocessing check image (appendix).
7. Ablation figure: γ^t weighting (C3), lr 0.001 (C4), γ sweep if run.
Tables: results summary (mean, std, min, max of the 500 rollouts for every run; greedy row; checkpoint spread), validation tables per part, runtime/throughput table.
All figures saved as PNG at ≥ 200 dpi in `figures/`, each with axis labels, units, and a one-line caption in the report that states what the reader should conclude.

---------------------------------------------------------------------------------------------------

## 10. Report (`report/report.md` → `report/report.pdf`)
Build with pandoc (+ a LaTeX engine) if installed, else weasyprint/markdown-to-HTML-to-PDF; **open the PDF and check that every figure and table renders and nothing overflows**. Structure:
1. **Title block + submission info**: GitHub username `<<GITHUB_USERNAME>>`, repository name `<<REPO_NAME>>`, link `<<GITHUB_LINK>>` (these three are placeholders the student must fill in – list them in `TODO_USER.md`; leave unmistakable placeholders, never invent them). Short AI-assistance statement (draft wording; the student confirms in `TODO_USER.md`).
2. **How to run** (setup, Part 1 commands, CartPole commands, Pong commands, validation commands, expected runtimes, hardware used: deepdish, one GPU, CPU count).
3. **Part 1**: model + assumptions, figures (a)(b)(c), results table, edge cases and tests (grouped table, measured values), discussion (threshold = 13; plateau 13–15; cap artifact; timing ambiguity harmless).
4. **Part 2 CartPole** and **Part 2 Pong**: method (estimator D1, returns D2, architecture, hyper-parameters, deviations from the spec and why, parallel actor design), plots, rollout stats, edge cases + tests (T-R*, T-C*, T-P*), honest discussion (including seeds, final-policy spread, termination causes, what did not work).
5. **Part 3**: which baselines and how implemented (D5), plots, rollout stats, tests (T-B*), the gradient-variance evidence, comparison with Part 2.
6. **Edge-case and validation summary table** across parts (name, what could go wrong, handling, test, measured result).
7. **"Given an additional 3 months…"** (answer in your own words, based on): many more seeds with confidence intervals and significance tests; hyper-parameter sweeps for lr/batch/γ/architecture; stronger estimators (GAE, A2C/PPO) and a CNN vs MLP ablation on Pong; proper optimal-baseline estimation and variance studies; running policy gradient on the shuttle MDP and comparing with the exact optimum from Part 1 (a clean exactly-solvable testbed); robustness of the shuttle policy to perturbed arrival distributions and a different cap semantics; evaluating Pong under sticky-action settings and across more evaluation seeds; ablating return reset at point boundaries.
8. **Appendix**: preprocessing figure, 10-step traces, full validation tables, LR sweep, throughput benchmark.

---------------------------------------------------------------------------------------------------

## 11. Timeline (Chicago time) and priorities
| when | what |
|---|---|
| Sun 19:00–20:30 | repo skeleton, local venv, `common/models/reinforce`, T-R* tests, CartPole trainer; copy `shuttle/` and re-run it |
| Sun 20:30–21:30 | remote setup (§2.4), T-C* tests, launch CartPole C0–C2 on deepdish |
| Sun 21:30–23:00 | Pong utils + worker pool + T-P* tests (local), benchmark and smoke test on deepdish |
| Sun 23:00–01:30 | LR sweep (§8.2 step 3); CartPole eval, ablations and T-B5 in parallel |
| **Mon 01:30 at the latest** | **launch the Pong full runs (P0, P1[, P2])** |
| Mon 01:30–06:30 | training (4 h cap); meanwhile: plots for Part 1/CartPole, report draft, STUDY_GUIDE |
| Mon 06:30–09:00 | Pong final evaluation, Pong figures |
| Mon 09:00–15:00 | finish report, build PDF, check, git repo ready |
| Mon 15:00 | **feature freeze**; remaining time = buffer (deadline 23:59) |
If you are behind, **cut in this order**: C6 γ sweep → Pong P2 (EMA) → CartPole C4/C5 → second CartPole ablation seeds → Pong standardized variant. Never cut: the validation tests, the 500-episode rollouts, histograms, the baseline experiments, or the report's honesty section.

---------------------------------------------------------------------------------------------------

## 12. GitHub repository
Prepare the repo; the student pushes. `git init`, `.gitignore` (venv, `__pycache__`, `*.pt` over a few MB, `results/**/ckpt_*.pt` except the final small policies if under ~50 MB, `.DS_Store`), commit with clear messages, README with the run instructions (same as the report). Try `gh auth status`; if authenticated, `gh repo create <name> --public --source . --push`, otherwise put the exact commands in `TODO_USER.md`. Make sure the repo contains **all code used** (Part 1, 2, 3, validation, plotting, scripts), `requirements.txt`, the figures, and the small result CSV/JSON files the report quotes. Do not commit credentials.

## 13. Definition of done (check every box before finishing)
- [ ] Part 1 reproduces the numbers in §3; report section complete.
- [ ] All `exact` validation tests pass (CartPole, Pong, baseline); `prediction` tests reported honestly, including failures.
- [ ] CartPole: C0, C1, C2 × 5 seeds trained and evaluated; Part 2 and Part 3 figures + histograms + mean/std; gradient-variance table (T-B5).
- [ ] Pong: P0 and P1 trained within budget, evaluated on 500 episodes, figures present; every deviation (lr, batch, variants) justified with evidence; the sweep figure included.
- [ ] Each of Parts 1–3 has an edge-case section and a "how we know it works" section with quantitative evidence and run instructions; the 3-months question answered.
- [ ] `report.pdf` opened and visually checked; placeholders for GitHub username/repo/link are clearly marked; `TODO_USER.md` lists everything the student must do (fill placeholders, push repo, confirm AI statement, submit on Canvas).
- [ ] `STUDY_GUIDE.md`, `DECISIONS.md`, `README.md`, `requirements.txt` written; no secrets committed.
