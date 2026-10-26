# Decisions log

Pre-registered decisions D1–D10 are in `PLAN.md` §4 and were not changed.
Deviations and on-the-fly calls are recorded here as they happen.

## D11 — GPU card
Use `CUDA_VISIBLE_DEVICES=1` (RTX PRO 5000 Blackwell, 48 GB). At setup time all
three cards were idle; card 1 is the largest idle card. Every process sees
exactly one device. Verified with `nvidia-smi` before launch.

## D12 — no tmux on deepdish
`tmux` and `screen` are not installed. Long jobs use `nohup setsid` and write
to `log.txt`. Polling is via `tail` on that file and `curve.csv`.

## D13 — EMA first-batch value
The first update uses `b = 0` (no previous episodes). The EMA is then updated
from that batch for the *next* update. This keeps T-B4's "previous only" rule
and is unbiased (just higher variance on update 1).

## D14 — value-net Adam lr
`1e-3` for both CartPole and Pong, as the plan allows. Pong value updates are
one pass over the batch in minibatches of 4096.

## D15 — CartPole runs on CPU
The 4→128→2 MLP is faster on CPU than on a shared GPU. Seeds run as separate
processes (`OMP_NUM_THREADS=1`). Pong learner uses the GPU; actors stay on CPU.

## D16 — first Pong frame difference is zeros
`prev = first` at reset, so `x_0 = 0`. Documented in `pong_utils.py` and
checked by T-P3. The agent has no motion cue on the first action.

## D18 — action sampling
`torch.distributions.Categorical.sample` does not accept `generator=` on
torch 2.14 / 2.11. We sample with `torch.multinomial(probs, 1, generator=)`
(the same primitive Categorical uses) so per-episode RNGs still replay (T-C4).

## D19 — scripted Pong tracker (T-P9)
The first controller masked columns `< 16` and `≥ 64` and chased the ball's
current row. That hid the ball on the approach, and one frameskipped action
moves the paddle about 11 px (measured: repeated action 2 shifts the centroid
by ~11 rows), so the sign flipped every step and the score was −21 on every
episode. `scripted_action` masks only the paddle columns, predicts the row
at the agent paddle with wall reflections, and holds position inside a 5 px
deadband by alternating actions. On the deterministic env that scores +9,
which clears the −18 bar. Training is unchanged. The frames themselves are
fine (ball and both paddles are visible; T-P2 passed).

## D20 — CartPole C0 mean MA100 is under 100
Final 100-episode means for C0 (lr 0.01, one episode per update), seeds 0–4:
146.8, 9.3, 181.0, 92.4, 9.3. The average is about 88, and seeds 1 and 4
collapsed below the random-policy length. C0 stays the Part 2 primary, as
the plan requires. Two labeled variants, three seeds each, 2000 episodes:
C7 (`--episodes_per_update 8`, lr 0.01) and C8 (lr 0.003, one episode per
update).

## D28 — How the PDF was built
`/opt/anaconda3/bin/pandoc` is an x86_64 binary and will not start on this
Mac. `scripts/build_report.py` converts `report/report.md` with
Python-Markdown, inlines the figures, and prints with headless Chrome.
That is the rebuild command in `TODO_USER.md`.

## D27 — Pong full runs stopped on the flat-MA rule
P0 stopped at episode 5232 (MA100 −2.08, peak +2.65). P1 stopped at 7488
(MA100 +10.65, peak +12.68) after the resume in D25. P2, the EMA baseline,
was started once those two had released their 16 actors each, and stopped
at episode 6016 (MA100 +4.10, peak +7.61). None reached MA100 +15. The
flat rule compares the moving average with its value 3000 episodes earlier,
so a curve can still wiggle inside that window. P2's 500-episode evaluation
is the same protocol as P0 and P1.

## D26 — γ = 0 was run; the rest of the γ sweep was not
T-C8 asks for γ = 0, three seeds, 1000 episodes. That run is
`results/cartpole/C6_gamma0`. Last-100 lengths are 9.3, 9.5, 9.4, outside
the random-policy range, so the "no learning" prediction is not held. The
optional sweep γ ∈ {0.5, 0.8, 0.99} is the C6 item the plan says to cut
first, and it was cut.

## D25 — Pong value-run GPU memory
The Part 3 run climbed to 41 GB on the 48 GB card while the no-baseline run
stayed near 6 GB. The value update indexed the concatenated batch with a
view, so every minibatch graph kept the whole batch alive, and the caching
allocator never gave the memory back as rallies got longer. The fix clones
each minibatch (`index_select` + `detach`) and drops the batch tensors
after the step (`empty_cache`). The value run was killed and resumed from
`latest.pt` (episode 2752, MA100 6.9). The first resume died because
`torch.save` turns the RNG `ByteTensor` into a plain uint8 tensor and
`set_rng_state` rejects it; `load_rng_state` now casts it back. The
in-flight batch (episodes 2753–2768) had been logged but not checkpointed,
so those CSV rows were trimmed and that batch was collected again. After
the fix the value run sits near 1 GB. The no-baseline run was left running.

## D24 — Pong learning rate
Sweep, baseline none, batch 16, 2000 episodes, seed 0. Final training MA100:

| lr | final MA100 | entropy at the end | mean rally length |
|---|---|---|---|
| 1e-2 | −0.98 | 0.044 | ~3400 |
| 1e-3 | −17.27 | 0.45 | ~1900 |
| 3e-4 | −20.09 | 0.69 | ~1100 |
| 1e-4 | −20.40 | 0.69 | ~1000 |

Selection uses the training reward, as pre-registered. **lr = 0.01** wins, and it is the learning rate the assignment asked for, so the full runs do not deviate from 0.01. The smaller rates barely leave the random-agent score (about −20.7). At 0.01 the policy becomes peaked (entropy 0.04) while the 100-episode mean rises by about 19 points and rallies get about 3× longer. The mean is still noisy at the end (it touched +0.8 a hundred episodes earlier), which is why the full run keeps going rather than stopping at 2000.

## D23 — which physical GPU `CUDA_VISIBLE_DEVICES` selects
`nvidia-smi` order is 5060 Ti, PRO 5000, 5060 Ti. Torch's order is the
PRO 5000, then the two 5060 Tis. `CUDA_VISIBLE_DEVICES=1` therefore lands
on a 5060 Ti (that is what the smoke test used: log says `visible=1`,
and `nvidia-smi` showed memory only on its index 0). The 48 GB card is
torch index 0. Every later Pong job sets `CUDA_VISIBLE_DEVICES=0` and is
checked on `nvidia-smi` index 1. Still exactly one card.

## D21 — Pong worker count
T-P11 on deepdish (random policy, one batch per setting): steps/s were
1826 (1 worker), 5234 (8), 9497 (16), 12726 (24), 14372 (32). The smallest
count within 10% of the best is **32** (24 is 88.5% of 32). The trainer
sets the pool size to `min(workers, episodes_per_update)`, because a batch
of 16 cannot keep 32 actors busy. Pre-registered runs keep the batch at 16
(D3), so they use **16 workers** (the largest pool that batch can feed).
Two concurrent jobs use 16+16 = 32 workers, which is the node cap. A later
ladder rung that raises the batch to 32 will also raise the pool to 32.
At 16 workers a random episode is about 9 episodes/s, so 2,000 episodes is
a few minutes and 25,000 episodes is under an hour if rallies stay short.
4-hour budget at 32 workers is ~197k episodes; we still cap at 25,000.

## D22 — T-B5 value-net fit
Adam's first-step property (T-R4) means each update moves parameters by
about the learning rate, not by the raw gradient. Forty full-batch steps
at lr 1e-3 cannot move a value head from 0 up to returns near 15, so an
earlier 40-epoch fit stayed near R² ≤ 0 (the net had not left its init).
The reported T-B5 fit keeps lr 1e-3 and runs until the training MSE
plateaus or 4,000 epochs, whichever comes first. That is "many epochs"
in the plan, not a different baseline.

## D17 — T-P8
Default `ALE/Pong-v5` uses sticky actions (p=0.25), so same seed + same action
sequence need not replay. If hashes differ we record that and skip (exact
"documented skip" record). Training still uses the default env (D9).
