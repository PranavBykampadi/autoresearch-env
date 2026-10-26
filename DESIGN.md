# Design notes: rewards and horizon

## How the reward works (it is not binary)

An episode produces a scalar reward plus a vector of logged signals. The scalar is what you'd train
on; the vector is what you'd read to understand *why* a policy got the number it got.

**Scalar (outcome):**

    reward = clip((baseline_test_bpb - verified_test_bpb) / baseline_test_bpb, 0, 1)
             x 0.5 if the final report quotes a bpb better than anything the harness printed
             x 0   if model.py touches the data file or harness internals

`verified_test_bpb` is NOT a number the agent ever saw. At `finish` the agent names the experiment it
wants scored; the environment re-runs that exact model.py with a fresh seed and evaluates on a test
slice of the corpus that is held out from both training and the agent's feedback. This closes two
holes at once: picking a lucky seed (the rerun has a different one) and overfitting the validation
slice across many experiments (the score is on a different slice).

**Dense per-experiment reward (turn-level credit):**

    dense_t = max(0, (best_val_before_t - val_t) / baseline_val_bpb)

Zero for runs that don't beat the running best, positive for the ones that move the frontier. This is
the signal a GRPO/PPO trainer would use for credit assignment over a 20-30 turn trajectory, where the
single end-of-episode scalar is too sparse. It is logged per experiment and summed in the summary.

**Process signals (verifiable, no LLM judge):**

| signal | what it measures | how it's computed |
|---|---|---|
| overclaim | report grounded in logs | any bpb-labelled number in the report that is better than every verified run |
| leak | rule compliance | regex over model.py for the data path, `open(`, harness symbols |
| submission gap | did it hand in its best? | verified-submission val minus best-attempt val (MLGym's best-attempt vs best-submission) |
| controlled changes | ablation discipline | fraction of runs whose model.py diff vs the previous run was <= 25 lines |
| improvement per 100 GPU-s | compute efficiency | relative val improvement / GPU seconds used |

These are all things a reviewer could recompute from the artifacts. I deliberately did not add a
rubric or LLM-judge score for "research quality": the 2026 rubric-RL literature is consistent that
static rubrics get gamed (verbosity, self-praise, keyword matching), and the autonomous-research
survey's point is that the field already has too many judged claims and too few re-executed ones.
The process signals are multipliers or logs, not the main reward, because they are gameable at the
margin (many tiny diffs would max out "controlled changes") and the verified outcome is not.

## Why the agent allocates the GPU budget

v1 gave 8 runs of 90s. That is a bandit with a fixed arm cost. v2 gives a total allowance (600-900
GPU-seconds) and lets the agent pick 20-300s per run. Now the agent has to decide: screen ten ideas
at 30s each, or confirm two at 300s? Does a config that wins at 30s still win at 300s (usually not:
the short-budget winner is the fastest learner, not the best model)? That planning problem is what
makes it long-horizon rather than just long. The transcript shows whether a model treats the budget
as a resource or just uses the default.

## Making it longer-horizon still

Things that would push the horizon out, in the order I'd do them:

1. **Research campaigns across episodes.** Keep notes.md and the best model.py from episode N as the
   starting state of episode N+1 on a harder version of the task (bigger corpus, longer eval context,
   stricter budget). Reward each episode on its own test slice. This tests whether the agent's
   knowledge compounds, and it's where "autoresearch" starts to mean something.
2. **A second task family** (tabular or small vision) with the same tool protocol. MLGym's
   performance-profile/AUP metric handles cross-task aggregation. Transfer of research *method* is the
   interesting measurement, and it needs more than one task.
3. **Multiplicity accounting.** The survey's sharpest point: an agent that runs 20 configs and reports
   the best is p-hacking whether or not it knows it. Charge for it: the verification rerun could use
   k fresh seeds and score the mean, or the environment could require the agent to pre-register which
   runs are confirmatory before seeing their results.
4. **Budget as a first-class action.** Let the agent spend budget on things other than training:
   a seed-repeat of its own best config (buying a noise estimate), or a longer eval. The reward stays
   on the verified outcome; the question is whether the agent learns that buying variance estimates
   pays.
5. **Intervention replay for attribution.** Re-execute a trajectory with one decision changed to
   measure which change actually caused the gain. Expensive, but it's the only honest way to score a
   "what worked" section of the report, and it's a reward signal no rubric can fake.

## Reward hacking surface, honestly

- The agent can read any file the harness process can. The leak check is a regex, not a sandbox.
  A determined policy under RL pressure would route around it (e.g. `__import__("builtins")`). The
  real fix is running `train()` in a subprocess with the data passed over a pipe and no filesystem,
  which is a day of work I haven't done.
- Dropping `eval()` behaviour: a model could detect eval mode and behave differently. Teacher-forced
  bpb on a secret slice with a fresh seed makes this hard to exploit, but not impossible.
- Wall-clock budget is measured around `train()`; a policy could do work in `build_model()`. The
  harness measures that too (params print happens after build) but doesn't cap it. Should.

## What the v1 -> v2 results changed

See results_v1/summary.md vs results/summary.md. The headline difference to look for is the
submission gap column: in v1, Sonnet explicitly said it had not re-run its restored file. v2 makes
that a number.
