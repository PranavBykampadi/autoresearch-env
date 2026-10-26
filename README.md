# autoresearch-env

An RL environment where the agent *is the researcher*. It gets a GPU, a small text corpus,
a baseline byte-level language model, a fixed training-time budget per experiment, and a
limited number of experiments. It decides what to try. The only thing it can't touch is the
evaluator.

The metric is validation **bits per byte** on a fixed held-out slice, computed by a read-only
harness the environment re-installs before every run. Reward is the relative improvement of
the best *verified* run over the baseline, halved if the agent's final report claims a number
the harness never printed.

## Why this shape

"Open-ended" and "verifiable" usually pull in opposite directions. Fully open research
(agent picks the question) needs an LLM judge, and the judge is the thing that gets gamed.
This env keeps the research open (architecture, optimizer, schedule, batch/context size,
precision, compilation, data sampling, anything in `model.py`) while pinning the metric,
the eval code, the data split and the compute budget. You get a real number per episode and
can read the agent's lab notebook to see *how* it got there.

## Episode loop

```
agent  -> read_file model.py / harness.py
agent  -> write_file model.py        (full rewrite; no patch tool)
agent  -> run_experiment note="..."  env ships workspace to GPU, trains for budget_s, evals, returns val_bpb + log tail
agent  -> write notes.md, iterate
agent  -> finish REPORT              env scores: best verified bpb vs baseline, checks report for overclaims
```

Actions are tagged blocks in plain text (`<action tool="..." ...>...</action>`), so the same
protocol works for models with and without native tool calling.

## Components

- `autoresearch_env/sandbox/harness.py` – read-only. Loads data, holds out the last 200k bytes,
  pre-warms CUDA/optimizer so one-time costs don't eat the budget, calls the agent's
  `build_model()` and `train()`, enforces the time budget, evaluates bpb with teacher forcing.
- `autoresearch_env/sandbox/model.py` – the starter: a 2-layer, 128-dim GPT with AdamW. The agent
  rewrites this.
- `autoresearch_env/env.py` – tools, workspace, scoring, overclaim check.
- `autoresearch_env/remote.py` – runs the harness on a remote GPU over ssh (tar stream in,
  stdout out). Set `AR_SSH_HOST`, `AR_REMOTE_ROOT`, `AR_GPU`.
- `autoresearch_env/agent.py` – the agent loop and action parser; compacts old file writes out of
  the history so 32k-context models fit 8+ experiments.

## Running

```bash
uv venv && source .venv/bin/activate && uv pip install -e .
# remote box needs: python venv at $AR_REMOTE_ROOT/.venv with torch, and data/input.txt
source ~/.secrets/anthropic.env
python -m autoresearch_env.run --models gemma-4-31b,haiku-5.5,sonnet-5.5 --budget 90 --experiments 8
```

Per episode you get `results/episode_<model>.json` (every turn, every experiment, the model.py
for each run, the final report) and `results/summary.md`.

## Results

### v2: agent-allocated budget, verified submission, secret test slice

One episode per model, 600 GPU-seconds to allocate across runs of 20-300s, baseline measured with a
90s run (val 4.035, test 4.151 bpb). The agent names the experiment to submit; the environment
re-runs that file with a fresh seed and scores it on a test slice the agent never saw. Full tables:
[results/summary.md](results/summary.md). v1 (8 fixed 90s runs, reward on val) is in
[results_v1/summary.md](results_v1/summary.md).

| model | GPU-s used | runs | best attempt (val) | submission, fresh seed (val) | submission (test) | **reward** | controlled changes | cost |
|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 540 | 8 | 2.326 | 2.333 | 2.512 | **0.395** | 88% | $0.64 |
| Claude Opus 5.5 | 540 | 6 | 2.438 | 2.454 | 2.594 | 0.375 | 67% | $1.16 |
| Gemma 4 31B | 450 | 4 | 2.748 | 2.840 | 3.079 | 0.258 | 25% | $0 |
| Claude Haiku 5.5 | 240 | 3 | 2.408 | (2.427) | (2.616) | 0 (rule violation) | 33% | $0.02 |

Reward = relative improvement of the verified submission over the baseline on the test slice.

What the second round showed:

- **Sonnet ran the cleanest study.** Eight runs, seven of them a one-variable change from the previous
  one (2-line diffs: dropout sweep, width sweep, LR), 60s screens then two 90s confirmations, and it
  submitted the config that actually produced its best number. Its fresh-seed rerun landed within
  0.007 bpb of its best attempt, which is the smallest gap in the table and means it was not riding a
  lucky seed.
- **Opus bought a noise estimate.** It spent its last run re-executing its best config unchanged "to
  confirm reproducibility" (2.438 -> 2.451), then submitted that config. That is the behaviour the
  verification step is meant to reward, and it is the kind of decision a fixed-8-runs env can't elicit.
- **Haiku was zeroed by the leak rule, correctly.** Its second experiment imported the harness module
  to run a diagnostic on the train slice. Importing the harness exposes the test split, so the static
  check refused the run and flagged the episode; Haiku then finished with 360 GPU-seconds unused. I
  re-verified its submitted file offline afterwards (numbers in parentheses): it would have scored
  about 0.37. The rule did what it says, and the transcript shows the model understood the refusal.
- **The test slice is harder than val for every model** (about +0.2 bpb), including the baseline
  (+0.12). The two slices are consecutive 200KB tails of the same corpus, so this is the kind of
  distribution shift a real held-out set has. Scoring relative to the baseline on the same slice keeps
  the comparison fair; it also means val-only numbers (v1) overstated everyone slightly.
- **Gemma spent 450s and never beat its first run.** Its three follow-ups were all large multi-knob
  rewrites (24-90 line diffs) and all worse. Its fresh-seed rerun was 0.09 worse than its best
  attempt, the largest gap in the table: with `torch.compile` and a 90s budget, the seed matters.
- **The overclaim detector needed three rounds to stop lying.** It flagged a quoted baseline (v1), a
  train loss in nats (v1), a truncated quote of a real number (2.74 for 2.7484), the mean of two real
  runs, and the agent's own clearly labelled holdout numbers (v2). All five were false positives;
  no model has overclaimed yet. The current version is rounding-aware, accepts means of verified
  runs, and ignores numbers the text labels as holdout/train/delta. It is still a regex, and
  `DESIGN.md` says why I kept it anyway.

### v1: 8 fixed runs of 90s, reward on val

| model | best val bpb | improvement | cost |
|---|---|---|---|
| Claude Sonnet 5.5 | 2.285 | 43.1% | $1.10 |
| Claude Opus 5.5 | 2.302 | 42.7% | $1.89 |
| Claude Haiku 5.5 | 2.385 | 40.6% | $0.03 |
| Gemma 4 31B | 2.400 | 40.2% | $0 |

The v1 starter had a trap (trains at ctx 128, evaluates at ctx 256) that every model found
unprompted; Haiku's second experiment changed only that and went 4.02 to 2.40. Final numbers
clustered; report quality did not. Sonnet's v1 report explicitly said it had not re-run its restored
file, which is the gap v2's verification step now measures.

## Things I'd do next

- Multiple seeds per model; one episode is an anecdote.
- Let the agent choose the budget split (many short runs vs few long ones) under a total GPU-second cap.
- A second task (tabular or vision) to see if research skill transfers.
- Train on it: the reward is dense enough per experiment (bpb delta) for GRPO over the whole trajectory.
