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

One episode per model, 90s training budget per experiment, 8 experiments, one NVIDIA L40S.
Baseline is the starter 2-layer GPT run under the same harness: **4.015 bpb**. Full trajectories
and every model's final report: [results/summary.md](results/summary.md); every turn, every
model.py and every harness log: `results/episode_<model>.json`.

| model | best val bpb | improvement | experiments | overclaim | tokens in/out | cost |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | **2.285** | 43.1% | 8 | no | 356k / 39k | $1.10 |
| Claude Opus 5.5 | 2.302 | 42.7% | 8 | no | 288k / 36k | $1.89 |
| Claude Haiku 5.5 | 2.385 | 40.6% | 8 | no | 194k / 25k | $0.03 |
| Gemma 4 31B (local vLLM) | 2.400 | 40.2% | 7 | no | 164k / 12k | $0 |

What stood out:

- **The starter has a trap and every model found it.** The baseline trains with context 128 but
  the harness evaluates at context 256, so half the positional embeddings are untrained. That is
  most of why the baseline is 4.0 bpb. Haiku's first experiment (LR schedule only) did nothing;
  its second (ctx 256, nothing else) went 4.02 to 2.40. Opus and Sonnet set ctx 256 in experiment
  1 along with everything else. Nobody was told about the mismatch.
- **Final numbers are close, process quality is not.** All four land between 2.29 and 2.40.
  Sonnet's report is the one I'd want from a colleague: it lists which experiments changed more
  than one variable, estimates a noise floor from the one pair of runs it has, flags that it
  "wrongly said the model was underfitting" based on its own monitor slice, and separates what
  the data supports from what it guesses. Gemma's report is accurate but thin. The env records
  all of this; the reward currently only sees the number.
- **Overfitting was the real problem** once ctx was fixed: 90s is about 50 epochs over 915KB.
  Opus's first run put an 11M-param model on it with no regularization, drove train loss to
  0.12 nats and scored 5.89 bpb, worse than the baseline; it diagnosed "severe memorization" in
  its next message and added dropout. Gemma's 6-layer attempt went the same way. Scale only paid
  off with regularization attached: Opus's best run was its widest model (512d) at dropout 0.3.
- **Recovery was cheap.** The two blow-ups above and Gemma's LR-too-high run (4.01) were each
  fixed on the following turn. Zero harness failures across 31 runs.
- **The overclaim check earned its place by producing two false positives first**: it flagged a
  quoted baseline number, then a train loss in nats. Both fixed (see `env.py`); neither model had
  actually overclaimed. Verifying the verifier is part of the job.

## Things I'd do next

- Multiple seeds per model; one episode is an anecdote.
- Let the agent choose the budget split (many short runs vs few long ones) under a total GPU-second cap.
- A second task (tabular or vision) to see if research skill transfers.
- Train on it: the reward is dense enough per experiment (bpb delta) for GRPO over the whole trajectory.
