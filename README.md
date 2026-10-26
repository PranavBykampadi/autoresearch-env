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

See [results/summary.md](results/summary.md).

## Things I'd do next

- Multiple seeds per model; one episode is an anecdote.
- Let the agent choose the budget split (many short runs vs few long ones) under a total GPU-second cap.
- A second task (tabular or vision) to see if research skill transfers.
- Train on it: the reward is dense enough per experiment (bpb delta) for GRPO over the whole trajectory.
