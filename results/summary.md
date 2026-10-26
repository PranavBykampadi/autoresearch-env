# Results

One episode per model. The agent allocates a total GPU-second budget across runs of 20-300s each. Best attempt = lowest val bpb the agent ever saw. Submission = the model.py it handed in, re-run with a fresh seed and scored on a test slice it never saw. Reward = relative improvement of the verified submission over the baseline on test (x0.5 if the report overclaims, 0 if model.py reads the data).

Baseline (90s run): val 4.0346, test 4.1506

| model | GPU s used | runs (failed) | best attempt val | submission val (fresh seed) | submission test | reward | gap (sub val - best) | controlled changes | dense sum | overclaim | tokens in/out | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | 450 | 4 (0) | 2.7484 | 2.8399 | 3.0791 | 0.258 | +0.091 | 25% | 0.319 | no | 57k/5k | 0.00 |
| haiku-5.5 | 240 | 3 (1) | 2.4084 | 2.4267 | 2.6161 | 0.000 | +0.018 | 33% | 0.403 | no | 83k/20k | 0.02 |
| opus-5.5 | 540 | 6 (0) | 2.4380 | 2.4535 | 2.5938 | 0.375 | +0.015 | 67% | 1.022 | no | 174k/22k | 1.16 |
| sonnet-5.5 | 540 | 8 (0) | 2.3262 | 2.3333 | 2.5122 | 0.395 | +0.007 | 88% | 0.423 | no | 206k/22k | 0.64 |

## Experiment trajectories

### gemma-4-31b

submitted: experiment 1

| # | budget | note | val_bpb | diff lines | dense r |
|---|---|---|---|---|---|
| 1 | 90s | Testing larger model with torch.compile and bfloat16 | 2.7484 | 90 | 0.319 |
| 2 | 120s | Testing larger model (N_EMBD=384, N_LAYER=8) with CosineAnnealingLR | 4.8306 | 50 | 0.000 |
| 3 | 120s | Testing N_EMBD=256, N_LAYER=6 with BATCH=128 and weight_decay=0.05 | 3.3701 | 27 | 0.000 |
| 4 | 120s | Testing smaller model (N_EMBD=128, N_LAYER=4) to reduce overfitting | 4.0093 | 24 | 0.000 |

### haiku-5.5

submitted: experiment 3

| # | budget | note | val_bpb | diff lines | dense r |
|---|---|---|---|---|---|
| 1 | 90s | exp1: LR 3e-3, CTX 256, batch 64, warmup + linear decay over budget, bf16 autoca | 4.0321 | 0 | 0.001 |
| 2 | 60s | exp2 diagnostic: same training as exp1, then score train slice under train/eval  | FAIL | 57 | 0.000 |
| 3 | 90s | exp4: clean rewrite of exp1 config (CTX 256, batch 64, LR 3e-3, warmup + linear  | 2.4084 | 30 | 0.402 |

### opus-5.5

submitted: experiment 6

| # | budget | note | val_bpb | diff lines | dense r |
|---|---|---|---|---|---|
| 1 | 90s | ctx256 6L384 bf16 sdpa cosine lr2e-3 | 6.5619 | 88 | 0.000 |
| 2 | 90s | dropout 0.25, 6L288, holdout early stopping w/ best ckpt | 2.4543 | 65 | 1.018 |
| 3 | 90s | dropout 0.35, wd 0.2 (vs exp2 0.25/0.1) | 2.4779 | 4 | 0.000 |
| 4 | 90s | dropout 0.15, wd 0.1 (vs exp2 0.25) | 2.4380 | 4 | 0.004 |
| 5 | 90s | exp4 config + tied head/embedding weights | 2.4577 | 13 | 0.000 |
| 6 | 90s | rerun exp4 config exactly (untied, dropout 0.15) to confirm reproducibility/nois | 2.4512 | 5 | 0.000 |

### sonnet-5.5

submitted: experiment 8

| # | budget | note | val_bpb | diff lines | dense r |
|---|---|---|---|---|---|
| 1 | 60s | v1: 4L/256d SDPA bf16, LR 2e-3, cosine, ctx256 bs64 | 2.6596 | 60 | 0.341 |
| 2 | 60s | v2: v1 + dropout 0.2 + wd 0.1 to test overfitting hypothesis | 2.3629 | 15 | 0.074 |
| 3 | 60s | v3: v2 with dropout 0.35 (only change) | 2.5617 | 2 | 0.000 |
| 4 | 60s | v4: v2 with dropout 0.1 (only change from v2/v3 dropout sweep) | 2.3262 | 2 | 0.009 |
| 5 | 60s | v5: v4 with N_EMBD=128 (capacity test, only change) | 2.5276 | 2 | 0.000 |
| 6 | 60s | v6: v4 with N_EMBD=384 (capacity test, only change) | 2.3657 | 2 | 0.000 |
| 7 | 90s | v7: v4 config (256d, drop 0.1, wd 0.1) at 90s budget, to check how it scales wit | 2.3494 | 2 | 0.000 |
| 8 | 90s | v8: v7 config at 90s with LR 1e-3 (only change from v7), testing gentler LR vs o | 2.3300 | 2 | 0.000 |

