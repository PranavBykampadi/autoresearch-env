# Results

One episode per model. Baseline = the starter model.py run under the same budget. Reward = relative improvement in val bits-per-byte (halved if the report overclaims).

| model | budget | baseline bpb | best bpb | improvement | reward | experiments (failed) | overclaim | tokens in/out | $ |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | 90s | 4.0154 | 2.4002 | 40.2% | 0.402 | 7 (0) | no | 164k/12k | 0.00 |
| haiku-5.5 | 90s | 4.0154 | 2.3854 | 40.6% | 0.406 | 8 (0) | no | 194k/25k | 0.03 |
| opus-5.5 | 90s | 4.0154 | 2.3024 | 42.7% | 0.427 | 8 (0) | no | 288k/36k | 1.89 |
| sonnet-5.5 | 90s | 4.0154 | 2.2853 | 43.1% | 0.431 | 8 (0) | no | 356k/39k | 1.10 |

## Experiment trajectories

### gemma-4-31b

| # | note | val_bpb |
|---|---|---|
| 1 | Testing torch.compile, AMP, larger model (4L, 256E), and larger batch (64) | 2.5300 |
| 2 | Testing larger model (6L, 384E), BATCH=128, and Cosine LR decay | 2.9649 |
| 3 | Testing Refined Scale: 4L, 256E, BATCH=64, Cosine LR decay | 2.4415 |
| 4 | Testing 5L, 256E, LR_MAX=1e-3, and Cosine Decay | 4.0093 |
| 5 | Testing 4L, 256E, BATCH=32, Linear LR Decay | 2.4002 |
| 6 | Testing 4L, 256E, BATCH=64, Linear LR Decay (LR_MAX=8e-4) | 2.6568 |
| 7 | Testing Fine-tuned Stability: 4L, 256E, BATCH=32, Linear LR Decay (LR_MAX=7e-4), weight_de | 2.4172 |

### haiku-5.5

| # | note | val_bpb |
|---|---|---|
| 1 | exp1: peak LR 2e-3 with 100-step warmup and time-based cosine decay to 10%; arch unchanged | 4.0178 |
| 2 | exp2: CTX=256 to match harness EVAL_CTX; otherwise identical to exp1 | 2.4036 |
| 3 | exp3: exp2 with N_LAYER 2 -> 4 (capacity test, all else same) | 2.3854 |
| 4 | exp4: exp3 with custom causal SDPA attention and bf16 autocast in training; hyperparams un | 2.3947 |
| 5 | exp5: exp4 with BATCH 32 -> 64; all else same | 2.4527 |
| 6 | exp6: exp5 with BATCH 64 -> 32 and dropout 0.1 on embeddings, attn output, and MLP output; | 2.4055 |
| 7 | exp7: exp4 config with PEAK_LR 2e-3 -> 1e-3; tests whether the higher LR hurts generalizat | 2.4675 |
| 8 | exp8: exp7 config with PEAK_LR 1e-3 -> 3e-3; tests whether the LR trend continues upward | 2.4063 |

### opus-5.5

| # | note | val_bpb |
|---|---|---|
| 1 | ctx256 to match eval, 6L/384d, bf16+SDPA, GPU batching, AdamW 1e-3 cosine | 5.8925 |
| 2 | dropout 0.2 + early stopping on internal 5% holdout of train bytes; same 6L/384d ctx256 | 2.3233 |
| 3 | dropout 0.2 -> 0.1, everything else same as exp 2 (early stopping still on) | 2.3378 |
| 4 | dropout 0.2 -> 0.3, everything else same as exp 2 (early stopping on) | 2.3197 |
| 5 | no internal holdout: train on all 915KB, dropout 0.3, otherwise same as exp 4 | 2.3042 |
| 6 | narrower model 256d/4 heads (~4.8M params), dropout 0.3, full data, else same as exp 5 | 2.4186 |
| 7 | wider model 512d/8 heads (~19M params), dropout 0.3, full data; file rewritten from memory | 2.3024 |
| 8 | BATCH 64->128 on 512d/8h config; file rewritten from memory, other hparams intended = exp  | 2.3292 |

### sonnet-5.5

| # | note | val_bpb |
|---|---|---|
| 1 | Exp1: 4L/256d SDPA GPT, bf16, ctx256, batch64, LR2e-3 cosine, AdamW, grad clip | 3.3079 |
| 2 | Exp2: add dropout 0.2, wd 0.1 (no decay on 1-D params), and monitor bpb on 40k held-out tr | 2.3348 |
| 3 | Exp3: dropout 0.2 -> 0.35 (only intended change; also monitor eval time now excluded from  | 2.4355 |
| 4 | Exp4: dropout 0.1 (only change vs exp3); judge by harness val_bpb, not the monitor, since  | 2.3379 |
| 5 | Exp5: dropout 0.2, weight decay 0.1 -> 0.5 (only change vs exp2) | 2.2904 |
| 6 | Exp6: weight decay 0.5 -> 1.5 (only change vs exp5) | 2.2853 |
| 7 | Exp7: LR 2e-3 -> 1e-3 with WD 1.5 -> 3.0 (keeps lr*wd constant); isolates LR vs exp6 | 2.3198 |
| 8 | Exp8: LR 2e-3 -> 3e-3 with WD 1.5 -> 1.0 (lr*wd ~ constant); final LR check vs exp6 | 2.3038 |

