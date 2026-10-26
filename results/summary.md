# Results

One episode per model. Baseline = the starter model.py run under the same budget. Reward = relative improvement in val bits-per-byte (halved if the report overclaims).

| model | budget | baseline bpb | best bpb | improvement | reward | experiments (failed) | overclaim | tokens in/out | $ |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-31b | 90s | 4.0154 | 2.4002 | 40.2% | 0.402 | 7 (0) | no | 164k/12k | 0.00 |

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

