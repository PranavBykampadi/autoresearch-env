"""Read-only training/eval harness. The environment overwrites this file before every run,
so edits to it are discarded. Agents edit model.py only.

Protocol: model.py must define
    build_model(vocab_size: int) -> torch.nn.Module     forward(idx[B,T]) -> logits[B,T,vocab]
    train(model, train_bytes: np.ndarray, budget_s: float, device) -> None   (train in place)

Data: the corpus tail is held out in two slices. VAL is reported to the agent after every run.
TEST is never shown; the environment scores the final submission on it with a fresh seed. Reading the
data file from model.py is against the rules and is checked for.
Metric: bits per byte with teacher forcing at context EVAL_CTX.
"""
import argparse, math, os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # harness runs under python -I
import numpy as np
import torch

VOCAB = 256
EVAL_CTX = 256
VAL_BYTES = 200_000
TEST_BYTES = 200_000


def load_split(path):
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    train = data[:-(VAL_BYTES + TEST_BYTES)].copy()
    val = data[-(VAL_BYTES + TEST_BYTES):-TEST_BYTES].copy()
    test = data[-TEST_BYTES:].copy()
    return train, val, test


@torch.no_grad()
def eval_bpb(model, split, device, ctx=EVAL_CTX, bs=64):
    model.eval()
    n_tok = (len(split) - 1) // ctx * ctx
    x = torch.from_numpy(split[:n_tok].astype(np.int64)).view(-1, ctx)
    y = torch.from_numpy(split[1:n_tok + 1].astype(np.int64)).view(-1, ctx)
    nll = 0.0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs].to(device), y[i:i + bs].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(xb)
        logits = logits.float()
        if logits.shape[-1] != VOCAB:
            raise ValueError(f"logits last dim must be {VOCAB}, got {logits.shape[-1]}")
        nll += torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), yb.view(-1), reduction="sum").item()
    return nll / n_tok / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--budget", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = load_split(a.data)
    print(f"device={device} train_bytes={len(train)} val_bytes={len(val)} budget_s={a.budget} seed={a.seed}", flush=True)
    # pay one-time lazy-import costs (optimizer construction is ~9s on this filesystem) before the clock starts
    _w = torch.zeros(4, device=device, requires_grad=True)
    torch.optim.AdamW([_w], lr=1e-3).step(); (_w * 2).sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    try:
        import model as M
        net = M.build_model(VOCAB).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"params={n_params}", flush=True)
        t0 = time.time()
        M.train(net, train, a.budget, device)
        elapsed = time.time() - t0
        if elapsed > a.budget + 30:  # setup inside train() is allowed, but not a second training run
            print(f"OVER_BUDGET elapsed={elapsed:.1f}s", flush=True)
            print("FINAL_VAL_BPB=inf", flush=True); print("FINAL_TEST_BPB=inf", flush=True)
            return
        vb = eval_bpb(net, val, device)
        tb = eval_bpb(net, test, device)
        print(f"train_elapsed_s={elapsed:.1f}", flush=True)
        print(f"FINAL_VAL_BPB={vb:.4f}", flush=True)
        print(f"FINAL_TEST_BPB={tb:.4f}", flush=True)
    except Exception:
        traceback.print_exc()
        print("FINAL_VAL_BPB=inf", flush=True); print("FINAL_TEST_BPB=inf", flush=True)


if __name__ == "__main__":
    main()
