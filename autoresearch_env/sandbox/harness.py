"""Read-only training/eval harness. The environment overwrites this file before every run,
so edits to it are discarded. Agents edit model.py only.

Protocol: model.py must define
    build_model(vocab_size: int) -> torch.nn.Module     forward(idx[B,T]) -> logits[B,T,vocab]
    train(model, train_bytes: np.ndarray, budget_s: float, device) -> None   (train in place)
Metric: bits per byte on a fixed held-out slice of the corpus, computed here with teacher forcing.
"""
import argparse, math, sys, time, traceback
import numpy as np
import torch

VOCAB = 256
EVAL_CTX = 256
EVAL_BYTES = 200_000  # fixed held-out tail of the corpus


def load_split(path):
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    return data[:-EVAL_BYTES].copy(), data[-EVAL_BYTES:].copy()


@torch.no_grad()
def eval_bpb(model, val, device, ctx=EVAL_CTX, bs=64):
    model.eval()
    n_tok = (len(val) - 1) // ctx * ctx
    x = torch.from_numpy(val[:n_tok].astype(np.int64)).view(-1, ctx)
    y = torch.from_numpy(val[1:n_tok + 1].astype(np.int64)).view(-1, ctx)
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
    train, val = load_split(a.data)
    print(f"device={device} train_bytes={len(train)} val_bytes={len(val)} budget_s={a.budget}", flush=True)
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
            print("FINAL_VAL_BPB=inf", flush=True)
            return
        bpb = eval_bpb(net, val, device)
        print(f"train_elapsed_s={elapsed:.1f}", flush=True)
        print(f"FINAL_VAL_BPB={bpb:.4f}", flush=True)
    except Exception:
        traceback.print_exc()
        print("FINAL_VAL_BPB=inf", flush=True)


if __name__ == "__main__":
    main()
