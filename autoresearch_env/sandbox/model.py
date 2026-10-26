"""Baseline: a small byte-level GPT trained with AdamW for the time budget.
Edit anything in this file: architecture, optimizer, schedule, batch size, data sampling, ...
Keep the two entry points build_model() and train() with the same signatures."""
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 128
N_LAYER = 2
N_HEAD = 4
N_EMBD = 128
BATCH = 32
LR = 3e-4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, N_EMBD)
        self.pos = nn.Embedding(4096, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def build_model(vocab_size):
    return GPT(vocab_size)


def get_batch(data, device):
    ix = np.random.randint(0, len(data) - CTX - 1, BATCH)
    x = torch.stack([torch.from_numpy(data[i:i + CTX].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + CTX + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def train(model, train_data, budget_s, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    step = 0
    while time.time() - t0 < budget_s:
        x, y = get_batch(train_data, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
    print(f"steps={step}", flush=True)
