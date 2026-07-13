"""Sanity-check every PEFT method: forward, backward, grad flow, param counts."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn as nn
from models.peft import build_model, apply_peft, METHODS
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
x = torch.randn(2, 3, 224, 224, device=device)
y = torch.randint(0, 10, (2,), device=device)
crit = nn.CrossEntropyLoss()

print(f"{'method':14s} {'trainable':>14s} {'total':>14s} {'pct%':>8s}  fwd  bwd  grad_ok")
for m in METHODS:
    model = build_model(10)
    model = apply_peft(model, m)
    model = model.to(device)
    model.train()
    tr, tot, pct = utils.count_parameters(model)
    try:
        out = model(x)
        fwd_ok = tuple(out.shape) == (2, 10)
        loss = crit(out, y)
        loss.backward()
        # every trainable param must receive a gradient
        miss = [n for n, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0)]
        grad_ok = len(miss) == 0
        print(f"{m:14s} {tr:>14,} {tot:>14,} {pct:>8.3f}  {'Y' if fwd_ok else 'N':>3}  "
              f"{'Y':>3}  {'Y' if grad_ok else 'N(' + str(len(miss)) + ')'}")
        if miss:
            print("    no-grad params (first 5):", miss[:5])
    except Exception as e:
        print(f"{m:14s} ERROR: {type(e).__name__}: {e}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
print("SANITY_DONE")
