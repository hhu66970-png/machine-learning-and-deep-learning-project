# -*- coding: utf-8 -*-
"""Interpretability analysis from saved PEFT checkpoints (outputs/ckpts/).

Generates:
  - ef_cka.png         : per-block CKA(pretrained, adapted) -> where adaptation happens
  - ef_layer_adapt.png : per-block learned-update magnitude (LoRA / SSF / AdaptFormer)
  - ef_loss_landscape.png : 1-D filter-normalized loss landscape (flatness/sharpness)
Uses GPU; run AFTER the campaign (group 5 'interp' checkpoints must exist).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # torch+matplotlib both link OpenMP on Windows
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn as nn
import timm
from models.peft import build_model, apply_peft
from datasets import get_loaders, DATASET_INFO

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 170
plt.rcParams["savefig.bbox"] = "tight"

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(PROJ, "outputs", "ckpts")
FIG = os.path.join(PROJ, "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL = {"linear_probe": "Linear", "bitfit": "BitFit", "vpt": "VPT", "ssf": "SSF",
         "lora": "LoRA", "adaptformer": "AdaptFormer", "lora_ssf": "LoRA+SSF(ours)",
         "full_ft": "Full FT"}
ORDER = ["linear_probe", "bitfit", "vpt", "ssf", "lora", "adaptformer", "lora_ssf", "full_ft"]


def load_adapted(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    nc = DATASET_INFO[meta["dataset"]]
    model = build_model(nc, model_name=meta["model"])
    cfg = {"lora_r": int(meta.get("lora_r", 8)), "lora_alpha": int(meta.get("lora_r", 8)),
           "adapter_dim": int(meta.get("adapter_dim", 64)), "vpt_prompts": int(meta.get("vpt_prompts", 20))}
    model = apply_peft(model, meta["method"], cfg)
    missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
    return model, meta


def get_batch(meta, n=256):
    model0 = build_model(DATASET_INFO[meta["dataset"]], model_name=meta["model"])
    dc = timm.data.resolve_model_data_config(model0)
    tr, te, _ = get_loaders(meta["dataset"], dc["mean"], dc["std"],
                            batch_size=n, num_workers=4)
    x, y = next(iter(te))
    return x[:n].to(DEVICE), y[:n].to(DEVICE)


@torch.no_grad()
def block_feats(model, x):
    feats = []
    hooks = []

    def mk(i):
        def hook(m, inp, out):
            feats.append((i, out.detach().float().mean(dim=1).cpu()))  # mean over tokens
        return hook
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(mk(i)))
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(x)
    for h in hooks:
        h.remove()
    feats.sort(key=lambda t: t[0])
    return [f for _, f in feats]


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xty = (X.t() @ Y).norm() ** 2
    xtx = (X.t() @ X).norm()
    yty = (Y.t() @ Y).norm()
    return (xty / (xtx * yty + 1e-12)).item()


def cka_analysis(ckpts):
    plt.figure(figsize=(7.5, 5))
    table = {}
    for cp in ckpts:
        model, meta = load_adapted(cp)
        m = meta["method"]
        if m == "linear_probe":
            continue  # backbone identical -> CKA=1 trivially
        x, _ = get_batch(meta)
        ref = build_model(DATASET_INFO[meta["dataset"]], model_name=meta["model"]).to(DEVICE)
        model = model.to(DEVICE)
        fa = block_feats(model, x)
        fr = block_feats(ref, x)
        ckas = [linear_cka(a, b) for a, b in zip(fa, fr)]
        table[m] = ckas
        plt.plot(range(1, len(ckas) + 1), ckas, "o-", label=LABEL.get(m, m))
        del model, ref
        torch.cuda.empty_cache()
    plt.xlabel("Transformer 层")
    plt.ylabel("CKA(预训练特征, 适配后特征)")
    plt.title("各层特征相似度：CKA 越低=该层被适配得越多 (Pets, ViT-B)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG, "ef_cka.png"))
    plt.close()
    print("saved ef_cka.png")
    return table


def layer_adapt(ckpts):
    plt.figure(figsize=(7.5, 5))
    found = False
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        meta = ck["meta"]; m = meta["method"]; sd = ck["trainable"]
        nblk = 24 if "large" in meta["model"] else (12 if "base" in meta["model"] else 12)
        mags = [0.0] * nblk
        if m in ("lora", "lora_ssf"):
            for i in range(nblk):
                bq = sd.get(f"blocks.{i}.attn.qkv.lora_B_q.weight")
                aq = sd.get(f"blocks.{i}.attn.qkv.lora_A_q.weight")
                if bq is not None and aq is not None:
                    mags[i] = (bq.float() @ aq.float()).norm().item()
            label = "LoRA ΔW 范数"
        elif m == "ssf":
            for i in range(nblk):
                g = sd.get(f"blocks.{i}.ssf_n1.scale")
                b = sd.get(f"blocks.{i}.ssf_n1.shift")
                if g is not None:
                    mags[i] = ((g.float() - 1).norm() + b.float().norm()).item()
            label = "SSF |γ-1|+|β| 范数"
        elif m == "adaptformer":
            for i in range(nblk):
                up = sd.get(f"blocks.{i}.adapter.up.weight")
                dn = sd.get(f"blocks.{i}.adapter.down.weight")
                if up is not None and dn is not None:
                    mags[i] = (up.float() @ dn.float()).norm().item()
            label = "Adapter ΔW 范数"
        else:
            continue
        found = True
        plt.plot(range(1, nblk + 1), mags, "o-", label=f"{LABEL.get(m,m)} ({label})")
    if found:
        plt.xlabel("Transformer 层")
        plt.ylabel("学习到的更新量（范数）")
        plt.title("各层适配强度：适配主要发生在哪些层 (Pets, ViT-B)")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIG, "ef_layer_adapt.png"))
        print("saved ef_layer_adapt.png")
    plt.close()


def loss_landscape(ckpts, n_alpha=21, span=1.0):
    crit = nn.CrossEntropyLoss()
    plt.figure(figsize=(7.5, 5))
    for cp in ckpts:
        model, meta = load_adapted(cp)
        m = meta["method"]
        model = model.to(DEVICE)
        x, y = get_batch(meta, n=256)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            continue
        theta0 = [p.detach().clone() for p in params]
        # filter-normalized random direction
        dirs = []
        for p, t0 in zip(params, theta0):
            d = torch.randn_like(p)
            d = d * (t0.norm() / (d.norm() + 1e-12))
            dirs.append(d)
        alphas = np.linspace(-span, span, n_alpha)
        losses = []
        model.eval()
        with torch.no_grad():
            for a in alphas:
                for p, t0, d in zip(params, theta0, dirs):
                    p.copy_(t0 + a * d)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = crit(model(x), y).item()
                losses.append(loss)
            for p, t0 in zip(params, theta0):
                p.copy_(t0)
        losses = np.array(losses) - min(losses)
        plt.plot(alphas, losses, "-", label=LABEL.get(m, m))
        del model
        torch.cuda.empty_cache()
    plt.xlabel("沿随机方向的扰动 α")
    plt.ylabel("损失增量 (相对最小值)")
    plt.title("一维损失地形：曲线越平坦=极小值越平坦/泛化越好 (Pets, ViT-B)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG, "ef_loss_landscape.png"))
    plt.close()
    print("saved ef_loss_landscape.png")


def main():
    ckpts = sorted(glob.glob(os.path.join(CKPT, "interp__*.pt")))
    if not ckpts:
        print("no interp checkpoints yet (run campaign group 5 first)")
        return
    print(f"found {len(ckpts)} checkpoints")
    # order by method
    def key(p):
        for i, m in enumerate(ORDER):
            if f"__{m}__" in p:
                return i
        return 99
    ckpts.sort(key=key)
    try:
        cka_analysis(ckpts)
    except Exception as e:
        print("[warn] cka failed:", e)
    try:
        layer_adapt(ckpts)
    except Exception as e:
        print("[warn] layer_adapt failed:", e)
    try:
        loss_landscape(ckpts)
    except Exception as e:
        print("[warn] loss_landscape failed:", e)
    print("INTERP_DONE")


if __name__ == "__main__":
    main()
