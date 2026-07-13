# -*- coding: utf-8 -*-
"""Extract the NEW experiment numbers (full ablations, ViT-S 8-dataset, multi-seed,
complete data-efficiency) into a markdown digest for the report-writing agents."""
import os
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(PROJ, "outputs", "results.csv"))
for c in ["best_acc", "trainable_params", "pct_trainable", "lora_r", "adapter_dim",
          "vpt_prompts", "train_fraction", "seed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

ML = {"linear_probe": "Linear", "bitfit": "BitFit", "vpt": "VPT", "ssf": "SSF",
      "lora": "LoRA", "adaptformer": "AdaptFormer", "lora_ssf": "LoRA+SSF(ours)", "full_ft": "Full FT"}
DL = {"cifar100": "CIFAR-100", "flowers102": "Flowers-102", "pets": "Pets", "dtd": "DTD",
      "cifar10": "CIFAR-10", "svhn": "SVHN", "eurosat": "EuroSAT", "gtsrb": "GTSRB"}
MO = ["linear_probe", "bitfit", "vpt", "ssf", "lora", "adaptformer", "lora_ssf", "full_ft"]
DO = ["cifar100", "flowers102", "pets", "dtd", "cifar10", "svhn", "eurosat", "gtsrb"]
out = []


def w(s=""):
    out.append(s)


w("# 报告增补数据 digest（新实验结果，供撰写各章节引用，数字以此为准）\n")

# ---- ViT-S 8-dataset broad ----
w("## A. ViT-S 在全部 8 个数据集上的结果（图6缩放已含，此处给完整表，用于07缩放章节）")
vs = df[(df.tag == "main") & (df.backbone == "vit_s") & (df.train_fraction == 1.0)]
vb = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0) & (df.seed == 42)]
w("| 方法 | " + " | ".join(DL[d] for d in DO) + " | ViT-S均值 | ViT-B均值 |")
w("|" + "---|" * (len(DO) + 3))
for m in MO:
    row = [ML[m]]
    svals = []
    for d in DO:
        v = vs[(vs.method == m) & (vs.dataset == d)]["best_acc"]
        row.append(f"{v.iloc[0]:.2f}" if len(v) else "-")
        if len(v):
            svals.append(v.iloc[0])
    bvals = vb[vb.method == m]["best_acc"]
    row.append(f"{sum(svals)/len(svals):.2f}" if svals else "-")
    row.append(f"{bvals.mean():.2f}" if len(bvals) else "-")
    w("| " + " | ".join(row) + " |")
w("")

# ---- LoRA alpha ----
w("## B. LoRA 缩放系数 α 消融（DTD, ViT-B, r=8 固定）-> 图14 (ef_abl_lora_alpha.png)")
al = df[df.tag.str.startswith("abl_loraAlpha", na=False)].copy()
al["alpha"] = al.tag.str.extract(r"_a(\d+)").astype(float)
al = al.sort_values("alpha")
w("| α | 缩放(α/r) | Top-1(%) |")
w("|---|---|---|")
for _, r in al.iterrows():
    w(f"| {int(r.alpha)} | {r.alpha/8:.2f} | {r.best_acc:.2f} |")
w("")

# ---- Adapter dim ----
w("## C. AdaptFormer 瓶颈维度消融（DTD, ViT-B）-> 图15 (ef_abl_adapter_dim.png)")
ad = df[df.tag == "abl_adapterDim"].sort_values("adapter_dim")
w("| 瓶颈维度 | 可训练参数 | Top-1(%) |")
w("|---|---|---|")
for _, r in ad.iterrows():
    w(f"| {int(r.adapter_dim)} | {int(r.trainable_params):,} | {r.best_acc:.2f} |")
w("")

# ---- VPT prompts ----
w("## D. VPT 提示 token 数消融（DTD, ViT-B）-> 图16 (ef_abl_vpt_prompts.png)")
vp = df[df.tag == "abl_vptPrompts"].sort_values("vpt_prompts")
w("| 提示数 | 可训练参数 | Top-1(%) |")
w("|---|---|---|")
for _, r in vp.iterrows():
    w(f"| {int(r.vpt_prompts)} | {int(r.trainable_params):,} | {r.best_acc:.2f} |")
w("")

# ---- LR sensitivity ----
w("## E. 学习率敏感性（DTD, ViT-B）-> 图17 (ef_abl_lr.png)")
lr = df[df.tag.str.startswith("abl_lr_", na=False)].copy()
lr["lr_val"] = lr.tag.str.extract(r"_([0-9.e+-]+)$").astype(float)
lr["mname"] = lr.tag.str.extract(r"abl_lr_(\w+?)_[0-9]")
w("| 方法 | lr=1e-4 | lr=3e-4 | lr=1e-3 | lr=3e-3 |")
w("|---|---|---|---|---|")
for mn in ["lora", "ssf", "full_ft"]:
    s = lr[lr.mname == mn].sort_values("lr_val")
    vals = {round(r.lr_val, 6): r.best_acc for _, r in s.iterrows()}
    w(f"| {ML[mn]} | " + " | ".join(f"{vals.get(x, float('nan')):.2f}" for x in [1e-4, 3e-4, 1e-3, 3e-3]) + " |")
w("")

# ---- Multi-seed variance ----
w("## F. 多随机种子方差（ViT-B 核心4数据集 flowers/dtd/pets/cifar100）-> 图18 (ef_seed_variance.png)")
core = ["flowers102", "dtd", "pets", "cifar100"]
ms = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0) & (df.dataset.isin(core))]
w("各方法在核心4数据集上、跨已完成种子(42/123)的均值±标准差(对每方法先按数据集取各种子均值，再跨数据集汇总):")
w("| 方法 | 种子数(每数据集) | 核心4平均 Top-1(%) | 跨种子标准差(百分点) |")
w("|---|---|---|---|")
for m in MO:
    sub = ms[ms.method == m]
    if sub.empty:
        continue
    nseed = sub.groupby("dataset").seed.nunique().max()
    mean = sub.best_acc.mean()
    # std across seeds: per dataset std then avg
    stds = sub.groupby("dataset").best_acc.std()
    w(f"| {ML[m]} | {int(nseed)} | {mean:.2f} | {stds.mean():.3f} |")
w("\n注:seed2024 与 ViT-L 仍在后台运行,完成后可再扩充为3种子与三点缩放。")
w("")

# ---- Data-efficiency complete (cifar100) ----
w("## G. 数据效率完整数据（ViT-B; 训练比例 5/10/25/50/100%）-> 图8(DTD) 图9(CIFAR-100)")
for ds in ["dtd", "cifar100"]:
    w(f"### {DL[ds]}")
    fe = df[(df.tag == "fracEff") & (df.dataset == ds)]
    full = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.dataset == ds) & (df.seed == 42) & (df.train_fraction == 1.0)]
    methods = ["linear_probe", "bitfit", "lora", "adaptformer", "lora_ssf"]
    w("| 方法 | 5% | 10% | 25% | 50% | 100% |")
    w("|---|---|---|---|---|---|")
    for m in methods:
        vals = []
        for fr in [0.05, 0.1, 0.25, 0.5]:
            v = fe[(fe.method == m) & (abs(fe.train_fraction - fr) < 1e-6)]["best_acc"]
            vals.append(f"{v.iloc[0]:.2f}" if len(v) else "-")
        f100 = full[full.method == m]["best_acc"]
        vals.append(f"{f100.iloc[0]:.2f}" if len(f100) else "-")
        w(f"| {ML[m]} | " + " | ".join(vals) + " |")
    w("")

open(os.path.join(PROJ, "outputs", "新数据digest.md"), "w", encoding="utf-8").write("\n".join(out))
print("saved outputs/新数据digest.md")
print("\n".join(out))
