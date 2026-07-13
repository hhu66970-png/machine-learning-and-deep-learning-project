# -*- coding: utf-8 -*-
"""Comprehensive figures & tables for the EXPANDED PEFT study.

Reads outputs/results.csv (schema with run_id/backbone/seed/train_fraction).
Robust to partial data: only renders what is available. Re-runnable anytime.
Outputs figures into figures/ (prefix ef_) and tables into outputs/.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 170
plt.rcParams["savefig.bbox"] = "tight"

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "outputs", "results.csv")
CURVE_DIR = os.path.join(PROJ, "outputs", "curves")
FIG = os.path.join(PROJ, "figures")
OUT = os.path.join(PROJ, "outputs")
os.makedirs(FIG, exist_ok=True)

METHOD_LABELS = {
    "linear_probe": "Linear", "bitfit": "BitFit", "vpt": "VPT", "ssf": "SSF",
    "lora": "LoRA", "adaptformer": "AdaptFormer", "lora_ssf": "LoRA+SSF(ours)",
    "full_ft": "Full FT",
}
METHOD_ORDER = ["linear_probe", "bitfit", "vpt", "ssf", "lora", "adaptformer", "lora_ssf", "full_ft"]
DS_LABEL = {"cifar100": "CIFAR-100", "flowers102": "Flowers-102", "pets": "Pets", "dtd": "DTD",
            "cifar10": "CIFAR-10", "svhn": "SVHN", "eurosat": "EuroSAT", "gtsrb": "GTSRB"}
DS_ORDER = ["cifar100", "flowers102", "pets", "dtd", "cifar10", "svhn", "eurosat", "gtsrb"]
CORE = ["flowers102", "dtd", "pets", "cifar100"]
BK_LABEL = {"vit_s": "ViT-S", "vit_b": "ViT-B", "vit_l": "ViT-L"}
BK_PARAMS = {"vit_s": 22, "vit_b": 86, "vit_l": 304}
BK_ORDER = ["vit_s", "vit_b", "vit_l"]
COLORS = dict(zip(METHOD_ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))


def ml(m):
    return METHOD_LABELS.get(m, m)


def load():
    if not os.path.exists(RESULTS):
        return pd.DataFrame()
    df = pd.read_csv(RESULTS)
    for c in ("best_acc", "final_acc", "pct_trainable", "trainable_params",
              "peak_mem_mb", "time_per_epoch_s", "throughput_img_s", "train_fraction",
              "lora_r", "adapter_dim", "vpt_prompts", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "backbone" not in df.columns:
        df["backbone"] = "vit_b"
    if "train_fraction" not in df.columns:
        df["train_fraction"] = 1.0
    return df


def present(values, order):
    s = set(values)
    return [x for x in order if x in s]


# --------------------------------------------------------------------------- #
# Primary ViT-B results (seed-averaged)
# --------------------------------------------------------------------------- #
def primary_vitb(df):
    m = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0)]
    if m.empty:
        return None
    agg = m.groupby(["method", "dataset"]).agg(
        acc=("best_acc", "mean"), acc_std=("best_acc", "std"),
        pct=("pct_trainable", "mean"), params=("trainable_params", "mean"),
        nseed=("best_acc", "count")).reset_index()
    return agg


def fig_acc_vs_params(df):
    agg = primary_vitb(df)
    if agg is None:
        return
    dss = present(agg.dataset, DS_ORDER)
    n = len(dss)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.8 * rows), squeeze=False)
    for ax, ds in zip(axes.flat, dss):
        sub = agg[agg.dataset == ds]
        for m in present(sub.method, METHOD_ORDER):
            r = sub[sub.method == m].iloc[0]
            ax.scatter(r["params"], r["acc"], s=70, color=COLORS[m], zorder=3)
            ax.annotate(ml(m), (r["params"], r["acc"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_xlabel("可训练参数量(对数)", fontsize=9)
        ax.set_ylabel("Top-1 (%)", fontsize=9)
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("准确率—可训练参数量权衡（ViT-B，左上角为最优）", fontsize=13, y=1.005)
    fig.savefig(os.path.join(FIG, "ef_acc_vs_params.png"))
    plt.close(fig)
    print("saved ef_acc_vs_params.png")


def fig_heatmap(df):
    agg = primary_vitb(df)
    if agg is None:
        return
    ms = present(agg.method, METHOD_ORDER)
    dss = present(agg.dataset, DS_ORDER)
    mat = np.full((len(ms), len(dss)), np.nan)
    for i, m in enumerate(ms):
        for j, ds in enumerate(dss):
            v = agg[(agg.method == m) & (agg.dataset == ds)]["acc"]
            if len(v):
                mat[i, j] = v.iloc[0]
    fig, ax = plt.subplots(figsize=(1.15 * len(dss) + 3, 0.62 * len(ms) + 2))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(dss)))
    ax.set_xticklabels([DS_LABEL.get(d, d) for d in dss], rotation=30, ha="right")
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([ml(m) for m in ms])
    for i in range(len(ms)):
        for j in range(len(dss)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("准确率热力图（ViT-B，方法 × 数据集）")
    fig.colorbar(im, ax=ax, label="Top-1 (%)")
    fig.savefig(os.path.join(FIG, "ef_heatmap.png"))
    plt.close(fig)
    print("saved ef_heatmap.png")


def fig_acc_bars(df):
    agg = primary_vitb(df)
    if agg is None:
        return
    ms = present(agg.method, METHOD_ORDER)
    dss = present(agg.dataset, DS_ORDER)
    x = np.arange(len(ms))
    w = 0.8 / max(len(dss), 1)
    fig, ax = plt.subplots(figsize=(1.5 * len(ms) + 3, 5))
    for i, ds in enumerate(dss):
        sub = agg[agg.dataset == ds]
        vals = [sub[sub.method == m]["acc"].iloc[0] if m in set(sub.method) else np.nan for m in ms]
        ax.bar(x + i * w, vals, w, label=DS_LABEL.get(ds, ds))
    ax.set_xticks(x + w * (len(dss) - 1) / 2)
    ax.set_xticklabels([ml(m) for m in ms], rotation=20, ha="right")
    ax.set_ylabel("Top-1 (%)")
    ax.set_ylim(bottom=max(0, np.nanmin(agg.acc) - 5))
    ax.set_title("各方法在各数据集上的准确率（ViT-B）")
    ax.legend(title="数据集", ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_acc_bars.png"))
    plt.close(fig)
    print("saved ef_acc_bars.png")


def fig_avg_rank(df):
    agg = primary_vitb(df)
    if agg is None:
        return
    piv = agg.pivot_table(index="method", columns="dataset", values="acc")
    ranks = piv.rank(ascending=False, axis=0)
    avg_rank = ranks.mean(axis=1).reindex([m for m in METHOD_ORDER if m in ranks.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = avg_rank.sort_values().index
    ax.barh([ml(m) for m in order], avg_rank[order].values,
            color=[COLORS[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlabel("平均排名（越小越好，跨所有数据集）")
    ax.set_title("各方法平均排名（ViT-B）")
    for i, m in enumerate(order):
        ax.text(avg_rank[m], i, f" {avg_rank[m]:.2f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_avg_rank.png"))
    plt.close(fig)
    print("saved ef_avg_rank.png")


# --------------------------------------------------------------------------- #
# Backbone scaling
# --------------------------------------------------------------------------- #
def fig_scaling(df):
    m = df[(df.tag == "main") & (df.train_fraction == 1.0)]
    bks = present(m.backbone, BK_ORDER)
    if len(bks) < 2:
        return
    # average over core datasets & seeds, per (backbone, method)
    mm = m[m.dataset.isin(CORE)]
    agg = mm.groupby(["backbone", "method"]).agg(acc=("best_acc", "mean")).reset_index()
    ms = present(agg.method, METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [BK_PARAMS[b] for b in bks]
    for m_ in ms:
        ys = []
        for b in bks:
            v = agg[(agg.backbone == b) & (agg.method == m_)]["acc"]
            ys.append(v.iloc[0] if len(v) else np.nan)
        ax.plot(xs, ys, "o-", color=COLORS[m_], label=ml(m_))
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("骨干参数量 (M, 对数)")
    ax.set_ylabel("核心数据集平均 Top-1 (%)")
    ax.set_title("PEFT 方法随骨干规模的缩放（ViT-S → B → L）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_scaling.png"))
    plt.close(fig)
    print("saved ef_scaling.png")

    # gap of best PEFT vs full_ft per backbone
    fig, ax = plt.subplots(figsize=(7.5, 5))
    peft = [x for x in ms if x not in ("full_ft", "linear_probe")]
    for b in bks:
        full = agg[(agg.backbone == b) & (agg.method == "full_ft")]["acc"]
        full = full.iloc[0] if len(full) else np.nan
        bestp = np.nanmax([agg[(agg.backbone == b) & (agg.method == p)]["acc"].iloc[0]
                           for p in peft if p in set(agg[agg.backbone == b].method)])
        ax.bar(BK_LABEL[b], bestp - full,
               color="tab:green" if bestp >= full else "tab:red")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("最优PEFT − 全量微调 (百分点)")
    ax.set_title("最优PEFT相对全量微调的优势随规模变化")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_scaling_gap.png"))
    plt.close(fig)
    print("saved ef_scaling_gap.png")


# --------------------------------------------------------------------------- #
# Multi-seed variance
# --------------------------------------------------------------------------- #
def fig_seed_var(df):
    m = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0)
           & (df.dataset.isin(CORE))]
    cnt = m.groupby(["method", "dataset"])["best_acc"].count()
    if cnt.max() < 2:
        return  # need >=2 seeds somewhere
    agg = m.groupby("method").agg(acc=("best_acc", "mean"), std=("best_acc", "std"),
                                  n=("best_acc", "count")).reset_index()
    ms = present(agg.method, METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(ms))
    means = [agg[agg.method == m_]["acc"].iloc[0] for m_ in ms]
    stds = [agg[agg.method == m_]["std"].iloc[0] for m_ in ms]
    ax.bar(xs, means, yerr=stds, capsize=4, color=[COLORS[m] for m in ms])
    ax.set_xticks(xs)
    ax.set_xticklabels([ml(m) for m in ms], rotation=20, ha="right")
    ax.set_ylabel("核心数据集平均 Top-1 (%)")
    ax.set_ylim(bottom=min(means) - 3)
    n = int(agg["n"].max())
    ax.set_title(f"多随机种子均值±标准差（ViT-B，{n} 个种子）")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_seed_variance.png"))
    plt.close(fig)
    print("saved ef_seed_variance.png")


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def fig_ablations(df):
    # LoRA rank
    a = df[df.tag == "abl_loraR"].sort_values("lora_r")
    if not a.empty:
        fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
        ax1.plot(a.lora_r, a.best_acc, "o-", color="tab:blue", label="准确率")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(a.lora_r)
        ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax1.set_xlabel("LoRA 秩 r")
        ax1.set_ylabel("Top-1 (%)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(a.lora_r, a.trainable_params / 1e3, "s--", color="tab:red")
        ax2.set_ylabel("可训练参数 (千)", color="tab:red")
        ax1.set_title("LoRA 秩 r 消融 (DTD, ViT-B)")
        ax1.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, "ef_abl_lora_rank.png"))
        plt.close(fig)
        print("saved ef_abl_lora_rank.png")

    # LoRA alpha (tag prefix)
    al = df[df.tag.str.startswith("abl_loraAlpha", na=False)].copy()
    if not al.empty:
        al["alpha"] = al.tag.str.extract(r"_a(\d+)").astype(float)
        al = al.sort_values("alpha")
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(al.alpha, al.best_acc, "o-", color="tab:purple")
        ax.set_xlabel("LoRA 缩放系数 α (r=8)")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title("LoRA α 消融 (DTD, ViT-B)")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, "ef_abl_lora_alpha.png"))
        plt.close(fig)
        print("saved ef_abl_lora_alpha.png")

    # Adapter dim
    ad = df[df.tag == "abl_adapterDim"].sort_values("adapter_dim")
    if not ad.empty:
        fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
        ax1.plot(ad.adapter_dim, ad.best_acc, "o-", color="tab:green")
        ax1.set_xscale("log", base=2)
        ax1.set_xticks(ad.adapter_dim)
        ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax1.set_xlabel("Adapter 瓶颈维度")
        ax1.set_ylabel("Top-1 (%)", color="tab:green")
        ax2 = ax1.twinx()
        ax2.plot(ad.adapter_dim, ad.trainable_params / 1e3, "s--", color="tab:red")
        ax2.set_ylabel("可训练参数 (千)", color="tab:red")
        ax1.set_title("AdaptFormer 瓶颈维度消融 (DTD, ViT-B)")
        ax1.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, "ef_abl_adapter_dim.png"))
        plt.close(fig)
        print("saved ef_abl_adapter_dim.png")

    # VPT prompts
    vp = df[df.tag == "abl_vptPrompts"].sort_values("vpt_prompts")
    if not vp.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(vp.vpt_prompts, vp.best_acc, "o-", color="tab:orange")
        ax.set_xlabel("VPT 提示 token 数")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title("VPT 提示数量消融 (DTD, ViT-B)")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, "ef_abl_vpt_prompts.png"))
        plt.close(fig)
        print("saved ef_abl_vpt_prompts.png")

    # LR sensitivity
    lr = df[df.tag.str.startswith("abl_lr_", na=False)].copy()
    if not lr.empty:
        lr["lr_val"] = lr.tag.str.extract(r"_([0-9.e+-]+)$").astype(float)
        lr["mname"] = lr.tag.str.extract(r"abl_lr_(\w+?)_[0-9]")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for mn in lr.mname.dropna().unique():
            s = lr[lr.mname == mn].sort_values("lr_val")
            ax.plot(s.lr_val, s.best_acc, "o-", label=ml(mn))
        ax.set_xscale("log")
        ax.set_xlabel("学习率")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title("学习率敏感性 (DTD, ViT-B)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, "ef_abl_lr.png"))
        plt.close(fig)
        print("saved ef_abl_lr.png")


# --------------------------------------------------------------------------- #
# Data efficiency
# --------------------------------------------------------------------------- #
def fig_data_efficiency(df):
    fe = df[df.tag == "fracEff"]
    if fe.empty:
        return
    # also pull the 100% point from main (vit_b)
    full = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0)]
    dss = present(fe.dataset, DS_ORDER)
    for ds in dss:
        sub = fe[fe.dataset == ds]
        ms = present(sub.method, METHOD_ORDER)
        fig, ax = plt.subplots(figsize=(7, 5))
        for m_ in ms:
            s = sub[sub.method == m_].sort_values("train_fraction")
            fracs = list(s.train_fraction * 100)
            accs = list(s.best_acc)
            f100 = full[(full.method == m_) & (full.dataset == ds)]["best_acc"]
            if len(f100):
                fracs.append(100.0)
                accs.append(f100.mean())
            ax.plot(fracs, accs, "o-", color=COLORS.get(m_), label=ml(m_))
        ax.set_xlabel("训练数据比例 (%)")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title(f"数据效率：{DS_LABEL.get(ds, ds)} (ViT-B)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, f"ef_data_efficiency_{ds}.png"))
        plt.close(fig)
        print(f"saved ef_data_efficiency_{ds}.png")


# --------------------------------------------------------------------------- #
# Convergence curves (per backbone+dataset, primary seed)
# --------------------------------------------------------------------------- #
def fig_convergence(df):
    files = glob.glob(os.path.join(CURVE_DIR, "main__vit_b__*__s42__f100__*.json"))
    by_ds = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        meta = d["meta"]
        if meta.get("train_fraction", 1.0) not in (1.0, "1.0"):
            continue
        by_ds.setdefault(meta["dataset"], []).append((meta["method"], d["curve"]))
    for ds, items in by_ds.items():
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        items.sort(key=lambda x: METHOD_ORDER.index(x[0]) if x[0] in METHOD_ORDER else 99)
        for m_, curve in items:
            ep = [c["epoch"] for c in curve]
            acc = [c["test_acc"] for c in curve]
            ax.plot(ep, acc, marker="o", ms=3, label=ml(m_), color=COLORS.get(m_))
        ax.set_xlabel("训练轮数 (epoch)")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title(f"{DS_LABEL.get(ds, ds)} 收敛曲线 (ViT-B)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, f"ef_convergence_{ds}.png"))
        plt.close(fig)
        print(f"saved ef_convergence_{ds}.png")


# --------------------------------------------------------------------------- #
# Efficiency (memory / time) on ViT-B
# --------------------------------------------------------------------------- #
def fig_efficiency(df):
    m = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0)]
    if m.empty:
        return
    ms = present(m.method, METHOD_ORDER)
    mem = [m[m.method == x]["peak_mem_mb"].mean() for x in ms]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar([ml(x) for x in ms], mem, color=[COLORS[x] for x in ms])
    a1.set_ylabel("峰值显存 (MB)")
    a1.set_title("训练显存开销 (ViT-B)")
    a1.tick_params(axis="x", rotation=20)
    # storage per task: trainable params * 4 bytes
    store = [m[m.method == x]["trainable_params"].mean() * 4 / 1e6 for x in ms]
    a2.bar([ml(x) for x in ms], store, color=[COLORS[x] for x in ms])
    a2.set_yscale("log")
    a2.set_ylabel("每任务存储 (MB, 对数)")
    a2.set_title("每个下游任务需保存的参数大小 (ViT-B)")
    a2.tick_params(axis="x", rotation=20)
    for a in (a1, a2):
        a.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "ef_efficiency.png"))
    plt.close(fig)
    print("saved ef_efficiency.png")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def make_tables(df):
    agg = primary_vitb(df)
    if agg is None:
        return
    piv = agg.pivot_table(index="method", columns="dataset", values="acc")
    piv = piv.reindex([m for m in METHOD_ORDER if m in piv.index])
    piv = piv[[d for d in DS_ORDER if d in piv.columns]]
    piv.columns = [DS_LABEL.get(c, c) for c in piv.columns]
    piv["平均"] = piv.mean(axis=1).round(2)
    pct = agg.groupby("method")["pct"].mean()
    piv.insert(0, "可训练%", [round(pct.get(m, np.nan), 3) for m in piv.index])
    piv.index = [ml(m) for m in piv.index]
    piv.to_csv(os.path.join(OUT, "ef_table_main.csv"), encoding="utf-8-sig")
    with open(os.path.join(OUT, "ef_table_main.md"), "w", encoding="utf-8") as f:
        f.write("# ViT-B 主结果表（8 数据集，种子平均 Top-1 %）\n\n")
        f.write(piv.to_markdown() + "\n")

    # scaling table
    m = df[(df.tag == "main") & (df.train_fraction == 1.0) & (df.dataset.isin(CORE))]
    sc = m.groupby(["backbone", "method"]).agg(acc=("best_acc", "mean")).reset_index()
    sc_piv = sc.pivot_table(index="method", columns="backbone", values="acc")
    sc_piv = sc_piv.reindex([m for m in METHOD_ORDER if m in sc_piv.index])
    sc_piv = sc_piv[[b for b in BK_ORDER if b in sc_piv.columns]]
    sc_piv.columns = [BK_LABEL.get(c, c) for c in sc_piv.columns]
    sc_piv.index = [ml(m) for m in sc_piv.index]
    sc_piv.to_csv(os.path.join(OUT, "ef_table_scaling.csv"), encoding="utf-8-sig")
    with open(os.path.join(OUT, "ef_table_scaling.md"), "w", encoding="utf-8") as f:
        f.write("# 骨干缩放表（核心4数据集平均 Top-1 %）\n\n")
        f.write(sc_piv.to_markdown() + "\n")
    print("saved ef_table_main.(csv/md), ef_table_scaling.(csv/md)")
    print(piv.to_string())


def main():
    df = load()
    if df.empty:
        print("no results yet")
        return
    print(f"loaded {len(df)} rows; backbones={sorted(set(df.backbone))}; "
          f"datasets={sorted(set(df.dataset))}; tags={sorted(set(df.tag))}")
    for fn in (fig_acc_vs_params, fig_heatmap, fig_acc_bars, fig_avg_rank,
               fig_scaling, fig_seed_var, fig_ablations, fig_data_efficiency,
               fig_convergence, fig_efficiency, make_tables):
        try:
            fn(df)
        except Exception as e:
            print(f"[warn] {fn.__name__} failed: {type(e).__name__}: {e}")
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
