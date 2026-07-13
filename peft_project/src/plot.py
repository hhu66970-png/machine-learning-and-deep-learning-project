# -*- coding: utf-8 -*-
"""Generate figures and summary tables from outputs/results.csv (+ curves).

Robust to partial results: only plots what is available. Re-runnable any time
while the matrix is still training, to refresh figures with the latest data.
"""
import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Chinese font (Windows) ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 180
plt.rcParams["savefig.bbox"] = "tight"

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "outputs", "results.csv")
CURVE_DIR = os.path.join(PROJ, "outputs", "curves")
FIG = os.path.join(PROJ, "figures")
OUT = os.path.join(PROJ, "outputs")
os.makedirs(FIG, exist_ok=True)

METHOD_LABELS = {
    "linear_probe": "Linear", "bitfit": "BitFit", "vpt": "VPT", "ssf": "SSF",
    "lora": "LoRA", "adaptformer": "AdaptFormer", "lora_ssf": "LoRA+SSF (ours)",
    "full_ft": "Full FT",
}
METHOD_ORDER = ["linear_probe", "bitfit", "vpt", "ssf", "lora", "adaptformer", "lora_ssf", "full_ft"]
DATASET_LABELS = {"cifar100": "CIFAR-100", "flowers102": "Flowers-102", "pets": "Pets", "dtd": "DTD"}
DATASET_ORDER = ["cifar100", "flowers102", "pets", "dtd"]
COLORS = dict(zip(METHOD_ORDER, plt.cm.tab10(np.linspace(0, 1, 10))))


def load():
    if not os.path.exists(RESULTS):
        return pd.DataFrame()
    df = pd.read_csv(RESULTS)
    return df


def mlabel(m):
    return METHOD_LABELS.get(m, m)


def methods_present(df):
    return [m for m in METHOD_ORDER if m in set(df["method"])]


def datasets_present(df):
    return [d for d in DATASET_ORDER if d in set(df["dataset"])]


# --------------------------------------------------------------------------- #
def fig_acc_vs_params(df):
    main = df[df["tag"] == "main"]
    dss = datasets_present(main)
    if not dss:
        return
    n = len(dss)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.8 * rows), squeeze=False)
    for ax, ds in zip(axes.flat, dss):
        sub = main[main["dataset"] == ds]
        for m in methods_present(sub):
            r = sub[sub["method"] == m].iloc[0]
            ax.scatter(r["trainable_params"], r["best_acc"], s=90,
                       color=COLORS[m], zorder=3)
            ax.annotate(mlabel(m), (r["trainable_params"], r["best_acc"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("可训练参数量 (对数刻度)")
        ax.set_ylabel("Top-1 测试准确率 (%)")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)}：准确率 vs 参数量")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(dss):]:
        ax.axis("off")
    fig.suptitle("各微调方法的「准确率—参数量」权衡（左上角为最优）", fontsize=14, y=1.02)
    fig.savefig(os.path.join(FIG, "fig1_acc_vs_params.png"))
    plt.close(fig)
    print("saved fig1_acc_vs_params.png")


def fig_acc_bars(df):
    main = df[df["tag"] == "main"]
    dss = datasets_present(main)
    ms = methods_present(main)
    if not dss or not ms:
        return
    x = np.arange(len(ms))
    w = 0.8 / max(len(dss), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(ms) + 2, 5))
    for i, ds in enumerate(dss):
        sub = main[main["dataset"] == ds]
        vals = [sub[sub["method"] == m]["best_acc"].max() if m in set(sub["method"]) else np.nan
                for m in ms]
        ax.bar(x + i * w, vals, w, label=DATASET_LABELS.get(ds, ds))
    ax.set_xticks(x + w * (len(dss) - 1) / 2)
    ax.set_xticklabels([mlabel(m) for m in ms], rotation=20, ha="right")
    ax.set_ylabel("Top-1 测试准确率 (%)")
    ax.set_title("各方法在不同数据集上的准确率对比")
    ax.legend(title="数据集")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig2_acc_bars.png"))
    plt.close(fig)
    print("saved fig2_acc_bars.png")


def fig_trainable_pct(df):
    main = df[df["tag"] == "main"]
    ms = methods_present(main)
    if not ms:
        return
    # average trainable% across datasets (depends only weakly on head size)
    vals = [main[main["method"] == m]["pct_trainable"].mean() for m in ms]
    fig, ax = plt.subplots(figsize=(1.3 * len(ms) + 2, 5))
    bars = ax.bar([mlabel(m) for m in ms], vals, color=[COLORS[m] for m in ms])
    ax.set_yscale("log")
    ax.set_ylabel("可训练参数占比 (%, 对数刻度)")
    ax.set_title("各方法的可训练参数占比")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig3_trainable_pct.png"))
    plt.close(fig)
    print("saved fig3_trainable_pct.png")


def fig_convergence(df):
    files = glob.glob(os.path.join(CURVE_DIR, "main_*.json"))
    by_ds = {}
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ds = d["meta"]["dataset"]
        by_ds.setdefault(ds, []).append((d["meta"]["method"], d["curve"]))
    for ds, items in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        items.sort(key=lambda x: METHOD_ORDER.index(x[0]) if x[0] in METHOD_ORDER else 99)
        for m, curve in items:
            ep = [c["epoch"] for c in curve]
            acc = [c["test_acc"] for c in curve]
            ax.plot(ep, acc, marker="o", ms=3, label=mlabel(m), color=COLORS.get(m))
        ax.set_xlabel("训练轮数 (epoch)")
        ax.set_ylabel("Top-1 测试准确率 (%)")
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)}：各方法收敛曲线")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(FIG, f"fig4_convergence_{ds}.png"))
        plt.close(fig)
        print(f"saved fig4_convergence_{ds}.png")


def fig_lora_ablation(df):
    abl = df[df["tag"] == "abl_loraR"]
    if abl.empty:
        return
    abl = abl.sort_values("lora_r")
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(abl["lora_r"], abl["best_acc"], "o-", color="tab:blue", label="准确率")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(abl["lora_r"])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("LoRA 秩 r")
    ax1.set_ylabel("Top-1 测试准确率 (%)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(abl["lora_r"], abl["trainable_params"] / 1e3, "s--", color="tab:red", label="可训练参数")
    ax2.set_ylabel("可训练参数量 (千)", color="tab:red")
    ax1.set_title("LoRA 秩 r 的消融实验 (DTD)")
    ax1.grid(True, alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig5_lora_rank_ablation.png"))
    plt.close(fig)
    print("saved fig5_lora_rank_ablation.png")


def fig_efficiency(df):
    main = df[df["tag"] == "main"]
    ms = methods_present(main)
    if not ms:
        return
    mem = [main[main["method"] == m]["peak_mem_mb"].mean() for m in ms]
    tpe = [main[main["method"] == m]["time_per_epoch_s"].mean() for m in ms]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.bar([mlabel(m) for m in ms], mem, color=[COLORS[m] for m in ms])
    a1.set_ylabel("峰值显存 (MB)")
    a1.set_title("训练显存开销")
    a1.tick_params(axis="x", rotation=20)
    a2.bar([mlabel(m) for m in ms], tpe, color=[COLORS[m] for m in ms])
    a2.set_ylabel("每轮训练时间 (秒)")
    a2.set_title("训练时间开销 (各数据集平均)")
    a2.tick_params(axis="x", rotation=20)
    for a in (a1, a2):
        a.grid(True, axis="y", alpha=0.3)
    fig.savefig(os.path.join(FIG, "fig6_efficiency.png"))
    plt.close(fig)
    print("saved fig6_efficiency.png")


def fig_heatmap(df):
    main = df[df["tag"] == "main"]
    ms = methods_present(main)
    dss = datasets_present(main)
    if not ms or not dss:
        return
    mat = np.full((len(ms), len(dss)), np.nan)
    for i, m in enumerate(ms):
        for j, ds in enumerate(dss):
            v = main[(main["method"] == m) & (main["dataset"] == ds)]["best_acc"]
            if len(v):
                mat[i, j] = v.max()
    fig, ax = plt.subplots(figsize=(1.4 * len(dss) + 3, 0.6 * len(ms) + 2))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(dss)))
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in dss])
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([mlabel(m) for m in ms])
    for i in range(len(ms)):
        for j in range(len(dss)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_title("准确率热力图 (方法 × 数据集)")
    fig.colorbar(im, ax=ax, label="Top-1 准确率 (%)")
    fig.savefig(os.path.join(FIG, "fig7_heatmap.png"))
    plt.close(fig)
    print("saved fig7_heatmap.png")


def make_tables(df):
    main = df[df["tag"] == "main"]
    if main.empty:
        return
    piv = main.pivot_table(index="method", columns="dataset", values="best_acc", aggfunc="max")
    piv = piv.reindex([m for m in METHOD_ORDER if m in piv.index])
    piv.columns = [DATASET_LABELS.get(c, c) for c in piv.columns]
    piv["平均"] = piv.mean(axis=1).round(2)
    pct = main.groupby("method")["pct_trainable"].mean()
    piv.insert(0, "可训练%", [round(pct.get(m, np.nan), 3) for m in piv.index])
    piv.index = [mlabel(m) for m in piv.index]
    piv.to_csv(os.path.join(OUT, "table_main.csv"), encoding="utf-8-sig")
    with open(os.path.join(OUT, "table_main.md"), "w", encoding="utf-8") as f:
        f.write("# 主结果表：各方法 Top-1 准确率 (%)\n\n")
        f.write(piv.to_markdown())
        f.write("\n")
    # full metrics dump
    main.to_csv(os.path.join(OUT, "table_full.csv"), index=False, encoding="utf-8-sig")
    print("saved table_main.csv / table_main.md / table_full.csv")
    print(piv.to_string())


def main():
    df = load()
    if df.empty:
        print("no results yet.")
        return
    print(f"loaded {len(df)} rows; methods={sorted(set(df['method']))}; "
          f"datasets={sorted(set(df['dataset']))}")
    fig_acc_vs_params(df)
    fig_acc_bars(df)
    fig_trainable_pct(df)
    fig_convergence(df)
    fig_lora_ablation(df)
    fig_efficiency(df)
    fig_heatmap(df)
    make_tables(df)
    print("PLOT_DONE")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    main()
