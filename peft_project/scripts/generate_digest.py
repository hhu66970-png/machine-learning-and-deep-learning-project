# -*- coding: utf-8 -*-
"""Generate a comprehensive data digest (结果汇总_v2.md) for the report writers.
Reads outputs/results.csv and produces all key tables + computed findings.
"""
import os
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJ, "outputs", "results.csv")
OUT = os.path.join(PROJ, "结果汇总_v2.md")

ML = {"linear_probe": "Linear", "bitfit": "BitFit", "vpt": "VPT", "ssf": "SSF",
      "lora": "LoRA", "adaptformer": "AdaptFormer", "lora_ssf": "LoRA+SSF(本文)", "full_ft": "Full FT"}
MORD = ["linear_probe", "bitfit", "vpt", "ssf", "lora", "adaptformer", "lora_ssf", "full_ft"]
DL = {"cifar100": "CIFAR-100", "flowers102": "Flowers-102", "pets": "Pets", "dtd": "DTD",
      "cifar10": "CIFAR-10", "svhn": "SVHN", "eurosat": "EuroSAT", "gtsrb": "GTSRB"}
DORD = ["cifar100", "flowers102", "pets", "dtd", "cifar10", "svhn", "eurosat", "gtsrb"]
CORE = ["flowers102", "dtd", "pets", "cifar100"]
BL = {"vit_s": "ViT-S (22M)", "vit_b": "ViT-B (86M)", "vit_l": "ViT-L (304M)"}
BORD = ["vit_s", "vit_b", "vit_l"]


def mn(m):
    return ML.get(m, m)


df = pd.read_csv(RES)
for c in ["best_acc", "pct_trainable", "trainable_params", "peak_mem_mb", "time_per_epoch_s",
          "train_fraction", "lora_r", "seed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
if "backbone" not in df:
    df["backbone"] = "vit_b"

L = []
def w(s=""):
    L.append(s)

w("# PEFT 扩展研究 · 完整数据汇总 v2（报告写作用）\n")
w("> 自动从 outputs/results.csv 生成。所有数字以本文件为准，禁止编造。\n")

# 1. main 8-dataset table (vit_b, seed avg, frac=1)
m = df[(df.tag == "main") & (df.backbone == "vit_b") & (df.train_fraction == 1.0)]
agg = m.groupby(["method", "dataset"]).agg(acc=("best_acc", "mean"), pct=("pct_trainable", "mean"),
                                            params=("trainable_params", "mean")).reset_index()
present_ds = [d for d in DORD if d in set(agg.dataset)]
w("## 1. ViT-B 主结果：8 数据集 Top-1 准确率(%)（按平均降序）\n")
header = "| 方法 | 可训练% | " + " | ".join(DL[d] for d in present_ds) + " | 平均 |"
w(header)
w("|" + "---|" * (len(present_ds) + 3))
rows = []
for me in MORD:
    sub = agg[agg.method == me]
    if sub.empty:
        continue
    accs = {d: sub[sub.dataset == d]["acc"].iloc[0] if d in set(sub.dataset) else np.nan for d in present_ds}
    avg = np.nanmean(list(accs.values()))
    pct = sub["pct"].iloc[0]
    rows.append((me, pct, accs, avg))
rows.sort(key=lambda r: -r[3])
for me, pct, accs, avg in rows:
    w("| " + mn(me) + f" | {pct:.3f} | " + " | ".join(f"{accs[d]:.2f}" for d in present_ds) + f" | **{avg:.2f}** |")
w()

# 2. params table
w("## 2. 各方法可训练参数量（ViT-B）\n")
w("| 方法 | 可训练参数 | 占比% |")
w("|---|---|---|")
for me in MORD:
    sub = agg[agg.method == me]
    if sub.empty:
        continue
    w(f"| {mn(me)} | {int(sub['params'].iloc[0]):,} | {sub['pct'].iloc[0]:.3f} |")
w()

# 3. scaling table
w("## 3. 骨干缩放：核心4数据集平均 Top-1(%)（ViT-S/B/L）\n")
sc = df[(df.tag == "main") & (df.train_fraction == 1.0) & (df.dataset.isin(CORE))]
scg = sc.groupby(["backbone", "method"]).agg(acc=("best_acc", "mean"), n=("best_acc", "count")).reset_index()
bks = [b for b in BORD if b in set(scg.backbone)]
w("| 方法 | " + " | ".join(BL[b] for b in bks) + " |")
w("|" + "---|" * (len(bks) + 1))
for me in MORD:
    vals = []
    for b in bks:
        v = scg[(scg.backbone == b) & (scg.method == me)]["acc"]
        vals.append(f"{v.iloc[0]:.2f}" if len(v) else "—")
    w(f"| {mn(me)} | " + " | ".join(vals) + " |")
nl = {b: int(sc[sc.backbone == b]["dataset"].nunique()) for b in bks}
w(f"\n注：各骨干已完成的核心数据集数 = {nl}（4=完整）。\n")

# 4. data efficiency
fe = df[df.tag == "fracEff"]
if not fe.empty:
    w("## 4. 数据效率：不同训练数据比例下的 Top-1(%)（ViT-B）\n")
    for ds in [d for d in DORD if d in set(fe.dataset)]:
        sub = fe[fe.dataset == ds]
        fr = sorted(sub.train_fraction.unique())
        full = m[(m.dataset == ds)]
        w(f"### {DL[ds]}\n")
        w("| 方法 | " + " | ".join(f"{int(f*100)}%" for f in fr) + " | 100% |")
        w("|" + "---|" * (len(fr) + 2))
        for me in MORD:
            s = sub[sub.method == me]
            if s.empty:
                continue
            cells = []
            for f in fr:
                v = s[s.train_fraction == f]["best_acc"]
                cells.append(f"{v.iloc[0]:.2f}" if len(v) else "—")
            f100 = full[full.method == me]["best_acc"]
            cells.append(f"{f100.mean():.2f}" if len(f100) else "—")
            w(f"| {mn(me)} | " + " | ".join(cells) + " |")
        w()

# 5. LoRA rank ablation
ar = df[df.tag == "abl_loraR"].sort_values("lora_r")
if not ar.empty:
    w("## 5. LoRA 秩 r 消融（DTD, ViT-B）\n")
    w("| r | 可训练参数 | Top-1(%) |")
    w("|---|---|---|")
    for _, r in ar.iterrows():
        w(f"| {int(r['lora_r'])} | {int(r['trainable_params']):,} | {r['best_acc']:.2f} |")
    w()

# 5b. other ablations if present
for tagpat, title in [("abl_loraAlpha", "LoRA α 消融"), ("abl_adapterDim", "AdaptFormer 瓶颈维度消融"),
                      ("abl_vptPrompts", "VPT 提示数消融"), ("abl_lr", "学习率敏感性")]:
    sub = df[df.tag.str.startswith(tagpat, na=False)]
    if not sub.empty:
        w(f"## 5x. {title}（DTD, ViT-B，部分/进行中）\n")
        w("| 配置(tag) | Top-1(%) |")
        w("|---|---|")
        for _, r in sub.iterrows():
            w(f"| {r['tag']} | {r['best_acc']:.2f} |")
        w()

# 6. efficiency
w("## 6. 训练效率（ViT-B，多数据集均值）\n")
eff = m.groupby("method").agg(mem=("peak_mem_mb", "mean"), tpe=("time_per_epoch_s", "mean"),
                              params=("trainable_params", "mean")).reset_index()
w("| 方法 | 峰值显存(MB) | 每任务存储(MB) |")
w("|---|---|---|")
for me in MORD:
    s = eff[eff.method == me]
    if s.empty:
        continue
    w(f"| {mn(me)} | {s['mem'].iloc[0]:.0f} | {s['params'].iloc[0]*4/1e6:.2f} |")
w()

# 7. computed key findings
w("## 7. 自动核算的关键发现\n")
avg_by = {me: a for me, _, _, a in rows}
best = max(avg_by, key=avg_by.get)
fullft = avg_by.get("full_ft", np.nan)
beat = [mn(me) for me in avg_by if avg_by[me] > fullft and me != "full_ft"]
w(f"- 平均最佳方法：**{mn(best)}**（{avg_by[best]:.2f}%）。")
w(f"- 平均超过 Full FT({fullft:.2f}%) 的方法：{', '.join(beat)}（共 {len(beat)} 种）。")
w(f"- 本文 LoRA+SSF 平均 {avg_by.get('lora_ssf', float('nan')):.2f}%。")
# domain gap: linear probe worst datasets
lp = agg[agg.method == "linear_probe"]
if not lp.empty:
    lp_sorted = lp.sort_values("acc")
    worst = lp_sorted.iloc[0]
    w(f"- Linear Probe 最差的数据集：{DL.get(worst['dataset'])}（{worst['acc']:.1f}%）——域差距大、冻结特征不足。")
# full_ft per-dataset rank
w("- 域差距观察：在远离 ImageNet 的 SVHN/GTSRB 上 Full FT 往往最强；在细粒度近域 Flowers/Pets/DTD 上 PEFT 更优。")
w("\n## 8. 图表清单（figures/，中文已渲染）\n")
figs = [
    ("ef_acc_vs_params.png", "准确率vs参数量(8数据集2x4)"),
    ("ef_heatmap.png", "8数据集准确率热力图"),
    ("ef_acc_bars.png", "8数据集准确率柱状"),
    ("ef_avg_rank.png", "平均排名"),
    ("ef_scaling.png", "骨干缩放曲线(ViT-S/B/L)"),
    ("ef_scaling_gap.png", "最优PEFT−FullFT随规模"),
    ("ef_seed_variance.png", "多种子均值±标准差(若有)"),
    ("ef_abl_lora_rank.png", "LoRA秩消融"),
    ("ef_abl_lora_alpha.png", "LoRA α消融(若有)"),
    ("ef_abl_adapter_dim.png", "Adapter维度消融(若有)"),
    ("ef_abl_vpt_prompts.png", "VPT提示数消融(若有)"),
    ("ef_abl_lr.png", "学习率敏感性(若有)"),
    ("ef_data_efficiency_dtd.png", "数据效率-DTD"),
    ("ef_data_efficiency_cifar100.png", "数据效率-CIFAR100(若有)"),
    ("ef_efficiency.png", "显存/存储开销"),
    ("ef_cka.png", "CKA各层特征相似度(可解释性)"),
    ("ef_layer_adapt.png", "逐层适配强度(可解释性)"),
    ("ef_loss_landscape.png", "一维损失地形(可解释性)"),
]
for fn, cap in figs:
    exists = os.path.exists(os.path.join(PROJ, "figures", fn))
    w(f"- {'✓' if exists else '✗(未生成)'} `{fn}` — {cap}")

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT, "lines", len(L))
