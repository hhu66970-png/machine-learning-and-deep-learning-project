# 机器学习与深度学习课程设计

本仓库用于保存《机器学习与深度学习》课程设计资料。

- 作者：胡昊铭
- 学号：3024210028
- 实验主题：面向预训练视觉 Transformer 的参数高效微调

## PEFT 实验项目

核心代码位于 [`peft_project/`](peft_project/)，对比以下八种微调策略：

`Full FT` | `Linear Probing` | `BitFit` | `LoRA` | `AdaptFormer` | `SSF` | `VPT` | `LoRA+SSF`

主要内容包括：

- `peft_project/src/models/peft.py`：八种微调策略的注入与冻结逻辑。
- `peft_project/src/train.py`：统一训练、评估和指标记录。
- `peft_project/src/datasets.py`：下游数据集和预处理。
- `peft_project/src/plot.py` 与 `analyze.py`：实验图表与结果汇总。
- `peft_project/scripts/`：数据下载、正确性检查和实验矩阵编排。

## 快速开始

```bash
cd peft_project

conda create -n dl_course python=3.11 -y
conda activate dl_course

pip install torch torchvision timm==1.0.25 pandas numpy matplotlib seaborn tensorboard fvcore

python scripts/download_data.py
python scripts/sanity.py

python src/train.py \
  --method lora \
  --dataset dtd \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --lora_r 8 \
  --lora_alpha 8
```

完整实验可运行：

```bash
python scripts/run_matrix.py
python scripts/run_campaign.py
```

已有实验结果位于 `peft_project/outputs/`，图表位于 `peft_project/figures/`。

## 实验结果

下表汇总 ViT-B、随机种子 42 在 CIFAR-100、Flowers-102、Oxford-IIIT Pets、DTD、CIFAR-10、SVHN、EuroSAT 和 GTSRB 八个完整数据集上的正式结果。统一采用预定训练轮数结束时的 `final_acc`，平均值为八个任务 Top-1 准确率的等权平均；可训练参数比例采用 10 类任务分类头作为统一结构预算口径。

| 方法 | 八任务平均 Top-1（%） | 可训练参数比例（%） |
|---|---:|---:|
| AdaptFormer | 95.51 | 1.376 |
| LoRA+SSF | 95.24 | 0.437 |
| LoRA | 95.21 | 0.351 |
| Full FT | 94.92 | 100.000 |
| SSF | 94.81 | 0.095 |
| BitFit | 94.60 | 0.129 |
| VPT | 92.49 | 0.027 |
| Linear Probing | 87.01 | 0.009 |

在当前预训练骨干、数据处理和固定训练预算下，多种 PEFT 方法能够以较小的任务参数保持与 Full FT 接近或更高的平均准确率。AdaptFormer取得当前最高平均值，但其适配参数也高于其他 PEFT 方法；LoRA+SSF与LoRA仅相差0.03个百分点，现有两个种子的描述性检查不足以支持“组合方法稳定优于LoRA”的结论。原始实验记录见 [`peft_project/outputs/results.csv`](peft_project/outputs/results.csv)，完整统计口径与报告结果以课程提交包为准。

## 注意事项

数据集、预训练权重、模型检查点和大量训练日志不建议直接提交到 GitHub。课程提交时建议保留核心源码、README、环境说明与必要的结果表。
