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
  --lora_alpha 64
```

完整实验可运行：

```bash
python scripts/run_matrix.py
python scripts/run_campaign.py
```

已有实验结果位于 `peft_project/outputs/`，图表位于 `peft_project/figures/`。

## 注意事项

数据集、预训练权重、模型检查点和大量训练日志不建议直接提交到 GitHub。课程提交时建议保留核心源码、README、环境说明与必要的结果表。
