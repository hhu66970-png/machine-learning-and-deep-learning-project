# 机器学习与深度学习课程设计

- 姓名：胡昊铭
- 学号：3024210028
- 题目：面向预训练视觉 Transformer 的参数高效微调

本项目比较了 Full FT、Linear Probing、BitFit、LoRA、AdaptFormer、SSF、VPT 和 LoRA+SSF 八种微调方法。实验代码、运行脚本和结果文件均放在 [`peft_project/`](peft_project/) 目录中。

## 文件说明

- `peft_project/src/models/peft.py`：各类微调方法的实现。
- `peft_project/src/train.py`：模型训练与测试入口。
- `peft_project/src/datasets.py`：数据集读取和预处理。
- `peft_project/src/plot.py`、`peft_project/analyze.py`：结果整理与绘图。
- `peft_project/scripts/`：数据下载、运行检查和批量实验脚本。
- `peft_project/outputs/results.csv`：本报告使用的实验记录。

## 运行环境

建议使用 Python 3.11。安装依赖：

```bash
conda create -n dl_course python=3.11 -y
conda activate dl_course
pip install torch torchvision timm==1.0.25 pandas numpy matplotlib seaborn tensorboard fvcore
```

## 运行方法

进入项目目录并下载数据：

```bash
cd peft_project
python scripts/download_data.py
python scripts/sanity.py
```

下面以 DTD 数据集上的 LoRA 实验为例：

```bash
python src/train.py \
  --method lora \
  --dataset dtd \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --lora_r 8 \
  --lora_alpha 8
```

批量运行实验：

```bash
python scripts/run_matrix.py
python scripts/run_campaign.py
```

## 实验结果

下表是 ViT-B 在八个数据集上的实验结果，随机种子为 42。平均准确率按八个数据集的最终轮准确率计算。

| 方法 | 平均准确率（%） | 可训练参数比例（%） |
|---|---:|---:|
| AdaptFormer | 95.51 | 1.376 |
| LoRA+SSF | 95.24 | 0.437 |
| LoRA | 95.21 | 0.351 |
| Full FT | 94.92 | 100.000 |
| SSF | 94.81 | 0.095 |
| BitFit | 94.60 | 0.129 |
| VPT | 92.49 | 0.027 |
| Linear Probing | 87.01 | 0.009 |

本次实验中，AdaptFormer 的平均准确率最高，为 95.51%。LoRA+SSF 和 LoRA 的结果分别为 95.24% 和 95.21%，两者只相差 0.03 个百分点，因此不能据此认为组合方法一定优于 LoRA。各数据集的详细结果见 [`peft_project/outputs/results.csv`](peft_project/outputs/results.csv)。

## 仓库说明

仓库中未上传数据集、预训练权重和模型检查点，运行前需要先下载数据。课程提交材料以实验报告和精简源码包为准，本仓库用于查看源码与实验结果。
