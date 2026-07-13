# 模型微调技术探索（PEFT）—— 参数高效微调方法对比

在 ImageNet-21k 预训练的 **ViT-Base/16** 上，系统对比 **8 种微调策略**在 4 个下游图像分类数据集上的
「准确率 / 可训练参数量 / 显存 / 训练时间」，并做 LoRA 秩消融与一个组合改进（LoRA+SSF）。

> 《机器学习与深度学习》课程设计 · 选题④ 模型微调技术探索。

---

## 1. 实验环境

| 项 | 版本 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5070 Ti Laptop (12GB) |
| Python | 3.11 (conda 环境 `dl_course`) |
| PyTorch | 2.10 + CUDA 12.8 |
| 关键库 | timm 1.0.25, torchvision 0.25, numpy, pandas, matplotlib, seaborn |

复现环境：
```bash
conda create -n dl_course python=3.11
conda activate dl_course
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm pandas matplotlib seaborn tensorboard fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 数据集下载

4 个数据集均通过 torchvision 自动下载到 `data/`（国内慢时脚本会自动尝试清华镜像）：

| 数据集 | 类别 | 类型 | 训练/测试 |
|---|---|---|---|
| CIFAR-100 | 100 | 通用物体 | 50000 / 10000 |
| Oxford Flowers-102 | 102 | 细粒度·花卉 | 2040 / 6149 |
| Oxford-IIIT Pet | 37 | 细粒度·宠物 | 3680 / 3669 |
| DTD | 47 | 纹理 | 3760 / 1880 |

```bash
python scripts/download_data.py            # 下载全部
python scripts/download_data.py cifar100   # 只下某个
```

## 3. 运行方式

```bash
# (a) 正确性自检：8 种方法的前向/反向/参数量
python scripts/sanity.py

# (b) 单个实验
python src/train.py --method lora --dataset dtd --epochs 30 --batch_size 64

# (c) 完整矩阵 + 消融（后台跑，结果增量写入 outputs/results.csv，可断点续跑）
python scripts/run_matrix.py

# (d) 由结果生成全部图表与汇总表
python src/plot.py
```

支持的方法 `--method`：
`full_ft` | `linear_probe` | `bitfit` | `lora` | `adaptformer` | `ssf` | `vpt` | `lora_ssf`

支持的数据集 `--dataset`：`cifar100` | `flowers102` | `pets` | `dtd`

## 4. 目录结构

```
peft_project/
├── configs/
├── data/                  数据集（自动下载）
├── src/
│   ├── models/peft.py     8 种 PEFT 方法实现（核心）
│   ├── datasets.py        数据加载与预处理
│   ├── train.py           训练/评估主程序（记录全部指标）
│   ├── plot.py            图表与汇总表生成
│   └── utils.py           参数计数、显存、计时
├── scripts/
│   ├── download_data.py   并行下载数据集（带重试/镜像）
│   ├── sanity.py          方法正确性自检
│   └── run_matrix.py      实验矩阵编排（可续跑）
├── outputs/
│   ├── results.csv        全部实验指标（每行一个实验）
│   ├── curves/            每个实验的逐轮曲线 (json)
│   ├── logs/              每个实验的训练日志
│   └── table_main.md      主结果表
├── figures/               论文用图 (fig1~fig7)
├── 实验方案.md             原理与方案（含报告大纲）
└── README.md
```

## 5. 实验结果

主结果表见 [outputs/table_main.md](outputs/table_main.md)；图表见 `figures/`：

| 图 | 内容 |
|---|---|
| fig1 | 准确率 vs 可训练参数量（核心权衡图）|
| fig2 | 各方法在各数据集的准确率柱状图 |
| fig3 | 各方法可训练参数占比 |
| fig4 | 各数据集的收敛曲线 |
| fig5 | LoRA 秩 r 消融 |
| fig6 | 显存与训练时间开销 |
| fig7 | 准确率热力图（方法 × 数据集）|

> 实验仍在运行时，可随时重跑 `python src/plot.py` 用最新结果刷新图表。
