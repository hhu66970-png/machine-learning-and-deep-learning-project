# 自动运行状态 / RESUME NOTES (for the agent)

## 终极目标
满分档 PEFT 课题：大规模扩展实验 + 约 80 页高质量中文报告（图文充实，不灌水）。

## 已确认参数（用户）
- 时间不限，笔记本保持开机插电。
- 骨干：ViT-S + ViT-B + ViT-L 全做（L 用 grad checkpointing，已验证仅 3GB 显存）。
- 专注视觉（不做 NLP）。
- 报告署名：胡昊铭，3024210028。

## 实验战役（run_campaign.py，task=boty1v3dn，已启动）
274 个 job（6 个旧消融已跳过，268 待跑），按优先级顺序，**可断点续跑（run_id 去重）**：
1. ViT-B 新4数据集(cifar10/svhn/eurosat/gtsrb)×8方法
2. ViT-S 全8数据集×8方法（缩放-小端）
3. 消融：LoRA秩/α、Adapter维度、VPT提示数、学习率敏感性
4. 数据效率：train_fraction∈{5,10,25,50}% × 5方法 × {dtd,cifar100}
5. 可解释性检查点：ViT-B/Pets 8方法 --save_ckpt（存到 outputs/ckpts/）
6. ViT-B core4 seed=123（多种子）
7. ViT-L core4×8方法（缩放-大端，慢）
8. ViT-B core4 seed=2024（多种子，凑3种子）
- 8 数据集：cifar100,flowers102,pets,dtd,cifar10,svhn,eurosat,gtsrb（已下载，Aircraft 因 2.75GB 太慢已弃用）。
- 旧 32 条 ViT-B core seed42 结果保留并参与分析（不重跑）。
- 预计 ~30-40 小时。结束标志日志含 CAMPAIGN_DONE。

## 关键文件
- 数据/指标：outputs/results.csv（新schema：run_id,backbone,train_fraction,...；旧38行已迁移，备份 .bak）
- 出图：`python src/analyze.py`（幂等，扩展版全套图 ef_*.png + ef_table_*.md），旧 `src/plot.py` 仍可用。
- 可解释性：`src/analyze_interp.py`（CKA/损失地形/注意力，需 outputs/ckpts/ 检查点，group5 跑完后可用）。
- 报告：report/sections/*.md → report/build_docx.py → 实验报告_胡昊铭_3024210028.docx

## 每次唤醒该做什么
1. 看进度：grep "OK (|FAILED|CAMPAIGN_DONE" 于 campaign 日志；或 results.csv 行数。
2. 刷新图表：`python src/analyze.py`。
3. group5(interp)跑完后：`python src/analyze_interp.py` 生成可解释性图。
4. CAMPAIGN_DONE 后：最终出图 → 启动报告写作 workflow（多智能体，~80页）→ build_docx → 校验。

## 5小时自主冲刺计划(18:24起，用户5h后回)
- campaign 已**分离进程**(Start-Process,PID每次重启变)+睡眠已禁，断网也不停。日志: outputs/campaign_run.log
- 已**重排序**：interp(8) → 数据效率(40) → 消融 → ViT-S新4 → 多种子 → ViT-L。
- **报告核心数据已就位**：ViT-B 8数据集✓、ViT-S核心4(缩放)✓、LoRA秩消融✓；仅缺 interp(跑完即得)。
- **报告构建触发点**：当 interp✓+数据效率✓(约3-3.7h) 或 elapsed~3.5h 时：
  1) 暂停campaign(kill python) 2) 跑 analyze_interp.py(GPU,需独占) + analyze.py(全图)
  3) 重启campaign(继续跑加分项) 4) Workflow多智能体写~80页报告(agent写作不占本地GPU,可与campaign并行)
  5) build_docx.py 组装 + 导PDF抽查。
- ViT-L/多种子为加分项，5h内大概率跑不完，报告写已有内容即可(很充实)。

## 报告 v1 已完成(94页)
- report/实验报告_胡昊铭_3024210028.docx + .pdf（94页,13图,21表,5.1万字）。
- sections2/ 为各章md；build_docx.py 组装；analyze.py/analyze_interp.py 出图；结果汇总_v2.md 为数据digest。
- KMP_DUPLICATE_LIB_OK=TRUE 是 analyze_interp.py 必需(torch+matplotlib OMP冲突)。

## 当前:继续跑完整campaign(用户要求全跑,含ViT-L,~30h)
- 剩余:完整消融(α/Adapter维/VPT/lr)→ViT-S新4→多种子123→ViT-L核心→多种子2024。
- **全部跑完(CAMPAIGN_DONE)后**:重跑 analyze.py(出ViT-L缩放/多种子误差棒/完整消融图)
  + analyze_interp.py(若需)+ generate_digest.py 更新digest → 用 Workflow 重写/增强相关章节
  (缩放、多种子、消融)→ build_docx.py 重建增强版报告。
- 注意:analyze_interp 需GPU,跑它时要先暂停campaign再恢复(分离进程,Start-Process)。

## 环境
python：C:\Users\29339\anaconda3\envs\dl_course\python.exe ；中文字体 Microsoft YaHei。

## 进度日志
- (旧) 38 条完成：ViT-B core4×8 seed42 + LoRA秩消融6。报告 v1（19页）已存在 report/。
- 大战役首次启动后发现大数据集太慢(3run/57min→预计80h)，已**削减大/易数据集轮数**
  (cifar10:3, svhn:2, eurosat:5, gtsrb:3; cifar100保持6以与旧seed42一致)。
- **当前活动 campaign task=bndt19s2c**（11:46 重启），waiter=bzt8llp20。
- 监控循环：每~57min 刷新 analyze.py 出图；group5后跑 analyze_interp.py；
  groups1-6完成(~15-18h)即可写报告，ViT-L(group7)+seed2024(group8)为加分项随后补。
