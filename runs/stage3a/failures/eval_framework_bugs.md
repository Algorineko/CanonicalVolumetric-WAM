# Stage 3a 评估框架 bug 记录(2026-11-18)

## Bug 1: action-shuffle 消融最初无效应(消融设计缺陷)

- 现象:`eval_p1_shuffle.csv` 与 `eval_p1.csv` 指标完全相同(4 位小数一致)。
- 根因:eval DataLoader shuffle=False,batch 内样本按 (episode, t, offset) 顺序排列 → 同批样本是**相邻帧**;15Hz 下相邻帧的 action chunk 几乎相同,批内置换 ≈ 无操作。
- 关键佐证:这不是"模型不使用动作"——act_only(零体素输入、仅 qpos+action)dynamic IoU 0.153 > 0,证明动作通路可用。
- 修复:改为**数据集级跨相位置换**(`_ShuffleActionsWrapper`,同 episode 内 t 循环平移半个数据集),保证动作块来自不同运动阶段。修复后 P1 dynamic IoU 0.295→0.277(-6%)、change F1 0.068→0.059(-12%):模型对动作**敏感但弱**——动态预测主要来自当前体素状态的局部外推。

## Bug 2: dropout 评估无输出后缀,覆盖主结果文件

- 现象:`eval_p0/p1/p1_v.csv` 被重写为 dropout 子集(344 行)结果,且无 `*_drop_*.csv`。
- 根因:`suffix` 只处理了 shuffle 分支,`--input-dropout` 没有生成后缀。
- 修复:后缀逻辑补全 `f"_drop_{subset}"`;重跑被覆盖的 3 个主评估 + 9 个 dropout 评估。
- 另修:`_DropoutWrapper` 索引错位(dropout 缓存只存 t 步长 8 的帧,原 wrapper 对每个 t 取 dropout npz 且 `base[idx]` 与过滤后的 items 对不上)——改为保留原索引列表 + 存在性过滤。

## 教训

消融必须先验证"扰动确实改变了输入"(数值差异检查),否则会把"消融无效应"误读为科学结论;文件命名必须让不同实验条件不可冲突。

## Bug 3(最重要): P1-V occupancy pos_weight 写反(验证工作流 blocker)

- 现象:P1-V occIoU 0.535,远低于 P1 的 0.832,报告差点将"解耦失败"作为结论。
- 根因:`train_tiny.py` 的 `pos_w = w_state[1]/w_state[0] = 2.82/23.0 = 0.123`,而 occupancy BCE 的正类是 hit,应上权 hit:`free/hit 比 = 8.15`(即 w_state[0]/w_state[1])。
- 影响:正类(命中体素)被 0.123 权重压低 → 模型系统性欠预测占据。
- 修复后重训(仅改该行,seed/步数/其余不变):occIoU 0.535→**0.846**、dynIoU 0.084→**0.381**(验证工作流对照重训一致)。**P1-V(解耦)在动态区域实际优于 P1(0.381 vs 0.295)**——修正后的结论与原结论相反。
- 教训:对抗性验证(blacker 级重构指标)+ 单一变量对照重训,是抓这类"静默损失配置错误"的唯一可靠手段。
