# Stage 3b Final Report(2026-11-19,持续更新)

审查方指令: docs-local/plan/current-plan.md。数字全部实测,回指 metrics/ 下文件;未完成项标 PENDING。

## 1. 机制归因裁决(审查方 §1.2)→ 决策 1: Factorization 是否保留?

**已裁决(2026-11-20,`metrics/attribution_verdict.md`,数据源 pareto v3 全 9 CSV)**:

- **P1-V − P1 在全部匹配预算 × Clean/Rand 上 ≤ 0**(clean: -0.044@0.005, -0.014@0.01, -0.001@0.02;rand 同向)——解耦监督在固定 split 上**无动态预测优势**
- **P1-W ≈ P1-V**(差 ±0.02 内):权重移植即可复现"解耦"的全部表现
- **P1-F 一致最差**(-0.063@0.01):精确 NLL 等价的因式参数化**主动伤害优化**(损失数值等价 ≠ 梯度路径等价)
- **P1(朴素逆频 CE)在严格预算最强**(0.005: 0.321 vs P1-V 0.277)
- P1-V 三 seed 稳定(std ≤ 0.023)——"无优势"结论本身是稳定的,不是噪声
- stage3a 的"P1-V 优势"(0.347 vs 0.295)确认为**旧 split + 未匹配操作点伪影**(旧 split 用 rand0/1 当 test,现已降级为 development)

**决策 1:Factorization 降级为训练技巧(按审查方 §1.2 规则),不再作为机制主张。**
机制主张重心移至:L4 观测者独立性(探针已证) + unknown/visibility 语义(可见性头 rev_acc 0.95,作为辅助输出) + 评估方法论(匹配 falseOcc pareto + persistence 基线 + 物体区域指标——这套证伪协议本身是可复用贡献)。

固定阈值旁证(door_eval)保留:`metrics/door/door_summary.csv`——P1 家族均可预测门运动,但该结果不依赖 factorization。

## 2. 独立物体动力学可预测性(审查方 §1.3)→ 决策 3

`metrics/door/`(独立 replay episode 427 帧,door_region GT,真实 qpos FK 机器人区域,mean 2.34mm 验证):

| 模型 | door_new_recall k=8 / k=16 | door_new_prec | robot_dynIoU k=16 | P0 baseline |
|---|---|---|---|---|
| P0 persistence | **0.0 / 0.0** | nan | 0.732(区域 IoU 静态部分) | — |
| P1 ×3 seeds | 0.58–0.80 / 0.76–0.87 | 1.0 | 0.870–0.882 | — |
| P1-V ×3 seeds | 0.61–0.78 / 0.79–0.86 | 1.0 | 0.874–0.876 | — |
| P1-W | 0.50 / 0.60 | 1.0 | 0.835 | — |
| P1-F | 0.18 / 0.25 | 1.0 | 0.817 | — |

**裁决:动作条件化的独立物体运动(门)可预测** — recall 0.58–0.87 vs persistence 0.0,精度 1.0(翻转预测零幻觉)。审查方停止条件("所有模型均无法预测独立对象运动")**不触发**。

qpos 映射勘误:replay npz 的 qpos 曾错误截取 robot_qpos_actual[:14](实际布局左臂在 dims [6,14,18,22,26,30]、左夹爪 dim34);已修正 426 帧(build_replay_geometry + /tmp/fix_replay_qpos.py),FK 验证 mean 2.34mm / p95 18.7mm = 已知 PhysX 接触发散量级。

## 3. Observer 干预探针(审查方 §4)→ 支撑决策 4

- `--head-cam-dis 0.05` 重放 ep10:**物理逐位一致**(qpos/门角/mic pose max diff = 0.0,427 帧,`metrics/probe_verification.json`),head 相机渲染已变(Δdepth 均值 5–7mm,边缘 ~700mm),left/right 相机逐位不变。
- **可见性对比(L4 规律实测)** `metrics/probe_vis_compare.md`:2.5mm 相机偏移 → 每帧 **occlude 103.8 + reveal 42.6 ≈ 146 个可见性翻转细胞**、hit ±~10,而物理零变化 → **观测过程变化速率远超物理变化速率**,与 Stage 2c 的 4.6–6.3× 定量同向。这就是 Observation Model 必须独立于 Geometry Dynamics 的直接证据(同几何 + 不同相机 = 不同 visibility,而 occupancy 目标不变)。
- hold-action / retreat 分支:NOT_RUN(时间盒)

## 4. Stage 3c 表示对照(审查方 §3)→ 决策 2: Voxel/Point/View

- 代码: `src/cvwam/models_repr.py`(P2/P2E 点云 scatter→解码器;P3 per-view pointmap+相机 FiLM→再投影;统一解耦头,参数 1.44M/1.44M/1.63M vs P1 家族 1.56M)
- 缓存: `cache/point_ep`(3 视角 depth+参数,点云在线构建)
- 训练: P2/P2E/P3 各 seed42(链式 `runs/stage3c/run_chain.sh`) — **PENDING**
- 胜者补 2 seeds 的规则:训练后按统一 eval(door/occ/dynIoU)裁决 — PENDING

## 5. Novelty(审查方 §2)

`research/novelty_matrix.md` 已更新:DSR-Net(最近先例,精确区分)/UNITAS(表示范式核心对照)/AWAM/OccWAM;"无先例"绝对化已替换为可检索的限定表述。

## 6. GPU 预算与 NOT_RUN

- GPU 累计:训练 0.45h + pareto/door 评估 ~1h + Stage 3c 训练(进行中);合计控制在 6h 内
- NOT_RUN: P1-D、Flow Matching、门角 PCA 拟合器(区域级指标已足够)、hold-action/retreat 探针分支、ECE 校准

## 7. 五项明确决策

1. Factorization 是否保留?**否——降级为训练技巧**(匹配预算全表 ≤0,P1-W 可复现,P1-F 反而最差;详见 §1)
2. Voxel/Point/View 正式表示?**体素(canonical voxel)**——阈值扫描定稿(stage3c/report.md):域内点表示前沿更优(clean +0.05–0.13)但随机化下 falseOcc 膨胀 8×、匹配 fo 处体素反超(0.474 vs 0.348);体素另承载 unknown/visibility 语义接口。域内无随机化场景点表示是合理替代。
3. 对象动力学是否真实可预测?**是**(door_new_recall 0.58–0.87 vs persistence 0.0,prec 1.0)——该结果不依赖 factorization。**反事实检验补充**(`metrics/action_counterfactual.json`):动作条件化在 logit 层精确聚焦于物理变化细胞(p1: |Δ| 0.572 vs 背景 0.026,22×),但强度不足以跨越二值决策边界(仅 2–3.5% GT 翻转细胞依赖动作)——同时解释了 precision=1.0/recall 中等,并给出下一代架构的明确改进方向:放大 action→geometry 通路。hold 仿真分支 NOT_RUN(时间盒;模型侧敏感性已测)
4. Observer-Factorized 架构实现条件?**具备且已验证**(L4 完整证据链):①同物理 2.5mm 相机偏移 → 146 cells/帧可见性翻转、物理 0 变化(probe_vis_compare);②**tiny Observation Model 只训相机配置 A,跨配置 B 迁移 visIoU 0.9859 vs 同配置 holdout 0.9874(损失 0.0015,TRANSFERS)**(`observer_proto/result.json`)——可见性是 (几何, 相机参数) 的函数,可作为独立专家学习。注:迁移域为"跨相机配置"(单 episode),跨场景迁移未测;偏移幅度 2.5mm,强干预 stress test 留待后续
5. 是否进入 Tiny Joint WAM?**有条件进入**:先 Observation Model tiny prototype(visibility 跨相机迁移验证,<1h GPU,见 research/next_model_design.md);Geometry-Action 闭环对照超出本批预算,列为下一阶段提案
