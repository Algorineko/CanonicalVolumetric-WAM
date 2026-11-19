# Stage 3b Final Report(2026-11-19,持续更新)

审查方指令: docs-local/plan/current-plan.md。数字全部实测,回指 metrics/ 下文件;未完成项标 PENDING。

## 1. 机制归因裁决(审查方 §1.2)→ 决策 1: Factorization 是否保留?

- Pareto 匹配 falseOcc 评估(VAL 定阈值→TEST 固定,9 模型增量 CSV):`metrics/pareto/`(P0/p1_s42/p1_s43 已出,其余运行中)
- 裁决脚本: `scripts/attribution_verdict.py` → `metrics/attribution_verdict.md` — **PENDING(等 9 CSV 齐)**
- **固定阈值旁证(door_eval,thr=0)**:`metrics/door/door_summary.csv`
  - P1 家族 dyn(k=16): 0.392/0.420/0.434(3 seeds);P1-V: 0.442/0.407/0.419
  - **P1-W(k=16 dyn 0.334)更差;P1-F(dyn 0.088)灾难性更差** — 权重移植与精确 NLL 等价形式都不能复现解耦收益
  - P1-F 反常:其 val_loss(0.0247)低于 P1(~0.05)但动态指标崩坏 → **损失数值等价 ≠ 优化/机制等价**(因式形式的梯度路径改变训练动力学)
- 预裁决(待 pareto 匹配预算确认):解耦监督的收益是**机制性**的(masking + 双头分解),factorization 倾向保留。

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

1. Factorization 是否保留?**PENDING**(预裁决:保留,等匹配预算数字)
2. Voxel/Point/View 正式表示?**PENDING**(Stage 3c)
3. 对象动力学是否真实可预测?**是**(door_new_recall 0.58–0.87 vs persistence 0.0,prec 1.0)
4. Observer-Factorized 架构实现条件?**PENDING**(探针已证明"同物理不同相机"可构造,可见性对比构建中)
5. 是否进入 Tiny Joint WAM?**PENDING**(决策 1–4 汇总后)
