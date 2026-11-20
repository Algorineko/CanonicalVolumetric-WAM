# Stage 4b — P0:提升空间审计(A0 压力测试 + 新任务)

更新: 2026-11-20 17:1x。执行规格: `docs-local/cvwam_assessment_roadmap_2026-11-20.md` §5 P0。

## 前置事故(影响基线口径,详见 `runs/stage4a/failures/fm_integration_sign_bug_note.md` 追加 1-6)

**盲策略事故(已修复)**:eval client 曾把 `extrinsic_cv`(world→cam 逆矩阵)当作
`extrinsics_matrix`(契约=cam2world_gl)→ 在线体素全零 → 2026-11-20 17:00 前的
全部闭环数字(含 a0 筛查 19/20×2、baseline 盲版 20/20 = git fe95f64)为**无条件
先验回放**,非体素条件控制。修复后 dump 验证 x_nonzero=1.0、同帧 observed IoU 0.973。

**采样器定论(修复输入下)**:x0jump 9/9→20/20;euler_minus 0/5(盲/真实输入均败,
离线 MSE 却 0.0043——闭环协变量偏移假说,见笔记追加 6)。**本阶段全部闭环以
`sampler: x0jump` 执行**。

## stage4a 变体 smoke 终版(修复输入,x0jump,5-ep seed 0)

| 变体 | 盲输入(作废) | 修复输入 |
|---|---|---|
| a0 | (5/5 盲) | 5/5(`rollout_results/a0_smoke5_rates_x0jump.txt`) |
| a1 | (5/5 盲) | **5/5**(`a1_smoke5_rates_x0jump.txt`) |
| a2 | 0/5 盲 | **5/5**(`a2_smoke5_rates_x0jump.txt`) |
| a3 | 挂死(误报,盲 5/5) | **5/5**(`a3_smoke5_rates_x0jump.txt`) |
| a3mask | 未跑(盲) | **5/5**(`a3mask_smoke5_rates_x0jump.txt`) |

全部变体在真实观测条件下通过闭环 smoke;"世界变体闭环失败"的旧叙事由伪影解释,
L9 的 world→action 内容不敏感仍仅由离线干预支撑。

## 目的

找到 A0(无世界模型基线)会失败、而世界信息理论上能帮上忙的场景。

## 预注册门 G0(实验前锁定,不得事后调整)

**至少一类扰动或一个任务上 A0 ≤ 70%(20 trials,Wilson 95% CI 上界 < 90%)。**
操作判据(wilson_ci.py 实测):**≤14/20**(14/20 Wilson 上界 0.855 < 0.90;15/20 失败于成功率条款 75% > 70%)。

- 过 → P1(oracle 未来上界,A-oracle vs A0 vs A-current-belief)入队。
- 未过 → 世界模型在本仿真设定下无提升空间 → roadmap §6 路线 B(停止候选;写决策备忘录 + autopilot HALT 等用户裁决)。

统计口径:每条件 20 trials;种子沿用 A0 Clean-20 筛查的同一 20-seed 配对集(--seed 0 --test_num 20,trial 种子 100000+i,expert-check 跳过模式记录于 metrics/stress/<cond>_seeds.txt);成功率附 Wilson 95% CI;条件间差异用配对 bootstrap(如需)。

## 条件矩阵(roadmap §5 P0.2–P0.3)

| 组 | 条件 | 档位 | 状态 |
|---|---|---|---|
| 相机平移 | head-cam-dis | 0.05 / 0.15 / 0.30 | NOT_RUN |
| 相机旋转 | Rodrigues(复用 probe 逻辑) | 7.5° / 20° | NOT_RUN |
| 视角缺失 | 推理时丢相机(在线版,复用 build_dropout_voxels 逻辑) | 丢头相机;丢随机一台 | NOT_RUN |
| 遮挡 | head 相机与门之间静态遮挡物;若 API 不便用观测侧 mask 模拟,两者分开报告 | 1 档 | NOT_RUN |
| 布局 OOD | 微波炉位姿随机化超出训练分布 | 1 档 | NOT_RUN |
| 新任务 | open_laptop / put_object_cabinet 各训 A0 + Clean 20 | 2 任务 | NOT_RUN |

全部在 open_microwave Clean 20-seed 集上配对;新任务除外。

## 结果

> **证据有效性(规划裁决 §0,2026-11-20)**:所有盲输入(空体素)闭环数字
> ——a0 筛查 19/20×2、a0 盲 baseline 20/20(git fe95f64)、盲 smoke 5/5×2——
> 标记 **INVALID**,不得用于论文性能声明;仅作为"无条件先验回放率"对照保留。
> 修复输入数字单列如下。

### A0 参照(修复输入,2026-11-20 18:20 完成)

- open_microwave Clean 20-trial:**20/20 = 100.0%**(`metrics/stress/baseline_rates.txt`,
  实际种子集见 `baseline_seeds.txt`;盲版同分 20/20——盲先验回放与条件控制在
  单任务 clean 上不可区分,正是需要扰动暴露差异的原因)。

### 压力条件(每条件 20 trials,配对 seed 0)

| 条件 | 状态 | 成功率 | 出处 |
|---|---|---|---|
| cam_t030(头相机平移 0.30m,真 E 视角 OOD) | **完成 0/20** | **0.0%,Wilson 上界 0.168 → 门条件满足** | `metrics/stress/cam_t030_rates.txt` |
| cam_r20(旋转 20°) | **完成 15/20** | 75.0%,Wilson 上界 0.888,成功率条款 0.75>0.70 → **单独不满足** | `metrics/stress/cam_r20_rates.txt` |
| drop1of3(随机丢一台相机,trial 内固定) | **完成 20/20** | 100%(阴性) | `metrics/stress/drop1of3_rates.txt` |
| cam_t030blind(偏移相机+零观测,§3 诊断) | **完成 5/5** | 100% —— 盲>偏移观测 | `metrics/stress/cam_t030blind_rates.txt` |
| openlaptop_a0 5-trial smoke | RUNNING | | `metrics/stress/openlaptop_a0_rates.txt` |
| blind_openlaptop 5-trial 对照 | RUNNING | | `metrics/stress/blind_openlaptop_rates.txt` |
| cam_t015 / cam_t005 / cam_r7.5 / drop_head / occ / lo | NOT_RUN(裁决 §2 冻结,不补齐) | | |

### P0 Failure Attribution(§3,2026-11-20 20:1x,详见 failures note 追加 8)

ep10(评测集内,seed 1000019)同物理状态 nominal/shifted 成对渲染,427 帧对齐,43 帧配对分析
(`metrics/attribution/att_{frames.csv,summary.json}`):

- **偏移相机观测覆盖 = 名义 2.07×**(15k→31k 格),门区域观测率 0.977(build_replay_geometry 分割门区域交叉验证;原近似盒 0.80 低估) —— **感知信息缺失排除(§10.1:否)**;
- 同噪声动作差仅 0.0025(EE 2.4cm)但闭环 0/13 ⇒ 小偏差自执行复合;
- 训练画布 observed ~11-21k,shift 31k 超上界 1.5× ⇒ **画布统计量分布偏移成立**;
- **cam_t030blind:零观测 > 偏移观测(§10.2:是)——偏移观测主动有害**;
- 含义:修复候选第一梯队 = B3 相机增广/输入归一化(画布统计量鲁棒),其次才是历史信念;B1/B2 融合也改变统计量,须 B3 对照剥离机制归因。

## 失败存档

failures/;未执行项标 NOT_RUN。

## G0 正式裁决(2026-11-20 20:4x,预注册判据)

**cam_t030:0/20 = 0.0%,Wilson 95% CI [0.000, 0.168],上界 < 0.90 且 0.0 ≤ 70% → G0 条件满足(MET)。**
cam_r20(15/20,75%,上界 0.888)与 drop1of3(20/20)单独不满足。剂量响应:平移 0.30m 完全致残(0/20)、旋转 20° 轻度退化(75%)、丢一台相机无影响(100%)。
按 roadmap §5:过门 → **P1(oracle 上界,重设计版见 p1_oracle_design.md)进入待授权队列**。
失败机制归因(P0 Failure Attribution):观测覆盖 2.07×/门 0.977 排除感知缺失;画布统计量 OOD + 闭环复合(偏移观测严格有害:盲 5/5 vs 观测 0/20)。
