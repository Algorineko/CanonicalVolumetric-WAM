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

### A0 参照(修复输入,2026-11-20 18:20 完成)

- open_microwave Clean 20-trial:**20/20 = 100.0%**(`metrics/stress/baseline_rates.txt`,
  实际种子集见 `baseline_seeds.txt`;盲版同分 20/20,留档 git fe95f64 —— 盲先验回放
  与条件控制在单任务 clean 上不可区分,正说明该任务对世界信息无需求,需扰动暴露差异)。

### 压力条件(每条件 20 trials,配对 seed 0)

| 条件 | 状态 | 成功率 | 出处 |
|---|---|---|---|
| cam_t030(头相机平移 0.30m,真 E 视角 OOD) | RUNNING | | `metrics/stress/cam_t030_rates.txt` |
| drop1of3(随机丢一台相机,trial 内固定) | RUNNING | | `metrics/stress/drop1of3_rates.txt` |
| cam_t015 / cam_t005 / cam_r20 / cam_r7.5 | NOT_RUN | | |
| drop_head | NOT_RUN | | |
| occ(观测侧遮挡盒) | NOT_RUN | | |
| lo(布局 OOD,xlim -0.38..-0.28) | NOT_RUN | | |
| openlaptop_a0 Clean-20 | NOT_RUN(训练完成,待筛查) | | |

## 失败存档

failures/;未执行项标 NOT_RUN。
