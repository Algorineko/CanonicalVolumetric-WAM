# Stage 4b — P0:提升空间审计(A0 压力测试 + 新任务)

更新: 2026-11-20 17:1x。执行规格: `docs-local/cvwam_assessment_roadmap_2026-11-20.md` §5 P0。

## 前置事故(影响基线口径,详见 `runs/stage4a/failures/fm_integration_sign_bug_note.md`)

serving 采样器 A/B 实证:x0jump 闭环 5/5(=筛查时代行为),euler_minus 0/3,
euler_plus 离线垃圾。**本阶段全部闭环运行以 `sampler: x0jump` 执行**(wrapper deploy
键锁定)。旧 a0 筛查(19/20×2)与新运行的采样器一致性由 x0jump smoke 复现背书;
baseline 20-trial 在新 wrapper(seeded、x0jump)下重测,作为 P0 的正式 A0 参照。

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

见 metrics/;汇总表待实验落盘后填写,数字一律回指 metrics/ 下文件。

## 失败存档

failures/;未执行项标 NOT_RUN。
