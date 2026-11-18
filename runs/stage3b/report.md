# Stage 3b — Mechanism Validation 报告(CV-WAM)

- **Date**: 2026-11-18;GPU 0;torch 2.7.1+cu128;训练 fp32(bf16 计划未实现,偏差记录)
- 冻结:Stage 3a 终态 commit `56278acf`(+AGENTS.md 后 `2033dd0`),freeze manifest `runs/stage3b/freeze_manifest.json`(sha12 d7b84f05);bug 版 P1-V 完整存档 `runs/stage3a/failures/poswbug_p1_v/`
- **开发集声明**:rand0/rand1 为 development(Stage 3a 用于模型选择/迭代);rand1 不进入本批任何 split;最终独立测试证据 = clean_test(5 新集)+ rand_test(5 新集),split 见 `split_manifest.json`(全模型共用)

## 1. §2 独立物体 GT(重放)

- 管线:`scripts/replay_episode_gt.py`(robottwin env):collect_data 二阶段语义(seed 交叉验证 ✓ + `_traj_data` 关节路径 + need_plan=False),逐帧捕获微波炉 root pose / 门角 / **全 link 世界位姿** / 三相机 actor 分割 / **机器人真实 qpos/qvel(38 维,原始 HDF5 从未存储)** / 三相机 K/T / 帧序号。自检:门 link_0 朝向变化 65.0° == 门角 65.0°(几何自洽)。
- 对齐裁决(`metrics/replay_verdict.md` + `replay_alignment_*.json`):head 外参逐帧 0.00e+00;RGB 均值 0.6–2.0(JPEG 噪声级);**左腕外参 0.00–21.2mm(快运动/接触相位尖峰)** → **重放 = 近孪生非逐位孪生(PhysX 接触求解非确定性)**。
- **按 §2 规则**:重放标签不得附加到原始数据 → 重放作为**独立再生完整 episode** 使用(自带 RGB-D + 自洽门/物体 GT;首次拥有真实 qpos)。差异空间分布集中于左臂+被抓住的门(桌面/墙体/微波炉体匹配)。
- amodal 完整几何:本批 NOT_RUN(仅 link 位姿 × mesh 点样本构建 door/object 区域,不宣称 amodal)。

<!-- REPLAY_GEO_RESULTS_PLACEHOLDER: NN 链验证 + 门区域体素 -->

## 2. §3 数据规模

<!-- DATA_SCALE_PLACEHOLDER: 实际缓存集数/帧数 -->

## 3. §4 matched-loss 对照(P1 / P1-W / P1-F / P1-V)

- 损失定义(全部有单测,`metrics/test_stage3b.log`):
  - P1:3-state CE,逆频权重 [22.65, 2.73, 0.386](train 统计)
  - P1-W:3-state CE,权重 [8.15, 1, 1](P1-V 的 pos_weight 移植)
  - P1-F:因式分解头 + **精确无权三分类 NLL**(L=(1/N_all)[Σ_all BCE_vis + Σ_obs BCE_occ];单测数值等价 |Δ|=6e-8)
  - P1-V:因式分解头 + mean_all[BCE_vis] + 8.15·mean_obs[BCE_occ]
- 概率关系单测:p_hit=p_obs·p_occ、p_free=p_obs·(1−p_occ)、p_unknown=1−p_obs(simplex ✓)

<!-- MATCHED_LOSS_PLACEHOLDER -->

## 4. §5 Pareto 评估(falseOcc 预算 0.2%/0.5%/1%/2%,VAL 定阈值 → TEST 固定应用)

<!-- PARETO_PLACEHOLDER -->

## 5. §7 物体级与动作条件化验证

<!-- OBJECT_LEVEL_PLACEHOLDER -->

## 6. §6 P1-D / §8 Point-View 对照

<!-- P1D_P2_PLACEHOLDER -->

## 7. 六问结论

<!-- FINAL_ANSWERS_PLACEHOLDER -->
