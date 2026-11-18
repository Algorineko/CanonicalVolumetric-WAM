# Stage 3b §2 重放对齐裁决(2026-11-18)

## 流程已实现并跑通

`scripts/replay_episode_gt.py`(robottwin env):collect_data 第二阶段语义(seed.txt[i] ↔ instruction seed 交叉验证 ✓;`_traj_data` 关节路径;need_plan=False)+ 逐帧捕获:
- 微波炉 root pose(p,q_wxyz)+ 门角 `get_qpos()[0]` + **全 link 世界位姿**(link_0=门)
- 三相机 actor 分割(240×320 原始 id)+ 静态体 id 名映射(ground/wall/table ✓)
- **机器人真实 qpos/qvel(38 维,原始 HDF5 从未存储)**
- 三相机 K(intrinsic_cv)/T(cam2world_gl 与 extrinsic_cv)
- 帧序号(时间 = 帧号/15Hz)
- 自检:门 link_0 朝向变化 65.0° == 门角 1.134 rad = 65.0°(几何自洽 ✓)

## 对齐验证(`replay_alignment_demo_clean_balanced_10.json` + 腕外参专项)

| 项 | 结果 |
|---|---|
| 帧数 | 重放 427 vs 原始 obs 425(多 2 帧为末帧重复捕获;前 425 行与原始 obs 0..424 对齐) |
| head 外参(静止相机) | 逐帧 **0.00e+00 精确** |
| **左腕外参(臂上相机)** | **0.00–21.2mm**,快运动/接触相位尖峰(t=24:21mm、t=192:2.6mm、t=360:2.1mm),准静态 <0.2mm |
| 右腕外参 | 恒 0.0(右臂静止任务) |
| RGB | 逐帧均值 0.6–2.0(JPEG 噪声级);快运动帧尖峰至 3–12 |
| depth(rint 后) | 均值 95.4% 逐像素精确;快运动帧 0.83–0.88;|diff| 中位 2mm(p95 43mm,深度边缘) |
| 差异空间分布 | **集中于左臂+被抓住的微波炉门/把手**;桌面/墙体/微波炉体匹配 |

## 裁决

**重放 = 近孪生,非逐位孪生**:臂运动学经 PhysX 动力学(接触求解非确定性)在动态相位与原始 episode 分叉至 ~2–21mm(臂面/被抓物面深度边缘)。桌面/场景/门角轨迹自洽。

依 §2 规则:**重放标签不得直接附加到原始 HDF5 数据**;重放作为**独立再生完整 episode** 使用:
1. 重放 episode 自成一体:用自己的 RGB-D + 自己的物体/门 GT(自洽),构成 door-region/object-region 评估的独立探针;
2. 原始 episode 的几何缓存不做任何修改;
3. 重放 episode **首次拥有机器人真实 qpos/qvel** ——后续可用于消除 FK 中段滞后(未来工作)。

## 与 amodal 声明的关系

仅三视角深度 ≠ 完整物理占用 GT;door/object 区域由 **link 位姿 × mesh 点样本** 构建(几何真值,非 amodal 完整几何);完整 amodal 几何若需要,使用仿真 mesh 单独构建(后续工作,本批 NOT_RUN)。
