# Stage 2c — 数据动态性验证报告(CV-WAM)

- **Date**: 2026-11-18;GPU: 无(本阶段全 CPU);数据: open_microwave,5 episodes(clean 10/25/40 + randomized 0/1,~2965 帧 @15Hz)
- 数据源: `/data/openmw_balanced_work`(逐帧内外参 + ffv1 GT depth);体素缓存 `/data/canonical-volumetric-wam/cache/voxels_ep/`
- 本阶段裁决性发现:**randomized 场景桌面更低(0.726–0.730 base)且场景延展更宽,原 bbox z_min=0.74 会切掉桌面**;已将 workspace 修正为 x[0.30,1.10], y±0.65, z[0.71,1.60](网格 40×65×45 @20mm),逐集桌高存入 manifest。

## 1. 3×3 Hit/Free/Unknown 转移矩阵(t+1/4/8/16)

全网格 headline 率(5 episodes 均值,体素占比;完整矩阵 `transition_stats.json`,热图 `plots/transition_heatmaps_all.png`):

| offset | physical flip (hit↔free) | visibility flip (obs↔unknown) | unknown→observed (reveal) |
|---|---:|---:|---:|
| t+1 | 0.000074 | 0.000340 | 0.000134 |
| t+4 | 0.000212 | 0.001236 | 0.000470 |
| t+8 | 0.000356 | 0.002260 | 0.000844 |
| t+16 | 0.000614 | 0.003820 | 0.001386 |

分区域(ep10 为例,均值趋势同):

| offset | table phys | table vis | robot phys | robot vis | all phys | all vis |
|---|---:|---:|---:|---:|---:|---:|
| t+1 | 0.00000 | 0.00098 | 0.00195 | 0.00583 | 0.00008 | 0.00049 |
| t+8 | 0.00000 | 0.00697 | 0.01020 | 0.03213 | 0.00041 | 0.00328 |
| t+16 | 0.00000 | 0.01269 | 0.01900 | 0.05955 | 0.00073 | 0.00531 |

要点:
- **可见性翻转率 4.6–6.3× 物理翻转率**(k=1 为 4.6×,k≥4 为 5.8–6.3×),且随 k 近线性增长;
- **桌面区域(clean)逐集物理翻转率 = 0**(池化计数口径 1.8e-6)而 20mm 网格上桌面可见性翻转在 k=16 达 1.27%(ep10)——物理静止表面仍有大量"观测状态"动态;
- 全网格物理变化仅 ~0.06–0.07% → "完全不变"基线可获得 99.9% 体素级准确率,动态区域指标与类平衡监督不可省略;
- 机器人区域(远端臂+手爪,网格内)翻转率最高;肩/肘段在网格外(显式声明)。

## 2. 物理变化 vs 可见性变化(解耦假设的核心证据)

三条独立证据:
1. **转移矩阵**:visibility 翻转率 5–7× physical 翻转率(全网格与桌面区域一致),桌面上物理=0 而可见性≈1.3%——两者的"状态分布"不同,不是同一过程的两份拷贝;
2. **归因分析**(`vis_attribution.csv`,40 对 × 3 相机):静止 head 相机新增 915 hit 体素 **100% 来自场景变化**(门摆开露出内腔——零相机运动下的可见性变化);左腕 1460 体素来自相机运动 + 227 来自场景变化;右腕 0(左臂任务)。可见性变化的成因(相机移动 / 遮挡物移动)与物理变化的成因(物体/机器人运动)可以独立发生;
3. **Camera dropout 结构**:同一时刻不同相机子集构建的 observed/unknown 分布差异显著(head_only 观测域占网格 9.4% vs 全视图 17.4%,`dropout_coverage.json`),而 hit 分布按子集可见性收缩(head_only hit 0.011 vs 全视图 0.021)——可见性是"观测过程"的函数,占据是"世界"的函数(占据状态在被共同观测的子集上跨配置一致)。

**结论:物理占用变化与可见性变化应当解耦生成**(P1-V 的设计依据成立);把 Hit/Free/Unknown 统一当作物理状态会把"门开露出内腔"这类纯可见性事件误编码为物理变化。

**可见性归因(相机运动 vs 场景变化;`vis_attribution.csv`,40 对采样 × 3 相机,k∈{8,16})**:
- **cam_head(静止相机): 新增 hit 体素 915,100% 归因于场景变化** —— 微波炉门摆开把内腔"暴露"给静止的头相机。可见性变化可以在零相机运动下发生。
- cam_left_wrist: 相机运动驱动 1460 体素(11 对,位移>1cm)+ 场景变化驱动 227 体素(29 对)。
- cam_right_wrist: 0 新增(左臂任务,右臂几乎不动)。
- 结论:可见性翻转的两个成因(相机移动、遮挡物移动)在数据中**都真实存在且量级显著**;把 Hit/Free/Unknown 当作统一"物理状态"生成会把第二种成因误编码为物理变化。

## 3. 诊断区域(独立 GT)

- **机器人区域**: AVAILABLE——`build_robot_region.py`:numpy FK(fl_base_joint→joint1..6)+ `state/*_ee_poses` 腕部精确校正 + curobo 碰撞球(left 71 + right 34,半径 0.010–0.035m)逐帧标记体素。FK-vs-记录位姿误差(左臂 ep10,`_robot_region/.../meta.json`):mean 1.41 / p95 4.30 / max 11.64mm——即 `state/*_arm_joint_states` 是 **PD 驱动目标而非真实 qpos**(侦察+实测证实),腕部用记录的 link6 真值校正,中段连杆滞后保留(标注为不确定度)。
- **机器人区域裁剪说明**: 肩/肘段(link1–4,x≈0.23–0.26,z≈0.84–0.94)在 bbox 外或边界上——**只标记网格内部分,显式声明**(不静默裁剪);运动诊断以远端臂+手爪为主。
- **物体区域**: **NOT_AVAILABLE(缓存数据无逐帧 object pose/门角/分割)**。可再生命径(侦察已核实,未执行):collect_data.py 第二阶段按 seed + `_traj_data` 关节路径确定性重放,~10 行 get_obs 补丁记录 `microwave.get_pose()/get_qpos()[0]`(门角;成功判据 ≥0.7×π/2)+ 同一 gbuffer pass 的 actor 分割(边际渲染成本≈0,但 pkl2hdf5 会丢弃分割,需自定义保存)。深度视频本身就是重放产物(帧数逐集 T+1 吻合),确定性有实证基础但需一次验证。
- **替代代理**: dynamic(物理翻转)∩ 网格内 减 robot 区域 = "object+未建模运动"代理区域,仅作参考,不冒充物体 GT。

## 4. Workspace 覆盖与 20mm 保留度

- **修正后**域内点占比(`workspace_coverage.json`,bbox 修正后重测):clean 99.71–99.79%,randomized 99.32–99.65%(五集 99.32–99.79%);**修正前** randomized 仅 33.8–38.4%(桌面被切,历史测量已存档)。
- 被排除点 100% 在 bbox 15cm 邻域内(桌底/远端边缘,z<0.71 或 y 略超,非操作区);关键操作区(微波炉正面 x≈0.75–0.9、双爪、门把手)在网格内。
- 20mm 保留度:smoke 阶段(旧 bbox)开门变化 62 hit 单元(10mm 216);10mm 转移参考(k=16):vis/phys = 0.0052/0.0004 = 13×(与 20mm 同向,信号保留);10mm 缓存(ep10)作精度参考。

## 5. Camera dropout 数据

- 子集:head_only / head_left / head_right(全 3 视图 = 主缓存"all");构建:`build_dropout_voxels.py`,eval 帧 t 步长 8(val 210 帧 + test 303 帧 = 513)。
- 语义(按研究规范):**被隐藏视角的观测绝不作为该视角实验的输入**;完整三视角观测仅定义共享的监督/目标区域(future GT 由 3 视图构建,对所有 dropout 条件一致)。

## 6. Episode 级体素缓存

- 20mm 全帧 × 5 episodes + 10mm × ep1(精度参考);每帧 npz(hit/free/observed/unknown/view_count/conflict + qpos + 16×14 action chunk);manifest 记录 T、频率、语言指令、逐集桌高、bbox。
- 训练时零重复 ray casting(缓存后每帧直接读 npz)。

## 7. 结论与对 Stage 3a 的输入

- 转移矩阵:可见性翻转 5–7× 物理翻转,桌面上物理=0/可见性 1.3%;物理变化极稀疏(0.06%)→ 动态区域指标必须;
- 解耦假设(物理占用变化 ⊥ 可见性变化)在数据层成立:量级、区域分布、成因三方面独立,支持 P1-V 的解耦监督设计;
- Stage 3a 数据集:train=clean 10/25/40(6096 对),val=randomized 0(2212 对),test=randomized 1(3212 对),按 episode 划分,无相邻帧泄漏;相机 dropout 输入体素已建(val 210 帧 + test 303 帧 = 513,3 子集);
- Object-region metrics:NOT_AVAILABLE(代理 = dynamic∩网格 减 robot 区域,仅参考);amodal occupancy completion 不可验证(无完整几何标签)。
