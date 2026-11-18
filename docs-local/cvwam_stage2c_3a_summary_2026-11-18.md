# CV-WAM Stage 2c + 3a 结论摘要(经对抗性验证修订)

> 2026-11-18 第二批 | 执行:Stage 2c(数据动态性验证)+ Stage 3a(tiny dynamics predictor),无完整 WAM 训练
> 研究方向调整(本轮生效):GT-Depth 固定、VGGT RGB-only 暂缓、20mm 全局体素(10mm 作精度参考)、**新假设:物理占用变化与可见性变化解耦**、无 TTT/物理记忆/Differential WAM
> 详细数据:`runs/stage2c/report.md`、`runs/stage3a/report.md`、`runs/*/metrics/`、`runs/stage3a/failures/`
> 本摘要所有数字来自实际实验并经 4 路独立对抗性审查(数字复算/泄漏/指标定义/Stage2c 复算)后修订。

## 1. Stage 2c:数据动态性验证(核心发现)

### 1.1 Hit/Free/Unknown 3×3 转移矩阵(t+1/4/8/16;5 episodes × ~2965 帧 @20mm,网格 40×65×45)

全网格体素占比(逐集率均值):

| offset | physical flip (hit↔free) | visibility flip (obs↔unknown) | unknown→observed (reveal) |
|---|---:|---:|---:|
| t+1 | 0.000074 | 0.000340 | 0.000134 |
| t+4 | 0.000212 | 0.001236 | 0.000470 |
| t+8 | 0.000356 | 0.002260 | 0.000844 |
| t+16 | 0.000614 | 0.003820 | 0.001386 |

- **可见性翻转率 = 物理翻转率的 4.6–6.3×**(k≥4 时 5.8–6.3×),随 k 近线性增长。
- **物理翻转极稀疏**(k=16 也仅 0.06%)→ "完全不变"基线可获得 99.9% 体素级准确率;动态区域指标是唯一可靠判据。
- **桌面区域(clean)逐集物理翻转率 = 0** 而 20mm 下桌面可见性翻转在 k=16 达 1.27%(ep10)——物理静止表面仍有大量观测状态动态。
- 机器人区域(远端臂+手爪)翻转率最高(phys 1.9%,vis 6.0% @k16);肩/肘段在网格外(显式声明)。

### 1.2 可见性变化归因(相机运动 vs 场景变化;40 对 × 3 相机)

- **静止的 head 相机新增 915 个 hit 体素,100% 归因于场景变化**——门摆开把内腔暴露给零运动的相机(逐帧外参恒定已核验)。
- 左腕:1460 体素来自相机运动 + 227 来自场景变化;右腕 0(左臂任务)。
- **解耦假设的证据链完整**:可见性翻转率 ≠ 物理翻转率(量级)、成因独立(相机移动 / 遮挡物移动 / 物体位移)、桌面反例(零物理变化 + 大量可见性变化)。

### 1.3 诊断区域(独立 GT;NOT_AVAILABLE 如实标注)

- **机器人区域 AVAILABLE**:numpy FK + `state/*_ee_poses` 腕部精确校正 + curobo 碰撞球(71+34)逐帧标记。FK-vs-记录位姿(左臂 ep10):mean 1.41 / p95 4.30 / max 11.64mm——`state/*_arm_joint_states` 是 **PD 驱动目标非真实 qpos**(中段连杆滞后保留并声明),`state/*_ee_poses` 是真实仿真 link6 位姿(**world 坐标,wxyz**——已实测裁决)。
- **物体区域 NOT_AVAILABLE**:缓存数据无逐帧 object pose / 门角 / 分割(mesh/actor 分割从未写入;pkl2hdf5 会丢弃 seg)。确定性再生命径已核实未执行:collect_data 第二阶段按 seed + `_traj_data` 关节路径重放 + ~10 行 get_obs 补丁记录 `microwave.get_pose()/get_qpos()[0]`(门角,成功判据 ≥0.7×π/2);分割在同一 gbuffer pass 边际成本≈0。代理区域(dynamic ∩ 网格 − robot)仅作参考。

### 1.4 Workspace 修正(重要数据事实)

- **randomized 场景桌面更低(0.726–0.730 base)且场景更宽**;原 bbox z_min=0.74 会切掉 randomized 桌面(randomized 域内占比仅 33.8–38.4%)。
- 修正后 bbox x[0.30,1.10], y±0.65, z[0.71,1.60]:**五集域内占比 99.32–99.79%**;排除点 100% 为桌底/边缘(非操作区)。逐集桌高写入 manifest(clean 0.7403 / rand0 0.7302 / rand1 0.7263)。

### 1.5 Camera dropout 数据与 episode 级缓存

- dropout 子集体素(head_only/head_left/head_right;目标恒为全 3 视图 GT,隐藏视角观测绝不进入输入):val 210 帧 + test 303 帧 = 513 × 3 子集。
- 观测域占比:head_only 9.4% vs 全 3 视图 17.4%(网格)。
- Episode 级体素缓存(20mm ×5 集 + 10mm ×1 集)使训练零重复 ray casting;10mm 参考转移(k=16):vis/phys = 13×(与 20mm 同向)。

## 2. Stage 3a:Tiny Dynamics Predictor

统一架构 1.56M 参数(3D CNN base32 + FiLM 条件化 + 4 offset 头 t+4/8/12/16,zero-init=持续性起点);输入 = 当前体素(4ch)+ qpos(14)+ action chunk(16×14,action 组=下一帧绝对关节目标,单测通过);**按 episode 划分**(train=clean 10/25/40,val=rand0,test=rand1;训练代码无 test 路径,审查确认);fp32(计划 bf16 未实现,如实记录);总 GPU **≈0.7h**(上限 6)。

### 2.1 主表(val+test 合并;val/test 分离值在 `val_test_split.json`)

| model | occIoU | occ-set F1@±1 | **dynIoU** | chgF1 | revAcc | falseOcc | visIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 persistence | **0.968** | **0.997** | 0.000 | 未定义 | 0.764 | **0.0016** | — |
| P1 (3-state CE) | 0.832 | 0.989 | 0.295 | 0.067 | 0.811 | 0.0082 | — |
| P1-R (residual) | 0.609 | 0.972 | 0.189 | 0.040 | 0.776 | 0.0283 | — |
| **P1-V (decoupled,修复后)** | **0.858** | 0.994 | **0.347** | **0.088** | **0.870** | 0.0202 | **0.955** |
| act-only | 0.313 | 0.776 | 0.153 | 0.026 | 0.687 | 0.0773 | — |

- P0 的聚合优势是表示偏置(0.06% 翻转率);**动态区域是能力判据**:P1-V 0.347 > P1 0.295 > P1-R 0.189 > act-only 0.153 > P0 0。
- **P1-V 在 test 上 dynIoU 无衰减(val=test=0.347)**;P1-R 在 test 上 occ 崩至 0.443(val 0.852,单 val 场景过拟合)。
- 真实 P1 逐 horizon:occIoU 0.859(k4)→0.808(k16),dynIoU 0.245(k4)→0.334(k16)。

### 2.2 消融

- **Action 跨轨迹置换**(~82% 供体动作来自不同 episode):P1 dynIoU −6%、chgF1 −12%;P1-V −7%/−11% → 模型对动作**敏感但弱**,动态预测主要来自当前体素局部外推;act-only 证明纯任务进度捷径存在(0.153)但远不充分。
- **Camera dropout**:P1/P1-V 的 **dynIoU 对相机子集鲁棒**(P1-V 0.332–0.348 跨全部子集;P1 0.264–0.295),静态 occupancy 随输入可见域收缩而降;persistence 无此性质(0.968→0.665 且动态恒 0)→ **学到的动力学跨相机配置迁移**。P1-V 的 vis 头正确适应输入配置(head_right 0.949 ≈ 全视图 0.955;head_only 0.714 = 子集输入与全 3 视图目标的差距)。
- **阈值校准前沿**(val 选阈值,test 应用):P1-V 默认点全面占优;压 falseOcc 至 P0 量级时 P1 保留更多动态(3 类 CE 的 unknown 锚点效应)。

### 2.3 四个明确问题的答案

1. **P1 超过 persistence?** 动态判据**是**(0.295 vs 0),聚合 IoU 否(表示偏置);校准后 P1 达 P0 同级 falseOcc 并保留部分动态 → 学习到真实动态信号。
2. **P1-R 有优势?** **否**,全指标劣于 P1 且 test occ 崩溃(0.443);放弃。
3. **P1-V 真正提供额外收益?** **是(修复 pos_weight 后)**:occ +0.027、dynIoU +18%(0.347 vs 0.295)、chgF1 +31%、revAcc +0.059、test 无衰减、vis 头 0.955;**且该收益建立在"unknown 不作 free-space 标签"的解耦监督上**——直接支持本轮新假设。**关键教训**:原"P1-V 无收益"的结论是 pos_weight 写反(0.123 vs 8.15)的训练 bug 伪影,验证工作流(blocker)+ 单变量重训翻转了结论。
4. **是否进入 Point/View 对照与 Flow Matching?** **值得进入**:Gate G3 前三条满足(overfit 成功、held-out 动态超 persistence、非静态复制),P1-V 给出 canonical 体素+可见性结构的正向证据(H3 方向)。顺序建议:P2(point)/P3(per-view pointmap)对照 + P1-V 多种子稳定性复算(当前单 seed,如实标注)→ 之后决定 FM(P4)。

## 3. 对抗性验证抓到的问题(全部修复并存档)

1. **BLOCKER**:P1-V pos_weight 写反 → 结论反转(最大教训:损失配置静默错误只能靠对抗性数字复算 + 单变量重训发现)。
2. 主表 val/test 混淆 → 分离后 P1-R/act-only 的 test 真实水平显著更差。
3. 报告数字误引 shuffle 变体、CE 权重出处错、bf16 未实现却声明、"surface F-score" 实为 occupied-set F1(命名已限定)、评估延迟含 57% 数据加载(前向仅 ~4ms)、dropout 行为 stride-8 子采样(已披露)。
4. Stage 2c:两格转移数字单集污染、workspace 数据过期(已换修正后实测)、若干绝对化表述已限定范围。
5. 评估框架两 bug(批内置换 shuffle 是无效消融→cross-trajectory 修复;dropout 无后缀覆盖主结果→重跑)。
存档:`runs/stage3a/failures/`(含 bug 版 P1-V 完整产物)。

## 4. 下一步(最低成本优先)

1. P1-V 多种子(3 seeds)稳定性复算(~0.7 GPU-h);
2. P2 point-set / P3 per-view pointmap 对照(缓存与管线就绪);
3. 若 P1-V 优势复现 → P4(voxel latent Flow Matching)小规模验证;
4. 数据再生(可选,解锁 object-region GT 与 amodal 声明):单集确定性重放验证 + object pose/seg 记录(~10 行补丁,成本 ~数分钟/集)。
