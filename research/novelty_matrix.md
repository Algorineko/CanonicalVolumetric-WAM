# CV-WAM Novelty Matrix

更新:2026-11-19(按审查方 current-plan.md §2:补 DSR-Net/UNITAS/AWAM/OccWAM,删除"无先例"绝对化表述)。核心差异声明:**Visibility-Decoupled Canonical 3D World Model** —— 物理占用与观测过程在生成目标层面分离(因式分解双头 + unknown 不作 free-space 标签),未来 3D 世界状态被独立物理指标证伪(dynIoU / door-region / falseOcc Pareto / persistence 基线)。

## 主表

| 论文 | 3D 表示 | 可见性解耦 | 物理可证伪世界评估 | 与本工作关系 |
|---|---|---|---|---|
| **DSR-Net** (arXiv:2011.01968, CoRL20) | **TSDF 体素动态场景表示**(多视角交互聚合);逐物体 amodal TSDF + SE(3) 运动预测 + MPC 推压 | ✗ TSDF 聚合无 unknown/可见性语义(全部积分为主) | 部分:object permanence/amodal completeness/逐物体 SE(3) 误差,但**无 action-chunk 条件化的多步未来占用预测**,无 falseOcc/可见性指标 | **最近先例,必须精确区分**:①它是"多视角 TSDF 融合+逐物体刚体运动假设"(SE(3) 每物体一个变换),我们预测的是**逐体素 occupancy/visibility 场**,动作 chunk 条件化,允许非刚体变化;②它没有 unknown 状态、没有可见性解耦(AModal 补全靠多视角聚合而非生成式 observer 分离);③评估对象是物体重建/跟踪误差,不是未来动态区域的占用预测。**我们的 door_new_recall/persistence 对照与其 object permanence 精神相近,但机制与监督不同** |
| **UNITAS** (arXiv:2610.12099) | 1.7B 3D-native WAM;**scene flow + action flow 统一度量 3D 系** | ✗(flow 表示无占用/自由/未知三分) | 部分:场景预测 ADE/flow 误差(RoboTwin 最优,比 PointWorld 低 49%) | **表示范式的核心对照**:点轨迹流 vs 我们的 occupancy+visibility 体素场。flow 擅长大变形/物体运动,但天然不表达 free-space(射线负空间)与 unknown(不可观测区),而门开启导致的 free-space 大变化正是我们 Stage 2c 解耦证据的主场。Stage 3c P2 对照即回应此竞争 |
| **AWAM** (arXiv:2610.23396) | 几何条件世界模型 + **主动虚拟视点想象** | 部分(想象未见视点=观测过程建模的一种,但以"新视点渲染"为目标) | ✗ 无物理 occupancy 指标 | 动机相邻、机制不同:它想象**换一个相机往回看**,我们预测**同一相机下未知区的 reveal/occlude**(visibility head 的 rev_acc 指标)。二者可互补(其视点选择可消费我们的 unknown 场) |
| **OccWAM** (arXiv:2610.17688) | 自动驾驶 occupancy + ego flow matching | ✗ 单一 occupancy 状态,无 free/unknown 分解、无可见性 | ✗ | 范式引用点:occupancy-WAM 存在,但驾驶场景 ego-motion 主导;操作场景臂体自遮挡+物体交互使可见性解耦成为必要(Stage 2c 实测 4.6–6.3×) |
| GeoMem-VLA (2610.29866, ICLR27 投稿) | 隐式 GeoPlan tokens(point map 蒸馏)+时序记忆;**明确放弃完整可渲染重建** | ✗ 无分离 | ✗ 不评估内化几何的物理正确性(只有 policy 成功率) | **最直接竞品**。差异:①显式可测量占用场 vs 隐式 token;②p_obs/p_occ 因式分解监督;③物理可证伪评估(门区 new_recall 0.58–0.87 vs persistence 0.0,精确复现见 runs/stage3b/metrics/door/)。其 teacher 蒸馏用法可借鉴,不构成差异点 |
| **ReconVLA** (arXiv:2508.10333) | 无显式 3D 表示;VLA 内以重建式辅助目标(重建注视区域视觉 token)强化感知 | ✗ | ✗(只有 policy 成功率) | **A1(Action FM + P1-V 辅助解码)的文献先例,不是因果验证**。Stage 4a §3 匹配消融(a1 vs a1nofm,3 seeds,masked dynIoU 0.830 vs 0.831,Δ=−0.0018)显示在本设定下 FM 共训练对世界预测无增益;ReconVLA 的机制在 CV-WAM 中**未被因果验证**。 |
| FoundObj (2605.27178, ICML26) | 点云超点分割(FM 先验作 RL 奖励) | ✗ | ✗ | 边缘:物体区域 GT 工具候选;非世界模型 |
| FastVGGT (2509.02560, ICLR26) | VGGT 加速(token 合并,4×) | ✗ | ✗ | 工程使能;不解决实测精度阻塞(MAE 43–78mm) |
| PointWAM (2610.02840) | 点轨迹 | ✗ | 部分(点跟踪误差) | 已在 plan.md §2;无可见性分离 |
| Flex-π (2608.10860) | view-indexed pointmap | ✗ | ✗ | view-indexed 基线 |
| VGGT-World (2603.12655) | 冻结 VGGT 特征 + latent 生成 | ✗ | ✗ | teacher 蒸馏先例(同用法,无解耦) |

## 撞题自查(2026-11-19,按审查方要求去掉绝对化)

- Q: GeoMem-VLA 是否已显式建模 unknown/free?A: 否——tokens 隐式,无占用/可见性语义,无 unknown 概念。
- Q: "可见性解耦"是否有先例?A: **限定表述**:在"canonical 体素占用 + 显式 visibility 头 + unknown 不作 free 标签"的**组合**上,检索范围内(PointWAM/MVG-WAM/WSA1/VGGT-World/OccWAM/GeoMem-VLA/DSR-Net/UNITAS/AWAM)未见同组合;AWAM 的视点想象与 DSR-Net 的多视角 amodal 聚合是两个最近的相邻机制,但均不分离"物理状态生成"与"观测过程"两个监督目标。此表述可证伪、可检索,不再使用"无先例"绝对化。
- Q: 与 DSR-Net 只差数据格式?A: 否——其逐物体 SE(3)+amodal TSDF 假设刚体运动且由多视角聚合补全;我们逐体素生成、动作条件化、显式建模不可观测区,且给出它没有的 persistence/门区基线对照。
- Q: 与 UNITAS 只差指标?A: 否——scene/action flow 与 occupancy 场是两种可对照的表示假说,Stage 3c 以同信息量对照裁决(P2/P2-E/P3 vs P1-V)。

## 本批新增实测证据(支撑差异声明)

- 门区 new_recall:P1/P1-V **0.58–0.87(精度 1.0)** vs persistence **0.0**(`runs/stage3b/metrics/door/door_summary.csv`)——独立物体运动可预测,GeoMem-VLA 类隐式方法不提供此项测量。
- P1-W(权重移植)更差、P1-F(精确 NLL 等价分解)更差(dyn 0.09 vs P1 0.42,固定阈值)——解耦收益非 loss 数值形式可复现,机制归因最终裁决待 pareto 匹配 falseOcc(运行中)。

## Stage 4a 待核实条目(2026-11-20)

以下论文仅记录了编号,本会话尚未重读全文,**未写入主表**(NOT_VERIFIED,避免凭记忆写关系):Robot-Factored World Models (2607.22535)、MVISTA-4D (2602.09878)、SelfWAM。UNITAS (2610.12099) 已在主表。待下一轮文献核对后补行。
