# CV-WAM Novelty Matrix

更新:2026-11-19(补录用户提供的 3 篇新文献)。核心差异声明:**Visibility-Decoupled Canonical 3D World Model** —— 物理占用与观测过程在生成目标层面分离(因式分解监督,unknown 不作 free-space 标签),未来 3D 世界状态被独立物理指标证伪(dynIoU / door-region / falseOcc Pareto / action 反事实)。

| 论文 | 3D 表示 | 可见性解耦 | 物理可证伪世界评估 | 与本工作关系 |
|---|---|---|---|---|
| GeoMem-VLA (2610.29866, ICLR27 投稿) | 隐式 GeoPlan tokens(point map 蒸馏)+时序记忆;**明确放弃完整可渲染重建** | ✗ 无分离 | ✗ 不评估内化几何的物理正确性(只有 policy 成功率) | **最直接竞品**。差异:①显式可测量占用场 vs 隐式 token;②p_obs/p_occ 因式分解监督(Stage 2c:可见性翻转 4.6–6.3× 物理翻转;桌面物理=0 可见性 1.3%);③action 反事实/门角基线。teacher 用法(训练期蒸馏)可借鉴,不构成差异点 |
| FoundObj (2605.27178, ICML26) | 点云超点分割(FM 先验作 RL 奖励) | ✗ | ✗ | 边缘:物体区域 GT 工具候选;非世界模型 |
| FastVGGT (2509.02560, ICLR26) | VGGT 加速(token 合并,4×,长序列) | ✗ | ✗ | 工程使能:若未来 WAM 需要 VGGT 在线前端,token 合并可用;不解决我们实测的精度阻塞(MAE 43–78mm) |
| PointWAM (2610.02840) | 点轨迹 | ✗ | 部分(点跟踪误差) | 已在 plan.md §2;无可见性分离 |
| OccWAM (2610.17688) | 自动驾驶 occupancy+ego FM | ✗(占用单一状态) | ✗ | 已在 plan.md §2;强调操作尺度差异 |
| Flex-π (2608.10860) | view-indexed pointmap | ✗ | ✗ | view-indexed 基线 |
| VGGT-World (2603.12655) | 冻结 VGGT 特征 + latent 生成 | ✗ | ✗ | teacher 蒸馏先例(与我们用法同,但无解耦) |

## 撞题自查(2026-11-19)

- Q: GeoMem-VLA 是否已显式建模 unknown/free?A: 否——tokens 隐式,无占用/可见性语义,无 unknown 概念。
- Q: 是否已有人做"可见性解耦"生成?A: 检索范围内无(PointWAM/MVG-WAM/WSA1/VGGT-World/OccWAM/GeoMem-VLA 均未分离物理与观测过程)。
- Q: 与 GeoMem-VLA 只差数据格式?A: 否——三点模型层面差异(显式可测量世界状态 / 可见性因式分解监督 / 物理可证伪评估),且其"放弃重建"哲学与我们"保留可证伪重建"互为对立假设,天然构成对照实验。
