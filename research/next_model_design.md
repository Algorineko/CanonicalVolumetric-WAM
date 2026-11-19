# 下一代模型设计:Observer-Factorized World-Action Flow(2026-11-19 草案,准备阶段)

状态:设计文档,未训练。依据 = Stage 2c/3a/3b 已实测的四条规律,每条标注证据回指。

## 设计依据(实测规律 → 设计约束)

| 规律 | 证据 | 对设计的约束 |
|---|---|---|
| L1 物理与可见性解耦 | 可见性翻转 4.6–6.3× 物理翻转(stage2c);P1-V 优于 P1(待 pareto v3 终裁) | 世界状态必须双头分解生成:(occupancy, visibility),unknown 不作 free 标签 |
| L2 独立物体运动可预测 | door_new_recall 0.58–0.87 vs persistence 0.0(stage3b/metrics/door/) | Geometry Expert 必须包含 object-centric 动态,不能只建模 robot-sweep |
| L3 表示:free-space 是实质信息 | P2E > P2 全预算(stage3c/report.md);匹配点对比 P2E/P3 ≥ P1-V(0.02/default),极严格预算体素独活 | 表示层需携带 free-space 结构;体素/点+free 均可行,选型待 P2E 多 seed 复测 |
| L4 观测过程独立可干预 | 同物理 2.5mm 相机偏移 → 146 cells/帧可见性翻转,物理 0 变化(probe_vis_compare) | Observation Model 必须是 geometry+camera 的独立函数,而非 dynamics 的副产品 |

## 架构:三专家 + 显式信息流

```
                ┌────────────────────────────────────────────┐
obs_t (RGB-D)──▶│ Perception: canonical geometry + p_obs 场  │
qpos_t, a_t..──▶│ (现役 voxel 管线,P2E 备选)                │
                └──────────────┬─────────────────────────────┘
                               │ z_world (canonical latent)
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
   ┌────────────────┐ ┌─────────────────┐ ┌──────────────────┐
   │ Geometry       │ │ Action Expert   │ │ Observation Model│
   │ Dynamics Expert│ │ (动作生成/策略)│ │ g(几何, camera) │
   │ z_world,a →    │ │ z_world,goal →  │ │ → 未来 visibility│
   │   未来 occupancy│ │   动作 chunk    │ │   (reveal/occlude)│
   └───────┬────────┘ └────────┬────────┘ └────────┬─────────┘
           │                   │                    │
           └───────┬───────────┴────────────┬───────┘
                   ▼                        ▼
        未来世界状态 (occ*, vis*)      渲染/重构观测 ô_{t+k}
                   └── 物理可证伪评估(dynIoU/door/falseOcc)
```

关键设计决策:

1. **Geometry Dynamics Expert**:以 canonical latent(体素或 P2E 点场)为状态,动作 chunk 条件化,输出未来 occupancy。**门/物体动态的监督来自重放 GT 的 region 场**(复用 replay_episode_gt 管线,已验证)。
2. **Action Expert**:同一 latent 条件化生成动作;训练期可用 BC 或 RL,**闭环对照必需**:current-only policy(无世界模型)/ auxiliary-only(世界模型仅作辅助损失)/ joint(共享 latent)——验证"世界模型改善动作"而非只是改善预测。
3. **Observation Model**:g(future geometry, camera config) → future visibility。**训练信号 = L4 探针管线**:同 seed 同关节路径 + `--head-cam-dis` 多相机配置重放,物理 GT 共享、visibility 目标各异——这是其他 WAM 工作没有的数据构造,也是本架构的核心可验证主张。
4. **因式分解双头本身不是主要创新**(P1-W/P1-F 已证 loss 形式不能复现收益,机制归因见 attribution_verdict.md);创新在于 **三个专家的接口与信息流**:同一生成几何对应多观测配置(L4),动作与几何通过 latent 显式交互,预测被物理指标证伪。

## Tiny Prototype 计划(待批准后执行)

- 输入:P1-V 现役管线输出 + 第二相机配置(`--head-cam-dis` 探针集,2–3 集)
- Observation Model:tiny CNN(几何 latent + camera params → visibility),~1M 参数
- 验证指标:visibility 迁移误差(probe 集:训练相机配置 → 换相机配置测试的 visIoU 保持率)
- GPU 预算:<1h
- 闭环对照(current-only/aux-only/joint):需要 policy 训练管线,超出本阶段预算,列为下一阶段提案

## 风险与开放问题

1. Observation Model 的相机外生性:部署时相机位姿已知(本体感知),不构成信息泄漏;但若未来相机随动作变化(腕部相机),camera 参数依赖动作 → Observation Model 输入包含 action-expert 的输出,信息流需要消融验证。
2. **action→geometry 通路需要放大(实测驱动)**:反事实检验显示 tiny 模型的动作条件化聚焦正确(p1 的 |Δlogit| 在 GT 翻转细胞 0.572 vs 背景 0.026)但过弱,仅 2–3.5% 的 GT 翻转细胞跨越二值边界(runs/stage3b/metrics/action_counterfactual.json)→ Geometry Expert 的动作注入需要结构性增强(如 object-centric 状态 + 动作到物体状态的直接通路),而非更多 FiLM。
3. latent 接口选择(体素 vs P2E 点场)已终审:体素(随机化鲁棒 + unknown 语义),见 stage3c/report.md。
4. 若 factorization 被 pareto 降级为训练技巧(已发生),L1 约束弱化为"双头是工程选择",架构主张重心移至 L4 的 Observation Model 独立性。
