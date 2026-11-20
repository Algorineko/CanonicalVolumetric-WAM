# P3 Belief Mechanism B0-B4 阶梯(规划裁决 §8)——设计文档,NOT_RUN

定稿 2026-11-20 20:0x。前置事实(追加 7 计算图 probe):plain a3 的世界流**不读当前体素**;a3mask 的 act←current 门三种子都停在 ≈0.019(近关闭);v_a 读的是 noisy_world 的 token 编码而非生成的 clean future。**任何"信念进入动作"的设计必须显式打通 当前体素/历史 → future 表征 → 动作流 三段,否则复刻 L9 的空通路。**

## 0. 候选定义(§8 原文 + 实现化)

| ID | 名称 | 实现 | 训练量 |
|---|---|---|---|
| B0 | Per-frame canonical voxel | =A0 现状(每帧重建) | 已有 |
| B1 | History aggregation | 解析并/交流水线(p1_oracle_design.md §2 规则),4 通道语义不变 | 零训练 |
| B2 | Analytic unknown-aware fusion | B1 + age/confidence 通道(6ch,enc 首层 in_ch=6) | 重训 a0 ~10min |
| B3 | Camera-augmented current-only | 当前体素 + 相机位姿扰动增广训练(数据侧:训练时随机加 cam shift/rot,让策略学会对齐) | 重训 a0(增广)~10min |
| B4 | Predictive Belief WAM | 学习式信念传播:enc 输入 B2 风格 6ch 融合画布 + 上一帧 belief latent;world stream 预测未来 belief;**动作流 cross-attn 未池化体素 patch(P2 候选 1,带 3D 位置编码)** | 新架构,~1-2h |

## 1. 语义纪律(§8,写死)

- unknown ≠ free:任何帧都未 observed 的格保持 unknown,不参与 free 通道;
- 不无条件保留旧占用:hit/free 记录 age;age > τ(2s) 且当前观测冲突 ⇒ 按当前观测覆盖;
- 通道:occupancy belief、confidence(hit 帧数/observed 帧数)、age_norm、当前观测证据(current-observed mask);
- 动态 stale/ghost:open_microwave 的门是运动部件——door 区域的 occupancy 必须按当前帧证据优先(§5 transient_occ 条件专测)。

## 2. 评测协议(预注册)

- 主条件:cam_t030(持续偏移)、cam_switch(中途偏移)、transient_occ(暂遮挡)——全部满足"目标在历史中曾可见"(§5);
- 每条件 20-trial,种子配对(冻结集);每条件先盲对照(G-Integrity);
- 次要:clean(不应回退)、drop1of3(阴性对照,不应显著变化);
- 统计:Wilson CI + 配对 bootstrap(B4 vs B1/B2/B3 差);
- **B4 立主贡献的门:B4 在 cam_switch 与 transient_occ 上同时优于 B1/B2/B3(两条件配对 bootstrap 95% CI 不跨 0)**;只赢 cam_t030(持续偏移,B1 即可解)不算。

## 3. 与竞品区分(§3 roadmap)

- AtlasVLA/Recursive Belief VLA:隐式 belief token,无显式体积 unknown 语义;
- SCOUT:遮挡下的物体级 rollout 规划,无逐格 belief/age/confidence;
- 本阶梯的差异化 = **显式体积 unknown + age/confidence 的解析-生成混合**:B2 是解析下界,B4 若胜出,增益归因于"预测式传播补足解析融合的死角"(动态物体/从未见区域),由 §10.6 的场景存在性证明。

## 4. 风险

- B2(解析)可能已够 → B4 不立项,主贡献改为"B 阶梯审计 + unknown-aware 融合"(按裁决如实降级);
- B3(相机增广)可能独立解决 cam_t030 → 削弱"信念"必要性,归因先行(E 节)是守门人;
- 训练分布:全部 B 候选只在 clean 10 集上训——跨条件增益必须来自机制而非数据(与 A0 同训集,公平)。
