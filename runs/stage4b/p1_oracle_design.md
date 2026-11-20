# P1 Oracle 重设计(规划裁决 §4)——设计文档,启动需用户授权预算(建议 6 GPU-h)

状态:设计定稿 2026-11-20 20:0x,NOT_RUN。前置:P0 Failure Attribution(E 节)结论——归因决定 Oracle-C 的解释框架。

## 0. 目的

回答裁决 §10.3/.4:"Oracle-C 能否恢复动作控制?""历史静态融合是否已够?"——并把 roadmap 原 P1(A-oracle 未来上界)按泄漏纪律重构。

## 1. 三层 Oracle 定义

### Oracle-C(当前观测恢复,直接可执行,复用归因资产)

同一模拟器物理状态下(确定性重放,episode 10 / seed 1000019 已有 nominal+shifted 两套渲染):

| 条件 | 策略输入 | 回答 |
|---|---|---|
| C0 shifted(已测) | 偏移相机在线体素 | 基线(closed-loop 0/x) |
| C1 zero(已测,cam_t030blind) | 全零体素 | 盲先验回放率 |
| **C2 nominal-render** | **同状态 nominal 相机渲染的体素**(离线成对帧,仿真状态=偏移 episode 的物理态) | **若 C2 恢复成功率 ⇒ 感知信息缺失是主因;若不恢复 ⇒ 输入分布/部署问题** |
| C3 nominal+shifted fusion | 解析 unknown-aware 融合(§2 规则) | 历史/多视角融合的解析上界(§10.4 前半) |

执行方式:C2/C3 无法在闭环内"换相机"(相机已物理移动)——采用**离线动作回放评估**:对 paired 帧序列逐帧取 oracle 体素 → 策略动作 → 与 expert 轨迹比对(开环一致性),并用重放闭环(replay-based closed-loop:动作序列在重放环境中执行,状态由重放物理推进,与 expert 分支时如实报告)两种口径。**口径差异必须分开报告**(裁决 §4:排除泄漏)。

### Oracle-B(历史信息三层,回应 §10.4)

| 层 | 输入构造 | 说明 |
|---|---|---|
| B0 | 每帧重建(=A0 现状) | 已有(cam_t030 0/x) |
| B1 | history aggregation:过去 k 帧体素的 observed-union 命中取并、free 取交集、**unknown 保持 unknown**(不在任何帧 observed 的格) | 纯解析,无学习 |
| B2 | 解析 unknown-aware fusion:B1 + 每格 confidence/age 通道(见 §3) | 纯解析,B4 的直接下界 |

判读:**若 B2(解析融合)在 cam_t030/cam_switch 条件下恢复 A0 成功率,则"简单基线解决问题"成立——按裁决 §10 如实调整核心机制**(生成式信念传播不立为主贡献,降级为 B1/B2 的改进项)。

### Oracle-F(未来几何诊断,特权信息,不入闭环)

- 离线:expert 重放的后续帧几何(已有 replay GT 管线)作为"真实未来"喂动作流,量化 oracle-future 条件下动作对 GT 的改善幅度(§7 干预语义:"改善"而非"改变")。
- **禁止**:把独立专家轨迹按时间索引注入实际闭环(泄漏专家动作)。真在线 oracle 只能从当前 sim state 分支预测或专家重规划(成本高,列为可选,默认不做)。

## 2. 解析 unknown-aware fusion 规则(B1/B2/C3 共用,写死防事后调整)

- observed_any[t-k..t] = 任一帧 observed ⇒ 该格**当前观测证据**;
- hit:任一帧 hit ⇒ hit(最新优先);无 hit 但有 observed ⇒ free(需 ≥1 帧 free 支持);任何帧都未 observed ⇒ **unknown(不标 free,不沿用旧占用)**;
- 动态物体的 stale/ghost:hit/free 记录 age(距最新证据帧数);age > τ(建议 2s≈100 步)且当前帧该格被 updated-observed 为 free ⇒ 降级;被 observed 为 hit ⇒ 覆盖。**B1 不带 age 通道(纯并/交),B2 带 age+confidence(该格 hit 帧数/observed 帧数)。**

## 3. 训练侧改动(B2 通道进入网络时的接口)

现输入 4 通道 [hit, free, observed, view_count/3]。B2 融合画布改 6 通道(+age_norm, +confidence)——a0 的 enc 首层 conv 需 in_ch=6(微改 + 重训 10min 级);B1 可直接用现 4 通道(并/交后在通道语义内表达)。**先跑 B1(零训练),再决定 B2 是否值得 6ch 重训。**

## 4. 执行顺序与预算(待授权)

1. B1 解析融合 + paired 帧离线动作一致性(归因资产复用,~0.5 GPU-h)
2. C2/C3 离线动作一致性 + 重放闭环成功率(~1 GPU-h)
3. B1/B2 在 cam_t030 条件的闭环(replay-based,~1.5 GPU-h)
4. Oracle-F 离线诊断(~0.5 GPU-h)
合计 ~3.5 GPU-h(6 上限内)。**闭环真机式评测(真 sim 分支)不在本预算内。**

## 5. 判读矩阵(预注册)

| 结果 | 行动 |
|---|---|
| C2 恢复 + B1/B2 恢复 | 感知缺失主因;解析历史融合足够 → 核心机制转向"信念维护的表征与策略消费"(§10.4=是),生成式部分降级 |
| C2 恢复 + B1/B2 不恢复 | 当前帧 nominal 即够、历史无用 → 主张收缩为"鲁棒感知对齐"(弱,考虑路线 B 复核) |
| C2 不恢复 | 输入分布/部署问题 → 回归 G-Integrity/归因,不上 P1 主实验 |
| Oracle-F 改善动作 + B2 不够 | 生成式信念传播有独立价值 → B4 立项(P3),需 §5 新条件佐证 |
