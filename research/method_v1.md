# Method v1: Observer-Factorized Joint World-Action Model(CV-WAM Stage 4a)

> 草案,随 4a 结果更新。主张-证据对应见 claim_evidence_matrix.md。

## 1. 问题设定

操作任务中的世界—动作联合生成:给定当前 canonical 体素场 x_t(hit/free/observed/unknown 四通道,20mm,robot-base 系)、本体感知 qpos、语言指令,模型 (i) 生成未来动作块 a_{t:t+16}(14 维绝对关节目标,FM 采样)与 (ii) 生成未来几何信念场(布局区域可观测性显式建模)。部署时 receding-horizon 执行(chunk 16,exec horizon 4)。

## 2. 三专家与信息流(硬门控接口)

- **Geometry/World Expert**:对世界潜token流做 FM 去噪,输出未来几何 belief(observed-occupancy 语义,unknown 不作 free)。
- **Action Expert**:对动作 token 流做 FM 去噪;通过 cross-attention 读取 World Expert 的中间 token(硬门控:不读当前几何——归因设计),qpos 作为第 17 token。
- **Observation Model(Observer)**:由生成的几何特征 + 未来相机参数(本体可得的配置量,非场景观测)预测未来可见性场;仅辅助监督,不进动作流。
- **A3-mask 变体**:所有 cross-attn 路径加可学习 sigmoid 门(γ 初始化:已有路径 0,新增 action←current 路径 −4 近闭合),最终 γ 值作为发现报告——回答"专家路由硬门控是否承重"(用户指定)。

## 3. 变体阶梯(匹配对照)

| 变体 | 几何来源 | 世界流 | Observer | 隔离的假设 |
|---|---|---|---|---|
| A0 | 当前编码 | 无 | 无 | 基线动作 FM |
| A1 | 当前编码 | 辅助预测头(不生成) | 无 | 辅助世界监督的感知塑造价值(ReconVLA 先例) |
| A2 | 当前编码 | 联合 FM 生成 | 无 | 联合生成的额外价值 |
| A3 | 当前编码 | 双流 + 双向 cross-attn | 有(辅助) | Observer-Factorized 结构价值 |
| A3-mask | 同 A3 + 可学习门 | 同 A3 | 同 A3 | 硬路由是否承重 |

匹配:编码器(AE-init)、动作专家容量、FM 配方(pi0 式)、数据 split、训练步数、评估 seed、Euler 步数、exec horizon。

## 4. 关键设计事实(实测依据)

- 编码器特征→占用预测的判别通路已证(dynIoU 0.42–0.49,P1-V 家族);tokenizer 有损瓶颈(8× 压缩不能表达薄层结构,三次门失败存档 runs/stage4a/gate_note.md)→ 世界 FM 目标用 E_frozen 潜码,世界侧评估以 A1 aux 为判别参照
- 动作条件化在 logit 层聚焦正确但过弱(stage3b 反事实检验)→ 动作专家用 cross-attn 读世界流(FiLM 被证不足)
- Observer 跨相机迁移已验证(visIoU 0.9859/0.9874)
- 在线体素构建 GPU 版 10.4ms(逐位等价 numpy)→ 闭环 replan 可行

## 5. 评估协议

离线:变体×seed 诊断(动作 x0-MSE、世界 masked dynIoU、aux dynIoU、observer visIoU + GT-geometry 上界)。闭环:XPolicyLab 适配器,同 seed 集,5 smoke → Clean 20/Randomized 20;task success(门 ≥60%)/final door angle/失败阶段/门区动态/延迟/显存。主比较 A1 vs A0 / A2 vs A1 / A3 vs A2 / A3-mask vs A3(小样本仅筛选)。
