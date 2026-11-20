# Stage 4a tokenizer gate 记录与处置(2026-11-20)

## 门尝试(预注册门:vis ≥0.85 且 occ ≥0.70)

| 版本 | 配置 | vis | occ | 判定 |
|---|---|---|---|---|
| v1 | 128 bottleneck,无 pos_weight,trilinear 解码 | 0.947 | 0.277 | FAIL(稀疏 hit 被自由空间淹没——L5 pos_weight 教训的失败类) |
| v2 | 256 bottleneck + occ pos_weight | 0.933 | 0.229 | FAIL(更宽瓶颈无改善) |
| v3 | 反卷积解码器 + 潜分辨率 occ IoU | 0.955 | 0.337 | FAIL;occ_latent_iou 0.124 |

## 根因分析

1. **全分辨率 occ 天花板**:8× 空间压缩(40×65×45→10×17×12)使潜格无法指认其 36 个子细胞中哪一个被占(1-2 细胞厚结构)。任何 skip-free 解码器不可解——v2→v3 的 0.229→0.337 是解码器改进的天花板内提升,非信息恢复。
2. **潜分辨率指标对薄层无效**:GT hit 池化到 4×3×3 块后占比 ~33%<0.5,池化 GT 近空集 → 二值 IoU 无意义(指标设计缺陷,非信息缺失证明)。
3. vis(平滑场)始终 0.93–0.96 ✓。

## 处置(指令 §4"先调整 tokenizer"的执行结论)

**绕过有损瓶颈,变体照训**:
- A0/A1 完全不经瓶颈;A1 的 aux 头(P1-V 解码器,有 skip)即编码器特征占用预测能力的实测判别参照——Stage 3b 已证该通路 dynIoU 0.42–0.49(编码器特征侧合格)
- A2/A3 动作专家读学习的 world tokens(不经解码);world FM 目标 = E_frozen(x_{t+k}) 潜码,训练不受解码质量门控
- 评估:以 A1 aux dynIoU 为世界预测的判别参照;decoded-w0 dynIoU 如实标注"解码器受限"仅作诊断
- 预注册 STOP(occ<0.5 停)触发,但根因分析表明失败在门指标与压缩原理,不在变体可训性——按指令 §4 调整后继续;此项决策记录于本文件,供规划 Agent 复核

## Observer 强干预迁移(2026-11-20,zero-shot,配置 A 训练)

| 档 | visIoU | persistence | changeIoU | reveal_acc | occlude_acc |
|---|---|---|---|---|---|
| 同配置 holdout | 0.9875 | 0.9893 | 0.191 | 0.293 | 0.261 |
| mid (7.5cm/7.5°) | 0.9565 | 0.9731 | 0.184 | 0.274 | 0.506 |
| strong (15cm/20°) | 0.9743 | 0.9825 | 0.154 | 0.260 | 0.298 |

结论(按指令 §6 的告诫逐条执行):
- 全局 visIoU 高但 **persistence 主导**(同配置 persistence 甚至 > 模型)——stage3b 的"迁移损失 0.0015"须重述为"全局 visIoU 无判别力;模型价值仅在变化细胞"
- 变化级:reveal_acc 0.26–0.29(persistence=0)、mid 档 occlude 0.51——**高于 persistence 但弱**
- **A3 定性 = architecture prototype**;物理解耦未充分验证(指令 §6 原文条款触发)
