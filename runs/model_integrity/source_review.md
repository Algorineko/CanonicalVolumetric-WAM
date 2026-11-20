# Source Review 判定表(规划审查 S1-S7,服务器本地核实 2026-11-20 20:3x)

审查来源: docs-local/review/review.md(公开 main 静态审查)。本地仓库与公开 main 同源;
本地有未提交修改(见 hash 表)。逐项判定: CONFIRMED(本地复现)/FIXED/NOT_REPRODUCED/NOT_RUN。

| 项 | 内容 | 判定 | 此前是否发现 | 佐证 |
|---|---|---|---|---|
| S1 | a3 world/action 流零依赖当前体素 | CONFIRMED | 是(追加 7) | tests/test_model_conditioning.py::test_s1_a3_no_current_conditioning |
| S2 | Observer 从未训练 | CONFIRMED | 是(追加 7) | 同上 observer 用例 |
| S3 | WorldTokenizer 无 3D 位置编码 | CONFIRMED | **否(遗漏)** | 同上 spatial 用例 |
| S4 | a2/a3 动作流无 t_act | CONFIRMED | 是(追加 7) | 同上 time 用例 |
| S5.1 | W0/W1 世界噪声重抽 | CONFIRMED | 是(my-wam) | 代码 world_intervention.py euler_chunk 内 randn |
| S5.2 | a2 W1 空操作 | CONFIRMED | 是(追加 7) | forward 无 block 引用 |
| S5.3 | W3 donor=batch roll 同 episode | CONFIRMED | **否(遗漏)** | world_intervention.py:151 |
| S5.4 | L9 FK 吃归一化动作 | CONFIRMED | **否(遗漏)** | world_intervention.py:171-182 |
| S6 | best.pt=final(best_val 从未更新) | CONFIRMED | **否(遗漏)** | train_stage4a.py:317/418-420 |
| S7 | dyn IoU inter 缺 gt_hit | CONFIRMED | **否(遗漏)** | train_stage4a.py:65/129 + eval_stage4a.py:143(a1 aux 同病,自查) |

自查新增: a1 aux 指标同病;归因 door 盒近似未验证;G-Integrity sampler-parity 未执行。
汇总: 10 项 CONFIRMED / 0 NOT_REPRODUCED / 修复状态见 test_results.json。

## RoboTwin 外部文件 hash(G-Integrity 记录)
见 runs/stage4b/metrics/gintegrity_baseline.json(model.py 91ec3ff7 / eval_policy 7b150849 / collect_data 0b940529;后续修改会变,以 json 为准)。
