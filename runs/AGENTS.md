# runs 实验产物

更新:2026-11-18。

| 目录 | 内容 |
| --- | --- |
| `smoke/` | Stage 0–2b 报告+指标+图(已冻结) |
| `stage2c/` | 数据动态性验证(转移矩阵/归因/覆盖) |
| `stage3a/` | tiny predictor(含 failures/poswbug_p1_v 完整 bug 版存档) |
| `stage3b/` | 机制验证(matched-loss/Pareto/重放 GT);split_manifest.json 在此 |
| `stage3b_freeze_commit.txt` | Stage 3a 冻结 commit |

## 规则

- 每批新实验 = 新 stage 目录;**禁止覆盖已有 stage 目录内容**(失败也算结果)。
- 每个 stage:`report.md`(数字必须回指 metrics/ 里的文件)+ `metrics/` + `plots/` + `failures/`;未执行项写 NOT_RUN。
- 报告完成后跑对抗性数字复算(Stage 3a 的 pos_weight blocker 即由此发现)。
- 大文件(缓存/权重)在 /data/canonical-volumetric-wam;这里只放报告/CSV/PNG。
