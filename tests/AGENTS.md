# tests

更新:2026-11-20。

| 文件 | 覆盖 |
| --- | --- |
| `test_geometry.py` | 投影↔反投影 roundtrip(<1e-5 m)、cam2world_gl 约定、base 系锚点、table 点 |
| `test_voxel.py` | plan.md §7.4 全表:索引/bbox 界外/墙 free 语义/双视角盒/缺视角;曾抓到 per-view 射线 bug |
| `test_stage3a.py` | episode 泄漏、action 语义(action 组非 state)、指标正确性(oracle=1、偏移惩罚)、P1-V unknown 掩码、模型前向 |
| `test_stage3b.py` | 因式分解 BCE ≡ 三分类 NLL(数值)、pos_weight 回归守卫(Bug-3)、P1-W/P1-F 结构 |
| `test_dynamic_metrics.py` | AUDIT-S7 masked dyn-IoU 公式回归:五类合成用例(Perfect/All-hit/All-free/Persistence/Inverted)手算期望,内置 pre-fix bug 版锚定新旧对照 |

运行:`/home/zhangjialiang/miniconda3/envs/openwam/bin/python tests/test_X.py`。改 src/cvwam 或 scripts 的坐标/损失/体素语义前先跑对应文件;新增机制必须带等价性或回归单测。

- **运行方式注意(2026-11-20)**:`test_geometry.py` 是独立脚本(直接 `<env python> tests/test_geometry.py`,自带 check 汇总),其 `test_project_unproject` 用手工参数化,pytest 收集会报 fixture 错误——以脚本模式结果为准;其余 test_*.py 均可用 pytest(`17 passed`)。
