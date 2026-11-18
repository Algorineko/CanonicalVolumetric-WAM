# tests

更新:2026-11-18。

| 文件 | 覆盖 |
| --- | --- |
| `test_geometry.py` | 投影↔反投影 roundtrip(<1e-5 m)、cam2world_gl 约定、base 系锚点、table 点 |
| `test_voxel.py` | plan.md §7.4 全表:索引/bbox 界外/墙 free 语义/双视角盒/缺视角;曾抓到 per-view 射线 bug |
| `test_stage3a.py` | episode 泄漏、action 语义(action 组非 state)、指标正确性(oracle=1、偏移惩罚)、P1-V unknown 掩码、模型前向 |
| `test_stage3b.py` | 因式分解 BCE ≡ 三分类 NLL(数值)、pos_weight 回归守卫(Bug-3)、P1-W/P1-F 结构 |

运行:`/home/zhangjialiang/miniconda3/envs/openwam/bin/python tests/test_X.py`。改 src/cvwam 或 scripts 的坐标/损失/体素语义前先跑对应文件;新增机制必须带等价性或回归单测。
