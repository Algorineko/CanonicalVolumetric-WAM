# src/cvwam 公共实现

更新:2026-11-18。

- `geometry.py`:坐标链、投影/反投影、PLY。**易错点**:T_base_cam 是 cam→base 方向(相机中心=平移列 `T[:3,3]`,不是 -Rᵀt);行向量约定 `p_target = (p_source - t) @ R`。所有变换改动必须先过 `tests/test_geometry.py`(锚点 [0.2,0.032,1.35] 误差 <1e-5 m)。
- `robotwin_adapter.py`:openmw HDF5/ffv1 读取 + canonical sample(plan.md §5.3 契约)。instructions 字段是 0-d object bytes,用 `d[()].decode()`;锚点断言(head 相机/K/深度量级)内建于 `_verify_anchors`,不许绕过。
- `voxel.py`:体素语义(hit/free/observed/unknown/view_count/conflict)。**射线按 view_id 分组,每相机只连自己的观测点**;bbox slab 裁剪必需(相机在 bbox 外);pitch/2 ray marching 是文档化近似。语义改动必须过 `tests/test_voxel.py`(§7.4 表)。
- `dynamics_dataset.py`:episode 级 (t→t+k) 数据集;action=HDF5 action 组(下一帧绝对关节目标);split 按 episode 传参,禁止相邻帧测试划分。
- `models_tiny.py`:P0/P1/P1-R/P1-V/P1-W/P1-F 家族;zero-init head=持续性起点;P1-V 的 occupancy pos_weight=free/hit(≈8.15,**曾写反致结论反转,回归测试在 tests/test_stage3b.py**)。
- `align.py`:Umeyama/相机中心相似/平面拟合/NN 指标;NN 空数组返回 NaN 不崩溃。
- `viz.py`:matplotlib Agg;用 `.venv-wam` python 跑。

公共契约见根 [AGENTS.md](../../AGENTS.md)。
