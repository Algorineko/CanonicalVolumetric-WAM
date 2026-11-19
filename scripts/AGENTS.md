# scripts 操作脚本

更新:2026-11-19。

## 入口地图

| 脚本 | 作用 | 环境 |
| --- | --- | --- |
| `export_frames.py` | episode→canonical 样本(RGB-D+外参+qpos/action) | openwam |
| `build_geometry.py` | GT 点云+PLY+跨视角一致性 | openwam |
| `smoke_vggt.py` / `smoke_omnivggt.py` | VGGT 重建 smoke(已暂缓) | openwam / venv-wam |
| `eval_geometry.py` / `eval_drift.py` | VGGT 对齐与指标(暂缓路线留档) | openwam |
| `cache_episode_voxels.py` | episode 级体素缓存(训练数据) | openwam |
| `build_robot_region.py` | FK+碰撞球机器人区域(可被 door_eval 复用) | openwam |
| `stage2c_transitions.py` / `stage2c_vis_attribution.py` / `stage2c_workspace_check.py` | Stage 2c 数据动态性验证 | openwam |
| `build_dropout_voxels.py` | 相机子集输入体素 | openwam |
| `train_tiny.py` / `eval_tiny.py` / `aggregate_stage3a.py` | tiny predictor 训练/评估/聚合 | openwam + GPU |
| `pareto_eval.py` | §5 Pareto 评估(v2 流式三趟:阈值→val→test;9 模型增量 CSV) | openwam + GPU |
| `door_eval.py` | §1.3 门/机器人/物体级评估(replay 427 帧;FK 用 qpos_full dims [6,14,18,22,26,30]+grip34) | openwam + GPU |
| `plot_voxels_ep.py` | canonical 体素可视化(overview/slices/dynamics,metric 空间) | venv-wam |
| `replay_episode_gt.py` | 确定性重放+物体 GT;`--head-cam-dis` observer 探针(**robottwin env**) | robottwin |
| `verify_replay_alignment.py` / `build_replay_geometry.py` | 重放对齐裁决 / 重放几何+门区域 | openwam |
| `verify_probe_physics.py` | observer 探针裁决(物理不变+相机已动) | openwam |
| `stage3c_point_cache.py` | Stage 3c 表示缓存(3 视角 depth+参数;点云在线构建) | openwam |
| `train_repr.py` | Stage 3c P2/P2E/P3 训练(P1-V 同款解耦监督) | openwam + GPU |
| `run_stage3a.sh` | Stage 3a 全序列 | — |
| `plot_*.py` | 可视化 | venv-wam |

## 规则

- 输出写 runs/stageX 或 /data/canonical-volumetric-wam;**临时输出/日志/权重不入源码**。
- 训练/评估单卡执行:按 /code/AGENTS.md 优先 1 号卡,被占则选空闲显存最大的卡(启动前 nvidia-smi 复核)。
- 新脚本先读根 [AGENTS.md](../AGENTS.md) 公共契约;坐标/体素语义相关改动先跑对应单测。
- 破坏性命令(覆盖已有 runs)禁止;失败保留并写 failures/。
- pareto_eval/door_eval 支持断点续跑(CSV 存在即跳过);长任务用 setsid nohup + 后台日志。
