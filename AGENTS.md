# CV-WAM 开发入口

更新:2026-11-18。只维护公共硬约束与任务路由;参数、操作和证据放对应目录的 AGENTS.md 与 runs/*/report.md。

## 项目一句话

Candidate A(Canonical Volumetric WAM):在 robot-base 坐标系维护 canonical 3D 体素世界状态(含可见性结构),验证"物理占用变化与可见性变化解耦"的生成式世界模型路线。研究规格是 `docs/plan.md`(只读规格,不改写);阶段产物在 `runs/stage*/`。

## 按任务阅读

| 要做的事 | 入口 | 主要目录 |
| --- | --- | --- |
| 数据导出/几何/点云/坐标系问题 | [src/cvwam](src/cvwam/AGENTS.md) | src/cvwam、scripts |
| 体素化/自由空间/可见性语义 | [src/cvwam](src/cvwam/AGENTS.md) + [tests](tests/AGENTS.md) | src/cvwam/voxel.py、tests |
| 训练 tiny predictor / 消融 / 评估 | [scripts](scripts/AGENTS.md) | scripts、runs/stage3a、runs/stage3b |
| 重放 GT / 物体门角真值 | [scripts](scripts/AGENTS.md) | scripts/replay_episode_gt.py、runs/stage3b |
| 报告与证据 | [runs](runs/AGENTS.md) | runs/stage*/report.md、metrics |
| 环境配置 | [configs](configs/AGENTS.md) | configs/smoke.yaml |
| 对外交接摘要 | docs-local | docs-local/*_summary_*.md |

## 公共契约(实测验证,勿凭记忆改写)

- 坐标链:`T_base_cam = inv(T_world_base) @ cam2world_gl @ diag(1,-1,-1)`;T_world_base=(Rz90°,[0,-0.65,0])(wxyz 四元数);head 相机 base 系锚点 [0.2,0.032,1.35];T_base_cam 方向是 **cam→base**(相机中心=平移列)。
- 数据:openmw HDF5 的 `extrinsics_matrix` 是 **cam2world_gl**;depth=ffv1 gray16le **uint16 毫米 z-depth,0=无效**,容器无帧数元数据必须解码计数断言;RGB 解码必须用 `RoboTwin/data/decode_image_bit.py`。
- `state/*_arm_joint_states` 是 **PD 驱动目标非真实 qpos**;`state/*_ee_poses` 是真实 link6 位姿(**world,wxyz**)。重放 episode(replay_gt)才有真实 qpos。
- 体素:20mm 全局 pitch;每相机射线只连**自己的**观测点;hit 后不标 free;无效深度≠free;矛盾计数,hit 优先。
- 实验纪律:新实验写新 runs/stageX 目录,禁止覆盖;失败与 bug 全部存档 failures/;未执行标 NOT_RUN;数字必须来自实测;每批结束跑对抗性数字复算(Stage 3a 的 pos_weight bug 即由此发现)。
- 环境:几何/训练 `miniconda3/envs/openwam`;仿真/重放 `envs/robottwin`;OmniVGGT/可视化 `/data/canonical-volumetric-wam/venv` 或 `PhysAdaptWAM/.venv-wam`;大文件一律 /data(根分区近满);只用 GPU 0。
