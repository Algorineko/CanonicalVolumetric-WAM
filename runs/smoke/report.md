# Candidate A / Server Smoke Report

- **Date**: 2026-11-18 13:55–15:45 CST;**GPU**: 2× RTX 4090 48GB(实际全部使用 GPU 0,空闲 ~23GB;GPU 1 被他人训练占用);driver 570.211.01,CUDA 12.8
- **torch/CUDA**: 2.7.1+cu128(`miniconda3/envs/openwam`,python 3.10.22;OmniVGGT 用 `/data/canonical-volumetric-wam/venv` 薄 venv,叠加 fastwam site-packages + 本 venv 补装 sympy/scipy/cv2/matplotlib/evo,未改动任何共享环境)
- **Repos**: RoboTwin `ea8b211`(RoboTwin 2.0,浅克隆,本地未提交改动**未触碰**);官方 vggt `a288dd0f1478`(2026-05-18,经 gh-proxy.com clone → `/data/canonical-volumetric-wam/third_party/vggt`);OmniVGGT-code-upstream `c8cbfc6`(sparse checkout)
- **Data**: `/data/openmw_balanced_work`(open_microwave,clean 50 + randomized 50 eps,每 ep ~424–820 帧 @15Hz;HDF5 逐帧内外参 + ffv1 GT depth mm)。**选样**: clean ep 10/25/40 + randomized ep 0/1 × 5 timesteps(0, T/4, T/2, 3T/4, T-2)+ ep10 相邻 6 帧(t=212..217,漂移测量)= **31 样本 × 3 相机**
- **Weights**: 官方 VGGT-1B `model.pt` 5,026,874,952 B(hf-mirror.com,629s,sha256 见 `metrics/vggt_weight_sha256.txt`);OmniVGGT.safetensors 4.87GB(本地已有,`verified.json` 校验通过)
- **任务选择说明**: plan.md §5.1 优先 stack_blocks_two/pick_dual_bottles,但两者本地无 GT-depth 数据;经确认改用已有 open_microwave 数据(零新采集)。微波炉门开合属强遮挡任务,与可见性假设(H3)方向一致。

## Gates

| Gate | PASS/FAIL/NOT_RUN | Evidence | Next decision |
|---|---|---|---|
| G1 GT camera geometry | **PASS** | 15/15 单测(roundtrip 1.4e-16 m;锚点 head cam base 系 [0.2,0.032,1.35] 误差 4.16e-17 m);桌面平面 z=0.740–0.741 vs 已知 0.7401;跨视角注册簇(head↔right 0.90–1.29mm,head↔left 1.63–1.93mm 中位,max 5.42mm);`metrics/test_geometry_gt.log`、`gt_geometry_summary.json` | 进入 Stage 2/2b;Stage 3 默认 gt_depth |
| G2 VGGT reconstruction | **PASS(运行)→ RGB-only 暂缓** | 输出全部有限点(203k/609k)、可对齐、日志可复现;但深度 MAE 43–78mm(最佳单视角)、acc median 28–42mm ≈ 1.5–2 voxel@20mm,三视角融合更差(48–152mm);部署型漂移(head-anchor)帧间 1.1–5.9mm(时间一致性可)但绝对深度系统性偏差 | **RGB-only(VGGT 为唯一几何源)暂缓;GT/RGB-D 路线继续**。H4 部分证伪(超接触精度已证实;对 rollout 的影响未测) |
| G2b ray/voxel | **PASS** | 13/13 体素语义单测(wall free/hit/unknown、双视角盒、bbox max 排除、缺视角);20mm 下微波炉区域开门变化 62 hit 单元(10mm:216)→ 20mm 为保留任务相关变化的最粗 pitch;`test_voxel.log`、`voxel_stats.csv` | Stage 3 用 20mm(10mm 作细化选项) |
| G3 future predictor | NOT_RUN | 本批不启动训练(plan 限定 Stage 0–2b) | 需 G1–G2b 批准后启动 |
| G4 policy | NOT_RUN | 同上 | — |

## Geometry

| Source | Views | Points | Alignment | Depth MAE (mm) | Point acc/compl (mm) | Static drift (mm) | Peak VRAM (GiB) | Latency (s) |
|---|---:|---:|---|---:|---|---:|---:|---:|
| GT | 3 | 230,400/帧 | known(逐帧标定) | 0(基准) | 跨视角注册簇 0.90–1.93 | — | — | 0.6s/帧(反投影+统计) |
| VGGT-1B | 1 | 203,056 | oracle Sim3 | 75.0 | 28.4 / 39.5 (F10 0.19) | — | 5.18 | 0.069 |
| VGGT-1B | 1 | 203,056 | deployable(桌高定尺度) | 77.9 | 42.1 / 49.0 | **1.1–5.9(帧间,部署型对齐)** | 5.18 | 0.069 |
| VGGT-1B | 1 | 203,056 | deployable G3(VGGT depth+GT 标定) | 72.8 | 37.2 / 55.0 | — | 5.18 | 0.069 |
| VGGT-1B | 3 | 609,168 | oracle Sim3 | 131.7 | 48.1 / 27.1 | 6.9–16.8(oracle) | 5.50 | 0.139 |
| VGGT-1B | 3 | 609,168 | deployable(相机中心) | 112.5 | 97.8 / 290.6(不稳定,3/4 帧可用) | 不可测(对齐帧间不稳定) | 5.50 | 0.139 |
| OmniVGGT | 1 | 203,056 | oracle | 48.3 | 29.4 / 28.6 | — | 5.19 | 0.070 |
| OmniVGGT | 3 | 609,168 | oracle | 131.9 | 38.0 / 23.6 | — | 5.83 | 0.141 |

(上表为 4 个代表帧均值;逐帧值见 `metrics/geometry_metrics.csv` 与 `geometry_summary.json`。OmniVGGT 为衍生变体对照,非官方基准。)

**关键观察**
1. **三视角融合反而差于单视角**(48 vs 28mm acc):并排渲染(`plots/compare_*.png`)显示两腕相机的重建呈互相矛盾的倾斜面片——极端近距/掠射的腕视角破坏 VGGT 联合重建。
2. **VGGT 自估相机是主要误差源之一**:G3(VGGT depth+GT 标定)depth MAE 72.8 < world_points oracle 75.0 < deployable(world_points)77.9;OmniVGGT 同趋势(G3 66.4 vs oracle 48.3 — 该例 oracle 反而差,规模性结论需更多帧)。
3. **单目尺度失真**:单视角 VGGT 重建中桌面呈 13° 倾斜曲面(RANSAC 平面 inlier 61%),导致基于平面拟合的定尺度失效;改用"锚定相机位姿 + 绕相机中心缩放 + z 下包络对齐已知桌高"的部署型构造。
4. **时间一致性 ≠ 绝对精度**:head-anchor 部署对齐下静态桌面帧间漂移仅 1.1–5.9mm(66ms 间隔),但深度 MAE 仍 4–10cm——VGGT 几何是"低频系统性偏差 + 高频稳定"。
5. GT 数据侧:depth_valid 全帧 1.0;帧对齐机制为源 observation 列表 T+1 项同切片(`[:-1]`),已亲读 `pkl2hdf5.py` 确认;ffv1 容器无帧数元数据,已解码计数断言 == HDF5 T。

| Voxel pitch | Grid shape | Hit % | Free % | Unknown % | Preprocess time | Cache size |
|---|---|---|---:|---:|---|---|
| 20 mm(ray) | 35×61×44 | 1.20 | 8.79 | 90.02 | 0.6s/帧 | 0.03 MiB/帧(压缩) |
| 10 mm(ray) | 70×121×87 | 0.55 | 8.41 | 91.03 | 1.4s/帧 | 0.03 MiB/帧 |
| 20 mm(point-only 对照) | 35×61×44 | 1.20 | 0 | 98.80 | 0.6s | 0.02 MiB |
| 10 mm(point-only 对照) | 70×121×87 | 0.55 | 0 | 99.45 | 1.4s | 0.02 MiB |

- conflict(hit/free 矛盾,已记录、hit 优先)ray 模式 0.5–1.1%。ray 融合使 observed 活跃单元比 point-only 多 8–16×(自由空间语义的贡献)。
- 微波炉区域(x 0.75–0.95, y −0.05–0.30, z 0.85–1.00):20mm 下 t212→t422(开门)hit 26→88(变化 62 单元);10mm 45→261(216 单元)→ 两个 pitch 都保留动态变化,**推荐 20mm**。
- NOT_RUN: 5mm 局部细化(可选,未执行);VGGT 源的体素缓存(G2 判定 RGB-only 暂缓后无增量信息,后置)。

## 实际执行的命令(摘要)

```bash
# 环境:openwam=main,/data/canonical-volumetric-wam/venv=OmniVGGT 薄 venv,PhysAdaptWAM/.venv-wam=可视化
python tests/test_geometry.py                       # 15/15 PASS
python tests/test_voxel.py                          # 13/13 PASS
python scripts/export_frames.py                     # 31 样本,7s(+6 相邻帧补导)
python scripts/build_geometry.py --source gt --ply-frames 0,5,10
python scripts/smoke_vggt.py --frames <4 帧+t=106> --view-configs head head+left_wrist+right_wrist
python scripts/smoke_omnivggt.py ...                # 同帧对照(cam_drop_prob=1.0, depth_drop_prob=1.0, 零 depth/mask 占位)
python scripts/eval_geometry.py --frames <4 帧> --view-configs head head+left_wrist+right_wrist
python scripts/eval_drift.py --times 212 213 214 215 216 217 --align {deployable,oracle,head_anchor}
python scripts/build_voxels.py                      # GT,20/10mm × ray/pointonly × 31 帧
python scripts/plot_{gt_geometry,voxels,vggt_compare}.py
git clone https://gh-proxy.com/https://github.com/facebookresearch/vggt.git   # a288dd0f
HF_ENDPOINT=https://hf-mirror.com hf_hub_download facebook/VGGT-1B model.pt  # 629s
```

## Runtime & storage

- 导出+几何构建:~40s;VGGT 前向:0.07s/0.14s per config(1/3 视角);评估:~9 min(KD-tree);体素化:31 帧×2 pitch×2 模式 ≈ 78s。
- 存储:`/data/canonical-volumetric-wam`:frames 9.5M,pointclouds 85M,ply 40M,voxels 2.5M,vggt caches 214M,weights 4.7G,third_party 129M;仓库 `runs/smoke` 4.6M。根分区未受影响。

## Artifacts

- GT:逐视角+融合 PLY 与相机轴(`cache/ply/gt/`,3 代表帧);31 帧点云 npz(`cache/pointclouds/gt`)
- 可视化(`runs/smoke/plots/`):3 视角 RGB 行、GT 融合点云(高度着色+view_id)、VGGT 官方/Omni 对照并排、体素 3 正交切片 ×(20/10mm)、体素 3D
- 指标:`metrics/{gt_geometry.csv,gt_geometry_summary.json,geometry_metrics.csv,geometry_summary.json,voxel_stats.csv,static_drift.csv,test_*.log}`
- 失败与诊断:`failures/stage1_diagnostics.md`(实现 bug 5 项、NN 双峰分布伪影与度量修正、滞后/偏移假设排除记录)、体素 per-view 射线语义 bug(§7.4 单测捕获)
- manifest/config:`cache/frames/manifest.jsonl`(31 行)、`configs/smoke.yaml`、依赖与权重 sha256 记录

## Decision

- **继续:canonical point / voxel,geometry_source = gt_depth(RGB-D)**。G1/G2b 通过;G2 运行成功但精度不达标 → **RGB-only(VGGT 为唯一几何输入)暂缓**,与 plan.md §6.4"若中位对齐误差超过 1–2 voxel,RGB-only 暂缓,继续 GT/RGB-D 路线"一致。
- **Blocking issue(对 RGB-only 而言)**:深度 MAE 43–78mm、单目尺度失真(桌面 13° 倾斜)、腕视角破坏多视角融合。三视角 + GT 标定外参(G3 路线)是 VGGT 侧最有希望的后续方向,但需要解决腕视角联合重建退化(如每视角独立尺度/相机固定),属后续实验。
- **Stage 3 判断:具备条件**——建议以 `geometry_source: gt_depth`、voxel 20mm(workspace x∈[0.30,1.00], y±0.60, z∈[0.74,1.60] base 系,bounds 由 GT 点云统计确认:桌面 0.740–0.741,物体高度 ≤0.95)启动小型 predictor;P0(persistence)/P1(deterministic)先行;VGGT 冻结、不参与 Stage 3。
- **下一步最低成本实验**:用现有 31 帧+相邻帧序列在 20mm 体素上 overfit P1(单 episode),验证"非复制当前帧"的动态预测信号(plan.md §8.5 步骤 1)。
- 假设状态:H4 部分证伪(对齐后误差超过接触精度,rollout 影响未测);H1–H3 未测(需 Stage 3+)。
