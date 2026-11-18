# configs

更新:2026-11-18。

- `smoke.yaml`:全阶段共享配置——数据路径、episode 选择、坐标系常量、workspace bbox、体素 pitch、VGGT/OmniVGGT 路径、环境 python 路径。
- **bbox 历史**:z_min 曾为 0.74,会切掉 randomized 低桌面(0.726–0.730)→ 现为 x[0.30,1.10] y±0.65 z[0.71,1.60](40×65×45 @20mm);逐集桌高在 voxels_ep 的 manifest,勿用全局 table_z_base 做 randomized 诊断。
- 修改 bbox/pitch = 改变全部体素缓存的网格,必须重缓存并记录在当批 report;split 以 `runs/stage3b/split_manifest.json` 为准(全模型共用,勿各自造 split)。
