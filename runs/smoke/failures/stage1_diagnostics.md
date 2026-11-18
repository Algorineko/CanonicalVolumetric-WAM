# Stage 1 诊断记录(2026-11-18)

## 修复过的实现 bug(单测/断言捕获)

1. `project_base_to_px` 行向量约定平移符号错(`+t` → `(p-t)@R`);z 回算单测同步修正。
2. `camera_center_in_base` 公式错:`T_base_cam` 是 cam→base 方向,相机中心 = 平移列 `T[:3,3]`;原 `-Rᵀt` 是 base 原点在相机系。锚点断言 [0.2,0.032,1.35] 误差从 1.04m → 4.16e-17m。
3. `write_ply` 1 维 extra 数组拼接失败;测试断言对 `%.3f` 格式错误。
4. `export_frames.py` 把 `T_world_base` 传给了需要 `T_base_world` 的形参(方向颠倒,锚点断言抓到 err 0.7565m)。
5. h5py 0-d object 字段:`asstr()` 返回惰性视图,json 需 `d[()].decode()`。

## NN 一致性度量的伪影与修正(重要)

- 初版用 `<50mm 掩码中位数`:left↔head 出现 10–45mm 的"误差",head↔right 却 1mm。
- 排查过程(全部存档于本目录脚本日志):
  - 帧滞后假设:**排除**(深度 t vs 外参 t-k,k=0..8,误差不变;臂速 0.1–1.7mm/帧的静止时刻也 ~42mm)。
  - bbox 裁剪影响:无 crop 42mm / 有 crop 42.9mm,基本不变,排除。
  - 恒定相机系安装偏移假设:**排除**(t=0 与 t=136 紧邻带 <30mm 内偏移均值 [-0.5,-1.0,0.1] / [0.2,-1.6,-3.5]mm,std 4.5–7.3mm,无聚簇)。
- **真相:NN 距离分布是双峰** —— 真重叠表面呈 <2mm 注册尖峰(量化水平),单侧可见表面呈 20–200mm 宽尾(遮挡/近距/掠射导致头相机不可见)。50mm 掩码的中位数落在哪个峰取决于两种点的混合比,因此不是配准误差。
- 佐证:左腕 t=136 时 66% 点距头相机最近点 >20mm,但其图像显示左腕相机距表面仅 5–27cm(z_cam 中位 6cm),观察的是头相机被手爪遮挡/视角外的把手区域。
- **修正后度量**:注册簇(`<10mm`)中位数 + `frac<2mm`(2mm ≈ 0.7m 处像素脚印)。最终值见 gt_geometry_summary.json:head↔right 0.90/1.29mm,head↔left 1.63/1.93mm(max 5.42mm),left↔right 有效重叠 ≈0(相机安装位置决定,如实记录为 NOT_OVERLAPPING)。

## 其他发现

- depth_valid 全帧 1.0(SAPIEN 深度无无效像素,alpha 混合后 0=无效在这批数据未出现)。
- 采集中源 observation 列表为 T+1 项,state=[:-1]、action=[1:]、colors/K/外参/depth 全部 [:-1] 同切片 → 帧 t ↔ ffv1 帧 t ↔ state[t] 严格对齐(pkl2hdf5.py:116-224 亲读确认)。
- ffv1 mkv 容器帧数元数据为 0,必须解码计数;已断言 == HDF5 T。

## Stage 2b 体素语义 bug(单测捕获)

- 初版 `voxelize_rays` 把**所有相机的点**作为每个相机的射线终点(射线穿过盒体内部打到别的相机的表面点,把 unknown 内部标成 free)。§7.4 "双视角静态盒"单测当场暴露:interior free=True。修复为按 view_id 分组、每相机只连自己的观测点;冲突计数与 hit-wins 语义同时厘清。
- 另修:逐轴 DDA(不沿真实直线)→ 换成 bbox slab 裁剪 + pitch/2 ray marching(轴对齐情形下精确,文档注明近似);`grid_stats` 缺 shape 键;相机在 bbox 外(x=0.2<0.30)必须裁剪后才遍历。
