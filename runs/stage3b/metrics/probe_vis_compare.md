# Observer 探针:可见性对比(同物理,head 相机 +2.5mm 抖动)

- 采样帧数: 43 (stride 10)

| 指标 | 值 |
|---|---|
| hit Jaccard (orig vs probe) | 0.9876 |
| free Jaccard | 0.9891 |
| unknown 率 orig / probe | 0.8671 / 0.8676 |
| reveal (unknown→seen) 均/帧 | 42.6 |
| occlude (seen→unknown) 均/帧 | 103.8 |
| hit 新增/丢失 均/帧 | 9.1 / 11.8 |

裁决: **VISIBILITY_CHANGED_PHYSICS_SAME** (物理逐位一致见 probe_verification.json)
