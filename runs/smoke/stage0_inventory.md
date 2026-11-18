## Stage 0 系统盘点 (CV-WAM smoke)
- Date: 2026-11-18 14:14:33

### 基本
```
/home/zhangjialiang/code/CanonicalVolumetric-WAM
zhangjialiang
/dev/nvme0n1p2  1.8T  1.7T   14G 100% /
/dev/nvme1n1p1  1.8T  1.6T  153G  92% /data
conda binary not on PATH; envs under /home/zhangjialiang/miniconda3
```

### GPU (nvidia-smi)
```
index, name, memory.total [MiB], memory.free [MiB], driver_version
0, NVIDIA GeForce RTX 4090, 49140 MiB, 22956 MiB, 570.211.01
1, NVIDIA GeForce RTX 4090, 49140 MiB, 12787 MiB, 570.211.01
```

### PyTorch (openwam env)
```
torch: 2.7.1+cu128
CUDA build: 12.8
CUDA available: True
GPU count: 2
GPU 0: NVIDIA GeForce RTX 4090
BF16: True
```

### Repos / commits
```
== RoboTwin ==
ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
 M env_cfg/task_config/demo_clean.yml
 M env_cfg/task_config/demo_randomized.yml
 M envs/_base_task.py
 M envs/utils/pkl2hdf5.py
?? env_cfg/task_config/demo_clean_balanced.yml
?? env_cfg/task_config/demo_clean_smoke.yml
?? env_cfg/task_config/demo_randomized_balanced.yml
?? env_cfg/task_config/demo_randomized_v21.yml
== OmniVGGT-code-upstream ==
c8cbfc6bcd57a07bf08a1f97804da9827847af8c
== SemWAM ==
3815ae13bfe2d225f17b00877ede67412fc9f400
?? experiments/qualitative/analyze_libero_noise.py
?? experiments/qualitative/analyze_task_structure.py
?? experiments/qualitative/cache_libero_text.py
?? experiments/qualitative/capture_libero_fog10.py
?? experiments/qualitative/capture_libero_noise.py
?? experiments/qualitative/fetch_libero_bundles.py
?? experiments/qualitative/fetch_libero_fog10_demos.py
?? experiments/qualitative/libero_noise_models.py
== CanonicalVolumetric-WAM (new) ==
not a git repo yet
```

### Data & weights
```
2.1G	/data/openmw_balanced_work/demo_clean_balanced
4.1G	/data/openmw_balanced_work/demo_randomized_balanced
50
-rw-rw-r-- 1 zhangjialiang zhangjialiang 4870150472 Nov 12 21:45 /data/physadapt_models/omnivggt/OmniVGGT.safetensors
```
