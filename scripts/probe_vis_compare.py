"""Observer-law measurement (Stage 3b §4 probe): same physics, different head
camera => compare canonical voxels of the ORIGINAL replay (demo_clean_balanced_
10_replay) vs the probe (probe_camdis05). Physics GT is bit-identical, so any
difference in hit/free/unknown fields is purely OBSERVATION-PROCESS change.

Metrics per frame: hit Jaccard, free Jaccard, unknown-rate delta, reveal cells
(unknown->observed), occlude cells (observed->unknown), flip rates. Writes
runs/stage3b/metrics/probe_vis_compare.json + md summary.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/canonical-volumetric-wam/cache/voxels_ep_replay")
A = ROOT / "demo_clean_balanced_10_replay" / "20mm"
B = ROOT / "probe_ep10_camdis05" / "20mm"
OUT = Path("runs/stage3b/metrics/probe_vis_compare")

n = len(list(A.glob("frame_*.npz")))
acc = dict(hit_jac=[], free_jac=[], unk_rate_a=[], unk_rate_b=[],
           reveal=[], occlude=[], hit_gain=[], hit_loss=[])
for t in range(0, n, 10):
    da = np.load(A / f"frame_{t:07d}.npz")
    db = np.load(B / f"frame_{t:07d}.npz")
    ha, fa_, ua = da["hit"].astype(bool), da["free"].astype(bool), da["unknown"].astype(bool)
    hb, fb_, ub = db["hit"].astype(bool), db["free"].astype(bool), db["unknown"].astype(bool)
    def jac(x, y):
        u = (x | y).sum()
        return float((x & y).sum() / u) if u else 1.0
    acc["hit_jac"].append(jac(ha, hb))
    acc["free_jac"].append(jac(fa_, fb_))
    acc["unk_rate_a"].append(float(ua.mean()))
    acc["unk_rate_b"].append(float(ub.mean()))
    acc["reveal"].append(int((ua & ~ub).sum()))       # unknown in orig, seen in probe
    acc["occlude"].append(int((~ua & ub).sum()))      # observed in orig, unknown in probe
    acc["hit_gain"].append(int((hb & ~ha).sum()))
    acc["hit_loss"].append(int((ha & ~hb).sum()))

res = {k: (float(np.mean(v)) if k.startswith(("hit", "free", "unk")) else float(np.mean(v)))
       for k, v in acc.items()}
res["frames_sampled"] = len(acc["hit_jac"])
res["verdict"] = ("VISIBILITY_CHANGED_PHYSICS_SAME"
                  if res["free_jac"] < 0.95 and res["hit_jac"] > 0.9 else "CHECK")
OUT.with_suffix(".json").write_text(json.dumps(res, indent=2))
md = ["# Observer 探针:可见性对比(同物理,head 相机 +2.5mm 抖动)", "",
      f"- 采样帧数: {res['frames_sampled']} (stride 10)", "",
      "| 指标 | 值 |", "|---|---|",
      f"| hit Jaccard (orig vs probe) | {res['hit_jac']:.4f} |",
      f"| free Jaccard | {res['free_jac']:.4f} |",
      f"| unknown 率 orig / probe | {res['unk_rate_a']:.4f} / {res['unk_rate_b']:.4f} |",
      f"| reveal (unknown→seen) 均/帧 | {res['reveal']:.1f} |",
      f"| occlude (seen→unknown) 均/帧 | {res['occlude']:.1f} |",
      f"| hit 新增/丢失 均/帧 | {res['hit_gain']:.1f} / {res['hit_loss']:.1f} |",
      "", f"裁决: **{res['verdict']}** (物理逐位一致见 probe_verification.json)",
      ""]
OUT.with_suffix(".md").write_text("\n".join(md))
print(json.dumps(res, indent=2))
