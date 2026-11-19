"""Stage 3b §1.2: mechanism-attribution verdict from pareto CSVs.

Answers the four review questions with numbers:
  1) Does P1-V beat P1 at matched falseOcc?  2) Do P1-W / P1-F reproduce it?
  3) Stable across seeds?                    4) Same direction Clean vs Rand?
Writes runs/stage3b/metrics/attribution_verdict.md (+ .csv table).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path("runs/stage3b/metrics/pareto")
OUT_MD = Path("runs/stage3b/metrics/attribution_verdict.md")
BUDGETS = ["0.002", "0.005", "0.01", "0.02"]
GROUPS = {"p1": ["p1_s42", "p1_s43", "p1_s44"],
          "p1_v": ["p1_v_s42", "p1_v_s43", "p1_v_s44"],
          "p1_w": ["p1_w_s42"], "p1_f": ["p1_f_s42"], "p0": ["p0"]}


def load():
    rows = []
    for f in sorted(P.glob("pareto_*.csv")):
        model = f.stem.replace("pareto_", "")
        for r in csv.DictReader(open(f)):
            r.setdefault("model", model)
            rows.append(r)
    return rows


def agg(rows, model, budget, split, group, metric="dyn_iou"):
    """mean over (episodes, offsets) of episode-mean values, for one seed."""
    vals = [float(r[metric]) for r in rows
            if r["model"] == model and r["budget"] == budget and
            r["split"] == split and r["test_group"] == group and r[metric] != ""]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def seed_stats(rows, seeds, budget, split, group, metric="dyn_iou"):
    vals = [agg(rows, s, budget, split, group, metric)[0] for s in seeds]
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0), len(vals)


def main():
    rows = load()
    lines = ["# Stage 3b §1.2 机制归因裁决(数据源: pareto_p*.csv,匹配 falseOcc 预算)",
             ""]
    table = []
    for split, group in (("test", "clean_test"), ("test", "rand_test")):
        lines.append(f"## split={split} group={group}")
        lines.append("")
        lines.append("| model | seeds | " +
                     " | ".join(f"dynIoU@{b}" for b in BUDGETS) +
                     " | " + " | ".join(f"chgF1@{b}" for b in BUDGETS) + " |")
        lines.append("|---|---|" + "---|" * (2 * len(BUDGETS)))
        for fam, seeds in GROUPS.items():
            cells, cells2 = [], []
            for b in BUDGETS:
                m, s, n = seed_stats(rows, seeds, b, split, group, "dyn_iou")
                cells.append(f"{m:.4f}±{s:.4f}" if n > 1 else f"{m:.4f}")
                m2, s2, _ = seed_stats(rows, seeds, b, split, group, "chg_f1")
                cells2.append(f"{m2:.4f}±{s2:.4f}" if n > 1 else f"{m2:.4f}")
                table.append(dict(split=group, family=fam, budget=b,
                                  dyn_iou_mean=m, dyn_iou_std=s,
                                  chg_f1_mean=m2, n_seeds=len(seeds)))
            lines.append(f"| {fam} | {len(seeds)} | " + " | ".join(cells) +
                         " | " + " | ".join(cells2) + " |")
        lines.append("")

    # verdict rules
    def fam_mean(fam, b, group, metric="dyn_iou"):
        seeds = GROUPS[fam]
        m, _, n = seed_stats(rows, seeds, b, "test", group, metric)
        return m

    verdicts = {}
    for group in ("clean_test", "rand_test"):
        d = {}
        for b in BUDGETS:
            d[b] = dict(
                p1v_vs_p1=fam_mean("p1_v", b, group) - fam_mean("p1", b, group),
                p1w_vs_p1v=fam_mean("p1_w", b, group) - fam_mean("p1_v", b, group),
                p1f_vs_p1v=fam_mean("p1_f", b, group) - fam_mean("p1_v", b, group))
        verdicts[group] = d
    lines.append("## 裁决(Δ = 相对 P1-V 的 dynIoU 差)")
    lines.append("")
    for group, d in verdicts.items():
        lines.append(f"### {group}")
        lines.append("| budget | P1-V − P1 | P1-W − P1-V | P1-F − P1-V |")
        lines.append("|---|---|---|---|")
        for b in BUDGETS:
            lines.append(f"| {b} | {d[b]['p1v_vs_p1']:+.4f} | "
                         f"{d[b]['p1w_vs_p1v']:+.4f} | {d[b]['p1f_vs_p1v']:+.4f} |")
        lines.append("")

    # automatic rule
    bmid = "0.01"
    gains = [verdicts[g][bmid]["p1v_vs_p1"] for g in verdicts]
    repro_w = [verdicts[g][bmid]["p1w_vs_p1v"] for g in verdicts]
    repro_f = [verdicts[g][bmid]["p1f_vs_p1v"] for g in verdicts]
    gate = min(gains)
    if gate > 0.02 and max(repro_w) < 0.01 and max(repro_f) < 0.01:
        rule = ("P1-V 在匹配预算下稳定优于 P1,且 P1-W/P1-F 不能复现 → "
                "factorization+masking 保留为机制主张(进入结构化设计)")
    elif gate > 0.02:
        rule = ("P1-V 优于 P1 但 matched-loss 变体部分复现 → "
                "factorization 降级为训练技巧,转入结构化 geometry-observer 设计")
    else:
        rule = "P1-V 无稳定优势 → 因式分解作为训练技巧;范式裁决交给 Stage 3c 表示对照"
    lines.append(f"**裁决规则(审查方 §1.2)@budget={bmid}: {rule}**")
    lines.append("")
    lines.append("_门区/物体级旁证(door_eval,固定阈值): P1/P1-V new_recall 0.58–0.87 "
                 "prec 1.0 vs P0 0.0;P1-W 更差、P1-F 显著更差(dyn 0.09 vs 0.42)——"
                 "见 runs/stage3b/metrics/door/door_summary.csv_")
    OUT_MD.write_text("\n".join(lines))
    with open(P / "attribution_verdict.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({c for r in table for c in r}))
        w.writeheader()
        for r in table:
            w.writerow(r)
    print("\n".join(lines[-25:]))
    print(f"\n-> {OUT_MD}")


if __name__ == "__main__":
    main()
