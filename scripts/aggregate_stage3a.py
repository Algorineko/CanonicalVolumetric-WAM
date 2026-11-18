"""Aggregate Stage 3a eval CSVs into comparison tables."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

M_KEYS = ["observed_occupancy_iou", "surface_f1_tol1", "dynamic_region_iou",
          "change_f1", "reveal_occ_acc", "reveal_occ_iou", "knownfree_false_occ",
          "vis_iou"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_all():
    out = {}
    for p in sorted(Path("runs/stage3a/metrics").glob("eval_*.csv")):
        m = re.match(r"eval_(.+?)(?:_shuffle)?(?:_drop_\w+)?\.csv$", p.name)
        tag = m.group(1)
        suffix = ""
        if "_drop_" in p.name:
            suffix = "_drop_" + p.name.split("_drop_")[1].replace(".csv", "")
        elif "_shuffle" in p.name:
            suffix = "_shuffle"
        rows = list(csv.DictReader(open(p)))
        # group by offset, average
        per_off = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for k in M_KEYS:
                if k in r and r[k] != "":
                    per_off[r["offset"]][k].append(f(r[k]))
        out[tag + suffix] = {off: {k: float(np.nanmean(v)) for k, v in d.items()}
                             for off, d in per_off.items()}
    return out


def main():
    data = load_all()
    # headline: average over offsets 4..16 per model
    rows = []
    for tag, offs in sorted(data.items()):
        acc = defaultdict(list)
        for off, md in offs.items():
            for k, v in md.items():
                acc[k].append(v)
        row = {k: round(float(np.nanmean(v)), 4) for k, v in acc.items()}
        row["tag"] = tag
        rows.append(row)
    # val/test separated: n-weighted means over PURE-episode rows only
    import csv as _csv
    import re as _re
    vt = {}
    for p in sorted(Path("runs/stage3a/metrics").glob("eval_*.csv")):
        m = _re.match(r"eval_(.+?)(?:_shuffle)?(?:_drop_\w+)?\.csv$", p.name)
        tag = m.group(1)
        suffix = ""
        if "_drop_" in p.name:
            suffix = "_drop_" + p.name.split("_drop_")[1].replace(".csv", "")
        elif "_shuffle" in p.name:
            suffix = "_shuffle"
        for r in _csv.DictReader(open(p)):
            if r["episodes"].endswith(":0"):
                ep_tag = "val"
            elif r["episodes"].endswith(":1"):
                ep_tag = "test"
            else:
                continue  # mixed boundary rows excluded
            g = vt.setdefault((tag + suffix, ep_tag), defaultdict(list))
            for k in M_KEYS:
                if r.get(k, "") not in ("", "nan"):
                    g[k].append(f(r[k]))
    json.dump({f"{t}/{e}": {k: round(float(np.nanmean(v)), 4) for k, v in d.items()}
               for (t, e), d in vt.items()},
              open("runs/stage3a/metrics/val_test_split.json", "w"), indent=1)
    print("\nval/test separated (pure-episode rows):")
    for (t, e) in sorted(vt):
        d = vt[(t, e)]
        if d.get("observed_occupancy_iou") or d.get("dynamic_region_iou"):
            print(f"{t:22s} {e:5s} occIoU={np.nanmean(d.get('observed_occupancy_iou',[np.nan])):.3f} "
                  f"dynIoU={np.nanmean(d.get('dynamic_region_iou',[np.nan])):.3f} "
                  f"chgF1={np.nanmean(d.get('change_f1',[np.nan])):.3f} "
                  f"falseOcc={np.nanmean(d.get('knownfree_false_occ',[np.nan])):.4f}")
    keys = ["tag"] + [k for k in M_KEYS if any(k in r for r in rows)]
    with open("runs/stage3a/metrics/aggregated_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # per-offset detail for the main variants
    detail = {tag: {off: {k: round(v, 4) for k, v in md.items()}
                    for off, md in offs.items()}
              for tag, offs in data.items()}
    json.dump(detail, open("runs/stage3a/metrics/aggregated_detail.json", "w"), indent=1)
    print(f"{'tag':22s} {'occIoU':>7s} {'surfF1':>7s} {'dynIoU':>7s} {'chgF1':>7s} "
          f"{'revAcc':>7s} {'falseOcc':>8s} {'visIoU':>7s}")
    for r in rows:
        print(f"{r['tag']:22s} "
              f"{r.get('observed_occupancy_iou', float('nan')):7.3f} "
              f"{r.get('surface_f1_tol1', float('nan')):7.3f} "
              f"{r.get('dynamic_region_iou', float('nan')):7.3f} "
              f"{r.get('change_f1', float('nan')):7.3f} "
              f"{r.get('reveal_occ_acc', float('nan')):7.3f} "
              f"{r.get('knownfree_false_occ', float('nan')):8.4f} "
              f"{r.get('vis_iou', float('nan')):7.3f}")


if __name__ == "__main__":
    main()
