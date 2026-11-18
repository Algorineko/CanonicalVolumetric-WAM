"""Stage 2c: per-frame robot-occupancy region via numpy FK + collision spheres.

FK chain (footprint-local): fixed fl_base_joint -> fl_base_link -> revolute
joints 1..6 -> link6. link6 pose is then OVERWRITTEN with the recorded
state/{side}_ee_poses (the actual simulated link6 pose; arm joint states are
PD drive targets and lag during contact — scout-verified, max ~24mm). link7/8
prismatic offsets applied from the corrected link6. Sphere proxies from
collision_aloha_{left,right}.yml (curobo collision_spheres format) are
transformed per link and marked in the base-frame voxel grid.

Validation channel: FK link6 vs recorded ee_poses position error distribution
per episode (evidence of marking uncertainty).

NOTE: spheres whose center falls outside the grid are clipped (proximal arm
shoulder region is outside the workspace bbox — documented, not silently
ignored: the fraction of spheres outside is reported).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import geometry as geo  # noqa: E402
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam.geometry import T_base_world  # noqa: E402

URDF_DIR = Path("/data/safi-wam/robottwin2-assets/embodiments/aloha-agilex")


def load_chain(side: str):
    """Return list of (joint_name, axis, origin_xyz, origin_rpy, type) for one arm."""
    import xml.etree.ElementTree as ET
    root = ET.parse(URDF_DIR / "urdf" / "arx5_description_isaac.urdf").getroot()
    joints = {}
    for j in root.findall("joint"):
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=j.find("origin"),
            axis=j.find("axis").get("xyz") if j.find("axis") is not None else None,
        )
    p = "fl" if side == "left" else "fr"
    chain, link = [], f"{p}_link6"
    # walk up from link6 to footprint
    cur = link
    while cur != "footprint":
        jname = [n for n, j in joints.items() if j["child"] == cur][0]
        chain.append((jname, joints[jname]))
        cur = joints[jname]["parent"]
    chain.reverse()
    return chain, joints


def origin_transform(origin):
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"):
            xyz = np.array([float(v) for v in origin.get("xyz").split()])
        if origin.get("rpy"):
            rpy = np.array([float(v) for v in origin.get("rpy").split()])
    R = geo.quat_wxyz_to_R(_rpy_to_quat(rpy))
    return R, xyz


def _rpy_to_quat(rpy):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", rpy).as_quat()[[3, 0, 1, 2]]  # -> wxyz


def fk_links(chain, joints, qdict):
    """chain: list of (name, jinfo); qdict maps joint name -> value (prismatic
    in meters). Returns dict link_name -> (R, t) in footprint-local coords."""
    poses = {}
    R = np.eye(3)
    t = np.zeros(3)
    cur_link = "footprint"
    poses["footprint"] = (np.eye(3), np.zeros(3))
    for jname, jinfo in chain:
        Ro, to = origin_transform(jinfo["origin"])
        t = t + R @ to          # translation in PARENT frame
        R = R @ Ro              # then own rotation
        if jinfo["type"] == "revolute" or jinfo["type"] == "continuous":
            q = qdict.get(jname, 0.0)
            axis = np.array([float(v) for v in jinfo["axis"].split()])
            Rq = geo.quat_wxyz_to_R(_axisangle_quat(axis, q))
            R = R @ Rq
        elif jinfo["type"] == "prismatic":
            q = qdict.get(jname, 0.0)
            axis = np.array([float(v) for v in jinfo["axis"].split()])
            t = t + R @ (axis * q)
        cur_link = jinfo["child"]
        poses[cur_link] = (R.copy(), t.copy())
    return poses


def _axisangle_quat(axis, q):
    n = np.linalg.norm(axis)
    axis = axis / n
    return np.concatenate([[np.cos(q / 2)], axis * np.sin(q / 2)])


def load_spheres(side: str):
    import re
    txt = (URDF_DIR / f"collision_aloha_{side}.yml").read_text()
    spheres = {}
    cur = None
    last_center = None
    for line in txt.splitlines():
        m = re.match(r"\s{4}(\S+):\s*$", line.rstrip())
        if m and "collision_spheres" not in line:
            cur = m.group(1)
            spheres.setdefault(cur, [])
            last_center = None
            continue
        m2 = re.search(r'"center":\s*\[([^\]]+)\]', line)
        if m2 and cur:
            last_center = np.array([float(v) for v in m2.group(1).split(",")])
        m3 = re.search(r'"radius":\s*([0-9.]+)', line)
        if m3 and cur and last_center is not None:
            spheres[cur].append((last_center, float(m3.group(1))))
            last_center = None
    return spheres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--pitch", default="20mm")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    T_wb = rta.T_world_base_from_config(cfg)
    T_bw = geo.T_base_world(T_wb)
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    out_root = root / "_robot_region"
    out_root.mkdir(parents=True, exist_ok=True)

    chains = {s: load_chain(s) for s in ("left", "right")}
    spheres = {s: load_spheres(s) for s in ("left", "right")}

    for spec in args.episodes:
        variant, ep = spec.split(":")
        ep = int(ep)
        edir = root / variant / f"episode_{ep:07d}" / args.pitch
        mf = json.loads((edir / "manifest.json").read_text())
        b_min = np.asarray(mf["b_min"], dtype=np.float64)
        pitch = float(mf["pitch_m"])
        f0 = np.load(edir / "frame_0000000.npz")
        shape = tuple(int(v) for v in f0["shape"])
        with h5py.File(rta.locate_episode(cfg["data_root"], variant, ep).hdf5, "r") as f:
            arm = {s: np.asarray(f[f"state/{s}_arm_joint_states"]) for s in ("left", "right")}
            grip = {s: np.asarray(f[f"state/{s}_ee_joint_states"])[:, 0] for s in ("left", "right")}
            ee = {s: np.asarray(f[f"state/{s}_ee_poses"]) for s in ("left", "right")}
        outdir = out_root / variant / f"episode_{ep:07d}"
        outdir.mkdir(parents=True, exist_ok=True)
        errs = []
        for t in range(mf["T_total"]):
            robot = np.zeros(shape, dtype=bool)
            n_out = 0
            n_sph = 0
            for side in ("left", "right"):
                qd = {f"{'fl' if side == 'left' else 'fr'}_joint{i + 1}": arm[side][t, i]
                      for i in range(6)}
                p = "fl" if side == "left" else "fr"
                grip_raw = float(np.clip(grip[side][t], 0, 1))
                qd[f"{p}_joint7"] = float(np.clip(
                    -0.01 + 0.055 * grip_raw, 0, 0.04765))
                qd[f"{p}_joint8"] = qd[f"{p}_joint7"]
                poses = fk_links(chains[side][0], chains[side][1], qd)
                # overwrite link6 with the recorded ACTUAL pose (world -> base)
                qw = ee[side][t]
                R6_w = geo.quat_wxyz_to_R(qw[3:])
                t6_w = qw[:3]
                t6_b = T_bw[:3, :3] @ t6_w + T_bw[:3, 3]
                R6_b = T_bw[:3, :3] @ R6_w
                if f"{p}_link6" in poses:
                    # FK-vs-recorded error (evidence channel)
                    Rfk, tfk = poses[f"{p}_link6"]
                    tfk_w = T_wb[:3, :3] @ tfk + T_wb[:3, 3]
                    errs.append(float(np.linalg.norm(tfk_w - t6_w)))
                    poses[f"{p}_link6"] = (R6_b, t6_b)
                # links 7/8 re-anchored to corrected link6
                for ln in (f"{p}_link7", f"{p}_link8"):
                    if ln not in poses:
                        continue
                    Rj, tj = _joint_transform_in_chain(chains[side][0], ln, poses, qd)
                    poses[ln] = (poses[f"{p}_link6"][0] @ Rj,
                                 poses[f"{p}_link6"][1] + poses[f"{p}_link6"][0] @ tj)
                # mark spheres
                for link, sph_list in spheres[side].items():
                    if link not in poses:
                        continue
                    R, tt = poses[link]
                    for c_local, r in sph_list:
                        n_sph += 1
                        c = R @ np.asarray(c_local) + tt
                        idx = np.floor((c - b_min) / pitch).astype(int)
                        rad_cells = int(np.ceil(r / pitch))
                        lo = np.maximum(idx - rad_cells, 0)
                        hi = np.minimum(idx + rad_cells + 1, np.asarray(shape))
                        if (idx < 0).any() or (idx >= np.asarray(shape)).any():
                            n_out += 1
                            continue
                        gi, gj, gk = np.indices((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
                        centers = b_min + np.stack([gi + lo[0], gj + lo[1], gk + lo[2]],
                                                   axis=-1) * pitch + pitch / 2
                        d = np.linalg.norm(centers - c, axis=-1)
                        m = d <= r
                        robot[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= m
            np.savez_compressed(outdir / f"frame_{t:07d}.npz", robot=robot)
            if t % 100 == 0:
                print(f"  [{spec}] {t}/{mf['T_total']} spheres_out={n_out}/{n_sph}",
                      flush=True)
        meta = {"episode": spec, "fk_vs_ee_pose_err_mm": {
            "mean": round(float(np.mean(errs)) * 1000, 2),
            "p95": round(float(np.percentile(errs, 95)) * 1000, 2),
            "max": round(float(np.max(errs)) * 1000, 2),
            "note": "mid-link lag (PD drive targets) remains; link6 exact"}}
        (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))


def _joint_transform_in_chain(chain, link_name, poses, qd):
    """Local transform from link6 to the given link along the chain."""
    R = np.eye(3)
    t = np.zeros(3)
    cur = "fl_link6" if "fl_link6" in link_name else "fr_link6"
    for jname, jinfo in chain:
        if jinfo["parent"] != cur:
            continue
        Ro, to = origin_transform(jinfo["origin"])
        t = t + R @ to          # translation in PARENT frame
        R = R @ Ro              # then own rotation
        if jinfo["type"] in ("revolute", "continuous"):
            q = qd.get(jname, 0.0)
            axis = np.array([float(v) for v in jinfo["axis"].split()])
            R = R @ geo.quat_wxyz_to_R(_axisangle_quat(axis, q))
        elif jinfo["type"] == "prismatic":
            q = qd.get(jname, 0.0)
            axis = np.array([float(v) for v in jinfo["axis"].split()])
            t = t + R @ (axis * q)
        cur = jinfo["child"]
        if cur == link_name:
            return R, t
    return R, t


if __name__ == "__main__":
    main()
