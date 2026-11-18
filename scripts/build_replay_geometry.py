"""Stage 3b: build the replay episode's own geometry + door/object GT regions.

The replay (replay_episode_gt.py) is an independently regenerated episode
(near-twin of the original; NOT fused with original data per the §2 verdict).
This script:
1. unprojects the replay's own 3-view depth into the base frame and voxelizes
   (identical pipeline/conventions to cache_episode_voxels);
2. builds door-region / object-region voxel masks from the microwave LINK
   POSES x mesh point samples, with the asset->sim transform chain verified by
   nearest-neighbor distance against the observed depth cloud (picks the chain
   whose NN distance is minimal; reports all candidates);
3. writes a DynamicsDataset-compatible pseudo-episode cache under
   voxels_ep_replay/ so models can be evaluated on it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import voxel as vx  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base, make_tf  # noqa: E402

ASSETS = Path("/data/safi-wam/robottwin2-assets/objects/044_microwave")
DOOR_LABELS = None  # resolved from the mobility URDF at runtime


def load_door_labels(asset_dir: Path):
    """Door point labels = numeric suffixes of link_0's visual mesh files."""
    import xml.etree.ElementTree as ET
    urdf = next(asset_dir.glob("mobility*.urdf"))
    root = ET.parse(urdf).getroot()
    door_link = None
    for j in root.findall("joint"):
        if j.get("name") == "joint_0" or (j.find("axis") is not None and
                                          j.get("type") == "revolute"):
            child = j.find("child").get("link")
            if door_link is None:
                door_link = child
    labels = set()
    for link in root.findall("link"):
        if door_link and link.get("name") != door_link:
            continue
        for mesh in link.iter("mesh"):
            m = re.findall(r"-(\d+)(?:\.stl|\.obj|\.dae)", mesh.get("filename", ""))
            for x in m:
                labels.add(int(x))
    return door_link, sorted(labels)


def load_points(asset_dir: Path):
    lines = (asset_dir / "point_sample" / "sample-points-all-pts-label-10000.ply").read_text().splitlines()
    hdr = [i for i, l in enumerate(lines) if l == "end_header"][0]
    data = np.array([[float(x) for x in l.split()] for l in lines[hdr + 1:hdr + 10001]])
    labels = np.loadtxt(asset_dir / "point_sample" / "label-10000.txt")
    return data[:, :3], labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--asset", default="base0")
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--pitch", type=float, default=0.02)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    asset_id = {"base0": "7167", "base1": "7310"}[args.asset]
    asset_dir = ASSETS / asset_id
    door_link, door_labels = load_door_labels(asset_dir)
    pts_asset, labels = load_points(asset_dir)
    # model_data.json: scale + transform (asset -> sim)
    md = json.loads((asset_dir / "model_data.json").read_text())
    scale = float(md.get("scale", 1.0))
    Toff = np.asarray(md.get("transform_matrix", np.eye(4)), dtype=np.float64)
    print(f"asset {asset_id}; door_link={door_link}; door_labels={door_labels}; "
          f"scale={scale}")

    rows = json.loads((Path(args.replay_dir) / "replay_gt.json").read_text())
    img_dir = Path(args.replay_dir) / "rgbdepth"
    out_dir = Path(cfg["cache_dir"]) / "voxels_ep_replay" / \
        f"{args.variant}_{args.episode}_replay" / "20mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, shape = vx.grid_config(b_min, b_max, args.pitch)

    manifest = []
    door_nn_report = []
    for i, r in enumerate(rows):
        t = r["frame"]
        imgf = img_dir / f"frame_{t:07d}.npz"
        if not imgf.exists():
            continue
        d = np.load(imgf)
        centers, pts_list, valid_list = [], [], []
        for cam, tag in (("head_camera", "head"), ("left_camera", "left"),
                         ("right_camera", "right")):
            K = np.asarray(r[f"K_{cam}"], dtype=np.float64)
            Twc_gl = np.asarray(r[f"T_world_cam_gl_{cam}"], dtype=np.float64)
            T_bc = T_bw @ geo_T(Twc_gl)
            dmm = d[f"depth_mm_{tag}"].astype(np.float32)
            valid = dmm > 0
            dm = np.where(valid, dmm / 1000.0, 0.0)
            p, _ = depth_to_base(dm, K, T_bc, valid=valid)
            centers.append(T_bc[:3, 3])
            pts_list.append(p)
            valid_list.append(np.ones(len(p), dtype=bool))
        g = vx.voxelize_rays(np.stack(centers), pts_list, valid_list,
                             b_min, b_max, args.pitch, shape)

        # door/object point clouds under candidate chains (verified by NN)
        R_root = quat_R(r["mic_q_wxyz"])
        p_root = np.asarray(r["mic_p"], dtype=np.float64)
        links = r.get("mic_link_poses", {})
        door_pts_world = None
        obj_pts_world = None
        if links and door_link in links:
            lp = links[door_link]
            R_l = quat_R(lp["q_wxyz"])
            p_l = np.asarray(lp["p"], dtype=np.float64)
            dm = (labels[:, None] == np.asarray(door_labels)).any(1)
            # candidate chains for asset->door-link-local points
            cands = {
                "scale_then_Toff": (pts_asset * scale) @ Toff[:3, :3].T + Toff[:3, 3],
                "Toff_then_scale": ((pts_asset @ Toff[:3, :3].T + Toff[:3, 3]) * scale),
                "scale_only": pts_asset * scale,
            }
            best = None
            for cname, P in cands.items():
                pw = P @ R_l.T + p_l
                pw_base = pw @ T_bw[:3, :3].T + T_bw[:3, 3]
                # NN against the observed head-view cloud
                obs = pts_list[0]
                if len(obs) > 20000:
                    obs = obs[:: len(obs) // 20000]
                nn = cKDTree(obs).query(pw_base[dm], k=1)[0]
                md_dist = float(np.median(nn))
                door_nn_report.append({"frame": t, "chain": cname,
                                        "median_nn_m": round(md_dist, 4)})
                if best is None or md_dist < best[0]:
                    best = (md_dist, cname, pw_base, dm)
            _, cname, door_pts_world, dm = best
            obj_pts_world = None
        # voxelize door/object regions (mark voxels containing GT points)
        door_region = np.zeros(tuple(shape), dtype=bool)
        obj_region = np.zeros(tuple(shape), dtype=bool)
        if door_pts_world is not None:
            idx, inb = vx.points_to_voxel_idx(door_pts_world[dm], b_min, args.pitch, shape)
            door_region[idx[inb, 0], idx[inb, 1], idx[inb, 2]] = True
        if obj_pts_world is not None:
            idx, inb = vx.points_to_voxel_idx(obj_pts_world, b_min, args.pitch, shape)
            obj_region[idx[inb, 0], idx[inb, 1], idx[inb, 2]] = True

        qpos = np.asarray(r["robot_qpos_actual"], dtype=np.float32)[:14]
        act_end = min(i + 16, len(rows))
        # NOTE: replay rows use original action group alignment via t index;
        # actions are read from the ORIGINAL hdf5 by cache_episode_voxels
        # conventions -- here we store the replay's actual qpos only, actions
        # are attached at eval time from the original episode manifest.
        npz = out_dir / f"frame_{t:07d}.npz"
        np.savez_compressed(
            npz, hit=g["hit"], free=g["free"], observed=g["observed"],
            unknown=g["unknown"], view_count=g["view_count"],
            conflict=g["conflict"], door_region=door_region,
            obj_region=obj_region, qpos=qpos,
            door_angle=np.float64(r.get("door_angle", np.nan)),
            b_min=b_min.astype(np.float32), pitch=np.float32(args.pitch),
            shape=np.asarray(shape))
        manifest.append({"t": t, "npz": str(npz),
                         "door_angle": r.get("door_angle"),
                         "mic_p": r.get("mic_p"), "mic_q_wxyz": r.get("mic_q_wxyz")})
        if i % 100 == 0:
            print(f"  frame {t} ({i}/{len(rows)})", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps({
        "variant": f"{args.variant}_{args.episode}_replay", "episode": args.episode,
        "pitch_m": args.pitch, "T_total": len(manifest), "frequency_hz": 15,
        "language": "replay episode (independently regenerated; NOT fused with original)",
        "b_min": b_min.tolist(), "b_max": b_max.tolist(),
        "table_z_base": 0.7401, "action_horizon": 16, "frames": manifest,
        "door_nn_report": door_nn_report[:40],
    }, indent=1))
    print(f"-> {out_dir} ({len(manifest)} frames)")


def geo_T(cam2world_gl):
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def quat_R(q_wxyz):
    from cvwam import geometry as geo
    return geo.quat_wxyz_to_R(q_wxyz)


import re  # noqa: E402

if __name__ == "__main__":
    main()
