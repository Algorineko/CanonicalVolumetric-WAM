"""Stage 3b: replay episode geometry + door/object regions (data-driven).

Regions derive ONLY from the replay's own captured data (no asset-frame
archaeology; the asset->sim chain could not be validated to mm and was
abandoned — see failures note):
  - static named ids (ground/wall/table) via Scene.render_id_to_visual_name;
  - robot pixels = points inside the original FK robot region dilated (the
    replay arm is within ~2.1 cm of the original; documented approximation);
  - remaining ids = microwave parts; the DOOR id is identified globally as
    the microwave id whose point centroid, expressed in the microwave ROOT
    frame, moves most with the door angle (root frame is per-frame captured).
Voxels: identical pipeline to cache_episode_voxels (20mm, per-view rays).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import voxel as vx  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402

CAM_OF = {"head_camera": "head", "left_camera": "left", "right_camera": "right"}


def geo_T(cam2world_gl):
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def mark_voxels(points, b_min, pitch, shape):
    idx, inb = vx.points_to_voxel_idx(points, b_min, pitch, shape)
    out = np.zeros(tuple(shape), dtype=bool)
    if inb.any():
        idx = idx[inb]
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--pitch", type=float, default=0.02)
    ap.add_argument("--id-probe-frames", type=int, default=24)
    ap.add_argument("--tag", default="replay",
                    help="output dir suffix: {variant}_{episode}_{tag}")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    rows = json.loads((Path(args.replay_dir) / "replay_gt.json").read_text())
    img_dir = Path(args.replay_dir) / "rgbdepth"
    seg_dir = Path(args.replay_dir) / "seg"
    out_dir = Path(cfg["cache_dir"]) / "voxels_ep_replay" / \
        f"{args.variant}_{args.episode}_{args.tag}" / "20mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, shape = vx.grid_config(b_min, b_max, args.pitch)

    # original robot region (dilated) as the robot-pixel classifier reference
    rob_root = Path(cfg["cache_dir"]) / "voxels_ep" / "_robot_region" / \
        args.variant / f"episode_{args.episode:07d}"
    rob_dil = None

    def robot_region(t):
        nonlocal rob_dil
        f = rob_root / f"frame_{t:07d}.npz"
        if f.exists():
            r = np.load(f)["robot"]
        else:
            r = np.zeros(tuple(shape), dtype=bool)
        return r

    # ---------- pass 1: classify seg ids (probe frames across episode) ----------
    ts_all = [i for i, r in enumerate(rows)
              if (seg_dir / f"frame_{r['frame']:07d}.npz").exists()]
    probe = np.linspace(0, len(ts_all) - 1, min(args.id_probe_frames, len(ts_all))).astype(int)
    id_stats = {}   # id -> list of (door_angle, centroid_in_root)
    for i in probe:
        r = rows[ts_all[i]]
        t = r["frame"]
        segf = seg_dir / f"frame_{t:07d}.npz"
        imgf = img_dir / f"frame_{t:07d}.npz"
        if not segf.exists() or not imgf.exists():
            continue
        sd = np.load(segf)
        dd = np.load(imgf)
        R_root = None
        if "mic_q_wxyz" in r:
            from cvwam import geometry as geo
            R_root = geo.quat_wxyz_to_R(r["mic_q_wxyz"])
            p_root = np.asarray(r["mic_p"], dtype=np.float64)
        rob = robot_region(t)
        for camkey in sd.files:
            cam = camkey.replace("seg_actor_", "")
            seg = sd[camkey]
            K = np.asarray(r[f"K_{cam}"], dtype=np.float64)
            T_bc = T_bw @ geo_T(np.asarray(r[f"T_world_cam_gl_{cam}"], dtype=np.float64))
            dmm = dd[f"depth_mm_{CAM_OF[cam]}"].astype(np.float32)
            valid = dmm > 0
            dm2 = np.where(valid, dmm / 1000.0, 0.0)
            p_pts, msk = depth_to_base(dm2, K, T_bc, valid=valid)
            vv, uu = np.where(msk)
            ids_at = seg[vv, uu]
            for sid in np.unique(ids_at):
                key = int(sid)
                pts_id = p_pts[ids_at == key]
                if len(pts_id) < 30:
                    continue
                inb = ((pts_id >= b_min) & (pts_id <= b_max)).all(1)
                if inb.sum() < 30:
                    continue
                vc = np.clip(np.floor((pts_id[inb] - b_min) / args.pitch).astype(int),
                             0, np.asarray(shape) - 1)
                rob_frac = float(rob[vc[:, 0], vc[:, 1], vc[:, 2]].mean())
                st = id_stats.setdefault(key, {"robot_frac": [], "cent_root": [],
                                               "door": [], "n": 0})
                st["n"] += 1
                st["robot_frac"].append(rob_frac)
                if R_root is not None:
                    cen_root = (R_root.T @ (pts_id[inb].mean(0) - p_root))
                    st["cent_root"].append(cen_root.tolist())
                st["door"].append(float(r.get("door_angle", np.nan)))

    # classify: robot-dominated, named static, microwave, door
    STATIC_NAMES = {"ground", "wall", "table", "front_camera", "floor"}
    id_role = {}
    named_ids = {}
    for r_p in rows:
        sai = r_p.get("scene_actor_ids")
        if sai:
            named_ids.update({int(v): k for k, v in sai.items()})
            break
    door_candidate = None
    door_motion_best = -1.0
    for key, st in id_stats.items():
        rf = np.mean(st["robot_frac"]) if st["robot_frac"] else 0.0
        nm = named_ids.get(key, "")
        if key in named_ids and any(nm == s for s in STATIC_NAMES):
            id_role[key] = f"static:{nm}"
            continue
        if rf > 0.5:
            id_role[key] = "robot"
            continue
        # microwave part: estimate root-frame centroid motion vs door angle
        cent = np.asarray(st["cent_root"])
        ang = np.asarray(st["door"])
        if len(cent) >= 6 and np.nanstd(ang) > 0.05:
            motion = float(np.linalg.norm(cent.max(0) - cent.min(0)))
        else:
            motion = 0.0
        if motion > door_motion_best and motion > 0.03:   # >3cm in root frame
            door_motion_best = motion
            door_candidate = key
    for key in id_stats:
        if key not in id_role:
            id_role[key] = "door" if key == door_candidate else "mic"
    print("id roles:", {k: v for k, v in sorted(id_role.items())})
    print(f"door id: {door_candidate} (root-frame centroid motion "
          f"{door_motion_best*1000:.0f} mm)")

    # ---------- pass 2: per-frame voxels + regions ----------
    manifest = []
    for i, r in enumerate(rows):
        t = r["frame"]
        segf = seg_dir / f"frame_{t:07d}.npz"
        imgf = img_dir / f"frame_{t:07d}.npz"
        if not segf.exists() or not imgf.exists():
            continue
        sd = np.load(segf)
        dd = np.load(imgf)
        centers, pts_list, valid_list = [], [], []
        for cam, tag in CAM_OF.items():
            K = np.asarray(r[f"K_{cam}"], dtype=np.float64)
            T_bc = T_bw @ geo_T(np.asarray(r[f"T_world_cam_gl_{cam}"], dtype=np.float64))
            dmm = dd[f"depth_mm_{tag}"].astype(np.float32)
            valid = dmm > 0
            dm2 = np.where(valid, dmm / 1000.0, 0.0)
            p_pts, _ = depth_to_base(dm2, K, T_bc, valid=valid)
            centers.append(T_bc[:3, 3])
            pts_list.append(p_pts)
            valid_list.append(np.ones(len(p_pts), dtype=bool))
        g = vx.voxelize_rays(np.stack(centers), pts_list, valid_list,
                             b_min, b_max, args.pitch, shape)
        door_region = np.zeros(tuple(shape), dtype=bool)
        obj_region = np.zeros(tuple(shape), dtype=bool)
        rob = robot_region(t)
        for camkey in sd.files:
            cam = camkey.replace("seg_actor_", "")
            seg = sd[camkey]
            K = np.asarray(r[f"K_{cam}"], dtype=np.float64)
            T_bc = T_bw @ geo_T(np.asarray(r[f"T_world_cam_gl_{cam}"], dtype=np.float64))
            dmm = dd[f"depth_mm_{CAM_OF[cam]}"].astype(np.float32)
            valid = dmm > 0
            dm2 = np.where(valid, dmm / 1000.0, 0.0)
            p_pts, msk = depth_to_base(dm2, K, T_bc, valid=valid)
            vv, uu = np.where(msk)
            ids_at = seg[vv, uu]
            for key, role in id_role.items():
                m = ids_at == key
                if not m.any():
                    continue
                if role == "door":
                    door_region |= mark_voxels(p_pts[m], b_min, args.pitch, shape)
                    obj_region |= mark_voxels(p_pts[m], b_min, args.pitch, shape)
                elif role == "mic":
                    obj_region |= mark_voxels(p_pts[m], b_min, args.pitch, shape)
        # qpos in TRAINING semantics [l_arm6, r_arm6, l_grip_norm, r_grip_norm].
        # sim entity qpos dim mapping (correlation-verified vs drive targets,
        # r>=0.998; FK link6 vs original ee_poses mean 2.34mm): left arm joints
        # at dims [6,14,18,22,26,30], left grip physical meters at dim 34
        # (q35 = mimic finger), right grip at dim 36; right arm at dims 0-5
        # (frozen 0 in this episode). Full raw vector kept as qpos_full.
        qraw = np.asarray(r["robot_qpos_actual"], dtype=np.float64)
        L_ARM_DIMS = (6, 14, 18, 22, 26, 30)
        l_arm = qraw[list(L_ARM_DIMS)]
        r_arm = qraw[0:6]
        l_grip_norm = float(np.clip((qraw[34] + 0.01) / 0.055, 0, 1))
        r_grip_norm = float(np.clip((qraw[36] + 0.01) / 0.055, 0, 1))
        qpos = np.concatenate([l_arm, r_arm,
                               [l_grip_norm, r_grip_norm]]).astype(np.float32)
        npz = out_dir / f"frame_{t:07d}.npz"
        np.savez_compressed(
            npz, hit=g["hit"], free=g["free"], observed=g["observed"],
            unknown=g["unknown"], view_count=g["view_count"],
            conflict=g["conflict"], door_region=door_region,
            obj_region=obj_region, qpos=qpos, qpos_full=qraw.astype(np.float32),
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
        "id_role": {str(k): v for k, v in id_role.items()},
        "door_id": door_candidate,
        "door_centroid_motion_m": round(door_motion_best, 4),
        "region_note": "door/obj regions from replay's own actor segmentation; "
                       "robot region approximated by original FK region (replay "
                       "arm within ~2.1cm); NOT amodal-complete geometry",
    }, indent=1))
    print(f"-> {out_dir} ({len(manifest)} frames)")


def rta_scene_name(row, sid):
    """Static-scene name for a seg id via the captured mapping."""
    return row.get("seg_id_names", {}).get(str(sid), "?")


if __name__ == "__main__":
    main()
