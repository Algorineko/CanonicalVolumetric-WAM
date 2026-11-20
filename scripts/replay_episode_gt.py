"""Stage 3b P0: deterministic replay of one open_microwave episode with
independent object GT recording (docs: runs/stage3b plan §2).

Replays the recorded joint paths (collect_data phase-2 semantics:
need_plan=False + _traj_data paths + seed) and captures per frame:
  - microwave root pose (p, q_wxyz) + door joint angle (qpos[0])
  - actor-level segmentation for head/left/right cameras (+ microwave actor id)
  - robot ACTUAL qpos/qvel (never stored in the original HDF5)
  - camera intrinsic_cv / extrinsic_cv / cam2world_gl
  - frame index (timing = index / frequency)

Then verifies replay-vs-original alignment: per-frame RGB (decoded, JPEG-noise
tolerance), depth (lossless -> exact), and camera extrinsics (exact). If
alignment fails, labels are NOT attached to the original data; the replay
stands alone as an independently regenerated episode.

Run inside the `robottwin` conda env (sapien + rendering stack).
"""

import argparse
import copy
import importlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import yaml

ROBOTWIN = Path("/home/zhangjialiang/code/RoboTwin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--head-cam-dis", type=float, default=0.0,
                    help="observer probe: override domain_randomization."
                         "random_head_camera_dis (camera jitter; RNG draw count "
                         "is dis-independent so physics stays identical)")
    ap.add_argument("--head-cam-rot-deg", type=float, default=0.0,
                    help="observer probe: rotate head camera around its left "
                         "axis by this many degrees (deterministic)")
    ap.add_argument("--head-cam-shift-cm", type=float, default=0.0,
                    help="observer probe: shift head camera position by this "
                         "many cm along a seeded random direction (deterministic)")
    ap.add_argument("--task", default="open_microwave",
                    help="RoboTwin task name (data path + env module)")
    args_cli = ap.parse_args()

    os.chdir(ROBOTWIN)
    sys.path.insert(0, str(ROBOTWIN))
    config = yaml.safe_load((ROBOTWIN / "env_cfg/task_config/demo_clean_balanced.yml").read_text())
    if args_cli.head_cam_dis != 0.0:
        config.setdefault("domain_randomization", {})["random_head_camera_dis"] = \
            args_cli.head_cam_dis
    robot = ROBOTWIN / "assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((robot / "config.yml").read_text())
    if args_cli.head_cam_rot_deg != 0.0 or args_cli.head_cam_shift_cm != 0.0:
        # deterministic head-camera intervention (physics untouched: camera
        # creation consumes the same RNG draws regardless of these values)
        embodiment = copy.deepcopy(embodiment)
        hc = embodiment["static_camera_list"][0]
        fwd = np.asarray(hc["forward"], dtype=np.float64)
        left = np.asarray(hc["left"], dtype=np.float64)
        if args_cli.head_cam_rot_deg != 0.0:
            ax = left / np.linalg.norm(left)
            th = np.radians(args_cli.head_cam_rot_deg)
            fwd = (fwd * np.cos(th) + np.cross(ax, fwd) * np.sin(th)
                   + ax * np.dot(ax, fwd) * (1 - np.cos(th)))
            hc["forward"] = fwd.tolist()
        if args_cli.head_cam_shift_cm != 0.0:
            rng = np.random.default_rng(args_cli.episode * 1000
                                        + int(args_cli.head_cam_shift_cm * 10)
                                        + int(args_cli.head_cam_rot_deg * 10))
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            pos = np.asarray(hc["position"], dtype=np.float64) \
                + d * (args_cli.head_cam_shift_cm / 100.0)
            hc["position"] = pos.tolist()
        print(f"[probe] head camera rot={args_cli.head_cam_rot_deg}deg "
              f"shift={args_cli.head_cam_shift_cm}cm -> fwd={hc['forward']} "
              f"pos={hc['position']}")
    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)

    data_root = Path("/data/openmw_balanced_work") / args_cli.variant / \
        args_cli.task / "aloha_agilex"
    seed = int((data_root / "seed.txt").read_text().split()[args_cli.episode])
    # cross-check against instruction json
    instr = json.loads((data_root / "instruction" / f"episode_{args_cli.episode:07d}.json").read_text())
    assert instr.get("seed") == seed, f"seed mismatch: seed.txt {seed} vs instruction {instr.get('seed')}"

    with open(data_root / "_traj_data" / f"episode{args_cli.episode}.pkl", "rb") as f:
        traj = pickle.load(f)

    config.update(
        left_robot_file=str(robot), right_robot_file=str(robot),
        dual_arm_embodied=True, left_embodiment_config=embodiment,
        right_embodiment_config=embodiment,
        save_path=str(out / "simulator_work"), save_data=False,
        render_freq=0, eval_mode=True, collect_data=True, save_freq=15,
        task_config="demo_clean_balanced", need_plan=False,
        left_joint_path=traj["left_joint_path"],
        right_joint_path=traj["right_joint_path"],
    )
    config["data_type"].update(rgb=True, depth=True, endpose=True, qpos=True,
                               pointcloud=False, actor_segmentation=True)

    task = args_cli.task
    env = getattr(importlib.import_module("envs." + task), task)()

    rows = []
    state = {"armed": False, "n": 0}

    def capture(observation, idx):
        mic = getattr(env, "microwave", None)
        row = {"frame": idx}
        if mic is not None:
            try:
                pose = mic.get_pose()
                row["mic_p"] = np.asarray(pose.p, dtype=np.float64).tolist()
                row["mic_q_wxyz"] = np.asarray(pose.q, dtype=np.float64).tolist()
            except Exception as e:
                row["mic_pose_error"] = str(e)
            try:
                row["door_angle"] = float(np.asarray(mic.get_qpos()).ravel()[0])
            except Exception as e:
                row["door_qpos_error"] = str(e)
            # ArticulationActor wraps the PhysxArticulation in .actor;
            # capture per-link WORLD poses (door region construction uses
            # link poses x mesh point samples — no segmentation dependency)
            try:
                links = mic.actor.get_links()
                row["mic_link_poses"] = {
                    ln.get_name(): {
                        "p": np.asarray(ln.get_pose().p, dtype=np.float64).tolist(),
                        "q_wxyz": np.asarray(ln.get_pose().q, dtype=np.float64).tolist(),
                    } for ln in links}
            except Exception as e:
                row["mic_link_poses_error"] = str(e)
        try:
            ent = env.robot._entity
            row["robot_qpos_actual"] = np.asarray(ent.get_qpos(), dtype=np.float64).tolist()
            try:
                row["robot_qvel_actual"] = np.asarray(ent.get_qvel(), dtype=np.float64).tolist()
            except Exception:
                pass
            try:
                row["robot_link_names"] = [ln.get_name() for ln in ent.get_links()]
            except Exception as e:
                row["robot_link_names_error"] = str(e)
        except Exception as e:
            row["robot_qpos_error"] = str(e)
        # render id -> visual name mapping (per Scene API of SAPIEN 3.0.0b1);
        # classify each seg id by its visual name (microwave mesh parts,
        # robot link meshes, table/walls)
        try:
            actors = env.scene.get_all_actors()
            row["scene_actor_ids"] = {a.get_name(): int(a.per_scene_id)
                                       for a in actors}
        except Exception as e:
            row["scene_actor_ids_error"] = str(e)
        # camera group: get_config() gives K/T per name; raw actor seg per camera
        try:
            cfgd = env.cameras.get_config()
            for name in ("head_camera", "left_camera", "right_camera"):
                c = cfgd[name]
                row[f"K_{name}"] = np.asarray(c["intrinsic_cv"], dtype=np.float64).tolist()
                row[f"T_world_cam_gl_{name}"] = np.asarray(c["cam2world_gl"], dtype=np.float64).tolist()
                row[f"T_world_cam_cv_{name}"] = np.asarray(c["extrinsic_cv"], dtype=np.float64).tolist()
        except Exception as e:
            row["cam_config_error"] = str(e)
        try:
            cams = env.cameras
            raw = {}
            for name in cams.static_camera_name:
                i = cams.static_camera_name.index(name)
                if name in ("head_camera", "front_camera"):
                    raw[name] = np.asarray(cams.static_camera_list[i].get_picture("Segmentation")[..., 1]).astype(np.int32)
            raw["left_camera"] = np.asarray(cams.left_camera.get_picture("Segmentation")[..., 1]).astype(np.int32)
            raw["right_camera"] = np.asarray(cams.right_camera.get_picture("Segmentation")[..., 1]).astype(np.int32)
            for name in ("head_camera", "left_camera", "right_camera"):
                row[f"seg_actor_{name}"] = raw[name]
            id2name = {}
            for name in ("head_camera", "left_camera", "right_camera"):
                for i in np.unique(raw[name]):
                    try:
                        id2name[int(i)] = env.scene.render_id_to_visual_name(int(i))
                    except Exception:
                        id2name[int(i)] = "?"
            row["seg_id_names"] = id2name
        except Exception as e:
            row["seg_error"] = str(e)
        # RGB + depth for replay-vs-original alignment verification
        try:
            obsd = observation["observation"]
            for name, tag in (("head_camera", "head"), ("left_camera", "left"),
                              ("right_camera", "right")):
                row[f"rgb_{tag}"] = np.asarray(obsd[name]["rgb"], dtype=np.uint8)
                row[f"depth_mm_{tag}"] = np.asarray(obsd[name]["depth"], dtype=np.float32)
        except Exception as e:
            row["rgbdepth_error"] = str(e)
        rows.append(row)

    def picture(self):
        self.FRAME_IDX += 1
        if state["armed"]:
            capture(self.get_obs(), self.FRAME_IDX)
            state["n"] += 1
            if args_cli.max_frames and state["n"] >= args_cli.max_frames:
                state["armed"] = False

    env._take_picture = types.MethodType(picture, env)
    try:
        env.setup_demo(now_ep_num=args_cli.episode, seed=seed, **{**config, "task_name": task})
        state["armed"] = True
        capture(env.get_obs(), 0)          # frame 0
        env.play_once()
        capture(env.get_obs(), state["n"])  # final frame
    finally:
        status = {"captured_frames": len(rows),
                  "plan_success": bool(getattr(env, "plan_success", False)),
                  "task_success": bool(env.check_success()) if hasattr(env, "check_success") else None,
                  "seed": seed, "episode": args_cli.episode, "variant": args_cli.variant}
        try:
            env.close_env()
        except Exception:
            pass

    (out / "replay_meta.json").write_text(json.dumps(status, indent=2))
    # big arrays saved separately as npz per frame
    seg_dir = out / "seg"
    seg_dir.mkdir(exist_ok=True)
    img_dir = out / "rgbdepth"
    img_dir.mkdir(exist_ok=True)
    for r in rows:
        big = {k: v for k, v in r.items()
               if k.startswith("seg_actor_") or k.startswith("rgb_") or k.startswith("depth_mm_")}
        for k in list(big):
            r.pop(k)
        np.savez_compressed(seg_dir / f"frame_{r['frame']:07d}.npz",
                            **{k: v for k, v in big.items() if k.startswith("seg_actor_")})
        np.savez_compressed(img_dir / f"frame_{r['frame']:07d}.npz",
                            **{k: v for k, v in big.items() if not k.startswith("seg_actor_")})
    (out / "replay_gt.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(status))
    print(f"rows -> {out/'replay_gt.json'}; seg -> {seg_dir}")


if __name__ == "__main__":
    main()
