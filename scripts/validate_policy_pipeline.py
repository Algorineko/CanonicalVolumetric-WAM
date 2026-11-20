"""Stage 4a.0: validate the closed-loop control interface via expert-action
replay, and measure online voxel construction latency.

In-process in the robottwin env (no policy server): setup_demo like the eval
script (expert plan runs for seed gating), then drive take_action directly
with the cached 14-dim actions CONVERTED from cache order
[Larm6, Rarm6, Lgrip, Rgrip] to eval take_action order
[Larm6, Lgrip, Rarm6, Rgrip]. Asserts task_success triggers and samples the
online depth->voxel latency (the FM policy adapter's input path).

Outputs runs/stage4a/pipeline_check.json.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import voxel as vx  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402

ROBOTWIN = Path("/home/zhangjialiang/code/RoboTwin")


def cache_to_eval_order(a14):
    """[Larm6, Rarm6, Lgrip, Rgrip] -> [Larm6, Lgrip, Rarm6, Rgrip]."""
    out = np.empty_like(a14)
    out[0:6] = a14[0:6]
    out[6] = a14[12]
    out[7:13] = a14[6:12]
    out[13] = a14[13]
    return out


class OnlineVoxelBuilder:
    """Live depth -> canonical voxel via the verified chain."""

    def __init__(self, cfg):
        self.T_bw = T_base_world(rta.T_world_base_from_config(cfg))
        mf = json.loads((Path("/data/canonical-volumetric-wam/cache/voxels_ep/"
                              "demo_clean_balanced/episode_0000010/20mm")
                         / "manifest.json").read_text())
        self.b_min = np.asarray(mf["b_min"], dtype=np.float64)
        self.b_max = np.asarray(mf["b_max"], dtype=np.float64)
        _, _, self.shape = vx.grid_config(self.b_min, self.b_max, float(mf["pitch_m"]))
        self.pitch = float(mf["pitch_m"])

    def build(self, cams_depth, cams_K, cams_gl):
        from cvwam import geometry as geo
        from voxelize_gpu import voxelize_rays_gpu
        t0 = time.perf_counter()
        cams_c, pts_list, valid_list = [], [], []
        for cam, dmm in cams_depth.items():
            T_wc = np.asarray(cams_gl[cam], dtype=np.float64)
            if T_wc.shape == (3, 4):
                T_wc = np.vstack([T_wc, [0.0, 0.0, 0.0, 1.0]])
            # verified chain: T_base_cam = T_bw @ world_cam_cv(cam2world_gl)
            T_base_cam = self.T_bw @ geo.cam2world_gl_to_world_cam_cv(T_wc)
            valid = dmm > 0
            p, msk = depth_to_base(np.where(valid, dmm / 1000.0, 0.0),
                                   np.asarray(cams_K[cam], dtype=np.float64),
                                   T_base_cam, valid=valid)
            cams_c.append(T_base_cam[:3, 3])
            pts_list.append(p)
            valid_list.append(np.ones(len(p), dtype=bool))
        g = voxelize_rays_gpu(np.stack(cams_c), pts_list, valid_list,
                              self.b_min, self.b_max, self.pitch, self.shape)
        return g, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0, help="0=all")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/smoke.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    os.chdir(ROBOTWIN)
    sys.path.insert(0, str(ROBOTWIN))
    paths = rta.locate_episode(cfg["data_root"], args.variant, args.episode,
                               task=cfg.get("task", "open_microwave"))
    builder = OnlineVoxelBuilder(cfg)

    data_root = Path("/data/openmw_balanced_work") / args.variant / "open_microwave" / "aloha_agilex"
    seed = int((data_root / "seed.txt").read_text().split()[args.episode])

    with h5py.File(paths.hdf5, "r") as f:
        act_l = np.asarray(f["action/left_arm_joint_states"], dtype=np.float32)
        act_r = np.asarray(f["action/right_arm_joint_states"], dtype=np.float32)
        gl = np.asarray(f["action/left_ee_joint_states"], dtype=np.float32)
        gr = np.asarray(f["action/right_ee_joint_states"], dtype=np.float32)
        T_total = int(act_l.shape[0])
    act = np.concatenate([act_l, act_r, gl, gr], axis=1)  # (T,14) cache order

    robot = "/data/safi-wam/robottwin2-assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((Path(robot) / "config.yml").read_text())
    base_cfg = yaml.safe_load((ROBOTWIN / "env_cfg/task_config/demo_clean_balanced.yml")
                              .read_text())
    base_cfg.update(
        left_robot_file=robot, right_robot_file=robot,
        dual_arm_embodied=True, left_embodiment_config=embodiment,
        right_embodiment_config=embodiment,
        save_path="/tmp/cvwam_pipeline_check", save_data=False,
        render_freq=0, eval_mode=True, collect_data=False, save_freq=15,
        task_config="demo_clean_balanced",
    )
    base_cfg["data_type"].update(rgb=True, depth=True, endpose=True, qpos=True,
                                 pointcloud=False, actor_segmentation=False)

    task = "open_microwave"
    env = getattr(importlib.import_module("envs." + task), task)()
    env.setup_demo(now_ep_num=args.episode, seed=seed, is_test=False,
                   **{**base_cfg, "task_name": task})

    N = T_total - 1 if args.max_frames == 0 else min(args.max_frames, T_total - 1)
    latencies, success_step, t_wall0 = [], None, time.perf_counter()
    for t in range(N):
        a_eval = cache_to_eval_order(act[t])
        env.take_action(a_eval, action_type="qpos")
        if env.eval_success and success_step is None:
            success_step = t
        if t % 8 == 0:
            obs = env.get_obs()
            cams_depth, cams_K, cams_gl = {}, {}, {}
            for cam in ("head_camera", "left_camera", "right_camera"):
                v = obs["observation"][cam]
                cams_depth[cam] = np.asarray(v["depth"], dtype=np.float64)
                cams_K[cam] = np.asarray(v["intrinsic_cv"], dtype=np.float64)
                cams_gl[cam] = np.asarray(v["cam2world_gl"], dtype=np.float64)
            g, dt = builder.build(cams_depth, cams_K, cams_gl)
            latencies.append(round(dt, 4))
        if success_step is not None and t > success_step + 20:
            break

    res = dict(
        episode=f"{args.variant}:{args.episode}", frames_stepped=int(N),
        take_action_calls=int(env.take_action_cnt),
        success_step=success_step, success=bool(env.eval_success),
        plan_success=bool(getattr(env, "plan_success", None)),
        door_qpos=float(np.asarray(env.microwave.get_qpos()).ravel()[0]),
        wall_s=round(time.perf_counter() - t_wall0, 1),
        voxel_n=len(latencies),
        voxel_latency_mean_ms=round(float(np.mean(latencies)) * 1000, 1) if latencies else None,
        voxel_latency_max_ms=round(float(np.max(latencies)) * 1000, 1) if latencies else None,
    )
    out = Path(__file__).resolve().parents[1] / "runs/stage4a/pipeline_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
