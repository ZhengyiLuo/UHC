"""
File: /eval_copycat.py
Created Date: Monday June 7th 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Monday June 7th 2021 3:57:49 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
"""

import argparse

import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp
import os

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np

from uhc.utils.flags import flags
from uhc.utils.config_utils.copycat_config import Config

from uhc.utils.image_utils import write_frames_to_video
import wandb
import mujoco_py
from uhc.utils.copycat_visualizer import CopycatVisualizer
from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose, qpos_to_smpl

class MultiVisulizer(CopycatVisualizer):

    starting = [np.random.random(2) * 10 for _ in range(10)]
     
    def update_pose(self):

        # if self.env_vis.viewer._record_video:
        #     print(self.fr)
        # print(self.fr)
        expert = self.agent.env.expert
        lim = self.agent.env.converter.new_nq + (
            expert["obj_pose"].shape[1] if expert["has_obj"] else 0
        )

        # self.data["pred"][self.fr][-14:] = expert["obj_pose"][self.fr]
        
        for i in range(num_people):
            self.env_vis.data.qpos[(lim * i):(lim * (i + 1))] = self.data[f"qpos_{i:03d}"][self.fr]

        for i in range(num_people):
            self.env_vis.data.qpos[lim * i:(lim * i + 2) ] += self.starting[i]
        

        if self.agent.cfg.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()
        # print(self.fr)
        
    def data_generator(self):
        
        if self.agent.cfg.mode != "disp_stats":
            for loader in self.agent.test_data_loaders:
                ten_keys = [k for k in loader.data_keys if loader.get_sample_len_from_key(k) > 50 and ("walk" in k or "run" in k)]
                ten_keys = np.random.choice(ten_keys, num_people, replace=False)
                
                for take_key in loader.data_keys:
                    self.cur_key = take_key
                    print(
                        f"Generating for {take_key} seqlen: {loader.get_sample_len_from_key(take_key)}"
                    )
                    # eval_res = self.agent.eval_seq(take_key, loader)
                    eval_res = {}
                    env = self.agent.env
                    for idx, k in enumerate(ten_keys):
                        data_curr = loader.get_sample_from_key(k, full_sample = True)
                        qpos = smpl_to_qpose(
                            pose=data_curr['pose_aa'],
                            mj_model=env.model,
                            trans=data_curr['trans'].squeeze(),
                            model=env.cc_cfg.robot_cfg.get("model", "smpl"),
                            count_offset=env.cc_cfg.robot_cfg.get("mesh", True),
                        )
                        eval_res[f'qpos_{idx:03d}'] = qpos[:200]


                    print(
                        "Agent Mass:",
                        mujoco_py.functions.mj_getTotalmass(self.agent.env.model),
                    )
                    
                    self.env_vis.reload_sim_model(
                        self.agent.env.smpl_robot.export_vis_string_self(
                          num = num_people 
                        ).decode("utf-8")
                    )

                    self.setup_viewing_angle()
                    self.set_video_path(
                        image_path=osp.join(
                            self.agent.cfg.output,
                            take_key,
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%04d.png",
                        ),
                        video_path=osp.join(
                            self.agent.cfg.output,
                            f"{take_key}_{self.agent.cfg.id}_{self.agent.cfg.epoch}_%01d.mp4",
                        ),
                    )
                    os.makedirs(osp.join(self.agent.cfg.output, take_key), exist_ok=True)
                    self.num_fr = np.min([v.shape[0] for v in eval_res.values()])
                    yield eval_res
        else:
            yield None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="singles")
    parser.add_argument("--mode", type=str, default="vis")
    parser.add_argument("--render_video", action="store_true", default=False)
    parser.add_argument("--render_rfc", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--hide_expert", action="store_true", default=False)
    parser.add_argument("--no_fail_safe", action="store_true", default=False)
    parser.add_argument("--focus", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--shift_expert", action="store_true", default=False)
    parser.add_argument("--smplx", action="store_true", default=False)
    parser.add_argument("--hide_im", action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    cfg.update(args)

    flags.debug = args.debug
    cfg.no_log = True
    if args.no_fail_safe:
        cfg.fail_safe = False

    cfg.output = osp.join("/hdd/zen/data/copycat/renderings/uhc/", f"{cfg.id}")
    os.makedirs(cfg.output, exist_ok=True)

    if args.data == "singles":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_train_singles.pkl"
    elif args.data == "single":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_train_single.pkl"
    elif args.data == "all":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5.pkl"
    elif args.data == "take5_test":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_test.pkl"
    elif args.data == "take5_test_small":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_test_small.pkl"
    elif args.data == "take5_single":
        cfg.data_specs[
            "file_path"
        ] = "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_single.pkl"
    elif args.data == "grab_test":
        cfg.data_specs["file_path"] = "/hdd/zen/data/ActBound/AMASS/grab_test.pkl"
    elif args.data == "grab":
        cfg.data_specs["file_path"] = "/hdd/zen/data/ActBound/AMASS/grab_take1.pkl"
    elif args.data == "usr":
        pass
    else:
        if osp.isfile(f"/hdd/zen/data/ActBound/AMASS/singles/amass_copycat_{args.data}.pkl"):
            cfg.data_specs[
            "file_path"
            ] = f"/hdd/zen/data/ActBound/AMASS/singles/amass_copycat_{args.data}.pkl"
        else:
            cfg.data_specs[
            "file_path"
            ] = f"/hdd/zen/data/ActBound/AMASS/amass_copycat_{args.data}.pkl"


    num_people = 10
    if "test_file_path" in cfg.data_specs:
        del cfg.data_specs["test_file_path"]

    if cfg.mode == "vis":
        cfg.num_threads = 1

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    # device = (
    #     torch.device("cuda", index=args.gpu_index)
    #     if torch.cuda.is_available()
    #     else torch.device("cpu")
    # )
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.smplx and cfg.robot_cfg["model"] == "smplh":
        cfg.robot_cfg["model"] = "smplx"

    from uhc.agents.agent_copycat import AgentCopycat

    agent = AgentCopycat(cfg, dtype, device, training=True, checkpoint_epoch=args.epoch)

    if args.mode == "stats":
        agent.eval_policy(epoch=args.epoch, dump=True)
    elif args.mode == "disp_stats":
        vis = MultiVisulizer(
            agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent
        )
        vis.display_coverage()
    else:
        vis = MultiVisulizer(
            agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent
        )
        vis.show_animation()
