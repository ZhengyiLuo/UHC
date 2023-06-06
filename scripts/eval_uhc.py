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
    parser.add_argument("--data", type=str, default="sample_data/amass_copycat_take5_test_small.pkl")
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

    cfg.output = osp.join("results/renderings/uhc/", f"{cfg.id}")
    os.makedirs(cfg.output, exist_ok=True)

    cfg.data_specs["file_path"] = args.data

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
        from uhc.utils.copycat_visualizer import CopycatVisualizer

        # if cfg.masterfoot:
        #     vis = CopycatVisualizer(cfg.vis_model_file, agent)
        # else:
        # vis = CopycatVisualizer(
        #     agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent
        # )
        vis = CopycatVisualizer(agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent)
        vis.display_coverage()
    else:
        from uhc.utils.copycat_visualizer import CopycatVisualizer

        # if cfg.masterfoot:
        #     vis = CopycatVisualizer(cfg.vis_model_file, agent)
        # else:
        # vis = CopycatVisualizer(
        #     agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent
        # )
        vis = CopycatVisualizer(agent.env.smpl_robot.export_vis_string().decode("utf-8"), agent)
        vis.show_animation()
