"""
File: /train_copycat.py
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

from torch.utils.tensorboard import SummaryWriter
from uhc.utils.flags import flags
from uhc.utils.config_utils.copycat_config import Config
import wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=40)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--show_noise", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_log", action="store_false", default=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--show_single", action="store_true", default=False)

    args = parser.parse_args()

    cfg = Config(cfg_id=args.cfg, create_dirs=not (args.render or args.iter > 0))
    cfg.update(args)
    flags.debug = args.debug

    if args.debug:
        # cfg.data_specs['file_path'] = "sample_data/amass_copycat_take3_test.pkl"
        cfg.num_threads = 1
        cfg.no_log = True

    if not args.no_log:
        wandb.init(
            project="copycat",
            resume=not args.resume is None,
            id=args.resume,
            notes=cfg.notes,
        )
        wandb.config.update(vars(cfg), allow_val_change=True)
        wandb.config.update(args, allow_val_change=True)
        wandb.run.name = args.cfg
        wandb.run.save()

    # if cfg.render:
    #     print("Rendering!!")
    #     from mujoco_py import load_model_from_path, MjSim
    #     from uhc.khrylib.rl.envs.common.mjviewer import MjViewer

    #     model = load_model_from_path(f"assets/mujoco_models/{cfg.mujoco_model_file}")
    #     sim = MjSim(model)
    #     viewer = MjViewer(sim)
    #     cfg.num_threads = 1

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=args.gpu_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    from uhc.agents.agent_copycat import AgentCopycat

    if args.show_single:
        cfg.data_specs[
            "file_path"
        ] = "sample_data/amass_copycat_take5_single.pkl"
        agent = AgentCopycat(cfg, dtype, device, training=True, checkpoint_epoch=0)
        take_key = "0-DanceDB_20140506_AnnaCortesi_AnnaCortesi_BellyDance2_C3D_poses"
        agent.load_curr()
        res = agent.eval_seq(take_key, agent.data_loader)
        print(res['succ'])
    else:
        agent = AgentCopycat(
            cfg, dtype, device, training=True, checkpoint_epoch=args.iter
        )
        agent.precision_mode = True
        agent.load_curr()
        done_keys = [k for k in os.listdir(f"{cfg.model_dir}_singles/")]

        take_keys = iter(agent.data_loader.data_keys)
        take_key = next(take_keys)
        while take_key in done_keys:
            take_key = next(take_keys)
        for epoch in range(args.iter, 99999):
            res = agent.eval_seq(take_key, agent.data_loader)
            if not res["succ"]:
                print(f"Fitting: {take_key} {res['succ']}")
                agent.fit_single_key = take_key
                agent.optimize_policy(epoch, save_model=False)
                agent.save_curr()
            else:
                print(f"************************Fitted {take_key} at {epoch}")
                agent.save_singles(epoch, take_key)
                try:
                    take_key = next(take_keys)
                    while take_key in done_keys:
                        take_key = next(take_keys)
                except StopIteration:
                    break
