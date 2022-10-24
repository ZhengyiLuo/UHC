import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import torch
import argparse
import numpy as np
import time
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.skeleton import Skeleton
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--template_id", type=str, default="humanoid_template")
    parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_start")
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--model", type=str, default="smpl")
    parser.add_argument("--bone_prefix", type=str, default="")
    args = parser.parse_args()

    t_s = time.time()
    template_file = "assets/mujoco_models/template/%s.xml" % args.template_id
    model_file = "assets/mujoco_models/%s.xml" % args.model_id
    skeleton = Skeleton()
    bone_prefix = args.bone_prefix

    device = torch.device("cpu")

    if args.model == "smpl":
        smpl_parser = SMPL_Parser("data/smpl", gender=args.gender)
        offset_smpl_dict, parents_dict, channels = smpl_parser.get_offsets()
    elif args.model == "smplh":
        smplh_parser = SMPLH_Parser(
            "data/smpl", gender=args.gender, create_transl=False, use_pca=False
        )
        smplh_parser.to(device)
        offset_smpl_dict, parents_dict, channels = smplh_parser.get_offsets()

    skeleton.load_from_offsets(offset_smpl_dict, parents_dict, 1, {}, channels, {})
    dt = time.time() - t_s
    print(dt)
    skeleton.write_xml(model_file, template_file, offset=np.array([0, 0, 1]))

    model = load_model_from_path(model_file)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim.data.qpos[:] = 0
    sim.data.qpos[2] = 1.0
    sim.forward()

    while args.render:
        viewer.render()
