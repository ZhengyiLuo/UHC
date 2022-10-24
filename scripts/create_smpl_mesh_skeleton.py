import os
import sys
import numpy as np

from uhc.data_process.amass_to_qpos import sample_seq_length

sys.path.append(os.getcwd())

import torch
import pickle
import yaml

from collections import defaultdict
from scipy.spatial import ConvexHull
from stl import mesh

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.skeleton_mesh import Skeleton
from uhc.utils.vis_model_utils import create_vis_model
from uhc.utils.geom import quadric_mesh_decimation, center_scale_mesh
from uhc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
from uhc.smpllib.smpl_robot import get_joint_geometries

# from smplpytorch.pytorch.smpl_layer import SMPL_Layer
# from h36m.utils.h36m_global import *


def create_skeleton(gender="neutral"):

    torch.set_grad_enabled(False)

    template_file = "assets/mujoco_models/template/humanoid_template.xml"
    model_dir = f"assets/mujoco_models/"
    model_file = f"{model_dir}/humanoid_smpl_{gender}_mesh_test.xml"
    vis_model_file = f"{model_dir}/humanoid_smpl_{gender}_mesh_vis.xml"
    os.makedirs(model_dir, exist_ok=True)

    # smpl_layer = SMPL_Layer(
    #     center_idx=0,
    #     gender='neutral',
    #     model_root='smplpytorch/native/models')

    # Load LBS template mesh
    # subject_data = pickle.load(open(f'{h36m_out_folder}/poses/S{subject_id}.pkl', 'rb'))
    # mean_shape = torch.tensor(subject_data['mean_shape']).unsqueeze(0).float()
    # mean_shape = torch.tensor(subject_data['action_seq'][0]['shape'][0]).unsqueeze(0).float()
    zero_pose = torch.zeros(1, 72).float()
    mean_shape = torch.zeros(1, 10).float()
    smpl_parser = SMPL_Parser(model_path="data/smpl/", gender=gender)

    verts, Jtr = smpl_parser.get_joints_verts(
        zero_pose, th_betas=mean_shape, th_trans=torch.zeros(1, 3)
    )

    joint_names = SMPL_BONE_ORDER_NAMES
    h36m_joint_parents = smpl_parser.parents.cpu().numpy()

    joint_pos = Jtr[0].numpy()
    joint_offsets = {
        joint_names[c]: (joint_pos[c] - joint_pos[p]) if c > 0 else joint_pos[c]
        for c, p in enumerate(h36m_joint_parents)
    }
    joint_parents = {
        x: joint_names[i] if i >= 0 else None
        for x, i in zip(joint_names, h36m_joint_parents)
    }
    joint_axes = {x: np.identity(3) for x in joint_names}
    joint_dofs = {x: ["z", "y", "x"] for x in joint_names}
    joint_range = {
        x: np.hstack([np.ones([3, 1]) * -np.pi, np.ones([3, 1]) * np.pi])
        for x in joint_names
    }
    joint_range["L_Elbow"] *= 4
    joint_range["R_Elbow"] *= 4
    contype = {0: joint_names}
    conaffinity = {1: joint_names}

    verts = verts[0].numpy()
    # skin_weights = smpl_layer.th_weights.numpy()
    skin_weights = smpl_parser.lbs_weights.numpy()
    get_joint_geometries(
        verts, joint_pos, skin_weights, joint_names, geom_dir=f"{model_dir}/geom"
    )

    skeleton = Skeleton(model_dir)
    skeleton.load_from_offsets(
        joint_offsets,
        joint_parents,
        joint_axes,
        joint_dofs,
        joint_range,
        sites={},
        scale=1,
        equalities={},
        collision_groups=contype,
        conaffinity=conaffinity,
        simple_geom=False,
    )
    # save skeleton
    skeleton.write_xml(model_file, template_file, offset=np.array([0, 0, 0]))
    print(model_file)
    # create vis model
    create_vis_model(model_file, vis_model_file, num=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    import time

    t_s = time.time()
    create_skeleton()
    dt = time.time() - t_s
    print(dt)

    model_dir = f"assets/mujoco_models"
    model_file = f"{model_dir}/humanoid_smpl_neutral_mesh_test.xml"
    model = load_model_from_path(model_file)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    print(f"qpos dim: {sim.data.qpos.shape}")
    sim.data.qpos[:] = 0
    sim.data.qpos[2] = 1.0
    sim.forward()

    while args.render:
        viewer.render()
