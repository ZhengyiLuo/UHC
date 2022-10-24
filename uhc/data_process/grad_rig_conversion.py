from mujoco_py import load_model_from_path
from uhc.khrylib.utils import *
import joblib
from scipy.spatial.transform import Rotation as sRot

import glob
import os
import sys
import pdb
import os.path as osp
import copy

sys.path.append(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

from uhc.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, rot6d_to_rotmat, convert_orth_6d_to_mat,
    angle_axis_to_rotation_matrix, angle_axis_to_quaternion)

from uhc.smpllib.smpl_parser import SMPL_Parser
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
from mujoco_py import load_model_from_path, MjSim
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    device = (torch.device("cuda", index=0)
              if torch.cuda.is_available() else torch.device("cpu"))
    print(device)
    smpl_p = SMPL_Parser("/hdd/zen/dev/copycat/Copycat/data/smpl",
                         gender="male")
    smpl_p.to(device)

    smpl_model_file = 'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    smpl_model = load_model_from_path(smpl_model_file)
    smpl_qpose_addr = get_body_qposaddr(smpl_model)
    smpl_bone_names = list(smpl_qpose_addr.keys())
    print(smpl_bone_names)
    khry_model_file = 'assets/mujoco_models/humanoid_1205_v1.xml'
    khry_model = load_model_from_path(khry_model_file)
    khry_qpose_addr = get_body_qposaddr(khry_model)
    khry_bone_names = list(khry_qpose_addr.keys())
    print(khry_bone_names)
    smpl_sim = MjSim(smpl_model)
    khry_sim = MjSim(khry_model)

    smpl_1 = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine',
        'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax',
        'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    smpl_2 = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', "LeftLeg", "RightLeg", "Spine1", "LeftFoot", \
            "RightFoot", "Spine2", "LeftToe", "RightToe", "Neck", "LeftChest", "RightChest", "Mouth", "LeftShoulder", \
            "RightShoulder", "LeftArm", "RightArm", "LeftWrist", "RightWrist", "LeftHand", "RightHand"
            ]

    smpl_chosen_bones_1 = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        'Mouth',
        "LeftChest",
        "LeftShoulder",
        "LeftArm",
        "LeftWrist",
        "RightChest",
        "RightShoulder",
        "RightArm",
        "RightWrist",
    ]

    khry_chosen_bones = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        'Head',
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    smpl_update_dict = {smpl_2[i]: smpl_1[i] for i in range(len(smpl_1))}
    smpl_chosen_bones = []
    for i in smpl_chosen_bones_1:
        smpl_chosen_bones.append(smpl_update_dict[i])

    khry_2_smpl = {
        khry_chosen_bones[i]: smpl_chosen_bones[i]
        for i in range(len(smpl_chosen_bones))
    }
    smpl_2_khry = {
        smpl_chosen_bones[i]: khry_chosen_bones[i]
        for i in range(len(smpl_chosen_bones))
    }

    smpl_chosen_bones_1_grad = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Mouth",
        "LeftShoulder",
        "LeftArm",
        "LeftWrist",
        "RightShoulder",
        "RightArm",
        "RightWrist",
    ]

    khry_chosen_bones_grad = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Head",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
    ]

    smpl_chosen_bones_1_grad = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Mouth",
        "LeftArm",
        "LeftWrist",
        "RightArm",
        "RightWrist",
    ]

    khry_chosen_bones_grad = [
        "Hips",
        "RightFoot",
        "LeftFoot",
        "RightLeg",
        "RightUpLeg",
        "LeftLeg",
        "LeftUpLeg",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Head",
        "LeftForeArm",
        "LeftHand",
        "RightForeArm",
        "RightHand",
    ]

    smpl_chosen_bones = []
    for i in smpl_chosen_bones_1_grad:
        smpl_chosen_bones.append(smpl_update_dict[i])

    smpl_chosen_grad = [smpl_1.index(i) for i in smpl_chosen_bones]
    khry_chosen_grad = [
        khry_bone_names.index(i) for i in khry_chosen_bones_grad
    ]

    egopose_mocap = joblib.load(
        "/hdd/zen/data/ActBound/AMASS/egopose_mocap_smpl.pkl")
    relive_mocap = joblib.load(
        "/hdd/zen/data/ActBound/AMASS/relive_mocap_smpl.pkl")
    relive_mocap.update(egopose_mocap)

    relive_mocap_grad = defaultdict(dict)

    for k in tqdm(relive_mocap.keys()):
        # if not k.startswith("step"):
        # continue

        pose_aa = relive_mocap[k]['pose']
        khry_qposes = relive_mocap[k]['khry_qpos'].copy()[:, -59:]

        khry_j3ds = []
        for curr_khry_qpos in khry_qposes:
            if khry_qposes.shape[1] != 59:
                khry_sim.data.qpos[:] = curr_khry_qpos[-59:]
            else:
                khry_sim.data.qpos[:] = curr_khry_qpos

            khry_sim.forward()
            curr_khry_j3d = khry_sim.data.body_xpos[1:].copy()
            curr_khry_j3d = curr_khry_j3d[khry_chosen_grad]
            curr_khry_j3d -= curr_khry_j3d[0]
            khry_j3ds.append(curr_khry_j3d)

        khry_j3d_torch = torch.tensor(khry_j3ds).to(device)
        trans = torch.from_numpy(relive_mocap[k]['trans'].copy()).to(device)
        smpl_chosen_idx_torch = torch.from_numpy(
            np.array(smpl_chosen_grad)).to(device)
        print(f'--- {k}')

        pose_aa_torch = torch.tensor(pose_aa).float().to(device)
        seq_len = pose_aa_torch.shape[0]

        shape = torch.zeros([1, 10]).to(device)
        pose_aa_torch_new = Variable(pose_aa_torch, requires_grad=True)
        shape_new = Variable(shape, requires_grad=True)
        trans_new = Variable(trans, requires_grad=True)

        optimizer_mesh = torch.optim.SGD([shape_new], lr=1)
        optimizer_pose = torch.optim.SGD([pose_aa_torch_new], lr=5)
        # optimizer_trans = torch.optim.Adadelta([trans_new], lr=5)

        scheduler_mesh = StepLR(optimizer_mesh, step_size=1, gamma=0.9995)
        scheduler_pose = StepLR(optimizer_pose, step_size=1, gamma=0.9995)
        # scheduler_trans = StepLR(optimizer_trans, step_size=1, gamma=0.9995)
        step_size = 10
        for j in range(2000):
            # pose_aa_torch_new = Variable(pose_aa_torch_new, requires_grad=True)
            scheduler_mesh.step()
            scheduler_pose.step()
            shapes = shape_new.repeat((pose_aa_torch_new.shape[0], 1))
            shapes.retain_grad()

            verts, smpl_j3d_torch = smpl_p.get_joints_verts(pose_aa_torch_new,
                                                            th_betas=shapes)

            smpl_j3d_torch -= smpl_j3d_torch[:, 0:1].clone()
            smpl_j3d_torch = smpl_j3d_torch[:, smpl_chosen_idx_torch]
            loss = torch.norm(smpl_j3d_torch - khry_j3d_torch, dim=2).mean()

            optimizer_mesh.zero_grad()
            # optimizer_trans.zero_grad()
            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_mesh.step()
            # optimizer_trans.step()
            optimizer_pose.step()
            # pose_aa_torch_new = (pose_aa_torch_new -
            #                      pose_aa_torch_new.grad * step_size)
            if j % 100 == 0:
                print(loss.item() * 1000)


        verts_zero, joints_zero = smpl_p.get_joints_verts(
            torch.zeros_like(pose_aa_torch_new),
            th_betas=shapes,
            th_trans=torch.zeros_like(trans))
        offset = joints_zero[:, 0, :]
        # import ipdb; ipdb.set_trace()
        trans_new = trans - offset

        verts_new, joints_new = smpl_p.get_joints_verts(pose_aa_torch_new,
                                                        th_betas=shapes,
                                                        th_trans=trans_new)
        delta_z = torch.min(verts_new[0, :, 2])
        trans_new[:, 2] -= delta_z


        relive_mocap_grad[k]['pose_aa'] = pose_aa_torch_new.detach().cpu(
        ).numpy()
        relive_mocap_grad[k]['beta'] = shape_new.repeat(
            (pose_aa_torch_new.shape[0], 1)).detach().cpu().numpy()
        relive_mocap_grad[k]['khry_qpos'] = relive_mocap[k]['khry_qpos']
        relive_mocap_grad[k]['trans'] = trans_new.detach().cpu().numpy()
        obj_pose = relive_mocap[k]['obj_pose']

        if k.split("-")[0] == "sit":
            relive_mocap_grad[k]['obj_info'] = {
                "chair": {
                    "geoms": [{
                        "contype": "1",
                        "conaffinity": "1",
                        "type": "box",
                        "size": "0.209 0.165 0.2",
                        "pos": "0.0 0.0 -0.18",
                        "euler": "0 0 0",
                        "rgba": "0 0.9 0 0.5",
                        "condim": "3",
                        "mass": "100000"
                    }, {
                        "contype": "1",
                        "conaffinity": "1",
                        "type": "box",
                        "size": "0.209 0.021 0.33",
                        "pos": "0.0 -0.20 0.1",
                        "euler": "14 0 0",
                        "rgba": "0 0 0.9 0.5",
                        "condim": "3",
                        "mass": "1"
                    }],
                    "obj_pose":
                    obj_pose
                }
            }
        elif k.split("-")[0] == "push":
            relive_mocap_grad[k]['obj_info'] = {
                "box": {
                    "geoms": [{
                        "contype": "1",
                        "conaffinity": "1",
                        "type": "box",
                        "size": "0.15 0.19 0.120",
                        "pos": "0 0 -0.1",
                        "euler": "0 0 0",
                        "rgba": "0.2 0.3 0.4 1",
                        "condim": "3",
                        "mass": "1"
                    }],
                    "obj_pose":
                    obj_pose[:, :7]
                },
                "table": {
                    "geoms": [
                        {
                            "contype": "1",
                            "conaffinity": "1",
                            "type": "box",
                            "size": "0.499 0.294 0.01",
                            "pos": "0 0 -0.1",
                            "euler": "0 0 0",
                            "rgba": "0 0.9 0 1",
                            "condim": "3",
                            "mass": "1"
                        },
                        {
                            "contype": "1",
                            "conaffinity": "1",
                            "type": "cylinder",
                            "size": "0.03 0.3",
                            "pos": "-0.35 -0.25 -0.44",
                            "euler": "0 0 0",
                            "rgba": "0 0.9 0 1",
                            "condim": "3",
                            "mass": "500"
                        },
                        {
                            "contype": "1",
                            "conaffinity": "1",
                            "type": "cylinder",
                            "size": "0.03 0.3",
                            "pos": "-0.35 0.25 -0.44",
                            "euler": "0 0 0",
                            "rgba": "0 0.9 0 1",
                            "condim": "3",
                            "mass": "500"
                        },
                        {
                            "contype": "1",
                            "conaffinity": "1",
                            "type": "cylinder",
                            "size": "0.03 0.3",
                            "pos": "0.35 -0.25 -0.44",
                            "euler": "0 0 0",
                            "rgba": "0 0.9 0 1",
                            "condim": "3",
                            "mass": "500"
                        },
                        {
                            "contype": "1",
                            "conaffinity": "1",
                            "type": "cylinder",
                            "size": "0.03 0.3",
                            "pos": "0.35 0.25 -0.44",
                            "euler": "0 0 0",
                            "rgba": "0 0.9 0 1",
                            "condim": "3",
                            "mass": "500"
                        },
                    ],
                    "obj_pose":
                    obj_pose[:, 7:14]
                }
            }
        elif k.split("-")[0] == "avoid":
            relive_mocap_grad[k]['obj_info'] = {
                "Can": {
                    "geoms": [{
                        "contype": "1",
                        "conaffinity": "1",
                        "type": "cylinder",
                        "size": "0.279 0.345",
                        "pos": "-0.031 0.004 -0.345",
                        "euler": "0 0 0",
                        "rgba": "0 0.9 0 0.5",
                        "condim": "3",
                        "mass": "100000"
                    }],
                    "obj_pose":
                    obj_pose
                }
            }
        elif k.split("-")[0] == "step":
            relive_mocap_grad[k]['obj_info'] = {
                "Can": {
                    "geoms": [{
                        "contype": "1",
                        "conaffinity": "1",
                        "type": "box",
                        "size": "0.4 0.4 0.17",
                        "pos": "0 0 -0.20",
                        "euler": "0 0 0",
                        "rgba": "0 0.9 0 0.5",
                        "condim": "3",
                        "mass": "40"
                    }],
                    "obj_pose":
                    obj_pose
                }
            }

    joblib.dump(
        relive_mocap_grad,
        "/hdd/zen/data/ActBound/AMASS/kinpoly_mocap_smpl_grad_test.pkl")
    # np.sum([np.sum(np.abs(relive_mocap_grad[k]['pose'] -  relive_mocap[k]['pose'])) for k in relive_mocap.keys()])
    # joblib.dump(relive_mocap_grad, "/insert_directory_here/egopose_mocap_smpl_grad_stepsize.pkl")
