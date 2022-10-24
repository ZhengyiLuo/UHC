from smplx.lbs import vertices2joints
from smplx import SMPL as _SMPL
from smplx import SMPL
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage import gaussian_filter1d
from uhc.khrylib.utils.transformation import (
    quaternion_slerp,
    quaternion_from_euler,
    euler_from_quaternion,
)
from tqdm import tqdm
from torch.autograd import Variable
from collections import defaultdict
from uhc.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    rotation_matrix_to_angle_axis,
    rot6d_to_rotmat,
    convert_orth_6d_to_mat,
    angle_axis_to_rotation_matrix,
    angle_axis_to_quaternion,
)
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from mujoco_py import load_model_from_path
from uhc.khrylib.utils import *
import joblib
from scipy.spatial.transform import Rotation as sRot

import glob
import os
import sys
import pdb
import os.path as osp
import argparse

sys.path.append(os.getcwd())


# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script


VIBE_DATA_DIR = "/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/VIBE/data/vibe_data"

# Map joints to SMPL joints
JOINT_MAP = {
    "OP Nose": 24,
    "OP Neck": 12,
    "OP RShoulder": 17,
    "OP RElbow": 19,
    "OP RWrist": 21,
    "OP LShoulder": 16,
    "OP LElbow": 18,
    "OP LWrist": 20,
    "OP MidHip": 0,
    "OP RHip": 2,
    "OP RKnee": 5,
    "OP RAnkle": 8,
    "OP LHip": 1,
    "OP LKnee": 4,
    "OP LAnkle": 7,
    "OP REye": 25,
    "OP LEye": 26,
    "OP REar": 27,
    "OP LEar": 28,
    "OP LBigToe": 29,
    "OP LSmallToe": 30,
    "OP LHeel": 31,
    "OP RBigToe": 32,
    "OP RSmallToe": 33,
    "OP RHeel": 34,
    "Right Ankle": 8,
    "Right Knee": 5,
    "Right Hip": 45,
    "Left Hip": 46,
    "Left Knee": 4,
    "Left Ankle": 7,
    "Right Wrist": 21,
    "Right Elbow": 19,
    "Right Shoulder": 17,
    "Left Shoulder": 16,
    "Left Elbow": 18,
    "Left Wrist": 20,
    "Neck (LSP)": 47,
    "Top of Head (LSP)": 48,
    "Pelvis (MPII)": 49,
    "Thorax (MPII)": 50,
    "Spine (H36M)": 51,
    "Jaw (H36M)": 52,
    "Head (H36M)": 53,
    "Nose": 24,
    "Left Eye": 26,
    "Right Eye": 25,
    "Left Ear": 28,
    "Right Ear": 27,
}
JOINT_NAMES = [
    "OP Nose",
    "OP Neck",
    "OP RShoulder",
    "OP RElbow",
    "OP RWrist",
    "OP LShoulder",
    "OP LElbow",
    "OP LWrist",
    "OP MidHip",
    "OP RHip",
    "OP RKnee",
    "OP RAnkle",
    "OP LHip",
    "OP LKnee",
    "OP LAnkle",
    "OP REye",
    "OP LEye",
    "OP REar",
    "OP LEar",
    "OP LBigToe",
    "OP LSmallToe",
    "OP LHeel",
    "OP RBigToe",
    "OP RSmallToe",
    "OP RHeel",
    "Right Ankle",
    "Right Knee",
    "Right Hip",
    "Left Hip",
    "Left Knee",
    "Left Ankle",
    "Right Wrist",
    "Right Elbow",
    "Right Shoulder",
    "Left Shoulder",
    "Left Elbow",
    "Left Wrist",
    "Neck (LSP)",
    "Top of Head (LSP)",
    "Pelvis (MPII)",
    "Thorax (MPII)",
    "Spine (H36M)",
    "Jaw (H36M)",
    "Head (H36M)",
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(VIBE_DATA_DIR, "J_regressor_extra.npy")
SMPL_MEAN_PARAMS = osp.join(VIBE_DATA_DIR, "smpl_mean_params.npz")
SMPL_MODEL_DIR = VIBE_DATA_DIR
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            "J_regressor_extra", torch.tensor(J_regressor_extra, dtype=torch.float32)
        )
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs["get_skin"] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(
            vertices=smpl_output.vertices,
            global_orient=smpl_output.global_orient,
            body_pose=smpl_output.body_pose,
            joints=joints,
            betas=smpl_output.betas,
            full_pose=smpl_output.full_pose,
        )
        return output


def get_smpl_faces():
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces


def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(
            quat[q - 1] + quat[q], axis=0
        ):
            quat[q] = -quat[q]
    return quat


def quat_smooth_window(quats):
    quats = quat_correct(quats)
    quats = gaussian_filter1d(quats, 30, axis=0)
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    return quats


def smooth_smpl_quat_window(pose_aa, ratio=0.2):
    batch = pose_aa.shape[0]
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch, -1, 4)
    pose_quat = pose_quat[:, :, [1, 2, 3, 0]].copy()

    quats_all = []
    for i in range(pose_quat.shape[1]):
        quats = pose_quat[:, i, :].copy()
        quats_all.append(quat_smooth_window(quats))

    pose_quat_smooth = np.stack(quats_all, axis=1)[:, :, [3, 0, 1, 2]]

    pose_rot_vec = (
        sRot.from_quat(pose_quat_smooth.reshape(-1, 4))
        .as_rotvec()
        .reshape(batch, -1, 3)
    )
    return pose_rot_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train")
    args = parser.parse_args()
    data = args.data
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    from smplpytorch.pytorch.smpl_layer import SMPL_Layer

    smpl_layer = SMPL_Layer(
        center_idx=0, gender="neutral", model_root="/hdd/zen/data/SMPL/smpl_models/"
    )
    smpl_layer.to(device)
    smpl_49 = SMPL(
        "/hdd/zen/data/SMPL/smpl_models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        batch_size=1,
        create_transl=False,
    )
    smpl_49.to(device)

    J_regressor_36m = np.load(
        "/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/VIBE/data/vibe_data/J_regressor_h36m_correct.npy"
    )
    J_regressor_36m = torch.from_numpy(J_regressor_36m).to(device)

    h36m_data = joblib.load(f"/hdd/zen/data/ActBound/AMASS/h36m_{data}_60_fitted.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_test_60_fitted.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_60_fitted_grad_test.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_filtered.p")

    h36m_grad = defaultdict(dict)
    lixel_order = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    spintoh36m = [39, 28, 29, 30, 27, 26, 25, 41, 37, 43, 38, 34, 35, 36, 33, 32, 31]
    num_epochs = 5000
    # J_regressor_new = Variable(J_regressor_36m, requires_grad = True)

    h36m_grad = joblib.load(
        f"/hdd/zen/data/ActBound/AMASS/h36m_{data}_60_fitted_smpl_grad_test.pkl"
    )

    for k in tqdm(h36m_data.keys()):
        # if k in h36m_grad:
        #     print(f"{k} done")
        #     continue
        if k in h36m_grad:
            pose_aa = h36m_grad[k]["pose"].copy()
            trans = h36m_grad[k]["trans"]
            shape = (
                torch.from_numpy(
                    h36m_grad[k]["shape"].mean(axis=0)[
                        None,
                    ]
                )
                .to(device)
                .float()
            )
        else:
            pose_aa = h36m_data[k]["pose"].copy()
            trans = h36m_data[k]["trans"]
            shape = (
                torch.from_numpy(
                    h36m_data[k]["shape"].mean(axis=0)[
                        None,
                    ]
                )
                .to(device)
                .float()
            )

        pose_aa = smooth_smpl_quat_window(pose_aa).reshape(-1, 72)
        gt_kps = torch.from_numpy(h36m_data[k]["S"]).to(device).float()[:, :, :3]
        pose_aa_torch = torch.tensor(pose_aa).float().to(device)
        trans = gaussian_filter1d(trans, 5, axis=0)
        trans = torch.from_numpy(trans).to(device).float()

        print("---" + k)
        pose_aa_torch_new = Variable(pose_aa_torch, requires_grad=True)
        trans_new = Variable(trans, requires_grad=True)
        shape_new = Variable(shape, requires_grad=True)

        # optimizer_mesh = torch.optim.SGD([pose_aa_torch_new, shape_new], lr=50)
        # optimizer_trans = torch.optim.SGD([trans_new], lr=5)

        optimizer_mesh = torch.optim.Adadelta([pose_aa_torch_new, shape_new], lr=50)
        optimizer_trans = torch.optim.Adadelta([trans_new], lr=5)
        # optimizer_regressor = torch.optim.Adadelta([J_regressor_new], lr=0.00001)

        # optimizer_mesh = torch.optim.Adagrad([pose_aa_torch_new, shape_new], lr=0.05)
        # optimizer_trans = torch.optim.Adagrad([trans_new], lr=0.01)

        # optimizer_mesh = torch.optim.AdamW([pose_aa_torch_new, shape_new], lr=0.001)
        # optimizer_trans = torch.optim.AdamW([trans_new], lr=0.0005)

        scheduler_mesh = StepLR(optimizer_mesh, step_size=1, gamma=0.9995)
        scheduler_trans = StepLR(optimizer_trans, step_size=1, gamma=0.9995)
        # scheduler_regressor = StepLR(optimizer_regressor, step_size=1, gamma=0.9995)

        for i in range(num_epochs):
            scheduler_mesh.step()
            scheduler_trans.step()
            # scheduler_regressor.step()

            shapes = shape_new.repeat((pose_aa.shape[0], 1))
            shapes.retain_grad()

            verts, Jtr = smpl_layer(
                pose_aa_torch_new, th_betas=shapes, th_trans=trans_new
            )
            smpl_j3d_torch = torch.matmul(J_regressor_36m.float(), verts)[
                :, lixel_order, :
            ]

            # curr_smpl_out = smpl_49(
            #     body_pose=pose_aa_torch_new[:, 3:], global_orient=pose_aa_torch_new[:, :3], shape=shape_new, transl=trans_new)
            # verts = curr_smpl_out.vertices
            # # smpl_j3d_torch = torch.matmul(J_regressor_new.float(), verts)[:, lixel_order, :]
            # smpl_j3d_torch = torch.matmul(J_regressor_36m.float(), verts)[
            #     :, lixel_order, :]

            # loss = torch.abs(smpl_j3d_torch - gt_kps).mean()
            loss = torch.norm(smpl_j3d_torch - gt_kps, dim=2).mean()

            # print(loss_l2.item() * 1000, 'Epoch:', i,'LR:', scheduler_mesh.get_lr(), scheduler_trans.get_lr())
            if i % 100 == 0:
                print(
                    loss.item() * 1000,
                    "Epoch:",
                    i,
                    "LR:",
                    scheduler_mesh.get_lr(),
                    scheduler_trans.get_lr(),
                )
            optimizer_mesh.zero_grad()
            optimizer_trans.zero_grad()
            # optimizer_regressor.zero_grad()
            loss.backward()
            optimizer_mesh.step()
            optimizer_trans.step()
            # optimizer_regressor.step()

        # pose_aa_torch_new[:, -6:] = 0
        # pose_aa_torch_new[:, 30:36] = 0
        # pose_aa_torch_new[:, 21:27] = 0

        h36m_grad[k] = copy.deepcopy(h36m_data[k])

        h36m_grad[k]["pose"] = pose_aa_torch_new.detach().cpu().numpy()
        h36m_grad[k]["shape"] = (
            shape_new.repeat((pose_aa.shape[0], 1)).detach().cpu().numpy()
        )
        h36m_grad[k]["trans"] = trans_new.detach().cpu().numpy()

        torch.cuda.empty_cache()
        import gc

        gc.collect()

        # joblib.dump(J_regressor_new.cpu().detach().numpy(), f"/hdd/zen/data/ActBound/AMASS/h36m_regressor.pkl")
        joblib.dump(
            h36m_grad,
            f"/hdd/zen/data/ActBound/AMASS/h36m_{data}_60_fitted_smpl_grad.pkl",
        )
