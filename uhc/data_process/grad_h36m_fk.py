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
from uhc.smpllib.torch_smpl_humanoid_batch import Humanoid_Batch

sys.path.append(os.getcwd())

# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/models/hmr.py
# Adhere to their licence to use this script



def smpl_op_to_op(pred_joints2d):
    new_2d = torch.cat([     pred_joints2d[..., [1, 4], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 1:7, :], \
                             pred_joints2d[..., [7, 8, 11], :].mean(axis = -2, keepdims = True), \
                             pred_joints2d[..., 9:11, :], \
                             pred_joints2d[..., 12:, :]], \
                             axis = -2)
    return new_2d


def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(
                quat[q - 1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_smooth_window(quats):
    quats = quat_correct(quats)
    quats = gaussian_filter1d(quats, 30, axis=0)
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    return quats


def smooth_smpl_quat_window(pose_aa, ratio=0.2):
    batch = pose_aa.shape[0]
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(
        batch, -1, 4)
    pose_quat = pose_quat[:, :, [1, 2, 3, 0]].copy()

    quats_all = []
    for i in range(pose_quat.shape[1]):
        quats = pose_quat[:, i, :].copy()
        quats_all.append(quat_smooth_window(quats))

    pose_quat_smooth = np.stack(quats_all, axis=1)[:, :, [3, 0, 1, 2]]

    pose_rot_vec = (sRot.from_quat(pose_quat_smooth.reshape(
        -1, 4)).as_rotvec().reshape(batch, -1, 3))
    return pose_rot_vec

MUJOCO_2_SMPL = np.array([ 0,  1,  5,  9,  2,  6, 10,  3,  7, 11,  4,  8, 12, 14, 19, 13, 15,
       20, 16, 21, 17, 22, 18, 23])

smpl2op_map = np.array([52, 12, 17, 19, 21, 16, 18, 20,  0,  2,  5,  8,  1,  4,  7, 53, 54,
       55, 56, 57, 58, 59, 60, 61, 62], dtype=np.int32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train")
    args = parser.parse_args()
    data = args.data
    device = (torch.device("cuda", index=0)
              if torch.cuda.is_available() else torch.device("cpu"))


    h36m_data = joblib.load(f"/hdd/zen/data/ActBound/AMASS/h36m_{data}_60_fitted.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_test_60_fitted.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_60_fitted_grad_test.pkl")
    # h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_filtered.p")

    h36m_grad = defaultdict(dict)
    hb = Humanoid_Batch(data_dir=osp.join("/hdd/zen/dev/copycat/Copycat", "data/smpl"))

    num_epochs = 5000

    for k in tqdm(h36m_data.keys()):
        pose_aa = h36m_data[k]["pose"].copy()
        trans = h36m_data[k]["trans"].copy()
        shape = (torch.from_numpy(h36m_data[k]["shape"][0:1]).to(device).float())

        B = pose_aa.shape[0]

        pose_aa = smooth_smpl_quat_window(pose_aa).reshape(-1, 72)
        gt_kps = torch.from_numpy(h36m_data[k]["S"]).to(device).float()[:, :, :3]
        pose_aa_torch = torch.tensor(pose_aa).float().to(device)
        # trans = gaussian_filter1d(trans, 5, axis=0)
        trans = torch.from_numpy(trans).to(device).float()

        j3ds_14 = torch.zeros([B, 14, 3]).to(device)
        j3ds_14[:, 0] = gt_kps[:, [11, 14], :].mean(axis=1)
        j3ds_14[:, [1, 2, 3, 4, 5, 6]] = gt_kps[:, [14, 15, 16, 11, 12, 13], :]
        j3ds_14[:, [7, 8, 9, 10]] = gt_kps[:, [0, 4, 5, 6], :]
        j3ds_14[:, [11, 12, 13]] = gt_kps[:, [1, 2, 3], :]
        j3ds_12 = smpl_op_to_op(j3ds_14)

        print("---" + k)
        pose_aa_torch_new = Variable(pose_aa_torch, requires_grad=True)
        trans_new = Variable(trans, requires_grad=True)
        shape_new = Variable(shape, requires_grad=True)

        # optimizer_mesh = torch.optim.SGD([pose_aa_torch_new, shape_new], lr=50)
        # optimizer_trans = torch.optim.SGD([trans_new], lr=5)

        optimizer_mesh = torch.optim.Adadelta([pose_aa_torch_new, shape_new],lr=50)
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
            hb.update_model(shape_new, torch.tensor([0]).to(device))
            trans_off = trans_new - hb._offsets[0, 0].to(device)
            fk_res = hb.fk_batch(pose_aa_torch_new[None, ], trans_off[None, ])
            wbpos = fk_res['wbpos'].squeeze()[..., MUJOCO_2_SMPL, :][..., smpl2op_map[smpl2op_map < 22], :]
            wbpos = smpl_op_to_op(wbpos)
            diff = j3ds_12 - wbpos

            j3ds_12_local = j3ds_12 - j3ds_12[..., 7:8, :]
            wbpos_local = wbpos - wbpos[..., 7:8, :]
            diff_local = j3ds_12_local - wbpos_local

            loss_g = diff.norm(dim = -1).mean()
            loss_local = diff_local.norm(dim = -1).mean()
            loss = loss_g + loss_local

            optimizer_mesh.zero_grad()
            optimizer_trans.zero_grad()
            # optimizer_regressor.zero_grad()
            loss.backward()
            optimizer_mesh.step()
            optimizer_trans.step()

            if i % 100 == 0:
                print(
                    loss_g.item() * 1000,
                    loss_local.item() * 1000,
                    "Epoch:",
                    i,
                    "LR:",
                    scheduler_mesh.get_last_lr(),
                    scheduler_trans.get_last_lr(),
                )
            scheduler_mesh.step()
            scheduler_trans.step()
            # scheduler_regressor.step()

        trans_off = trans_new - hb._offsets[0, 0].to(device)
        h36m_grad[k] = copy.deepcopy(h36m_data[k])

        h36m_grad[k]["pose"] = pose_aa_torch_new.detach().cpu().numpy()
        h36m_grad[k]["shape"] = (shape_new.repeat((pose_aa.shape[0], 1)).detach().cpu().numpy())
        h36m_grad[k]["trans"] = trans_off.detach().cpu().numpy()

        torch.cuda.empty_cache()
        import gc
        gc.collect()

        joblib.dump(
            h36m_grad,
            f"/hdd/zen/data/ActBound/AMASS/h36m_{data}_60_fitted_smpl_grad_test.pkl",
        )
