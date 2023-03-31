import os
import sys
import pdb

sys.path.append(os.getcwd())

import numpy as np
import glob
import pickle as pk
import joblib
import torch
import argparse

from tqdm import tqdm
from uhc.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    rotation_matrix_to_angle_axis,
    rot6d_to_rotmat,
)
from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Viewer
from mujoco_py import load_model_from_path, MjSim
from uhc.utils.config_utils.copycat_config import Config
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.utils.tools import get_expert
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.smpllib.smpl_parser import SMPLH_Parser
from uhc.utils.flags import flags


def count_consec(lst):
    consec = [1]
    for x, y in zip(lst, lst[1:]):
        if x == y - 1:
            consec[-1] += 1
        else:
            consec.append(1)
    return consec


def fix_height_smpl(pose_aa, th_trans, th_betas, gender, seq_name):
    with torch.no_grad():
        offset = 0.03
        gender = gender.item() if isinstance(gender, np.ndarray) else gender
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")

        if gender == "neutral":
            smpl_parser = smpl_parser_n
        elif gender == "male":
            smpl_parser = smpl_parser_m
        elif gender == "female":
            smpl_parser = smpl_parser_f
        else:
            print(gender)
            raise Exception("Gender Not Supported!!")

        batch_size = pose_aa.shape[0]
        verts, jts = smpl_parser.get_joints_verts(pose_aa[0:1],
                                                th_betas.repeat((1, 1)),
                                                th_trans=th_trans[0:1])

        # vertices = verts[0].numpy()
        zs = verts[:, :, 2]
        if torch.sum(zs < 0) > 0:
            gp = torch.mean(zs[zs < 0])
        else:
            gp = zs.min()

        if gp > 0.1:
            print(f"Starting too high: {seq_name}")
            return None

        th_trans[:, 2] -= gp
        
        
        # verts, jts = smpl_parser.get_joints_verts(pose_aa,
        #                                         th_betas.repeat((batch_size, 1)),
        #                                         th_trans=th_trans)

        # conseq = count_consec(
        #     torch.nonzero(torch.sum(jts[:, [10, 11], 2] > 0.2, axis=1) > 1))
        # if np.max(conseq) > 30:
        #     ## Too high
        #     print(
        #         f"{seq_name} too high sequence invalid for copycat: {np.max(conseq)}"
        #     )
        #     return None
        return th_trans


if __name__ == "__main__":
    np.random.seed(0)
    amass_base = "sample_data/"
    take_num = "copycat_take5"
    amass_seq_data = {}
    seq_length = -1
    cfg = Config(cfg_id="copycat_30", create_dirs=False)

    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    np.random.seed(0)
    smpl_parser_n = SMPLH_Parser(model_path="data/smpl",
                                 gender="neutral",
                                 use_pca=False,
                                 create_transl=False)
    smpl_parser_m = SMPLH_Parser(model_path="data/smpl",
                                 gender="male",
                                 use_pca=False,
                                 create_transl=False)
    smpl_parser_f = SMPLH_Parser(model_path="data/smpl",
                                 gender="female",
                                 use_pca=False,
                                 create_transl=False)
    data_all = joblib.load("sample_data/h36m_all_sit_30_qpos.pkl")
    # data_all = joblib.load("sample_data/kinpoly_mocap_smpl_grad_test.pkl")

    for k, v in tqdm(data_all.items()):
        

        new_trans = fix_height_smpl(torch.cat([torch.from_numpy(v["pose_aa"]), torch.zeros([v["pose_aa"].shape[0], 84])], dim = 1),
                        torch.from_numpy(v["trans"]),
                        torch.cat([torch.from_numpy(v['beta']), torch.zeros(6)]),
                        v['gender'],
                        # "neutral", 
                        k)
        v['trans']  = new_trans.cpu().numpy()
    joblib.dump(data_all, "sample_data/h36m_all_sit_30_qpos_height.pkl")
    # joblib.dump(data_all, "sample_data/kinpoly_mocap_smpl_grad_height.pkl")