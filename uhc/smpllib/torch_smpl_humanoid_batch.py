import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp
from uhc.utils.torch_ext import dict_to_torch

sys.path.append(os.getcwd())

from uhc.utils.torch_utils import *
from uhc.utils.transform_utils import *
from scipy.spatial.transform import Rotation as sRot
import joblib
from mujoco_py import load_model_from_path
from uhc.smpllib.smpl_mujoco import SMPLConverter, smpl_to_qpose, smpl_to_qpose_torch, SMPL_BONE_ORDER_NAMES
from uhc.smpllib.smpl_parser import SMPL_EE_NAMES
from uhc.utils.tools import get_expert, get_expert_master
import uhc.utils.pytorch3d_transforms as tR
from uhc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)


class Humanoid_Batch:
    def __init__(self, smpl_model = "smpl", data_dir="data/smpl"):
        self.smpl_model = smpl_model
        if self.smpl_model == "smpl":
            self.smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
            self.smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
            self.smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
        elif self.smpl_model == "smplh":
            self.smpl_parser_n = SMPLH_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLH_Parser(
                model_path=data_dir, gender="male", use_pca=False, create_transl=False
            )
            self.smpl_parser_f = SMPLH_Parser(
                model_path=data_dir, gender="female", use_pca=False, create_transl=False
            )
        elif self.smpl_model == "smplx":
            self.smpl_parser_n = SMPLX_Parser(
                model_path=data_dir,
                gender="neutral",
                use_pca=False,
                create_transl=False,
            )
            self.smpl_parser_m = SMPLX_Parser(
                model_path=data_dir, gender="male", use_pca=False, create_transl=False
            )
            self.smpl_parser_f = SMPLX_Parser(
                model_path=data_dir, gender="female", use_pca=False, create_transl=False
            )

        self.model_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        self._parents = [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
        self.smpl_index = [SMPL_BONE_ORDER_NAMES.index(i) for i in self.model_names]

    def update_model(self, betas, gender):
        betas, gender = betas.cpu().float(), gender.cpu().long()
        B, _ = betas.shape
        betas_f = betas[gender == 2]
        if len(betas_f) > 0:
            _, _, _, _, joint_offsets_f, _, _, _, _, _, _, = self.smpl_parser_f.get_mesh_offsets_batch(betas=betas_f[:, :10])

        betas_n = betas[gender == 0]
        if len(betas_n) > 0:
            _, _, _, _, joint_offsets_n, _, _, _, _, _, _, = self.smpl_parser_n.get_mesh_offsets_batch(betas=betas_n[:, :10])
        
        betas_m = betas[gender == 1]
        if len(betas_m) > 0:
            _, _, _, _, joint_offsets_m, _, _, _, _, _, _, = self.smpl_parser_m.get_mesh_offsets_batch(betas=betas_m[:, :10])

        joint_offsets_all = dict()
        for n in SMPL_BONE_ORDER_NAMES:
            joint_offsets_all[n] = torch.zeros([B, 3]).float()
            if len(betas_f) > 0: joint_offsets_all[n][gender == 2] = joint_offsets_f[n]
            if len(betas_n) > 0: joint_offsets_all[n][gender == 0] = joint_offsets_n[n]
            if len(betas_m) > 0: joint_offsets_all[n][gender == 1] = joint_offsets_m[n]

        off_sets = []
        for n in self.model_names:
            off_sets.append(joint_offsets_all[n])

        # self._offsets = torch.from_numpy(np.stack(off_sets, axis=1))
        self._offsets = torch.from_numpy(np.round(np.stack(off_sets, axis=1), decimals = 5))
        self.trans2joint = -self._offsets[:, 0:1]
        self.trans2joint[:, :, 2] = 0
        # self._offsets = joblib.load("curr_offset.pkl")[None, ]
 

    def fk_batch(self, pose, trans, convert_to_mat = True, count_offset = True):
        device, dtype = pose.device, pose.dtype
        B, seq_len = pose.shape[:2]
        if convert_to_mat:
            pose_mat = tR.axis_angle_to_matrix(pose.reshape(B * seq_len, -1, 3)).reshape(B, seq_len, -1, 3, 3)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:    
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root

        if count_offset:
            trans = trans + self._offsets[:, 0:1].to(device)

        pose_mat_ordered  = pose_mat[:, :, self.smpl_index]
        
        wbody_pos, wbody_mat = self.forward_kinematics_batch(
            pose_mat_ordered[:, :, 1:], pose_mat_ordered[:, :, 0:1], trans
        )
        return_dic = {}
        return_dic["wbpos"] = wbody_pos
        return_dic["wbmat"] = wbody_mat
        
        return return_dic

    def get_ee_pos(self, body_xpos, root_q, transform):
        ee_name = SMPL_EE_NAMES
        ee_pos = []
        root_pos = body_xpos[:, 0, :]
        for name in ee_name:
            bone_id = self.model._body_name2id[name] - 1
            bone_vec = body_xpos[:, bone_id]

            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec_batch(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)

        return torch.swapaxes(torch.stack(ee_pos, dim=0), 0, 1)

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []
        expanded_offsets = (
            self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype)
        )
        
        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    torch.matmul(
                        rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]
                    ).squeeze(-1) + positions_world[self._parents[i]]
                )

                rot_mat = torch.matmul(
                    rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :]
                )

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
   

if __name__ == "__main__":
    import mujoco_py
    from uhc.smpllib.smpl_robot import Robot
    from uhc.smpllib.torch_smpl_humanoid import Humanoid
    from uhc.utils.config_utils.copycat_config import Config
    from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
    from uhc.utils.torch_ext import dict_to_torch
    from uhc.smpllib.smpl_mujoco import smpl_to_qpose_torch, smplh_to_smpl
    torch.manual_seed(0)

    cfg = Config(
        cfg_id="copycat_44",
        create_dirs=False,
    )

    smpl_robot = Robot(
            cfg.robot_cfg,
            data_dir=osp.join(cfg.base_dir, "data/smpl"),
            masterfoot=False,
        )
    dataset = DatasetAMASSSingle(cfg.data_specs, "test")
    humanoid_batch = Humanoid_Batch()

    data_test = dataset.sample_seq()
    data_test = dict_to_torch(data_test)
    pose_aa = data_test['pose_aa']
    trans = data_test['trans']
    beta = data_test['beta']
    gender = data_test['gender']
    count_offset = True
    
    smpl_robot.load_from_skeleton(beta[0:1, :].cpu().float(), gender=gender, objs_info=None)
    model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
    humanoid = Humanoid(model = model)
    qpos = smpl_to_qpose_torch(pose_aa, mj_model = model, trans = trans, count_offset=count_offset)
    fk_res = humanoid.qpos_fk(qpos)
    
    
    pose_aa_smpl = smplh_to_smpl(pose_aa)
    humanoid_batch.update_model(beta[0:1], gender[0:1])
    return_dict = humanoid_batch.fk_batch(pose_aa_smpl[None, ], trans[None, ], count_offset=count_offset)
    diff = return_dict['wbpos'].reshape(-1, 24, 3) - fk_res['wbpos'].reshape(-1, 24, 3)
    print(diff.abs().sum())
    import ipdb; ipdb.set_trace()

