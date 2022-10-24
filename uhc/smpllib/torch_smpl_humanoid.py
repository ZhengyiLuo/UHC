import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from uhc.utils.torch_utils import *
from uhc.utils.transform_utils import *
from scipy.spatial.transform import Rotation as sRot
import joblib
from mujoco_py import load_model_from_path
from uhc.smpllib.smpl_mujoco import SMPLConverter
from uhc.smpllib.smpl_parser import SMPL_EE_NAMES
from uhc.utils.tools import get_expert, get_expert_master


class Humanoid:
    def __init__(self, model_file=None, model=None):
        self.model = model
        if model_file is not None:
            self.model = load_model_from_path(model_file)  # load mujoco model
        elif model is not None:
            pass

        self.update_model(self.model)

    def update_model(self, model):
        self.body_name = model.body_names[1:]
        offsets = model.body_pos[1:len(self.body_name) + 1]
        i_offsets = model.body_ipos[1:len(self.body_name) + 1]
        parents = model.body_parentid[1:len(self.body_name) + 1] - 1
        parents[0] = -1

        """global coordinate to local link coordinate"""
        # offsets = np.round(offsets, 3)
        self._offsets = torch.tensor(offsets)
        self._i_offsets = torch.tensor(i_offsets)
        self._parents = np.array(parents)

        self.map_length = 0.6
        self.voxel_num = 32

        self._compute_metadata()
        # import ipdb; ipdb.set_trace()

    def get_head_idx(self):
        return self.model._body_name2id["Head"] - 1

    def to4x4(self, rotmat):
        device, dtype = rotmat.device, rotmat.dtype
        obj_rot_tmp = torch.eye(4).to(device).type(dtype)
        obj_rot_tmp[:3, :3] = rotmat
        obj_rot = obj_rot_tmp.clone()
        return obj_rot

    def get_body_occup_map(self, qpos, body_name):
        """
        Input qpos: 1 x J (object pos, quat, root pos, root orientation, joint orientation)
        Output occupancy map: J x Vx x Vy x Vz x 1
        """
        device, dtype = qpos.deivce, qpos.dtype
        tot_num = self.voxel_num * self.voxel_num * self.voxel_num
        grid = self.base_grid.clone()

        occup_grid_size = (
            len(body_name),
            self.voxel_num,
            self.voxel_num,
            self.voxel_num,
        )

        root_rot = qpos[3:7]
        root_pos = qpos[:3]
        joint_rot = qpos[7:]

        body_pos, body_quat = self.qpos_fk(
            torch.unsqueeze(qpos, dim=0), select_joints=body_name
        )
        body_pos = body_pos.view(-1, 3)
        body_quat = body_quat.view(-1, 4)

        ## World -> object root transformation
        obj_pos = qpos[:3]
        obj_quat = qpos[3:7]
        obj_rot = self.to4x4(quaternion_matrix(obj_quat).t())

        obj_rot[:3, 3] = -obj_pos

        ## Body -> World transformation
        body_rot = torch.stack(
            [
                self.to4x4(quaternion_matrix(get_heading_q(b_quat)))
                for b_quat in body_quat
            ],
            dim=0,
        )
        body_rot[:, :3, 3] = body_pos

        ## Body -> object root transformation
        body_trans = torch.einsum(
            "bij, bjk -> bik",
            torch.repeat_interleave(
                torch.unsqueeze(obj_rot, dim=0), len(body_name), dim=0
            ),
            body_rot,
        )

        ## Object root -> object part transformation
        obj_loc_trans = torch.stack(
            [self.to4x4(quaternion_matrix(quat).T) for quat in self.obj_loc_quat], dim=0
        )
        obj_loc_trans[:, :3, 3] = -self.obj_loc_pos

        ## Body -> object part transformation
        total_trans = torch.einsum(
            "blij, bljk -> blik",
            torch.repeat_interleave(
                torch.unsqueeze(obj_loc_trans, dim=1), len(body_name), dim=1
            ),
            torch.repeat_interleave(
                torch.unsqueeze(body_trans, dim=0), self.obj_geom_num, dim=0
            ),
        )

        grid_h = torch.cat(
            (
                grid.view(tot_num, 3).t(),
                torch.ones(1, tot_num, dtype=dtype, device=device),
            ),
            dim=0,
        )
        trans_grid = torch.einsum("bkij,jl->bkil", total_trans, grid_h)[
            :, :, :-1, :
        ]  ## object part, body num, xyz1, point num
        obj_sizes = torch.repeat_interleave(
            torch.repeat_interleave(
                torch.unsqueeze(torch.unsqueeze(self.obj_sizes, dim=1), dim=-1),
                len(body_name),
                dim=1,
            ),
            tot_num,
            dim=3,
        )
        cond = torch.abs(trans_grid) < obj_sizes / 2.0
        occup_grid_batch = (
            cond.all(dim=2).any(dim=0).view(occup_grid_size).type(dtype).to(device)
        )

        return occup_grid_batch.unsqueeze(-1)

    def qpos_fk(self, qpos, select_joints=None, fix_quat=False, to_numpy=True, full_return = True):
        """
        qpos: body representation (1->3: root position, 3->7: root orientation, 7->end: joint orientation)
        Rotations are represented in euler angles.
        Note that some joints have 1 DoF.
        B = batch size, J = number of joints
        Input: rotations (B, L), root_rotations (B, 4), root_positions (B, 3),
        Output: (B, J, 3) J is number of joints or selected joints
        """
        device, dtype = qpos.device, qpos.dtype
        qpos = qpos.clone()
        batch_size = qpos.shape[0]
        rotations = qpos[:, 7:]
        root_rotations = qpos[:, 3:7]
        root_positions = qpos[:, :3]

        if fix_quat:
            new_rot_norm = torch.norm(root_rotations, dim=1).view(-1, 1)
            root_rotations = root_rotations / new_rot_norm
            qpos = torch.cat(
                [qpos[:, :3], root_rotations, qpos[:, 7:]], dim=1
            )  # had to do this for gradient flow

        assert rotations.size()[0] == root_positions.size()[0]
        B = rotations.size()[0]
        J = len(self.body_name) - 1  # Exclude root

        # padded_rot = self._rotation_padding(rotations)
        padded_rot = rotations.reshape(-1, 3)
        body_quats = quaternion_from_euler(
            padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes="rzyx"
        )
        body_quats = body_quats.reshape(B, J, 4)
        wbody_pos, body_com, wbody_quat = self.forward_kinematics_batch(
            body_quats, root_rotations, root_positions
        )
        if full_return:
            """ Get sim full vals """
            body_quats_full = quaternion_from_euler(
                padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes="rzyx"
            )


            body_quats_full = torch.cat(
                (root_rotations[:, None, :], body_quats_full.reshape(B, J, 4)), dim=1
            )

            if len(qpos) > 1:
                qvel = get_qvel_fd_batch(qpos[:-1], qpos[1:], 1 / 30)
            else:
                qvel = torch.zeros((0, 75)).to(qpos)

            qvel = torch.cat((qvel[0:1], qvel), dim=0)  # padding the first qvel
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:, :3].clone()
            rlinv_local = transform_vec_batch(qvel[:, :3], qpos[:, 3:7])
            rangv = qvel[:, 3:6].clone()

            if len(qpos) > 1:
                bangvel = get_angvel_fd_batch(body_quats_full[:-1], body_quats_full[1:], 1 / 30)
            else:
                bangvel = torch.zeros((0, 96)).to(qpos)

            bangvel = torch.cat((bangvel[0:1], bangvel), dim=0)
            ee_wpos = self.get_ee_pos(wbody_pos, root_rotations, transform=None)
            ee_pos = self.get_ee_pos(wbody_pos, root_rotations, transform="root")

            if select_joints is None:
                pass
            else:
                wbody_pos_list = []
                wbody_quat_list = []
                for joint in select_joints:
                    jidx = self.body_name.index(joint)
                    wbody_pos_list.append(wbody_pos[:, jidx, :])
                    wbody_quat_list.append(wbody_quat[:, jidx, :])
                wbody_pos = torch.cat(wbody_pos_list).view(B, len(select_joints), 3)
                wbody_quat = torch.cat(wbody_quat_list).view(B, len(select_joints), 4)

            return_dic = {
                "qpos": qpos,
                "qvel": qvel,
                "wbpos": wbody_pos.reshape(batch_size, -1),
                "wbquat": wbody_quat.reshape(batch_size, -1),
                "bquat": body_quats_full.reshape(batch_size, -1),
                "body_com": body_com.reshape(batch_size, -1),
                "rlinv": rlinv.reshape(batch_size, -1),
                "rlinv_local": rlinv_local.reshape(batch_size, -1),
                "rangv": rangv.reshape(batch_size, -1),
                "bangvel": bangvel.reshape(batch_size, -1),
                "ee_wpos": ee_wpos.reshape(batch_size, -1),
                "ee_pos": ee_pos.reshape(batch_size, -1),
                "com": body_com[:, 0].reshape(batch_size, -1),
            }

            return_dic["height_lb"] = return_dic["qpos"][:, 2].min()
        else:
            return_dic = {
                "qpos": qpos,
                "wbpos": wbody_pos.reshape(batch_size, -1),
                "wbquat": wbody_quat.reshape(batch_size, -1),
            }

        if to_numpy:
            return_dic = {k: v.cpu().numpy() for k, v in return_dic.items()}
        return_dic["len"] = return_dic["qpos"].shape[0]
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

    def forward_kinematics(self, rotations, root_rotations, root_positions):
        positions_world = []
        rotations_world = []
        device, dtype = root_rotations.device, root_rotations.dtype
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:

                positions_world.append(
                    quat_mul_vec(rotations_world[self._parents[i]], self._offsets[i])
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[i]
                        )
                    )
                else:
                    rotations_world.append(None)

        return torch.stack(positions_world), torch.stack(rotations_world)

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        assert len(rotations.shape) == 3
        assert rotations.size()[-1] == 4

        device, dtype = root_rotations.device, root_rotations.dtype
        B = rotations.size()[0]
        J = self._offsets.shape[0]
        positions_world = []
        positions_world_com = []
        rotations_world = []
        expanded_offsets = (
            self._offsets.expand(B, J, self._offsets.shape[1]).to(device).type(dtype)
        )
        expanded_i_offsets = (
            self._i_offsets.expand(B, J, self._i_offsets.shape[1])
            .to(device)
            .type(dtype)
        )

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                positions_world_com.append(
                    root_positions
                    + quat_mul_vec_batch(root_rotations, expanded_i_offsets[:, 0, :])
                )
                rotations_world.append(root_rotations)
            else:
                jpos = (
                    quat_mul_vec_batch(
                        rotations_world[self._parents[i]], expanded_offsets[:, i, :]
                    )
                    + positions_world[self._parents[i]]
                )

                jquat = quaternion_multiply_batch(
                    rotations_world[self._parents[i]], rotations[:, i - 1, :]
                )

                positions_world.append(jpos)
                rotations_world.append(jquat)
                positions_world_com.append(
                    quat_mul_vec_batch(jquat, expanded_i_offsets[:, i, :]) + jpos
                )

        positions_world = torch.stack(positions_world, dim=2)
        positions_world_com = torch.stack(positions_world_com, dim=2)
        rotations_world = torch.stack(rotations_world, dim=2)
        return (
            positions_world.permute(0, 2, 1),
            positions_world_com.permute(0, 2, 1),
            rotations_world.permute(0, 2, 1),
        )

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def qpos_2_6d(self, qpos):
        rotations = qpos[:, 7:]
        root_rotations = qpos[:, 3:7]
        root_positions = qpos[:, :3]

        assert rotations.size()[0] == root_positions.size()[0]
        B = rotations.size()[0]
        J = len(self.body_name) - 1  # Exclude root

        padded_rot = self._rotation_padding(rotations)
        quats = quaternion_from_euler(
            padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes="rxyz"
        )

        quats = quats.reshape(B, J, 4)

        rot_6d = convert_quat_to_6d(quats)

        return qpos[:, :7], rot_6d

    def qpos_from_6d(self, orth6d):
        B, J, _ = orth6d.shape
        quats = convert_6d_to_quat(orth6d)

        ## This will no longer be differentiable
        quat_numpy = quats.cpu().numpy()
        quat_numpy_flat = quat_numpy.reshape(-1, 4)[:, [1, 2, 3, 0]]

        euler_numpy = sRot.from_quat(quat_numpy_flat).as_euler("XYZ").reshape(B, J, -1)
        qpos_numpy = euler_numpy.reshape(B, -1)[:, self.qpos_pad_indices]
        return qpos_numpy

    def qpos_from_euler(self, euler_angles):
        B, J, _ = euler_angles.shape
        qpos = euler_angles.reshape(B, -1)[:, self.qpos_pad_indices]
        return qpos


def eval_expert(expert_qpos):
    wbody_pos_mujoco = []
    wbody_quat_mujoco = []
    body_com_mujoco = []
    body_quat_mujoco = []

    occup_map = []
    import time

    t1 = time.time()

    expert = get_expert(expert_qpos, {"cyclic": False, "seq_name": "test"}, env)

    t2 = time.time()
    print("======================================================")
    print("Env:", t2 - t1)
    ## Singles
    from collections import defaultdict

    fk_res = defaultdict(list)
    # [
    #     [
    #         fk_res[k].append(v)
    #         for k, v in env_humanoid.qpos_fk(torch.tensor(qpos)).items()
    #     ]
    #     for qpos in expert_qpos
    # ]
    t3 = time.time()
    # fk_res = {k: np.stack(v) for k, v in fk_res.items()}
    print("Seq fk:", t3 - t2)
    ## Batch
    fk_res = env_humanoid.qpos_fk(torch.from_numpy(expert_qpos))


    t4 = time.time()
    print("Batch Fk fk:", t4 - t3)
    for k, v in expert.items():
        if k in fk_res:
            print(
                k,
                np.sum(np.abs(expert[k] - fk_res[k])),
            )

    # assert np.array_equal(np.round(joint_pos_mujoco, 6), np.round(joint_pos_torch.cpu().numpy(), 6)), "Joint position error"
    # assert np.array_equal(np.round(joint_quat_mujoco, 6), np.round(joint_quat_torch.cpu().numpy(), 6)), "Joint orientation error"


if __name__ == "__main__":
    import argparse

    # from sceneplus.data_loaders.statereg_dataset import Dataset
    # from sceneplus.envs.humanoid_v2 import HumanoidEnv
    # from sceneplus.utils.egomimic_config import Config as EgoConfig

    model_id = "1205"

    action = "sit"
    from uhc.envs.humanoid_im import HumanoidEnv
    from uhc.utils.config_utils.copycat_config import Config
    from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle

    cfg = Config(
        cfg_id="copycat_30",
        create_dirs=False,
    )
    cfg.mujoco_model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml"
    # cfg.mujoco_model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_masterfoot.xml"
    smpl_model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml"
    # master_foot_model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_masterfoot.xml"
    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    cfg.masterfoot = False
    env = HumanoidEnv(
        cfg, init_expert=init_expert, data_specs=cfg.data_specs, mode="test"
    )

    dtype = torch.float64
    device = torch.device("cpu")

    # env_humanoid = Humanoid(model_file=cfg.mujoco_model_file)
    env_humanoid = Humanoid(model=env.model)
    smpl_model = load_model_from_path(smpl_model_file)
    # sim_model = load_model_from_path(master_foot_model_file)
    sim_model = load_model_from_path(smpl_model_file)
    converter = SMPLConverter(smpl_model, sim_model)
    # # occup_joints = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Hips']

    data_load = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_qpos_grad.pkl")
    for k in data_load.keys():
        expert_qpos = data_load[k]["qpos"]
        eval_expert(converter.qpos_smpl_2_new(expert_qpos))
