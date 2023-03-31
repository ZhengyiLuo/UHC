import torch
import numpy as np
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.utils.transform_utils import *
from uhc.utils.transformation import (
    quaternion_from_euler_batch,
    quaternion_multiply_batch,
    quat_mul_vec,
    quat_mul_vec_batch,
    quaternion_from_euler,
    quaternion_multiply,
    quaternion_matrix,
)

from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_mujoco import SMPLConverter
import joblib


class Humanoid:
    def __init__(self, model_file):
        print(model_file)
        self.model = load_model_from_path(model_file)  # load mujoco model

        offsets = self.model.body_pos[1:]  # exclude world and chair
        i_offsets = self.model.body_ipos[1:]  # exclude world and chair
        parents = self.model.body_parentid[1:] - 1
        parents[0] = -1

        """global coordinate to local link coordinate"""
        self._offsets = offsets
        self._i_offsets = i_offsets
        self._parents = np.array(parents)
        self.body_name = self.model.body_names[1:]

        # chair offset

        self.map_length = 0.6
        self.voxel_num = 32

        self._compute_metadata()
        # self._set_local_map()
        # self._set_rotpad_indices()
        # self._set_qpos_padding()

    def get_head_idx(self):
        return self.model._body_name2id["Head"] - 1

    def _set_local_map(self):

        x = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        y = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        z = np.linspace(-self.map_length / 2.0, self.map_length / 2.0, self.voxel_num)
        X, Y, Z = np.meshgrid(x, y, z)
        self.base_grid = np.concatenate(
            (X[:, :, :, np.newaxis], Y[:, :, :, np.newaxis], Z[:, :, :, np.newaxis]),
            axis=3,
        )

        obj_idx = self.model.body_names.index(self.obj_names[-1])  # ZL: not cool
        g_ids = [i for i, x in enumerate(self.model.geom_bodyid) if x == obj_idx]

        self.obj_geom_num = len(g_ids)
        self.obj_sizes = self.model.geom_size[g_ids]
        self.obj_loc_pos = self.model.geom_pos[g_ids]
        self.obj_loc_quat = self.model.geom_quat[g_ids]

    def to4x4(self, rotmat):
        obj_rot_tmp = np.eye(4)
        obj_rot_tmp[:3, :3] = rotmat
        obj_rot = obj_rot_tmp.copy()
        return obj_rot

    def get_body_occup_map(self, qpos, body_name):
        """
        Input qpos: 1 x J (object pos, quat, root pos, root orientation, joint orientation)
        Output occupancy map: J x Vx x Vy x Vz x 1
        """
        tot_num = self.voxel_num * self.voxel_num * self.voxel_num
        grid = self.base_grid.copy()

        occup_grid_size = (
            len(body_name),
            self.voxel_num,
            self.voxel_num,
            self.voxel_num,
        )

        root_rot = qpos[3:7]
        root_pos = qpos[:3]
        joint_rot = qpos[7:]

        body_pos, body_quat = self.qpos_fk_batch(
            torch.unsqueeze(qpos, axis=0), select_joints=body_name
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
            axis=0,
        )
        body_rot[:, :3, 3] = body_pos

        ## Body -> object root transformation
        body_trans = torch.einsum(
            "bij, bjk -> bik",
            torch.repeat_interleave(
                torch.unsqueeze(obj_rot, axis=0), len(body_name), axis=0
            ),
            body_rot,
        )

        ## Object root -> object part transformation
        obj_loc_trans = torch.stack(
            [self.to4x4(quaternion_matrix(quat).T) for quat in self.obj_loc_quat],
            axis=0,
        )
        obj_loc_trans[:, :3, 3] = -self.obj_loc_pos

        ## Body -> object part transformation
        total_trans = torch.einsum(
            "blij, bljk -> blik",
            torch.repeat_interleave(
                torch.unsqueeze(obj_loc_trans, axis=1), len(body_name), axis=1
            ),
            torch.repeat_interleave(
                torch.unsqueeze(body_trans, axis=0), self.obj_geom_num, axis=0
            ),
        )

        grid_h = torch.cat(
            (
                grid.view(tot_num, 3).t(),
                torch.ones(1, tot_num, dtype=self.dtype, device=self.device),
            ),
            axis=0,
        )
        trans_grid = torch.einsum("bkij,jl->bkil", total_trans, grid_h)[
            :, :, :-1, :
        ]  ## object part, body num, xyz1, point num
        obj_sizes = torch.repeat_interleave(
            torch.repeat_interleave(
                torch.unsqueeze(torch.unsqueeze(self.obj_sizes, axis=1), axis=-1),
                len(body_name),
                axis=1,
            ),
            tot_num,
            axis=3,
        )
        cond = torch.abs(trans_grid) < obj_sizes / 2.0
        occup_grid_batch = (
            cond.all(axis=2)
            .any(axis=0)
            .view(occup_grid_size)
            .type(self.dtype)
            .to(self.device)
        )

        return occup_grid_batch.unsqueeze(-1)

    def qpos_fk_batch(self, qpos, select_joints=None, fix_quat=True):
        """
        qpos: body representation (1->3: root position, 3->7: root orientation, 7->end: joint orientation)
        Rotations are represented in euler angles.
        Note that some joints have 1 DoF.
        B = batch size, J = number of joints
        Input: rotations (B, L), root_rotations (B, 4), root_positions (B, 3),
        Output: (B, J, 3) J is number of joints or selected joints
        """
        qpos = qpos.copy()
        rotations = qpos[:, 7:]
        root_rotations = qpos[:, 3:7]
        root_positions = qpos[:, :3]

        if fix_quat:
            root_rotations /= np.linalg.norm(root_rotations, axis=1)[:, None]
            qpos[:, 3:7] = root_rotations

        assert rotations.shape[0] == root_positions.shape[0]
        B = rotations.shape[0]
        J = len(self.body_name) - 1  # Exclude root

        # padded_rot = self._rotation_padding(rotations)
        padded_rot = rotations.reshape(-1, 3)
        body_quats = quaternion_from_euler_batch(
            padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes="rzyx"
        )
        body_quats = body_quats.reshape(B, J, 4)

        wbody_pos, body_com, wbody_quat = self.forward_kinematics_batch(
            body_quats, root_rotations, root_positions
        )

        """ Get sim full vals"""
        body_quats_full = quaternion_from_euler_batch(
            padded_rot[:, 0], padded_rot[:, 1], padded_rot[:, 2], axes="sxyz"
        )
        body_quats_full = np.concatenate(
            (root_rotations[:, None, :], body_quats_full.reshape(B, J, 4)), axis=1
        )

        if select_joints is None:
            pass
        else:
            wbody_pos_list = []
            wbody_quat_list = []
            for joint in select_joints:
                jidx = self.body_name.index(joint)
                wbody_pos_list.append(wbody_pos[:, jidx, :])
                wbody_quat_list.append(wbody_quat[:, jidx, :])
            wbody_pos = np.concatenate(wbody_pos_list).reshape(B, len(select_joints), 3)
            wbody_quat = np.concatenate(wbody_quat_list).reshape(
                B, len(select_joints), 4
            )

        return {
            "qpos": qpos,
            "wbpos": wbody_pos,
            "wbquat": wbody_quat,
            "bquat": body_quats_full,
            "body_com": body_com,
        }

    def qpos_fk(self, qpos, select_joints=None, fix_quat=True):
        """
        qpos: body representation (1->3: root position, 3->7: root orientation, 7->end: joint orientation)
        Rotations are represented in euler angles.
        Note that some joints have 1 DoF.
        B = batch size, J = number of joints
        Input: rotations (B, L), root_rotations (B, 4), root_positions (B, 3),
        Output: (B, J, 3) J is number of joints or selected joints
        """
        qpos_use = qpos.copy()
        rotations = qpos_use[7:]
        root_rotations = qpos_use[3:7]
        root_positions = qpos_use[:3]

        if fix_quat:
            root_rotations /= np.linalg.norm(root_rotations)
            qpos[3:7] = root_rotations

        J = len(self.body_name) - 1  # Exclude root

        rotations = rotations.reshape(-1, 3)
        body_quats = quaternion_from_euler_batch(
            rotations[:, 0], rotations[:, 1], rotations[:, 2], axes="rzyx"
        )

        wbody_pos, body_com, wbody_quat = self.forward_kinematics(
            body_quats, root_rotations, root_positions
        )

        """ Get sim full vals"""
        body_quats_full = quaternion_from_euler_batch(
            rotations[:, 0], rotations[:, 1], rotations[:, 2], axes="sxyz"
        )

        body_quats_full = np.concatenate(
            (
                root_rotations[
                    None,
                ],
                body_quats_full.reshape(J, 4),
            ),
            axis=0,
        )

        if select_joints is None:
            pass
        else:
            wbody_pos_list = []
            wbody_quat_list = []
            for joint in select_joints:
                jidx = self.body_name.index(joint)
                wbody_pos_list.append(wbody_pos[jidx, :])
                wbody_quat_list.append(wbody_quat[jidx, :])
            wbody_pos = np.concatenate(wbody_pos_list).reshape(len(select_joints), 3)
            wbody_quat = np.concatenate(wbody_quat_list).reshape(len(select_joints), 4)

        return {
            "qpos": qpos,
            "wbpos": wbody_pos,
            "wbquat": wbody_quat,
            "bquat": body_quats_full,
            "body_com": body_com,
        }

    def forward_kinematics(self, rotations, root_rotations, root_positions):
        positions_world = []
        rotations_world = []
        positions_world_com = []
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
                positions_world_com.append(
                    root_positions + quat_mul_vec(root_rotations, self._i_offsets[0, :])
                )
            else:

                jpos = (
                    quat_mul_vec(rotations_world[self._parents[i]], self._offsets[i, :])
                    + positions_world[self._parents[i]]
                )
                jquat = quaternion_multiply(
                    rotations_world[self._parents[i]], rotations[i - 1, :]
                )

                positions_world.append(jpos)
                rotations_world.append(jquat)
                positions_world_com.append(quat_mul_vec(jquat, self._i_offsets[i, :]))

        return (
            np.stack(positions_world),
            np.stack(positions_world_com),
            np.stack(rotations_world),
        )

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        assert len(rotations.shape) == 3
        assert rotations.shape[-1] == 4
        B = rotations.shape[0]
        J = self._offsets.shape[0]
        positions_world = []
        positions_world_com = []
        rotations_world = []

        expanded_offsets = np.broadcast_to(
            self._offsets, (B, J, self._offsets.shape[1])
        )
        expanded_i_offsets = np.broadcast_to(
            self._i_offsets, (B, J, self._i_offsets.shape[1])
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

        positions_world = np.stack(positions_world, axis=2)
        positions_world_com = np.stack(positions_world_com, axis=2)
        rotations_world = np.stack(rotations_world, axis=2)

        return (
            positions_world.transpose(0, 2, 1),
            positions_world_com.transpose(0, 2, 1),
            rotations_world.transpose(0, 2, 1),
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


def get_expert(expert_qpos):
    wbody_pos_mujoco = []
    wbody_quat_mujoco = []
    body_com_mujoco = []
    body_quat_mujoco = []

    occup_map = []
    import time

    t1 = time.time()

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:] = qpos.copy()
        env.sim.forward()

        wbody_pos_mujoco.append(env.get_wbody_pos().reshape(-1, 3))
        wbody_quat_mujoco.append(env.get_wbody_quat().reshape(-1, 4))
        body_com_mujoco.append(env.get_body_com().reshape(-1, 3))
        body_quat_mujoco.append(env.get_body_quat().reshape(-1, 4))

        # _map, _ = env.get_body_occup_map(occup_joints)
        # _map_torch = env_humanoid.get_body_occup_map(torch.tensor(qpos, dtype=dtype, device=device), occup_joints).squeeze()
        # print(torch.sum(_map_torch))
        # assert np.array_equal(_map, _map_torch.cpu().numpy()), "Map creation error"
    t2 = time.time()
    print("======================================================")
    print("Env:", t2 - t1)
    ## Singles
    from collections import defaultdict

    fk_res = defaultdict(list)
    [
        [fk_res[k].append(v) for k, v in env_humanoid.qpos_fk(qpos).items()]
        for qpos in expert_qpos
    ]
    t3 = time.time()
    # fk_res = {k: np.stack(v) for k, v in fk_res.items()}
    print("Seq fk:", t3 - t2)
    ## Batch
    fk_res = env_humanoid.qpos_fk_batch(expert_qpos)
    t4 = time.time()
    print("Batch Fk fk:", t4 - t3)
    print(
        np.sum(np.abs(wbody_pos_mujoco - fk_res["wbpos"])),
        np.sum(np.abs(wbody_quat_mujoco - fk_res["wbquat"])),
        np.sum(np.abs(body_quat_mujoco - fk_res["bquat"])),
        np.sum(np.abs(body_com_mujoco - fk_res["body_com"])),
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
    # cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh_all.xml"
    cfg.mujoco_model_file = "assets/mujoco_models/humanoid_smpl_neutral_masterfoot.xml"
    smpl_model_file = "assets/mujoco_models/humanoid_smpl_neutral_mesh.xml"
    master_foot_model_file = "assets/mujoco_models/humanoid_smpl_neutral_masterfoot.xml"
    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    cfg.masterfoot = True
    env = HumanoidEnv(
        cfg, init_expert=init_expert, data_specs=cfg.data_specs, mode="test"
    )

    dtype = torch.float64
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    env_humanoid = Humanoid(model_file=cfg.mujoco_model_file)
    smpl_model = load_model_from_path(smpl_model_file)
    sim_model = load_model_from_path(master_foot_model_file)
    converter = SMPLConverter(smpl_model, sim_model)
    # # occup_joints = ['LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Hips']

    data_load = joblib.load("sample_data/relive_mocap_qpos_grad.pkl")
    for k in data_load.keys():
        expert_qpos = data_load[k]["qpos"]
        get_expert(converter.qpos_smpl_2_new(expert_qpos))
