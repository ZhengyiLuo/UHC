"""
File: /kin_policy.py
Created Date: Friday July 16th 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Friday July 16th 2021 8:05:22 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
"""

import torch.nn as nn
import torch
import pickle

from tqdm import tqdm

from uhc.khrylib.rl.core.distributions import DiagGaussian
from uhc.khrylib.rl.core.policy import Policy
from uhc.utils.math_utils import *
from uhc.khrylib.models.mlp import MLP
from uhc.utils.flags import flags
from scipy.ndimage import gaussian_filter1d
from uhc.utils.torch_ext import get_scheduler
from uhc.models import KinNet

from uhc.utils.torch_utils import (
    get_heading_batch,
    get_heading_q,
    quaternion_multiply,
    quaternion_inverse,
    get_heading_q_batch,
    transform_vec_batch,
    quat_from_expmap_batch,
    quat_mul_vec_batch,
    get_qvel_fd_batch,
    transform_vec,
    rotation_from_quaternion,
    de_heading_batch,
    quat_mul_vec,
    quat_from_expmap,
    quaternion_multiply_batch,
    quaternion_inverse_batch,
)
from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.losses.loss_function import (
    compute_mpjpe_global,
    pose_rot_loss,
    root_pos_loss,
    root_orientation_loss,
    end_effector_pos_loss,
    linear_velocity_loss,
    angular_velocity_loss,
    action_loss,
    position_loss,
    orientation_loss,
    compute_error_accel,
    compute_error_vel,
)


class SuperNet(KinNet):
    def __init__(self, cfg, data_sample, device, dtype, mode="train"):
        super(KinNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.specs = cfg.model_specs
        self.mode = mode
        self.epoch = 0
        self.register_buffer("base_rot", torch.tensor([[0.7071, 0.7071, 0.0, 0.0]]))
        self.model_v = self.specs.get("model_v", 1)
        self.mlp_hsize = mlp_hsize = self.specs.get("mlp_hsize", [1024, 512])
        self.htype = htype = self.specs.get("mlp_htype", "relu")
        self.sim = {}
        self.qpos_lm = 74
        self.qvel_lm = 75
        self.pose_start = 7
        self.remove_base = True
        self.pose_delta = True

        self.get_dim(data_sample)
        self.model_mlp = MLP(self.state_dim, mlp_hsize, htype)
        self.fk_model = Humanoid(model_file=cfg.mujoco_model_file)
        self.setup_optimizer()

    def get_dim(self, data):
        # qpos_curr = data[f"qpos"][:, 0, :]
        zero_qpos = torch.zeros((1, 76)).to(self.device).type(self.dtype)
        zero_qpos[:, 3] = 1
        zero_qvel = torch.zeros((1, 75)).to(self.device).type(self.dtype)
        self.set_sim(zero_qpos, zero_qvel)

        self.state_dim = data["obs"].shape[-1] + data["action"].shape[-1]
        self.action_dim = 80

    def setup_optimizer(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if cfg.policy_optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        elif cfg.policy_optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)

        self.scheduler = get_scheduler(
            self.optimizer,
            policy="lambda",
            nepoch_fix=self.cfg.num_epoch_fix,
            nepoch=self.cfg.num_epoch,
        )

    def forward(self, state):
        action = self.model_mlp(state)
        return action

    def step(self, action, dt=1 / 30):
        curr_qpos = self.sim["qpos"].clone()
        curr_qvel = self.sim["qvel"].clone()

        curr_pos, curr_rot = curr_qpos[:, :3], curr_qpos[:, 3:7]

        if self.remove_base:
            curr_rot = self.remove_base_rot_batch(curr_rot)
        curr_heading = get_heading_q_batch(curr_rot)

        body_pose = action[:, (self.pose_start - 2) : self.qpos_lm].clone()
        if self.pose_delta:
            body_pose = body_pose + curr_qpos[:, self.pose_start :]
            body_pose[body_pose > np.pi] -= 2 * np.pi
            body_pose[body_pose < -np.pi] += 2 * np.pi

        next_qpos = torch.cat(
            [curr_pos[:, :2], action[:, : (self.pose_start - 2)], body_pose], dim=1
        )
        root_qvel = action[:, self.qpos_lm :]
        linv = quat_mul_vec_batch(curr_heading, root_qvel[:, :3])
        next_qpos[:, :2] += linv[:, :2] * dt

        angv = quat_mul_vec_batch(curr_rot, root_qvel[:, 3:6])
        angv_quat = quat_from_expmap_batch(angv * dt)
        new_rot = quaternion_multiply_batch(angv_quat, curr_rot)
        if self.remove_base:
            new_rot = self.add_base_rot_batch(new_rot)

        new_rot_norm = new_rot / torch.norm(new_rot, dim=1).view(-1, 1)

        next_qpos[:, 3:7] = new_rot_norm
        self.sim["qpos"] = next_qpos
        self.sim["qvel"] = get_qvel_fd_batch(curr_qpos, next_qpos, dt, transform=None)
        return self.sim["qpos"], self.sim["qvel"]

    def update_supervised(self, data_sample, num_epoch=20):
        pbar = tqdm(range(num_epoch))
        states, actions, gt_target_qpos, curr_qpos, res_qpos = (
            data_sample["states"],
            data_sample["actions"],
            data_sample["gt_target_qpos"],
            data_sample["curr_qpos"],
            data_sample["res_qpos"],
        )
        for _ in pbar:
            action_mean = self.forward(torch.cat([states, actions], dim=1))
            self.set_sim(curr_qpos)
            next_qpos, _ = self.step(action_mean)
            loss, loss_idv = self.compute_loss_lite(next_qpos, res_qpos)
            self.optimizer.zero_grad()
            loss.backward()  # Testing GT
            self.optimizer.step()  # Testing GT
            pbar.set_description_str(
                f"Super loss: {loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_idv])}] lr: {self.scheduler.get_last_lr()[0]:.5f}"
            )

            # total_loss, loss_dict, loss_unweighted_dict = self.kin_net.compute_loss_lite(next_qpos, target_qpos)
            # self.optimizer.zero_grad()
            # total_loss.backward()   # Testing GT
            # import ipdb; ipdb.set_trace()
            # self.optimizer.step()  # Testing GT
            # pbar.set_description_str(f"Per-step loss: {total_loss.cpu().detach().numpy():.3f} [{' '.join([str(f'{i * 1000:.3f}') for i in loss_unweighted_dict.values()])}] lr: {self.scheduler.get_lr()[0]:.5f}")

    def compute_loss_lite(self, pred_qpos, gt_qpos):
        w_rp, w_rr, w_p, w_ee = 50, 50, 1, 10
        fk_res_pred = self.fk_model.qpos_fk(pred_qpos, to_numpy=False)
        fk_res_gt = self.fk_model.qpos_fk(gt_qpos, to_numpy=False)

        pred_wbpos = fk_res_pred["wbpos"].reshape(pred_qpos.shape[0], -1)
        gt_wbpos = fk_res_gt["wbpos"].reshape(pred_qpos.shape[0], -1)

        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()  # pose loss
        ee_loss = end_effector_pos_loss(
            gt_wbpos, pred_wbpos
        ).mean()  # End effector loss

        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss + w_ee * ee_loss

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss, p_rot_loss, ee_loss]]