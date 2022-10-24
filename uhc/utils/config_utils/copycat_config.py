import yaml
import os
import os.path as osp
import glob
import numpy as np
from os import path

from uhc.khrylib.utils import recreate_dirs
from uhc.utils.config_utils.base_config import Base_Config


class Config(Base_Config):
    def __init__(self, mujoco_path="%s.xml", **kwargs):
        super().__init__(**kwargs)

        # training config
        self.gamma = self.cfg_dict.get("gamma", 0.95)
        self.tau = self.cfg_dict.get("tau", 0.95)
        self.policy_htype = self.cfg_dict.get("policy_htype", "relu")
        self.policy_hsize = self.cfg_dict.get("policy_hsize", [300, 200])
        self.policy_optimizer = self.cfg_dict.get("policy_optimizer", "Adam")
        self.policy_lr = self.cfg_dict.get("policy_lr", 5e-5)
        self.policy_momentum = self.cfg_dict.get("policy_momentum", 0.0)
        self.policy_weightdecay = self.cfg_dict.get("policy_weightdecay", 0.0)
        self.value_htype = self.cfg_dict.get("value_htype", "relu")
        self.value_hsize = self.cfg_dict.get("value_hsize", [300, 200])
        self.value_optimizer = self.cfg_dict.get("value_optimizer", "Adam")
        self.value_lr = self.cfg_dict.get("value_lr", 3e-4)
        self.value_momentum = self.cfg_dict.get("value_momentum", 0.0)
        self.value_weightdecay = self.cfg_dict.get("value_weightdecay", 0.0)
        self.adv_clip = self.cfg_dict.get("adv_clip", np.inf)
        self.clip_epsilon = self.cfg_dict.get("clip_epsilon", 0.2)
        self.log_std = self.cfg_dict.get("log_std", -2.3)
        self.fix_std = self.cfg_dict.get("fix_std", False)
        self.num_optim_epoch = self.cfg_dict.get("num_optim_epoch", 10)
        self.min_batch_size = self.cfg_dict.get("min_batch_size", 50000)
        self.mini_batch_size = self.cfg_dict.get("mini_batch_size", self.min_batch_size)
        self.save_n_epochs = self.cfg_dict.get("save_n_epochs", 100)
        self.reward_id = self.cfg_dict.get("reward_id", "quat")
        self.reward_weights = self.cfg_dict.get("reward_weights", None)
        self.end_reward = self.cfg_dict.get("end_reward", False)
        self.actor_type = self.cfg_dict.get("actor_type", "gauss")
        if self.actor_type == "mcp":
            self.num_primitive = self.cfg_dict.get("num_primitive", 8)
            self.composer_dim = self.cfg_dict.get("composer_dim", [[300, 200]])

        # adaptive parameters
        self.adp_iter_cp = np.array(self.cfg_dict.get("adp_iter_cp", [0]))
        self.adp_noise_rate_cp = np.array(self.cfg_dict.get("adp_noise_rate_cp", [1.0]))
        self.adp_noise_rate_cp = np.pad(
            self.adp_noise_rate_cp,
            (0, self.adp_iter_cp.size - self.adp_noise_rate_cp.size),
            "edge",
        )
        self.adp_log_std_cp = np.array(
            self.cfg_dict.get("adp_log_std_cp", [self.log_std])
        )
        self.adp_log_std_cp = np.pad(
            self.adp_log_std_cp,
            (0, self.adp_iter_cp.size - self.adp_log_std_cp.size),
            "edge",
        )
        self.adp_policy_lr_cp = np.array(
            self.cfg_dict.get("adp_policy_lr_cp", [self.policy_lr])
        )
        self.adp_policy_lr_cp = np.pad(
            self.adp_policy_lr_cp,
            (0, self.adp_iter_cp.size - self.adp_policy_lr_cp.size),
            "edge",
        )
        self.adp_noise_rate = None
        self.adp_log_std = None
        self.adp_policy_lr = None

        # env config
        self.mujoco_model_file = self.find_asset(
            mujoco_path % self.cfg_dict["mujoco_model"]
        )

        self.vis_model_file = self.find_asset(mujoco_path % self.cfg_dict["vis_model"])

        self.env_start_first = self.cfg_dict.get("env_start_first", False)
        self.env_init_noise = self.cfg_dict.get("env_init_noise", 0.0)
        self.env_episode_len = self.cfg_dict.get("env_episode_len", 200)
        self.env_term_body = self.cfg_dict.get("env_term_body", "head")
        self.env_expert_trail_steps = self.cfg_dict.get("env_expert_trail_steps", 0)

        self.obs_v = self.cfg_dict.get("obs_v", 0)
        self.obs_type = self.cfg_dict.get("obs_type", "full")
        self.obs_coord = self.cfg_dict.get("obs_coord", "root")
        self.obs_phase = self.cfg_dict.get("obs_phase", True)
        self.obs_heading = self.cfg_dict.get("obs_heading", False)
        self.obs_vel = self.cfg_dict.get("obs_vel", "full")
        self.root_deheading = self.cfg_dict.get("root_deheading", False)
        self.action_type = self.cfg_dict.get("action_type", "position")
        self.action_v = self.cfg_dict.get("action_v", 0)
        self.reactive_v = self.cfg_dict.get("reactive_v", 0)
        self.no_root = self.cfg_dict.get("no_root", False)
        self.reactive_rate = self.cfg_dict.get("reactive_rate", 0.3)
        self.sampling_temp = self.cfg_dict.get("sampling_temp", 0.2)
        self.sampling_freq = self.cfg_dict.get("sampling_freq", 0.75)

        # virutual force
        self.residual_force = self.cfg_dict.get("residual_force", False)
        self.residual_force_scale = self.cfg_dict.get("residual_force_scale", 200.0)
        self.residual_force_lim = self.cfg_dict.get("residual_force_lim", 100.0)
        self.residual_force_mode = self.cfg_dict.get("residual_force_mode", "implicit")
        self.residual_force_bodies = self.cfg_dict.get("residual_force_bodies", "all")
        self.residual_force_torque = self.cfg_dict.get("residual_force_torque", True)
        self.rfc_decay = self.cfg_dict.get("rfc_decay", False)

        # meta pd
        self.meta_pd = self.cfg_dict.get("meta_pd", False)
        self.meta_pd_joint = self.cfg_dict.get("meta_pd_joint", False)

        # masterfoot
        self.masterfoot = self.cfg_dict.get("masterfoot", False)
        self.fail_safe = self.cfg_dict.get("fail_safe", True)

        # robot config
        self.robot_cfg = self.cfg_dict.get("robot", {})
        if len(self.robot_cfg) == 0:
            self.robot_cfg["model"] = "smpl"
            self.robot_cfg["mesh"] = "mesh" in self.mujoco_model_file
        self.has_shape = self.cfg_dict.get("has_shape", False)

        # joint param
        if "joint_params" in self.cfg_dict:
            jparam = zip(*self.cfg_dict["joint_params"])
            jparam = [np.array(p) for p in jparam]
            self.jkp, self.jkd, self.a_ref, self.a_scale, self.torque_lim = jparam[1:6]
            self.a_ref = np.deg2rad(self.a_ref)
            jkp_multiplier = self.cfg_dict.get("jkp_multiplier", 1.0)
            jkd_multiplier = self.cfg_dict.get("jkd_multiplier", jkp_multiplier)
            self.jkp *= jkp_multiplier
            self.jkd *= jkd_multiplier
            torque_limit_multiplier = self.cfg_dict.get("torque_limit_multiplier", 1.0)
            self.torque_lim *= torque_limit_multiplier

        # body param
        if "body_params" in self.cfg_dict:
            bparam = zip(*self.cfg_dict["body_params"])
            bparam = [np.array(p) for p in bparam]
            self.b_diffw = bparam[1]
            self.jpos_diffw = np.concatenate([[1], self.b_diffw])

        ## Agent Name
        self.agent_name = self.cfg_dict.get("agent_name", "agent_copycat")
        self.model_name = self.cfg_dict.get("model_name", "super_net")

    def update_adaptive_params(self, i_iter):
        cp = self.adp_iter_cp
        ind = np.where(i_iter >= cp)[0][-1]
        nind = ind + int(ind < len(cp) - 1)
        t = (
            (i_iter - self.adp_iter_cp[ind]) / (cp[nind] - cp[ind])
            if nind > ind
            else 0.0
        )
        self.adp_noise_rate = (
            self.adp_noise_rate_cp[ind] * (1 - t) + self.adp_noise_rate_cp[nind] * t
        )
        self.adp_log_std = (
            self.adp_log_std_cp[ind] * (1 - t) + self.adp_log_std_cp[nind] * t
        )
        self.adp_policy_lr = (
            self.adp_policy_lr_cp[ind] * (1 - t) + self.adp_policy_lr_cp[nind] * t
        )

    def find_asset(self, asset_path):
        if not path.exists(asset_path):
            fullpath = path.join(
                self.base_dir, "assets/mujoco_models", path.basename(asset_path)
            )
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            return fullpath
        else:
            return path
