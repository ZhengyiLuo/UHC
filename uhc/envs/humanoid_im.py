import joblib
from scipy.linalg import cho_solve, cho_factor
import time
import pickle
import mujoco_py
from mujoco_py import functions as mjf
from gym import spaces
import os
import sys
import os.path as osp

from torch import from_numpy

from uhc.utils.torch_utils import quaternion_matrix_batch

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, load_model_from_xml
import copy
from uhc.utils.flags import flags
from uhc.utils.tools import get_expert, get_expert_master
from uhc.khrylib.utils.transformation import quaternion_from_euler
from uhc.khrylib.utils import *
from uhc.khrylib.rl.envs.common import mujoco_env
from Cython.Compiler.Errors import local_errors

from uhc.utils.transformation import (
    quaternion_from_euler_batch,
    quaternion_multiply_batch,
    quat_mul_vec,
    quat_mul_vec_batch,
    quaternion_from_euler,
    quaternion_inverse_batch,
)
from uhc.utils.math_utils import *
from uhc.smpllib.smpl_mujoco import SMPLConverter
from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.smpllib.smpl_robot import Robot, in_hull
from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose, qpos_to_smpl
from uhc.smpllib.smpl_parser import (
    SMPL_EE_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
)
from scipy.spatial.transform import Rotation as sRot


class HumanoidEnv(mujoco_env.MujocoEnv):

    def __init__(self, cfg, init_expert, data_specs, mode="train", no_root=False):
        # env specific
        self.use_quat = cfg.robot_cfg.get("ball", False)
        self.smpl_robot_orig = Robot(cfg.robot_cfg, data_dir=osp.join(cfg.base_dir, "data/smpl"))
        self.smpl_robot = Robot(
            cfg.robot_cfg,
            data_dir=osp.join(cfg.base_dir, "data/smpl"),
            masterfoot=cfg.masterfoot,
        )
        self.xml_str = self.smpl_robot.export_xml_string().decode("utf-8")
        # if cfg.masterfoot:
        #     mujoco_env.MujocoEnv.__init__(self, cfg.mujoco_model_file)
        # else:
        #     mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        mujoco_env.MujocoEnv.__init__(self, self.xml_str, 15)
        self.setup_constants(cfg, data_specs, mode, no_root)
        self.netural_data = joblib.load(self.netural_path)
        self.load_expert(init_expert)
        self.set_action_spaces()
        self.set_obs_spaces()
        self.weight = mujoco_py.functions.mj_getTotalmass(self.model)

    def setup_constants(self, cfg, data_specs, mode, no_root):
        self.cc_cfg = cfg
        self.set_cam_first = set()
        self.smpl_model = load_model_from_xml(self.smpl_robot_orig.export_xml_string().decode("utf-8"))

        # if self.cc_cfg.masterfoot:
        #     self.sim_model = load_model_from_path(cfg.mujoco_model_file)
        # else:
        #     self.sim_model = load_model_from_xml(
        #         self.smpl_robot.export_xml_string().decode("utf-8")
        #     )
        self.sim_model = load_model_from_xml(self.smpl_robot.export_xml_string().decode("utf-8"))
        self.expert = None
        self.base_rot = data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.netural_path = data_specs.get("neutral_path", "/hdd/zen/data/ActBound/AMASS/standing_neutral.pkl")
        self.no_root = no_root
        self.body_diff_thresh = cfg.get("body_diff_thresh", 0.5)
        self.body_diff_thresh_test = cfg.get("body_diff_thresh_test", 0.5)
        # self.body_diff_thresh_test = cfg.get("body_diff_thresh_test", 0.5)
        self.mode = mode
        self.end_reward = 0.0
        self.start_ind = 0
        self.rfc_rate = 1 if not cfg.rfc_decay else 0
        self.prev_bquat = None
        self.load_models()
        self.set_model_base_params()
        self.bquat = self.get_body_quat()
        self.humanoid = Humanoid(model=self.model)
        self.curr_vf = None  # storing current vf
        self.curr_torque = None  # Strong current applied torque at each joint

    def set_mode(self, mode):
        self.mode = mode

    def load_models(self):
        self.converter = SMPLConverter(
            self.smpl_model,
            self.sim_model,
            smpl_model=self.cc_cfg.robot_cfg.get("model", "smpl"),
        )
        self.sim_iter = 15
        self.qpos_lim = self.converter.get_new_qpos_lim()
        self.qvel_lim = self.converter.get_new_qvel_lim()
        self.body_lim = self.converter.get_new_body_lim()
        self.jpos_diffw = self.converter.get_new_diff_weight()[:, None]
        self.body_diffw = self.converter.get_new_diff_weight()[1:]
        self.body_qposaddr = get_body_qposaddr(self.model)

        self.jkd = self.converter.get_new_jkd() * self.cc_cfg.get("pd_mul", 1)
        self.jkp = self.converter.get_new_jkp() * self.cc_cfg.get("pd_mul", 1)

        self.a_scale = self.converter.get_new_a_scale()
        self.torque_lim = self.converter.get_new_torque_limit() * self.cc_cfg.get("tq_mul", 1)
        self.set_action_spaces()

    def convert_2_smpl_params(self, res):
        gt_qpos = (self.converter.qpos_new_2_smpl(res["gt"]) if self.converter is not None else res["gt"])
        pred_qpos = (self.converter.qpos_new_2_smpl(res["pred"]) if self.converter is not None else res["pred"])
        beta = self.expert["beta"][:gt_qpos.shape[0], :]  # for models with futures, need to account for the difference in number of frames

        gt_pose_aa, gt_trans = qpos_to_smpl(gt_qpos, self.smpl_model, self.cc_cfg.robot_cfg.get("model", "smpl"))
        pred_pose_aa, pred_trans = qpos_to_smpl(pred_qpos, self.smpl_model, self.cc_cfg.robot_cfg.get("model", "smpl"))

        with torch.no_grad():
            gt_vertices, gt_joints = self.smpl_robot.get_joint_vertices(
                pose_aa=torch.from_numpy(gt_pose_aa),
                th_betas=torch.from_numpy(beta) if self.cc_cfg.has_shape else None,
                th_trans=torch.from_numpy(gt_trans),
            )
            pred_vertices, pred_joints = self.smpl_robot.get_joint_vertices(
                pose_aa=torch.from_numpy(pred_pose_aa),
                th_betas=torch.from_numpy(beta) if self.cc_cfg.has_shape else None,
                th_trans=torch.from_numpy(pred_trans),
            )
        res["gt_vertices"] = gt_vertices
        res["gt_joints"] = gt_joints
        res["pred_vertices"] = pred_vertices
        res["pred_joints"] = pred_joints
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def reset_robot(self):
        beta = self.expert["beta"].copy()
        gender = self.expert["gender"].copy()
        obj_info = self.expert.get("obj_info", None)
        obj_pose = self.expert.get("obj_pose", None)

        if self.smpl_robot.smpl_model == "smplx":
            self.smpl_robot.load_from_skeleton(
                v_template=torch.from_numpy(self.expert["v_template"]).float(),
                gender=gender,
                objs_info=obj_info,
                obj_pose=obj_pose,
            )
        else:
            # import ipdb; ipdb.set_trace()
            self.smpl_robot.load_from_skeleton(torch.tensor(beta[0:1, :]).float(), gender=gender, objs_info=obj_info)

        xml_str = self.smpl_robot.export_xml_string().decode("utf-8")

        self.reload_sim_model(xml_str)
        if obj_info is None:
            self.weight = mujoco_py.functions.mj_getTotalmass(self.model)
        else:
            self.weight = self.smpl_robot.weight

        return xml_str

    def load_expert(self, expert_data, reload_robot=True):
        expert_meta = {"cyclic": False, "seq_name": expert_data["seq_name"]}
        self.expert = copy.deepcopy(expert_data)

        if self.cc_cfg.has_shape:
            pose_aa = expert_data["pose_aa"]
            trans = expert_data["trans"]
            self.expert_save = copy.deepcopy(expert_data)
            if reload_robot:
                self.reset_robot()

            self.humanoid.update_model(self.model)
            if self.use_quat:
                expert_qpos_quat = smpl_to_qpose(
                    pose=pose_aa,
                    mj_model=self.model,
                    trans=trans.squeeze(),
                    model=self.cc_cfg.robot_cfg.get("model", "smpl"),
                    use_quat=self.use_quat,
                    count_offset=self.cc_cfg.robot_cfg.get("mesh", True),
                )

            expert_qpos = smpl_to_qpose(
                pose=pose_aa,
                mj_model=self.model,
                trans=trans.squeeze(),
                model=self.cc_cfg.robot_cfg.get("model", "smpl"),
                count_offset=self.cc_cfg.robot_cfg.get("mesh", True),
            )

            self.expert["meta"] = expert_meta
            self.expert.update(self.humanoid.qpos_fk(torch.from_numpy(expert_qpos)))  # Not using this since the masterfoot stuff...
            # if self.use_quat: self.expert['qpos'] = expert_qpos_quat

            # self.expert.update(get_expert_master(expert_qpos, expert_meta, self))

        else:
            expert_qpos = expert_data["qpos"]

            if self.cc_cfg.masterfoot:
                self.expert.update(get_expert_master(expert_qpos, expert_meta, self))
            else:
                self.expert.update(get_expert(expert_qpos, expert_meta, self))

        # import ipdb; ipdb.set_trace()
        # self.expert['wbquat'][0] - expert_qpos_quat[0, 3:]

        if self.cc_cfg.obs_v == 3:
            self.expert["len"] -= 30

    def reload_curr_expert(self):
        # Reloading expert based on the new humanoid definition
        self.load_expert(self.expert_save, reload_robot=False)

    def set_model_base_params(self):
        if self.cc_cfg.action_type == "torque" and hasattr(self.cc_cfg, "j_stiff"):
            self.model.jnt_stiffness[1:] = self.cc_cfg.j_stiff
            self.model.dof_damping[6:] = self.cc_cfg.j_damp

    def set_action_spaces(self):
        cfg = self.cc_cfg
        self.vf_dim = 0
        self.meta_pd_dim = 0
        self.ndof = self.model.actuator_ctrlrange.shape[0]
        body_id_list = self.model.geom_bodyid.tolist()
        if cfg.residual_force:
            if cfg.residual_force_mode == "implicit":
                self.vf_dim = 6
            else:
                if cfg.residual_force_bodies == "all":
                    self.vf_bodies = SMPL_BONE_ORDER_NAMES
                else:
                    self.vf_bodies = cfg.residual_force_bodies

                self.vf_geoms = [body_id_list.index(self.model._body_name2id[body]) for body in self.vf_bodies]
                self.body_vf_dim = 6 + cfg.residual_force_torque * 3
                self.vf_dim = (self.body_vf_dim * len(self.vf_bodies) * cfg.get("residual_force_bodies_num", 1))

        if cfg.meta_pd:
            self.meta_pd_dim = 2 * 15
        elif cfg.meta_pd_joint:
            self.meta_pd_dim = 2 * self.jkp.shape[0]

        self.action_dim = self.ndof + self.vf_dim + self.meta_pd_dim
        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim),
            high=np.ones(self.action_dim),
            dtype=np.float32,
        )

    def set_obs_spaces(self):
        self.obs_dim = self.get_obs().size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def add_base_rot(self, quat):
        return quaternion_multiply(quat, self.base_rot)

    def get_obs(self):
        if self.cc_cfg.obs_type == "full":
            if self.cc_cfg.obs_v == 0:
                obs = self.get_full_obs()
            elif self.cc_cfg.obs_v == 1:
                obs = self.get_full_obs_v1()
            elif self.cc_cfg.obs_v == 2:
                if self.use_quat:
                    obs = self.get_full_obs_v2_quat()
                else:
                    obs = self.get_full_obs_v2()
            elif self.cc_cfg.obs_v == 3:
                obs = self.get_full_obs_v3()
            elif self.cc_cfg.obs_v == 4:
                obs = self.get_full_obs_v4()
            elif self.cc_cfg.obs_v == 5:
                obs = self.get_full_obs_v5()
            elif self.cc_cfg.obs_v == 6:
                obs = self.get_full_obs_v6()
        return obs

    def get_full_obs(self):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()
        obs = []
        # pos
        if self.cc_cfg.obs_heading:
            obs.append(np.array([get_heading(qpos[3:7])]))
        if self.cc_cfg.root_deheading:
            qpos[3:7] = de_heading(qpos[3:7])
        obs.append(qpos[2:])
        # vel
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)

        obs.append(self.get_expert_kin_pose())

        # phase
        if self.cc_cfg.obs_phase:
            phase = self.get_phase()
            obs.append(np.array([phase]))

        obs = np.concatenate(obs)
        return obs

    def get_phase(self):
        return self.cur_t / self.expert["len"]

    def get_full_obs_v1(self):

        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)

        ################ Body pose and z ################
        target_body_qpos = self.get_expert_qpos(delta_t=1)  # target body pose (1, 76)
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))

        obs.append(target_body_qpos[2:])  # obs: target z + body pose (1, 74) # ZL: shounldn't you remove base root here???
        obs.append(qpos[2:])  # obs: current z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        obs.append(np.array([rel_h]))

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint/com positions ################
        target_jpos = self.get_expert_joint_pos(delta_t=1)
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
        # obs: target body frame joint position (1, 72)
        obs.append(r_jpos.ravel())

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        target_com = self.get_expert_com_pos(delta_t=1)  # body frame position
        curr_com = self.data.xipos[1:self.body_lim].copy()

        r_com = curr_com - qpos[None, :3]
        r_com = transform_vec_batch(r_com, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(r_com.ravel())  # obs: current target body frame com position  (1, 72)
        diff_com = target_com.reshape(-1, 3) - curr_com
        diff_com = transform_vec_batch(diff_com, curr_root_quat, self.cc_cfg.obs_coord)
        # obs: current body frame com position (1, 72)
        obs.append(diff_com.ravel())

        ################ target/relative global joint quaternions ################
        # target_quat = self.get_expert_bquat(delta_t=1).reshape(-1, 4)
        target_quat = self.get_expert_wbquat(delta_t=1).reshape(-1, 4)

        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        for i in range(r_quat.shape[0]):
            r_quat[i] = quaternion_multiply(quaternion_inverse(hq), r_quat[i])
        # obs: current target body quaternion (1, 92)
        obs.append(r_quat.ravel())

        rel_quat = np.zeros_like(cur_quat)
        for i in range(rel_quat.shape[0]):
            rel_quat[i] = quaternion_multiply(quaternion_inverse(cur_quat[i]), target_quat[i])
        # obs: current target body quaternion (1, 92)
        obs.append(rel_quat.ravel())

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v2(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)
        # ZL : why use heading????. Should remove this...

        ######## target body pose #########
        target_body_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)
        ################ Body pose and z ################
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])
        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))

        obs.append(target_body_qpos[2:])  # obs: target z + body pose (1, 74)
        obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()  # ZL: I think this one has some issues. You are doing this twice.
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)  # full qvel, 75

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        obs.append(np.array([rel_h]))

        rel_pos = target_root_quat[:3] - qpos[:3]  # ZL: this is wrong. Makes no sense. Should be target_root_pos. Should be fixed.
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
        # obs: target body frame joint position (1, 72)
        obs.append(r_jpos.ravel())
        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative global joint quaternions ################
        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        hq_invert = quaternion_inverse(hq)
        hq_invert_batch = np.repeat(
            hq_invert[None,],
            r_quat.shape[0],
            axis=0,
        )

        obs.append(quaternion_multiply_batch(hq_invert_batch, r_quat).ravel())  # obs: current target body quaternion (1, 96) # this contains redundant information
        obs.append(quaternion_multiply_batch(quaternion_inverse_batch(cur_quat), target_quat).ravel())  # obs: current target body quaternion (1, 96)

        if self.cc_cfg.has_shape and self.cc_cfg.get("has_shape_obs", True):
            obs.append(self.get_expert_shape_and_gender())

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v5(self, delta_t=0):
        # No diff, no heading. Ablation
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        obs = []

        ######## target body pose #########
        target_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)
        ################ Body pose and z ################

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        # curr_root_quat = qpos[3:7]
        target_root_quat = self.remove_base_rot(target_qpos[3:7])
        # target_root_quat = target_qpos[3:7]

        hq_quat = get_heading_q_new(curr_root_quat)

        qpos[3:7] = de_heading_new(curr_root_quat)  # deheading the root
        diff_qpos = target_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))

        obs.append(target_qpos[2:])  # obs: target z + body pose (1, 74)
        obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec_new(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)  # full qvel, 75

        ################ relative heading and root position ################
        rel_h = get_heading_new(target_root_quat) - get_heading_new(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        obs.append(np.array([rel_h]))

        rel_pos = target_qpos[:3] - qpos[:3]
        rel_pos = transform_vec_new(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target joint + relative positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
        # obs: target body frame joint position (1, 72)
        obs.append(r_jpos.ravel())

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch_new(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        ################ target global joint quaternions ################
        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()

        hq_invert = quaternion_inverse(hq_quat)
        hq_invert_batch = np.repeat(
            hq_invert[None,],
            r_quat.shape[0],
            axis=0,
        )

        obs.append(quaternion_multiply_batch(hq_invert_batch, r_quat).ravel())  # obs: current target body quaternion (1, 96) [global] # this contains redundant information
        obs.append(quaternion_multiply_batch(quaternion_inverse_batch(cur_quat), target_quat).ravel())  # obs: current target body [global] quaternion (1, 96)

        if self.cc_cfg.has_shape and self.cc_cfg.get("has_shape_obs", True):
            obs.append(self.get_expert_shape_and_gender())

        obs = np.concatenate(obs)

        return obs

    def get_full_obs_v6(self, delta_t=0):
        # Most up to date and concise obs.
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq_quat = get_heading_q_new(curr_root_quat)
        # hq_quat = curr_root_quat

        ######## target body pose #########
        target_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)
        target_root_quat = self.remove_base_rot(target_qpos[3:7])

        ################ Root position and orientation ################
        rel_h = get_heading_new(target_root_quat) - get_heading_new(curr_root_quat)

        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        rel_pos = target_qpos[:3] - qpos[:3]
        rel_pos = transform_vec_new(rel_pos, hq_quat).ravel()

        obs.append(rel_pos)  # obs: x, y, z difference in current heading frame
        obs.append(np.array([rel_h]))  # Relative heading
        rel_root_quat = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat)).ravel()
        obs.append(rel_root_quat)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec_new(qvel[:3], hq_quat).ravel()  # First 3 velocities are in world frame, so needs to be transformed
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)  # full qvel, 75

        ################ target/difference joint positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()
        # difference between target joint positions and current joint positions
        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch_new(r_jpos, hq_quat)[1:]  # joint position in current body frame
        obs.append(r_jpos.ravel())

        diff_jpos = (target_jpos.reshape(-1, 3) - curr_jpos)[1:]
        diff_jpos = transform_vec_batch_new(diff_jpos, hq_quat)
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative local joint 6d rotations ################
        target_bquat = self.get_expert_bquat(delta_t=1).reshape(-1, 4)[1:].copy()
        cur_bquat = self.get_body_quat().reshape(-1, 4)[1:].copy()

        curr_jrot_quat = torch.from_numpy(cur_bquat).numpy().ravel()

        diff_jrot_quat = torch.from_numpy(quaternion_multiply_batch(quaternion_inverse_batch(cur_bquat), target_bquat)).numpy().ravel()

        obs.append(curr_jrot_quat)
        obs.append(diff_jrot_quat)

        if self.cc_cfg.has_shape and self.cc_cfg.get("has_shape_obs", True):
            obs.append(self.get_expert_shape_and_gender())

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v2_quat(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qpos_copy = qpos.copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)

        ######## target body pose #########
        target_body_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)

        ################ Body pose and z ################
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]

        obs.append(target_body_qpos[2:3])  # target z
        obs.append(qpos[2:3])  # current z
        obs.append(diff_qpos[2:3])  # different z

        diff_qpos[3:7] = target_root_quat  # getting removed base root
        qpos_copy[3:7] = curr_root_quat  # getting removed base root
        obs.append(quaternion_multiply_batch(quaternion_inverse_batch(qpos_copy[3:].reshape(-1, 4)), diff_qpos[3:].reshape(-1, 4)).ravel())  # diff in quat

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()  # ZL: I think this one has some issues
        if self.cc_cfg.obs_vel == "root":
            obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            obs.append(qvel)  # full qvel, 75

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        obs.append(np.array([rel_h]))

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # body frame position
        # obs: target body frame joint position (1, 72)
        obs.append(r_jpos.ravel())

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative global joint quaternions ################
        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        hq_invert = quaternion_inverse(hq)
        hq_invert_batch = np.repeat(
            hq_invert[None,],
            r_quat.shape[0],
            axis=0,
        )

        obs.append(quaternion_multiply_batch(hq_invert_batch, r_quat).ravel())  # obs: current target body quaternion (1, 96) # this contains redundant information

        obs.append(quaternion_multiply_batch(quaternion_inverse_batch(cur_quat), target_quat).ravel())  # obs: current target body quaternion (1, 96)

        if self.cc_cfg.has_shape and self.cc_cfg.get("has_shape_obs", True):
            obs.append(self.get_expert_shape_and_gender())

        obs = np.concatenate(obs)
        return obs

    def get_full_obs_v3(self):
        fut_frames = self.cc_cfg.get("fut_frames", 10)
        obs_acc = []
        skip = self.cc_cfg.get("skip", 10)
        for i in range(0, fut_frames * skip, skip):
            obs = self.get_full_obs_v2(delta_t=i)
            obs_acc.append(obs)
        obs_acc = np.concatenate(obs_acc)

        return obs_acc

    def get_full_obs_v4(self, delta_t=0):
        ## Global and Local obs seperated.
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()

        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
        global_obs = []
        local_obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        global_obs.append(hq)  # obs: heading (4,)

        ######## target body pose #########
        target_body_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)

        ################ Body pose and z ################
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])
        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root (global)
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(target_root_quat, quaternion_inverse(curr_root_quat))

        global_obs.append(target_body_qpos[2:7])
        global_obs.append(qpos[2:7])
        global_obs.append(diff_qpos[2:7])

        local_obs.append(target_body_qpos[7:].reshape(-1, 3))
        local_obs.append(qpos[7:].reshape(-1, 3))
        local_obs.append(diff_qpos[7:].reshape(-1, 3))
        ################ vels ################
        # vel
        qvel[:3] = transform_vec(qvel[:3], curr_root_quat, self.cc_cfg.obs_coord).ravel()  # ZL: I think this one has some issues
        if self.cc_cfg.obs_vel == "root":
            global_obs.append(qvel[:6])
        elif self.cc_cfg.obs_vel == "full":
            global_obs.append(qvel[:6])
            local_obs.append(qvel[6:].reshape(-1, 3))

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        global_obs.append(np.array([rel_h]))

        rel_pos = target_body_qpos[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        global_obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()

        # translate to body frame (zero-out root)
        r_jpos = curr_jpos - qpos[None, :3]
        r_jpos = transform_vec_batch(r_jpos, curr_root_quat, self.cc_cfg.obs_coord).T  # body frame position
        local_obs.append(r_jpos.reshape(-1, 3)[1:, :])  # obs: target body frame joint position (1, 72)

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord).T
        local_obs.append(diff_jpos.reshape(-1, 3)[1:, :])  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative global joint quaternions ################
        cur_quat = self.data.body_xquat.copy()[1:self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        hq_invert = quaternion_inverse(hq)
        hq_invert_batch = np.repeat(
            hq_invert[None,],
            r_quat.shape[0],
            axis=0,
        )

        local_obs.append(quaternion_multiply_batch(hq_invert_batch, r_quat)[1:, :])  # obs: current target body quaternion (1, 96) # this contains redundant information
        local_obs.append(quaternion_multiply_batch(quaternion_inverse_batch(cur_quat), target_quat)[1:, :])  # obs: current target body quaternion (1, 96)

        if self.cc_cfg.has_shape:
            global_obs.append(self.get_expert_shape_and_gender())

        local_obs = np.hstack(local_obs)
        global_obs = np.concatenate(global_obs)
        obs_full = np.concatenate([global_obs, local_obs.ravel()])
        return obs_full, local_obs, global_obs

    def get_seq_obs_explicit(self, delta_t=0):
        data = self.data
        qpos = data.qpos[:self.qpos_lim].copy()
        qvel = data.qvel[:self.qvel_lim].copy()
        # transform velocity
        qvel[:3] = transform_vec(qvel[:3], qpos[3:7], self.cc_cfg.obs_coord).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])

        ######## target body pose #########
        target_body_qpos = self.get_expert_qpos(delta_t=1 + delta_t)  # target body pose (1, 76)
        target_quat = self.get_expert_wbquat(delta_t=1 + delta_t).reshape(-1, 4)
        target_jpos = self.get_expert_joint_pos(delta_t=1 + delta_t)
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        ################ relative heading and root position ################
        rel_h = get_heading(target_root_quat) - get_heading(curr_root_quat)
        if rel_h > np.pi:
            rel_h -= 2 * np.pi
        if rel_h < -np.pi:
            rel_h += 2 * np.pi
        # obs: heading difference in angles (1, 1)
        obs.append(np.array([rel_h]))

        rel_pos = target_body_qpos[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        curr_jpos = self.data.body_xpos[1:self.body_lim].copy()

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(diff_jpos, curr_root_quat, self.cc_cfg.obs_coord)  # Transfrom jpos to relative frame
        obs.append(diff_jpos.ravel())  # obs: current diff body frame joint position  (1, 72)

        obs = np.concatenate(obs)
        return obs

    def fail_safe(self):
        self.data.qpos[:self.qpos_lim] = self.get_expert_qpos()
        self.data.qvel[:self.qvel_lim] = self.get_expert_qvel()
        self.sim.forward()

    def get_head_idx(self):
        return self.model._body_name2id["Head"] - 1

    def get_ee_pos(self, transform):
        data = self.data
        ee_name = SMPL_EE_NAMES
        ee_pos = []
        root_pos = data.qpos[:3]
        root_q = data.qpos[3:7].copy()
        for name in ee_name:
            bone_id = self.model._body_name2id[name]
            bone_vec = self.data.body_xpos[bone_id]
            if transform is not None:
                bone_vec = bone_vec - root_pos
                bone_vec = transform_vec(bone_vec, root_q, transform)
            ee_pos.append(bone_vec)
        return np.concatenate(ee_pos)

    def get_body_quat(self):
        qpos = self.get_humanoid_qpos()
        if self.use_quat:
            body_quat = [qpos[3:7]]
            for body in self.model.body_names[1:self.body_lim]:
                if body == "Pelvis" or not body in self.body_qposaddr:
                    continue
                start, end = self.body_qposaddr[body]
                body_quat.append(qpos[start:end])
            body_quat = np.concatenate(body_quat)

        else:
            body_quat = [qpos[3:7]]
            for body in self.model.body_names[1:self.body_lim]:
                if body == "Pelvis" or not body in self.body_qposaddr:
                    continue
                start, end = self.body_qposaddr[body]
                euler = np.zeros(3)
                euler[:end - start] = qpos[start:end]
                quat = quaternion_from_euler(euler[0], euler[1], euler[2], "rzyx")
                body_quat.append(quat)
            body_quat = np.concatenate(body_quat)
        return body_quat

    def get_wbody_quat(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xquat[1:self.body_lim].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xquat[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_com(self):
        # return self.data.subtree_com[0, :].copy()
        # return self.data.subtree_com[1, :].copy()
        return self.data.get_body_xipos("Pelvis")

    def get_head(self):
        bone_id = self.model._body_name2id["Head"]
        head_pos = self.data.body_xpos[bone_id]
        head_quat = self.data.body_xquat[bone_id]
        return np.concatenate((head_pos, head_quat))

    def get_wbody_pos(self, selectList=None):
        body_pos = []
        if selectList is None:
            # body_names = self.model.body_names[1:] # ignore plane
            return self.data.body_xpos[1:self.body_lim].copy().ravel()
        else:
            body_names = selectList
        for body in body_names:
            bone_idx = self.model._body_name2id[body]
            bone_vec = self.data.body_xpos[bone_idx]
            body_pos.append(bone_vec)
        return np.concatenate(body_pos)

    def get_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            # ignore plane
            body_names = self.model.body_names[1:self.body_lim]
        else:
            body_names = selectList

        for body in body_names:
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)

        return np.concatenate(body_pos)

    def get_full_body_com(self, selectList=None):
        body_pos = []
        if selectList is None:
            # ignore plane
            body_names = self.model.body_names[1:self.body_lim]
        else:
            body_names = selectList

        for body in body_names:
            bone_vec = self.data.get_body_xipos(body)
            body_pos.append(bone_vec)

        return np.concatenate(body_pos)

    def compute_desired_accel(self, qpos_err, qvel_err, k_p, k_d):
        dt = self.model.opt.timestep
        nv = self.model.nv

        M = np.zeros(nv * nv)
        mjf.mj_fullM(self.model, M, self.data.qM)
        M.resize(nv, nv)
        M = M[:self.qvel_lim, :self.qvel_lim]
        C = self.data.qfrc_bias.copy()[:self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()

    def compute_torque(self, ctrl, i_iter=0):
        cfg = self.cc_cfg
        dt = self.model.opt.timestep
        ctrl_joint = ctrl[:self.ndof]
        qpos = self.get_humanoid_qpos()
        qvel = self.get_humanoid_qvel()

        if self.cc_cfg.action_v == 1:
            base_pos = self.get_expert_kin_pose(delta_t=1)  # should use the target pose instead of the current pose
            while np.any(base_pos - qpos[7:] > np.pi):
                base_pos[base_pos - qpos[7:] > np.pi] -= 2 * np.pi
            while np.any(base_pos - qpos[7:] < -np.pi):
                base_pos[base_pos - qpos[7:] < -np.pi] += 2 * np.pi
        elif self.cc_cfg.action_v == 0:
            base_pos = cfg.a_ref
        target_pos = base_pos + ctrl_joint

        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])

        if cfg.meta_pd:
            meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
            curr_jkp = self.jkp.copy() * np.clip((meta_pds[i_iter] + 1), 0, 10)
            curr_jkd = self.jkd.copy() * np.clip((meta_pds[i_iter + self.sim_iter] + 1), 0, 10)
            # if flags.debug:
            # import ipdb; ipdb.set_trace()
            # print((meta_pds[i_iter + self.sim_iter] + 1), (meta_pds[i_iter] + 1))
        elif cfg.meta_pd_joint:
            num_jts = self.jkp.shape[0]
            meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim + self.meta_pd_dim)]
            curr_jkp = self.jkp.copy() * np.clip((meta_pds[:num_jts] + 1), 0, 10)
            curr_jkd = self.jkd.copy() * np.clip((meta_pds[num_jts:] + 1), 0, 10)
        else:
            curr_jkp = self.jkp.copy()
            curr_jkd = self.jkd.copy()

        k_p[6:] = curr_jkp
        k_d[6:] = curr_jkd
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - target_pos))
        qvel_err = qvel
        q_accel = self.compute_desired_accel(qpos_err, qvel_err, k_p, k_d)
        qvel_err += q_accel * dt
        torque = -curr_jkp * qpos_err[6:] - curr_jkd * qvel_err[6:]
        return torque

    """ RFC-Explicit """

    def rfc_explicit(self, vf):
        qfrc = np.zeros(self.data.qfrc_applied.shape)
        num_each_body = self.cc_cfg.get("residual_force_bodies_num", 1)
        residual_contact_only = self.cc_cfg.get("residual_contact_only", False)
        residual_contact_only_ground = self.cc_cfg.get("residual_contact_only_ground", False)
        residual_contact_projection = self.cc_cfg.get("residual_contact_projection", False)
        vf_return = np.zeros(vf.shape)
        for i, body in enumerate(self.vf_bodies):
            body_id = self.model._body_name2id[body]
            foot_pos = self.data.get_body_xpos(body)[2]
            has_contact = False

            geom_id = self.vf_geoms[i]
            for contact in self.data.contact[:self.data.ncon]:
                g1, g2 = contact.geom1, contact.geom2
                if (g1 == 0 and g2 == geom_id) or (g2 == 0 and g1 == geom_id):
                    has_contact = True
                    break

            if residual_contact_only_ground:
                pass
            else:
                has_contact = foot_pos <= 0.12

            if not (residual_contact_only and not has_contact):
                for idx in range(num_each_body):
                    contact_point = vf[(i * num_each_body + idx) * self.body_vf_dim:(i * num_each_body + idx) * self.body_vf_dim + 3]
                    if residual_contact_projection:
                        contact_point = self.smpl_robot.project_to_body(body, contact_point)

                    force = (vf[(i * num_each_body + idx) * self.body_vf_dim + 3:(i * num_each_body + idx) * self.body_vf_dim + 6] * self.cc_cfg.residual_force_scale)
                    torque = (vf[(i * num_each_body + idx) * self.body_vf_dim + 6:(i * num_each_body + idx) * self.body_vf_dim + 9] * self.cc_cfg.residual_force_scale if self.cc_cfg.residual_force_torque else np.zeros(3))

                    contact_point = self.pos_body2world(body, contact_point)

                    force = self.vec_body2world(body, force)
                    torque = self.vec_body2world(body, torque)

                    vf_return[(i * num_each_body + idx) * self.body_vf_dim:(i * num_each_body + idx) * self.body_vf_dim + 3] = contact_point
                    vf_return[(i * num_each_body + idx) * self.body_vf_dim + 3:(i * num_each_body + idx) * self.body_vf_dim + 6] = (force / self.cc_cfg.residual_force_scale)

                    # print(np.linalg.norm(force), np.linalg.norm(torque))
                    mjf.mj_applyFT(
                        self.model,
                        self.data,
                        force,
                        torque,
                        contact_point,
                        body_id,
                        qfrc,
                    )
        self.curr_vf = vf_return
        self.data.qfrc_applied[:] = qfrc

    """ RFC-Implicit """

    def rfc_implicit(self, vf):
        vf *= self.cc_cfg.residual_force_scale * self.rfc_rate
        curr_root_quat = self.remove_base_rot(self.get_humanoid_qpos()[3:7])
        hq = get_heading_q(curr_root_quat)
        # hq = get_heading_q(self.get_humanoid_qpos()[3:7])
        vf[:3] = quat_mul_vec(hq, vf[:3])
        vf = np.clip(vf, -self.cc_cfg.residual_force_lim, self.cc_cfg.residual_force_lim)
        self.data.qfrc_applied[:vf.shape[0]] = vf

    def do_simulation(self, action, n_frames):
        t0 = time.time()
        cfg = self.cc_cfg
        ctrl = action
        # import ipdb; ipdb.set_trace()

        # meta_pds = ctrl[(self.ndof + self.vf_dim):(self.ndof + self.vf_dim +
        #                                            self.meta_pd_dim)]
        # print(np.max(meta_pds), np.min(meta_pds))
        self.curr_torque = []
        for i in range(n_frames):
            if cfg.action_type == "position":
                torque = self.compute_torque(ctrl, i_iter=i)
            elif cfg.action_type == "torque":
                torque = ctrl * self.a_scale * 100
            torque = np.clip(torque, -self.torque_lim, self.torque_lim)

            # torque[(self.get_expert_kin_pose() == 0)] = 0
            self.curr_torque.append(torque)
            self.data.ctrl[:] = torque
            """ Residual Force Control (RFC) """
            if cfg.residual_force:
                vf = ctrl[(self.ndof):(self.ndof + self.vf_dim)].copy()
                if cfg.residual_force_mode == "implicit":
                    self.rfc_implicit(vf)
                else:
                    self.rfc_explicit(vf)
            # if flags.debug:
            #     self.data.qpos[: self.qpos_lim] = self.get_expert_qpos(
            #         delta_t=-1
            #     )  # debug
            #     self.sim.forward()  # debug
            self.sim.step()
            # try:
            #     self.sim.step()
            # except Exception as e:
            #     # if flags.debug:
            #     #     import ipdb
            #     #     ipdb.set_trace()
            #     print("Exception in do_simulation step:", e)
            # pass

            # self.render()

        if self.viewer is not None:
            self.viewer.sim_time = time.time() - t0

    def step(self, a):

        cfg = self.cc_cfg

        self.prev_qpos = self.get_humanoid_qpos()
        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()

        # do simulation
        # if np.isnan(a).any():
        #     print(self.data_loader.curr_key)
        #     print(a)

        fail = False
        # self.do_simulation(a, self.frame_skip)
        try:
            self.do_simulation(a, self.frame_skip)
        except Exception as e:
            print("Exception in do_simulation", e, self.cur_t)
            fail = True

        # if flags.debug:
        #     self.data.qpos[: self.qpos_lim] = self.get_expert_qpos(delta_t = 1)  # debug
        #     self.sim.forward()  # debug

        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        head_pos = self.get_wbody_pos(["Head"])
        reward = 1.0
        if cfg.env_term_body == "Head":
            body_fail = (self.expert is not None and head_pos[2] < self.expert["head_height_lb"] - 0.1)
        elif cfg.env_term_body == "root":
            body_fail = (self.expert is not None and self.get_humanoid_qpos()[2] < self.expert["height_lb"] - 0.1)
        elif cfg.env_term_body == "body":
            body_diff = self.calc_body_diff()
            body_fail = body_diff > self.body_diff_thresh if self.mode == "train" else body_diff > self.body_diff_thresh_test

        fail = fail or body_fail
        end = (self.cur_t >= cfg.env_episode_len) or (self.cur_t + self.start_ind >= self.expert["len"] + cfg.env_expert_trail_steps - 1)
        done = fail or end
        # if done:
        #     print("done!!!", fail, end)

        percent = self.cur_t / (self.expert["len"] - 1)
        obs = self.get_obs()
        return obs, reward, done, {"fail": fail, "end": end, "percent": percent}

    def reset_model(self):
        cfg = self.cc_cfg
        ind = 0
        self.start_ind = 0

        init_pose_exp = self.expert["qpos"][ind, :].copy()
        init_vel_exp = self.expert["qvel"][ind, :].copy()  # Using GT joint velocity
        if self.mode == "train":
            init_pose_exp[7:] += self.np_random.normal(loc=0.0, scale=cfg.env_init_noise, size=self.qpos_lim - 7)

        if cfg.reactive_v == 0:
            # self.set_state(init_pose, init_vel)
            pass
        elif cfg.reactive_v == 1:
            if self.mode == "train" and np.random.binomial(1, 1 - cfg.reactive_rate):
                # self.set_state(init_pose, init_vel)
                pass
            elif self.mode == "test":
                # self.set_state(init_pose, init_vel)
                # netural_qpos = self.netural_data['qpos']
                # init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
                # init_vel_exp = self.netural_data['qvel']
                pass
            else:
                netural_qpos = self.netural_data["qpos"]
                init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
                init_vel_exp = self.netural_data["qvel"]

            # self.set_state(init_pose, init_vel)
        else:
            init_pose = self.get_humanoid_qpos()
            init_pose[2] += 1.0
            self.set_state(init_pose, self.data.qvel)
        self.bquat = self.get_body_quat()
        # print("reactive!!!")
        # netural_qpos = self.netural_data['qpos']
        # init_pose_exp = self.match_heading_and_pos(init_pose_exp, netural_qpos)
        # init_vel_exp = self.netural_data['qvel']

        if self.expert["has_obj"]:
            obj_pose = self.expert["obj_pose"][ind, :].copy()
            init_pose = np.concatenate([init_pose_exp, obj_pose])
            init_vel = np.concatenate([init_vel_exp, np.zeros(self.expert["num_obj"] * 6)])

        else:
            init_pose = init_pose_exp
            init_vel = init_vel_exp
        self.set_state(init_pose, init_vel)
        # print("Resetting model")

        return self.get_obs()

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.lookat[:2] = self.get_humanoid_qpos()[:2]
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 33
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.2
            self.viewer.cam.elevation = -20
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)

    def match_heading_and_pos(self, qpos_1, qpos_2):
        posxy_1 = qpos_1[:2]
        qpos_1_quat = self.remove_base_rot(qpos_1[3:7])
        qpos_2_quat = self.remove_base_rot(qpos_2[3:7])
        heading_1 = get_heading_q(qpos_1_quat)
        qpos_2[3:7] = de_heading(qpos_2[3:7])
        qpos_2[3:7] = quaternion_multiply(heading_1, qpos_2[3:7])
        qpos_2[:2] = posxy_1
        return qpos_2

    def get_expert_index(self, t):
        return ((self.start_ind + t) % self.expert["len"] if self.expert["meta"]["cyclic"] else min(self.start_ind + t, self.expert["len"] - 1))

    def get_expert_offset(self, t):
        if self.expert["meta"]["cyclic"]:
            n = (self.start_ind + t) // self.expert["len"]
            offset = self.expert["meta"]["cycle_offset"] * n
        else:
            offset = np.zeros(2)
        return offset

    def get_expert_attr(self, attr, ind):
        return self.expert[attr][ind].copy()

    def get_expert_qpos(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        expert_qpos = self.get_expert_attr("qpos", ind)

        # if self.no_root:
        # expert_qpos[:3] = self.data.qpos[:3]
        return expert_qpos

    def get_expert_qvel(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        expert_vel = self.get_expert_attr("qvel", ind)

        return expert_vel

    def get_expert_kin_pose(self, delta_t=0):
        return self.get_expert_qpos(delta_t=delta_t)[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        ind = self.get_expert_index(self.cur_t + delta_t)
        wbpos = self.get_expert_attr("wbpos", ind)
        if self.no_root:
            all_wbpos = wbpos.reshape(-1, 3).copy()
            curr_root_pos = all_wbpos[0]
            curr_sim_root_pos = self.data.body_xpos[1:self.body_lim][0]
            all_wbpos[:, :] += (curr_sim_root_pos - curr_root_pos)[:3]
            wbpos = all_wbpos.flatten()

        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        ind = self.get_expert_index(self.cur_t + delta_t)
        body_com = self.get_expert_attr("body_com", ind)

        if self.no_root:
            all_body_com = body_com.reshape(-1, 3).copy()
            curr_root_pos = all_body_com[0]
            curr_sim_root_pos = self.get_body_com()[:3]
            all_body_com[:, :] += (curr_sim_root_pos - curr_root_pos)[:3]
            body_com = all_body_com.flatten()

        return body_com

    def get_expert_bquat(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        bquat = self.get_expert_attr("bquat", ind)
        return bquat

    def get_expert_wbquat(self, delta_t=0):
        ind = self.get_expert_index(self.cur_t + delta_t)
        wbquat = self.get_expert_attr("wbquat", ind)
        return wbquat

    def get_expert_shape_and_gender(self):
        cfg = self.cc_cfg
        shape = self.get_expert_attr("beta", 0)
        gender = self.get_expert_attr("gender", 0)
        obs = []
        if cfg.get("has_pca", True):
            obs.append(shape)

        obs.append([gender])

        if cfg.get("has_weight", False):
            obs.append([self.weight])

        if cfg.get("has_bone_length", False):
            obs.append(self.smpl_robot.bone_length)

        return np.concatenate(obs)

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        # jpos_dist = np.linalg.norm(
        #     diff[self.jpos_diffw.squeeze().astype(bool)], axis=1
        # ).mean()  # Taking the mean since we want to make sure the number of joints does not affect (as compared to sum). Should we just do Max??
        jpos_dist = np.linalg.norm(diff[self.jpos_diffw.squeeze().astype(bool)], axis=1).max()  # Taking the mean since we want to make sure the number of joints does not affect (as compared to sum). Should we just do Max??

        return jpos_dist

    def get_humanoid_qpos(self):
        return self.data.qpos.copy()[:self.qpos_lim]

    def get_humanoid_qvel(self):
        return self.data.qvel.copy()[:self.qvel_lim]

    def get_obj_qpos(self):
        return self.data.qpos.copy()[self.qpos_lim:]

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.qvel_lim:]

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self._viewers[mode] = self.viewer
        self.viewer_setup("rgb")
        return self.viewer

    def reload_sim_model(self, xml_str):
        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers

        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}

    def get_world_vf(self):
        return self.curr_vf

    def get_curr_torque(self):
        # Return current torque as list
        return self.curr_torque


if __name__ == "__main__":
    pass
