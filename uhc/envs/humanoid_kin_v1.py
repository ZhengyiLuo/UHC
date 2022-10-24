import os
import sys

sys.path.append(os.getcwd())

from uhc.khrylib.rl.envs.common import mujoco_env
from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.models.policy_mcp import PolicyMCP
from uhc.utils.flags import flags
from uhc.envs.humanoid_im import HumanoidEnv

from uhc.smpllib.numpy_smpl_humanoid import Humanoid
from gym import spaces
from mujoco_py import functions as mjf
import pickle
import time
from scipy.linalg import cho_solve, cho_factor
import joblib


class HumanoidKinEnv(HumanoidEnv):
    # Wrapper class that wraps around Copycat agent
    def __init__(
        self, kin_cfg, cc_cfg, init_context, cc_iter=-1, mode="train", ar_mode=False
    ):
        mujoco_env.MujocoEnv.__init__(self, kin_cfg.scene_mujoco_file, 15)
        self.cc_cfg = cc_cfg
        self.kin_cfg = kin_cfg
        self.mode = mode
        self.set_cam_first = set()

        # env specific
        self.base_rot = cc_cfg.data_specs.get("base_rot", [0.7071, 0.7071, 0.0, 0.0])
        self.qpos_lim = 76
        self.qvel_lim = 75
        self.body_lim = 25
        self.rfc_rate = 1
        self.num_obj = self.get_obj_qpos().shape[0] // 7
        self.end_reward = 0.0
        self.start_ind = 0
        self.action_index_map = [0, 7, 21, 28]
        self.action_len = [7, 14, 7, 7]
        self.action_names = ["sit", "push", "avoid", "step"]
        self.smpl_humanoid = Humanoid(model_file=kin_cfg.mujoco_model_file)

        self.netural_path = kin_cfg.data_specs.get(
            "neutral_path", "/hdd/zen/data/ActBound/AMASS/standing_neutral.pkl"
        )
        self.netural_data = joblib.load(self.netural_path)

        self.body_qposaddr = get_body_qposaddr(self.model)
        self.bquat = self.get_body_quat()
        self.prev_bquat = None
        self.prev_hpos = None
        self.set_model_base_params()
        self.load_context(init_context)
        self.policy_v = kin_cfg.policy_specs["policy_v"]
        self.pose_delta = self.kin_cfg.model_specs.get("pose_delta", False)
        self.ar_model_v = self.kin_cfg.model_specs.get("model_v", 1)
        self.ar_mode = ar_mode

        self.set_spaces()
        self.jpos_diffw = np.array(
            kin_cfg.reward_weights.get(
                "jpos_diffw",
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            )
        )[:, None]

        """ Load CC Controller """
        state_dim = self.get_cc_obs().shape[0]
        cc_action_dim = self.action_dim
        if cc_cfg.actor_type == "gauss":
            self.cc_policy = PolicyGaussian(
                cc_cfg, action_dim=cc_action_dim, state_dim=state_dim
            )
        elif cc_cfg.actor_type == "mcp":
            self.cc_policy = PolicyMCP(
                cc_cfg, action_dim=cc_action_dim, state_dim=state_dim
            )

        self.cc_value_net = Value(
            MLP(state_dim, cc_cfg.value_hsize, cc_cfg.value_htype)
        )
        print(cc_cfg.model_dir)
        if cc_iter != -1:
            cp_path = "%s/iter_%04d.p" % (cc_cfg.model_dir, cc_iter)
        else:
            cc_iter = np.max(
                [
                    int(i.split("_")[-1].split(".")[0])
                    for i in os.listdir(cc_cfg.model_dir)
                ]
            )
            cp_path = "%s/iter_%04d.p" % (cc_cfg.model_dir, cc_iter)
        print(("loading model from checkpoint: %s" % cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))
        self.cc_running_state = model_cp["running_state"]
        self.cc_policy.load_state_dict(model_cp["policy_dict"])
        self.cc_value_net.load_state_dict(model_cp["value_dict"])

    def load_context(self, data_dict):
        self.ar_context = {
            k: v[0].detach().cpu().numpy() if v.requires_grad else v[0].cpu().numpy()
            for k, v in data_dict.items()
        }
        self.ar_context["len"] = self.ar_context["qpos"].shape[0] - 1
        self.gt_targets = self.smpl_humanoid.qpos_fk_batch(self.ar_context["qpos"])
        self.target = self.smpl_humanoid.qpos_fk(self.ar_context["ar_qpos"][0])

    def get_obs(self):
        ar_obs = self.get_ar_obs_v1()
        return ar_obs

    def get_cc_obs(self):
        if self.cc_cfg.obs_v == 0:
            cc_obs = self.get_full_obs()
        elif self.cc_cfg.obs_v == 1:
            cc_obs = self.get_full_obs_v1()
        elif self.cc_cfg.obs_v == 2:
            cc_obs = self.get_full_obs_v2()
        return cc_obs

    def remove_base_rot(self, quat):
        return quaternion_multiply(quat, quaternion_inverse(self.base_rot))

    def add_base_rot(self, quat):
        return quaternion_multiply(quat, self.base_rot)

    def get_head_idx(self):
        return self.model._body_name2id["Head"] - 1

    def get_ar_obs_v1(self):
        data = self.data
        qpos = data.qpos[: self.qpos_lim].copy()
        qvel = data.qvel[: self.qvel_lim].copy()
        input_qpos = self.ar_context["qpos"][self.cur_t + 1].copy()
        target = self.smpl_humanoid.qpos_fk(input_qpos)
        # transform velocity
        qvel[:3] = transform_vec(
            qvel[:3], qpos[3:7], self.cc_cfg.obs_coord
        ).ravel()  # body angular velocity
        obs = []

        curr_root_quat = self.remove_base_rot(qpos[3:7])
        hq = get_heading_q(curr_root_quat)
        obs.append(hq)  # obs: heading (4,)

        ################ Body pose and z ################
        target_body_qpos = input_qpos
        target_root_quat = self.remove_base_rot(target_body_qpos[3:7])

        qpos[3:7] = de_heading(curr_root_quat)  # deheading the root
        diff_qpos = target_body_qpos.copy()
        diff_qpos[2] -= qpos[2]
        diff_qpos[7:] -= qpos[7:]
        diff_qpos[3:7] = quaternion_multiply(
            target_root_quat, quaternion_inverse(curr_root_quat)
        )

        obs.append(target_body_qpos[2:])  # obs: target z + body pose (1, 74)
        obs.append(qpos[2:])  # obs: target z +  body pose (1, 74)
        obs.append(diff_qpos[2:])  # obs:  difference z + body pose (1, 74)

        ################ vels ################
        # vel
        qvel[:3] = transform_vec(
            qvel[:3], curr_root_quat, self.cc_cfg.obs_coord
        ).ravel()
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
        obs.append(np.array([rel_h]))  # obs: heading difference in angles (1, 1)

        rel_pos = target_root_quat[:3] - qpos[:3]
        rel_pos = transform_vec(rel_pos, curr_root_quat, self.cc_cfg.obs_coord).ravel()
        obs.append(rel_pos[:2])  # obs: relative x, y difference (1, 2)

        ################ target/difference joint positions ################
        target_jpos = target["wbpos"]
        curr_jpos = self.data.body_xpos[1 : self.body_lim].copy()
        r_jpos = curr_jpos - qpos[None, :3]  # translate to body frame (zero-out root)
        r_jpos = transform_vec_batch(
            r_jpos, curr_root_quat, self.cc_cfg.obs_coord
        )  # body frame position
        obs.append(r_jpos.ravel())  # obs: target body frame joint position (1, 72)

        diff_jpos = target_jpos.reshape(-1, 3) - curr_jpos
        diff_jpos = transform_vec_batch(
            diff_jpos, curr_root_quat, self.cc_cfg.obs_coord
        )
        obs.append(
            diff_jpos.ravel()
        )  # obs: current diff body frame joint position  (1, 72)

        ################ target/relative global joint quaternions ################
        # target_quat = self.get_expert_bquat(delta_t=1).reshape(-1, 4)
        target_quat = target["wbquat"]
        cur_quat = self.data.body_xquat.copy()[1 : self.body_lim]

        if cur_quat[0, 0] == 0:
            cur_quat = target_quat.copy()

        r_quat = cur_quat.copy()
        for i in range(r_quat.shape[0]):
            r_quat[i] = quaternion_multiply(
                quaternion_inverse(hq), r_quat[i]
            )  # ZL: you have gotta batch this.....
        obs.append(
            r_quat.ravel()
        )  # obs: current target body quaternion (1, 96) # this contains redundent information

        rel_quat = np.zeros_like(cur_quat)
        for i in range(rel_quat.shape[0]):
            rel_quat[i] = quaternion_multiply(
                quaternion_inverse(cur_quat[i]), target_quat[i]
            )  # ZL: you have gotta batch this.....
        obs.append(rel_quat.ravel())  # obs: current target body quaternion (1, 96)

        obs = np.concatenate(obs)
        return obs

    def step_ar(self, a, dt=1 / 30):
        cfg = self.kin_cfg
        qpos_lm = 74
        pose_start = 7
        curr_qpos = self.data.qpos[: self.qpos_lim].copy()
        curr_qvel = self.data.qvel[: self.qvel_lim].copy()

        curr_pos, curr_rot = curr_qpos[:3], curr_qpos[3:7]
        if cfg.remove_base:
            curr_rot = self.remove_base_rot(curr_rot)
        curr_heading = get_heading_q(curr_rot)

        body_pose = a[(pose_start - 2) : qpos_lm]

        if self.pose_delta:
            body_pose += curr_qpos[pose_start:]
            body_pose[body_pose > np.pi] -= 2 * np.pi
            body_pose[body_pose < -np.pi] += 2 * np.pi

        next_qpos = np.concatenate(
            [curr_pos[:2], a[: (pose_start - 2)], body_pose], axis=0
        )
        root_qvel = a[qpos_lm:]
        linv = quat_mul_vec(curr_heading, root_qvel[:3])
        next_qpos[:2] += linv[:2] * dt

        angv = quat_mul_vec(curr_rot, root_qvel[3:6])
        angv_quat = quat_from_expmap(angv * dt)
        new_rot = quaternion_multiply(angv_quat, curr_rot)
        if cfg.remove_base:
            new_rot = self.add_base_rot(new_rot)

        new_rot = new_rot / np.linalg.norm(new_rot)

        next_qpos[3:7] = new_rot
        return next_qpos

    def step(self, a):
        cfg = self.kin_cfg
        cc_cfg = self.cc_cfg
        # record prev state
        self.prev_qpos = self.get_humanoid_qpos()
        self.prev_qvel = self.get_humanoid_qvel()
        self.prev_bquat = self.bquat.copy()
        self.prev_hpos = self.get_head().copy()

        next_qpos = self.step_ar(a.copy())
        self.target = self.smpl_humanoid.qpos_fk(next_qpos)  # forming target from arnet

        # if flags.debug:
        #     self.target = self.smpl_humanoid.qpos_fk(self.ar_context['qpos'][self.cur_t + 1]) # GT
        # self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) # Debug

        # if self.ar_mode:4
        #     self.target = self.smpl_humanoid.qpos_fk(self.ar_context['ar_qpos'][self.cur_t + 1]) #

        cc_obs = self.get_cc_obs()
        cc_obs = self.cc_running_state(cc_obs, update=False)
        cc_a = self.cc_policy.select_action(
            torch.from_numpy(cc_obs)[
                None,
            ],
            mean_action=True,
        )[
            0
        ].numpy()  # CC step

        if flags.debug:
            # self.do_simulation(cc_a, self.frame_skip)
            self.data.qpos[: self.qpos_lim] = self.get_expert_qpos()  # debug
            # self.data.qpos[:self.qpos_lim] = self.ar_context['qpos'][self.cur_t + 1] # debug
            # self.data.qpos[:self.qpos_lim] = self.gt_targets['qpos'][self.cur_t + 1] # debug
            # self.data.qpos[:self.qpos_lim] = self.ar_context['ar_qpos'][self.cur_t + 1] # ARNet Qpos
            self.sim.forward()  # debug
            # self.do_simulation(cc_a, self.frame_skip)
        else:
            self.do_simulation(cc_a, self.frame_skip)

        self.cur_t += 1

        self.bquat = self.get_body_quat()
        # get obs
        reward = 1.0

        if cfg.env_term_body == "body":
            # body_diff = self.calc_body_diff()
            # fail = body_diff > 8
            body_diff = self.calc_body_diff()
            if self.mode == "train":
                body_gt_diff = self.calc_body_gt_diff()
                fail = body_diff > 10 or body_gt_diff > 10
            else:
                fail = body_diff > 10

        else:
            raise NotImplemented()

        end = (self.cur_t >= cc_cfg.env_episode_len) or (
            self.cur_t + self.start_ind >= self.ar_context["len"]
        )
        done = fail or end

        # if done: # ZL: Debug
        #     exit()
        # print("done!!!", self.cur_t, self.ar_context['len'] )

        percent = self.cur_t / self.ar_context["len"]
        obs = self.get_obs()
        return obs, reward, done, {"fail": fail, "end": end, "percent": percent}

    def set_mode(self, mode):
        self.mode = mode

    def ar_fail_safe(self):
        self.data.qpos[: self.qpos_lim] = self.ar_context["ar_qpos"][self.cur_t + 1]
        # self.data.qpos[:self.qpos_lim] = self.get_target_qpos()
        self.data.qvel[: self.qvel_lim] = self.ar_context["ar_qvel"][self.cur_t + 1]
        self.sim.forward()

    def reset_model(self):
        cfg = self.kin_cfg
        ind = 0
        self.start_ind = 0

        if self.ar_mode:
            init_pose_exp = self.ar_context["ar_qpos"][0].copy()
            init_vel_exp = self.ar_context["ar_qvel"][0].copy()
        else:
            init_pose_exp = self.ar_context["init_qpos"].copy()
            init_vel_exp = self.ar_context["init_qvel"].copy()

        if flags.debug:
            init_pose_exp = self.ar_context["qpos"][0].copy()
            init_vel_exp = self.ar_context["qvel"][0].copy()

        # obj_pose = self.convert_obj_qpos(self.ar_context["action_one_hot"][0], self.ar_context['obj_pose'][0])
        # obj_pose = self.ar_context['obj_pose'][0]

        init_pose = np.concatenate([init_pose_exp])
        init_vel = np.concatenate([init_vel_exp])

        self.set_state(init_pose, init_vel)
        self.target = self.smpl_humanoid.qpos_fk(init_pose_exp)

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

    def get_expert_qpos(self, delta_t=0):
        expert_qpos = self.target["qpos"].copy()
        return expert_qpos

    def get_target_kin_pose(self, delta_t=0):
        return self.get_expert_qpos()[7:]

    def get_expert_joint_pos(self, delta_t=0):
        # world joint position
        wbpos = self.target["wbpos"]
        return wbpos

    def get_expert_com_pos(self, delta_t=0):
        # body joint position
        body_com = self.target["body_com"]
        return body_com

    def get_expert_bquat(self, delta_t=0):
        bquat = self.target["bquat"]
        return bquat

    def get_expert_wbquat(self, delta_t=0):
        wbquat = self.target["wbquat"]
        return wbquat

    def calc_body_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.get_expert_joint_pos().reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_ar_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        # e_wbpos = self.get_target_joint_pos().reshape(-1, 3)
        e_wbpos = self.ar_context["ar_wbpos"][self.cur_t + 1].reshape(-1, 3)
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def calc_body_gt_diff(self):
        cur_wbpos = self.get_wbody_pos().reshape(-1, 3)
        e_wbpos = self.gt_targets["wbpos"][self.cur_t]
        diff = cur_wbpos - e_wbpos
        diff *= self.jpos_diffw
        jpos_dist = np.linalg.norm(diff, axis=1).sum()
        return jpos_dist

    def get_obj_qpos(self, action_one_hot=None):
        obj_pose_full = self.data.qpos.copy()[self.qpos_lim :]
        if action_one_hot is None:
            return obj_pose_full
        elif np.sum(action_one_hot) == 0:
            return np.array([0, 0, 0, 1, 0, 0, 0])

        action_idx = np.nonzero(action_one_hot)[0][0]
        obj_start = self.action_index_map[action_idx]
        obj_end = obj_start + self.action_len[action_idx]

        return obj_pose_full[obj_start:obj_end][
            :7
        ]  # ZL: only support handling one obj right now...

    def convert_obj_qpos(self, action_one_hot, obj_pose):
        if np.sum(action_one_hot) == 0:
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            for i in range(self.num_obj):
                obj_qos[(i * 7) : (i * 7 + 3)] = [(i + 1) * 100, 100, 0]
            return obj_qos
        else:
            action_idx = np.nonzero(action_one_hot)[0][0]
            obj_qos = np.zeros(self.get_obj_qpos().shape[0])
            # setting defult location for objects
            for i in range(self.num_obj):
                obj_qos[(i * 7) : (i * 7 + 3)] = [(i + 1) * 100, 100, 0]

            obj_start = self.action_index_map[action_idx]
            obj_end = obj_start + self.action_len[action_idx]
            obj_qos[obj_start:obj_end] = obj_pose

            return obj_qos

    def get_obj_qvel(self):
        return self.data.qvel.copy()[self.qvel_lim :]


if __name__ == "__main__":
    pass
