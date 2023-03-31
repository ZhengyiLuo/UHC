import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import numpy as np

import torch
import numpy as np
import pickle as pk
from tqdm import tqdm
from collections import defaultdict
import random
import argparse

from uhc.envs.humanoid_im import HumanoidEnv
from uhc.utils.config_utils.copycat_config import Config
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.utils.transformation import euler_from_quaternion, quaternion_matrix
from uhc.utils.math_utils import *
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES


def get_root_matrix(poses):
    matrices = []
    for pose in poses:
        mat = np.identity(4)
        root_pos = pose[:3]
        root_quat = pose[3:7]
        mat = quaternion_matrix(root_quat)
        mat[:3, 3] = root_pos
        matrices.append(mat)
    return matrices


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i + 1], dt, "heading")
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_frobenious_norm(x, y):
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i]
        y_mat_inv = np.linalg.inv(y[i])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(4)
        error += np.linalg.norm(ident_mat - error_mat, "fro")
    return error / len(x)


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_mean_abs(x):
    return np.abs(x).mean()


def compute_metrics(results, dt=1 / 30):
    if results is None:
        return

    res_dict = defaultdict(list)
    action_suss = defaultdict(list)

    for take in tqdm(results.keys()):

        res = results[take]
        traj_pred = res["qpos"].copy()
        traj_gt = res["qpos_gt"].copy()
        obj_pose = res["obj_pose"].copy()
        vels_gt = get_joint_vels(traj_gt, dt)
        accels_gt = get_joint_accels(vels_gt, dt)
        vels_pred = get_joint_vels(traj_pred, dt)

        accels_pred = get_joint_accels(vels_pred, dt)
        jpos_pred = compute_physcis_metris(traj_pred, obj_pose, res=res)
        jpos_gt = compute_physcis_metris(traj_gt, obj_pose, res=None)

        jpos_pred = jpos_pred.reshape(-1, 24, 3)
        jpos_gt = jpos_gt.reshape(-1, 24, 3)

        root_mat_pred = get_root_matrix(traj_pred)
        root_mat_gt = get_root_matrix(traj_gt)
        root_dist = get_frobenious_norm(root_mat_pred, root_mat_gt)

        vel_dist = get_mean_dist(vels_pred, vels_gt)

        accel_dist = np.mean(compute_error_accel(jpos_pred, jpos_gt)) * 1000

        smoothness = get_mean_abs(accels_pred)
        smoothness_gt = get_mean_abs(accels_gt)

        jpos_pred -= jpos_pred[:, 0:1]  # zero out root
        jpos_gt -= jpos_gt[:, 0:1]
        mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2).mean() * 1000

        # print(succ, succ_gt, take, slide_pred)
        succ = not res["fail_safe"]

        res_dict["root_dist"].append(root_dist)
        res_dict["mpjpe"].append(mpjpe)
        res_dict["accel_dist"].append(accel_dist)
        res_dict["succ"].append(succ)

        # res_dict["accels_pred"].append(smoothness)
        # res_dict["accels_gt"].append(smoothness_gt)
        res_dict["vel_dist"].append(vel_dist)

    # res_dict = {k: np.mean(v) for k, v in res_dict.items()}

    return res_dict


def compute_physcis_metris(traj, obj_pose, res=None):

    env.reset()
    joint_pos = []

    for fr in range(len(traj)):
        env.data.qpos[: env.qpos_lim] = traj[fr, :]
        # env.data.qpos[env.qpos_lim:] = obj_pose[fr]
        env.sim.forward()
        # env.render()
        margin = 0.005
        pen_acc = []
        pen_acc_check = []
        seq_len = len(obj_pose)
        # print(len(env.data.contact), env.data.ncon)

        # env.sim.model.geom_name2id("Hips")
        # https://github.com/rlworkgroup/gym-sawyer/blob/master/sawyer/mujoco/sawyer_env.py
        joint_pos.append(env.get_wbody_pos(selectList=SMPL_BONE_ORDER_NAMES))
    joint_pos = np.array(joint_pos)
    return joint_pos


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def compute_vel(joints):
    velocities = joints[1:] - joints[:-1]
    velocity_normed = np.linalg.norm(velocities, axis=2)
    return np.mean(velocity_normed, axis=1)


def compute_error_vel(joints_gt, joints_pred, vis=None):
    vel_gt = joints_gt[1:] - joints_gt[:-1]
    vel_pred = joints_pred[1:] - joints_pred[:-1]
    normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    return np.mean(normed[new_vis], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--mode", default="stats")
    parser.add_argument("--data", default="test")
    parser.add_argument("--wild", action="store_true", default=False)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=20)
    parser.add_argument("--action", type=str, default="all")
    parser.add_argument("--algo", type=str, default="statear")
    args = parser.parse_args()

    # data_res = joblib.load(f"results/motion_im/{args.cfg}/results/{args.iter:04d}_relive_all_coverage_full.pkl")
    data_res = joblib.load(
        f"results/motion_im/{args.cfg}/results/{args.iter:04d}_all_coverage_full.pkl"
    )

    cfg = Config(cfg_id=args.cfg, create_dirs=False)
    # cfg.mujoco_model_file = f"humanoid_smpl_neutral_mesh_all_h36m.xml"
    cfg.data_specs[
        "file_path"
    ] = "sample_data/h36m_train_no_sit_30_qpos.pkl"
    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = HumanoidEnv(
        cfg, mode="test", init_expert=init_expert, data_specs=cfg.data_specs
    )

    sr_res = defaultdict(dict)
    for k, v in tqdm(data_res.items()):
        sr_res[k] = {
            "qpos": np.array(v["pred"]),
            "qpos_gt": np.array(v["gt"]),
            "obj_pose": np.array(v["obj_pose"]),
            "fail_safe": v["fail_safe"] if "fail_safe" in v else False,
        }
    print(len(sr_res))
    # compute_metrics(sr_res)
    jobs = list(sr_res.items())

    from multiprocessing import Pool

    num_jobs = args.num_threads
    if num_jobs == 1:
        compute_metrics(sr_res)
    else:
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        jobs = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
        job_args = [({k: v for k, v in jobs[i]},) for i in range(len(jobs))]
        print(len(job_args))
        try:
            pool = Pool(num_jobs)  # multi-processing
            res_dict_list = pool.starmap(compute_metrics, job_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

        res_dict = defaultdict(list)
        for i in res_dict_list:
            [res_dict[k].extend(v) for k, v in i.items()]

    res_dict = {k: np.mean(v) for k, v in res_dict.items()}
    res_dict["coverage"] = int(res_dict["succ"] * len(sr_res))
    res_dict["succ"] *= 100
    prt_string = (
        "".join([f"{k}:{v:.3f} \t " for k, v in res_dict.items()])
        + f"--{args.cfg} | {args.iter} | {args.algo} | wild? {args.wild}"
    )
    print(prt_string)