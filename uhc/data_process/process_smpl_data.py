import os
import sys
import pdb

sys.path.append(os.getcwd())

import numpy as np
import glob
import pickle as pk
import joblib
import torch

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


np.random.seed(1)
left_right_idx = [
    0,
    2,
    1,
    3,
    5,
    4,
    6,
    8,
    7,
    9,
    11,
    10,
    12,
    14,
    13,
    15,
    17,
    16,
    19,
    18,
    21,
    20,
    23,
    22,
]


def left_to_rigth_euler(pose_euler):
    pose_euler[:, :, 0] = pose_euler[:, :, 0] * -1
    pose_euler[:, :, 2] = pose_euler[:, :, 2] * -1
    pose_euler = pose_euler[:, left_right_idx, :]
    return pose_euler


def flip_smpl(pose, trans=None):
    """
    Pose input batch * 72
    """
    curr_spose = sRot.from_rotvec(pose.reshape(-1, 3))
    curr_spose_euler = curr_spose.as_euler("ZXY", degrees=False).reshape(
        pose.shape[0], 24, 3
    )
    curr_spose_euler = left_to_rigth_euler(curr_spose_euler)
    curr_spose_rot = sRot.from_euler(
        "ZXY", curr_spose_euler.reshape(-1, 3), degrees=False
    )
    curr_spose_aa = curr_spose_rot.as_rotvec().reshape(pose.shape[0], 24, 3)
    if trans != None:
        pass
        # target_root_mat = curr_spose.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # root_mat = curr_spose_rot.as_matrix().reshape(pose.shape[0], 24, 3, 3)[:, 0]
        # apply_mat = np.matmul(target_root_mat[0], np.linalg.inv(root_mat[0]))

    return curr_spose_aa.reshape(-1, 72)


def fix_height(
    expert, expert_meta, env, gnd_threh=-0.15, feet_offset=-0.015, begin_feet_thresh=0.3
):
    wbpos = expert["wbpos"]
    wbpos = wbpos.reshape(wbpos.shape[0], 24, 3)
    begin_feet = min(wbpos[0, 4, 2], wbpos[0, 8, 2])
    if begin_feet > begin_feet_thresh:
        print(expert_meta["seq_name"], "sequence invalid for copycat")
        return expert

    begin_feet += feet_offset  # Hypter parameter to tune
    qpos = expert["qpos"]
    qpos[:, 2] -= begin_feet
    new_expert = get_expert(qpos, expert_meta, env)
    new_wpos = new_expert["wbpos"]
    new_wpos = new_wpos.reshape(new_wpos.shape[0], 24, 3)
    ground_pene = min(np.min(new_wpos[:, 4, 2]), np.min(new_wpos[:, 8, 2]))
    if ground_pene < gnd_threh:
        print(
            expert_me_["seq_name"],
            "negative sequence invalid for copycat",
            ground_pene,
        )
        return expert
    return new_expert


def smpl_2_entry(
    seq_name,
    env,
    smpl_dict,
    gnd_threh=-0.15,
    feet_offset=-0.015,
    begin_feet_thresh=0.3,
    fix_feet=True,
    full=False,
):
    pose_aa = smpl_dict["pose"]
    trans = smpl_dict["trans"]
    seq_len = pose_aa.shape[0]
    shape = smpl_dict["shape"] if "shape" in v else np.zeros([seq_len, 10])
    gender = smpl_dict["gender"] if "gender" in v else "neutral"
    obj_pose = smpl_dict["obj_pose"] if "obj_pose" in v else None

    seq_length = pose_aa.shape[0]
    if seq_length < 10:
        return None
    pose_aa = torch.from_numpy(pose_aa)
    humanoid_model = env.model
    pose_seq_6d = convert_aa_to_orth6d(pose_aa).reshape(-1, 144)
    qpos = smpl_to_qpose(pose=pose_aa, model=humanoid_model, trans=trans)

    expert_meta = {"cyclic": False, "seq_name": seq_name}
    expert_res = get_expert(qpos, expert_meta, env)
    if fix_feet:
        expert_res = fix_height(
            expert_res,
            expert_meta,
            env,
            gnd_threh=gnd_threh,
            feet_offset=feet_offset,
            begin_feet_thresh=begin_feet_thresh,
        )

    trans = expert_res["qpos"][:, :3]
    # print(expert_res['qpos'].shape)
    entry = None
    if not expert_res is None:
        entry = {
            "pose_aa": pose_aa.numpy(),
            "pose_6d": pose_seq_6d.numpy(),
            "qpos": qpos,
            "trans": trans,
            "beta": shape[:10] if not shape is None else np.zeros(10),
            "seq_name": seq_name,
            "gender": gender,
            "expert": expert_res,
        }
        if not obj_pose is None:
            entry["obj_pose"] = obj_pose

    return entry


if __name__ == "__main__":
    data_dir = "/hdd/zen/dev/copycat/MST10192 Final Working/sample.pkl"
    # data_dir = '/hdd/zen/data/video_pose/prox/results/sample.pkl'
    # data_dir = "sample_data/h36m_test_30_fitted_grad.pkl"
    # data_dir = "sample_data/h36m_train_30_fitted_grad_test.pkl"
    # data_dir = "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_full.p"
    # data_dir = "sample_data/h36m_train_30_fitted_grad.pkl"
    # data_dir = "sample_data/egopose_mocap_smpl_grad.pkl"
    # data_dir = "sample_data/h36m_all_smpl.pkl"
    # data_dir = "sample_data/relive_mocap_smpl_grad.pkl"
    # data_dir = "sample_data/relive_wild_smpl.pkl"
    # data_dir = "sample_data/relive_ar_smpl.pkl"
    # data_dir = "sample_data/relive_third_smpl.pkl"
    # data_dir = "/hdd/zen/data/copycat/seqs/AIST++/aist_smpl.pkl"
    # fix_feet = False
    fix_feet = True
    data_res = {}
    seq_length = -1
    cfg = Config(cfg_id="copycat_5", create_dirs=False)

    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    random_expert = data_loader.sample_seq()
    env = HumanoidEnv(
        cfg, init_expert=random_expert, data_specs=cfg.data_specs, mode="test"
    )

    # target_frs = [20,30,40] # target framerate
    video_annot = {}
    counter = 0
    seq_counter = 0
    # gnd_threh = -0.15
    # feet_offset = -0.015
    # begin_feet_thresh = 0.3
    gnd_threh = -1
    feet_offset = -0.015
    begin_feet_thresh = 50

    # model_file = f'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    data_db = joblib.load(data_dir)
    all_data = list(data_db.items())
    np.random.shuffle(all_data)
    pbar = tqdm(all_data)
    for (k, v) in pbar:
        pbar.set_description(k)
        entry = smpl_2_entry(
            env=env,
            seq_name=k,
            smpl_dict=v,
            gnd_threh=gnd_threh,
            feet_offset=feet_offset,
            begin_feet_thresh=begin_feet_thresh,
            fix_feet=fix_feet,
        )
        if not entry is None:
            data_res[k] = entry
            counter += 1
        # if counter > 10:
        # break

    # output_file_name = "sample_data/h36m_all_qpos.pkl"
    # output_file_name = "sample_data/relive_mocap_qpos_grad.pkl"
    # output_file_name = "sample_data/relive_wild_qpos.pkl"
    # output_file_name = "sample_data/relive_ar_qpos.pkl"
    # output_file_name = "sample_data/relive_third_qpos.pkl"
    # output_file_name = "/hdd/zen/data/copycat/seqs/AIST++/aist_qpos.pkl"
    # output_file_name = "sample_data/egopose_mocap_qpos_grad.pkl"
    # output_file_name = "sample_data/h36m_train_30_qpos.pkl"
    # output_file_name = "sample_data/h36m_test_30_qpos.pkl"
    # output_file_name = "sample_data/h36m_train_30_qpos_test.pkl"
    # output_file_name = "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_qpos_full.p"
    # output_file_name = "sample_data/prox_sample.pkl"
    output_file_name = "sample_data/dais_sample.pkl"
    print(output_file_name, len(data_res))
    joblib.dump(data_res, open(output_file_name, "wb"))
