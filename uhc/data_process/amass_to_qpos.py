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


def sample_seq_length(seq, tran, seq_length=150):
    if seq_length != -1:
        num_possible_seqs = seq.shape[0] // seq_length
        max_seq = seq.shape[0]

        start_idx = np.random.randint(0, 10)
        start_points = [max(0, max_seq - (seq_length + start_idx))]

        for i in range(1, num_possible_seqs - 1):
            start_points.append(i * seq_length + np.random.randint(-10, 10))

        if num_possible_seqs >= 2:
            start_points.append(max_seq - seq_length -
                                np.random.randint(0, 10))

        seqs = [seq[i:(i + seq_length)] for i in start_points]
        trans = [tran[i:(i + seq_length)] for i in start_points]
    else:
        seqs = [seq]
        trans = [tran]
        start_points = []
    return seqs, trans, start_points


def process_qpos_list(qpos_list):
    amass_res = {}
    counter = 0
    pbar = tqdm(qpos_list)
    for (k, v) in pbar:
        pbar.set_description(k)
        amass_pose = v["poses"]
        amass_trans = v["trans"]
        betas = v["betas"]
        gender = v["gender"]
        seq_length = amass_pose.shape[0]

        if seq_length < 10:
            continue

        amass_fr = v["mocap_framerate"]
        # skips = np.unique([int(amass_fr/target_fr) for target_fr in target_frs]).astype(int)

        pose_aa = torch.tensor(amass_pose)
        curr_trans = amass_trans

        pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa)).reshape(
            -1, 144)
        qpos = smpl_to_qpose(pose=pose_aa,
                             mj_model=humanoid_model,
                             trans=curr_trans)
        qpos[:,
             3:7] /= np.linalg.norm(qpos[:, 3:7],
                                    axis=1)[:,
                                            None]  ## Normalize qpos quaternion

        expert_meta = {"cyclic": False, "seq_name": k}
        expert_res = get_expert(qpos, expert_meta, env)
        if not expert_res is None:
            amass_res[k] = {
                "pose_aa": pose_aa.numpy(),
                "pose_6d": pose_seq_6d.numpy(),
                "qpos": qpos,
                "trans": curr_trans,
                "beta": betas[:10],
                "gender": gender,
                "expert": expert_res,
                "mocap_framerate": amass_fr,
            }
            counter += 1
            # if counter > 1:
            #     break

    return amass_res


if __name__ == "__main__":
    amass_base = "sample_data/"
    # amass_cls_data = pk.load(open(os.path.join(amass_base, "amass_class.pkl"), "rb"))
    amass_seq_data = {}
    seq_length = -1
    cfg = Config(cfg_id="copycat_5", create_dirs=False)

    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = HumanoidEnv(cfg,
                      init_expert=init_expert,
                      data_specs=cfg.data_specs,
                      mode="test")

    # target_frs = [20,30,40] # target framerate
    target_frs = [30]  # target framerate
    counter = 0
    seq_counter = 0
    amass_db = joblib.load("sample_data/amass_db.pt")

    model_file = f"assets/mujoco_models/humanoid_smpl_neutral_mesh.xml"
    humanoid_model = load_model_from_path(model_file)
    all_data = list(amass_db.items())
    np.random.shuffle(all_data)

    from multiprocessing import Pool

    num_jobs = 20
    jobs = all_data
    chunk = np.ceil(len(jobs) / num_jobs).astype(int)
    jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i], ) for i in range(len(jobs))]
    print(len(job_args))

    try:
        pool = Pool(num_jobs)  # multi-processing
        job_res = pool.starmap(process_qpos_list, job_args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    except Exception as e:
        import ipdb

        ipdb.set_trace()
    # job_res = [process_qpos_list(all_data)]
    import ipdb

    ipdb.set_trace()

    # try:
    #     [amass_seq_data.update(j) for j in job_res]
    #     # amass_output_file_name = "sample_data/amass_qpos_30.pkl"
    #     # amass_output_file_name = "sample_data/amass_qpos_test.pkl"
    #     amass_output_file_name = "sample_data/amass_qpos.pkl"
    #     # amass_output_file_name = "sample_data/amass_{}.pkl".format(take_num)
    #     print(amass_output_file_name, len(amass_seq_data))
    #     joblib.dump(amass_seq_data, open(amass_output_file_name, "wb"),  compress=9, protocol=4)
    # except:
    #     import ipdb; ipdb.set_trace()
