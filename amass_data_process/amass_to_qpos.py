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
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, rot6d_to_rotmat
)
from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Viewer
from mujoco_py import load_model_from_path, MjSim
from uhc.utils.config import Config
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.utils.tools import get_expert
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle

np.random.seed(1)
left_right_idx = [ 0,  2,  1,  3,  5,  4,  6,  8,  7,  9, 11, 10, 12, 14, 13, 15, 17,16, 19, 18, 21, 20, 23, 22]

def sample_seq_length(seq, tran, seq_length = 150):
    if seq_length != -1:
        num_possible_seqs = seq.shape[0] // seq_length
        max_seq = seq.shape[0]

        start_idx = np.random.randint(0, 10)
        start_points = [max(0, max_seq - (seq_length + start_idx))]

        for i in range(1, num_possible_seqs - 1):
            start_points.append(i * seq_length + np.random.randint(-10, 10))

        if num_possible_seqs >= 2:
            start_points.append(max_seq - seq_length - np.random.randint(0, 10))

        seqs = [seq[i:(i + seq_length)] for i in start_points]
        trans = [tran[i:(i + seq_length)] for i in start_points]
    else:
        seqs = [seq]
        trans = [tran]
        start_points = []
    return seqs, trans, start_points

if __name__ == "__main__":
    amass_seq_data = {}
    seq_length = -1

    # target_frs = [20,30,40] # target framerate
    target_frs = [30] # target framerate
    video_annot = {}
    counter = 0 
    seq_counter = 0
    amass_db = joblib.load("/insert_directory_here/amass_db.pt")
    
    model_file = f'assets/mujoco_models/humanoid_smpl_neutral_mesh.xml'
    humanoid_model =load_model_from_path(model_file)
    all_data = list(amass_db.items())
    np.random.shuffle(all_data)
    pbar = tqdm(all_data)
    for (k, v) in pbar:
        pbar.set_description(k)
        amass_pose = v['poses']
        amass_trans = v['trans']
        betas = v['betas']
        gender = v['gender']
        seq_length = amass_pose.shape[0]
        
        if seq_length < 10:
            continue

        amass_pose_back = amass_pose[::-1].copy() 
        amass_trans_back = amass_trans[::-1].copy()
        amass_fr = v['mocap_framerate']
        
        
        # amass_pose_flip = flip_smpl(amass_pose)
        skips = np.unique([int(amass_fr/target_fr) for target_fr in target_frs]).astype(int)
        
        seqs, start_points, trans = [],[],[]
        for skip in skips:
            seqs_org, trans_org, start_points_org = sample_seq_length(amass_pose[::skip], amass_trans[::skip], seq_length)
            seqs = seqs + seqs_org 
            trans = trans + trans_org 
            start_points = start_points + start_points_org
            
        seq_counter += len(seqs)
        for idx in range(len(seqs)):
            pose_aa = torch.tensor(seqs[idx])      
            curr_trans = trans[idx]
            
            pose_seq_6d = convert_aa_to_orth6d(torch.tensor(pose_aa))
            pose_seq_6d = pose_seq_6d.reshape(pose_seq_6d.shape[0],-1,3)
   
            seq_name =  f"{idx}-{k}"

            expert_meta = {
                "cyclic": False,
                "seq_name": seq_name
            }
            expert_res = get_expert(qpos, expert_meta, env)
            if not expert_res is None:
                amass_seq_data[seq_name] = {
                    "pose_aa": pose_aa.numpy(),
                    "pose_6d":pose_seq_6d.numpy(),
                    'trans': curr_trans,
                    'beta': betas[:10],
                    "seq_name": seq_name,
                    "gender": gender
                }

                counter += 1

        # if counter > 1000:
            # break
    amass_output_file_name = "/insert_directory_here/amass_qpos_30.pkl"
    print(amass_output_file_name, len(amass_seq_data))
    joblib.dump(amass_seq_data, open(amass_output_file_name, "wb"))
