import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import pickle as pk
import argparse
import math
import numpy as np
import cv2
import joblib
from tqdm import tqdm
import torch

# from uhc.khrylib.utils import *
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from mujoco_py import load_model_from_path, MjSim, MjRenderContextOffscreen
from uhc.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Renderer
from uhc.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root,
    rotation_matrix_to_angle_axis, convert_orth_6d_to_mat
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh')
    parser.add_argument('--offset_z', type=float, default=0.0)
    parser.add_argument('--start_take', type=str, default='S1,Directions 1')
    parser.add_argument('--data', type=str, default='test_data')
    parser.add_argument('--output', type=str, default='test')
    args = parser.parse_args()

    model_file = f"assets/mujoco_models/{args.model_id}.xml"
    smpl_renderer = SMPL_M_Renderer(model_file=model_file)

    amass_lag_data = joblib.load("/hdd/zen/data/ActBound/Language/amass_lag_take2.pkl")
    i = 0
    for k, v in tqdm(amass_lag_data.items()):
        text = v['text']
        full_pose = v['pose']
        X_r = convert_orth_6d_to_aa(torch.tensor(full_pose[:,3:]))
        tran = full_pose[:,:3]
        print("Rendering: ", text)
        smpl_renderer.render_smpl(body_pose = X_r, tran = tran, output_name="rendering/{}_{}.mp4".format(args.output, i), frame_rate=30, add_text = text)
        i += 1
        
        