import os
import glob
import argparse
import cdflib
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--raw_dir', type=str, default='~/datasets/h36m/raw')
parser.add_argument('--save_dir', type=str, default='~/datasets/h36m')
args = parser.parse_args()

raw_dir = os.path.expanduser(args.raw_dir)
save_dir = os.path.expanduser(args.save_dir)
data = {}
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
for subject in subjects:
    data[subject] = {}
    file_list = glob.glob(f'{raw_dir}/{subject}/MyPoseFeatures/D3_Angles/*.cdf')
    file_list.sort()
    for file in file_list:
        action = os.path.splitext(os.path.basename(file))[0]
        cdf = cdflib.CDF(file)
        poses = cdf.varget("Pose").squeeze(0)
        data[subject][action] = poses
pickle.dump(data, open(f'{save_dir}/data_pose_h36m.p', 'wb'))
