import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
import pickle as pk
import argparse
import glfw
import math
from uhc.khrylib.utils import *

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_mesh")
parser.add_argument("--offset_z", type=float, default=0.0)
parser.add_argument("--start_take", type=str, default="S1,Directions 1")
args = parser.parse_args()

model_file = f"assets/mujoco_models/{args.model_id}.xml"
model = load_model_from_path(model_file)
sim = MjSim(model)

viewer = MjViewer(sim)

glfw.set_window_size(viewer.window, 10, 10)
glfw.set_window_pos(viewer.window, 400, 0)


T = 10
paused = False
stop = False
reverse = False
offset_z = args.offset_z
viewer._hide_overlay = True
viewer.cam.distance = 10
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
# viewer.custom_key_callback = key_callback


def update_mocap():
    sim.data.qpos[:] = qpos_all[fr % qpos_all.shape[0]]
    # sim.data.qpos[7:] = 0
    sim.data.qpos[7]
    sim.data.qpos[2] += offset_z
    sim.forward()


qpos_all, trans = joblib.load("smpl_temp.npy")
qpos_all = smpl_to_qpose(qpos_all, model)
qpos_all[:, :3] = trans

# amass_data[:,2] = 0.91437225
# kin_qpose = joblib.load(f"/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/traj/1213_take_34_traj.p")
# print(kin_qpose.shape)
# rots = np.repeat(np.array([1, 0, 0, 0])[None, :], repeats = kin_qpose.shape[0], axis = 0)
# rots = np.repeat(np.array([0.7071, 0.7071, 0, 0])[None, :], repeats = kin_qpose.shape[0], axis = 0)

# test_data = joblib.load("sample_data/relive_all_smpl.pkl")
# amass_data = test_data['sit']['1001_take_01']
# smpl_pose = amass_data['pose_aa']
# trans = amass_data['trans']


print(qpos_all.shape)

t = 0
fr = 0
while not stop:
    # import pdb
    # pdb.set_trace()
    if t >= math.floor(T):
        update_mocap()
        fr += 1
        t = 0

    viewer.render()
    if not paused:
        t += 1
