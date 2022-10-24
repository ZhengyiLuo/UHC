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

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='humanoid_1205_v1')
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--start_take', type=str, default='S1,Directions 1')
args = parser.parse_args()

model_file = f'assets/mujoco_models/{args.model_id}.xml'
model = load_model_from_path(model_file)
sim = MjSim(model)

viewer = MjViewer(sim)

glfw.set_window_size(viewer.window, 10, 10)
glfw.set_window_pos(viewer.window, 400, 0)



T = 20
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
    # print(fr % kin_qpose.shape[0])
    sim.data.qpos[:] = kin_qpose[fr % kin_qpose.shape[0]]
    # sim.data.qpos[3:7] = np.array([0,0,0,1])
    # sim.data.qpos[:] = kin_qpose[80]
    # print(sim.data.qpos[38:41])
    # sim.data.qpos[38:41] = 0
    # print(sim.data.qpos[55:56], sim.data.qpos[48:49])
    # sim.data.qpos[55:56] = 0
    # sim.data.qpos[48:49] = 0
    sim.data.qpos[2] += offset_z
    sim.forward()

# amass_data = pk.load(open("data/rendering/test_data.pkl", "rb"))
# kin_qpose = joblib.load(f"/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/traj/1011_take_05_traj.p")
# kin_qpose = kin_qpose[:,7:]
# kin_qpose = joblib.load(f"/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/traj/1213_take_51_traj.p")
# kin_qpose = kin_qpose[:,14:]
kin_qpose, trans  = joblib.load("smpl_temp.npy")
# kin_qpose[:,7:] = 0
print(kin_qpose.shape)

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
        



