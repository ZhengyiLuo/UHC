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
import joblib

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose
from scipy.spatial.transform import Rotation as Rot

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_double')
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--start_take', type=str, default='S1,Directions 1')
args = parser.parse_args()

model_file = f'assets/mujoco_models/{args.model_id}.xml'
model = load_model_from_path(model_file)
smpl_model_file = 'assets/mujoco_models/humanoid_smpl_neutral.xml'
smpl_model = load_model_from_path(smpl_model_file)
sim = MjSim(model)
print(model._body_name2id)
viewer = MjViewer(sim)

glfw.set_window_size(viewer.window, 10, 10)
glfw.set_window_pos(viewer.window, 400, 0)



T = 5
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
    ## 23 * 3 + 7 + 59
    # sim.data.qpos[:] = amass_data[fr % amass_data.shape[0]]
    sim.data.qpos[:] = 0
    sim.data.qpos[3:7] = Rot.from_euler("xyz", np.array([np.pi,0,np.pi/2])).as_quat()
    sim.data.qpos[2] += offset_z
    
    sim.forward()

    tpose["smpl_j3d"] = sim.data.body_xpos[1:25].copy()
    tpose["humanoid_j3d"] = sim.data.body_xpos[25:].copy()

# amass_data = pk.load(open("data/rendering/test_data.pkl", "rb"))
amass_data = pk.load(open("/hdd/zen/data/ActBound/Language/netural_pose.pkl", "rb"))
amass_data = smpl_6d_to_qpose(amass_data.squeeze(), smpl_model)
amass_data[:,2] = 0.91437225
# print(amass_data.shape)


t = 0
fr = 0
tpose = {
    "smpl_j3d" : 0 ,
    "humanoid_j3d" : 0
}

while not stop:
    # import pdb
    # pdb.set_trace()
    if t >= math.floor(T):
        update_mocap()
        fr += 1
        t = 0
        out_file_name = f"/hdd/zen/dev/reallite/ContextEgoPose/results/smpl_t_pose.pkl"
        joblib.dump(tpose, out_file_name)

    viewer.render()
    if not paused:
        t += 1
    



