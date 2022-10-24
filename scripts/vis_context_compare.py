import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import pickle as pk
import argparse
import glfw
import math
import torch
from mujoco_py import load_model_from_path, MjSim

from uhc.khrylib.utils import *
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose
from scipy.spatial.transform import Rotation as Rot
from uhc.utils.transform_utils import vertizalize_smpl_root

from scripts.cal_context_compare import take_names


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

smpl_start = 0
smpl_end  = 76

# viewer.custom_key_callback = key_callback

def update_mocap():
    ## [7 + 23 * 3]  + [7 + 14 * 3]
    ## [1 + 23 + 22]
    sim.data.qpos[:smpl_end] = smpl_qpose[fr % smpl_qpose.shape[0]]
    sim.data.qpos[smpl_end:] = kin_qpose[fr % kin_qpose.shape[0]]
    

    sim.data.qpos[2] += offset_z
    sim.forward()

def get_heading_q(q):
    hq = q.copy()
    hq[1] = 0.0
    hq[2] = 0.0
    hq /= np.linalg.norm(hq)
    return hq

# amass_data = pk.load(open("data/rendering/test_data.pkl", "rb"))
# amass_data = pk.load(open("/hdd/zen/data/ActBound/Language/netural_pose.pkl", "rb"))
# amass_data = smpl_6d_to_qpose(amass_data.squeeze(), smpl_model)
# amass_data[:,2] = 0.91437225


meva_data = joblib.load("/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/meva/meva_res.pkl")


# take_name = "08-30-2020-16-16-49"
# take_name = "08-30-2020-16-20-24"
take_name = "08-30-2020-17-10-20"
# take_name = "08-18-2020-13-26-14" ..... doesnot work
obj_offset = 7
action = 'sit'

# take_name = '08-30-2020-16-50-15' 
# take_name =  "08-30-2020-16-54-49"
# take_name =  "08-30-2020-17-04-03"
# take_name =  "08-30-2020-17-05-22"
# # take_name = "08-30-2020-17-07-25"
# obj_offset = 14
# action = 'push'

# take_name = '08-30-2020-16-37-23' 
# take_name = "08-30-2020-16-38-44"
# obj_offset = 7
# action = 'avoid'


### Kinematic
# kin_exp = joblib.load(f"/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/fpv_slams/{take_name}_traj.p")
# kin_qpose = kin_exp['kin_expert']['qpos'][:,obj_offset:]
# fr_margin = 0

### FIT
traj_pred, _ = joblib.load(f"/hdd/zen/dev/reallite/ContextEgoPose/results/{action}/egomimic/subject_04/ft/{take_name}/results/iter_3000_ft_cam.p")
kin_qpose = traj_pred['traj_pred'][take_name][:,obj_offset:]
fr_margin = 5

slam_data = joblib.load(f"/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/traj/{take_name}_cam_traj.p")
slam_pos = slam_data


tpv_dict = joblib.load("/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/meva/context_tpv_take1_dict.pkl")
meva_key = tpv_dict[take_name]['tpv_name']
# print()
capture_offset = tpv_dict[take_name]['offset']

take_meta = take_names[take_name]
take_start = take_meta['offset']
take_end = take_meta['end']

smpl_pose = meva_data[meva_key]['thetas_smoothed']
smpl_j3d = meva_data[meva_key]['j3ds']
smpl_cam = meva_data[meva_key]['cam']
smpl_pose = vertizalize_smpl_root(torch.tensor(smpl_pose))
smpl_qpose = smpl_to_qpose(smpl_pose, smpl_model, normalize=True)

capture_start = take_start+capture_offset
# smpl_qpose[:,:3] = smpl_cam[:,:3]
smpl_head_root_dif =   smpl_j3d[:,0] -  smpl_j3d[:,15]
# smpl_head_root_dif[:,[2]] *= -1
smpl_head_root_dif = smpl_head_root_dif[capture_start:capture_start+kin_qpose.shape[0]]

smpl_qpose = smpl_qpose[capture_start:capture_start+kin_qpose.shape[0]]

# smpl_qpose[:,[2]] = kin_qpose[:,[2]]
# smpl_qpose[:,:2] = kin_qpose[:,:2]
smpl_head_root_dif = smpl_head_root_dif[:,[2,0,1]]

slam_pos = slam_pos[take_start + fr_margin:-take_end - 2 - fr_margin][:,[0,2,1]] # Head position
smpl_qpose[:,:3] = slam_pos[:,:3]  - smpl_head_root_dif 

neutral_height = 0.91437225
heaight_adjust =  neutral_height - smpl_qpose[0,2]
smpl_qpose[:,2]  += heaight_adjust

# 2: up and down
kin_init_heading_quat = kin_qpose[fr_margin, 3:7]
kin_init_euler = Rot.from_quat(kin_init_heading_quat).as_euler("xyz")
smpl_init_euler = Rot.from_quat(smpl_qpose[fr_margin][3:7]).as_euler("xyz")
heading_delta =  kin_init_euler[0] - smpl_init_euler[0] # Synced up initial heading  
print(meva_key, capture_start)
for i in range( smpl_qpose.shape[0]):

    kin_init_heading_quat = kin_qpose[i, 3:7]
    kin_init_euler = Rot.from_quat(kin_init_heading_quat).as_euler("xyz")
    root_quat = Rot.from_euler( "xyz",kin_init_euler + np.array([0, 0, np.pi/2])).as_quat()
    smpl_qpose[i][3:7] = root_quat

    # smpl_quat = smpl_qpose[i][3:7]
    # root_euler = Rot.from_quat(smpl_quat).as_euler("xyz")
    # root_euler[0] += heading_delta
    # root_quat = Rot.from_euler( "xyz",root_euler).as_quat()
    # smpl_qpose[i][3:7] = root_quat

# print(amass_data.shape)

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
        



