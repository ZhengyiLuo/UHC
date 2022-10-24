import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, load_model_from_xml, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
import pickle as pk
import argparse
import glfw
import math
import joblib
from uhc.khrylib.utils import *
from uhc.smpllib.smpl_robot import Robot

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose
from scipy.spatial.transform import Rotation as sRot
from uhc.utils.config_utils.copycat_config import Config


parser = argparse.ArgumentParser()
# parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_bigfoot")
parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_start")
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_masterfoot')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_vis')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_all_single_vis_plain')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_sit')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_push')
parser.add_argument("--offset_z", type=float, default=0.0)
parser.add_argument("--start_take", type=str, default="S1,Directions 1")
args = parser.parse_args()


def update_mocap():
    # sim.data.qpos[:76] = amass_data[0]
    sim.data.qpos[:] = amass_data[fr % amass_data.shape[0]]
    # sim.data.qpos[:] = smpl_qpos_2_masterfoot(amass_data[fr % amass_data.shape[0]])
    # sim.data.qpos[7:] = 0
    # sim.data.qpos[76:83] = obj_data[fr % amass_data.shape[0]]
    # sim.data.qpos[:76] = amass_data_gt[fr % amass_data.shape[0]]
    # sim.data.qpos[76:] = amass_data_gt[fr % amass_data.shape[0]]
    # sim.data.qpos[2] += offset_z
    sim.forward()
    # sim.step()


# model_file = f"assets/mujoco_models/{args.model_id}.xml"

cfg = Config(cfg_id="copycat_33", create_dirs=False)
smpl_model = "smpl"
cfg.robot_cfg["model"] = smpl_model
cfg.robot_cfg["mesh"] = True
smpl_robot = Robot(cfg.robot_cfg)

ngeom = 24
# model.geom_rgba[ngeom + 1: ngeom * 2 - 1] = np.array([1, 0, 0, 1])

amass_data_all = joblib.load(
    "/hdd/zen/data/ActBound/AMASS/amass_copycat_take5_test.pkl"
)
# amass_data_all = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_db_smplx_test.pt")
print(amass_data_all.keys())
key = "KIT_424_walking_slow06_poses"
# key = "KIT_8_WalkInClockwiseCircle02_poses"
# key = "TCD_handMocap_ExperimentDatabase_DirectionA_poses"
# key = "CMU_78_78_23_poses"
# key = "BioMotionLab_NTroje_rub056_0028_jumping1_poses"

pose_aa = amass_data_all[key]["pose_aa"]
trans = amass_data_all[key]["trans"]
beta = amass_data_all[key]["beta"]
gender = amass_data_all[key]["gender"]
# pose_aa = amass_data_all[key]["poses"]
# trans = amass_data_all[key]["trans"]
# beta = amass_data_all[key]["betas"]


# amass_raw = dict(
#     np.load(
#         "/hdd/zen/data/ActBound/AMASS/AMASS_Complete/BioMotionLab_NTroje/rub056/0028_jumping1_poses.npz"
#     )
# )
# pose_aa = amass_raw["poses"]
# trans = amass_raw["trans"]
# beta = amass_raw["betas"]
# gender = amass_raw["gender"]
if gender == "neutral":
    gender = [0]
elif gender == "male":
    gender = [1]
elif gender == "female":
    gender = [2]

smpl_robot.load_from_skeleton(
    betas=torch.tensor(
        beta[
            None,
        ]
    ).float(),
    gender=gender,
)
model = load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))
sim = MjSim(model)

viewer = MjViewer(sim)
glfw.set_window_size(viewer.window, 10, 10)
glfw.set_window_pos(viewer.window, 400, 0)

amass_data = smpl_to_qpose(
    pose=pose_aa, mj_model=model, trans=trans.squeeze(), model=smpl_model
)
print(amass_data.shape)
T = 10
paused = False
stop = False
reverse = False
offset_z = args.offset_z
viewer._hide_overlay = True
viewer.cam.distance = 5
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
# viewer.custom_key_callback = key_callback

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
