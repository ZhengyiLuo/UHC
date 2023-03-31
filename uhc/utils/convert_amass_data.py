import os
import sys

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
import pickle
import argparse
from scipy.spatial.transform import Rotation as sRot
import joblib
from tqdm import tqdm

from data_process.h36m_specs import *
from data_process.smpl import SMPL_Parser, SMPL_BONE_ORDER_NAMES
from uhc.khrylib.utils import *
from uhc.khrylib.utils.torch_geometry_transforms import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from uhc.khrylib.utils.transformation import quaternion_from_euler
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.pose import interpolated_traj


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral")
parser.add_argument("--mocap_fr", type=int, default=50)
parser.add_argument("--dt", type=float, default=1 / 30)
parser.add_argument("--render", action="store_true", default=False)
parser.add_argument("--fname", type=str, default="data_amass_v1")
args = parser.parse_args()

model_file = "assets/mujoco_models/%s.xml" % args.model_id
model = load_model_from_path(model_file)
body_qposaddr = get_body_qposaddr(model)
# orig_data = pickle.load(open(os.path.expanduser('~/datasets/h36m/data_pose_h36m.p'), 'rb'))
amass_db = joblib.load("sample_data/amass_db.pt")


def smpl_to_qpose(pose, trans, joint_idx):
    """
    Expect pose to be batch_size x 72
    """
    pose = torch.tensor(pose)
    trans = torch.tensor(trans)
    curr_pose_mat = angle_axis_to_rotation_matrix(pose.reshape(-1, 3)).reshape(
        pose.shape[0], -1, 4, 4
    )
    curr_spose = sRot.from_matrix(curr_pose_mat[:, :, :3, :3].reshape(-1, 3, 3).numpy())
    curr_spose_euler = curr_spose.as_euler("ZXY", degrees=False).reshape(
        curr_pose_mat.shape[0], -1
    )
    curr_spose_euler = curr_spose_euler.reshape(-1, 24, 3)[:, joint_idx, :].reshape(
        -1, 72
    )
    root_quat = rotation_matrix_to_quaternion(curr_pose_mat[:, 0, :3, :])

    curr_qpos = np.concatenate((curr_trans, root_quat, curr_spose_euler[:, 3:]), axis=1)
    return curr_qpos


joint_idx = [
    SMPL_BONE_ORDER_NAMES.index(q) for q in list(get_body_qposaddr(model).keys())
]
model = load_model_from_path(model_file)
sim = MjSim(model)
# viewer = MjViewer(sim)


qpos_data = {}
for idx, (k, v) in tqdm(enumerate(amass_db.items())):
    curr_pose = v["poses"]
    curr_trans = v["trans"]

    qpos_traj = smpl_to_qpose(curr_pose, curr_trans, joint_idx)
    subject = k.split("_")[0].strip() + "_" + k.split("_")[1].strip()
    action = k.split("_")[2].strip()
    if subject in qpos_data:
        qpos_data[subject][action] = qpos_traj
    else:
        qpos_data[subject] = {action: qpos_traj}

    if idx > 10:
        break
    if args.render:
        for i in range(qpos_traj.shape[0]):
            sim.data.qpos[:] = qpos_traj[i]
            sim.forward()
            viewer.render()

if not args.render:
    pickle.dump(qpos_data, open(os.path.expanduser(f"data/amass/{args.fname}.p"), "wb"))
