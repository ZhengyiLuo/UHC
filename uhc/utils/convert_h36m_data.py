import os
import sys

sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.pose import interpolated_traj
import pickle
import argparse
from data_process.h36m_specs import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='human36m_orig')
parser.add_argument('--mocap_fr', type=int, default=50)
parser.add_argument('--dt', type=float, default=1/30)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--fname', type=str, default='data_qpos_h36m_v2')
args = parser.parse_args()

model_file = 'assets/mujoco_models/%s.xml' % args.model_id
model = load_model_from_path(model_file)
body_qposaddr = get_body_qposaddr(model)
orig_data = pickle.load(open(os.path.expanduser('data/h36m/data_pose_h36m.p'), 'rb'))
offsets = {'S1': -0.025, 'S6': -0.117, 'S7': -0.02, 'S8': -0.045, 'S9': -0.075, 'S11': -0.067}


def get_qpos(pose):
    qpos = np.zeros(model.nq)
    for bone_name, ind2 in body_qposaddr.items():
        ind1 = bone_addr[bone_name]
        if ind1[0] == 0:
            trans = pose[ind1[0]:ind1[0] + 3] * 0.001
            angles = pose[ind1[0] + 3:ind1[1]]
            quat = quaternion_from_euler(angles[0], angles[1], angles[2], 'rzxy')
            qpos[ind2[0]:ind2[0] + 3] = trans
            qpos[ind2[0] + 3:ind2[1]] = quat
        else:
            angles = pose[ind1[0]:ind1[1]]
            qpos[ind2[0]:ind2[1]] = angles
    return qpos


# def angle_fix(poses, start_ind):
#     for i in range(1, poses.shape[0]):
#         diff = poses[i] - poses[i-1]
#         diff[:start_ind] = 0
#         poses[i:, diff > np.pi] -= 2 * np.pi
#         poses[i:, diff < -np.pi] += 2 * np.pi
#     return poses


def angle_fix(poses, start_ind):
    poses_t = poses[:, start_ind:]
    while np.any(poses_t > np.pi):
        poses_t[poses_t > np.pi] -= 2 * np.pi
    while np.any(poses_t < -np.pi):
        poses_t[poses_t < -np.pi] += 2 * np.pi
    return poses


def get_qpos_traj(poses):
    poses[:, 3:] = np.deg2rad(poses[:, 3:])
    poses = angle_fix(poses, 3)
    poses_samp = interpolated_traj(poses, args.dt, mocap_fr=args.mocap_fr)
    # poses_t = poses_samp[:, 3:]
    # p_min = poses_t.min(axis=0)
    # p_max = poses_t.max(axis=0)
    # print(poses_t.min(), poses_t.max(), (p_max - p_min))
    qpos_traj = []
    for i in range(poses_samp.shape[0]):
        cur_pose = poses_samp[i, :]
        cur_qpos = get_qpos(cur_pose)
        qpos_traj.append(cur_qpos)
    qpos_traj = np.vstack(qpos_traj)
    qpos_traj[:, 2] += offset_z
    return qpos_traj


model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)

qpos_data = {}
for subject, s_data in orig_data.items():
    qpos_data[subject] = {}
    for action, poses in s_data.items():
        offset_z = offsets.get(subject, 0.0)
        qpos_traj = get_qpos_traj(poses)
        qpos_data[subject][action] = qpos_traj

        if args.render:
            for i in range(qpos_traj.shape[0]):
                sim.data.qpos[:] = qpos_traj[i]
                sim.forward()
                viewer.render()

# if not args.render:
#     pickle.dump(qpos_data, open(os.path.expanduser(f'~/datasets/h36m/{args.fname}.p'), 'wb'))

