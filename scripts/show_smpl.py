import os
import sys
sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose
import pickle
import argparse
import glfw
import math
import joblib

def key_callback(key, action, mods):
    global T, fr, paused, stop, offset_z, take_ind, reverse, repeat

    if action != glfw.RELEASE:
        return False
    elif key == glfw.KEY_D:
        T *= 1.5
    elif key == glfw.KEY_F:
        T = max(1, T / 1.5)
    elif key == glfw.KEY_Q:
        stop = True
    elif key == glfw.KEY_R:
        repeat = not repeat
    elif key == glfw.KEY_W:
        fr = 0
        update_mocap()
    elif key == glfw.KEY_S:
        reverse = not reverse
    elif key == glfw.KEY_C:
        take_ind = (take_ind + 1) % len(takes)
        load_take()
        update_mocap()
    elif key == glfw.KEY_Z:
        take_ind = (take_ind - 1) % len(takes)
        load_take()
        update_mocap()
    elif key == glfw.KEY_RIGHT:
        if fr < qpos_traj.shape[0] - 1:
            fr += 1
        update_mocap()
    elif key == glfw.KEY_LEFT:
        if fr > 0:
            fr -= 1
        update_mocap()
    elif key == glfw.KEY_UP:
        offset_z += 0.001
        update_mocap()
    elif key == glfw.KEY_DOWN:
        offset_z -= 0.001
        update_mocap()
    elif key == glfw.KEY_SPACE:
        paused = not paused
    else:
        return False
    return True


def update_mocap():
    sim.data.qpos[:] = qpos_traj[fr]
    sim.data.qpos[2] += offset_z
    sim.forward()


def load_take():
    global qpos_traj, fr, take
    take = takes[take_ind]
    fr = 0
    # qpos_traj = smpl_6d_to_qpose(take['pose'], model)
    # qpos_traj = smpl_to_qpose(take['pose'], model, take['tran'], True, random_root = True, euler_order="zxy")
    print(take.keys())
    qpos_traj = smpl_to_qpose(take['pose_aa'], model, take['trans'], True, random_root = False)
    print("Current Annotation:", take['seq_name'], f"length: {take['pose_aa'].shape[0]}")



parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh')
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--start_take', type=str, default=None)
parser.add_argument('--dataset', type=str, default='h36m/data_qpos_h36m')
args = parser.parse_args()

model_file = f'assets/mujoco_models/{args.model_id}.xml'
print(model_file)
model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)

# amass_lag_data = joblib.load("/hdd/zen/data/ActBound/Language/amass_lag_take3_sal.pkl")
amass_lag_data = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take1_test.pkl")
takes = list(amass_lag_data.values())
# takes = [v for v in takes if "back" in v['text']]


qpos_traj = None
take = None
take_ind = 0 if args.start_take is None else takes.index(tuple(args.start_take.split(',')))
fr = 0
offset_z = args.offset_z
# load_take()

T = 10
paused = False
stop = False
reverse = False
repeat = True
glfw.set_window_size(viewer.window, 1000, 960)
glfw.set_window_pos(viewer.window, 400, 0)
viewer._hide_overlay = True
viewer.cam.distance = 10
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
viewer.custom_key_callback = key_callback

load_take()
update_mocap()
t = 0
while not stop:
    if t >= math.floor(T):
        if not reverse:
            if repeat:
                fr = (fr + 1) % qpos_traj.shape[0]
            elif fr < qpos_traj.shape[0] - 1:
                fr += 1

            update_mocap()
        elif reverse :
            if repeat:
                fr = (fr - 1) % qpos_traj.shape[0]
            elif fr > 0:
                fr -= 1
            update_mocap()

        
        t = 0

    viewer.render()
    if not paused:
        t += 1



