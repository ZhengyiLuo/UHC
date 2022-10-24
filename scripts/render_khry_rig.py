import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from mujoco_py import load_model_from_path, MjSim, MjRenderContextOffscreen
import pickle as pk
import argparse
import glfw
import math
from uhc.khrylib.utils import *

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose
from uhc.utils.image_utils import write_frames_to_video
import joblib
from collections import defaultdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='humanoid_1205_v1')
    parser.add_argument('--offset_z', type=float, default=0.0)
    parser.add_argument('--start_take', type=str, default='S1,Directions 1')
    parser.add_argument('--output', type=str, default='test')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model_file = f'assets/mujoco_models/{args.model_id}.xml'
    model = load_model_from_path(model_file)
    sim = MjSim(model)

    viewer = MjRenderContextOffscreen(sim)

    # glfw.set_window_size(viewer.window, 10, 10)
    # glfw.set_window_pos(viewer.window, 400, 0)

    T = 10
    paused = False
    stop = False
    reverse = False
    offset_z = args.offset_z
    viewer._hide_overlay = True
    viewer.cam.distance = 6
    viewer.cam.elevation = -20
    viewer.cam.azimuth = 90
    # viewer.custom_key_callback = key_callback
    size = (960, 480)

    frame_rate = 30
    # data_dict = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_smpl_grad.pkl")

    h36m_qpos = pk.load(open("/hdd/zen/dev/ActMix/actmix/DataGen/MotionSyn/data/h36m/data_qpos_h36m.p", "rb"))
    data_dict = defaultdict(dict)
    for sub, action_sub in h36m_qpos.items():
        for action, action_qpos in action_sub.items():
            data_dict[f'{sub}-{action}']['khry_qpos'] = action_qpos

    for k, v in data_dict.items():
        khry_qposes = v["khry_qpos"]

        if khry_qposes.shape[1] == 73:
            khry_qposes = khry_qposes[:, 14:]
        elif khry_qposes.shape[1] == 66:
            khry_qposes = khry_qposes[:, 7:]

        images = []
        print("Rendering: ", khry_qposes.shape)
        for fr in range(khry_qposes.shape[0]):
            sim.data.qpos[:] = khry_qposes[fr]
            sim.data.qpos[2] += offset_z
            sim.forward()
            viewer.render(size[0], size[1])
            data = np.asarray(viewer.read_pixels(size[0], size[1], depth=False)[::-1, :, :], dtype=np.uint8)
            images.append(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

        write_frames_to_video(images, out_file_name = "{}/{}.mp4".format(args.output, k), frame_rate = frame_rate, add_text = None)


        