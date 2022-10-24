import os
import argparse
import numpy as np
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.skeleton import Skeleton
from data_process.h36m_specs import *

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=True)
parser.add_argument('--template_id', type=str, default='humanoid_template')
parser.add_argument('--model_id', type=str, default='human36m_orig')
args = parser.parse_args()

template_file = 'assets/mujoco_models/template/%s.xml' % args.template_id
model_file = 'assets/mujoco_models/%s.xml' % args.model_id
skeleton = Skeleton()
skeleton.load_from_offsets(offsets, parents, 0.01, exclude_bones, channels, spec_channels)
print(template_file)
skeleton.write_xml(model_file, template_file, offset=np.array([0, 0, 1]))

# model = load_model_from_path(model_file)
# sim = MjSim(model)
# viewer = MjViewer(sim)
# sim.data.qpos[:] = 0
# sim.data.qpos[2] = 1.0
# sim.forward()

# while args.render:
#     viewer.render()
