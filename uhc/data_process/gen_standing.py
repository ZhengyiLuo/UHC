import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import pdb

from uhc.utils.config_utils.copycat_config import Config
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.utils.tools import get_expert

if __name__ == "__main__":
    
    cfg = Config("copycat_1", "train", create_dirs=False)
    env = HumanoidEnv(cfg)
    amass_cc = joblib.load("sample_data/amass_copycat_take1.pkl")
    amass_data = amass_cc['0-ACCAD_Male1General_c3d_General A1 - Stand_poses']
    neutral_dump = {}

    expert_meta = {
            "cyclic": False,
            "seq_name": "standing"
        }

    neutral_dump['pose_aa'] = amass_data['pose_aa']
    neutral_dump['pose_6d'] = amass_data['pose_6d']
    expert = get_expert(amass_data['qpos'], expert_meta, env)
    neutral_dump['qpos'] = expert['qpos'][10, :]
    neutral_dump['qpos'][2] = 0.91437225
    neutral_dump['qvel'] = expert['qvel'][10, :]

    joblib.dump(neutral_dump, "sample_data/standing_neutral.pkl")

