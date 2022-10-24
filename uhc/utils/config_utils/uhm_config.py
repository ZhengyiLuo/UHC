import yaml
import os
import os.path as osp
import glob
import numpy as np

from uhc.khrylib.utils import recreate_dirs
from uhc.utils.config_utils.base_config import Base_Config

class Config(Base_Config):

    def __init__(self, mujoco_path = '%s.xml', **kwargs):
        super().__init__( **kwargs)

        self.mujoco_model_file = mujoco_path % self.cfg_dict['mujoco_model']

        
        self.policy_optimizer = self.cfg_dict['policy_optimizer']
        self.policy_specs = self.cfg_dict.get("policy_specs", {})
        self.scene_specs = self.cfg_dict.get("scene_specs", {})
        self.scene_mujoco_file = mujoco_path % self.scene_specs.get("scene_mujoco_file", "humanoid_smpl_neutral_mesh_all_h36m")
        self.cc_cfg = self.policy_specs.get("cc_cfg", "copycat_9")
        self.agent_name = self.cfg_dict.get("agent_name", "agent_uhm")
        self.model_name = self.model_specs.get("model_name", "kin_net")
        self.policy_name = self.policy_specs.get("policy_name", "kin_policy")
        self.env_name = self.scene_specs.get("env_name", "humanoid_kin_v1")
        ## Model Specs
        self.autoregressive = self.model_specs.get("autoregressive", True)
        self.remove_base = self.model_specs.get("remove_base", True)

        # Policy Specs
        self.reward_weights = self.policy_specs.get("reward_weights", {})
        self.env_term_body = self.policy_specs.get("env_term_body", "body")
        self.env_episode_len = self.policy_specs.get("env_episode_len", "body")
        self.obs_vel = self.cfg_dict.get('obs_vel', 'full')

        ## Data Specs
        self.fr_num = self.data_specs.get("fr_num", 80)
     
        