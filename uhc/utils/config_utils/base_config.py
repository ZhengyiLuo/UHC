import yaml
import os
import os.path as osp
import glob
import numpy as np
from uhc.khrylib.utils import recreate_dirs

class Base_Config:

    def __init__(self, cfg_id, base_dir = "", create_dirs=False, cfg_dict=None):
        self.id = cfg_id
        base_dir = base_dir if base_dir else ''
        self.base_dir = os.path.expanduser(base_dir)

        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = osp.join(self.base_dir, f"config/**/{cfg_id}.yml")
            files = glob.glob(cfg_path, recursive=True)
            assert(len(files) == 1)
            cfg_name = files[0]
            cfg = yaml.safe_load(open(cfg_name, 'r'))

        # create dirs
        self.cfg_dict = cfg
        self.main_result_dir = osp.join(self.base_dir, "results")
        self.proj_name = proj_name = cfg.get("proj_name", "motion_im")

        self.cfg_dir = osp.join(self.main_result_dir, proj_name, cfg_id)
        self.model_dir = osp.join(self.cfg_dir, "models")

        self.output_dir = self.result_dir = osp.join(self.cfg_dir, "results")
        self.log_dir = osp.join(self.cfg_dir, "log")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if create_dirs and not osp.exists(self.log_dir):
            recreate_dirs(self.log_dir)
        self.seed = self.cfg_dict.get('seed', 1)
        self.notes = cfg.get('notes', "exp notes")

        # group specs
        self.data_specs = cfg.get('data_specs', {})
        self.loss_specs = cfg.get('loss_specs', {})
        self.model_specs = cfg.get('model_specs', {})

        # Default training configs
        self.lr = self.cfg_dict.get("lr", 3.e-4)
        self.num_epoch = self.cfg_dict.get("num_epoch", 100)
        self.num_epoch_fix = self.cfg_dict.get("num_epoch_fix", 10)
        self.save_n_epochs = self.cfg_dict.get("save_n_epochs", 20)
        self.eval_n_epochs = self.cfg_dict.get("eval_n_epochs", 20)

        self.num_samples = self.data_specs.get("num_samples", 5000)
        self.batch_size = self.data_specs.get("batch_size", 5000)


    def get(self, key, default = None):
        return self.cfg_dict.get(key, default)

    def update(self, dict):
        for k, v in vars(dict).items():
            self.__setattr__(k, v)