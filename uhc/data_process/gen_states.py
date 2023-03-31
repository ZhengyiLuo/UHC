import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import numpy as np
import pickle as pk
from tqdm.notebook import tqdm
from collections import defaultdict
from tqdm import tqdm
import joblib

from uhc.envs.humanoid_im import HumanoidEnv
from uhc.utils.config_utils.copycat_config import Config
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.models.mlp import MLP
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle

def run_seq(seqs):
    states_acc = defaultdict(list)
    for k in tqdm(seqs):
        curr_expert = data_loader.pickle_data[k]['expert']
        env.load_expert(curr_expert)
        seq_qpos = curr_expert["qpos"]
        seq_qvel = curr_expert["qvel"]
        states = []
        env.start_ind = 0

        for i in range(curr_expert['len'] - data_loader.t_min):
            env.cur_t = i
            env.set_state(seq_qpos[i], seq_qvel[i])
            state = env.get_obs()
            states.append(state)
        states_acc[k] = np.array(states)
    return states_acc


if __name__ == "__main__":
    # cfg = Config("copycat_19", False, create_dirs=False)
    cfg = Config("copycat_9", False, create_dirs=False)
    # cfg.data_specs['test_file_path'] = "sample_data/amass_copycat_take3_test.pkl"
    # cfg.data_specs['test_file_path'] = "sample_data/amass_copycat_take3.pkl"
    cfg.data_specs['test_file_path'] = "sample_data/relive_copycat.pkl"
    data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="test")
    init_expert = data_loader.sample_seq()
    env = HumanoidEnv(cfg, init_expert = init_expert, data_specs = cfg.data_specs, mode="test")

    

    jobs = list(data_loader.pickle_data.keys())
    
    # np.random.shuffle(jobs)
    # jobs= jobs[:1000]

    data_res_full = {}

    from torch.multiprocessing import Pool
    num_jobs = 20
    chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    job_args = [(jobs[i],) for i in range(len(jobs))]
    print(len(job_args))
    try:
        pool = Pool(num_jobs)   # multi-processing
        job_res = pool.starmap(run_seq, job_args)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    
    [data_res_full.update(j) for j in job_res]
    joblib.dump(data_res_full, "sample_data/relive_copycat_states.pkl")
    # joblib.dump(data_res_full, "sample_data/amass_copycat_take3_states_obs_v2.pkl")
    # joblib.dump(data_res_full, "sample_data/amass_copycat_take3_test_states.pkl")
    

