'''
File: /train_ar_policy.py
Created Date: Monday February 15th 2021
Author: Zhengyi Luo
-----
Last Modified: Monday February 15th 2021 8:39:12 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
'''
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import glob
import wandb

from uhc.utils.flags import flags

from uhc.agents import agent_dict
from uhc.utils.config_utils.uhm_config import Config
from uhc.khrylib.utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_threads', type=int, default=30)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--show_noise', action='store_true', default=False)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = Config(cfg_id = args.cfg, create_dirs=not (args.epoch > 0), mujoco_path = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/%s.xml")
    cfg.update(args)
    flags.debug = args.debug
    if cfg.render or flags.debug: cfg.num_threads = 1

    if not args.no_log:
        wandb.init(project='uhm goal', resume=not args.resume is None , id=args.resume)
        wandb.config.update(cfg, allow_val_change = True)
        wandb.run.name = args.cfg
        wandb.run.save()

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available(
    ) else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    print(f"Using: {device}")

    agent = agent_dict[cfg.agent_name](cfg, dtype, device, mode = "train", checkpoint_epoch=args.epoch)
 
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if not cfg.test:
        for i_iter in range(cfg.epoch, cfg.num_epoch):
            agent.optimize_policy(i_iter)
            if cfg.save_n_epochs > 0 and (i_iter+1) % cfg.save_n_epochs == 0:
                agent.save_checkpoint(i_iter)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        print('training done!')
    else:
        if cfg.render:
            agent.eval_seq(0, agent.data_loader, render = True)
        else:
            res_dicts = agent.eval_policy(data_mode = "test")
            import ipdb; ipdb.set_trace()
