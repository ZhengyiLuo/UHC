'''
File: /agent_uhm.py
Created Date: Tuesday June 22nd 2021
Author: Zhengyi Luo
Comment:
-----
Last Modified: Tuesday June 22nd 2021 5:33:25 pm
Modified By: Zhengyi Luo at <zluo2@cs.cmu.edu>
-----
Copyright (c) 2021 Carnegie Mellon University, KLab
-----
'''

import joblib
import os.path as osp
import pdb
import sys
import glob
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from collections import defaultdict
import multiprocessing
import math
import time
import os
import torch
import wandb

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.rl.agents import AgentPPO
from uhc.khrylib.rl.core import estimate_advantages
from uhc.khrylib.utils.torch import *
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.rl.core import LoggerRL
from uhc.khrylib.rl.core.critic import Value
from uhc.khrylib.utils import get_eta_str
from uhc.utils.flags import flags
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from uhc.khrylib.utils.logger import create_logger

from uhc.envs import env_dict
from uhc.models import policy_dict
from uhc.data_loaders.dataset_amass_batch import DatasetAMASSBatch
from uhc.losses.uhm_rewards import reward_func


class AgentUHM(AgentPPO):
    def __init__(self, cfg, dtype, device, mode="train", checkpoint_epoch=0):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.iter = checkpoint_epoch

        self.setup_vars()
        self.setup_data_loader()
        self.setup_policy()
        self.setup_env()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
            self.load_curr()

        super().__init__(env=self.env,
                         dtype=dtype,
                         device=device,
                         running_state=None,
                         custom_reward=self.expert_reward,
                         mean_action=cfg.render and not cfg.show_noise,
                         render=cfg.render,
                         num_threads=cfg.num_threads,
                         data_loader=self.data_loader,
                         policy_net=self.policy_net,
                         value_net=self.value_net,
                         optimizer_policy=self.optimizer_policy,
                         optimizer_value=self.optimizer_value,
                         opt_num_epochs=cfg.policy_specs['num_optim_epoch'],
                         gamma=cfg.policy_specs['gamma'],
                         tau=cfg.policy_specs['tau'],
                         clip_epsilon=cfg.policy_specs['clip_epsilon'],
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         end_reward=cfg.policy_specs['end_reward'],
                         use_mini_batch=False,
                         mini_batch_size=0)
        if self.iter == 0:
            self.train_init()

    def setup_vars(self):
        cfg = self.cfg

        self.value_net = None
        self.cc_cfg = None
        self.env = None
        self.freq_dict = None
        self.data_loader = None
        self.test_data_loaders = []
        self.expert_reward = None
        self.running_state = None
        self.optimizer_value = None
        self.optimizer_policy = None
        self.policy_net = None

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger.info(
            f"========================== {type(self).__name__} ==========================="
        )
        self.logger.info(f"env_name: {cfg.env_name}")
        self.logger.info(f"policy_name: {cfg.policy_name}")
        self.logger.info(f"model_name: {cfg.model_name}")
        self.logger.info(
            f"sampling temp: {cfg.policy_specs.get('sampling_temp', 0.5)}")
        self.logger.info(
            f"sampling freq: {cfg.policy_specs.get('sampling_freq', 0.5)}")
        self.logger.info(
            f"init_update_iter: {cfg.policy_specs.get('num_init_update', 3)}")
        self.logger.info(
            f"step_update_iter: {cfg.policy_specs.get('num_step_update', 10)}")
        # self.logger.info(f"add_noise: {cfg.add_noise}")
        # self.logger.info(f"Data file: {cfg.data_file}")
        # self.logger.info(f"Feature Version: {cfg.use_of}")
        self.logger.info(
            "============================================================")

    def setup_policy(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        data_sample = self.data_loader.sample_seq()
        data_sample = {
            k: v.to(device).clone().type(dtype)
            for k, v in data_sample.items()
        }
        self.policy_net = policy_net = policy_dict[cfg.policy_name](
            cfg, data_sample, device=device, dtype=dtype, mode=self.mode)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        state_dim = self.policy_net.state_dim
        action_dim = self.env.action_space.shape[0]
        self.value_net = Value(
            MLP(state_dim, self.cc_cfg.value_hsize, self.cc_cfg.value_htype))
        to_device(device, self.value_net)

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        """load CC model"""

        self.cc_cfg = cc_cfg = CC_Config(
            cfg_id=cfg.cc_cfg, base_dir="")
        cc_cfg.mujoco_model_file = cfg.scene_mujoco_file

        with torch.no_grad():
            data_sample = self.data_loader.sample_seq()
            data_sample = {
                k: v.to(device).clone().type(dtype)
                for k, v in data_sample.items()
            }
            context_sample = self.policy_net.init_context(data_sample)
        self.env = env_dict[self.cfg.env_name](cfg,
                                               cc_cfg=cc_cfg,
                                               init_context=context_sample,
                                               cc_iter=cfg.policy_specs.get(
                                                   'cc_iter', -1),
                                               mode="train")
        self.env.seed(cfg.seed)

    def setup_optimizer(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if cfg.policy_specs.get("policy_optimizer", 'Adam'):
            self.optimizer_policy = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=cfg.policy_specs['policy_lr'],
                weight_decay=cfg.policy_specs['policy_weightdecay'])
        else:
            self.optimizer_policy = torch.optim.SGD(
                self.policy_net.parameters(),
                lr=cfg.policy_specs['policy_lr'],
                momentum=cfg.policy_specs['policy_momentum'],
                weight_decay=cfg.policy_specs['policy_weightdecay'])

        if cfg.policy_specs.get("value_optimizer", 'Adam'):
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(),
                lr=cfg.policy_specs['value_lr'],
                weight_decay=cfg.policy_specs['value_weightdecay'])
        else:
            self.optimizer_value = torch.optim.SGD(
                self.value_net.parameters(),
                lr=cfg.policy_specs['value_lr'],
                momentum=cfg.policy_specs['policy_momentum'],
                weight_decay=cfg.policy_specs['value_weightdecay'])

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        freq_path = osp.join(cfg.result_dir, "freq_dict.pt")
        # try:
        #     self.freq_dict = {k: [] for k in self.data_loader.data_keys} if not osp.exists(
        #         freq_path) else joblib.load(freq_path)
        # except:
        #     print("error parsing freq_dict, using empty one")

        self.freq_dict = {k: [] for k in self.data_loader.data_keys}
        self.logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'), True)

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[
            cfg.policy_specs['reward_id']]

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        log = info['log']

        c_info = log.avg_c_info
        log_str = f"Ep: {self.iter}\t {cfg.id} \tT_s {info['T_sample']:.2f}\t  \
                T_u { info['T_update']:.2f} \t expert_R_avg {log.avg_c_reward:.4f} {np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=',')} \
                \texpert_R_range ({log.min_c_reward:.4f}, {log.max_c_reward:.4f})\teps_len {log.avg_episode_len:.2f}"

        self.logger.info(log_str)
        if not cfg.no_log:
            wandb.log({
                "rewards": log.avg_c_info,
                "eps_len": log.avg_episode_len,
                "avg_rwd": log.avg_c_reward
            }, step=self.iter)

        if "log_eval" in info and not cfg.no_log:
            [wandb.log(test, step=self.iter) for test in info["log_eval"]]

    def pre_epoch_update(self, epoch):
        pass

    def optimize_policy(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.iter = epoch
        t0 = time.time()
        self.pre_epoch_update(epoch)
        batch, log = self.sample(cfg.policy_specs['min_batch_size'])

        if cfg.policy_specs['end_reward']:
            self.env.end_reward = log.avg_c_reward * cfg.policy_specs[
                'gamma'] / (1 - cfg.policy_specs['gamma'])
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            'log': log,
            'T_sample': t1 - t0,
            'T_update': t2 - t1,
            'T_total': t2 - t0
        }

        if (self.iter + 1) % 10 == 0:
            self.save_curr()

        if (self.iter + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(self.iter)
            log_eval = self.eval_policy("test")
            info['log_eval'] = log_eval

        self.log_train(info)
        joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def setup_data_loader(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.test_data_loaders = []
        # self.data_loader = data_loader = DatasetAMASSBatch(cfg, data_mode="train", seed = cfg.seed, multiproess = not cfg.debug)
        self.data_loader = data_loader = DatasetAMASSBatch(cfg,
                                                           data_mode="train",
                                                           seed=cfg.seed,
                                                           multiproess=True)
        self.test_data_loaders.append(
            DatasetAMASSBatch(cfg,
                              data_mode="test",
                              seed=cfg.seed,
                              multiproess=True))

    def load_checkpoint(self, i_iter):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if i_iter > 0:
            cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter)

            if osp.exists(cp_path):
                self.logger.info('loading model from checkpoint: %s' % cp_path)
                model_cp = pickle.load(open(cp_path, "rb"))
                self.policy_net.load_state_dict(model_cp['policy_dict'])

                self.value_net.load_state_dict(model_cp['value_dict'])
                self.running_state = model_cp['running_state']
                self.iter = i_iter
                if 'freq_dict' in model_cp:
                    self.freq_dict = model_cp['freq_dict']
                to_device(device, self.value_net)
            else:
                print("model does not exist, load curr!!!")
                self.load_curr()

    def save_checkpoint(self, i_iter):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        # self.tb_logger.flush()
        policy_net, value_net, running_state = self.policy_net, self.value_net, self.running_state
        with to_cpu(policy_net, value_net):
            cp_path = '%s/iter_%04d.p' % (cfg.model_dir, i_iter + 1)
            model_cp = {
                'policy_dict': policy_net.state_dict(),
                'value_dict': value_net.state_dict(),
                'running_state': running_state,
                'freq_dict': self.freq_dict
            }
            pickle.dump(model_cp, open(cp_path, 'wb'))

    def save_curr(self):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path_best = f"{cfg.model_dir}/iter_curr.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
                'freq_dict': self.freq_dict,
            }
            pickle.dump(model_cp, open(cp_path_best, "wb"))

    def load_curr(self):
        cfg = self.cfg
        cp_path_curr = f"{cfg.model_dir}/iter_curr.p"
        self.logger.info('loading model from checkpoint: %s' % cp_path_curr)
        model_cp = pickle.load(open(cp_path_curr, "rb"))
        self.policy_net.load_state_dict(model_cp['policy_dict'])

        self.value_net.load_state_dict(model_cp['value_dict'])
        self.running_state = model_cp['running_state']
        if 'freq_dict' in model_cp:
            self.freq_dict = model_cp['freq_dict']

        self.logger.info(
            f'loading model from checkpoint: {cp_path_curr} for {self.global_start_fr}'
        )

    def train_init(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.policy_net.train_full_supervised(
            cfg,
            self.data_loader,
            device=self.device,
            dtype=self.dtype,
            scheduled_sampling=0.3,
            num_epoch=self.cfg.policy_specs.get('warm_update_full', 50),
            eval_freq=self.cfg.policy_specs.get('warm_update_eval', 50),
            save_func=self.save_checkpoint)
        self.policy_net.setup_optimizers()

    def eval_policy(self, epoch=0, dump=False):
        cfg = self.cfg
        data_loaders = self.test_data_loaders

        res_dicts = []
        for data_loader in data_loaders:
            num_jobs = self.num_threads
            # num_jobs = 20
            jobs = data_loader.data_keys
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
            data_res_coverage = {}
            with to_cpu(*self.sample_modules):
                with torch.no_grad():
                    queue = multiprocessing.Queue()
                    for i in range(len(jobs) - 1):
                        worker_args = (jobs[i + 1], data_loader, queue, i)
                        worker = multiprocessing.Process(target=self.eval_seqs,
                                                         args=worker_args)
                        worker.start()
                    res = self.eval_seqs(jobs[0], data_loader, None)
                    data_res_coverage.update(res)
                    for i in tqdm(range(len(jobs) - 1)):
                        res = queue.get()
                        data_res_coverage.update(res)

                for k, res in data_res_coverage.items():
                    [
                        self.freq_dict[k].append([res["succ"], 0])
                        for _ in range(1 if res["succ"] else 3)
                        if k in self.freq_dict
                    ]

                metric_names = [
                    "pa_mpjpe",
                    "mpjpe",
                    "mpjpe_g",
                    "accel_dist",
                    "vel_dist",
                    "succ",
                    "reward",
                    "root_dist",
                    "pentration",
                    "skate",
                ]
                data_res_metrics = defaultdict(list)
                [[
                    data_res_metrics[k].append(v)
                    for k, v in res.items() if k in metric_names
                ] for k, res in data_res_coverage.items()]
                data_res_metrics = {
                    k: np.mean(np.concatenate(v)) if len(v[0]) > 0 else np.mean(v)
                    for k, v in data_res_metrics.items()
                }
                coverage = int(data_res_metrics["succ"] *
                               data_loader.get_len())
                print_str = " \t".join(
                    [f"{k}: {v:.3f}" for k, v in data_res_metrics.items()])

                self.logger.info(
                    f"Coverage {data_loader.name} of {coverage} out of {data_loader.get_len()} | {print_str}"
                )
                data_res_metrics.update({
                    "mean_coverage":
                    coverage / data_loader.get_len(),
                    "num_coverage":
                    coverage,
                    "all_coverage":
                    data_loader.get_len(),
                })
                del data_res_metrics["succ"]
                res_dicts.append(
                    {f"coverage_{data_loader.name}": data_res_metrics})

                if dump:
                    res_dir = osp.join(
                        cfg.output_dir,
                        f"{epoch}_{data_loader.name}_coverage_full.pkl")
                    print(res_dir)
                    joblib.dump(data_res_coverage, res_dir)
                    

        return res_dicts

    def eval_seqs(self, take_keys, data_loader, queue, id=0):
        res = {}
        for take_key in take_keys:
            res[take_key] = self.eval_seq(take_key, data_loader)

        if queue == None:
            return res
        else:
            queue.put(res)

    def eval_cur_seq(self):
        return self.eval_seq(self.fit_ind, self.data_loader)

    def eval_seq(self, take_key, loader):
        curr_env = self.env
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)
                self.policy_net.set_mode('test')
                curr_env.set_mode('test')
                context_sample = loader.get_sample_from_key(take_key=take_key,
                                                            full_sample=True,
                                                            return_batch=True)
                curr_env.load_context(
                    self.policy_net.init_context(context_sample))
                state = curr_env.reset()

                if self.running_state is not None:
                    state = self.running_state(state)
                for t in range(10000):
                    res['gt'].append(
                        curr_env.context_dict['qpos'][self.env.cur_t])
                    res['target'].append(curr_env.target['qpos'])
                    res['pred'].append(curr_env.get_humanoid_qpos())

                    state_var = tensor(state).unsqueeze(0).double()
                    trans_out = self.trans_policy(state_var)
                    action = self.policy_net.select_action(
                        trans_out, mean_action=True)[0].numpy()
                    action = int(
                        action
                    ) if self.policy_net.type == 'discrete' else action.astype(
                        np.float64)
                    next_state, env_reward, done, info = curr_env.step(action)

                    # c_reward, c_info = self.custom_reward(curr_env, state, action, info)
                    # res['reward'].append(c_reward)
                    if self.cfg.render:
                        curr_env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state)

                    if done:
                        res = {k: np.vstack(v) for k, v in res.items()}
                        # print(info['percent'], context_dict['ar_qpos'].shape[1], loader.curr_key, np.mean(res['reward']))
                        res['percent'] = info['percent']
                        return res
                    state = next_state

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

    def set_mode(self, mode):
        self.policy_net.set_mode(mode)
        self.env.set_mode(mode)

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls()
        self.policy_net.set_mode('test')
        self.env.set_mode('train')
        freq_dict = defaultdict(list)

        while logger.num_steps < min_batch_size:
            context_sample = self.data_loader.sample_seq(
                freq_dict=self.freq_dict,
                sampling_temp=self.cfg.policy_specs.get("sampling_temp", 0.5),
                sampling_freq=self.cfg.policy_specs.get("sampling_freq", 0.9))
            # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = self.cfg.policy_specs.get("sampling_temp", 0.5), sampling_freq = self.cfg.policy_specs.get("sampling_freq", 0.9), full_sample = True if self.data_loader.get_seq_len(self.fit_ind) < 1000 else False)
            # context_sample = self.data_loader.sample_seq(freq_dict = self.freq_dict, sampling_temp = 0.5)
            # context_sample = self.data_loader.sample_seq()

            # should not try to fix the height during training!!!
            context_dict = self.policy_net.init_context(context_sample,
                                                        fix_height=False)

            self.env.load_context(context_dict)
            state = self.env.reset()

            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or self.env.np_random.binomial(
                    1, 1 - self.noise_rate)

                action = self.policy_net.select_action(trans_out,
                                                       mean_action)[0].numpy()

                action = int(
                    action
                ) if self.policy_net.type == 'discrete' else action.astype(
                    np.float64)
                #################### ZL: Jank Code.... ####################
                gt_qpos = self.env.context_dict['qpos'][self.env.cur_t + 1]
                # gt_qpos = self.env.context_dict['ar_qpos'][self.env.cur_t + 1]
                curr_qpos = self.env.get_humanoid_qpos()
                #################### ZL: Jank Code.... ####################

                next_state, env_reward, done, info = self.env.step(action)
                res_qpos = self.env.get_humanoid_qpos()

                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(
                        self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                # if flags.debug:
                #     np.set_printoptions(precision=4, suppress=1)
                #     print(c_reward, c_info)

                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state,
                                 reward, exp, gt_qpos, curr_qpos, res_qpos)

                if pid == 0 and self.render:
                    self.env.render()

                if done:
                    freq_dict[self.data_loader.curr_key].append(
                        [info['percent'], self.data_loader.fr_start])
                    # print(self.data_loader.curr_key, info['percent'])
                    break

                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict

    def push_memory(self, memory, state, action, mask, next_state, reward, exp,
                    gt_target_qpos, curr_qpos, res_qpos):
        v_meta = np.array([
            self.data_loader.curr_take_ind, self.data_loader.fr_start,
            self.data_loader.fr_num
        ])
        memory.push(state, action, mask, next_state, reward, exp, v_meta,
                    gt_target_qpos, curr_qpos, res_qpos)

    def sample(self, min_batch_size):
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules)
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(
                    math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads

                for i in range(self.num_threads - 1):
                    worker_args = (i + 1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker,
                                                     args=worker_args)
                    worker.start()
                memories[0], loggers[0], freq_dict = self.sample_worker(
                    0, None, thread_batch_size)
                self.freq_dict = {
                    k: v + freq_dict[k]
                    for k, v in self.freq_dict.items()
                }

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger, freq_dict = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger

                    self.freq_dict = {
                        k: v + freq_dict[k]
                        for k, v in self.freq_dict.items()
                    }

                self.freq_dict = {
                    k: v if len(v) < 5000 else v[-5000:]
                    for k, v in self.freq_dict.items()
                }
                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = self.traj_batch(memories)
                logger = self.logger_cls.merge(loggers)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(
            self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(
            self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        v_metas = torch.from_numpy(batch.v_metas).to(self.dtype).to(
            self.device)
        gt_target_qpos = torch.from_numpy(batch.gt_target_qpos).to(
            self.dtype).to(self.device)
        curr_qpos = torch.from_numpy(batch.curr_qpos).to(self.dtype).to(
            self.device)
        res_qpos = torch.from_numpy(batch.res_qpos).to(self.dtype).to(
            self.device)

        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(
                    self.trans_value(states[:, :self.policy_net.state_dim]))
        self.policy_net.set_mode('train')
        self.policy_net.initialize_rnn((masks, v_metas))
        """get advantage estimation from the trajectories"""
        print("==================================================>")

        if self.cfg.policy_specs.get("rl_update", False):
            print("RL:")
            advantages, returns = estimate_advantages(rewards, masks, values,
                                                      self.gamma, self.tau)
            self.update_policy(states, actions, returns, advantages, exps)

        if self.cfg.policy_specs.get(
                "init_update", False) or self.cfg.policy_specs.get(
                    "step_update", False) or self.cfg.policy_specs.get(
                        "full_update", False):
            print("Supervised:")

        if self.cfg.policy_specs.get("init_update", False):
            self.policy_net.update_init_supervised(
                self.cfg,
                self.data_loader,
                device=self.device,
                dtype=self.dtype,
                num_epoch=int(self.cfg.policy_specs.get("num_init_update", 5)))

        if self.cfg.policy_specs.get("step_update", False):
            self.policy_net.update_supervised(
                states,
                gt_target_qpos,
                curr_qpos,
                num_epoch=int(self.cfg.policy_specs.get("num_step_update",
                                                        10)))

        if self.cfg.policy_specs.get("full_update", False):
            self.policy_net.train_full_supervised(self.cfg,
                                                  self.data_loader,
                                                  device=self.device,
                                                  dtype=self.dtype,
                                                  num_epoch=1,
                                                  scheduled_sampling=0.3)

        self.policy_net.step_lr()

        return time.time() - t0

    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            # trans_value = self.trans_value(states[:, :self.policy_net.obs_lim])
            trans_value = self.trans_value(states)

            values_pred = self.value_net(trans_value)
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(
                    self.trans_policy(states), actions)
        pbar = tqdm(range(self.opt_num_epochs))
        for _ in pbar:
            ind = exps.nonzero(as_tuple=False).squeeze(1)
            self.update_value(states, returns)
            surr_loss, ratio = self.ppo_loss(states, actions, advantages,
                                             fixed_log_probs, ind)
            self.optimizer_policy.zero_grad()
            surr_loss.backward()
            self.clip_policy_grad()
            self.optimizer_policy.step()
            pbar.set_description_str(
                f"PPO Loss: {surr_loss.cpu().detach().numpy():.3f}| ratio: {ratio.mean().cpu().detach().numpy():.3f}"
            )

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(
            self.trans_policy(states)[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss, ratio

    def action_loss(self, actions, gt_actions):
        pass
