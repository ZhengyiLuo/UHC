'''
File: /agent_copycat.py
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

import math
import time
import os
import fasteners
import torch

os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import pickle
from collections import defaultdict
import glob
import os
import sys
import os.path as osp
from tqdm import tqdm
import wandb

from uhc.khrylib.utils import to_device, create_logger, ZFilter, get_eta_str
from uhc.khrylib.rl.core import LoggerRL
from uhc.khrylib.utils.memory import Memory
from uhc.khrylib.utils.torch import *
from uhc.khrylib.rl.core import estimate_advantages
from uhc.khrylib.rl.agents import AgentPPO
from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
from uhc.khrylib.rl.core.critic import Value
from uhc.losses.reward_function import reward_func
from uhc.models.policy_mcp import PolicyMCP
from uhc.khrylib.models.mlp import MLP
from uhc.envs.humanoid_im import HumanoidEnv
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.smpllib.smpl_parser import SMPL_BONE_ORDER_NAMES
from uhc.smpllib.smpl_eval import compute_metrics
from uhc.utils.flags import flags
import multiprocessing
from uhc.utils.tools import CustomUnpickler


class AgentCopycat(AgentPPO):

    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        self.cfg = cfg
        self.cc_cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training
        self.max_freq = 50

        self.setup_vars()
        self.setup_data_loader()
        self.setup_env()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_reward()
        self.seed(cfg.seed)
        self.print_config()
        if checkpoint_epoch > 0:
            self.load_checkpoint(checkpoint_epoch)
            self.epoch = checkpoint_epoch

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            running_state=self.running_state,
            custom_reward=self.expert_reward,
            mean_action=cfg.render and not cfg.show_noise,
            render=cfg.render,
            num_threads=cfg.num_threads,
            data_loader=self.data_loader,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.num_optim_epoch,
            gamma=cfg.gamma,
            tau=cfg.tau,
            clip_epsilon=cfg.clip_epsilon,
            policy_grad_clip=[(self.policy_net.parameters(), 40)],
            end_reward=cfg.end_reward,
            use_mini_batch=False,
            mini_batch_size=0,
        )

    def setup_vars(self):
        self.epoch = 0
        self.running_state = None
        self.fit_single_key = ""
        self.precision_mode = self.cc_cfg.get("precision_mode", False)

        pass

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger.info("==========================Agent Copycat===========================")
        self.logger.info(f"Feature Version: {cfg.obs_v}")
        self.logger.info(f"Meta Pd: {cfg.meta_pd}")
        self.logger.info(f"Meta Pd Joint: {cfg.meta_pd_joint}")
        self.logger.info(f"Actor_type: {cfg.actor_type}")
        self.logger.info(f"Precision mode: {self.precision_mode}")
        self.logger.info(f"State_dim: {self.state_dim}")
        self.logger.info("============================================================")

    def setup_data_loader(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.data_loader = data_loader = DatasetAMASSSingle(cfg.data_specs, data_mode="train")
        self.test_data_loaders = []
        self.test_data_loaders.append(data_loader)
        if len(cfg.data_specs.get("test_file_path", [])) > 0:
            self.test_data_loaders.append(DatasetAMASSSingle(cfg.data_specs, data_mode="test"))

    def setup_env(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        random_expert = self.data_loader.sample_seq()
        self.env = HumanoidEnv(
            cfg,
            init_expert=random_expert,
            data_specs=cfg.data_specs,
            mode="train",
            no_root=cfg.no_root,
        )

    def setup_policy(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        actuators = env.model.actuator_names
        self.state_dim = state_dim = env.observation_space.shape[0]
        self.action_dim = action_dim = env.action_space.shape[0]
        """define actor and critic"""
        if cfg.actor_type == "gauss":
            self.policy_net = PolicyGaussian(cfg, action_dim=action_dim, state_dim=state_dim)
        elif cfg.actor_type == "mcp":
            self.policy_net = PolicyMCP(cfg, action_dim=action_dim, state_dim=state_dim)
        self.running_state = ZFilter((state_dim,), clip=5)
        to_device(device, self.policy_net)

    def setup_value(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
        to_device(device, self.value_net)

    def setup_optimizer(self):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        if cfg.policy_optimizer == "Adam":
            self.optimizer_policy = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                weight_decay=cfg.policy_weightdecay,
            )
        else:
            self.optimizer_policy = torch.optim.SGD(
                self.policy_net.parameters(),
                lr=cfg.policy_lr,
                momentum=cfg.policy_momentum,
                weight_decay=cfg.policy_weightdecay,
            )
        if cfg.value_optimizer == "Adam":
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                weight_decay=cfg.value_weightdecay,
            )
        else:
            self.optimizer_value = torch.optim.SGD(
                self.value_net.parameters(),
                lr=cfg.value_lr,
                momentum=cfg.value_momentum,
                weight_decay=cfg.value_weightdecay,
            )

    def setup_reward(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.expert_reward = expert_reward = reward_func[cfg.reward_id]

    def save_checkpoint(self, epoch):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, epoch + 1)
            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))
            joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def save_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path = f"{cfg.model_dir}_singles/{key}.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path, "wb"))

    def save_curr(self):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self.policy_net, self.value_net):
            cp_path_best = f"{cfg.model_dir}/iter_best.p"

            model_cp = {
                "policy_dict": self.policy_net.state_dict(),
                "value_dict": self.value_net.state_dict(),
                "running_state": self.running_state,
            }
            pickle.dump(model_cp, open(cp_path_best, "wb"))

    def load_curr(self):
        cfg = self.cfg
        cp_path_best = f"{cfg.model_dir}/iter_best.p"
        self.logger.info("loading model from checkpoint: %s" % cp_path_best)
        model_cp = CustomUnpickler(open(cp_path_best, "rb")).load()
        self.policy_net.load_state_dict(model_cp["policy_dict"])
        self.value_net.load_state_dict(model_cp["value_dict"])
        self.running_state = model_cp["running_state"]

    def load_singles(self, epoch, key):
        cfg = self.cfg
        # self.tb_logger.flush()
        if epoch > 0:
            cp_path = f"{cfg.model_dir}/iter_{(epoch+ 1):04d}_{key}.p"
            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

    def load_checkpoint(self, iter):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg = self.cfg
        if iter > 0:
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, iter)
            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = CustomUnpickler(open(cp_path, "rb")).load()
            self.policy_net.load_state_dict(model_cp["policy_dict"])
            self.value_net.load_state_dict(model_cp["value_dict"])
            self.running_state = model_cp["running_state"]

        to_device(device, self.policy_net, self.value_net)

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        freq_path = osp.join(cfg.result_dir, "freq_dict.pt")
        try:
            self.freq_dict = ({k: [] for k in self.data_loader.data_keys} if not osp.exists(freq_path) else joblib.load(freq_path))
            for k in self.data_loader.data_keys:
                if not k in self.freq_dict:
                    raise Exception("freq_dict is not initialized")

            for k in self.freq_dict:
                if not k in self.data_loader.data_keys:
                    raise Exception("freq_dict is not initialized")
        except:
            print("error parsing freq_dict, using empty one")
            self.freq_dict = {k: [] for k in self.data_loader.data_keys}
        self.logger = create_logger(os.path.join(cfg.log_dir, "log.txt"))

    def per_epoch_update(self, epoch):
        cfg, device, dtype, env = self.cfg, self.device, self.dtype, self.env
        cfg.update_adaptive_params(epoch)
        self.set_noise_rate(cfg.adp_noise_rate)
        set_optimizer_lr(self.optimizer_policy, cfg.adp_policy_lr)
        if cfg.rfc_decay:
            if self.epoch < cfg.get("rfc_decay_max", 10000):
                self.env.rfc_rate = lambda_rule(self.epoch, cfg.get("rfc_decay_max", 10000), cfg.num_epoch_fix)
            else:
                self.env.rfc_rate = 0.0

        # epoch
        # adative_iter = cfg.data_specs.get("adaptive_iter", -1)
        # if epoch != 0 and adative_iter != -1 and epoch % adative_iter == 0 :
        # agent.data_loader.hard_negative_mining(agent.value_net, agent.env, device, dtype, running_state = running_state, sampling_temp = cfg.sampling_temp)

        if cfg.fix_std:
            self.policy_net.action_log_std.fill_(cfg.adp_log_std)
        return

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        log = info["log"]

        c_info = log.avg_c_info
        log_str = f"Ep: {self.epoch}\t {cfg.id} \tT_s {info['T_sample']:.2f}\t \
                    T_u { info['T_update']:.2f}\tETA {get_eta_str(self.epoch, cfg.num_epoch, info['T_total'])} \
                \texpert_R_avg {log.avg_c_reward:.4f} {np.array2string(c_info, formatter={'all': lambda x: '%.4f' % x}, separator=',')}\
                 \texpert_R_range ({log.min_c_reward:.4f}, {log.max_c_reward:.4f})\teps_len {log.avg_episode_len:.2f}"

        self.logger.info(log_str)

        if not cfg.no_log:
            wandb.log(
                data={
                    "rewards": log.avg_c_info,
                    "eps_len": log.avg_episode_len,
                    "avg_rwd": log.avg_c_reward,
                    "rfc_rate": self.env.rfc_rate,
                },
                step=self.epoch,
            )

            if "log_eval" in info:
                [wandb.log(data=test, step=self.epoch) for test in info["log_eval"]]

    def optimize_policy(self, epoch, save_model=True):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.epoch = epoch
        t0 = time.time()
        self.per_epoch_update(epoch)
        batch, log = self.sample(cfg.min_batch_size)

        if cfg.end_reward:
            self.env.end_reward = log.avg_c_reward * cfg.gamma / (1 - cfg.gamma)
        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()
        info = {
            "log": log,
            "T_sample": t1 - t0,
            "T_update": t2 - t1,
            "T_total": t2 - t0,
        }

        if save_model and (self.epoch + 1) % cfg.save_n_epochs == 0:
            self.save_checkpoint(epoch)
            log_eval = self.eval_policy(epoch)
            info["log_eval"] = log_eval

        self.log_train(info)
        # joblib.dump(self.freq_dict, osp.join(cfg.result_dir, "freq_dict.pt"))

    def eval_policy(self, epoch=0, dump=False):
        cfg = self.cfg
        data_loaders = self.test_data_loaders

        res_dicts = []
        for data_loader in data_loaders:
            num_jobs = 20
            # num_jobs = self.num_threads
            jobs = data_loader.data_keys
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            jobs = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
            data_res_coverage = {}
            with to_cpu(*self.sample_modules):
                with torch.no_grad():
                    queue = multiprocessing.Queue()
                    for i in range(len(jobs) - 1):
                        worker_args = (jobs[i + 1], data_loader, queue, i)
                        worker = multiprocessing.Process(target=self.eval_seqs, args=worker_args)
                        worker.start()
                    res = self.eval_seqs(jobs[0], data_loader, None)
                    data_res_coverage.update(res)
                    for i in tqdm(range(len(jobs) - 1)):
                        res = queue.get()
                        data_res_coverage.update(res)

                for k, res in data_res_coverage.items():
                    [self.freq_dict[k].append([res["succ"][0], 0]) for _ in range(1 if res["succ"][0] else 3) if k in self.freq_dict]  # first item is scuccess or not, second indicates the frame number

                metric_names = [
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
                [[data_res_metrics[k].append(v if np.ndim(v) == 0 else np.mean(v)) for k, v in res.items() if k in metric_names] for k, res in data_res_coverage.items()]
                data_res_metrics = {k: np.mean(v) for k, v in data_res_metrics.items()}
                coverage = int(data_res_metrics["succ"] * data_loader.get_len())
                print_str = " \t".join([f"{k}: {v:.3f}" for k, v in data_res_metrics.items()])

                self.logger.info(f"Coverage {data_loader.name} of {coverage} out of {data_loader.get_len()} | {print_str}")
                data_res_metrics.update({
                    "mean_coverage": coverage / data_loader.get_len(),
                    "num_coverage": coverage,
                    "all_coverage": data_loader.get_len(),
                })
                del data_res_metrics["succ"]
                res_dicts.append({f"coverage_{data_loader.name}": data_res_metrics})

                if dump:
                    res_dir = osp.join(cfg.output_dir, f"{epoch}_{data_loader.name}_coverage_full.pkl")
                    print(res_dir)
                    joblib.dump(data_res_coverage, res_dir)

        return res_dicts

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

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
        self.env.set_mode("test")
        fail_safe = False
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                res = defaultdict(list)

                self.env.load_expert(loader.get_sample_from_key(take_key=take_key, full_sample=True, fr_start=-1))

                state = self.env.reset()
                if self.running_state is not None:
                    state = self.running_state(state)

                for t in range(10000):
                    res["gt"].append(self.env.get_expert_attr("qpos", self.env.get_expert_index(t)).copy())

                    res["pred"].append(self.env.data.qpos.copy())

                    res["gt_jpos"].append(self.env.get_expert_attr("wbpos", self.env.get_expert_index(t)).copy())
                    res["pred_jpos"].append(self.env.get_wbody_pos().copy())
                    state_var = tensor(state).unsqueeze(0)
                    trans_out = self.trans_policy(state_var)

                    action = (self.policy_net.select_action(trans_out, mean_action=True)[0].cpu().numpy())
                    next_state, env_reward, done, info = self.env.step(action)
                    if (self.cc_cfg.residual_force and self.cc_cfg.residual_force_mode == "explicit"):
                        res["vf_world"].append(self.env.get_world_vf())

                    if self.render:
                        for i in range(10):
                            self.env.render()

                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    # np.set_printoptions(precision=4, suppress=1)
                    # print(c_info)
                    res["reward"].append(c_reward)
                    # self.env.render()
                    if self.running_state is not None:
                        next_state = self.running_state(next_state, update=False)

                    if done:
                        if self.cfg.fail_safe and info["percent"] != 1:
                            self.env.fail_safe()
                            fail_safe = True
                        else:
                            res = {k: np.vstack(v) for k, v in res.items()}
                            res["percent"] = info["percent"]
                            res["fail_safe"] = fail_safe
                            if self.cfg.get("full_eval", False):
                                self.env.convert_2_smpl_params(res)
                            res.update(compute_metrics(res, self.env.converter))
                            return res
                    state = next_state

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)
        if hasattr(self.env, "np_random"):
            self.env.np_random.random(pid)
        memory = Memory()
        logger = self.logger_cls()
        freq_dict = defaultdict(list)
        while logger.num_steps < min_batch_size:
            if self.fit_single_key != "":
                self.env.load_expert(self.data_loader.get_sample_from_key(
                    self.fit_single_key,
                    full_sample=False,
                    freq_dict=self.freq_dict,
                    precision_mode=self.precision_mode,
                ))
            else:
                self.env.load_expert(self.data_loader.sample_seq(
                    freq_dict=self.freq_dict,
                    full_sample=False,
                    sampling_temp=self.cfg.sampling_temp,
                    sampling_freq=self.cfg.sampling_freq,
                    precision_mode=self.precision_mode,
                ))
            # self.env.load_expert(self.data_loader.sample_seq(freq_dict = self.freq_dict, full_sample = True))

            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                action = self.policy_net.select_action(trans_out, mean_action)[0].numpy()
                action = (int(action) if self.policy_net.type == "discrete" else action.astype(np.float64))
                next_state, env_reward, done, info = self.env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward

                # add end reward
                if self.end_reward and info.get("end", False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if pid == 0 and self.render:
                    for i in range(10):
                        # for i in range(500):
                        self.env.render()

                if done:

                    freq_dict[self.data_loader.curr_key].append([info["percent"], self.data_loader.fr_start])
                    break
                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger, freq_dict])
        else:
            return memory, logger, freq_dict

    def sample(self, min_batch_size):
        self.env.set_mode("train")
        t_start = time.time()
        self.pre_sample()
        to_test(*self.sample_modules)
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads - 1):
                    worker_args = (i + 1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0], freq_dict = self.sample_worker(0, None, thread_batch_size)

                self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger, freq_dict = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger

                    self.freq_dict = {k: v + freq_dict[k] for k, v in self.freq_dict.items()}

                # print(np.sum([len(v) for k, v in self.freq_dict.items()]), np.mean(np.concatenate([self.freq_dict[k] for k in self.freq_dict.keys()])))
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers)

        self.freq_dict = {k: v if len(v) < self.max_freq else v[-self.max_freq:] for k, v in self.freq_dict.items()}
        logger.sample_time = time.time() - t_start
        return traj_batch, logger