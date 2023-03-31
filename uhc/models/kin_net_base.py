import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())
from sys import flags
from torch import is_tensor, nn
from collections import defaultdict
import joblib
import pickle
import time
import wandb
from tqdm import tqdm

from uhc.khrylib.utils import to_device, create_logger
from uhc.khrylib.models.mlp import MLP
from uhc.khrylib.utils.torch import lambda_rule, get_scheduler
from uhc.utils.flags import flags
from uhc.utils.torch_ext import *
from uhc.khrylib.models.rnn import RNN
from uhc.utils.torch_utils import (
    get_heading_batch,
    get_heading_q,
    quaternion_multiply,
    quaternion_inverse,
    get_heading_q_batch,
    transform_vec_batch,
    quat_from_expmap_batch,
    quat_mul_vec_batch,
    get_qvel_fd_batch,
    transform_vec,
    rotation_from_quaternion,
    de_heading_batch,
    quat_mul_vec,
    quat_from_expmap,
    quaternion_multiply_batch,
    quaternion_inverse_batch,
)
from uhc.losses.loss_function import (
    compute_mpjpe_global,
    pose_rot_loss,
    root_pos_loss,
    root_orientation_loss,
    end_effector_pos_loss,
    linear_velocity_loss,
    angular_velocity_loss,
    action_loss,
    position_loss,
    orientation_loss,
    compute_error_accel,
    compute_error_vel,
)
import mujoco_py
from uhc.smpllib.smpl_robot import Robot
from uhc.smpllib.torch_smpl_humanoid import Humanoid
from uhc.smpllib.torch_smpl_humanoid_batch import Humanoid_Batch

class KinNetBase(nn.Module):
    def __init__(self, cfg, data_sample, device, dtype, mode="train"):
        super(KinNetBase, self).__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model_specs = cfg.model_specs
        self.mode = mode
        self.epoch = 0
        self.register_buffer("base_rot", torch.tensor([[0.7071, 0.7071, 0.0, 0.0]]))
        self.model_v = self.model_specs.get("model_v", 1)
        self.pose_delta = self.model_specs.get("pose_delta", False)
        self.remove_base = self.model_specs.get("remove_base", False)
        self.num_jobs = self.model_specs.get("num_jobs", 10)

        self.gt_rate = 0
        self.fk_model = Humanoid(model_file=cfg.mujoco_model_file)

        self.htype = htype = self.model_specs.get("mlp_htype", "relu")
        self.mlp_hsize = mlp_hsize = self.model_specs.get("mlp_hsize", [1024, 512])
        self.rnn_hdim = rnn_hdim = self.model_specs.get("rnn_hdim", 512)
        self.rnn_type = rnn_type = self.model_specs.get("rnn_type", "gru")
        self.cnn_fdim = cnn_fdim = self.model_specs.get("cnn_fdim", 128)

        self.sim = dict()
        self.get_dim(data_sample)

        if self.model_v == 1 or self.model_v == 0:
            self.action_rnn = RNN(self.state_dim, rnn_hdim, rnn_type)
            self.action_rnn.set_mode("step")
            self.action_mlp = MLP(rnn_hdim + self.state_dim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.action_dim)
        elif self.model_v == 2:
            self.action_mlp = MLP(self.state_dim, mlp_hsize, htype)
            self.action_fc = nn.Linear(mlp_hsize[-1], self.action_dim)

        self.context_rnn = RNN(self.context_dim, rnn_hdim, rnn_type)
        self.context_rnn.set_mode("batch")
        self.context_mlp = MLP(rnn_hdim, mlp_hsize, htype)
        self.context_fc = nn.Linear(mlp_hsize[-1], self.init_dim)
        self.qpos_lm = 74
        self.qvel_lm = 75
        self.pose_start = 7

        # Netural data
        self.netural_data = joblib.load(
            "sample_data/standing_neutral.pkl"
        )
        fk_res = self.fk_model.qpos_fk(
            torch.from_numpy(
                self.netural_data["qpos"][
                    None,
                ]
            )
            .to(device)
            .type(dtype)
        )
        fk_res["qvel"] = (
            torch.from_numpy(self.netural_data["qvel"]).to(device).type(dtype)
        )
        self.netural_target = fk_res
        self.setup_logging()
        self.setup_optimizer()
        self.load_humanoid()
        print(
            f"Context dim: {self.context_dim}, State dim: {self.state_dim}, Init dim: {self.init_dim}, Action dim: {self.action_dim}",
            f"Adding noise? {self.model_specs.get('add_noise', True)}",
        )

    def set_mode(self, mode = "train"):
        self.mode = mode
        
    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def load_humanoid(self):
        self.smpl_robot = Robot(
            self.cfg.cc_cfg.robot_cfg,
            data_dir=osp.join(self.cfg.cc_cfg.base_dir, "data/smpl"),
            masterfoot=self.cfg.cc_cfg.masterfoot,
        )
        self.model = mujoco_py.load_model_from_xml(self.smpl_robot.export_xml_string().decode("utf-8"))
        self.humanoid = Humanoid(model = self.model)
        self.human_b = Humanoid_Batch(data_dir=osp.join(self.cfg.cc_cfg.base_dir, "data/smpl"),)

    def reload_humanoid(self, beta, gender = "neutral"):
        self.smpl_robot.load_from_skeleton(
                beta[0:1, :].cpu().float(), gender=gender, objs_info=None
            )
        self.model = mujoco_py.load_model_from_xml(self.smpl_robot.export_xml_string().decode("utf-8"))
        self.humanoid = Humanoid(model = self.model,)

    def set_schedule_sampling(self, gt_rate):
        self.gt_rate = gt_rate

    def get_dim(self, data):
        raise NotImplementedError

    def set_sim(self, qpos, qvel=None):
        self.sim["qpos"] = (
            qpos
            if torch.is_tensor(qpos)
            else torch.from_numpy(qpos).to(self.device).type(self.dtype)
        )
        if not qvel is None:
            self.sim["qvel"] = (
                qvel
                if torch.is_tensor(qvel)
                else torch.from_numpy(qvel).to(self.device).type(self.dtype)
            )
        else:
            self.sim["qvel"] = (
                torch.zeros(self.qvel_lm).to(self.device).type(self.dtype)
            )

    def get_context_dim(self, data):
        context_d = 0
        # if self.cfg.use_of: context_d += self.cnn_fdim
        # if self.cfg.use_img: context_d += data['img_feats'].shape[-1]
        # if self.cfg.use_2d: context_d += data['bbox_vels'].shape[-1]
        return context_d

    def get_context_feat(self, data):
        data_acc = []
        return None

    def init_states(self, data):
       raise NotImplementedError

    def forward(self, data):
       raise NotImplementedError


    def get_obs(self, data, t):
       raise NotImplementedError

    def step(self, action, dt=1 / 30):
        raise NotImplementedError

    def get_action(self, state):
        x = self.action_mlp(state)
        action = self.action_fc(x)
        return action

    def fix_qvel(self, feature_pred):
        pred_qvel = feature_pred["qvel"]
        feature_pred["qvel"] = torch.cat(
            (pred_qvel[:, 1:, :], pred_qvel[:, -2:-1, :]), dim=1
        )

    def setup_optimizer(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        optimizer = cfg.model_specs.get("optimizer")
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=cfg.lr)
        elif optimizer == "Adamx":
            self.optimizer = torch.optim.Adamax(self.parameters(), lr=cfg.lr)
            
        self.scheduler = get_scheduler(
            self.optimizer,
            policy="lambda",
            nepoch_fix=self.cfg.num_epoch_fix,
            nepoch=self.cfg.num_epoch,
        )

    def save_checkpoint(self, epoch):
        cfg = self.cfg
        # self.tb_logger.flush()
        with to_cpu(self):
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, epoch + 1)
            model_cp = {"model_dict": self.state_dict()}
            pickle.dump(model_cp, open(cp_path, "wb"))

    def load_checkpoint(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        cfg = self.cfg
        if epoch > 0:
            self.epoch = epoch
            cp_path = "%s/iter_%04d.p" % (cfg.model_dir, epoch)
            self.logger.info("loading model from checkpoint: %s" % cp_path)
            model_cp = pickle.load(open(cp_path, "rb"))
            self.load_state_dict(model_cp["model_dict"])
        to_device(self)

    def setup_logging(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger = create_logger(os.path.join(cfg.log_dir, "log.txt"))

    def per_epoch_update(self, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        if self.model_specs.get("gt_rate_decay", True):
            self.gt_rate = self.model_specs.get("gt_rate", 0.3) * lambda_rule(
                self.epoch, self.cfg.num_epoch, self.cfg.num_epoch_fix
            )
        self.scheduler.step()
        return

    def log_train(self, info):
        """logging"""
        cfg, device, dtype = self.cfg, self.device, self.dtype
        info["gt_rate"] = self.gt_rate
        logger_str = f"Ep {self.epoch} Loss: {info['total_loss']:.3f} \t losses: {[f'{k} : {v:.3f}' for k, v in info['loss_dict'].items()]}"

        if not self.cfg.no_log:
            wandb.log(data=info, step=self.epoch)
            self.logger.info(logger_str)
        else:
            print(logger_str)

        if "log_eval" in info:
            [wandb.log(data=test, step=self.epoch) for test in info["log_eval"]]

    def training_epoch(self, train_loader, epoch = 0, max_epoch = 100):
        self.epoch = epoch
        info = {
            "total_loss": [],
            "loss_dict": defaultdict(list),
            "loss_unweighted_dict": defaultdict(list),
        }
        pbar = tqdm(train_loader)
        for data_batch in pbar:
            data_batch = {
                k: v.clone().to(self.device).type(self.dtype)  if isinstance(v, torch.Tensor) else v
                for k, v in data_batch.items()
            }
            info_step = self.training_step(data_batch, epoch, max_epoch)
            pbar.set_description_str(f"{info_step['total_loss'].item():.3f}")
            info["total_loss"].append(info_step["total_loss"])
            [info["loss_dict"][k].append(v) for k, v in info_step["loss_dict"].items()]
            [
                info["loss_unweighted_dict"][k].append(v)
                for k, v in info_step["loss_unweighted_dict"].items()
            ]
        info["total_loss"] = torch.mean(torch.tensor(info["total_loss"]))
        info["loss_dict"] = {
            k: torch.mean(torch.tensor(v)) for k, v in info["loss_dict"].items()
        }
        info["loss_unweighted_dict"] = {
            k: torch.mean(torch.tensor(v))
            for k, v in info["loss_unweighted_dict"].items()
        }
        self.log_train(info)
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        return info

    def train_first_frame_epoch(self, train_loader, epoch):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.epoch = epoch
        pbar = train_loader
        info = {
            "total_loss": [],
            "loss_dict": defaultdict(list),
            "loss_unweighted_dict": defaultdict(list),
        }
        for data_dict in pbar:
            data_dict = {
                k: v.clone().to(device).type(dtype) for k, v in data_dict.items()
            }
            res_dict = self.init_states(data_dict)
            total_loss, loss_dict, loss_unweighted_dict = self.compute_loss_init(
                res_dict, data_dict
            )
            self.optimizer.zero_grad()
            total_loss.backward()  # Testing GT
            self.optimizer.step()  # Testing GT
            info["total_loss"].append(total_loss)
            [info["loss_dict"][k].append(v) for k, v in loss_dict.items()]
            [
                info["loss_unweighted_dict"][k].append(v)
                for k, v in loss_unweighted_dict.items()
            ]

        info["total_loss"] = torch.mean(torch.tensor(info["total_loss"]))
        info["loss_dict"] = {
            k: torch.mean(torch.tensor(v)) for k, v in info["loss_dict"].items()
        }
        info["loss_unweighted_dict"] = {
            k: torch.mean(torch.tensor(v))
            for k, v in info["loss_unweighted_dict"].items()
        }
        return info

    def training_step(self, train_batch, epoch = 0, max_epoch = 100):
        self.mode = "train"
        feature_pred, train_batch_udpate = self.forward(train_batch)
        total_loss, loss_dict, loss_unweighted_dict = self.compute_loss(
            feature_pred, train_batch_udpate, epoch, max_epoch
        )
        info = {
            "total_loss": total_loss,
            "loss_dict": loss_dict,
            "loss_unweighted_dict": loss_unweighted_dict,
        }

        if flags.debug:
            with torch.no_grad():
                self.mode = "test"
                feature_pred, train_batch_udpate = self.forward(train_batch)
                metrics = self.compute_metrics(feature_pred, train_batch)
                # print(metrics)
        else:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return info

    def eval_model(self, val_dataset, multi_process=True):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.mode = "test"
        self.to("cpu")
        jobs = list(val_dataset.iter_data().items())
        eval_res = {}
        import multiprocessing

        with torch.no_grad():
            if multi_process:
                num_jobs = self.num_jobs
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                jobs = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
                queue = multiprocessing.Queue()
                for i in range(len(jobs) - 1):
                    worker_args = (jobs[i + 1], queue)
                    worker = multiprocessing.Process(
                        target=self.eval_seqs, args=worker_args
                    )
                    worker.start()
                res = self.eval_seqs(jobs[0], None)
                eval_res.update(res)
                for i in range(len(jobs) - 1):
                    res = queue.get()
                    eval_res.update(res)
            else:
                self.eval_seqs(jobs)

            # for k, val_seq in pbar:
            #     val_seq =  {k:v.clone().to(device).type(dtype) for k, v in val_seq.items()}
            #     feature_pred = self.forward(val_seq)
            #     metrics = self.compute_metrics(feature_pred, val_seq)
            #     total_metrices[k] = metrics
            #     eval_res[k]['pred'], eval_res[k]['gt']  = {k: v.cpu().numpy().squeeze() for k, v in feature_pred.items()}, {k: v.cpu().numpy().squeeze() for k, v in val_seq.items()}
            #     pbar.set_description_str(f"Loss: {metrics['mpjpe_local']}")
            #     print(metrics)
        avg_metrics = defaultdict(list)
        exclude = ["gt_qpos", "pred_qpos"]
        [
            [avg_metrics[m].append(value) for m, value in v.items() if not m in exclude]
            for k, v in eval_res.items()
        ]
        avg_metrics = {k: torch.mean(torch.tensor(v)) for k, v in avg_metrics.items()}

        if not self.cfg.no_log:
            self.logger.info(
                f"Eval metrics {[f'{k} : {v:.3f}' for k, v in avg_metrics.items()]}"
            )
            wandb.log({"eval_loss": avg_metrics})
        else:
            print(f"Eval metrics {[f'{k} : {v:.3f}' for k, v in avg_metrics.items()]}")
        self.to(device)
        return eval_res, avg_metrics

    def eval_seqs(self, cur_jobs, queue):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        res = defaultdict(dict)
        for k, val_seq in cur_jobs:
            feature_pred, val_seq_update= self.forward(val_seq)
            metrics = self.compute_metrics(feature_pred, val_seq_update)
            feature_pred, val_seq = {
                k: v.cpu().numpy().squeeze() if torch.is_tensor(v) else v for k, v in feature_pred.items()
            }, {k: v.cpu().numpy().squeeze() if torch.is_tensor(v) else v for k, v in val_seq.items()}

            res[k].update(metrics)
            # res[k]["pred_qpos"], res[k]["gt_qpos"] = (
            #     feature_pred["qpos"],
            #     val_seq["qpos"],
            # )

        if queue == None:
            return res
        else:
            queue.put(res)

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

    # def compute_loss_lite(self, pred_qpos, gt_qpos):
    #     cfg, device, dtype = self.cfg, self.device, self.dtype
    #     weights = cfg.policy_specs.get("super_weights", {})

    #     fk_res_pred = self.fk_model.qpos_fk(pred_qpos)
    #     fk_res_gt = self.fk_model.qpos_fk(gt_qpos)

    #     pred_wbpos = fk_res_pred['wbpos'].reshape(pred_qpos.shape[0], -1)
    #     gt_wbpos = fk_res_gt['wbpos'].reshape(pred_qpos.shape[0], -1)

    #     r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
    #     r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
    #     p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()    # pose loss
    #     ee_loss = end_effector_pos_loss(gt_wbpos, pred_wbpos).mean() # End effector loss

    #     total_loss = 0
    #     loss_dict = {}
    #     loss_unweighted_dict = {}
    #     for k, v in weights.items():
    #         if k in locals():
    #             loss = eval(k) * v
    #             total_loss += loss; loss_dict[k] = loss.detach().item(); loss_unweighted_dict[k + "-uw"] = eval(k).detach().item()
    #     return total_loss, loss_dict, loss_unweighted_dict

    def compute_loss_lite(self, pred_qpos, gt_qpos):
        w_rp, w_rr, w_p, w_ee = 50, 50, 1, 10
        fk_res_pred = self.fk_model.qpos_fk(pred_qpos)
        fk_res_gt = self.fk_model.qpos_fk(gt_qpos)

        pred_wbpos = fk_res_pred["wbpos"].reshape(pred_qpos.shape[0], -1)
        gt_wbpos = fk_res_gt["wbpos"].reshape(pred_qpos.shape[0], -1)

        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()  # pose loss
        ee_loss = end_effector_pos_loss(
            gt_wbpos, pred_wbpos
        ).mean()  # End effector loss

        loss = w_rp * r_pos_loss + w_rr * r_rot_loss + w_p * p_rot_loss + w_ee * ee_loss

        return loss, [i.item() for i in [r_pos_loss, r_rot_loss, p_rot_loss, ee_loss]]

    def compute_loss(self, feature_pred, data, epoch = 0, max_epoch = 100):
        total_seq_loss, loss_seq_dict, loss_seq_unweighted_dict = self.compute_loss_seq(
            feature_pred, data, epoch, max_epoch
        )
        (
            total_init_loss,
            loss_init_dict,
            loss_init_unweighted_dict,
        ) = self.compute_loss_init(feature_pred, data)
        total_loss = total_seq_loss + total_init_loss
        loss_dict = {**loss_seq_dict, **loss_init_dict}
        loss_unweighted_dict = {**loss_seq_unweighted_dict, **loss_init_unweighted_dict}
        return total_loss, loss_dict, loss_unweighted_dict

    def compute_loss_seq(self, feature_pred, data, epoch = 0, max_epoch = 100):
        weights = self.model_specs.get("weights", {})

        b_size, seq_len, nq = feature_pred["qpos"].shape
        pred_qpos = feature_pred["qpos"].reshape(b_size * seq_len, -1)
        gt_qpos = data["qpos"].reshape(b_size * seq_len, -1)

        pred_qvel = feature_pred["qvel"][:, :-1, :].reshape(b_size * (seq_len - 1), -1)
        gt_qvel = data["qvel"][:, 1:, :].reshape(
            b_size * (seq_len - 1), -1
        )  # ZL: GT qvel is one step ahead

        # action = feature_pred['action'].reshape(b_size * seq_len, -1)

        pred_wbpos = feature_pred["pred_wbpos"].reshape(b_size * seq_len, -1)
        gt_w_pos = data["wbpos"].reshape(b_size * seq_len, -1)
        # pred_wbquat = feature_pred['pred_wbquat'].view(b_size, seq_len, -1)
        # wbquat = data['wbquat'].view(b_size, seq_len, -1)
        target_action = data["target"].reshape(b_size * seq_len, -1)

        r_pos_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
        p_rot_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()  # pose loss
        vl_loss = linear_velocity_loss(
            gt_qvel, pred_qvel
        ).mean()  # Root angular velocity loss
        va_loss = angular_velocity_loss(
            gt_qvel, pred_qvel
        ).mean()  # Root angular velocity loss
        ee_loss = end_effector_pos_loss(
            gt_w_pos, pred_wbpos
        ).mean()  # End effector loss

        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for k, v in weights.items():
            if k in locals():
                loss = eval(k) * v
                total_loss += loss
                loss_dict[k] = loss.detach().item()
                loss_unweighted_dict[k + "-uw"] = eval(k).detach().item()

        return total_loss, loss_dict, loss_unweighted_dict

    def compute_loss_init(self, feature_pred, data):
        weights = self.model_specs.get("weights", {})
        pred_qpos = feature_pred["init_qpos"]
        gt_qpos = data["qpos"][:, 0, :]
        fk_res_init_pred = self.fk_model.qpos_fk(pred_qpos)
        fk_res_init_gt = self.fk_model.qpos_fk(gt_qpos)

        pred_init_wbpos = fk_res_init_pred["wbpos"].reshape(pred_qpos.shape[0], -1)
        gt_init_wbpos = fk_res_init_gt["wbpos"].reshape(pred_qpos.shape[0], -1)

        r_pos_init_loss = root_pos_loss(gt_qpos, pred_qpos).mean()
        r_rot_init_loss = root_orientation_loss(gt_qpos, pred_qpos).mean()
        p_rot_init_loss = pose_rot_loss(gt_qpos, pred_qpos).mean()  # pose loss
        ee_init_loss = end_effector_pos_loss(
            gt_init_wbpos, pred_init_wbpos
        ).mean()  # End effector loss
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for k, v in weights.items():
            if k in locals():
                loss = eval(k) * v
                total_loss += loss
                loss_dict[k] = loss.detach().item()
                loss_unweighted_dict[k + "-uw"] = eval(k).detach().item()
        return total_loss, loss_dict, loss_unweighted_dict

    def compute_metrics(self, feature_pred, data):
        pred_jpos = (
            feature_pred["pred_wbpos"].squeeze().reshape((-1, 24, 3)).clone().clone()
        )
        gt_jpos = data["wbpos"].squeeze().reshape((-1, 24, 3)).clone()
        mpjpe_global = (
            np.linalg.norm((pred_jpos - gt_jpos).detach().cpu().numpy(), axis=2).mean()
            * 1000
        )

        pred_jpos_local = pred_jpos - pred_jpos[:, 0:1, :]
        gt_jpos_local = gt_jpos - gt_jpos[:, 0:1, :]
        mpjpe_local = (
            np.linalg.norm(
                (pred_jpos_local - gt_jpos_local).detach().cpu().numpy(), axis=2
            ).mean()
            * 1000
        )
        acc_err = (
            compute_error_accel(
                pred_jpos.detach().cpu().numpy(), gt_jpos.detach().cpu().numpy()
            ).mean()
            * 1000
        )
        vel_err = (
            compute_error_vel(
                pred_jpos.detach().cpu().numpy(), gt_jpos.detach().cpu().numpy()
            ).mean()
            * 1000
        )
        return {
            "mpjpe_local": mpjpe_local,
            "mpjpe_global": mpjpe_global,
            "acc_err": acc_err,
            "vel_err": vel_err,
        }
