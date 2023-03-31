import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
os.environ['OMP_NUM_THREADS'] = "1"
import argparse
import os
import sys
import pickle
import time
import subprocess
import shutil
from collections import defaultdict
from tqdm import tqdm 
import joblib
import numpy as np

from uhc.khrylib.rl.utils.visualizer import Visualizer
from uhc.utils.config_utils.copycat_config import Config
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.data_loaders.dataset_smpl_obj import DatasetSMPLObj



class MyVisulizer(Visualizer):

    def __init__(self, vis_file):
        super().__init__(vis_file)
        ngeom = 24
        self.env_vis.model.geom_rgba[ngeom + 1: ngeom * 2 - 1] = np.array([0.7, 0.0, 0.0, 1])

        self.env_vis.viewer.cam.lookat[2] = 1.0
        self.env_vis.viewer.cam.azimuth = args.azimuth
        self.env_vis.viewer.cam.elevation = -8.0
        self.env_vis.viewer.cam.distance = 5.0
        self.T = 12

    def data_generator(self):
        while True:
            if args.input:
                print("please input your requested sequence:")
                seq_name = input()
                data_loader.set_singles(seq_name)

            poses = defaultdict(list)
            env.load_expert(data_loader.iter_seq())
            state = env.reset()
            if running_state is not None:
                state = running_state(state, update=False)

            for t in range(1000000):
                epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
                env.render()
                poses['gt'].append(epos)
                poses['pred'].append(env.get_humanoid_qpos())
                poses['obj_pose'].append(env.get_obj_qpos())
                state_var = tensor(state, dtype=dtype).unsqueeze(0).to(device)
                action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                next_state, reward, done, res = env.step(action)

                if running_state is not None:
                    next_state = running_state(next_state, update=False)
                if done:
                    print(res['percent'], data_loader.curr_key)
                    break
                state = next_state

            poses ={k: np.vstack(v) for k, v in poses.items()}

            self.num_fr = poses['pred'].shape[0]
            yield poses

    def update_pose(self):
        self.env_vis.data.qpos[:76] = self.data['pred'][self.fr]
        self.env_vis.data.qpos[76:152] = self.data['gt'][self.fr]
        self.env_vis.data.qpos[152:] = self.data['obj_pose'][self.fr]

        # self.env_vis.data.qpos[env.model.nq] += 1.0
        # if args.record_expert:
            # self.env_vis.data.qpos[:env.model.nq] = self.data['gt'][self.fr]
        # if args.hide_expert:
            # self.env_vis.data.qpos[env.model.nq + 2] = 100.0
        if args.focus:
            self.env_vis.viewer.cam.lookat[:2] = self.env_vis.data.qpos[:2]
        self.env_vis.sim_forward()

    def display_coverage(self):
        res_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_coverage_full.pkl")
        # res_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_relive_coverage_full.pkl")
        print(res_dir)
        data_res = joblib.load(res_dir)
        print(len(data_res))

        

        def vis_gen():
            keys = sorted(list(data_res.keys()))
            keys = sorted([k for k in list(data_res.keys()) if data_res[k]['percent'] != 1 or ("fail_safe" in data_res[k] and data_res[k]['fail_safe'])])
            for k in keys:
                v = data_res[k]
                print(f"{v['percent']:.3f} | {np.mean(v['values']):.3f} | {k}")
                self.num_fr = len(v['pred'])
                yield v
        self.data_gen = iter(vis_gen())
        self.show_animation()

        

    def record_video(self):
        frame_dir = f'{args.video_dir}/frames'
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir, exist_ok=True)
        for fr in range(self.num_fr):
            self.fr = fr
            self.update_pose()
            for _ in range(20):
                self.render()
            if not args.preview:
                t0 = time.time()
                save_screen_shots(self.env_vis.viewer.window, f'{frame_dir}/%04d.png' % fr)
                print('%d/%d, %.3f' % (fr, self.num_fr, time.time() - t0))

        if not args.preview:
            out_name = f'{args.video_dir}/{args.cfg}_{"expert" if args.record_expert else args.iter}.mp4'
            cmd = ['/usr/local/bin/ffmpeg', '-y', '-r', '30', '-f', 'image2', '-start_number', '0',
                '-i', f'{frame_dir}/%04d.png', '-vcodec', 'libx264', '-crf', '5', '-pix_fmt', 'yuv420p', out_name]
            subprocess.call(cmd)

def run_seq(data_idx):
    results = {}
    policy_net.load_state_dict(model_cp['policy_dict'])
    value_net.load_state_dict(model_cp['value_dict'])
    pbar = tqdm(tqdm(data_idx))
    for idx in pbar:
        data_loader.set_seq_counter(idx)
        env.load_expert(data_loader.iter_seq())
        state = env.reset()
        cur_key = data_loader.curr_key
        seq_result = defaultdict(list)
        
        if running_state is not None:
            state = running_state(state, update=False)
        seq_result['fail_safe'] = False
        with torch.no_grad():
            for t in range(1000000):
                epos = env.get_expert_attr('qpos', env.get_expert_index(t)).copy()
                seq_result['gt'].append(epos)
                seq_result['pred'].append(env.get_humanoid_qpos())
                seq_result['obj_pose'].append(env.get_obj_qpos())
                state_var = tensor(state, dtype=dtype).unsqueeze(0).clone()
                action = policy_net.select_action(state_var, mean_action=True)[0].cpu().numpy()
                values = value_net(state_var)[0].cpu().numpy()
                seq_result['values'].append(values)
                next_state, reward, done, res = env.step(action)
                
                if running_state is not None:
                    next_state = running_state(next_state, update=False)

                if done:
                    seq_result['percent'] = res['percent']
                    # pbar.set_description(f"{res['percent']} | {np.mean(seq_result['values']):.3f} | {cur_key}")
                    if args.fail_safe and res['percent'] != 1:
                        print(f"Triggering fail safe at timestep {env.cur_t}: {data_loader.curr_key} | {res['percent']:.3f}")
                        env.fail_safe()
                        seq_result['fail_safe'] = True
                    else:
                        break

                state = next_state
        results[cur_key] = seq_result
    return results


def test_coverage():
    jobs = list(range(data_loader.get_len()))
    # jobs = list(range(10))
    # run_seq(jobs)

    from torch.multiprocessing import Pool
    coverage = 0
    data_res_full = {}
    data_res_coverage = {}
    num_jobs = args.num_threads
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
    for k, res in data_res_full.items():
        data_res_coverage[k] = {
            "percent": res['percent'], 
            "values": res['values']
        }
        print(f"{res['percent']}  | {k} | {res['fail_safe']}")
        if res["percent"] == 1 and not res["fail_safe"] :
            coverage += 1
        
    print(f"Coverage of {coverage} out of {data_loader.get_len()}")
    # res_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_coverage.pkl")
    # res_full_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_coverage_full.pkl")
    res_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_relive_coverage.pkl")
    res_full_dir = osp.join(cfg.output_dir, f"{args.iter}_{action}_relive_coverage_full.pkl")
    joblib.dump(data_res_coverage, res_dir)
    joblib.dump(data_res_full, res_full_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--vis_model_file', default='humanoid_smpl_neutral_mesh_vis')
    parser.add_argument('--iter', type=int, default=-1)
    parser.add_argument('--focus', action='store_true', default=True)
    parser.add_argument('--hide_expert', action='store_true', default=False)
    parser.add_argument('--preview', action='store_true', default=False)
    parser.add_argument('--azimuth', type=float, default=45)
    parser.add_argument('--video_dir', default='out/videos/motion_im')
    parser.add_argument('--mode', type=str, default='vis')
    parser.add_argument('--input', action='store_true', default=False)
    parser.add_argument('--num_threads',  type = int, default=20)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--action', type=str, default="all")
    parser.add_argument('--record_expert', action='store_true', default=False)
    parser.add_argument('--fail_safe', action='store_true', default=False)

    args = parser.parse_args()
    cfg = Config(cfg_id = args.cfg, create_dirs=False)

    action_meta = [("sit", 7), ("push", 14), ("avoid", 7), ("step", 7)]

    """environment"""
    action = args.action
    cfg.data_specs['test_file_path'] = "sample_data/h36m_train_no_sit_30_qpos.pkl"
    data_loader = DatasetSMPLObj(cfg.data_specs, data_mode="test")
    # data_loader.data_keys = [k for k in data_loader.data_keys if k.startswith(action)]
    data_loader.data_keys = data_loader.data_keys
    
    init_expert = data_loader.sample_seq()

    # cfg.mujoco_model_file = f"humanoid_smpl_neutral_mesh_{action}_h36m.xml"

    cfg.mujoco_model_file = f"humanoid_smpl_neutral_bigfoot_{action}_h36m.xml"
    args.vis_model_file = f"humanoid_smpl_neutral_mesh_{action}_h36m_vis"

    """make and seed env"""
    model = load_model_from_path(f'assets/mujoco_models/{args.vis_model_file}.xml')
    if args.mode != "stats" :
        sim = MjSim(model)
        viewer = MjViewer(sim)

    from uhc.khrylib.models.mlp import MLP
    from uhc.envs.humanoid_im import HumanoidEnv
    from uhc.khrylib.utils import *
    from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
    from uhc.models.policy_mcp import PolicyMCP
    from uhc.khrylib.rl.core.critic import Value
    cfg.env_start_first = True
    device = torch.device("cpu")
    
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.manual_seed(cfg.seed)
    torch.set_grad_enabled(False)
    env = HumanoidEnv(cfg, init_expert = init_expert, data_specs = cfg.data_specs, mode="test")
    env.seed(cfg.seed)
    actuators = env.model.actuator_names
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    """load learner policy"""
    if cfg.actor_type == "gauss":
        policy_net = PolicyGaussian(cfg, action_dim = action_dim, state_dim = state_dim)
    elif cfg.actor_type == "mcp":
        policy_net = PolicyMCP(cfg, action_dim = action_dim, state_dim = state_dim)

    value_net = Value(MLP(state_dim, cfg.value_hsize, cfg.value_htype))
    cp_path = '%s/iter_%04d.p' % (cfg.model_dir, args.iter)
    logger.info('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    running_state = model_cp['running_state']

    

    if args.mode == "stats":
        test_coverage()
    else:
        policy_net.load_state_dict(model_cp['policy_dict'])
        value_net.load_state_dict(model_cp['value_dict'])
        policy_net.to(device)
        value_net.to(device)
        vis = MyVisulizer(f'assets/mujoco_models/{args.vis_model_file}.xml')
        if args.mode == "vis":
            vis.show_animation()
        elif args.mode == "record":
            vis.record_video()
        elif args.mode == "disp_stats":
            vis.display_coverage()

    # if args.record:
    #     vis.record_video()
    # else:
    #     vis.show_animation()