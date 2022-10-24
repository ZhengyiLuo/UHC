import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import pickle as pk
import argparse
import math
import numpy as np
import cv2
import joblib
from tqdm import tqdm
import torch
from mujoco_py import load_model_from_path, MjSim
os.environ['OMP_NUM_THREADS'] = "1"
# from uhc.khrylib.utils import *
from uhc.smpllib.smpl_mujoco import smpl_to_qpose, SMPL_M_Renderer, normalize_smpl_pose

from uhc.utils.transform_utils import (
    convert_aa_to_orth6d, convert_orth_6d_to_aa, vertizalize_smpl_root, vertizalize_smpl_root_and_trans, 
    rotation_matrix_to_angle_axis, convert_orth_6d_to_mat
)

def render_amass_db_list(data_list, target_fr = 30):
    smpl_renderer = SMPL_M_Renderer()
    pbar = tqdm(data_list)
    for k, v in pbar:
        # text = v['text']
        amass_fr = v['mocap_framerate']
        skip = int(amass_fr/target_fr)
        full_pose = v['poses'][::skip]
        tran = v['trans'][::skip]
        # full_pose, tran = normalize_smpl_pose(full_pose, trans = tran, random_root = True)
        # print("Rendering: ", text)
        out_name = "{}/{}.mp4".format(args.output, k)

        if not osp.isfile(out_name):
            smpl_renderer.render_smpl(body_pose = full_pose, tran = tran,  output_name=out_name, frame_rate=30)
            pbar.set_description(k)
        else:
            print("done", out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh')
    parser.add_argument('--offset_z', type=float, default=0.0)
    parser.add_argument('--start_take', type=str, default='S1,Directions 1')
    parser.add_argument('--data', type=str, default='test_data')
    parser.add_argument('--output', type=str, default='test')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # smpl_renderer = SMPL_M_Renderer()
    # amass_lag_data = joblib.load("/hdd/zen/data/ActBound/Language/amass_lag_take2.pkl")
    # i = 0
    # for k, v in tqdm(amass_lag_data.items()):
    #     text = v['text']
    #     full_pose = v['pose']
    #     X_r = convert_orth_6d_to_aa(torch.tensor(full_pose[:,3:]))
    #     tran = full_pose[:,:3]
    #     print("Rendering: ", text)
    #     smpl_renderer.render_smpl(body_pose = X_r, tran = tran, output_name="rendering/{}_{}.mp4".format(args.output, i), frame_rate=30, add_text = text)
    #     i += 1

    ##### AMASS DB #####
    amass_data = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_db_smplx.pt")
    jobs = list(amass_data.items())

    # from multiprocessing import Pool
    # num_jobs = 10
    # chunk = np.ceil(len(jobs)/num_jobs).astype(int)
    # jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    # argumenets = [(jobs[i],) for i in range(len(jobs))]
    # print(len(argumenets))
    # try:
    #     pool = Pool(num_jobs)   # multi-processing
    #     pool.starmap(render_amass_db_list, argumenets)
    # except KeyboardInterrupt:
    #     pool.terminate()
    #     pool.join()
    render_amass_db_list(jobs)
    
        
        
    ##### MOVI DB #####
    # smpl_data = joblib.load("/hdd/zen/data/video_pose/movi/meva_res/movi_S1_PG1.pkl")
    # smpl_data = joblib.load("/hdd/zen/data/Reallite/contextegopose/EgoPoseObjectDataset/meva/meva_res.pkl")
    # for k, v in tqdm(smpl_data.items()):
    #     print(k)
    #     # full_pose = v['thetas']
    #     full_pose = v['thetas_smoothed']
    #     full_pose = vertizalize_smpl_root(torch.from_numpy(full_pose))
    #     # print("Rendering: ", text)
    #     smpl_renderer.render_smpl(body_pose = full_pose,  output_name="{}/{}.mp4".format(args.output, k), frame_rate=30)

    # pose_data = joblib.load("/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/MEVA/output/meva/S1_PG1_Subject_17_L/vibe_output.pkl")
    # full_pose = pose_data['S1_PG1_Subject_17_L']['pose'][:300]
    # full_pose = vertizalize_smpl_root(torch.from_numpy(full_pose))
    # # print("Rendering: ", text)
    # smpl_renderer.render_smpl(body_pose = full_pose,  output_name="{}/{}.mp4".format(args.output, "test"), frame_rate=30)

    ##### H36M DB #####
    # smpl_renderer = SMPL_M_Renderer()
    # smpl_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_all_smpl.pkl")
    # for k, v in tqdm(smpl_data.items()):
    #     body_pose = v['pose']
    #     full_trans = v['trans']
    #     # full_pose = v['thetas_smoothed']
    #     body_pose = vertizalize_smpl_root(torch.from_numpy(body_pose))
    #     print("Rendering: ", k)
    #     smpl_renderer.render_smpl(body_pose = body_pose, tran = full_trans, output_name="{}/{}.mp4".format(args.output, k), frame_rate=30)

    ##### Relive DB #####
    # smpl_renderer = SMPL_M_Renderer()
    # smpl_data = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_copycat.pkl")
    # # smpl_data = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_smpl_grad.pkl")
    # for k, v in tqdm(smpl_data.items()):
    #     body_pose = v['pose_aa']
    #     full_trans = v['trans']
    #     # full_pose = v['thetas_smoothed']
    #     # body_pose = vertizalize_smpl_root(torch.from_numpy(body_pose))
    #     print("Rendering: ", k)
    #     smpl_renderer.render_smpl(body_pose = body_pose, tran = full_trans, output_name="{}/{}.mp4".format(args.output, k), frame_rate=30)

    ###### H36M ######
    # follow = True
    # from uhc.utils.image_utils import write_frames_to_video, write_individaul_frames
    # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh_all_h36m.xml")
    # smpl_data = joblib.load("/hdd/zen/dev/copycat/Copycat/results/motion_im/copycat_9/results/18000_all_coverage_full.pkl")

    # render_size = (2048, 1024)
    # azimuth = 75
    # distance = 5
    # smpl_renderer.viewer.cam.distance = distance
    # smpl_renderer.viewer.cam.elevation = -6
    # smpl_renderer.viewer.cam.azimuth = azimuth
    # smpl_renderer.viewer.cam.lookat[2] = 1.0
    
    # renders = ["S1-Discussion 1"]
    # for k, v in tqdm(smpl_data.items()):
    #     if not k in renders:
    #         continue
    #     print(v.keys())
    #     # qpos = np.array(v['pred'])
    #     qpos = np.array(v['gt'])
    #     obj_pose = np.array(v['obj_pose'])
    #     qpos = np.concatenate([qpos, obj_pose], axis=1)
    #     print("Rendering: ", k)
    #     images  = smpl_renderer.render_qpose(qpos, frame_rate=30, size = render_size, follow = follow)
    #     output_name="{}/{}.mp4".format(args.output, k)
    #     img_output_folder = osp.join(args.output, k)
    #     os.makedirs(img_output_folder, exist_ok=True)
    #     write_individaul_frames(images, output_dir = img_output_folder)
    #     write_frames_to_video(images, out_file_name = output_name, frame_rate = 30, add_text = None)


    ###### VIBE ######
    # pose_data = joblib.load("/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/MEVA/output/meva/S1_PG1_Subject_17_L/vibe_output.pkl")
    # full_pose = pose_data['S1_PG1_Subject_17_L']['pose'][:300]
    # full_pose = vertizalize_smpl_root(torch.from_numpy(full_pose))
    # # print("Rendering: ", text)
    # smpl_renderer.render_smpl(body_pose = full_pose,  output_name="{}/{}.mp4".format(args.output, "test"), frame_rate=30)

    ###### Prox ######
    # follow = False
    # render_size = (1500, 1024)
    # from uhc.utils.image_utils import write_frames_to_video, write_individaul_frames
    # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml")
    # smpl_data = joblib.load("/hdd/zen/dev/copycat/Copycat/results/motion_im/copycat_30/results/13600_usr_coverage_full.pkl")
    # smpl_renderer.model.geom_rgba[1:] = np.array([0.7, 0.0, 0.0, 1])
    # for k, v in tqdm(smpl_data.items()):
    #         # qpos = np.array(v['pred'])
    #         qpos = np.array(v['gt'])
    #         print("Rendering: ", k)
    #         smpl_renderer.render_qpose_and_write(qpos, output_name="{}/{}.mp4".format(args.output, "reference"), frame_rate=30, size = render_size, follow = follow)

    ###### DAIS ######
    # follow = False
    # render_size = (750, 512)
    # from uhc.utils.image_utils import write_frames_to_video, write_individaul_frames
    # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml")
    # smpl_data = joblib.load("/hdd/zen/dev/copycat/Copycat/results/motion_im/copycat_30/results/13600_usr_coverage_full.pkl")
    # # smpl_renderer.model.geom_rgba[1:] = np.array([0.7, 0.0, 0.0, 1])
    # for k, v in tqdm(smpl_data.items()):
    #         qpos = np.array(v['pred'])
    #         # qpos = np.array(v['gt'])
    #         print("Rendering: ", k)
    #         smpl_renderer.render_qpose_and_write(qpos, output_name="{}/{}.mp4".format(args.output, k), frame_rate=30, size = render_size, follow = follow)

    ### Big foot ###
    # follow = False
    # render_size = (750, 512)
    # from uhc.utils.image_utils import write_frames_to_video, write_individaul_frames
    # # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml")
    # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_bigfoot.xml")
    # # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_masterfoot.xml")
    # smpl_data = joblib.load("/hdd/zen/dev/copycat/Copycat/results/motion_im/bigfoot_9/results/7800_all_coverage_full.pkl")
    # # smpl_data = joblib.load("/hdd/zen/dev/copycat/Copycat/results/motion_im/bigfoot_fut_1/results/1300_all_coverage_full.pkl")
    # # smpl_renderer.model.geom_rgba[1:] = np.array([0.7, 0.0, 0.0, 1])
    # for k, v in tqdm(smpl_data.items()):
    #     qpos = np.array(v['pred'])
    #     # qpos = np.array(v['gt'])
    #     print("Rendering: ", k)

    #     out_name = "{}/{}.mp4".format(args.output, k) if v['percent'] == 1 else "f_{}/{}.mp4".format(args.output, k)
    #     smpl_renderer.render_qpose_and_write(qpos, output_name=out_name, frame_rate=30, size = render_size, follow = follow)

 
    ### 3dpw ###
    # follow = False
    # render_size = (512, 512)
    # from uhc.utils.image_utils import write_frames_to_video, write_individaul_frames
    # smpl_renderer = SMPL_M_Renderer(model_file = "/hdd/zen/dev/copycat/Copycat/assets/mujoco_models/humanoid_smpl_neutral_mesh.xml")
    # model_file = f'assets/mujoco_models/{args.model_id}.xml'
    # model = load_model_from_path(model_file)
    # data_3dpw = pk.load(open("/hdd/zen/data/video_pose/3dpw/sequenceFiles/test/downtown_arguing_00.pkl", 'rb'), encoding = "latin1")
    # pose_aa = data_3dpw['poses'][0]
    # trans = data_3dpw['trans'][0]
    # pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa), trans = torch.from_numpy(trans).double())
    # trans[:, 2] += 0.89 - trans[:, 2]
    # qpos = smpl_to_qpose(pose = pose_aa, model = model, trans = trans.squeeze())
    # # qpos = np.array(v['gt'])

    # out_name = "{}/{}.mp4".format(args.output, '3dpw') 
    # smpl_renderer.render_qpose_and_write(qpos, output_name=out_name, frame_rate=30, size = render_size, follow = follow)

