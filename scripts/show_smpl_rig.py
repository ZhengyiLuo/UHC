import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
import pickle as pk
import argparse
import glfw
import math
import joblib
from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import (
    quaternion_matrix,
    quaternion_about_axis,
    quaternion_inverse,
    quaternion_multiply,
    rotation_from_quaternion,
    rotation_from_matrix,
)

from uhc.utils.transform_utils import (
    convert_aa_to_orth6d,
    convert_orth_6d_to_aa,
    vertizalize_smpl_root,
    vertizalize_smpl_root_and_trans,
    rotate_smpl_root_and_trans,
    rotation_matrix_to_angle_axis,
    convert_orth_6d_to_mat,
)

from uhc.utils.torch_utils import (
    # get_heading_batch,
    # get_heading_q,
    quaternion_multiply,
    quaternion_inverse,
    # get_heading_q_batch,
    # transform_vec_batch,
    # quat_from_expmap_batch,
    # quat_mul_vec_batch,
    # get_qvel_fd_batch,
    # transform_vec,
    # rotation_from_quaternion,
    # de_heading_batch,
    # quat_mul_vec,
    # quat_from_expmap,
    quaternion_multiply_batch,
    quaternion_inverse_batch,
)

from uhc.smpllib.smpl_mujoco import smpl_6d_to_qpose, smpl_to_qpose
from scipy.spatial.transform import Rotation as sRot

parser = argparse.ArgumentParser()
# parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_bigfoot")
# parser.add_argument("--model_id", type=str, default="humanoid_smpl_neutral_start")
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_masterfoot')
parser.add_argument("--model_id",
                    type=str,
                    default="humanoid_smpl_neutral_mesh_test")
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_vis')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_all_single_vis_plain')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_sit')
# parser.add_argument('--model_id', type=str, default='humanoid_smpl_neutral_mesh_push')
parser.add_argument("--offset_z", type=float, default=0.0)
parser.add_argument("--start_take", type=str, default="S1,Directions 1")
args = parser.parse_args()


def update_mocap():
    # sim.data.qpos[:76] = amass_data[0]
    # print(fr % amass_data.shape[0])
    sim.data.qpos[:76] = amass_data[fr % amass_data.shape[0]]

    model.geom_pos[25:, :] = np.random.random(model.geom_pos[25:, :].shape)
    model.geom_quat[25:, :] = np.random.random(model.geom_quat[25:, :].shape)
    model.geom_size[25:, :] = np.random.random(
        model.geom_size[25:, :].shape) * 100

    model.geom_rgba[50:, :] = np.array([0.0, 0.8, 1.0, 1.0])

    # sim.data.qpos[:] = smpl_qpos_2_masterfoot(amass_data[fr % amass_data.shape[0]])
    # sim.data.qpos[7:] = 0
    # sim.data.qpos[76:83] = obj_data[fr % amass_data.shape[0]]
    # sim.data.qpos[:76] = amass_data_gt[fr % amass_data.shape[0]]
    # sim.data.qpos[76:] = amass_data_gt[fr % amass_data.shape[0]]
    # sim.data.qpos[2] += offset_z
    sim.forward()
    # sim.step()


model_file = f"assets/mujoco_models/{args.model_id}.xml"
model = load_model_from_path(model_file)
sim = MjSim(model)

ngeom = 24
# model.geom_rgba[ngeom + 1: ngeom * 2 - 1] = np.array([1, 0, 0, 1])

viewer = MjViewer(sim)

glfw.set_window_size(viewer.window, 10, 10)
glfw.set_window_pos(viewer.window, 400, 0)


def remove_base_rot(quat, base_rot):
    return quaternion_multiply(quat, quaternion_inverse(base_rot))


# amass_data = pk.load(open("data/rendering/test_data.pkl", "rb"))
# amass_data = pk.load(open("/hdd/zen/data/ActBound/Language/netural_pose.pkl", "rb"))
# amass_test_data = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take2_test.pkl")
# amass_test_data = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take2_test.pkl")
# amass_test_data = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_all_qpos.pkl")
# amass_test_data= joblib.load("/hdd/zen/data/ActBound/AMASS/standing_neutral.pkl")
# amass_data = amass_test_data['qpos'][None,]
# seq_data = amass_test_data['0-BioMotionLab_NTroje_rub097_0002_treadmill_slow_poses']
# seq_data = amass_test_data['1001_take_01']
# amass_data = seq_data['qpos']

# expert = seq_data['expert']
# wbpos = expert['wbpos']
# wbpos = wbpos.reshape(wbpos.shape[0], 24, 3)
# begin_feet = min(wbpos[0, 4, 2],  wbpos[0, 8, 2])
# begin_feet -= 0.015
# print(begin_feet, wbpos[0, 4, 2],  wbpos[0, 8, 2])
# expert["qpos"][:, 2] -= begin_feet
# amass_data = expert["qpos"]

# amass_data = amass_data[:150, :]
# amass_data[:, :] = amass_data[10, :]
# amass_data[:, 16:22] = 0

# amass_data[:, 3:7] = [ 0, 0, -0.5440211, -0.8390715 ]

# base_rot = [0.7071, 0.7071, 0.0, 0.0]
# quat = remove_base_rot(amass_data[0, 3:7], base_rot)
# hq = get_heading_q(quat)
# print(hq)
# amass_data[:, 3:7] = base_rot
# # amass_data[:, 3:7] = [1, 0, 0, 0]
# print(get_heading_q(amass_data[0, 3:7]))

# amass_data[:, 3:7] = quaternion_multiply(quaternion_inverse(hq), amass_data[0, 3:7])
# amass_data[:, 3:7] = de_heading(amass_data[0, 3:7])
# sim.data.qpos[:] = amass_data[0]
# data_load= joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_qpos.pkl")
# data_load= joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_qpos_grad.pkl")
# test_data = data_load['push-1219_take_118']
# test_data = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_wild_qpos.pkl")['08-18-2020-13-26-14']
# full_pose = vertizalize_smpl_root(torch.from_numpy(test_data['pose_aa']))
# data_load= joblib.load("/hdd/zen/data/ActBound/AMASS/relive_third_qpos.pkl")
# test_data = data_load[list(data_load.keys())[0]]
# data_load = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_train_singles.pkl")
# amass_data = data_load["0-ACCAD_Male2Running_c3d_C20 - run to pickup box_poses"]["qpos"]

############################################################################################################################################################
# data_load = joblib.load("/hdd/zen/dev/copycat/ACTOR/gen_test.pkl")

# idx = 0
# # full_pose = data_load['gt_pose_aa'][idx]
# # trans = data_load['gt_pose_trans'][idx]
# full_pose = data_load['out_pose_aa'][idx]
# trans = data_load['out_pose_trans'][idx]
# # import ipdb; ipdb.set_trace()
# amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)
############################################################################################################################################################
# data_load = joblib.load("/hdd/zen/dev/copycat/MEVA/test.pkl")
# pose_aa = data_load['gt']
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), mj_model = model)

# data_load = joblib.load("/hdd/zen/data/video_pose/3dpw/test/test.pkl")
# pose_aa = data_load['poses']
# trans = data_load['trans']

# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa), trans = torch.from_numpy(trans).double())
# pose_aa, trans = pose_aa.numpy(), trans.numpy().squeeze()
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), trans = trans, mj_model = model)

# data_load = joblib.load('/hdd/zen/data/video_pose/GPA/smpl_fits/gpa_dataset_smplx_grad_fitted_test.pkl')
# data_load = joblib.load('/hdd/zen/data/video_pose/GPA/smpl_fits/gpa_dataset_smplx_fitted_test.pkl')
# data_load = joblib.load("/hdd/zen/data/video_pose/GPA/smpl_fits_test/data_split000.pkl")
# data_load = data_load['zw_static_00_12_02']
# pose_aa = data_load['pose']
# trans = data_load['trans']
# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa).float(), trans = torch.from_numpy(trans).float())
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), trans = trans, mj_model = model)

# data_load = joblib.load("/hdd/zen/dev/copycat/Sceneplus/debug.pkl")[1]
# pose_aa = data_load['pose_aa'].detach().cpu().numpy()[0].reshape(-1, 72)
# trans = data_load['trans'].detach().cpu().numpy()[0].squeeze()
# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa).float(), trans = torch.from_numpy(trans).float())
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), trans = trans, mj_model = model)

# humor_res = dict(np.load("/hdd/zen/dev/copycat/humor/out/proxd_fitting_sub_next/results_out/MPH1Library_00145_01_0033_16462461743/stage3_results.npz"))
# humor_res = dict(np.load("/hdd/zen/dev/copycat/humor/out/proxd_no_overlap_tet/results_out/MPH1Library_03301_01_0020_16462717360/stage3_results.npz"))
# humor_res = dict(np.load("/hdd/zen/dev/copycat/humor/out/proxd_fitting_sub2/results_out/N3Office_03301_01_0035_16461852865/stage3_results.npz"))
# humor_res = dict(np.load("/hdd/zen/dev/copycat/humor/out/proxd_fitting_sub1_test/results_out/N3Office_03301_01_0034_16463642790/stage3_results.npz"))
# # humor_res = dict(np.load("/hdd/zen/dev/copycat/humor/out/proxd_fitting_sub1_humor/results_out/MPH1Library_00145_01_0033_16463528001/stage3_results.npz"))
# b_size, _ = humor_res['pose_body'].shape
# pose_aa = np.concatenate([humor_res['root_orient'], humor_res['pose_body'], np.zeros([b_size, 6])], axis = 1)
# pose_aa = pose_aa
# trans = humor_res['trans']
# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa).float(), trans = torch.from_numpy(trans).float())
# trans[:, 2] += 0.89 - trans[0, 2]
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), trans = trans, mj_model = model)

# relive_mocap_grad = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_smpl.pkl")
# # relive_mocap_grad = joblib.load("/hdd/zen/data/ActBound/AMASS/relive_mocap_smpl_grad.pkl")
# # # relive_mocap_grad = joblib.load("/hdd/zen/data/ActBound/AMASS/egopose_mocap_smpl_grad.pkl")
# print(relive_mocap_grad.keys())
# # k = "sit-1011_take_04"
# # k = "avoid-1213_take_44"
# k = "step-1219_take_63"
# full_pose = relive_mocap_grad[k]['pose']
# trans = relive_mocap_grad[k]['trans']
# amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)

# relive_mocap_grad = joblib.load("/hdd/zen/data/ActBound/AMASS/singles/amass_copycat_take5_singles_run.pkl")
# print(relive_mocap_grad.keys())
# k = "0-ACCAD_Male2Running_c3d_C3 - run_poses"
# full_pose = relive_mocap_grad[k]['pose_aa']
# trans = relive_mocap_grad[k]['trans']
# amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)

# relive_mocap_grad = joblib.load("/hdd/zen/data/video_pose/Tennis/demo/denmarkopen_2021_mens_semifinal_leecheukyiu_axelsen/test.pkl")
# print(relive_mocap_grad.keys())
# k = "tennis"
# full_pose = relive_mocap_grad[k]['pose_aa']
# trans = relive_mocap_grad[k]['trans']
# amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)


proxd_res = joblib.load("/hdd/zen/dev/copycat/Sceneplus/results/third/proxd_processed.pkl")
k = "MPH1Library_00145_01"
full_pose = proxd_res[k]['pose_aa']
trans = proxd_res[k]['trans']
amass_data = smpl_to_qpose(pose = full_pose, mj_model = model, trans = trans)


# pose_test = joblib.load("/hdd/zen/data/video_pose/mot/pose_test.pkl")
# pose_test = joblib.load("/hdd/zen/data/video_pose/mot/youtube_0/smpl_processed.pkl")
# track_id = "27"
# pose_aa = pose_test[track_id]['pose_aa']
# trans = pose_test[track_id]['trans']
# trans[:, 2] += 0.89 - trans[0, 2]
# amass_data = smpl_to_qpose(pose=pose_aa, mj_model=model, trans=trans)


# # import ipdb; ipdb.set_trace()
# seq_len = amass_data.shape[0]
# base_rot = torch.tensor([[0.7071, 0.7071, 0.0, 0.0]])
# base_rot_batch = base_rot.repeat(seq_len, 1)

# amass_data[:, 3:7] = quaternion_multiply_batch(torch.from_numpy(amass_data[:, 3:7]).float(), quaternion_inverse_batch(base_rot_batch)).numpy()

# from uhc.utils.torch_geometry_transforms import (
#     angle_axis_to_rotation_matrix as aa2mat, rotation_matrix_to_angle_axis as
#     mat2aa)

# # test_data = joblib.load("/hdd/zen/dev/copycat/humor/test.pkl")
# test_data = joblib.load("/hdd/zen/dev/copycat/Sceneplus/test.pkl")

# # trans = []
# # pose_body = []
# # root_orient = []
# # for data_dict in test_data:
# #     trans.append(data_dict['trans'])
# #     pose_body.append(data_dict['pose_body'])
# #     root_orient.append(data_dict['root_orient'])
# # trans, pose_body, root_orient = torch.cat(trans), torch.cat(
# #     pose_body), torch.cat(root_orient)

# trans, root_orient, pose_body = test_data['trans'].squeeze().detach(
# ), test_data['root_orient'].squeeze().detach(), test_data['pose_body'].squeeze(
# ).detach()

# seq_len = trans.shape[0]
# body_pose_aa = mat2aa(pose_body.reshape(seq_len * 21, 3,
#                                         3)).reshape(seq_len, 63)
# root_aa = mat2aa(root_orient.reshape(seq_len, 3, 3)).reshape(seq_len, 3)
# pose_aa = torch.cat(
#     [root_aa, body_pose_aa,
#      torch.zeros(seq_len, 6).to(root_aa)], dim=1)

# amass_data = smpl_to_qpose(pose=pose_aa.cpu().numpy(),
#                            mj_model=model,
#                            trans=trans.cpu().numpy().squeeze())

# headings_q = np.stack([get_heading_q_new(hq) for hq in amass_data[:, 3:7].copy()])

# headings_q = np.stack([get_heading_q(hq) for hq in root_quat_remove])
# import ipdb; ipdb.set_trace()

# amass_data[:, 3:7] = quaternion_multiply_batch(quaternion_inverse_batch(torch.from_numpy(headings_q).float()), torch.from_numpy(amass_data[:, 3:7]).float())
# print(get_heading_new(amass_data[0, 3:7]))

# # humor_res = joblib.load("/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_test.pkl")
# humor_res = joblib.load("/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap.pkl")
# humor_res = joblib.load("/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd_overlap_test.pkl")
# # # # humor_res = joblib.load("/hdd/zen/data/video_pose/prox/qualitative/thirdeye_anns_proxd.pkl")
# keys = ['MPH1Library_00034_01', 'MPH1Library_00145_01', 'MPH1Library_03301_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01', 'N0Sofa_00145_01', 'N0Sofa_03403_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01', 'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01', 'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00139_02', 'N3Office_00150_01', 'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']
# # # import ipdb; ipdb.set_trace()
# # # N3Library_03301_02, N3Library_03375_01, N3Library_03375_02
# # # humor_res = humor_res['N0Sofa_00145_01']
# # ['N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01', 'N0Sofa_00145_01', 'N0Sofa_03403_01']
# # key = keys[23]
# key = "N0Sofa_00145_01"
# print(key, key, key, key, key, keys.index(key))
# humor_res = humor_res[key]
# # humor_res = humor_res['MPH1Library_03301_01']
# b_size, _ = humor_res['pose_aa'].shape
# pose_aa = humor_res['pose_aa']
# trans = humor_res['trans']
# amass_data = smpl_to_qpose(pose = pose_aa.reshape(-1, 24, 3), trans = trans, mj_model = model)

# rotation = ["MPH1Library_00145_01", "N0Sofa_00145_01", "N0Sofa_03403_01"]

# 316, 1167, 1365; 7

# 22-23
# 26-27

# print(test_data.keys())
# data_load= joblib.load("/hdd/zen/data/copycat/seqs/AIST++/aist_qpos.pkl")
# test_data = data_load[list(data_load.keys())[0]]
# full_pose = test_data['pos\
# amass_data = test_data['expert']['qpos']
# smpl_test, _ = joblib.load("/hdd/zen/dev/copycat/Relive/results/all/statear/smpl_ar_1/results/iter_0480_test.p")
# amass_data = smpl_test['traj_pred']['sit-1011_take_15']

# obj_pose =  test_data['obj_pose']
# trans = test_data['trans']

# full_pose, trans = joblib.load("/hdd/zen/data/copycat/seqs/test_rig.pkl")
# amass_data = smpl_to_qpose(pose = full_pose, model = model, trans = trans)

# ego_data, _ = pk.load(open("/hdd/zen/dev/copycat/EgoPose/results/statereg/all_01_traj/results/iter_0030_test.p", "rb"))

# ego_data, _ = pk.load(open("/hdd/zen/dev/copycat/EgoPose/results/statereg/all_01_traj/results/iter_0095_test.p", "rb"))
# print(ego_data["traj_pred"].keys())
# k = "push-08-30-2020-16-52-02"
# amass_data = ego_data["traj_pred"][k]
# orig_pose = ego_data["traj_orig"][k]
# print(amass_data.shape, orig_pose.shape)
# data_load= joblib.load("/hdd/zen/data/copycat/seqs/AIST++/aist_smpl.pkl")

# test_data = joblib.load("/hdd/zen/dev/ActMix/actmix/DataGen/MotionCapture/VIBE/test.npy")
# full_pose = torch.from_numpy(test_data['pose'])
# full_pose = vertizalize_smpl_root(full_pose)
# amass_data = smpl_to_qpose(pose = full_pose, model = model, trans = test_data['trans'])

# k = 'gWA_sBM_cAll_d26_mWA5_ch03'
# full_pose = data_load[k]['pose']
# trans = data_load[k]['trans']
# trans[:, 2] += 3
# full_pose = vertizalize_smpl_root(torch.from_numpy(full_pose))

# movi_data = joblib.load('/hdd/zen/data/video_pose/movi/movi_data.pkl')
# print(movi_data.keys())
# test_data = movi_data['PG1-Subject_79-move_2']
# full_pose = torch.from_numpy(test_data['pose'])
# amass_data = smpl_to_qpose(pose = full_pose, model = model, trans = test_data['trans'])

# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_test_60_fitted_smpl_grad.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_60_fitted_smpl_grad.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_30_fitted_grad_test.pkl")
# h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_test_60_fitted.p")
# h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_full.p")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_30_fitted_grad_qpos_full.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_test_30_qpos.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_no_sit_30_qpos.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_test_60_fitted_grad_test.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_60_fitted_grad.pkl")
# h36m_data = joblib.load("/hdd/zen/data/ActBound/AMASS/h36m_train_60_fitted_test.pkl")
# h36m_data = joblib.load(
#     "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_30_fitted_grad_full_test.p"
# )

# cam_space_data = joblib.load(
#     "/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_fitted.p"
# )
# print(h36m_data.keys())
# amass_data = h36m_data["S11-Directions 1"]["qpos"]
# cam = np.array(h36m_data["S1-Directions 1.54138969"]["cam"]["R"])
# t = np.array(h36m_data["S1-Directions 1.54138969"]["cam"]["t"]) / 1000
# root_vecs = amass_data[:, 3:7]
# rot_rots = sRot.from_quat(root_vecs[:, [1, 2, 3, 0]])
# rot_rots_new = sRot.from_matrix(cam.T) * rot_rots
# amass_data[:, 3:7] = rot_rots_new.as_quat()[:, [3, 0, 1, 2]]
# amass_data[:, :3] -= t
# amass_data[:, :3] = np.dot(amass_data[:, :3], cam)
# print(amass_data.shape, amass_data[:, :3])

# pose_aa = h36m_data["S1-Directions 1.54138969"]["pose"]
# trans = h36m_data["S1-Directions 1.54138969"]["trans"]
# amass_data = smpl_to_qpose(pose=pose_aa, model=model, trans=trans)

# test_data = h36m_data['S1-WalkingDog']
# test_data = h36m_data['S5-SittingDown']
# amass_data = h36m_data["S1-Eating"]["qpos"]
# # obj_data =  h36m_data['S1-Eating']['obj_pose']
# skip = 1

# skip = 2
# test_data = h36m_data['S1-WalkingDog 1.60457274']
# test_data = h36m_data['S1-SittingDown 2.60457274']
# test_data = h36m_data['S11-Directions 1']
# cam = np.array(test_data['cam'])
# pose_aa = test_data['pose'][::skip]
# trans = test_data['trans'][::skip]
# amass_data = smpl_to_qpose(pose = pose_aa, model = model, trans = trans)

# test_data = joblib.load('/hdd/zen/data/video_pose/prox/results/sample.pkl')
# pose_aa = test_data['sample']['pose'].squeeze()
# trans = test_data['sample']['trans'].squeeze()

# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa), trans = torch.from_numpy(trans).double())
# pose_aa, trans = pose_aa.numpy(), trans.numpy().squeeze()
# trans[:, 2] += 0.89 - trans[0, 2]

# # root_rot = sRot.from_rotvec(pose_aa[:, :3])
# # print(trans[:, :3])

# h36m_data = joblib.load("/hdd/zen/data/video_pose/h36m/data_fit/h36m_train_60_fitted_grad_qpos_full.p")
# amass_data = h36m_data['S1-WalkingDog 1.60457274']['qpos']

# amass_data = joblib.load("/hdd/zen/data/ActBound/tests/uhm_res.pkl")['qpos']

# res_data = joblib.load("/hdd/zen/dev/copycat/Sceneplus/results/video_pose/video_pose_init/results/eval_res_0020.pt")
# amass_data = res_data['pred']['qpos'].cpu().numpy().squeeze()
# amass_data_gt = res_data['gt']['qpos'].cpu().numpy().squeeze()

# data_load = joblib.load("/mnt/hdd/zen/dev/copycat/MST10192 Final Working/res/A_ _p_e_r_s_o_n_ _w_a_l_k_s_ _f_o_r_w_a_r_d.pkl")
# pose_aa = data_load['pose']
# trans = data_load['trans']

# data_3dpw = pk.load(open("/hdd/zen/data/video_pose/3dpw/sequenceFiles/test/downtown_arguing_00.pkl", 'rb'), encoding = "latin1")
# pose_aa = data_3dpw['poses'][0]
# trans = data_3dpw['trans'][0]
# pose_aa, trans = vertizalize_smpl_root_and_trans(torch.from_numpy(pose_aa), trans = torch.from_numpy(trans).double())
# print(pose_aa.shape, trans.shape)
# amass_data = smpl_to_qpose(pose = pose_aa, model = model, trans = trans.squeeze())

# amass_data = smpl_to_qpose(pose = pose_aa, model = model, trans = trans.squeeze())
print(amass_data.shape)
T = 10
paused = False
stop = False
reverse = False
offset_z = args.offset_z
viewer._hide_overlay = True
viewer.cam.distance = 5
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
# viewer.custom_key_callback = key_callback

t = 0
fr = 0
while not stop:
    # import pdb
    # pdb.set_trace()
    if t >= math.floor(T):
        update_mocap()
        fr += 1
        t = 0

    viewer.render()
    if not paused:
        t += 1
