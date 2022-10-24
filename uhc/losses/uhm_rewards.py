import numpy as np
import math

from uhc.utils.math_utils import multi_quat_norm_v2, get_angvel_fd, multi_quat_diff
from uhc.utils.flags import flags

 
def dynamic_supervision_v1(env, state, action, info):
    # V1 uses GT 
    # V1 now does not regulate the action using GT, and only has act_v 
    cfg = env.kin_cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))


    # Comparing with GT
    gt_bquat = env.ar_context['bquat'][ind].flatten()
    gt_prev_bquat = env.ar_context['bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    pose_gt_diff = multi_quat_norm_v2(multi_quat_diff(gt_bquat, cur_bquat)).mean()
    
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(gt_prev_bquat, gt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    gt_p_reward = math.exp(-k_act_p * pose_gt_diff)

    reward =  w_p * p_reward + w_jp * jp_reward + w_act_p * gt_p_reward + w_act_v * act_v_reward

    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(reward, np.array([p_reward, jp_reward, gt_p_reward, act_v_reward]))
    
    return reward, np.array([p_reward, jp_reward, gt_p_reward, act_v_reward])



def dynamic_supervision_v2(env, state, action, info):
    # V2 uses no GT
    # velocity loss is from AR-Net , reguralize the actions by running the model kinematically
    # This thing makes 0 sense rn 
    # cfg = env.cc_cfg
    # ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_v, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #      ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_v', 1.0),  ws.get('w_act_p', 1.0)
    # k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_v, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
    #     ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_v', 0.1), ws.get('k_act_p', 0.1)
    # v_ord = ws.get('v_ord', 2)
    
    # ind = env.cur_t
    # # Head losses
    # tgt_hpos = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    # cur_hpos = env.get_head().copy()
    # prev_hpos = env.prev_hpos.copy()

    # hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    # hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # # head orientation reward
    # hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    # hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # # head velocity reward 
    # # hpvel = (cur_hpos[:3] - prev_hpos[:3]) / env.dt
    # # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpos[3:], env.dt)
    # # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    # hv_reward = 0
    
    # cur_bquat = env.get_body_quat()
    # cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    # tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    # pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    # pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    # p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    # jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))

    # ## ARNet Action supervision
    # act_qpos = env.target['qpos']
    # tgt_qpos = env.ar_context['ar_qpos'][ind]

    # act_bquat = env.target['bquat'].flatten()
    # tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    # tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    # prev_bquat = env.prev_bquat
    

    # rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    # pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    # cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    # vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    # act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    # rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    # act_p_reward = math.exp(-k_act_p * (pose_action_diff))
    # # rq_reward = 0
    # # rp_reward = 0
    # # act_p_reward = 0

    
    # reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + \
    #     w_jp * jp_reward + w_rp * rp_reward + w_rq * rq_reward  + w_act_v * act_v_reward + w_act_p * act_p_reward
    # print(reward)
    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_v_reward, act_p_reward])

def dynamic_supervision_v3(env, state, action, info):
    # V3 is V2 mutiplicative
    # This is wrong, very wrong. This does not work since you should compare the simulated with the estimated!!!!!!
    cfg = env.cc_cfg
    ws = cfg.policy_specs['reward_weights']
    # w_hp, w_hq, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
    #     ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq,  k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0),   \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)
    
    ind = env.cur_t
    # Head losses
    tgt_hpos = env.ar_context['head_pose'][ind]
    tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpos = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpos[:3] - tgt_hpos[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpos[3:], tgt_hpos[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    cur_bquat = env.get_body_quat()

    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat, tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))

    ## ARNet Action supervision
    act_qpos = env.target['qpos']
    tgt_qpos = env.ar_context['ar_qpos'][ind]

    act_bquat = env.target['bquat'].flatten()
    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    rp_dist = np.linalg.norm(tgt_qpos[:3] - act_qpos[:3])
    rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    pose_action_diff = multi_quat_norm_v2(multi_quat_diff(tgt_bquat, act_bquat)).mean()

    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))
    # act_v_reward = 0

    rq_reward = math.exp(-k_rq * (rq_dist ** 2))
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    act_p_reward = math.exp(-k_act_p * (pose_action_diff))

    # import pdb; pdb.set_trace()

    # reward = hp_reward * hq_reward  *  p_reward * jp_reward * rp_reward  * rq_reward  * act_p_reward * act_v_reward
    reward = hp_reward * hq_reward  *  p_reward * jp_reward * rp_reward  * rq_reward  * act_p_reward 
    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward]))
    
    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward, act_v_reward])


def dynamic_supervision_v4(env, state, action, info):
    # V4 does not have the action terms (does not regularize the action)
    cfg = env.cc_cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)
    
    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)


    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # head velocity reward 
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    
    
    reward = w_hp * hp_reward + w_hq * hq_reward + w_hv * hv_reward + w_p * p_reward + w_jp * jp_reward 

    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])

def dynamic_supervision_v5(env, state, action, info):
    # V5 is V4 with multiplicative reward
    cfg = env.cc_cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1)
    
    ind = env.cur_t
    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]
    # tgt_hvel = env.ar_context['head_vels'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hpvel = (cur_hpose[:3] - prev_hpos[:3]) / env.dt
    # hqvel = get_angvel_fd(prev_hpos[3:], cur_hpose[3:], env.dt)


    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))

    # head velocity reward 
    # hpvel_dist = np.linalg.norm(hpvel - tgt_hvel[:3])
    # hqvel_dist = np.linalg.norm(hqvel - tgt_hvel[3:])
    # hv_reward = math.exp(-hpvel_dist - k_hv * hqvel_dist)
    hv_reward = 0
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    
    
    reward =  hp_reward  * hq_reward    * p_reward  * jp_reward 

    # if flags.debug:
        # np.set_printoptions(precision=4, suppress=1)
        # print(np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward, rp_reward, rq_reward, act_p_reward]))
    
    return reward, np.array([hp_reward, hq_reward, hv_reward, p_reward, jp_reward])


def dynamic_supervision_v6(env, state, action, info):
    # no head reward anymore 
    cfg = env.cc_cfg
    ws = cfg.policy_specs['reward_weights']
    w_hp, w_hq, w_hv, w_p, w_jp, w_rp, w_rq, w_act_p, w_act_v = ws.get('w_hp', 1.0), ws.get('w_hq', 1.0),\
         ws.get('w_hv', 0.05), ws.get('w_p', 1.0), ws.get('w_jp', 1.0), ws.get('w_rp', 1.0), ws.get('w_rq', 1.0), ws.get('w_act_p', 1.0), ws.get('w_act_v', 1.0)
    k_hp, k_hq, k_hv, k_p, k_jp, k_rp, k_rq, k_act_p, k_act_v = ws.get('k_hp', 1.0), ws.get('k_hq', 1.0), ws.get('k_hv', 1.0 ), \
        ws.get('k_p', 1.0), ws.get('k_jp', 0.1), ws.get('k_rp', 0.1), ws.get('k_rq', 0.1), ws.get('k_act_p', 0.1), ws.get('k_act_v', 0.1)
    v_ord = ws.get('v_ord', 2)

    ind = env.cur_t

    # Head losses
    tgt_hpose = env.ar_context['head_pose'][ind]

    cur_hpose = env.get_head().copy()
    prev_hpos = env.prev_hpos.copy()

    hp_dist = np.linalg.norm(cur_hpose[:3] - tgt_hpose[:3])
    hp_reward = math.exp(-k_hp * (hp_dist ** 2)) 
    
    # head orientation reward
    hq_dist = multi_quat_norm_v2(multi_quat_diff(cur_hpose[3:], tgt_hpose[3:])).mean()
    hq_reward = math.exp(-k_hq * (hq_dist ** 2))
    
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))
    

    tgt_bquat = env.ar_context['ar_bquat'][ind].flatten()
    tgt_prev_bquat = env.ar_context['ar_bquat'][ind - 1].flatten()
    prev_bquat = env.prev_bquat

    
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    tgt_bangvel = get_angvel_fd(tgt_prev_bquat, tgt_bquat, env.dt)
    vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    reward =   w_hp * hp_reward + w_hq * hq_reward + w_p * p_reward + w_jp * jp_reward + w_act_v * act_v_reward

    if flags.debug:
        import pdb; pdb.set_trace()
        np.set_printoptions(precision=4, suppress=1)
        print(reward, np.array([p_reward, jp_reward, act_v_reward]))
    
    return reward, np.array([hp_reward, hq_reward, p_reward, jp_reward, act_v_reward])


def constant_reward(env, state, action, info):
    reward = 1.0
    if info['end']:
        reward += env.end_reward
    return 1.0, np.zeros(1)

def forward_v1(env, state, action, info):
    # V1 uses GT 
    # V1 now does not regulate the action using GT, and only has act_v 
    cfg = env.kin_cfg
    ws = cfg.policy_specs['reward_weights']
    k_rp = ws.get('k_rp', 1.0)


    ind = env.cur_t
    
    cur_bquat = env.get_body_quat()
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)
    tgt_bquat, tgt_wbpos = env.target['bquat'], env.target['wbpos']

    # pose_quat_diff = multi_quat_norm_v2(multi_quat_diff(cur_bquat.flatten(), tgt_bquat.flatten())).mean()
    # pose_pos_diff = np.linalg.norm(cur_wbpos - tgt_wbpos, axis = 1).mean()

    # p_reward = math.exp(-k_p * (pose_quat_diff ** 2))
    # jp_reward = math.exp(-k_jp * (pose_pos_diff ** 2))


    # # Comparing with GT
    # gt_bquat = env.ar_context['bquat'][ind].flatten()
    # gt_prev_bquat = env.ar_context['bquat'][ind - 1].flatten()
    # prev_bquat = env.prev_bquat

    # pose_gt_diff = multi_quat_norm_v2(multi_quat_diff(gt_bquat, cur_bquat)).mean()
    
    # cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # tgt_bangvel = get_angvel_fd(gt_prev_bquat, gt_bquat, env.dt)
    # vel_dist = np.linalg.norm(cur_bangvel - tgt_bangvel, ord=v_ord)
    # act_v_reward = math.exp(-k_act_v * (vel_dist ** 2))

    
    # # rq_dist = multi_quat_norm_v2(multi_quat_diff(tgt_qpos[3:7], act_qpos[3:7])).mean()
    
    # # rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    # gt_p_reward = math.exp(-k_act_p * pose_gt_diff)

    # reward =  w_p * p_reward + w_jp * jp_reward + w_act_p * gt_p_reward + w_act_v * act_v_reward
    agent_qpos = env.data.qpos.copy()
    target_pos = np.array([10, 10, 0.9])
    rp_dist = np.linalg.norm(agent_qpos[:3] - target_pos)
    rp_reward = math.exp(-k_rp * (rp_dist ** 2))
    reward = rp_reward

    # if flags.debug:
    #     import pdb; pdb.set_trace()
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(reward, np.array([p_reward, jp_reward, gt_p_reward, act_v_reward]))
    
    return reward, np.array([rp_reward])


reward_func = {
               "dynamic_supervision_v1": dynamic_supervision_v1, 
               "dynamic_supervision_v2": dynamic_supervision_v2, 
               "dynamic_supervision_v3": dynamic_supervision_v3,
               "dynamic_supervision_v4": dynamic_supervision_v4,
               "dynamic_supervision_v5": dynamic_supervision_v5,
               "dynamic_supervision_v6": dynamic_supervision_v6,
               "forward_v1": forward_v1
               }