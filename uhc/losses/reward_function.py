from uhc.khrylib.utils import *
from uhc.utils.flags import flags
from uhc.utils.transformation import (
    quaternion_from_euler_batch,
    quaternion_multiply_batch,
    quat_mul_vec,
    quat_mul_vec_batch,
    quaternion_from_euler,
    quaternion_inverse_batch,
)

def world_rfc_implicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
    )
    k_p, k_v, k_e, k_c, k_vf = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = (
        env.cur_t
    )  # current time step (this is after the step function is called), so we are using the target motion for (t+1)
    ind = env.get_expert_index(t)

    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)

    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    cur_com = env.get_com()
    e_com = env.get_expert_attr("com", ind).copy()

    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert = env.expert
    
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward

    cur_bangvel = (cur_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()
    e_bangvel = (e_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()

    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    if cfg.residual_force > 0.0:
        vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    # overall reward
    reward = (
        w_p * pose_reward
        + w_v * vel_reward
        + w_e * ee_reward
        + w_c * com_reward
        + w_vf * vf_reward
    )
    reward /= w_p + w_v + w_e + w_c + w_vf
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])



def world_rfc_implicit_reward_quat(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
    )
    k_p, k_v, k_e, k_c, k_vf = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = (
        env.cur_t
    )  # current time step (this is after the step function is called), so we are using the target motion for (t+1)
    ind = env.get_expert_index(t)

    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)

    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    cur_com = env.get_com()
    e_com = env.get_expert_attr("com", ind).copy()

    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert = env.expert
    
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward

    cur_bangvel = (cur_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()
    e_bangvel = (e_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()

    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    if cfg.residual_force > 0.0:
        vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    # overall reward
    reward = (
        w_p * pose_reward
        + w_v * vel_reward
        + w_e * ee_reward
        + w_c * com_reward
        + w_vf * vf_reward
    )
    reward /= w_p + w_v + w_e + w_c + w_vf
    # if flags.debug:
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward]))
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])


def world_rfc_implicit_v1_mul(env, state, action, info):
   # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights

    w_p, w_v, w_e, w_c, w_vf = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
    ) 

    k_p, k_v, k_e, k_c, k_vf = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = (
        env.cur_t
    )  # current time step (this is after the step function is called), so we are using the target motion for (t+1)
    ind = env.get_expert_index(t)

    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)

    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    cur_com = env.get_com()
    e_com = env.get_expert_attr("com", ind).copy()

    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert = env.expert

    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward

    cur_bangvel = (cur_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()
    e_bangvel = (e_bangvel.reshape(-1, 3) * env.jpos_diffw).flatten()

    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    if cfg.residual_force > 0.0:
        vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0

    # overall reward
    reward = (
         pose_reward * vel_reward * ee_reward * com_reward * (vf_reward if w_vf != 0.0 else 1.0)
    )
    # if flags.debug:
    #     np.set_printoptions(precision=4, suppress=1)
    #     print(np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward]))
    return reward, np.array([pose_reward, vel_reward, ee_reward, com_reward, vf_reward])


def world_rfc_explicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf, w_cp = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
        ws.get("w_cp", 0.0),
    )
    k_p, k_v, k_e, k_c, k_vf, k_cp = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
        ws.get("k_cp", 1),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    e_com = env.get_expert_attr("com", ind).copy()
    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert = env.expert
    if expert["meta"]["cyclic"]:
        init_pos = expert["init_pos"]
        cycle_h = expert["cycle_relheading"]
        cycle_pos = expert["cycle_pos"]
        orig_rpos = e_rpos.copy()
        e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
        e_com = quat_mul_vec(cycle_h, e_com - orig_rpos) + e_rpos
        for i in range(e_ee.shape[0] // 3):
            e_ee[3 * i : 3 * i + 3] = (
                quat_mul_vec(cycle_h, e_ee[3 * i : 3 * i + 3] - orig_rpos) + e_rpos
            )

    if not expert["meta"]["cyclic"] and env.start_ind + t >= expert["len"]:
        e_bangvel = np.zeros_like(e_bangvel)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
    vf_loss = 0
    # cp_loss = 0
    for i, body in enumerate(env.vf_bodies):
        contact_point = vf[i * env.body_vf_dim : i * env.body_vf_dim + 3]
        force = vf[i * env.body_vf_dim + 3 : (i + 1) * env.body_vf_dim]
        vf_loss += np.linalg.norm(force) ** 2
        # cp_loss += np.linalg.norm(contact_point) ** 2
    vf_reward = math.exp(-k_vf * vf_loss)
    # cp_reward = math.exp(-k_cp * cp_loss)
    # overall reward
    reward = (
        w_p * pose_reward
        + w_v * vel_reward
        + w_e * ee_reward
        + w_c * com_reward
        + w_vf * vf_reward
    )
    reward /= w_p + w_v + w_e + w_c + w_vf 
    return reward, np.array(
        [pose_reward, vel_reward, ee_reward, com_reward, vf_reward]
    )




def world_rfc_explicit_mul_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_c, w_vf, w_cp = (
        ws.get("w_p", 0.6),
        ws.get("w_v", 0.1),
        ws.get("w_e", 0.2),
        ws.get("w_c", 0.1),
        ws.get("w_vf", 0.0),
        ws.get("w_cp", 0.0),
    )
    k_p, k_v, k_e, k_c, k_vf, k_cp = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_c", 1000),
        ws.get("k_vf", 1),
        ws.get("k_cp", 1),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    # learner
    cur_ee = env.get_ee_pos(None)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_com = env.get_com()
    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_ee = env.get_expert_attr("ee_wpos", ind).copy()
    e_com = env.get_expert_attr("com", ind).copy()
    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    expert = env.expert
    if expert["meta"]["cyclic"]:
        init_pos = expert["init_pos"]
        cycle_h = expert["cycle_relheading"]
        cycle_pos = expert["cycle_pos"]
        orig_rpos = e_rpos.copy()
        e_rpos = quat_mul_vec(cycle_h, e_rpos - init_pos) + cycle_pos
        e_com = quat_mul_vec(cycle_h, e_com - orig_rpos) + e_rpos
        for i in range(e_ee.shape[0] // 3):
            e_ee[3 * i : 3 * i + 3] = (
                quat_mul_vec(cycle_h, e_ee[3 * i : 3 * i + 3] - orig_rpos) + e_rpos
            )

    if not expert["meta"]["cyclic"] and env.start_ind + t >= expert["len"]:
        e_bangvel = np.zeros_like(e_bangvel)
    # pose reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff[1:] *= env.body_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel - e_bangvel, ord=v_ord)
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # com reward
    com_dist = np.linalg.norm(cur_com - e_com)
    com_reward = math.exp(-k_c * (com_dist ** 2))
    # residual force reward
    vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
    vf_loss = 0
    # cp_loss = 0
    for i, body in enumerate(env.vf_bodies):
        contact_point = vf[i * env.body_vf_dim : i * env.body_vf_dim + 3]
        force = vf[i * env.body_vf_dim + 3 : (i + 1) * env.body_vf_dim]
        vf_loss += np.linalg.norm(force) ** 2
        # cp_loss += np.linalg.norm(contact_point) ** 2
    vf_reward = math.exp(-k_vf * vf_loss)
    # cp_reward = math.exp(-k_cp * cp_loss)
    # overall reward
    reward = (
         pose_reward * vel_reward * ee_reward * com_reward * vf_reward
    )
    return reward, np.array(
        [pose_reward, vel_reward, ee_reward, com_reward, vf_reward]
    )



def local_rfc_implicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv, w_vf = (
        ws.get("w_p", 0.5),
        ws.get("w_v", 0.0),
        ws.get("w_e", 0.2),
        ws.get("w_rp", 0.1),
        ws.get("w_rv", 0.1),
        ws.get("w_vf", 0.1),
    )
    k_p, k_v, k_e, k_vf = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_vf", 1),
    )
    k_rh, k_rq, k_rl, k_ra = (
        ws.get("k_rh", 300),
        ws.get("k_rq", 300),
        ws.get("k_rl", 5.0),
        ws.get("k_ra", 0.5),
    )
    v_ord = ws.get("v_ord", 2)

    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd_new(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rlinv_local = env.get_expert_attr("rlinv_local", ind)
    e_rangv = env.get_expert_attr("rangv", ind)
    e_rq_rmh = env.get_expert_attr("rq_rmh", ind)
    e_ee = env.get_expert_attr("ee_pos", ind)
    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    # pose reward
    pose_diff = multi_quat_norm(
        multi_quat_diff(cur_bquat[4:], e_bquat[4:])
    )  # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(
        -k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2)
    )
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(
        -k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2)
    )
    # residual force reward
    if w_vf > 0.0:
        vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
        vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))
    else:
        vf_reward = 0.0
    # overall reward
    reward = (
        w_p * pose_reward
        + w_v * vel_reward
        + w_e * ee_reward
        + w_rp * root_pose_reward
        + w_rv * root_vel_reward
        + w_vf * vf_reward
    )
    reward /= w_p + w_v + w_e + w_rp + w_rv + w_vf
    return reward, np.array(
        [
            pose_reward,
            vel_reward,
            ee_reward,
            root_pose_reward,
            root_vel_reward,
            vf_reward,
        ]
    )


def local_rfc_explicit_reward(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_v, w_e, w_rp, w_rv, w_vf, w_cp = (
        ws.get("w_p", 0.4),
        ws.get("w_v", 0.0),
        ws.get("w_e", 0.2),
        ws.get("w_rp", 0.1),
        ws.get("w_rv", 0.1),
        ws.get("w_vf", 0.1),
        ws.get("w_cp", 0.1),
    )
    k_p, k_v, k_e, k_vf, k_cp = (
        ws.get("k_p", 2),
        ws.get("k_v", 0.005),
        ws.get("k_e", 20),
        ws.get("k_vf", 20),
        ws.get("k_cp", 10),
    )
    k_rh, k_rq, k_rl, k_ra = (
        ws.get("k_rh", 300),
        ws.get("k_rq", 300),
        ws.get("k_rl", 5.0),
        ws.get("k_ra", 0.5),
    )
    v_ord = ws.get("v_ord", 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat
    prev_qpos = env.prev_qpos
    # learner
    cur_qpos = env.data.qpos.copy()
    cur_qvel = get_qvel_fd_new(prev_qpos, cur_qpos, env.dt, cfg.obs_coord)
    cur_rlinv_local = cur_qvel[:3]
    cur_rangv = cur_qvel[3:6]
    cur_rq_rmh = de_heading(cur_qpos[3:7])
    cur_ee = env.get_ee_pos(cfg.obs_coord)
    cur_bquat = env.get_body_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    # expert
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rlinv_local = env.get_expert_attr("rlinv_local", ind)
    e_rangv = env.get_expert_attr("rangv", ind)
    e_rq_rmh = env.get_expert_attr("rq_rmh", ind)
    e_ee = env.get_expert_attr("ee_pos", ind)
    e_bquat = env.get_expert_attr("bquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    # pose reward
    pose_diff = multi_quat_norm(
        multi_quat_diff(cur_bquat[4:], e_bquat[4:])
    )  # ignore root
    pose_diff *= cfg.b_diffw
    pose_dist = np.linalg.norm(pose_diff)
    pose_reward = math.exp(-k_p * (pose_dist ** 2))
    # velocity reward
    vel_dist = np.linalg.norm(cur_bangvel[3:] - e_bangvel[3:], ord=v_ord)  # ignore root
    vel_reward = math.exp(-k_v * (vel_dist ** 2))
    # ee reward
    ee_dist = np.linalg.norm(cur_ee - e_ee)
    ee_reward = math.exp(-k_e * (ee_dist ** 2))
    # root position reward
    root_height_dist = cur_qpos[2] - e_qpos[2]
    root_quat_dist = multi_quat_norm(multi_quat_diff(cur_rq_rmh, e_rq_rmh))[0]
    root_pose_reward = math.exp(
        -k_rh * (root_height_dist ** 2) - k_rq * (root_quat_dist ** 2)
    )
    # root velocity reward
    root_linv_dist = np.linalg.norm(cur_rlinv_local - e_rlinv_local)
    root_angv_dist = np.linalg.norm(cur_rangv - e_rangv)
    root_vel_reward = math.exp(
        -k_rl * (root_linv_dist ** 2) - k_ra * (root_angv_dist ** 2)
    )
    # residual force reward
    vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
    vf_loss = 0
    cp_loss = 0
    for i, body in enumerate(env.vf_bodies):
        contact_point = vf[i * env.body_vf_dim : i * env.body_vf_dim + 3]
        force = vf[i * env.body_vf_dim + 3 : (i + 1) * env.body_vf_dim]
        vf_loss += np.linalg.norm(force) ** 2
        cp_loss += np.linalg.norm(contact_point) ** 2
    vf_reward = math.exp(-k_vf * vf_loss)
    cp_reward = math.exp(-k_cp * cp_loss)
    # overall reward
    reward = (
        w_p * pose_reward
        + w_v * vel_reward
        + w_e * ee_reward
        + w_rp * root_pose_reward
        + w_rv * root_vel_reward
        + w_vf * vf_reward
        + w_cp * cp_reward
    )
    reward /= w_p + w_v + w_e + w_rp + w_rv + w_vf + w_cp
    return reward, np.array(
        [
            pose_reward,
            vel_reward,
            ee_reward,
            root_pose_reward,
            root_vel_reward,
            vf_reward,
            cp_reward,
        ]
    )


def world_rfc_implicit_v2(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    k_p, k_wp, k_v, k_j, k_c, k_vf, jpos_diffw = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_c", 100),
        ws.get("k_vf", 1),
        ws.get("jpos_diffw", [1] * 24),
    )
    jpos_diffw = np.array(jpos_diffw)

    v_ord = ws.get("v_ord", 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat

    # Current policy states
    cur_bquat = env.get_body_quat()
    cur_wbquat = env.get_wbody_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_body_com = env.get_body_com().reshape(-1, 3)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_bquat = env.get_expert_attr("bquat", ind)
    e_wbquat = env.get_expert_attr("wbquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    e_wbpos = env.get_expert_attr("wbpos", ind).reshape(-1, 3)
    e_body_com = env.get_expert_attr("body_com", ind).reshape(-1, 3)
    expert = env.expert

    # body quat reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff *= jpos_diffw
    pose_diff = (pose_diff ** 2).mean()
    pose_reward = math.exp(-k_p * pose_diff)

    # global quat reward
    wpose_diff = multi_quat_norm(multi_quat_diff(cur_wbquat, e_wbquat))
    wpose_diff *= jpos_diffw
    wpose_diff = (wpose_diff ** 2).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_bangvel - e_bangvel
    vel_dist = (vel_dist ** 2).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body com reward
    diff = e_body_com - cur_body_com
    diff *= jpos_diffw[:, None]
    bcom_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    com_reward = math.exp(-k_c * bcom_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    diff *= jpos_diffw[:, None]
    jpos_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # residual force reward
    vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
    vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))

    # overall reward
    reward = (
        pose_reward * wpose_reward * com_reward * jpos_reward * vel_reward * vf_reward
    )
    # np.set_printoptions(precision=5, suppress=1)
    # print(np.array([reward, vel_dist, pose_diff, wpose_diff]), \
    # np.array([pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]))
    return reward, np.array(
        [pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]
    )


def world_rfc_implicit_v3(env, state, action, info):
    # reward coefficients
    cfg = env.cc_cfg
    ws = cfg.reward_weights
    w_p, w_wp, w_v, w_j, w_c, w_vf = (
        ws.get("w_p", 0.4),
        ws.get("w_wp", 0.4),
        ws.get("w_v", 0.005),
        ws.get("w_j", 100),
        ws.get("w_c", 100),
        ws.get("w_vf", 1),
    )

    k_p, k_wp, k_v, k_j, k_c, k_vf, jpos_diffw = (
        ws.get("k_p", 0.4),
        ws.get("k_wp", 0.4),
        ws.get("k_v", 0.005),
        ws.get("k_j", 100),
        ws.get("k_c", 100),
        ws.get("k_vf", 1),
        ws.get("jpos_diffw", [1] * 24),
    )
    jpos_diffw = np.array(jpos_diffw)

    v_ord = ws.get("v_ord", 2)
    # data from env
    t = env.cur_t
    ind = env.get_expert_index(t)
    prev_bquat = env.prev_bquat

    # Current policy states
    cur_bquat = env.get_body_quat()
    cur_wbquat = env.get_wbody_quat()
    cur_bangvel = get_angvel_fd(prev_bquat, cur_bquat, env.dt)
    cur_body_com = env.get_body_com().reshape(-1, 3)
    cur_wbpos = env.get_wbody_pos().reshape(-1, 3)

    # Expert States
    e_qpos = env.get_expert_attr("qpos", ind)
    e_rpos = e_qpos[:3]
    e_bquat = env.get_expert_attr("bquat", ind)
    e_wbquat = env.get_expert_attr("wbquat", ind)
    e_bangvel = env.get_expert_attr("bangvel", ind)
    e_wbpos = env.get_expert_attr("wbpos", ind).reshape(-1, 3)
    e_body_com = env.get_expert_attr("body_com", ind).reshape(-1, 3)
    expert = env.expert

    # body quat reward
    pose_diff = multi_quat_norm(multi_quat_diff(cur_bquat, e_bquat))
    pose_diff *= jpos_diffw
    pose_diff = (pose_diff ** 2).mean()
    pose_reward = math.exp(-k_p * pose_diff)

    # global quat reward
    wpose_diff = multi_quat_norm(multi_quat_diff(cur_wbquat, e_wbquat))
    wpose_diff *= jpos_diffw
    wpose_diff = (wpose_diff ** 2).mean()
    wpose_reward = math.exp(-k_wp * wpose_diff)

    # velocity reward
    vel_dist = cur_bangvel - e_bangvel
    vel_dist = (vel_dist ** 2).mean()
    vel_reward = math.exp(-k_v * vel_dist)

    # body com reward
    diff = e_body_com - cur_body_com
    diff *= jpos_diffw[:, None]
    bcom_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    com_reward = math.exp(-k_c * bcom_dist)

    # body joint reward
    diff = cur_wbpos - e_wbpos
    diff *= jpos_diffw[:, None]
    jpos_dist = np.power(np.linalg.norm(diff, axis=1), 2).mean()
    jpos_reward = math.exp(-k_j * jpos_dist)

    # residual force reward
    vf = action[(env.ndof) : (env.ndof + env.vf_dim)]
    vf_reward = math.exp(-k_vf * (np.linalg.norm(vf) ** 2))

    # overall reward
    reward = (
        w_p * pose_reward
        + w_wp * wpose_reward
        + w_c * com_reward
        + w_j * jpos_reward
        + w_v * vel_reward
        + w_vf * vf_reward
    )
    # np.set_printoptions(precision=5, suppress=1)
    # print(np.array([reward, vel_dist, pose_diff, wpose_diff]), \
    # np.array([pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]))
    return reward, np.array(
        [pose_reward, wpose_reward, com_reward, jpos_reward, vel_reward, vf_reward]
    )


reward_func = {
    "local_rfc_implicit": local_rfc_implicit_reward,
    "local_rfc_explicit": local_rfc_explicit_reward,
    "world_rfc_implicit": world_rfc_implicit_reward,
    "world_rfc_implicit_quat": world_rfc_implicit_reward_quat,
    "world_rfc_implicit_v1_mul": world_rfc_implicit_v1_mul,
    "world_rfc_explicit": world_rfc_explicit_reward,
    "world_rfc_explicit_mul": world_rfc_explicit_mul_reward,
    "world_rfc_implicit_v2": world_rfc_implicit_v2,
    "world_rfc_implicit_v3": world_rfc_implicit_v3,
}