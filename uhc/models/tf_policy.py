import torch.nn as nn
from khrylib.rl.core.distributions import DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.rl.core.running_norm import RunningNorm
from khrylib.models.mlp import MLP
from khrylib.utils.math import *
from design_opt.models.utils import *
from design_opt.models.pos_enc import PositionalEncoding


class PolicyTransformer(Policy):
    def __init__(self, cfg, agent):
        super().__init__()
        self.type = 'gaussian'
        self.cfg = cfg
        self.agent = agent
        self.attr_obs_dim = agent.attr_obs_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.design_obs_dim = agent.design_obs_dim
        self.control_state_dim = self.attr_obs_dim + self.sim_obs_dim + self.design_obs_dim
        self.design_state_dim = self.attr_obs_dim + self.design_obs_dim
        self.control_action_dim = agent.control_action_dim
        self.model_action_dim = agent.design_obs_dim
        self.action_dim = self.control_action_dim + self.model_action_dim

        """ control """
        self.control_norm = RunningNorm(self.control_state_dim) if cfg.get('control_norm', True) else None
        cur_dim = self.control_state_dim
        # transformer
        tf_cfg = cfg['control_transformer']
        self.control_pos_enc = PositionalEncoding(tf_cfg['model_dim'], cur_dim, enc_type='original')
        tf_layers = nn.TransformerEncoderLayer(tf_cfg['model_dim'], tf_cfg['nhead'], tf_cfg['ff_dim'], tf_cfg['dropout'])
        self.control_tf = nn.TransformerEncoder(tf_layers, tf_cfg['nlayer'])
        cur_dim = tf_cfg['model_dim']
        # dist
        self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
        self.control_action_log_std = nn.Parameter(torch.ones(1, self.control_action_dim) * cfg['control_log_std'], requires_grad=not cfg['fix_control_std'])
        init_fc_weights(self.control_action_mean)

        """ design """
        self.model_norm = RunningNorm(self.design_state_dim) if cfg.get('model_norm', True) else None
        cur_dim = self.design_state_dim
        # transformer
        tf_cfg = cfg['model_transformer']
        self.model_pos_enc = PositionalEncoding(tf_cfg['model_dim'], cur_dim, enc_type='original')
        tf_layers = nn.TransformerEncoderLayer(tf_cfg['model_dim'], tf_cfg['nhead'], tf_cfg['ff_dim'], tf_cfg['dropout'])
        self.model_tf = nn.TransformerEncoder(tf_layers, tf_cfg['nlayer'])
        cur_dim = tf_cfg['model_dim']
        # dist
        self.model_action_mean = nn.Linear(cur_dim, self.model_action_dim)
        self.model_action_log_std = nn.Parameter(torch.ones(1, self.model_action_dim) * cfg['model_log_std'], requires_grad=not cfg['fix_model_std'])
        init_fc_weights(self.model_action_mean)

    def batch_data(self, x):
        obs, edges, use_design_action, num_nodes = zip(*x)
        obs = torch.cat(obs)
        use_design_action = np.concatenate(use_design_action)
        num_nodes = np.concatenate(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        num_nodes_cum = np.cumsum(num_nodes)
        return obs, edges_new, use_design_action, num_nodes, num_nodes_cum

    def forward(self, x):
        # attr_obs, sim_obs = torch.split(obs, [self.attr_obs_dim, self.sim_obs_dim])
        control_x, design_x = [], []
        node_design_mask = []
        design_mask = []
        total_num_nodes = 0
        for i, x_i in enumerate(x):
            num = x_i[-1].item()
            is_design = x_i[-2].item() == 1
            (control_x, design_x)[is_design].append(x_i)
            node_design_mask += [is_design] * num
            design_mask.append(is_design)
            total_num_nodes += num
        node_design_mask = torch.BoolTensor(node_design_mask)
        design_mask = torch.BoolTensor(design_mask)
        # control
        if len(control_x) > 0:
            obs, edges, use_design_action, num_nodes, num_nodes_cum_control = self.batch_data(control_x)
            x = obs
            if self.control_norm is not None:
                x = self.control_norm(x)
            
            n = int(num_nodes.mean())
            x = x.view(-1, n, x.shape[-1]).transpose(0, 1).contiguous()
            x = self.control_pos_enc(x)
            x = self.control_tf(x)
            x = x.transpose(0, 1).reshape(-1, x.shape[-1])

            control_action_mean = self.control_action_mean(x)
            control_action_std = self.control_action_log_std.expand_as(control_action_mean).exp()
            control_dist = DiagGaussian(control_action_mean, control_action_std)
        else:
            num_nodes_cum_control = None
            control_dist = None
            
        # design
        if len(design_x) > 0:
            obs, edges, use_design_action, num_nodes, num_nodes_cum_design = self.batch_data(design_x)
            obs = torch.cat((obs[:, :self.attr_obs_dim], obs[:, -self.design_obs_dim:]), dim=-1)
            x = obs
            if self.model_norm is not None:
                x = self.model_norm(x)
            
            n = int(num_nodes.mean())
            x = x.view(-1, n, x.shape[-1]).transpose(0, 1).contiguous()
            x = self.model_pos_enc(x)
            x = self.model_tf(x)
            x = x.transpose(0, 1).reshape(-1, x.shape[-1])

            model_action_mean = self.model_action_mean(x)
            model_action_std = self.model_action_log_std.expand_as(model_action_mean).exp()
            model_dist = DiagGaussian(model_action_mean, model_action_std)
        else:
            num_nodes_cum_design = None
            model_dist = None
        return control_dist, model_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, x[0][0].device

    def select_action(self, x, mean_action=False):
        
        control_dist, model_dist, node_design_mask, _, total_num_nodes, _, _, device = self.forward(x)
        if control_dist is not None:
            control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
        else:
            control_action = None

        if model_dist is not None:
            model_action = model_dist.mean_sample() if mean_action else model_dist.sample()
        else:
            model_action = None

        action = torch.zeros(total_num_nodes, self.action_dim).to(device)
        if control_action is not None:
            action[~node_design_mask, :self.control_action_dim] = control_action
        if model_action is not None:
            action[node_design_mask, self.control_action_dim:] = model_action
        return action

    def get_log_prob(self, x, action):
        action = torch.cat(action)
        control_dist, model_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, device = self.forward(x)
        action_log_prob = torch.zeros(design_mask.shape[0], 1).to(device)
        # control log prob
        if control_dist is not None:
            control_action = action[~node_design_mask, :self.control_action_dim]
            control_action_log_prob_nodes = control_dist.log_prob(control_action)
            control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
            control_action_log_prob_cum = control_action_log_prob_cum[torch.LongTensor(num_nodes_cum_control) - 1]
            control_action_log_prob = torch.cat([control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
            action_log_prob[~design_mask] = control_action_log_prob
        # model log prob
        if model_dist is not None:
            model_action = action[node_design_mask, self.control_action_dim:]
            model_action_log_prob_nodes = model_dist.log_prob(model_action)
            model_action_log_prob_cum = torch.cumsum(model_action_log_prob_nodes, dim=0)
            model_action_log_prob_cum = model_action_log_prob_cum[torch.LongTensor(num_nodes_cum_design) - 1]
            model_action_log_prob = torch.cat([model_action_log_prob_cum[[0]], model_action_log_prob_cum[1:] - model_action_log_prob_cum[:-1]])
            action_log_prob[design_mask] = model_action_log_prob
        return action_log_prob


