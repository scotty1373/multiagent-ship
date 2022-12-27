# -*-  coding=utf-8 -*-
# @Time : 2022/8/23 15:53
# @Author : Scotty1373
# @File : multi_agnet_model.py
# @Software : PyCharm
import torch
from torch import nn
import numpy as np
from utils_tools.utils import uniform_init, orthogonal_init

class ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_state = nn.Sequential(
            nn.Linear(self.state_dim-5*4, 300),
            nn.ReLU(inplace=True)
        )
        self.fc_state2 = nn.Sequential(
            nn.Linear(self.state_dim-24*2*4, 100),
            nn.ReLU(inplace=True)
        )
        self.mean_fc1 = nn.Sequential(
            uniform_init(nn.Linear(400, 300), a=-3e-3, b=3e-3),
            nn.ReLU(inplace=True))
        self.mean_fc2 = uniform_init(nn.Linear(300, self.action_dim), a=-3e-3, b=3e-3)
        self.mean_fc2act = nn.Tanh()

    def forward(self, state_vect):
        b = state_vect.shape[:1].numel()
        state_vect = state_vect.reshape(b, 4, -1)
        state_pos = state_vect[:, :, :5].reshape(b, -1)
        state_ray = state_vect[:, :, 5:].reshape(b, -1)
        ray_feature = self.fc_state(state_ray)
        pos_feature = self.fc_state2(state_pos)
        common = torch.cat((ray_feature, pos_feature), dim=-1)
        action_mean = self.mean_fc1(common)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc2act(action_mean)
        return action_mean


class ActionCriticModel(nn.Module):
    def __init__(self, state_dim, agent_num, action_dim):
        super(ActionCriticModel, self).__init__()
        self.state_dim = agent_num * state_dim
        self.action_dim = agent_num * action_dim
        self.fc_state = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim, 400),
            nn.ReLU(inplace=True)
        )
        # self.fc_action = nn.Sequential(
        #     nn.Linear(self.action_dim, 200),
        #     nn.ReLU(inplace=True)
        # )
        # q1 network
        self.fc1_q1 = nn.Sequential(
            orthogonal_init(nn.Linear(400, 300), gain=np.sqrt(2)),
            nn.ReLU(inplace=True))
        self.fc2_q1 = nn.Sequential(
            orthogonal_init(nn.Linear(300, 64), gain=0.01),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(64, 1), gain=0.01))

        # q2 network
        self.fc1_q2 = nn.Sequential(
            orthogonal_init(nn.Linear(400, 300), gain=np.sqrt(2)),
            nn.ReLU(inplace=True))
        self.fc2_q2 = nn.Sequential(
            orthogonal_init(nn.Linear(300, 64), gain=0.01),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(64, 1), gain=0.01))

    def forward(self, state_vect, action):
        fusion_vect = torch.cat((state_vect, action), dim=-1)
        fusion_state = self.fc_state(fusion_vect)
        # action_vect = self.fc_action(action)
        # fusion_common = torch.cat((fusion_state, action_vect), dim=-1)

        # q1 network
        q1_critic = self.fc1_q1(fusion_state)
        q1_critic = self.fc2_q1(q1_critic)

        # q2 network
        q2_critic = self.fc1_q2(fusion_state)
        q2_critic = self.fc2_q2(q2_critic)
        return q1_critic, q2_critic

    def q_theta1(self, state_vect, action):
        fusion_vect = torch.cat((state_vect, action), dim=-1)
        fusion_state = self.fc_state(fusion_vect)
        # action_vect = self.fc_action(action)
        # fusion_common = torch.cat((fusion_state, action_vect), dim=-1)

        # q1 network
        q1_critic = self.fc1_q1(fusion_state)
        q1_critic = self.fc2_q1(q1_critic)
        return q1_critic


if __name__ == '__main__':
    agent = 2
    state_len = 3
    act_dim = 1
    actor = ActorModel(state_len, act_dim)
    critic = ActionCriticModel(state_len, agent, act_dim)

    x1 = torch.randn((10, state_len))
    x2 = torch.randn((10, state_len))
    cat_state = torch.cat((x1, x2), dim=-1)
    y = torch.randn((10, agent*act_dim))

    action = actor(x1)
    value1, value2 = critic(cat_state, y)

    value = critic.q_theta1(cat_state, y)