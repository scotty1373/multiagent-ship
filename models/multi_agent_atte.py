# -*- coding=utf-8 -*-
# @Time : 2022/8/23 15:53
# @Author : Scotty1373
# @File : multi_agent_atte.py
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
            nn.Linear(self.state_dim, 400),
            nn.ReLU(inplace=True)
        )
        self.mean_fc1 = nn.Sequential(
            uniform_init(nn.Linear(400, 300), a=-3e-3, b=3e-3),
            nn.ReLU(inplace=True))
        self.mean_fc2 = uniform_init(nn.Linear(300, self.action_dim), a=-3e-3, b=3e-3)
        self.mean_fc2act = nn.Tanh()

    def forward(self, state_vect):
        action_mean = self.fc_state(state_vect)
        action_mean = self.mean_fc1(action_mean)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc2act(action_mean)
        return action_mean


class ActionCriticModel(nn.Module):
    def __init__(self, state_dim, agent_num, action_dim):
        super(ActionCriticModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num

        self.fc_head = nn.Linear(state_dim+action_dim, 64)

        self.trans_block1 = TransBlock(64, 4)
        self.trans_block2 = TransBlock(64, 4)

        # q1 network
        self.fc1_q1 = nn.Sequential(
            orthogonal_init(nn.Linear(64*agent_num, 300), gain=np.sqrt(2)),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(300, 1), gain=0.01))

        # q2 network
        self.fc1_q2 = nn.Sequential(
            orthogonal_init(nn.Linear(64*agent_num, 300), gain=np.sqrt(2)),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(300, 1), gain=0.01))

    def forward(self, state_vect, act):
        b = state_vect.shape[:1].numel()
        state_vect = state_vect.reshape(b, self.agent_num, -1)
        act = act.reshape(b, self.agent_num, -1)

        common_vect = torch.cat([state_vect, act], dim=-1)
        assert common_vect.shape[:2] == torch.Size([b, self.agent_num])

        out = self.trans_block1(nn.functional.relu(self.fc_head(common_vect)))
        out = self.trans_block2(out)

        # q1 network
        q1_critic = self.fc1_q1(out.view(b, -1).contiguous())

        # q2 network
        q2_critic = self.fc1_q2(out.view(b, -1).contiguous())
        return q1_critic, q2_critic

    def q_theta1(self, state_vect, act):
        b = state_vect.shape[:1].numel()
        state_vect = state_vect.reshape(b, self.agent_num, -1)
        act = act.reshape(b, self.agent_num, -1)

        common_vect = torch.cat([state_vect, act], dim=-1)
        assert common_vect.shape[:2] == torch.Size([b, self.agent_num])

        out = self.trans_block1(nn.functional.relu(self.fc_head(common_vect)))
        out = self.trans_block2(out)

        # q1 network
        q1_critic = self.fc1_q1(out.view(b, -1).contiguous())
        return q1_critic


class TransBlock(nn.Module):
    def __init__(self, k, h):
        super(TransBlock, self).__init__()
        self.self_attention = ScaledDotProductAttention(k, k, k, h)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.fc = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k, k)
        )

    def forward(self, x):
        out = self.self_attention(x)
        out = self.norm1(out + x)
        out = self.fc(out)
        return self.norm2(out)


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, d_k, d_v, h):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(.1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        """
        Computes
        :param x: input qkv (b_s, nq, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        """
        queries = keys = values = x
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


# noinspection DuplicatedCode
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
