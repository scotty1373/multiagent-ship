# -*-  coding=utf-8 -*-
# @Time : 2022/8/22 19:43
# @Author : Scotty1373
# @File : agent.py
# @Software : PyCharm
import numpy as np
import torch
import random
from collections import deque
from copy import deepcopy
from torch.distributions import Normal
from itertools import chain
from models.multi_agnet_model import ActorModel, ActionCriticModel
from utils_tools.utils import RunningMeanStd, cut_requires_grad

DISTRIBUTION_INDEX = [0, 0.3]


class Agent:
    def __init__(self, state_dim, action_dim, agent_num, device, agent_idx, train=True, logger=None):
        self.action_space = np.array((-1, 1))
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.device = device
        self.agent_idx = agent_idx
        self.train = train
        self.logger = logger

        # 初始化actor + 双q网络
        self._init()
        self.tua = 0.01
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.discount_index = 0.98
        self.smooth_regular = 0.1
        self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1])
        self.target_model_regular_noise = Normal(0, 0.1)

        # optimizer init
        self.actor_opt = torch.optim.Adam(params=self.actor_model.parameters(), lr=self.lr_actor)
        self.critic_uni_opt = torch.optim.Adam(params=self.critic_model.parameters(), lr=self.lr_critic)

        # loss init
        self.critic_loss = torch.nn.MSELoss()

        # loss history
        self.actor_loss_history = 0.0
        self.critic_loss_history = 0.0

    def _init(self):
        self.actor_model = ActorModel(self.state_dim, self.action_dim).to(self.device)
        self.critic_model = ActionCriticModel(self.state_dim, self.agent_num, self.action_dim).to(self.device)
        self.actor_target = ActorModel(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = ActionCriticModel(self.state_dim, self.agent_num, self.action_dim).to(self.device)
        # model first hard update
        self.model_hard_update(self.actor_model, self.actor_target)
        self.model_hard_update(self.critic_model, self.critic_target)
        # target model requires_grad设为False
        cut_requires_grad(self.actor_target.parameters())
        cut_requires_grad(self.critic_target.parameters())

    def get_action(self, vect_state, explore=True, noise_clip=0.2):
        vect = torch.FloatTensor(vect_state).to(self.device)
        logits = self.actor_model(vect)
        if self.train and explore:
            noise = self.noise.sample()
            # acc 动作裁剪
            logits += noise.clamp(min=-noise_clip, max=noise_clip)
            logits = logits.clamp(min=self.action_space.min(), max=self.action_space.max())
        else:
            logits = logits.clamp(min=self.action_space.min(), max=self.action_space.max())
        return logits.detach().cpu().numpy()

    def save_model(self, file_name):
        checkpoint = {'actor': self.actor_model.state_dict(),
                      'critic': self.critic_model.state_dict(),
                      'opt_actor': self.actor_opt.state_dict(),
                      'opt_critic': self.critic_uni_opt.state_dict()}
        torch.save(checkpoint, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.actor_model.load_state_dict(checkpoint['actor'])
        self.critic_model.load_state_dict(checkpoint['critic'])
        self.actor_opt.load_state_dict(checkpoint['opt_actor'])
        self.critic_uni_opt.load_state_dict(checkpoint['opt_critic'])

    def model_hard_update(self, current, target):
        weight_model = deepcopy(current.state_dict())
        target.load_state_dict(weight_model)

    def model_soft_update(self, current, target):
        for target_param, source_param in zip(target.parameters(),
                                              current.parameters()):
            target_param.data.copy_((1 - self.tua) * target_param + self.tua * source_param)

