# -*- coding: utf-8 -*-
import numpy as np
import torch
from copy import deepcopy
from torch.distributions import Normal
from MATD3.agent import Agent


class MATD3:
    def __init__(self, frame_overlay, state_length, action_dim, batch_size, agent_num, device, train=True, logger=None):
        self.agent_num = agent_num
        self.frame_overlay = frame_overlay
        self.state_length = state_length
        self.state_dim = self.frame_overlay * self.state_length
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device
        self.train = train
        self.logger = logger

        # 初始化agent actor + 双q网络 训练参数
        self._init()
        self.t = 0
        self.ep = 0
        self.start_train = 2000
        self.delay_update = 5
        self.loss_history_critic = 0
        self.loss_history_actor = 0

    def _init(self):
        self.agent_list = [Agent(self.state_dim, self.action_dim,
                                 self.agent_num, self.device,
                                 i, self.train, self.logger) for i in range(self.agent_num)]

    def get_action(self, vect_state):
        """
        # 获取各agent动作
        :param vect_state: vect_state.shape = (agent_num, state_dim)
        :return: actions.shape = (agent_num, action_dim)
        """
        actions = []
        for idx in range(self.agent_num):
            actions.append(self.agent_list[idx].get_action(vect_state[idx].reshape(1, -1)))
        return np.concatenate(actions, axis=0)

    def update(self, replay_buffer):
        if replay_buffer.size < self.start_train:
            return
        # batch数据处理
        actor_vect, actor_next_vect, actor_action, vect_state, next_vect_state, reward, done = replay_buffer.get_batch(self.batch_size)

        # 计算用于critic和actor更新的action
        # 目标网络计算action， 用于td-error计算
        agent_target_action = []
        # 实际环境交互action，用于计算td-error
        agent_online_action = []

        for idx in range(self.agent_num):
            with torch.no_grad():
                agent_target_action.append(self.agent_list[idx].actor_target(actor_next_vect[idx]))
                agent_online_action.append(actor_action[idx])

        # [todo: 检查是否按agent_num维度拼接, 如果是二维动作，则需要增加维度]
        # flatten agent action
        agent_target_action = torch.cat(agent_target_action, dim=1)
        agent_online_action = torch.cat(agent_online_action, dim=1)

        # loss record
        critic_loss = 0
        actor_loss = 0

        # [todo: 测试维度是否匹配...]
        # critic更新
        for idx in range(self.agent_num):
            reward_critic = reward[..., idx].unsqueeze(1)
            done_critic = done[..., idx].unsqueeze(1)

            critic_loss += self.critic_update(idx, vect_state, next_vect_state, agent_target_action, agent_online_action, reward_critic, done_critic)

            # actor延迟更新
            if self.t % self.delay_update == 0:
                actor_loss += self.action_update(vect_state=vect_state, actor_vect=actor_vect, agent_idx=idx)
                with torch.no_grad():
                    self.agent_list[idx].model_soft_update(self.agent_list[idx].actor_model,
                                                           self.agent_list[idx].actor_target)
                    self.agent_list[idx].model_soft_update(self.agent_list[idx].critic_model,
                                                           self.agent_list[idx].critic_target)

        self.logger.add_scalar(tag='actor_loss',
                               scalar_value=critic_loss,
                               global_step=self.t)
        self.logger.add_scalar(tag='critic_loss',
                               scalar_value=actor_loss,
                               global_step=self.t)

    # critic theta1为更新主要网络，theta2用于辅助更新
    def action_update(self, vect_state, actor_vect, agent_idx):
        # 在线网络计算action，用于actor更新
        agent_actor_action = []

        for idx in range(self.agent_num):
            agent_actor_action.append(self.agent_list[idx].actor_model(actor_vect[idx]))

        agent_actor_action = torch.cat(agent_actor_action, dim=1)

        critic_q1 = self.agent_list[agent_idx].critic_model.q_theta1(vect_state, agent_actor_action)
        actor_loss = - torch.mean(critic_q1)
        self.agent_list[agent_idx].critic_uni_opt.zero_grad()
        self.agent_list[agent_idx].actor_opt.zero_grad()
        actor_loss.backward()
        self.agent_list[agent_idx].actor_opt.step()
        return actor_loss.detach().cpu().item()

    def critic_update(self, agent_idx, vect_state, next_vect_state, target_action, action_env, reward, done):
        # [todo: critic network输入维度是否匹配...]
        with torch.no_grad():
            noise = self.agent_list[agent_idx].target_model_regular_noise.sample(sample_shape=target_action.shape).to(self.device)
            epsilon = torch.clamp(noise, min=-self.agent_list[agent_idx].smooth_regular,
                                  max=self.agent_list[agent_idx].smooth_regular)
            smooth_tg_act = target_action + epsilon
            smooth_tg_act.clamp_(min=self.agent_list[agent_idx].action_space.min(), max=self.agent_list[agent_idx].action_space.max())

            target_q1, target_q2 = self.agent_list[agent_idx].critic_target(next_vect_state, smooth_tg_act)
            target_q1q2 = torch.cat([target_q1, target_q2], dim=1)

            # 根据论文附录中遵循deep Q learning，增加终止状态
            td_target = reward + self.agent_list[agent_idx].discount_index * (1 - done) * torch.min(target_q1q2, dim=1)[0].reshape(-1, 1)

        q1_curr, q2_curr = self.agent_list[agent_idx].critic_model(vect_state, action_env)
        loss_q1 = self.agent_list[agent_idx].critic_loss(q1_curr, td_target)
        loss_q2 = self.agent_list[agent_idx].critic_loss(q2_curr, td_target)
        loss_critic = loss_q1 + loss_q2

        self.agent_list[agent_idx].critic_uni_opt.zero_grad()
        loss_critic.backward()
        self.agent_list[agent_idx].critic_uni_opt.step()
        return loss_critic.detach().cpu().item()

    def load_model(self, ckpt_path):
        ckpt_epoch = int(ckpt_path.split('/')[-1].split('_')[-1][2:])
        for idx, agent in enumerate(self.agent_list):
            agent_ckpt_name = ckpt_path + "/" + f'save_model_ep{ckpt_epoch}_agent{idx}.pth'
            agent.load_model(agent_ckpt_name)

    def save_model(self, ckpt_path):
        for idx, agent in enumerate(self.agent_list):
            agent_ckpt_name = ckpt_path + "/" + f'save_model_ep{self.ep}_agent{idx}.pth'
            agent.save_model(agent_ckpt_name)
