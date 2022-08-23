# -*- coding: utf-8 -*-
import numpy as np
import torch
from copy import deepcopy
from torch.distributions import Normal
from MATD3.agent import Agent


class MATD3:
    def __init__(self, frame_overlay, state_length, action_dim, batch_size, overlay, agent_num, device, train=True, logger=None):
        self.agent_num = agent_num
        self.frame_overlay = frame_overlay
        self.state_length = state_length
        self.state_dim = self.frame_overlay * self.state_length
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.overlay = overlay
        self.device = device
        self.train = train
        self.logger = logger

        # 初始化agent actor + 双q网络 训练参数
        self._init()
        self.t = 0
        self.ep = 0
        self.start_train = 2000

    def _init(self):
        self.agent_list = [Agent(self.state_dim, self.action_dim,
                                 self.agent_num, self.device,
                                 i, self.train, self.logger) for i in range(self.agent_num)]

    def get_action(self, vect_state):
        pixel = torch.FloatTensor(pixel_state).to(self.device)
        vect = torch.FloatTensor(vect_state).to(self.device)
        logits = self.actor_model(pixel, vect)
        if self.train:
            # acc 动作裁剪
            logits[..., 0] = (logits[..., 0] + self.noise.sample()).clamp_(min=0.3, max=1)
            # ori 动做裁剪
            logits[..., 1] = (logits[..., 1] + self.noise.sample()).clamp_(min=self.action_space.min(),
                                                                           max=self.action_space.max())
        else:
            logits[..., 0] = logits[..., 0].clamp_(min=0.5, max=1)
            logits[..., 1] = logits[..., 1].clamp_(min=self.action_space.min(), max=self.action_space.max())
        return logits.detach().cpu().numpy()

    def update(self, replay_buffer):
        if replay_buffer.size < self.start_train:
            return
        # batch数据处理
        pixel, next_pixel, vect, next_vect, reward, action, done = replay_buffer.get_batch(self.batch_size)

        # critic更新
        self.critic_update(pixel, vect, action, next_pixel, next_vect, reward, done)

        # actor延迟更新
        if self.t % self.delay_update == 0:
            self.action_update(pixel_state=pixel, vect_state=vect)
            with torch.no_grad():
                self.model_soft_update(self.actor_model, self.actor_target)
                self.model_soft_update(self.critic_model, self.critic_target)

        self.logger.add_scalar(tag='actor_loss',
                               scalar_value=self.actor_loss_history,
                               global_step=self.t)
        self.logger.add_scalar(tag='critic_loss',
                               scalar_value=self.critic_loss_history,
                               global_step=self.t)

    # critic theta1为更新主要网络，theta2用于辅助更新
    def action_update(self, pixel_state, vect_state):
        act = self.actor_model(pixel_state, vect_state)
        critic_q1 = self.critic_model.q_theta1(pixel_state, vect_state, act)
        actor_loss = - torch.mean(critic_q1)
        self.critic_uni_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_loss_history = actor_loss.item()

    def critic_update(self, pixel_state, vect_state, action, next_pixel_state, next_vect_state, reward, done):
        with torch.no_grad():
            target_action = self.actor_target(next_pixel_state, next_vect_state)

            noise = self.target_model_regular_noise.sample(sample_shape=target_action.shape).to(self.device)
            epsilon = torch.clamp(noise, min=-self.smooth_regular, max=self.smooth_regular)
            smooth_tg_act = target_action + epsilon
            smooth_tg_act[..., 0].clamp_(min=0, max=self.action_space.max())
            smooth_tg_act[..., 1].clamp_(min=self.action_space.min(), max=self.action_space.max())

            target_q1, target_q2 = self.critic_target(next_pixel_state, next_vect_state, smooth_tg_act)
            target_q1q2 = torch.cat([target_q1, target_q2], dim=1)

            # 根据论文附录中遵循deep Q learning，增加终止状态
            td_target = reward + self.discount_index * (1 - done) * torch.min(target_q1q2, dim=1)[0].reshape(-1, 1)

        q1_curr, q2_curr = self.critic_model(pixel_state, vect_state, action)
        loss_q1 = self.critic_loss(q1_curr, td_target)
        loss_q2 = self.critic_loss(q2_curr, td_target)
        loss_critic = loss_q1 + loss_q2

        self.critic_uni_opt.zero_grad()
        loss_critic.backward()
        self.critic_uni_opt.step()
        self.critic_loss_history = loss_critic.item()

