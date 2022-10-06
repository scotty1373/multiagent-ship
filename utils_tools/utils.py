# -*-  coding=utf-8 -*-
# @Time : 2022/5/16 9:57
# @Author : Scotty1373
# @File : utils.py
# @Software : PyCharm
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from Envs.heatmap import normalize

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/2000):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect

# def heat_map_trans(vect, *, remap_sacle=REMAP_SACLE, ratio=REMAP_SACLE/ORG_SCALE):
#     remap_vect = np.array((vect[0] * ratio + remap_sacle/2, vect[1] * ratio), dtype=np.uint8)
#     return remap_vect


# 数据帧叠加
def state_frame_overlay(new_state, old_state, frame_num):
    agent_num = len(new_state)
    new_frame_overlay = []
    for agent_idx in range(agent_num):
        new_frame_overlay.append(np.concatenate((new_state[agent_idx].reshape(1, -1),
                                                 old_state[agent_idx].reshape(frame_num, -1)[:(frame_num - 1), ...]),
                                                axis=0).reshape(1, -1))
    new_frame_overlay = np.concatenate(new_frame_overlay, axis=0)
    return new_frame_overlay


# 基于图像的数据帧叠加
def pixel_based(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state,
                                       old_state[:, :(frame_num - 1), ...]),
                                       axis=1)
    return new_frame_overlay

def img_proc(img, resize=(80, 80)):
    img = Image.fromarray(img.astype(np.uint8))
    img = np.array(img.resize(resize, resample=Image.BILINEAR))
    # img = img_ori.resize(resize, resample=Image.NEAREST)
    # img.show()
    # img = img_ori.resize(resize, resample=Image.BILINEAR)
    # img.show()
    # img = img_ori.resize(resize, Image.BICUBIC)
    # img.show()
    # img = img_ori.resize(resize, Image.ANTIALIAS)
    # img.show()
    img = rgb2gray(img).reshape(1, 1, 80, 80)
    img = normalize(img)
    return img


def record(global_ep, global_ep_r, ep_r, res_queue, worker_ep, name, idx):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put([idx, ep_r])

    print(f'{name}, '
          f'Global_EP: {global_ep.value}, '
          f'worker_EP: {worker_ep}, '
          f'EP_r: {global_ep_r.value}, '
          f'reward_ep: {ep_r}')


def first_init(env, args):
    trace_history = [[] for i in range(env.env.agent_num)]
    obs_cat = []
    # 类装饰器不改变类内部调用方式
    obs, _, done, _ = env.reset()
    for idx in range(env.env.agent_num):
        '''利用广播机制初始化state帧叠加结构，不使用stack重复对数组进行操作'''
        obs_cat.append((np.ones((args.frame_overlay, args.state_length)) * obs[idx]).reshape(1, -1))
    obs_cat = np.concatenate(obs_cat, axis=0)
    return trace_history, obs_cat, done


def cut_requires_grad(params):
    for param in params:
        param.requires_grad = False


def uniform_init(layer, *, a=-3e-3, b=3e-3):
    torch.nn.init.uniform_(layer.weight, a, b)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


def orthogonal_init(layer, *, gain=1):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class MultiAgentReplayBuffer:
    def __init__(self, max_lens, frame_overlay, state_length, action_dim, agent_num, device):
        self.ptr = 0
        self.size = 0
        self.max_lens = max_lens
        self.state_length = state_length
        self.frame_overlay = frame_overlay
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.device = device

        # state store for critic
        # vect -> List[t1->tuple(agent1_vect, agent2_vect), t2, ...]
        self.vect_state = np.zeros((self.max_lens, self.state_length * self.frame_overlay * self.agent_num))
        self.next_vect_state = np.zeros((self.max_lens, self.state_length * self.frame_overlay * self.agent_num))
        self.reward = np.zeros((self.max_lens, self.agent_num))
        self.done = np.zeros((self.max_lens, self.agent_num))

        # state store for actor
        self.vect = []
        self.next_vect = []
        self.action = []
        # vect -> List[agent1->List[t1, t2, ...], agent2->List[t1, t2, ...], ...]
        for i in range(self.agent_num):
            self.vect.append(np.zeros((self.max_lens, self.state_length * self.frame_overlay)))
            self.next_vect.append(np.zeros((self.max_lens, self.state_length * self.frame_overlay)))
            self.action.append(np.zeros((self.max_lens, self.action_dim)))

    def add(self, vect, next_vect, reward, action, done):
        self.vect_state[self.ptr] = vect.reshape(1, -1).astype(np.float32)
        self.next_vect_state[self.ptr] = next_vect.reshape(1, -1).astype(np.float32)
        self.reward[self.ptr] = reward.astype(np.float32)
        self.done[self.ptr] = done.astype(np.float32)

        for i in range(self.agent_num):
            self.vect[i][self.ptr] = vect[i].astype(np.float32)
            self.next_vect[i][self.ptr] = next_vect[i].astype(np.float32)
            self.action[i][self.ptr] = action[i].astype(np.float32)

        self.ptr = (self.ptr + 1) % self.max_lens
        self.size = min(self.size + 1, self.max_lens)

    def get_batch(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        actor_vect = []
        actor_next_vect = []
        actor_action = []
        for i in range(self.agent_num):
            actor_vect.append(torch.FloatTensor(self.vect[i][ind]).to(self.device))
            actor_next_vect.append(torch.FloatTensor(self.next_vect[i][ind]).to(self.device))
            actor_action.append(torch.FloatTensor(self.action[i][ind]).to(self.device))

        return (actor_vect,
                actor_next_vect,
                actor_action,
                torch.FloatTensor(self.vect_state[ind]).to(self.device),
                torch.FloatTensor(self.next_vect_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.done[ind]).to(self.device))


class NormData:
    def pos_norm(self, pos_cur, pos_min, pos_max):
        """
        # 位置状态归一化
        :param pos_cur: current axis position
        :param pos_min: min axis position
        :param pos_max: max axis position
        :return:
        """
        if pos_min == pos_max:
            pos_cur = 0.0
        else:
            pos_cur = (pos_cur - pos_min) / (pos_max - pos_min)
        return pos_cur

    def ang_norm(self, ang_cur):
        """
        # 朝向角归一化
        :param ang_cur: current angle [0, 360]
        :return:
        """
        ang_cur = (180 - ang_cur) / 180
        return ang_cur

    def rwd_norm(self, rwd_cur, rwd_max):
        """
        # reward归一化
        :param rwd_cur: current reward value
        :param rwd_max: maximum reward value
        :return:
        """
        rwd_cur = rwd_cur / rwd_max
        return rwd_cur