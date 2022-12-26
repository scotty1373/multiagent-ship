# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Envs.multiagent import RoutePlan
from wrapper.wrapper import SkipEnvFrame
from MATD3.MATD3 import MATD3
from utils_tools.common import TIMESTAMP, seed_torch
from utils_tools.utils import state_frame_overlay, pixel_based, first_init, trace_trans
from utils_tools.utils import MultiAgentReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(
        description='MATD3 config option')
    parser.add_argument('--mode',
                        default='4Ship_CrossAway',
                        type=str,
                        help='environment name')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=1000,
                        type=int)
    parser.add_argument('--train',
                        help='Train or not',
                        default=False,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False,
                        type=bool)
    parser.add_argument('--checkpoint_path',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default='./log/4Ship_CrossAway_1671609054/save_model_ep75',
                        type=str)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=256,
                        type=int)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42,
                        type=int)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=64,
                        type=int)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=2,
                        type=int)
    parser.add_argument('--frame_overlay',
                        help='data frame overlay',
                        default=4,
                        type=int)
    # parser.add_argument('--state_length',
    #                     help='state data vector length',
    #                     default=5+24*2)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=5,
                        type=int)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False,
                        type=bool)
    parser.add_argument('--device',
                        help='data device',
                        default='cpu',
                        type=str)
    parser.add_argument('--replay_buffer_size',
                        help='Replay Buffer Size',
                        default=12000,
                        type=int)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)
    test_saver_path = f'./log/{args.mode}_{TIMESTAMP}_test'
    if not os.path.exists(test_saver_path):
        os.makedirs(test_saver_path)

    # 是否随机初始化种子
    if args.seed is not None:
        seed = args.seed
    else:
        seed = None

    # 环境与agent初始化
    env = RoutePlan(mode=args.mode, seed=seed)
    # env.seed(13)
    env = SkipEnvFrame(env, args.frame_skipping)
    assert isinstance(args.batch_size, int)

    """初始化agent"""
    agent = MATD3(frame_overlay=args.frame_overlay,
                  state_length=args.state_length,
                  action_dim=1,
                  batch_size=args.batch_size,
                  agent_num=env.env.agent_num,
                  device=device,
                  train=args.train)

    # 载入预训练模型
    checkpoint = args.checkpoint_path
    agent.load_model(checkpoint)

    done = [True] * agent.agent_num
    done_coll = [False] * agent.agent_num

    # NameSpace
    trace_history = []
    dis_history = []
    obs = None

    env.reset()
    trace_image = env.render(mode='rgb_array')
    trace_image = Image.fromarray(trace_image)
    trace_path = ImageDraw.Draw(trace_image)

    steps = tqdm(range(30000), leave=False, position=0, colour='green')
    for step in steps:
        if all(done):
            """轨迹记录"""
            trace_history = {f'agent_{i}': [] for i in range(agent.agent_num)}
            dis_history = []
            _, obs, done = first_init(env, args)
            # """初始化agent探索轨迹追踪"""
            trace_image = env.render(mode='rgb_array')
            trace_image = Image.fromarray(trace_image)
            trace_path = ImageDraw.Draw(trace_image)

        act = agent.get_action(obs)
        # 环境交互
        for i in done:
            if i and not any(done_coll):
                act[i] = 0.0
        obs_t1, reward, done, info = env.step(act)
        done_coll = info['done_coll']
        dis_closest = info['dis_closest']

        # 历史数据记录
        for i in range(agent.agent_num):
            trace_history[f'agent_{i}'].append(tuple(trace_trans(env.ships[i].position, ratio=0.24)))
        dis_history.append(dis_closest)

        # 随机漫步如果为1则不进行数据庞拼接
        if args.frame_overlay == 1:
            pass
        else:
            obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)

        # 记录timestep, reward＿sum
        agent.t += 1
        obs = obs_t1

        # trace_history.append(tuple(trace_trans(env.env.ship.position)))
        steps.set_description(f"step: {step}, "
                              f"ori_agent1: {act[0, 0].item():.2f}, "
                              f"ori_agent2: {act[1, 0].item():.2f}, "
                              f"reward_agent1: {reward[0]:.2f}, "
                              f"reward_agent2: {reward[1]:.2f}, "
                              f"done_agent1: {done[0]}, "
                              f"done_agent2: {done[1]}, ")

        if all(done) and not any(done_coll):
            dis_history_tmp = np.array(dis_history)
            for idx, node in enumerate(trace_history.values()):
                node.append(tuple(trace_trans(env.env.ships_goal[idx], ratio=0.24)))
                trace_path.point(node, fill='Blue')
            trace_image.save(f'./log/{args.mode}_{TIMESTAMP}_test/track_{step}.png', quality=95)
            for dist_enum in range(dis_history_tmp.shape[1]):
                plt.plot(range(dis_history_tmp.shape[0]), dis_history_tmp[:, dist_enum])

            plt.savefig(f'./log/{args.mode}_{TIMESTAMP}_test/dist_{step}.png')

        agent.ep += 1
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
