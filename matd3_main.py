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

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def parse_args():
    parser = argparse.ArgumentParser(
        description='MATD3 config option')
    parser.add_argument('--mode',
                        default='2Ship_CrossAway',
                        type=str,
                        help='environment name')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=1000,
                        type=int)
    parser.add_argument('--train',
                        help='Train or not',
                        default=True,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False,
                        type=bool)
    parser.add_argument('--checkpoint_path',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default='./log/1659972659/save_model_ep800.pth',
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
                        default=1,
                        type=int)
    # parser.add_argument('--state_length',
    #                     help='state data vector length',
    #                     default=5+24*2)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=3,
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
                        default=8000,
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
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)

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

    replay_buffer = MultiAgentReplayBuffer(max_lens=args.replay_buffer_size,
                                           frame_overlay=args.frame_overlay,
                                           state_length=args.state_length,
                                           action_dim=1,
                                           agent_num=env.env.agent_num,
                                           device=device)

    """初始化agent"""
    agent = MATD3(frame_overlay=args.frame_overlay,
                  state_length=args.state_length,
                  action_dim=1,
                  batch_size=args.batch_size,
                  agent_num=env.env.agent_num,
                  device=device,
                  train=args.train,
                  logger=tb_logger)

    # pretrained 选项，载入预训练模型
    if args.pre_train:
        if args.checkpoint_path is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    # """初始化agent探索轨迹追踪"""
    # env.reset()
    # trace_image = env.render(mode='rgb_array')
    # trace_image = Image.fromarray(trace_image)
    # trace_path = ImageDraw.Draw(trace_image)

    done = [True] * agent.agent_num
    # ep_history = []

    # NameSpace
    trace_history = None
    pixel_obs = None
    obs = None

    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')
    for epoch in epochs:
        reward_history = np.zeros((agent.agent_num,))
        if any(done):
            """轨迹记录"""
            _, obs, done = first_init(env, args)

        # timestep 样本收集
        steps = tqdm(range(0, args.max_timestep), leave=False, position=1, colour='red')
        for t in steps:
            act = agent.get_action(obs)
            # 环境交互
            obs_t1, reward, done, info = env.step(act)
            done_coll = info['done_coll']
            dis_closest = info['dis_closest']

            # 随机漫步如果为1则不进行数据庞拼接
            if args.frame_overlay == 1:
                pass
            else:
                obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)

            if args.train:
                # 达到最大timestep则认为单幕完成
                if t == args.max_timestep - 1:
                    done = [True] * agent.agent_num
                    done = np.array(done, dtype=bool)
                # 状态存储
                replay_buffer.add(obs, obs_t1, reward, act, done)
                agent.update(replay_buffer)

            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            reward_history += reward

            # trace_history.append(tuple(trace_trans(env.env.ship.position)))
            steps.set_description(f"epochs: {epoch}, "
                                  f"time_step: {agent.t}, "
                                  f"ep_reward_agent1: {reward_history[0].item():.2f}, "
                                  f"ep_reward_agent2: {reward_history[1].item():.2f}, "
                                  f"ori_agent1: {act[0, 0].item():.2f}, "
                                  f"ori_agent2: {act[1, 0].item():.2f}, "
                                  f"reward_agent1: {reward[0]:.2f}, "
                                  f"reward_agent2: {reward[1]:.2f}, "
                                  f"done_agent1: {done[0]}, "
                                  f"done_agent2: {done[1]}, "
                                  f"actor_loss: {agent.loss_history_actor:.2f}, "
                                  f"critic_loss: {agent.loss_history_critic:.2f}")

            # 单幕数据收集完毕
            if any(done):
                # 单幕结束显示轨迹
                _, obs, done = first_init(env, args)

        # ep_history.append(reward_history)
        # log_ep_text = {'epochs': epoch,
        #                'time_step': agent.t,
        #                'ep_reward': reward_history}
        agent.ep += 1

        # tensorboard logger
        for agent_idx in range(agent.agent_num):
            tb_logger.add_scalar(tag=f'Reward/ep_reward_agent_{agent_idx}',
                                 scalar_value=reward_history[agent_idx],
                                 global_step=epoch)
        # tb_logger.add_image(tag=f'Image/Trace',
        #                     img_tensor=np.array(trace_image),
        #                     global_step=epoch,
        #                     dataformats='HWC')

        # 环境重置
        if not epoch % 25:
            env.close()
            env = RoutePlan(mode=args.mode, seed=seed)
            env = SkipEnvFrame(env, args.frame_skipping)
            agent_save_path = f'./log/{TIMESTAMP}/save_model_ep{epoch}'
            if not os.path.exists(agent_save_path):
                os.mkdir(agent_save_path)
            agent.save_model(agent_save_path)
    env.close()
    tb_logger.close()


def EnvBarrierReset(ep, *, start_barrier_num, train=True):
    if ep <= 100 and train:
        return start_barrier_num
    elif 100 < ep <= 250:
        return start_barrier_num * 2
    elif 250 < ep <= 450:
        return start_barrier_num * 3
    elif 450 < ep or not train:
        return start_barrier_num * 5


if __name__ == '__main__':
    config = parse_args()
    main(config)
