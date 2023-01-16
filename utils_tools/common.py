# -*- coding: utf-8 -*-
import csv
import json
import sys
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from pathlib import Path
import os
import torch
from sys import platform

TIMESTAMP = str(round(time.time()))
KEYs_Train = ['epochs', 'time_step', 'ep_reward', 'entropy_mean']

FILE_NAME = './csv_log/sa_critic/train_loga'


# FILE_NAME = '../log/ppo_origin'


class log2json:
    def __init__(self, filename='train_log', type_json=True, log_path='log', logger_keys=None):
        self.root_path = os.getcwd()
        self.log_path = os.path.join(self.root_path, log_path, TIMESTAMP)

        # 创建当前训练log保存目录
        try:
            os.makedirs(self.log_path)
        except FileExistsError as e:
            print(e)

        filename = filename + TIMESTAMP
        if type_json:
            filename = os.path.join(self.log_path, filename + '.json').replace('\\', '/')
            self.fp = open(filename, 'w')
        else:
            filename = os.path.join(self.log_path, filename + '.csv')
            self.fp = open(filename, 'w', encoding='utf-8', newline='')
            self.csv_writer = csv.writer(self.fp)
            self.csv_keys = logger_keys
            self.csv_writer.writerow(self.csv_keys)

    def flush_write(self, string):
        self.fp.write(string + '\n')
        self.fp.flush()

    def write2json(self, log_dict):
        format_str = json.dumps(log_dict)
        self.flush_write(format_str)

    def write2csv(self, log_dict):
        format_str = list(log_dict.values())
        self.csv_writer.writerow(format_str)


class visualize_result:
    def __init__(self):
        sns.set_style("dark")
        sns.axes_style("darkgrid")
        sns.despine()
        sns.set_context("paper")

    @staticmethod
    def json2DataFrame(file_path):
        result_train = dict()
        # 创建train mode数据键值
        for key_str in KEYs_Train:
            result_train[key_str] = []
        # 文件读取
        with open(file_path, 'r') as fp:
            while True:
                json_line_str = fp.readline().rstrip('\n')
                if not json_line_str:
                    break
                _, temp_dict = json_line_extract(json_line_str)
                for key_iter in result_train.keys():
                    result_train[key_iter].append(temp_dict[key_iter])

        assert len(result_train['mode']) == len(result_train['ep_reward'])
        df_train = pd.DataFrame.from_dict(result_train)
        df_train.head()
        return df_train

    # @staticmethod
    def reward(self, logDataFrame, save_root):
        col = logDataFrame[0].shape[0]
        df_collect = pd.DataFrame(np.zeros((len(logDataFrame) * logDataFrame[0].shape[0], 3)),
                                  columns=['epochs', 'worker', 'ep_reward'])

        for idx, df_log in enumerate(logDataFrame):
            df_collect.iloc[idx * col:(idx + 1) * col, 0] = df_log['epochs']
            df_collect['worker'][idx * col:(idx + 1) * col] = f'worker_{idx}'
            smooth_data = self._tensorboard_smoothing(df_log['ep_reward'], 0.93)
            df_collect.iloc[idx * col:(idx + 1) * col, 2] = smooth_data
        df_collect.head()

        # df_insert = pd.melt(df_insert, 'epochs', var_name='workers', value_name='ep_reward')

        # 使用长数据格式用于不同agent累计回报可视化
        self.sns_plot(df_collect, logDataFrame[0].shape[0])

        """
        宽数据：宽数据是指数据集对所有的变量进行了明确的细分，各变量的值不存在重复循环的情况也无法归类。
            数据总体的表现为 变量多而观察值少。每一列为一个变量，每一行为变量所对应的值。
        长数据：长数据一般是指数据集中的变量没有做明确的细分，即变量中至少有一个变量中的元素存在值严重重复循环的情况（可以归为几类），
            表格整体的形状为长方形，即 变量少而观察值多。一列包含了所有的变量，而另一列则是与之相关的值。
        """

        # 将数据从长数据形式重塑成宽数据形式
        df_collect = df_collect.pivot(index='epochs', columns='worker', values='ep_reward')
        df_collect.to_csv(os.path.join(FILE_NAME + f'/{save_root}', 'smoothed_reward.csv'))

    def reward_all_agent(self, logDataFrame, save_root):
        log_num = len(logDataFrame)
        col = logDataFrame[0].shape[0]
        df_collect = pd.DataFrame(np.zeros((col, 2)),
                                  columns=['epochs', 'ep_reward'])
        # epochs 数据同步

        for row_idx, df_log in enumerate(range(col)):
            buffer = 0
            for log_ep_reward_idx in range(log_num):
                buffer += logDataFrame[log_ep_reward_idx]['ep_reward'][row_idx]
            df_collect.iloc[row_idx, 1] = buffer / log_num

        smooth_data = self._tensorboard_smoothing(df_collect['ep_reward'], 0.93)
        df_collect['epochs'] = logDataFrame[0]['epochs']
        df_collect['ep_reward'] = smooth_data
        df_collect.head()

        df_collect.to_csv(os.path.join(FILE_NAME + f'/{save_root}', 'smoothed_reward_all_agent.csv'))

    def uni_loss(self):
        pass

    def average_value(self):
        pass

    def _tensorboard_smoothing(self, values, smooth=0.97):
        # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
        norm_factor = smooth + 1
        x = values[0]
        res = [x]
        for i in range(1, len(values)):
            x = x * smooth + values[i]  # 指数衰减
            res.append(x / norm_factor)

            norm_factor *= smooth
            norm_factor += 1

        return res

    @staticmethod
    def sns_plot(df_collect, x_lim):
        sns.set_style('darkgrid')
        sns.set_context('paper')
        figure = sns.lineplot(data=df_collect, x='epochs', y='ep_reward', hue='worker')
        figure.set_xlim(0, x_lim)
        figure.set_ylim(-200, 200)
        # figure.set_yticks([-1500, -1000, -500, 0, 400])
        plt.xlabel('epoch', fontdict={'weight': 'bold',
                                      'size': 12})
        plt.ylabel('ep_reward', fontdict={'weight': 'bold', 'size': 12})
        # plt.subplots_adjust(left=0.2, bottom=0.2)
        # plt.title(y_trick, fontdict={'weight': 'bold',
        #                              'size': 20})
        # plt.legend(labels=['ppo ep reward'], loc='lower right', fontsize=10)
        plt.show()

    @staticmethod
    def dataframe_collect(file_name):
        df_train = pd.read_csv(file_name)
        df_train = df_train.drop(columns='Wall time')
        df_train.columns = ['epochs', 'ep_reward']
        return df_train

    @staticmethod
    def sort_agent_log(file_p, target_p, file_name):
        type_list = ['fc_critic', 'sa_critic']
        train_logs = os.listdir(os.path.join(file_p, type_list[0]))
        env_list = [i for i in os.listdir(os.path.join(file_p,
                                                       type_list[0],
                                                       train_logs[0]))
                    if Path(os.path.join(file_p,
                                         type_list[0],
                                         train_logs[0], i)).is_dir()]
        for env in env_list:
            df_common = None
            agent_num = None
            for log_idx, log in enumerate(train_logs):
                df_fc_critic = pd.read_csv(os.path.join(file_p, 'fc_critic', log, env, file_name))
                df_sa_critic = pd.read_csv(os.path.join(file_p, 'sa_critic', log, env, file_name))
                env_agent = df_fc_critic.keys()[1:]
                for idx, agent in enumerate(env_agent):
                    # 变更columns名称
                    tmp = list(df_fc_critic.columns)
                    tmp[idx+1] = f'agent_{idx}_fc_{log}'
                    df_fc_critic.columns = tmp
                    tmp = list(df_sa_critic.columns)
                    tmp[idx+1] = f'agent_{idx}_sa_{log}'
                    df_sa_critic.columns = tmp

                # 数据做拼接处理
                df_sa_critic = df_sa_critic.drop('epochs', axis=1)
                if log_idx == 0:
                    agent_num = len(env_agent)
                    df_common = pd.concat([df_fc_critic, df_sa_critic], axis=1)
                else:
                    df_fc_critic = df_fc_critic.drop('epochs', axis=1)
                    df_common = pd.concat([df_common, df_fc_critic, df_sa_critic], axis=1)
                    opt_fc = 0
                    opt_sa = 0
                    for agent_idx in range(agent_num):
                        tmp = df_common.pop(df_common.keys()[1+log_idx*2*agent_num+agent_idx+opt_sa])
                        df_common.insert(1+log_idx+agent_idx+opt_fc, tmp.name, tmp)
                        opt_fc += 1
                        if agent_idx == agent_num - 1:
                            continue
                        else:
                            tmp = df_common.pop(df_common.keys()[1+log_idx*2*agent_num+agent_num+agent_idx])
                        df_common.insert(1+log_idx+agent_idx+agent_num+opt_fc+opt_sa, tmp.name, tmp)
                        opt_sa += 1

                    # # [done:] testing 嵌入顺序
                    # test_list = list(df_common.columns)
                    # opt_fc = 0
                    # opt_sa = 0
                    # # for i in range(2*log_idx*len(env_agent), 2*(log_idx+1)*len(env_agent)):
                    # for agent_idx in range(agent_num):
                    #     tmp = test_list.pop(1+log_idx*2*agent_num+agent_idx+opt_sa)
                    #     test_list.insert(1+log_idx+agent_idx+opt_fc, tmp)
                    #     opt_fc += 1
                    #     if agent_idx == agent_num - 1:
                    #         continue
                    #     else:
                    #         tmp = test_list.pop(1+log_idx*2*agent_num+agent_num+agent_idx)
                    #     test_list.insert(1+log_idx+agent_idx+agent_num+opt_fc+opt_sa, tmp)
                    #     opt_sa += 1

            df_common.to_csv(f'{target_p}/{env}.csv')

    @staticmethod
    def collect_all_agent_log(file_p, target_p, file_name):
        type_list = ['fc_critic', 'sa_critic']
        train_logs = os.listdir(os.path.join(file_p, type_list[0]))
        env_list = [i for i in os.listdir(os.path.join(file_p,
                                                       type_list[0],
                                                       train_logs[0]))
                    if Path(os.path.join(file_p,
                                         type_list[0],
                                         train_logs[0], i)).is_dir()]
        for env in env_list:
            df_common = None
            agent_num = None
            for log_idx, log in enumerate(train_logs):
                df_fc_critic = pd.read_csv(os.path.join(file_p, 'fc_critic', log, env, file_name))
                df_fc_critic.columns = ['0', 'epochs', 'ep_reward']
                df_fc_critic = df_fc_critic.drop('0', axis=1)
                df_sa_critic = pd.read_csv(os.path.join(file_p, 'sa_critic', log, env, file_name))
                df_sa_critic.columns = ['0', 'epochs', 'ep_reward']
                df_sa_critic = df_sa_critic.drop('0', axis=1)
                env_agent = df_fc_critic.keys()[1:]
                for idx, agent in enumerate(env_agent):
                    # 变更columns名称
                    tmp = list(df_fc_critic.columns)
                    tmp[idx + 1] = f'agent_{idx}_fc_{log}'
                    df_fc_critic.columns = tmp
                    tmp = list(df_sa_critic.columns)
                    tmp[idx + 1] = f'agent_{idx}_sa_{log}'
                    df_sa_critic.columns = tmp

                # 数据做拼接处理
                df_sa_critic = df_sa_critic.drop('epochs', axis=1)
                if log_idx == 0:
                    agent_num = len(env_agent)
                    df_common = pd.concat([df_fc_critic, df_sa_critic], axis=1)
                else:
                    df_fc_critic = df_fc_critic.drop('epochs', axis=1)
                    df_common = pd.concat([df_common, df_fc_critic, df_sa_critic], axis=1)

            df_common.to_csv(f'{target_p}/{env}_all_agent.csv')


def dirs_creat():
    if platform.system() == 'windows':
        temp = os.getcwd()
        CURRENT_PATH = temp.replace('\\', '/')
    else:
        CURRENT_PATH = os.getcwd()
    ckpt_path = os.path.join(CURRENT_PATH, 'save_Model')
    log_path = os.path.join(CURRENT_PATH, 'log')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)


# 设置相同训练种子
def seed_torch(seed=2331):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为当前CPU 设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前的GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)  # 当使用多块GPU 时，均设置随机种子
        torch.backends.cudnn.deterministic = True  # 设置每次返回的卷积算法是一致的
        torch.backends.cudnn.benchmark = True  # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False
        torch.backends.cudnn.enabled = True  # pytorch使用cuDNN加速


def json_line_extract(json_format_str):
    return json.loads(json_format_str)


if __name__ == '__main__':
    vg = visualize_result()
    log_proc = input('Processing Or Sorting:')
    if bool(log_proc):
        for root_list in os.listdir(FILE_NAME):
            df = []
            file_list = glob.glob(FILE_NAME + f'/{root_list}' + '/*.csv')
            for fp in file_list:
                path_csv_log = fp.replace("\\", "/")
                df.append(vg.dataframe_collect(path_csv_log))

            vg.reward(df, root_list)
            vg.reward_all_agent(df, root_list)
    else:
        file_path = os.path.join(os.getcwd(), 'csv_log')
        target_path = os.path.join(os.getcwd(), 'processed_log')
        if not os.path.exists(target_path):
            os.mkdir(target_path)

        vg.sort_agent_log(file_path, target_path, 'smoothed_reward.csv')
        vg.collect_all_agent_log(file_path, target_path, 'smoothed_reward_all_agent.csv')


