import math
import numpy as np

from utils_tools.functions import *

ships_num = 4

ships_init = np.zeros((ships_num, 2))
ships_goal = np.zeros((ships_num, 2))
ships_speed = np.zeros((ships_num, 1))
ships_head = np.zeros((ships_num, 1))
ships_length = np.zeros((ships_num, 1))

ships_init[0, :] = np.array([0, 1900])
ships_goal[0, :] = np.array([0, 100])
ships_speed[0] = 5
ships_head[0] = -90
ships_length[0] = 50

ships_init[1, :] = np.array([-900, 900])
ships_goal[1, :] = np.array([900, 900])
ships_speed[1] = 5
ships_head[1] = 0
ships_length[1] = 50

ships_init[2, :] = np.array([0, 100])
ships_goal[2, :] = np.array([0, 1900])
ships_speed[2] = 5
ships_head[2] = 90
ships_length[2] = 50

ships_init[3, :] = np.array([900, 900])
ships_goal[3, :] = np.array([-900, 900])
ships_speed[3] = 5
ships_head[3] = 180
ships_length[3] = 50

# actions of ships
ship_action_space = 1
angle_limit = 10  # heading angle changing range (-30, 30)

# distance redundant
dis_redundant = 50

# --------------------------------------------------
# calculate below data based on given data
ships_given_pos = np.vstack((ships_init.reshape(-1), ships_goal.reshape(-1)))
ships_pos_min = ships_given_pos.min(0)
ships_pos_max = ships_given_pos.max(0)
ships_x_min = []
ships_y_min = []
ships_x_max = []
ships_y_max = []
ships_dis = []
# 用于state归一化
for ship_idx in range(ships_num):
    ships_x_min.append(ships_pos_min[ship_idx * 2]), ships_y_min.append(ships_pos_min[ship_idx * 2 + 1])
    ships_x_max.append(ships_pos_max[ship_idx * 2]), ships_y_max.append(ships_pos_max[ship_idx * 2 + 1])
    ships_dis.append(euc_dist(ships_init[ship_idx, 0], ships_goal[ship_idx, 0],
                              ships_init[ship_idx, 1], ships_goal[ship_idx, 1]))
ships_dis_max = np.array(ships_dis).max(-1)
ships_vel_min = ships_speed.min(0)
ship_max_length = np.array(ships_length).max()
ship_max_speed = np.array(ships_speed).max()
