# -*-  coding=utf-8 -*-
# @Time : 2022/8/20 10:46
# @Author : Scotty1373
# @File : multiagent
# @Software : PyCharm
import math
import time

import numpy as np
import keyboard

import Box2D
from Box2D import (b2CircleShape, b2FixtureDef,
                   b2PolygonShape, b2ContactListener,
                   b2Distance, b2RayCastCallback,
                   b2Vec2, b2_pi, b2Dot)
from utils_tools.functions import *
"""
b2FixtureDef: 添加物体材质
b2PolygonShape： 多边形构造函数，初始化数据
b2CircleShape: 圆形构造函数
b2ContactListener：碰撞检测监听器
"""

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from utils_tools.utils import img_proc
import config
from utils_tools.check_state import CheckState
from utils_tools.utils import NormData

SCALE = 0.24
FPS = 60

VIEWPORT_W = 480
VIEWPORT_H = 480

INITIAL_RANDOM = 20
MAIN_ENGINE_POWER = 0.3
MAIN_ORIENT_POWER = b2_pi/6
SIDE_ENGINE_POWER = 5

#           Background           PolyLine
PANEL = [(0.19, 0.72, 0.87), (0.10, 0.45, 0.56),  # ship
         (0.22, 0.16, 0.27), (0.31, 0.30, 0.31),  # barriers
         (0.87, 0.4, 0.23), (0.58, 0.35, 0.28),  # reach area
         (0.25, 0.41, 0.88), (0, 0, 0)]

RAY_CAST_LASER_NUM = 24

SHIP_POLY_BP = [
    (-5, +8), (-5, -8), (0, -8),
    (+8, -6), (+8, +6), (0, +8)
    ]

SHIP_POSITION = [(-6.5, 1.5), (6.5, 14.5)]

element_wise_weight = 0.8
SHIP_POLY = [
    (SHIP_POLY_BP[0][0]*element_wise_weight, SHIP_POLY_BP[0][1]*element_wise_weight),
    (SHIP_POLY_BP[1][0]*element_wise_weight, SHIP_POLY_BP[1][1]*element_wise_weight),
    (SHIP_POLY_BP[2][0]*element_wise_weight, SHIP_POLY_BP[2][1]*element_wise_weight),
    (SHIP_POLY_BP[3][0]*element_wise_weight, SHIP_POLY_BP[3][1]*element_wise_weight),
    (SHIP_POLY_BP[4][0]*element_wise_weight, SHIP_POLY_BP[4][1]*element_wise_weight),
    (SHIP_POLY_BP[5][0]*element_wise_weight, SHIP_POLY_BP[5][1]*element_wise_weight)
    ]

REACH_POLY = [(-1, 1), (-1, -1), (1, -1), (1, 1)]


class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        """
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        """
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        # NOTE: You will get this error:
        #   "TypeError: Swig director type mismatch in output value of
        #    type 'float32'"
        # without returning a value
        return fraction


class PosIter:
    def __init__(self, x):
        self.val = x
        self.next = None


class RoutePlan(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, mode, seed=None):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.seed_num = seed
        self.mode = config.load(mode)

        # 环境物理结构变量
        self.world = Box2D.b2World(gravity=(0, 0))
        self.reef = []
        self.ground = None
        self.time_step = 0

        # agent生成参数
        self.agent_num = self.mode.ships_num
        self.ships_init = self.mode.ships_init
        self.ships_goal = self.mode.ships_goal
        self.ships_speed = self.mode.ships_speed
        self.ships_head = self.mode.ships_head
        self.ships_length = self.mode.ships_length
        self.angle_limit = self.mode.angle_limit

        # 归一化参数
        self.ships_x_min = self.mode.ships_x_min
        self.ships_y_min = self.mode.ships_y_min
        self.ships_x_max = self.mode.ships_x_max
        self.ships_y_max = self.mode.ships_y_max
        self.ships_dis = self.mode.ships_dis
        self.ships_dis_max = self.mode.ships_dis_max
        self.ship_max_length = self.mode.ships_length.max()
        self.ship_max_speed = self.mode.ships_speed.max()

        # agent渲染列表
        self.ships = []
        self.term_points = []

        # 终止状态检测 + reward计算
        self.state_store = None
        self.norm = NormData()
        self.check_state = CheckState(self.mode)
        self.ships_done = np.array([False] * self.agent_num, dtype=bool)
        self.ships_coll = np.array([False] * self.agent_num, dtype=bool)

        # Raycast船体半径
        self.ship_radius = 0.36*element_wise_weight
        # 渲染列表
        self.draw_list = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (1,), dtype=np.float32)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.ships:
            return
        if not self.term_points:
            return
        # 船体和终止点移除
        for idx in range(self.agent_num):
            self.world.DestroyBody(self.ships[idx])
            self.world.DestroyBody(self.term_points[idx])
        self.ships.clear()
        self.term_points.clear()
        # 海下礁石移除
        for reef in self.reef:
            self.world.DestroyBody(reef)
        self.reef.clear()

    # 验证礁石生成位置是否合法
    def isValid(self, fixture_center, barrier_dict=None, reef_dict=None):
        if barrier_dict is not None:
            for idx in range(len(barrier_dict['center_point'])):
                if Distance_Cacul(barrier_dict['center_point'][idx], fixture_center) > 2.5 * barrier_dict['radius'][idx]:
                    continue
                else:
                    return False
        if reef_dict is not None:
            for idx_reef in range(len(reef_dict['center_point'])):
                if Distance_Cacul(reef_dict['center_point'][idx_reef], fixture_center) > 1.25:
                    continue
                else:
                    return False
        return True

    def reset(self):
        self._destroy()
        # 环境变量重置
        self.time_step = 0
        self.state_store = None
        # 终止状态重置
        self.ships_done = np.array([False] * self.agent_num, dtype=bool)
        self.ships_coll = np.array([False] * self.agent_num, dtype=bool)

        # 重建环境
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        """设置边界范围"""
        # self.ground = self.world.CreateBody(position=(0, VIEWPORT_H/SCALE/2))
        # self.ground.CreateEdgeFixture(vertices=[(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
        #                                         (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
        #                                         (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
        #                                         (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)],
        #                               friction=1.0,
        #                               density=1.0)
        # self.ground.CreateEdgeChain(
        #     [(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
        #      (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
        #      (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
        #      (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
        #      (-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)])

        """暗礁生成"""
        """
        # reef generate test
        reef_dict = {'center_point': [],
                     'radius': []}
        if self.seed_num is not None:
            self.np_random.seed(self.seed_num)
        while len(self.reef) < 5:
            reef_position = (self.np_random.uniform(-5, 2),
                             self.np_random.uniform(8, 13))
            if self.isValid(reef_position, reef_dict):
                reef = self.world.CreateStaticBody(shapes=b2CircleShape(pos=reef_position,
                                                   radius=0.36))
                reef.hide = True
                self.reef.append(reef)
                reef_dict['center_point'].append(reef_position)
                reef_dict['radius'].append(0.36)
        """

        for idx in range(self.agent_num):
            """ship生成"""
            """
            >>>help(Box2D.b2BodyDef)
            angularDamping: 角度阻尼
            angularVelocity: 角速度
            linearDamping：线性阻尼
            #### 增加线性阻尼可以使物体行动有摩擦
            """
            ship = self.world.CreateKinematicBody(
                position=self.ships_init[idx],
                angle=math.radians(self.ships_head[idx]),
                angularDamping=20,
                linearDamping=10,
                fixedRotation=True,
                fixtures=b2FixtureDef(
                    shape=b2PolygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in SHIP_POLY]),
                    density=3.5,
                    friction=1,
                    categoryBits=0x0010,
                    maskBits=0x001,     # collide only with ground
                    restitution=0.0)    # 0.99 bouncy
                    )
            # 碰撞出现 -> contact为True
            # 船与船之间相撞 -> coll为True
            # 船体与抵达点相接触 -> game_over为True
            ship.color_bg = PANEL[idx]
            ship.color_fg = PANEL[idx+1]
            self.ships.append(ship)

            """抵达点生成"""
            # 设置抵达点位置
            circle_shape = b2PolygonShape(vertices=[(x/SCALE*5, y/SCALE*5) for x, y in REACH_POLY])
            reach_area = self.world.CreateStaticBody(position=self.ships_goal[idx],
                                                     fixtures=b2FixtureDef(
                                                         shape=circle_shape)
                                                     )
            reach_area.color = ship.color_bg
            self.term_points.append(reach_area)
        # self.draw_list = self.ships + [self.ground] + self.term_points
        self.draw_list = self.ships + self.term_points
        """
        # reward Heatmap构建
        # 使heatmap只生成一次
        # profile测试发现heatmap生成所消耗时间是Box2D环境生成所消耗时间3000倍以上
        # 减少heatmap在reset中重置次数
        if not self.hard_update:
            bound_list = self.barrier + self.reef + [self.reach_area] + [self.ground]
            heat_map_init = HeatMap(bound_list, positive_reward=self.positive_heat_map)
            self.pathFinding = heat_map_init.rewardCal4TraditionalMethod
            self.heatmap_mapping_ra = heat_map_init.ra
            self.heatmap_mapping_bl = heat_map_init.bl
            self.heat_map = heat_map_init.rewardCal
            self.heat_map += heat_map_init.ground_rewardCal_redesign
            self.heat_map += heat_map_init.reach_rewardCal
            self.heat_map = normalize(self.heat_map) - 1
            self.hard_update = True
        """
        """seaborn heatmap"""
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, axes = plt.subplots(1, 1)
        # sns.heatmap(self.heat_map.T, annot=False, ax=axes).invert_yaxis()
        # plt.show()
        """matplotlib 3d heatmap"""
        # from mpl_toolkits import mplot3d
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # x, y = np.meshgrid(np.linspace(0, 79, 80), np.linspace(0, 79, 80))
        # ax.plot_surface(x, y, self.heat_map.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # plt.show()
        return self.step(self.np_random.uniform(0, 0, size=(self.agent_num, 1)))

    def step(self, action_samples: np.array):
        """
        # gym environment step
        :param action_samples: -> action_samples.shape == (self.agent_num, action_dim)
        :return: obs_n: np.ndarray -> obs_n.shape = (self.agent_num, state_dim)
                 reward_n: np.ndarray -> reward_n.shape == (self.agent_num,)
                 done_term: np.ndarray -> done_term.shape == (self.agent_num,)
                 info: dict -> info.keys = ('done_coll', 'dis_closest')
                               info['done_coll'].shape == (self.agent_num,)
                               info['dis_closest'].shape == (self.agent_num,)\
        """
        assert action_samples.shape == (self.agent_num, 1)
        # action_samples[..., 0] = np.clip(action_samples[..., 0], 0, 1).astype('float32')
        action_samples[..., 0] = np.clip(action_samples[..., 0], -1, 1).astype('float32')

        # multi-agent 返回值统计
        obs_n = []
        raw_obs_n = []
        info_n = {}

        """船体与世界交互得到各agent observation"""
        for idx in range(self.agent_num):
            if not self.ships_done[idx]:
                self._set_action(action_samples, idx)

        self.world.Step(1.0 / FPS, 10, 10)
        for agent in range(self.agent_num):
            raw_obs, obs = self._get_observation(agent)
            raw_obs_n.append(raw_obs)
            obs_n.append(obs)
        raw_obs_n = np.array(raw_obs_n)
        reward_n, done_term, done_coll, dis_closest = self._compute_done_reward(self.state_store, raw_obs_n)
        self.state_store = raw_obs_n
        # print(f"shipa_reward: {reward_n[0]}, shipb_reward: {reward_n[1]}")
        # [todo: 确定obs_n, reward_n, done_term 数值存储统一使用list还是ndarray]

        self.time_step += 1

        obs_n = np.stack(obs_n, axis=0)
        reward_n = np.array(reward_n)
        done_term = np.array(done_term)
        info_n['done_coll'] = np.array(done_coll)
        info_n['dis_closest'] = dis_closest
        return obs_n, reward_n, done_term, info_n

    def _set_action(self, act, idx):
        """
        # 动作应用环境计算
        :param act: np.ndarray -> act.shape == (self.agent_num, action_dim)
        :param idx: int -> agent index
        :return:
        """
        """船体推进位置及动力大小计算"""
        # speed2ship = self.remap(act[idx, 0], MAIN_ENGINE_POWER)
        orient2ship = self.remap(act[idx, 0], math.radians(self.angle_limit))
        # 映射逆时针为正方向
        # agent.angle = np.clip(-b2_pi/6, b2_pi/6, orient2ship-agent.angle)
        self.ships[idx].angle = wrap_to_pi(self.ships[idx].angle + orient2ship)
        # if not idx:
        #     print(f'angle: {self.ships[idx].angle}, orient: {math.degrees(orient2ship)}')

        self.ships[idx].position[0] = math.cos(self.ships[idx].angle)*self.ships_speed[idx].item() + self.ships[idx].position[0]
        self.ships[idx].position[1] = math.sin(self.ships[idx].angle)*self.ships_speed[idx].item() + self.ships[idx].position[1]

    def _get_observation(self, idx):
        # 11 维传感器数据字典
        sensor_raycast = {"points": np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'normal': np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'distance': np.zeros((RAY_CAST_LASER_NUM, 2))}
        # 传感器扫描
        length = self.ship_radius * 5  # Set up the raycast line
        point1 = self.ships[idx].position
        for vect in range(RAY_CAST_LASER_NUM):
            ray_angle = self.ships[idx].angle - b2_pi / 2 + (b2_pi * 2 / RAY_CAST_LASER_NUM * vect)
            d = (length * math.cos(ray_angle), length * math.sin(ray_angle))
            point2 = point1 + d

            # 初始化Raycast callback函数
            callback = RayCastClosestCallback()
            self.world.RayCast(callback, point1, point2)
            if callback.hit:
                sensor_raycast['points'][vect] = callback.point
                sensor_raycast['normal'][vect] = callback.normal
                if callback.fixture == self.term_points[idx].fixtures[0]:
                    sensor_raycast['distance'][vect] = (2, Distance_Cacul(point1, callback.point) - self.ship_radius)
                # elif callback.fixture in self.ground.fixtures:
                #     sensor_raycast['distance'][vect] = (2, Distance_Cacul(point1, callback.point) - self.ship_radius)
                else:
                    sensor_raycast['distance'][vect] = (1, Distance_Cacul(point1, callback.point) - self.ship_radius)
            else:
                sensor_raycast['distance'][vect] = (0, 5 * self.ship_radius)
        sensor_raycast['distance'][..., 1] /= self.ship_radius * 5

        # 当前船体相较于世界位置
        pos = self.ships[idx].position

        # 根据速度矢量计算当前船体行驶方位
        # 取余操作在对负数取余时，在Python当中,如果取余的数不能够整除，那么负数取余后的结果和相同正数取余后的结果相加等于除数。
        # 将负数角度映射到正确的范围内
        if self.ships[idx].angle < 0:
            angle_unrotate = - ((b2_pi*2) - self.ships[idx].angle % (b2_pi * 2))
        else:
            angle_unrotate = self.ships[idx].angle % (b2_pi * 2)
        # 角度映射到 [-pi, pi]
        if angle_unrotate < -b2_pi:
            angle_unrotate += (b2_pi * 2)
        elif angle_unrotate > b2_pi:
            angle_unrotate -= (b2_pi * 2)
        ship_head_radians = wrap_to_2pi(self.ships[idx].angle)
        ship_head_degrees = round(math.degrees(ship_head_radians), 2)

        # 原始状态值
        state = [
            pos[0],
            pos[1],
            ship_head_degrees
        ]
        assert len(state) == 3

        # 状态值归一化
        norm_state = [
            self.norm.pos_norm(pos[0], self.ships_x_min[idx], self.ships_x_max[idx]),
            self.norm.pos_norm(pos[1], self.ships_y_min[idx], self.ships_y_max[idx]),
            self.norm.ang_norm(ship_head_degrees)
        ]
        assert len(norm_state) == 3

        return np.hstack(state), np.hstack(norm_state)

    def _compute_done_reward(self, state, next_state):
        reward_done = self.check_state.check_done(next_state, self.ships_done)
        reward_ang_keep = self.check_state.check_rela_ang(next_state)
        if self.agent_num == 1:
            reward = reward_done + reward_ang_keep
        else:
            reward_coll, dis_closest = self.check_state.check_coll(next_state, self.ships_coll)
            self.ships_done = self.ships_done | self.ships_coll
            reward_cpa = self.check_state.check_CPA(next_state)
            if self.time_step == 0:
                reward_corleg = 0.0
            else:
                reward_corleg, _ = self.check_state.check_CORLEGs(state, next_state)
            reward = reward_done + reward_ang_keep + reward_coll + reward_cpa + reward_corleg
            reward = self.norm.rwd_norm(reward, self.check_state.reward_max)
            # debug位
            if reward.min() < -1 or reward.max() > 1:
                time.time()
        return reward, self.ships_done, self.ships_coll, dis_closest

    def render(self, mode='human', hide=True):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(-VIEWPORT_W/SCALE/2, VIEWPORT_W/SCALE/2, 0, VIEWPORT_H/SCALE)

        for obj in self.draw_list:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    if hasattr(obj, 'hide'):
                        if not hide:
                            self.viewer.draw_circle(f.shape.radius,
                                                    20,
                                                    color=PANEL[3],
                                                    filled=False,
                                                    linewidth=2).add_attr(t)
                    else:
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[2]).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[3], filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    if hasattr(obj, 'color_bg'):
                        self.viewer.draw_polygon(path, color=obj.color_bg)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=obj.color_fg, linewidth=2)
                    elif hasattr(obj, 'color'):
                        self.viewer.draw_polygon(path, color=obj.color)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=PANEL[5], linewidth=2)
                    else:
                        self.viewer.draw_polygon(path, color=PANEL[4])
                        self.viewer.draw_polyline(path, color=PANEL[7], linewidth=10)
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return self.viewer.render(return_rgb_array=True)

    @staticmethod
    def remap(action, remap_range):
        # assert isinstance(action, np.ndarray)
        return (remap_range * 2) / (1.0 * 2) * (action + 1) - remap_range

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def Distance_Cacul(pointA, pointB):
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.sqrt(cx * cx + cy * cy)


def Orient_Cacul(pointA, pointB):
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.atan2(cy, cx)


def manual_control(key):
    global action
    if key.event_type == 'down' and key.name == 'a':
        if key.event_type == 'down' and key.name == "w":
            action[1] = 1
        elif key.event_type == 'down' and key.name == 's':
            action[1] = -1
        action[0] = -1
    if key.event_type == 'down' and key.name == 'd':
        if key.event_type == 'down' and key.name == "w":
            action[1] = 1
        elif key.event_type == 'down' and key.name == 's':
            action[1] = -1
        action[0] = 1


def demo_route_plan():
    env = RoutePlan(mode='2Ship_CrossAway')
    total_reward = 0
    steps = 0
    done = None
    while True:
        if not steps % 5:
            action = np.random.uniform(-0.01, 0.01, size=(2, 2))
            action[..., 0] = abs(action[..., 0])
            s, r, done, info = env.step(action)
            done_coll = info['done_coll']
            dis_closest = info['dis_closest']
            total_reward += r

        still_open = env.render()
        if still_open is False:
            break
        # if steps % 20 == 0 or done:
        #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if any(done) or steps % 2048 == 0 or any(done_coll):
            env.reset()
    env.close()


def demo_TraditionalPathPlanning(env, seed=None):
    env.seed(seed)
    from pathfinding.core.diagonal_movement import DiagonalMovement
    from pathfinding.core.grid import Grid
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.finder.dijkstra import DijkstraFinder
    from pathfinding.finder.ida_star import IDAStarFinder
    from PIL import Image, ImageDraw

    grid = Grid(matrix=env.pathFinding)
    ship_position = heat_map_trans(env.ship.position)
    start_point = grid.node(ship_position[0], ship_position[1])
    end_point = grid.node(env.heatmap_mapping_ra['position'][0], env.heatmap_mapping_ra['position'][1])

    print(f'current height: {grid.height}, current width: {grid.width}')
    finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
    start_time = time.time()
    path, runs = finder.find_path(start_point, end_point, grid)
    end_time = time.time() - start_time
    print(end_time)
    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start_point, end=end_point))
    print(path)


if __name__ == '__main__':
    demo_route_plan()
    # demo_TraditionalPathPlanning(RoutePlan('2Ship_CrossAway', seed=42))
