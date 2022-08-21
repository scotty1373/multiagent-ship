# -*-  coding=utf-8 -*-
# @Time : 2022/4/10 10:46
# @Author : Scotty1373
# @File : sea_env.py
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
from .heatmap import HeatMap, heat_map_trans, normalize
from utils_tools.utils import img_proc

SCALE = 30
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


class ContactDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # 两船 相撞
        if contact.fixtureA.body in self.env.ships and contact.fixtureB.body in self.env.ships:
            contactA_index = self.env.ships.index(contact.fixtureA.body)
            contactB_index = self.env.ships.indecx(contact.fixtureB.body)
            self.env.ships[contactA_index].contact = True
            self.env.ships[contactA_index].coll = True
            self.env.ships[contactB_index].contact = True
            self.env.ships[contactB_index].coll = True
        # 碰撞(触礁/边界)
        if contact.fixtureA.body or contact.fixtureB.body in self.env.ships:
            contact_index = self.env.ships.index(contact.fixtureA.body) \
                if contact.fixtureA.body in self.env.ships else self.env.ships.index(contact.fixtureB.body)
            self.env.ships[contact_index].contact = True

    def EndContact(self, contact):
        if contact.fixtureA.body or contact.fixtureB.body in self.env.ships:
            pass


class PosIter:
    def __init__(self, x):
        self.val = x
        self.next = None


class RoutePlan(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, agent_num, seed=None, ship_pos_fixed=None, positive_heatmap=None):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.seed_num = seed
        self.ship_pos_fixed = ship_pos_fixed
        if positive_heatmap is not None:
            self.positive_heat_map = True
        else:
            self.positive_heat_map = None
        self.hard_update = False

        # 环境物理结构变量
        self.world = Box2D.b2World(gravity=(0, 0))
        self.barrier = []
        self.reef = []
        self.ship = None
        self.reach_area = None
        self.ground = None

        # agent数量
        self.agent_num = agent_num
        self.ships = []

        # 抵达点数量
        self.reachPoints = []

        # 障碍物生成边界
        self.barrier_bound_x = 0.8
        self.barrier_bound_y = 0.9
        self.dead_area_bound = 0.03
        self.ship_radius = 0.36*element_wise_weight

        # game状态记录
        self.timestep = 0
        self.game_over = None
        self.ground_contact = None
        self.dist_record = None
        self.draw_list = None
        self.dist_norm = 14.38
        self.dist_init = None
        self.iter_ship_pos = None

        # 生成环形链表
        self.loop_ship_posGenerator()

        # heatmap生成状态记录
        self.heatmap_mapping_ra = None
        self.heatmap_mapping_bl = None
        self.heat_map = None
        self.pathFinding = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        self.reset()

    def loop_ship_posGenerator(self):
        self.iter_ship_pos = cur = PosIter(SHIP_POSITION[0])
        for x in SHIP_POSITION:
            tmp = PosIter(x)
            cur.next = tmp
            cur = tmp
        cur.next = self.iter_ship_pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if self.ship is None:
            return
        # 清除障碍物
        if self.barrier:
            for idx, barr in enumerate(self.barrier):
                self.world.DestroyBody(barr)
        self.barrier.clear()
        if self.reef:
            for reef in self.reef:
                self.world.DestroyBody(reef)
        self.reef.clear()
        # 清除reach area
        self.world.DestroyBody(self.reach_area)
        self.reach_area = None
        # 清除船体
        self.world.DestroyBody(self.ship)
        self.ship = None

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
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.ground_contact = False
        self.dist_record = None
        self.timestep = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        """设置边界范围"""
        self.ground = self.world.CreateBody(position=(0, VIEWPORT_H/SCALE/2))
        self.ground.CreateEdgeFixture(vertices=[(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
                                                (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
                                                (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
                                                (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)],
                                      friction=1.0,
                                      density=1.0)
        self.ground.CreateEdgeChain(
            [(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
             (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
             (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
             (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
             (-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)])

        """暗礁生成"""
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

        for idx in range(self.agent_num):
            """ship生成"""
            """
            >>>help(Box2D.b2BodyDef)
            angularDamping: 角度阻尼
            angularVelocity: 角速度
            linearDamping：线性阻尼
            #### 增加线性阻尼可以使物体行动有摩擦
            """
            ship = self.world.CreateDynamicBody(
                position=SHIP_POSITION[idx],
                angle=0.0,
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
            ship.contact = False
            ship.coll = False
            ship.game_over = False
            ship.color_bg = PANEL[idx]
            ship.color_fg = PANEL[idx+1]
            ship.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                                     self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), wake=True)
            self.ships.append(ship)

            """抵达点生成"""
            # 设置抵达点位置
            circle_shape = b2PolygonShape(vertices=[(x/5, y/5) for x, y in REACH_POLY])
            reach_area = self.world.CreateStaticBody(position=SHIP_POSITION[-idx],
                                                     fixtures=b2FixtureDef(
                                                         shape=circle_shape
                                                    ))
            reach_area.color = ship.color_bg
            self.reachPoints.append(reach_area)
        self.draw_list = self.ships + self.reach_area + [self.ground]
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

        end_info = b2Distance(shapeA=self.ship.fixtures[0].shape,
                              idxA=0,
                              shapeB=self.reach_area.fixtures[0].shape,
                              idxB=0,
                              transformA=self.ship.transform,
                              transformB=self.reach_area.transform,
                              useRadii=True)
        self.dist_init = end_info.distance
        return self.step(self.np_random.uniform(-1, 1, size=(2,)))

    def step(self, action_samples: np.array):
        assert action_samples.shape == np.array(self.agent_num, 2)
        action_samples[..., 0] = np.clip(action_samples[..., 0], 0, 1).astype('float32')
        action_samples[..., 1] = np.clip(action_samples[..., 1], -1, 1).astype('float32')

        # multi-agent 返回值统计
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        if not self.ships:
            return

        """船体与世界交互得到各agent observation"""
        for agent in self.ships:
            self._set_action(action_samples, agent)

        self.world.Step(1.0 / FPS, 10, 10)
        for agent in self.ships:
            obs_n.append(self._get_observation(agent))
            reward_n.append(self._compute_reward(agent))
            done_n.append(self._compute_done(agent))
            info_n.append({})

        '''失败终止状态定义在训练迭代主函数中，由主函数给出失败终止状态惩罚reward'''
        return obs_n, reward_n, done_n, info_n

    def _set_action(self, act, agent):
        """船体推进位置及动力大小计算"""
        speed2ship = self.remap(act[0], MAIN_ENGINE_POWER)
        orient2ship = self.remap(act, MAIN_ORIENT_POWER)
        self.ship.angle = np.clip(-b2_pi/6, b2_pi/6, orient2ship+self.ship.angle)

        self.ship.position[0] = math.cos(self.ship.angle)*speed2ship + self.ship.position[0]
        self.ship.position[1] = math.sin(self.ship.angle)*speed2ship + self.ship.position[1]

    def _get_observation(self, agent, landmark):
        # 11 维传感器数据字典
        sensor_raycast = {"points": np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'normal': np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'distance': np.zeros((RAY_CAST_LASER_NUM, 2))}
        # 传感器扫描
        length = self.ship_radius * 5  # Set up the raycast line
        point1 = agent.position
        for vect in range(RAY_CAST_LASER_NUM):
            ray_angle = agent.angle - b2_pi / 2 + (b2_pi * 2 / RAY_CAST_LASER_NUM * vect)
            d = (length * math.cos(ray_angle), length * math.sin(ray_angle))
            point2 = point1 + d

            # 初始化Raycast callback函数
            callback = RayCastClosestCallback()

            self.world.RayCast(callback, point1, point2)

            if callback.hit:
                sensor_raycast['points'][vect] = callback.point
                sensor_raycast['normal'][vect] = callback.normal
                if callback.fixture == landmark.fixtures[0]:
                    sensor_raycast['distance'][vect] = (3, Distance_Cacul(point1, callback.point) - self.ship_radius)
                elif callback.fixture in self.ground.fixtures:
                    sensor_raycast['distance'][vect] = (2, Distance_Cacul(point1, callback.point) - self.ship_radius)
                else:
                    sensor_raycast['distance'][vect] = (1, Distance_Cacul(point1, callback.point) - self.ship_radius)
            else:
                sensor_raycast['distance'][vect] = (0, 5 * self.ship_radius)
        sensor_raycast['distance'][..., 1] /= self.ship_radius * 5

        # 当前船体相较于世界位置
        pos = agent.position
        # 当前船体速度矢量
        vel = agent.linearVelocity

        # 根据速度矢量计算当前船体行驶方位
        ship_head_radians = wrap_to_2pi(agent.angle)
        # ship速度标量计算
        ship_vel_scaler = Distance_Cacul(vel, b2Vec2(0, 0))

        state = [
            pos[0],
            pos[1],
            ship_head_radians
        ]
        assert len(state) == 3

        return np.hstack(state)

    def _get_done(self, agent):
        done = np.zeros((self.agent_num, 1))
        for idx, agent in enumerate(self.ships):
            if agent.contact:
                done[idx] = True
            else:
                done[idx] = False    # 船体碰撞终止状态
        return done

    def _compute_done(self, agent):
        # [todo: 终止条件判断]
        pass

    def _compute_reward(self, agent):
        # [todo: 1.终止条件奖励计算
        #      2.船体碰撞奖励计算
        #      3.CPA奖励计算
        #      4.符合CORLEGs规则奖励计算]
        pass

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
                        self.viewer.draw_polygon(path, color=PANEL[4])
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


def demo_route_plan(env, seed=None, render=False):
    global action
    env.seed(seed)
    total_reward = 0
    steps = 0
    done = False
    keyboard.hook(manual_control)
    while True:
        if not steps % 5:
            s, r, done, info = env.step(action)
            action = [0, 0]
            total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        # if steps % 20 == 0 or done:
        #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done and env.game_over:
            break
    env.close()
    return total_reward


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
    # demo_route_plan(RoutePlan(seed=42), render=True)
    demo_TraditionalPathPlanning(RoutePlan(barrier_num=3, seed=42))
