import gym
from gym.utils import seeding
import logging
import numpy as np
import networkx as ne  # 导入建网络模型包，命名ne
import matplotlib.pyplot as mp
import random

logger = logging.getLogger(__name__)


class SWEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    bandwidth = []
    occupy_ratio = []
    time_shake = []
    transmit_ratio = []
    info_accord = []

    for i in range(30):
        # 初始化概率，用于控制指标进一步生成数值的概率
        prob_transmit = random.random()
        prob_accord = random.random()
        prob_band_shake = random.random()
        prob_occupy_ratio = random.random()
        # 带宽性能随机生成
        t1 = random.uniform(1, 10)
        bandwidth.append(round(t1, 2))
        # 链路占用比
        if prob_occupy_ratio > 0.4:
            o1 = random.uniform(0.1, 0.4)
        else:
            o1 = random.uniform(0.4, 1)
        occupy_ratio.append(round(o1, 4))
        # 时延抖动按0.4-0.6生成
        if prob_band_shake > 0.35:
            s1 = random.uniform(10, 60)
        else:
            s1 = random.uniform(60, 200)
        time_shake.append(round(s1, 0))
        # 按照0.8的概率生成转发率良好的节点
        if prob_transmit > 0.35:
            tr = random.uniform(0.7, 1)
        else:
            tr = random.uniform(0.3, 0.6)
        transmit_ratio.append(round(tr, 4))
        # 按照0.7的概率生成信息符合度良好的节点
        if prob_accord > 0.5:
            i1 = random.uniform(0.7, 1)
        else:
            i1 = random.uniform(0.2, 0.6)
        info_accord.append(round(i1, 4))


    def __init__(self):
        # 小世界网络生成基本设置
        self.NETWORK_SIZE = 30
        self.K = 4
        self.reconnect_p = 0.5
        self.cm = None  # 邻接矩阵
        self.cm_weight = None
        self.ws = None  # 图 networkx
        self.ps = None  # 图框架
        self.max_weight = 5
        self.change_weight = 1.5
        self.create_weight = np.random.rand
        self.terminal = 13
        # 状态空间设置
        self.observation_space = list(range(30))  # 状态空间
        self.__terminal_finial = 13  # 终止状态为字典格式

        # 状态转移
        pass

        # 环境基本设置
        self.__gamma = 0.8  # 折扣因子
        self.viewer = None  # 环境可视化
        self.__state = None  # 当前状态
        self.seed()  # 随机种子
        self.boot_reward = True  # 引导奖励

    def _reward(self, state, action):
        """
        回报
        :param state:
        :param action
        :return:
        """
        r = 0.0
        step_num = state[0]
        local_point = state[1]
        cm = state[2]
        cm_weight = state[3]
        # 终止
        if cm[local_point, action] == 0 or step_num > 30:
            return -10000
        # 到达
        if self.__terminal_finial == action:
            return 1000
        # dijkstra boot
        if self.boot_reward:
            G = ne.Graph()  # 小世界网络转出至另一个图中
            for i in range(30):  # 添加边
                for j in range(i, 30):
                    if cm[i, j] != 0:
                        G.add_edge(i, j, weight=cm_weight[i, j])
            path = ne.dijkstra_path(G, source=local_point, target=self.__terminal_finial)
            mid = path[1]
            if action == mid:
                return 0

        # 经过
        return -cm_weight[local_point, action]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def transform(self, state, action):
        """
        状态转换
        :param state:
        :param action:
        :return:
        """
        step_num = state[0]
        local_point = state[1]
        cm = state[2]
        cm_weight = state[3]

        # 计算回报
        r = self._reward(state, action)

        # 判断是否终止
        if cm[local_point, action] == 0 or step_num > 30 or self.__terminal_finial == action:
            return state, self._reward(state, action), True, {}
        next_state = [[], [], [], []]
        # 状态转移
        next_state[0] = state[0] + 1
        next_state[1] = action
        next_state[2] = cm
        # 浮动
        # cm_weight = cm_weight + (self.create_weight(self.cm[0], self.cm[1]) - 0.5) * self.change_weight
        next_state[3] = cm_weight
        # 判断是否终止
        is_terminal = False

        return next_state, r, is_terminal, {}

    def step(self, action):
        """
        交互
        :param action: 动作值
        :return: 下一个状态，回报，是否停止，调试信息
        """
        state = self.__state
        next_state, r, is_terminal, _ = self.transform(state, action)
        self.__state = next_state
        return next_state, r, is_terminal, {}

    def change_graph(self):
        # 重新生成小世界网络
        self.ws = ne.watts_strogatz_graph(self.NETWORK_SIZE, self.K, self.reconnect_p)
        self.ps = ne.circular_layout(self.ws)  # 布置框架
        # 可视化
        pass
        # self.viewer = ne.draw(self.ws, self.ps, with_labels=False, node_size=self.NETWORK_SIZE)
        self.cm = np.array(ne.adjacency_matrix(self.ws).todense())  # 邻接矩阵
        self.cm_weight = self.cm * self.create_weight(self.cm.shape[0], self.cm.shape[1]) * self.max_weight  # 邻接
        for i in range(30):
            for j in range(30):
                self.cm_weight[i,j]=self.cm_weight[j,i]

    def reset(self):
        """
        重置环境
        :return: [步数，位置，邻接矩阵，带权矩阵]
        """
        # 设置起点
        self.__state = [0, 0, self.cm, self.cm_weight]  # 步数，位置，邻接矩阵，带权矩阵
        return self.__state  # 步数，位置，邻接矩阵，带权矩阵

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                mp.show()

    def indexes_update(self):                           #指标的范围更新
        for j in range(30):
            float_rate = random.uniform(0.7, 1.2)       #链路占用比
            self.occupy_ratio[j] = round(self.occupy_ratio[j] * float_rate, 4)
            if self.occupy_ratio[j] > 1:
                self.occupy_ratio[j] *= 0.8
            if self.occupy_ratio[j] < 0.03:
                self.occupy_ratio[j] += 0.3
            float_rate = random.uniform(0.7, 1.2)
            self.time_shake[j] = round(self.time_shake[j] * float_rate, 2)#时延抖动
            if self.time_shake[j] > 200:
                self.time_shake[j] *= 0.6
            if self.time_shake[j] < 10:
                self.time_shake[j] += 30
            float_rate = random.uniform(0.98, 1.02)
            self.transmit_ratio[j] = round(self.transmit_ratio[j] * float_rate, 4)#转发率
            if self.transmit_ratio[j] > 1:
                self.transmit_ratio[j] *= 0.8
            if self.transmit_ratio[j] < 0.03:
                self.transmit_ratio[j] += 0.3
            float_rate = random.uniform(0.98, 1.02)
            self.info_accord[j] = round(self.info_accord[j] * float_rate, 4)#信息符合度
            if self.info_accord[j] > 1:
                self.info_accord[j] *= 0.8
            if self.info_accord[j] < 0.03:
                self.info_accord[j] += 0.3
        return


if __name__ == '__main__':
    env = SWEnv()
    env.reset()
    env.render(closs=True)
